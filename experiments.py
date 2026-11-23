import numpy as np
from sklearn.svm import SVC

from QSVM import run_qsvm, shots
from preprocessing import load_split, sample_training_subset, preprocess, seed

'''
 Purpose of this file:
    1. Classical baseline: SVM with an RBF kernel (same C as the QSVM)
    2. Run the QSVM and the classical SVM on identical, identically preprocessed data
    3. Print the results tables and the comparison with the Phase 2 prediction
'''

C = 1.0  # same regularization for both models so only the kernel differs


def run_classical_svm(X_train, y_train, X_test, y_test, C=C):
    '''Classical reference: scikit-learn SVC with the RBF kernel (gamma="scale", the default).'''
    model = SVC(kernel="rbf", C=C).fit(X_train, y_train)
    train_acc = float(model.score(X_train, y_train))
    test_acc = float(model.score(X_test, y_test))
    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "generalization_gap": train_acc - test_acc,
    }


def run_test_case(n_features, n_train, seed=seed, shots=shots):
    '''
    One test case: one training subset, one feature count, three models on the same split
        - QSVM with the shot-based kernel (AerSimulator, `shots` per kernel entry)
        - QSVM with the exact statevector kernel (noiseless reference for the shot-based kernel)
        - classical RBF SVM (reference answer)
    '''
    X_pool, X_test_raw, y_pool, y_test = load_split()
    X_train_raw, y_train = sample_training_subset(X_pool, y_pool, n_train, seed)
    X_train, X_test, prep = preprocess(X_train_raw, X_test_raw, n_features)

    qsvm = run_qsvm(X_train, y_train, X_test, y_test, C=C, shots=shots, verbose=False)
    qsvm_exact = run_qsvm(X_train, y_train, X_test, y_test, C=C, exact=True, verbose=False)
    rbf = run_classical_svm(X_train, y_train, X_test, y_test)

    # always predicting the larger training class --> the accuracy any useful model must beat
    majority_class = np.bincount(y_train).argmax()

    return {
        "n_features": n_features,
        "n_train": n_train,
        "n_test": len(y_test),
        "shots": shots,
        "train_circuits": n_train * (n_train - 1) // 2,  # upper triangle of K_train
        "test_circuits": len(y_test) * n_train,          # every test point vs every training point
        "train_classes": np.bincount(y_train).tolist(),   # [malignant, benign]
        "pca_variance": float(prep.named_steps["pca"].explained_variance_ratio_.sum()),
        "majority_acc": float(np.mean(y_test == majority_class)),
        "qsvm": qsvm,
        "qsvm_exact": qsvm_exact,
        "rbf": rbf,
    }


def print_test_case(case):
    '''Results table for one test case.'''
    n, m = case["n_features"], case["n_train"]
    print(f"\n=== Test case: {n} PCA features / {n} qubits, {m} training samples ===")
    print("Ran on: local Qiskit AerSimulator (noiseless), seed_simulator = seed_transpiler = 42")
    print(f"Shots: {case['shots']} per kernel entry; "
          f"{case['train_circuits']} training + {case['test_circuits']} test kernel circuits")
    malignant, benign = case["train_classes"]
    print(f"Training subset: {malignant} malignant / {benign} benign; "
          f"test set: {case['n_test']} patients (fixed 20% split)")
    print(f"PCA keeps {case['pca_variance']:.1%} of the training variance")
    print()
    print(f"{'Model':<26} | {'Kernel':<22} | {'Train acc':>9} | {'Test acc':>8} | {'Gap':>6}")
    print("-" * 82)
    rows = [
        ("QSVM", f"quantum, {case['shots']} shots", case["qsvm"]),
        ("QSVM (exact reference)", "quantum, statevector", case["qsvm_exact"]),
        ("Classical SVM (reference)", "RBF, C=1", case["rbf"]),
    ]
    for name, kernel, r in rows:
        print(f"{name:<26} | {kernel:<22} | {r['train_acc']:>9.1%} | {r['test_acc']:>8.1%} | "
              f"{r['generalization_gap']:>+6.1%}")
    print(f"{'Majority-class baseline':<26} | {'-':<22} | {'-':>9} | {case['majority_acc']:>8.1%} | {'-':>6}")

    q = case["qsvm"]
    print()
    print("Kernel checks (shot-based training kernel):")
    print(f"  min K(x, x) over training points: {q['min_self_kernel']:.4f}   (ideal: 1)")
    print(f"  K_train symmetric:                {q['is_symmetric']}")
    print(f"  max |K_shots - K_exact|:          {q['max_kernel_error_vs_exact']:.4f}   "
          f"(1 std of shot noise <= 1/(2*sqrt({case['shots']})) = {0.5 / np.sqrt(case['shots']):.4f})")
    print(f"  mean off-diagonal K:              {q['mean_off_diagonal_K']:.4f}")


def print_prediction_check(cases):
    '''Compare each test case with the Phase 2 prediction: QSVM test accuracy >= RBF SVM test accuracy.'''
    print("\n=== Phase 2 prediction: QSVM performs at least as well as the classical RBF SVM ===")
    print(f"{'Test case':<22} | {'QSVM test':>9} | {'RBF test':>8} | {'QSVM - RBF':>10} | Prediction holds?")
    print("-" * 74)
    for case in cases:
        q, r = case["qsvm"]["test_acc"], case["rbf"]["test_acc"]
        label = f"{case['n_features']} qubits, {case['n_train']} train"
        print(f"{label:<22} | {q:>9.1%} | {r:>8.1%} | {q - r:>+10.1%} | {'yes' if q >= r else 'no'}")


def repeat_test_case(n_features, n_train, n_repeats=10):
    '''
    Is one 10-sample draw representative? Repeat the test case on n_repeats different
    training subsets (seeds 0..n_repeats-1), same fixed test set.
    Uses the exact statevector kernel: it matched the shot-based kernel above and runs in seconds.
    '''
    X_pool, X_test_raw, y_pool, y_test = load_split()
    qsvm_acc, rbf_acc = [], []
    for s in range(n_repeats):
        X_train_raw, y_train = sample_training_subset(X_pool, y_pool, n_train, s)
        X_train, X_test, _ = preprocess(X_train_raw, X_test_raw, n_features)
        qsvm_acc.append(run_qsvm(X_train, y_train, X_test, y_test, C=C, exact=True, verbose=False)["test_acc"])
        rbf_acc.append(run_classical_svm(X_train, y_train, X_test, y_test)["test_acc"])
    return np.array(qsvm_acc), np.array(rbf_acc)


def print_repeat_check(test_cases, n_repeats=10):
    print(f"\n=== Robustness check: {n_repeats} random training subsets per test case "
          f"(exact kernel, same test set) ===")
    print(f"{'Test case':<22} | {'QSVM test (mean ± std)':>22} | {'RBF test (mean ± std)':>21}")
    print("-" * 71)
    for n, m in test_cases:
        q, r = repeat_test_case(n, m, n_repeats)
        label = f"{n} qubits, {m} train"
        print(f"{label:<22} | {q.mean():>14.1%} ± {q.std():>5.1%} | {r.mean():>13.1%} ± {r.std():>5.1%}")


def run_results(test_cases=((2, 10), (4, 10))):
    '''Results section: run every (n_features, n_train) test case and print all tables.'''
    cases = [run_test_case(n, m) for n, m in test_cases]
    for case in cases:
        print_test_case(case)
    print_prediction_check(cases)
    print_repeat_check(test_cases)
    return cases
