import qiskit as qk
from qiskit_aer import AerSimulator
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
import numpy as np
from sklearn.svm import SVC

'''
 Purpose of this file:
    1. Take in preprocessed data and features  (n=2,4 where n is the number of features)
    2. Implement QSVM (kernel is main difference from SVM)
    3. Train and test model on split from preprocessing data

Assignment requires:
Fixing of random seeds
such as seed_simulator and seed_transpiler in Qiskit, so a rerun gives the same numbers.


'''

seed_simulator = 42 # rerun gives same number with fixed seed
shots = 1024 # shots per kernel entry (default is 1024)
repititions = 2 # repitions of ZZ feature map

simulator = AerSimulator(seed_simulator=seed_simulator) # simulator with fixed seed

def feature_map(n_features, reps=2):
    '''
    Creation of feature map, using built-in ZZ feature map in qiskit
    --> Turns one data point into a quantum state to be used for later

    repitions sets how many times feature map block in circuit will be run
        --> default 2 to mix up data-dependent features (with H gates), making it difficult for classical computers to compute efficiently
    
    Entanglement: CNOT --> P(2g) --> CNOT gates
        --> shown in qiskit circuit phase 2 of U(x)
        --> since we want all qubits/features to be entangled, set to "full"
    '''
    return zz_feature_map(n_features, reps=reps, entanglement="full")

def quantum_circuit(feature_map, x, x_p, measure=True):
    '''
    Circuit to create K(x,x'), which will give a similarity score between these 2 points
    '''
    # provides feature_map with data points from x (breast cancer, n=2 or n=4 feautures)
    encode = feature_map.assign_parameters(x) # U(x)
    
    # gives inverse of the feature_map with x' as data points
    unencode = feature_map.assign_parameters(x_p).inverse() # U^t(x′)
    
    # Gives one long circuit [ U(x) ] [ U^t(x′) ]
    qc = encode.compose(unencode) 
    assert qc is not None # gets rid of tedious pylance errors that kept coming up in circuit_stats

    if measure:
        # measures every qubit to find prob of every state
        # --> done because kernel is the prob that all qubits come out 0
        qc.measure_all() # measurement on every qubit at the end, 
    return qc

def quantum_kernel(feature_map, A, B=None, shots=shots):
    '''
    A --> 1st pair of data points
    B --> 2nd pair of data points to be compared to A

    B = None means we are building training kernel (A against itself) 
        --> This is done because training should only be done on training points (avoids leakage)

    Training gives us a neat trick:
    --> since the data points are symmetric (we are comparing A to A), we only need to compare half of the matrix (upper triangle), and can copy over the rest of it to complete it, saving time on training
    '''

    symmetric = (B is None) # if B is None return True, else False

    if symmetric:
        B = A
        K = np.eye(len(A)) # Creates diagonal using just A
    else:
        K = np.zeros( (len(A),len(B)) ) # test kernel --> rows = A (test points), columns = B (training points), all zeros to start
            # (len(A),len(B)) is shape, while len(A),len(B) are 2 diff args which doesnt work on zeros

    circuits, positions = [], [] # since we will be running the kernel in batches, we need to hold all of the circuits and positions in lists

    for i in range(len(A)): # building list of circuits
       
        if symmetric: # upper triangle --> only if symmetric
            start = i + 1
        else: start = 0      
        
        for j in range(start, len(B)):
            circuits.append( quantum_circuit(feature_map, A[i], B[j]) )
            positions.append( (i, j) )

    # Print check
    # print("circuits: ", circuits)
    # print("\n---------------------\n")
    # print("positions: ", positions)

    compiled = qk.transpile(circuits, simulator, seed_transpiler=seed_simulator) # circuit --> gates simulator understands
    result = simulator.run(compiled, shots=shots).result() # runs simulator on transpiler with 1024 shots (default value)

    all_zeros = "0" * feature_map.num_qubits # 2 or 4, counts feature_map feature count
    # --> "00" or "0000"

    '''
    example of loop:
    # Training data so symmetry (A=B), only upper triangle needed
    -    positions = [(0, 1), (0, 2), (1, 2)]   # which cell each circuit belongs to
    -    K = [[1, 0, 0],                        # from np.eye(3): diagonal already 1
            [0, 1, 0],
            [0, 0, 1]]
    Steps through loop:    
        1. k = 0, (i, j) = (0, 1) --> compare patient 0 vs patient 1
        result:
            - result.get_counts(0)   # {'00': 903, '01': 9, '10': 104, '11': 8}
            - K[0, 1] = 903 / 1024 = 0.882
            - K[1, 0] = 0.882        # mirrored
        2. k = 1, (i, j) = (0, 2) --> compare patient 0 vs patient 2
        result:
            - result.get_counts(1)   # {'00': 565, '01': 33, '10': 275, '11': 151}
            - K[0, 2] = 565 / 1024 = 0.552
            - K[2, 0] = 0.552
        3. k = 2, (i, j) = (1, 2)
        result:
            - result.get_counts(2)   # {'00': 272, '01': 14, '10': 606, '11': 132}
            - K[1, 2] = 272 / 1024 = 0.266
            - K[2, 1] = 0.266
        
    After loop --> K to be returned
            p0     p1     p2
    p0  [ 1.000  0.882  0.552 ]
    p1  [ 0.882  1.000  0.266 ]
    p2  [ 0.552  0.266  1.000 ]

    '''
    
    for k, (i, j) in enumerate(positions): # i == j would be 1 so we skip it
        counts = result.get_counts(k) # gets counts for circuit[k], taken from simulator trained on number of shots
        # counts format --> {'00': 250, '01': 150, '10': 60, '11': 52 }
        K[i, j] = counts.get(all_zeros, 0) / shots # counts "00" / 1024 (default) to get probability
        # --> probability: similarity score between 2 patients or how much two quantum states overlap
        if symmetric: # training data so we use upper/lower triangle trick
            K[j, i] = K[i, j] # mirror for lower triangle

    return K

def exact_kernel(feature_map, A, B=None):
    """
    quantum_kernel: uses shots and counts how many times "00" is measured, retrieving a similarity score K, which due to noise (randomness), isn't precise

    exact_kernel: calculates full quantum states directly removing noise using inner product of statevectors (patients)
                --> takes |<phi(b)|phi(a)>|^2 directly
    """
    if B is None: # Training data
        B = A

    # one row per patient as quantum states
    states_A = np.array([Statevector( feature_map.assign_parameters(a) ).data for a in A])
    '''
    breakdown:
        - feature_map.assign_parameters(a) --> U(a) encoding patient
        - Statevector( ... ) --> simulate U(a)|00>
        - ( ... ).data --> 4 numbers describing state
       
        # states_A example output: [0.428+0.143j, 0.474+0.082j, 0.590+0.182j, 0.429+0.012j] from a = [.5, 1]
    '''
    states_B = np.array([Statevector(feature_map.assign_parameters(b)).data for b in B])

    # calculate <phi(b)|phi(a)>
    overlaps = states_A @ states_B.conj().T  

    return (np.abs(overlaps) ** 2)

def check_kernel(feature_map, X, shots=shots):
    '''
    Checks integrity of kernel making sure K(x,x)=1 for every training point

    returns smallest K(x,x) found
    '''
    circuits = [quantum_circuit(feature_map, x, x) for x in X] # list of circuits built off of only x (x vs x)

    compiled = qk.transpile(circuits, simulator, seed_transpiler=seed_simulator) # same circuit --> mathematical gates

    result = simulator.run(compiled, shots=shots).result() # runs simulator, .result() gives us object holding all outcomes of every circuit in job

    all_zeros = "0" * feature_map.num_qubits
    # --> "00" or "0000"

    values = [] # will hold all x vs x similarity score
    for k in range(len(circuits)):
        counts = result.get_counts(k) # {'00': 250, '01': 150, '10': 60, '11': 52 }
        values.append(counts.get(all_zeros, 0) / shots) # retrieves "00" measurements / shots --> probability score, appended to values
    
    return min(values) # correct kernel will have min(values) == 1

def circuit_stats(n_features, reps=repititions):
    '''
    ### Implementation Section ###
    circuit with:
        - qubit count
        - depth
        - two-qubit gate count
    '''
    fm = feature_map(n_features, reps) # feature map with n features, repeated 2 times on default

    x = np.linspace(0.1, 1.0, n_features)
    # --> [0.1, 1.0]            --> 2-feature point
    # --> [0.1, 0.4, 0.7, 1.0]  --> 4-feature point
    qc = quantum_circuit(fm, x, x[::-1]) # builds one real kernel circuit going through x as x, and inverse(x') as x'

    compiled = qk.transpile( 
                         qc, 
                         basis_gates=["cx", "rz", "sx", "x"], # universal set of gates
                         optimization_level=0, # no optimization
                         seed_transpiler=seed_simulator # fixed seed for repeatable results
                        )
    
    return {
        "qubits": qc.num_qubits,                            # number of qubits
        "logical_depth": qc.depth(),                        # depth of the circuit as designed (before transpiling)
        "transpiled_depth": compiled.depth(),               # depth post-transpiling 
        "two_qubit_gates": compiled.num_nonlocal_gates(),   # number of CNOT's after transpiling (nonlocal gates need multiple qubits in the gate)
        "circuit": qc,                                      # circuit itself
    }


### Full QSVM Run (Train, Test, Evaluate) ###
def run_qsvm(X_train, y_train, 
                X_test, y_test, 
                C=1.0, shots=shots, exact=False, verbose=True):
    '''
    exact=False: shot-based kernel on the AerSimulator (real experiment)
    exact=True:  statevector kernel (no shots).

    Returns a dict with every number we report, so the notebook can print it.
    '''

    # Turning all inputted data into numpy arrays
    X_train, X_test = np.asarray(X_train, float), np.asarray(X_test, float)
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)

    n = X_train.shape[1] # number of features

    # Validating data, making sure its properly scaled
    # --> must be within [0, pi]
    # 1e-9: tolerance for floating-point rounding
    if X_train.min() < 0 or X_train.max() > np.pi + 1e-9:
        print("WARNING: training features are outside [0, pi]; check preprocessing")

    model = QSVM(C=C, shots=shots, exact=exact).fit(X_train, y_train)

    K_test = model._kernel(X_test, X_train) # quantum kernel -> matrix of similarity scores between X_test and X_train. Used for predictions.

    # training accuracy (using score() would create new kernel)
    train_acc = float(np.mean(model.svm.predict(model.K_train) == y_train))
    # test accuracy
    test_acc = float(np.mean(model.svm.predict(K_test) == y_test))


    off_diag = ~np.eye(len(X_train), dtype=bool) # skip diagonal because its always 1
        # [[False,  True,  True],
        #  [ True, False,  True],
        #  [ True,  True, False]]

    results = {
        "n_features": n,                        # number of features 
        "n_train": len(X_train),                # length of training examples
        "n_test": len(X_test),                  # length of test examples
        "shots": "exact" if exact else shots,   # exact --> exact_kernel
        "train_acc": train_acc,                 # computed training accuracy
        "test_acc": test_acc,                   # computed test accuracy
        # measures overfitting: .9 (train) - .6 (test) = .3 --> training did much better, could be result of overfitting
        "generalization_gap": train_acc - test_acc,
        # average similarity of matrix (excluding diagonal), can tell us why our model is overfitting (if similarity between points are too low, every patient looks unrelated to every other, low generalization --> bad model)
        "mean_off_diagonal_K": float(model.K_train[off_diag].mean()) if off_diag.any() else float("nan"),
        # symmetric --> training model, A=B
        "is_symmetric": bool(np.allclose(model.K_train, model.K_train.T)),
    }

    if  not exact:
        # Checks that only make sense for the shot-based kernel
        K_exact = exact_kernel(model.feature_map, X_train) # statevector compuation
        
        # proves K(x,x) = 1 for every point, showing circuit correctness
        results["min_self_kernel"] = check_kernel(model.feature_map, X_train, shots)
        # compares measured kernel with exact one (gives idea on shot noise's influence)
        results["max_kernel_error_vs_exact"] = float(np.abs(model.K_train - K_exact).max())

    if verbose: # prints results dictionary 
        for key, value in results.items():
            print(f"{key:>28}: {value:.4f}" if isinstance(value, float) else f"{key:>28}: {value}")
    return results

### Repeating Trials of QSVM run###


'''
 ### QSVM Class ###

- packaging all functions in ordered steps that make sense 

__init__
- C : regularization parameter (toleration of mistakes)
# (Both SVM and QSVM get same C to not skew results)

-shots: 512,1024 number of shots that get ran through simulator

- exact : exact_kernel if True, quantum_kernel if False
'''

class QSVM:
    def __init__(self, C=1.0, shots=shots, 
                 exact=False): 
        # Regularization (default = 1)
        self.C = C
        self.shots = shots    
        self.exact = exact
        self.reps = repititions                      

        # a normal SVM that accepts a kernel matrix, will which be replaced by our quantum kernel
        self.svm = SVC(kernel="precomputed", C=C) 

    def _kernel(self, A, B=None):
        if self.exact:
            return exact_kernel(self.feature_map, A, B)
        return quantum_kernel(self.feature_map, A, B, self.shots)

    def fit(self, X_train, y_train): 
        '''
        Train the QSVM: --> X_train data points vs y_train true predictions
        - build the feature map
        - compute the training kernel (training points vs each other)
        - fit SVM on it (with quantum or exact kernel, dependent on self.exact)
        '''
        # turn inputted data into an numpy array
        X_train = np.asarray(X_train, dtype=float)

        # needed to build K_test when we run predctions
        self.X_train = X_train

        self.feature_map = feature_map(X_train.shape[1], self.reps) # build feature map

        # returns quantum kernel based on feature_map built above and X_train (A=B since we are using training points)
        self.K_train = self._kernel(X_train)

        # train model on svc using quantum kernel
        self.svm.fit(self.K_train, y_train)

        return self

    def predict(self, X): 
        # predict data points and return the list of predictions

        # A --> new data points (test), B --> training points (X_train)
        # Creates matrix of similarity scores between new points and training points
        K = self._kernel(np.asarray(X, dtype=float), self.X_train) # quantum_kernel
        
        return self.svm.predict(K) # array([1, 0, 1, 1, 0, ..., 1]) of predictions for each new data point

    def score(self, X, y):
        '''
        accuracy of our predictions

        self.predict(X) --> model's guess with X as data points
        np.asarray(y) --> true labels
        # compare models guess vs true labels
        # return (correc) / number_of_predictions) == accuracy
        '''
        
        return float(np.mean(self.predict(X) == np.asarray(y)))


### Visualizations and summary of circuits ###
def print_circuit_table():
    """Print qubits, depth and CNOT (cx) count for n=2, n=4"""
    print(f"{'Qubits':>6} | {'Depth before':>12} | {'Depth after':>11} | {'Two-qubit gates':>15}")
    for n in (2, 4):
        s = circuit_stats(n)
        print(f"{s['qubits']:>6} | {s['logical_depth']:>12} | {s['transpiled_depth']:>11} | {s['two_qubit_gates']:>15}")

def draw_smallest_circuit(path="kernel_circuit_2q.png"):
    """Save Figure 1, the 2-qubit kernel circuit as Qiskit draws it.

    Needs `pip install pylatexenc` for the matplotlib drawer.
    """
    fig = circuit_stats(2)["circuit"].draw("mpl", fold=-1)
    fig.savefig(path, bbox_inches="tight")
    return fig