# Quantum_Programming_Project

QSVM with a ZZ-feature-map quantum kernel vs. a classical RBF SVM on scikit-learn's breast cancer dataset, trained on small amounts of data (PH-UY 2012 A course project).

## Files

- `QSVM.py`: feature map, kernel circuit, shot-based and exact (statevector) kernels, QSVM class, circuit statistics
- `preprocessing.py`: data loading, fixed stratified 80/20 split, standardize → PCA → scale to [0, π] (fitted on training data only)
- `experiments.py`: classical RBF SVM baseline, test-case runner, results tables
- `main.py`: runs everything and prints every reported number
- `Phase3.ipynb`: the Phase 3 submission (code, outputs and the four written sections)

## How to run

```bash
pip install -r requirements.txt
python main.py                 # prints the circuit table and all results, saves kernel_circuit_2q.png
jupyter notebook Phase3.ipynb  # or run the notebook top to bottom (needs `pip install notebook`)
```

Seeds are fixed (`seed_simulator = seed_transpiler = 42`, `random_state = 42`), so reruns print the same numbers. A full run takes about 15 seconds on a laptop.
