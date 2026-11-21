import qiskit as qk
from qiskit_aer import AerSimulator
from qiskit.circuit.library import zz_feature_map
import numpy as np

'''
 Purpose of this file:
    1. Take in preprocessed data and features  (n=2,4 where n is the number of features)
    2. Implement QSVM (kernel is main difference from SVM)
    3. Train and test model using train_test_split

Assignment requires:
Fixing of random seeds
such as seed_simulator and seed_transpiler in Qiskit, so a rerun gives the same numbers.


'''

seed_simulator = 42 # rerun gives same number with fixed seed
shots = [512,1024] # different amount of shots to test for accuracy
repititions = 2 # repitions of ZZ feature map to "entangle" features

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
        --> This is done because training should only be done on training points

    Training gives us a neat trick, since the data points are symmetric (we are comparing A to A), we only need to compare half of the matrix (upper triangle), and can copy over the rest of it to complete it, saving time on training
    '''

    symmetric = (B is None) # if B is None return True, else False

    if symmetric:
        B = A
        K = np.eye(len(A)) # Creates diagonal using just A
    else:
        K = np.zeroes(len(A),len(B)) # Creates diagonal with A and B on opposing sides

    circuits, positions = [], [] # since we will be running the kernel in batches, we need to hold all of the circuits and positions in lists

    for i in range(len(A)): # building list of circuits
       
        if symmetric: # upper triangle --> only if symmetric
            start = i + 1
        else: start = 0      
        
        for j in range(start, len(B)):
            circuits.append( quantum_circuit(feature_map, A[i], B[j]) )
            positions.append( (i, j) )

    # Print check
    print("circuits: ", circuits)
    print("\n---------------------\n")
    print("positions: ", positions)

    compiled = qk.transpile(circuits, SIMULATOR, seed_transpiler=SEED)
    result = SIMULATOR.run(compiled, shots=shots).result()

