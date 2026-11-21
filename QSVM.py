import qiskit as qk
from qiskit_aer import AerSimulator
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
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
shots = 1024 # different amount of shots to test for accuracy
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
        K = np.zeros( (len(A),len(B)) ) # Creates diagonal with A and B on opposing sides
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
    print("circuits: ", circuits)
    print("\n---------------------\n")
    print("positions: ", positions)

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
        1. k = 0, (i, j) = (0, 1) --> compare patient 1 vs patient 2
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
        counts = result.get_counts(k) # gets counts for circuit[k], taken from simulator trained on 512,1024 shots
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

def circuit_stats(n_features, reps=repititions): # 
    '''
    ### Implementation Section ###
    circuit with:
        - qubit count
        - depth
        - two-qubit gate count
    '''
    fm = feature_map(n_features, reps) # feature map with 

    x = np.linspace(0.1, 1.0, n_features)
    # --> [0.1, 1.0]            --> 2-feature point
    # --> [0.1, 0.4, 0.7, 1.0]  --> a 4-feature point
    qc = quantum_circuit(fm, x, x[::-1]) # builds one real kernel circuit going through x as x and inverse(x) as x'

    compiled = qk.transpile( 
                         qc, 
                         basis_gates=["cx", "rz", "sx", "x"], # universal set of gates
                         optimization_level=0, 
                         seed_transpiler=seed_simulator
                        )
    
    return {
        "qubits": qc.num_qubits,                            # number of qubits
        "logical_depth": qc.depth(),                        # depth of the circuit as designed (before transpiling)
        "transpiled_depth": compiled.depth(),               # depth post-transpiling 
        "two_qubit_gates": compiled.num_nonlocal_gates(),   # number of CNOT's after transpiling (nonlocal gates need multiple qubits in the gate)
        "circuit": qc,                                      # circuit itself
    }
