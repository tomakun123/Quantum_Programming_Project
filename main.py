from QSVM import print_circuit_table, draw_smallest_circuit
from experiments import run_results
from preprocessing import print_split_summary


def main():
    '''
    Implementation section: circuit size table and Figure 1
    '''
    print_circuit_table()

    draw_smallest_circuit("kernel_circuit_2q.png")

    '''
    Results section: 2 and 4 qubits with 10 training samples, QSVM vs classical RBF SVM
    '''
    print_split_summary()
    run_results()


if __name__ == "__main__":
    main()
