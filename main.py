from QSVM import run_qsvm, print_circuit_table, draw_smallest_circuit, shots

import numpy as np

def main():
    '''
    Implementation section: circuit size table and Figure 1
    '''
    print_circuit_table()

    draw_smallest_circuit("kernel_circuit_2q.png")


if __name__ == "__main__":
    main()