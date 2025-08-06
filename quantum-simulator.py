import numpy as np

# states 
zero = np.array([[1], [0]], dtype=complex)
one = np.array([[0], [1]], dtype=complex)
plus = (zero + one) / np.sqrt(2)
minus = (zero - one) / np.sqrt(2)

zero_zero = np.kron(zero, zero)
one_one = np.kron(one, one)
plus_plus = np.kron(plus, plus)
minus_minus = np.kron(minus, minus)

plus_minus = np.kron(plus, minus)
minus_plus = np.kron(minus, plus)  
zero_one = np.kron(zero, one)
one_zero = np.kron(one, zero)

zero_plus = np.kron(zero, plus)
zero_minus = np.kron(zero, minus)
one_plus = np.kron(one, plus)
one_minus = np.kron(one, minus)
minus_zero = np.kron(minus, zero)  
minus_one = np.kron(minus, one)
plus_zero = np.kron(plus, zero)
plus_one = np.kron(plus, one)

# gates 
X = np.array([[0, 1],
              [1, 0]], dtype=complex)
Y = np.array([[0, -1j],
[1j, 0]], dtype=complex)
Z = np.array([[1, 0],
              [0, -1]], dtype=complex)
H = (1 / np.sqrt(2)) * np.array([[1, 1],
                                 [1, -1]], dtype=complex)
T = np.array([[1, 0],
              [0, np.exp(1j * np.pi / 4)]], dtype=complex)
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)

CZ = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, -1]], dtype=complex)

I = np.eye(2, dtype=complex)

'''
gates = {
    'X': X,
    'Y': Y,
    'Z': Z,
    'H': H,
    'T': T,
    'CNOT': CNOT,
    'CZ': CZ,
    'I': I
}

for name, matrix in gates.items():
    print(f'{name}:\n{matrix}\n\n')
'''

'''
# testing 
result = H @ zero  # superposition circuit
print('|+> = ',plus,'\n')
print(f'Result of H|0> is:\n{result}\n\n') 

result2 = CNOT @ one_one
print('|10> =', one_zero, '\n')
print(f'Result of CNOT|11> is:\n{result2}\n')
'''


# Bell States Circuit starting with x=|0> y=|0>
'''
(1) H|0> = |+>
(2) CNOT|+0> = |Φ+> as CNOT flips second qubit if first one is |1> and |+> = (|0> + |1>)/sqrt(2)

result_1 = H @ zero   # same with np.dot() and np.matmul()
result_2 = CNOT @ np.kron(result_1, zero)  # CNOT on |+0> where result_1 is |+> obvi

label = "|Φ+> ="
matrix_str = str(result_2)
matrix_lines = matrix_str.splitlines()
print()
print(f'Bell States Circuit:')
print(f'{label} {matrix_lines[0]}')
for line in matrix_lines[1:]:
    print(' ' * len(label), line)
print()
'''

# bell states 
Phi_plus = (zero_zero + one_one)/np.sqrt(2)
Phi_minus = (zero_zero - one_one)/np.sqrt(2)
Psi_plus = (zero_one + one_zero)/np.sqrt(2)
Psi_minus = (zero_one - one_zero)/np.sqrt(2)

def bell_circuit(initial_state_1, initial_state_2):
    state_1 = H @ initial_state_1
    bell_state = CNOT @ np.kron(state_1, initial_state_2)
    return bell_state

label = "|Φ+> ="
ms = str(bell_circuit(zero, zero))
ml = ms.splitlines()
print()
print(f'Bell States Circuit:')
print(f'{label} {ml[0]}')
for line in ml[1:]:
    print(' ' * len(label), line)
print()


# quantum teleportation circuit

'''
Psi_C is the unknown qubit state to be teleported: Ψc = a|0> + b|1>
Phi_AB_plus is the entangled Bell state shared between Alice and Bob: |Φ+> = (|00> + |11>)/√2

=> so: 
(1) CNOT @ |Psi_c,Phi_A_plus>
(2) H @ (first qubit from the result of (1) aka Psi_c) => cnot does not change the first qubit
(3) measurements top 2 wires  
(4) Bob's corrections: 00 -> no change, 01 -> Z, 10 -> X, 11 -> XZ
'''

def teleportation_circuit(psi_c):

    state = np.kron(psi_c, Phi_plus)  # full initial state
    CNOT_dim = np.kron(CNOT, I)
    state = CNOT_dim @ state

    Hc = np.kron(H, np.kron(I, I))
    state = Hc @ state

    probs = np.abs(state.flatten())**2

    rng = np.random.default_rng()
    outcome = rng.choice(8, p=probs) 
    print(f'Outcome: {outcome} with probabilities {probs}')

    collapsed = np.zeros_like(state)
    collapsed[outcome, 0] = 1.0

    bits = format(outcome, '03b')
    a, b, c = int(bits[0]), int(bits[1]), int(bits[2])
    print(a,b,c)

    bob_qubit = None
    if a == 0 and b == 0:  # outcome 00 → no correction
        bob_qubit = psi_c
        print('No correction applied.')
    elif a == 0 and b == 1:  # outcome 01 → apply X
        bob_qubit = X @ psi_c
        print('Z correction applied.')
    elif a == 1 and b == 0:  # outcome 10 → apply Z
        bob_qubit = Z @ psi_c
        print('X correction applied.')
    elif a == 1 and b == 1:  # outcome 11 → apply XZ
        bob_qubit = X @ (Z @ psi_c)
        print('XZ correction applied.')

    return bits, bob_qubit

teleportation_circuit(tryy_zero := np.array([[1], [0]], dtype=complex))
print(f'Teleportation Circuit for |0>:\n{tryy_zero.flatten()}\n')

teleportation_circuit(tryy_one := np.array([[0], [1]], dtype=complex))
print(f'Teleportation Circuit for |1>:\n{tryy_one.flatten()}\n')

teleportation_circuit(tryy_one := np.array([[1], [1]], dtype=complex)* 1/np.sqrt(2))
print(f'Teleportation Circuit for |+>:\n{tryy_one.flatten()}\n')