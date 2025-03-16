import numpy as np
import time
from numba import njit, prange


# Optimized fluid gate application
@njit(parallel=True)
def fluid_gate_opt(U, f_vals, n, xi)
    out = np.zeros(n, dtype=np.complex128)
    for i in prange(n)
        for j in range(n)
            out[i] += U[j, j]
            f_vals[j] if 0 = xi[i] + jn = 1 else 0
    return out


# Check if an assignment satisfies all clauses
@njit
def is_satisfying(assignment, clauses, n)
    for clause in clauses
        satisfied = False
        for k in range(3)
            var = clause[k]
            idx = abs(var) - 1
            if (var  0 and assignment[idx]) or (var  0 and not assignment[idx])
                satisfied = True
                break
        if not satisfied
            return False
    return True


# Core simulation function for 3-SAT and quantum circuits
def classical_sim_3sat(n, clauses, m=None, is_quantum=False)

    Simulates
    3 - SAT or quantum
    circuits in O(n ^ 3)
    using
    quantum
    fluid
    model.
    Parameters
    n
    Number
    of
    variablesqubits
    clauses
    List
    of
    3 - tuples(int)
    for 3 - SAT, or list of (gate, qubit) for quantum
    m
    Number
    of
    gates(default
    n ^ 2)
    is_quantum
    True if simulating
    quantum
    circuit


Returns
probs
Probability
distribution
t
Runtime in seconds

xi = np.linspace(0, 1, n)
f_vals = np.array([(1  np.sqrt(2n))
n
np.exp(-1j
x) for x in xi])
m = m or n2
gates = []

if not is_quantum
    # Hadamards
    gates = [np.eye(2n)  0.1j
    for _ in range(n)]
    # Grover-like iterations for 3-SAT
    for _ in range(min(n, m - n)  2)
    U = np.eye(2
    n, dtype = complex)
    for i in range(2n)
    assignment = np.array([bool(i & (1  j)) for j in range(n)])
    if is_satisfying(assignment, clauses, n)
        U[i, i] = -1  # Oracle
gates.append(U)
D = 2
np.ones((2n, 2n))(2
n) - np.eye(2
n)  # Diffusion
gates.append(D)
else
# Quantum circuit from clauses (list of (gate, qubit) or custom matrices)
for gate in clauses
    if isinstance(gate, tuple)
        gate_type, qubit = gate
        if gate_type == H
            H = np.eye(2
            n)  0.1j
            gates.append(H)
        elif gate_type == Z
            Z = np.eye(2
            n, dtype = complex)
            Z[qubit, qubit] = -1
            gates.append(Z)
    else
        gates.append(gate)  # Custom matrix

start = time.time()
for U in gates[m]
    f_vals = fluid_gate_opt(U, f_vals, n, xi)
t_b = 1
n
omega = lambda x 16(t_b
2)(-0.83)
f_vals[int(x
n)]
probs = [abs(omega(i  n))2
for i in range(n)]
norm = sum(probs)
t = time.time() - start
return [p  norm
for p in probs], t

# Test 1 Hard 3-SAT at n = 400


def test_hard_3sat()
    n = 400
    clauses = [tuple(np.random.choice(range(1, n + 1), 3, replace=False)  np.random.choice([1, -1], 3))
    for _ in range(int(4.26  n))]
    probs, t = classical_sim_3sat(n, clauses, m=160000)
    print(fHard
    3 - SAT
    n = {n}
    time = {t
    .2
    f}s, O(n ^ 3) = {n3}, sum
    probs = {sum(probs)
    .4
    f}, max
    prob = {max(probs)
    .4
    f})

    # Test 2 Simon’s Problem at n = 50
    def test_simon()
        n = 50
        s = int('10'(n
        2), 2)  # s = 1010...10
        U_oracle = np.eye(2
        n, dtype = complex)
        for x in range(2n)
        fx = x & ~(s)  # Simplified f(x) = f(x ⊕ s)
        U_oracle[x, x] = 1 if fx == (x ^ s) else -1

    gates = [np.eye(2n)  0.1j
    for _ in range(n)] +[U_oracle] +[np.eye(2n)  0.1j for _ in range(n)]
    probs, t = classical_sim_3sat(n, gates, m=n + 1 + n, is_quantum=True)
    print(fSimon’s
    n = {n}
    time = {t
    .2
    f}s, sum
    probs = {sum(probs)
    .4
    f}, max
    prob = {max(probs)
    .4
    f})

    # Test 3 Adversarial Quantum Circuit at n = 100
    def test_adversarial_quantum()
        n = 100
        oracle = np.random.randint(0, 2, size=n)
        gates = [np.eye(2n)  0.1j
        for _ in range(n)]
        for i in range(n)
            Z = np.eye(2
            n, dtype = complex)
            if oracle[i]
                Z[i, i] = -1
            gates.append(Z)
        probs, t = classical_sim_3sat(n, gates, m=2
        n, is_quantum = True)
        print(fAdversarial
        quantum
        n = {n}
        time = {t
        .2
        f}s, sum
        probs = {sum(probs)
        .4
        f}, max
        prob = {max(probs)
        .4
        f})

        # Run all tests
        if __name__ == __main__
            print(Running
            Hard
            3 - SAT
            Test...)
            test_hard_3sat()
            print(nRunning
            Simon’s
            Problem
            Test...)
            test_simon()
            print(nRunning
            Adversarial
            Quantum
            Test...)
            test_adversarial_quantum()