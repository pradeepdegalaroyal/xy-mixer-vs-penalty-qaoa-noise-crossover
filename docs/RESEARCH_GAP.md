# Research gap

## State of the art
Quantum Approximate Optimization Algorithm (QAOA) is a leading noisy intermediate-scale quantum (NISQ) candidate for combinatorial optimisation, but in its standard form it handles hard
constraints poorly (Wołk et al., 2027, doi:10.1016/j.ins.2026.124187). Two routes exist: add a quadratic
penalty to the cost (cheap circuits, but the sampled distribution leaks into infeasible states), or restrict
the mixer to the feasible subspace, e.g. an XY (excitation-number-preserving) mixer on a Dicke initial state (the output is feasible by
construction, but circuits are much larger). The same work notes that the larger circuit "reduces noise
robustness" and proposes a gate-count-reduced hypercube mixer. On the hardware side, parallel two-qubit gates
at about 99.5% fidelity have been reported on neutral atoms (Evered et al., 2023, doi:10.1038/s41586-023-06481-y),
which puts realistic two-qubit error near 5e-3.

## Gap
The trade-off "constrained mixer = feasible but deeper" is stated, but I could not find in the provided
literature a *controlled, equal-footing measurement of where it flips*: at what two-qubit error rate does the
gate overhead of a constrained mixer cancel its feasibility advantage, compared with penalty QAOA, and does
the Dicke-state initial state matter?

## Hypothesis (falsifiable)
H1: with parameters trained noiselessly, depth-2 XY-mixer QAOA keeps a higher feasible-weighted approximation
ratio than penalty QAOA at every p2 up to 2e-2, but its *own noiseless value* is lost
(it falls below penalty QAOA's noiseless value) at some p2* inside the 1e-3 to 1e-2 range.
H2: removing the Dicke initial state removes the advantage.
Success metric: feasible-weighted approximation ratio (mean ± std over 5 random graphs), plus P(feasible).
Scale: exact statevector / Pauli-trajectory simulation, n = 8 qubits, k = 4 — a simulation study, not hardware.
