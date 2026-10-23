# Method

- **Problem**: densest-k-subgraph, G(8, 0.5) random graphs, k = 4 (70 feasible of 256 states). 5 graph seeds.
- **Methods** (depth p = 1, 2, 3): penalty QAOA (X mixer, uniform init, penalty 2·(|x|−k)²) — the standard
  baseline; XY-ring-mixer QAOA with Dicke initial state (constrained baseline from the literature).
- **Classical baseline**: random-restart swap local search (200 cost evaluations). **Null**: best of 100
  random feasible subsets.
- **Training**: noiseless, COBYLA, 6 random restarts; evaluation of the fixed parameters under noise
  (parameters are not retrained under noise).
- **Noise**: Pauli-trajectory depolarising model (60 trajectories). Each qubit gets a random Pauli per layer
  with probability 1−(1−p2)^c, c = number of CX gates touching it in that layer (cost layer: penalty 2(n−1),
  XY 2·deg; XY ring mixer: 8; X mixer: 0). Dicke-state preparation is assumed ideal (stated limitation).
  Sweep p2 ∈ {0, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2}.
- **Metrics**: P(feasible); feasible-weighted approximation ratio Σ P(x)·C(x)·1[feasible] / C_opt;
  best feasible of 100 shots / C_opt.
- **Ablation**: XY mixer with uniform (non-Dicke) initial state, depth 2. **Scaling in depth**: p = 1–3.
