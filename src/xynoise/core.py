"""Statevector QAOA for densest-k-subgraph with Pauli-trajectory noise."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.optimize import minimize


def bits(n: int) -> np.ndarray:
    """(2^n, n) array of bit values; qubit q is bit q of the index."""
    idx = np.arange(2**n)
    return ((idx[:, None] >> np.arange(n)) & 1).astype(np.int8)


@dataclass
class Problem:
    """Densest-k-subgraph: maximise edges inside a size-k vertex subset."""

    graph: nx.Graph
    k: int

    def __post_init__(self) -> None:
        self.n = self.graph.number_of_nodes()
        self.B = bits(self.n)
        self.weight = self.B.sum(1)
        c = np.zeros(2**self.n)
        for i, j in self.graph.edges:
            c += self.B[:, i] * self.B[:, j]
        self.cost = c
        self.feasible = self.weight == self.k
        self.opt = c[self.feasible].max()

    def penalised(self, lam: float) -> np.ndarray:
        """Cost to maximise: edges minus lam*(|x|-k)^2."""
        return self.cost - lam * (self.weight - self.k) ** 2


def random_problem(n: int, k: int, p_edge: float, seed: int) -> Problem:
    """Random G(n, p) instance with at least k edges (deterministic in seed)."""
    for s in itertools.count(seed * 1000):
        g = nx.gnp_random_graph(n, p_edge, seed=s)
        if g.number_of_edges() >= k:
            return Problem(g, k)
    raise RuntimeError


def _pauli_x_sum(n: int) -> np.ndarray:
    d = 2**n
    h = np.zeros((d, d))
    idx = np.arange(d)
    for q in range(n):
        h[idx ^ (1 << q), idx] += 1
    return h


def _xy_ring(n: int) -> np.ndarray:
    """Sum over ring of (XX+YY)/2: hops a particle between neighbours, preserves Hamming weight."""
    d = 2**n
    h = np.zeros((d, d))
    for q in range(n):
        a, b = q, (q + 1) % n
        for x in range(d):
            if ((x >> a) & 1) != ((x >> b) & 1):
                h[x ^ (1 << a) ^ (1 << b), x] += 1
    return h


class Qaoa:
    """QAOA on a Problem with mixer 'x' (penalty) or 'xy' (constrained ring)."""

    def __init__(self, prob: Problem, mixer: str, depth: int, init: str = "auto", lam: float = 2.0):
        self.prob, self.mixer, self.depth, self.n = prob, mixer, depth, prob.n
        self.phase_cost = prob.penalised(lam) if mixer == "x" else prob.cost.astype(float)
        h = _pauli_x_sum(self.n) if mixer == "x" else _xy_ring(self.n)
        self.evals, self.evecs = np.linalg.eigh(h)
        d = 2**self.n
        if init == "auto":
            init = "uniform" if mixer == "x" else "dicke"
        psi = prob.feasible.astype(complex) if init == "dicke" else np.ones(d, complex)
        self.psi0 = psi / np.linalg.norm(psi)
        deg = dict(prob.graph.degree)
        if mixer == "x":  # penalty: all-pairs ZZ (2 CX each); X mixer is CX-free
            self.cx_cost = np.full(self.n, 2 * (self.n - 1))
            self.cx_mix = np.zeros(self.n, int)
        else:  # ZZ on graph edges; ring of RXX+RYY (4 CX per pair, 2 pairs per qubit)
            self.cx_cost = np.array([2 * deg[q] for q in range(self.n)])
            self.cx_mix = np.full(self.n, 8)
        idx = np.arange(d)
        self.perm = [idx ^ (1 << q) for q in range(self.n)]
        self.zs = [1 - 2 * ((idx >> q) & 1) for q in range(self.n)]

    def _mix(self, psi: np.ndarray, beta: float) -> np.ndarray:
        """exp(-i beta H) applied to rows of psi via the eigendecomposition of H."""
        return ((psi @ self.evecs) * np.exp(-1j * beta * self.evals)) @ self.evecs.T

    def _noise(self, psi: np.ndarray, cx: np.ndarray, p2: float, rng: np.random.Generator) -> np.ndarray:
        """Depolarising Pauli error per qubit with prob 1-(1-p2)^cx (rows = trajectories)."""
        t = psi.shape[0]
        for q in range(self.n):
            hit = rng.random(t) < 1 - (1 - p2) ** cx[q]
            if not hit.any():
                continue
            e = rng.integers(1, 4, t) * hit  # 0 none, 1 X, 2 Y, 3 Z
            fx = (e == 1) | (e == 2)
            fz = (e == 2) | (e == 3)
            psi = np.where(fx[:, None], psi[:, self.perm[q]], psi)
            psi = np.where(fz[:, None], psi * self.zs[q], psi)
        return psi

    def run(self, params: np.ndarray, p2: float = 0.0, traj: int = 1, rng=None) -> np.ndarray:
        """Output probability distribution (mean over Pauli trajectories)."""
        g, b = params[: self.depth], params[self.depth:]
        t = traj if p2 > 0 else 1
        psi = np.tile(self.psi0, (t, 1))
        for gl, bl in zip(g, b):
            psi = psi * np.exp(-1j * gl * self.phase_cost)
            if p2 > 0:
                psi = self._noise(psi, self.cx_cost, p2, rng)
            psi = self._mix(psi, bl)
            if p2 > 0:
                psi = self._noise(psi, self.cx_mix, p2, rng)
        return (np.abs(psi) ** 2).mean(0)

    def train(self, restarts: int, seed: int) -> np.ndarray:
        """Noiseless optimisation of the expected phase cost; best of several random starts."""
        rng = np.random.default_rng(seed)
        best, bv = None, -np.inf
        for _ in range(restarts):
            x0 = rng.uniform(0, np.pi, 2 * self.depth)
            r = minimize(lambda x: -self.run(x) @ self.phase_cost, x0, method="COBYLA",
                         options={"maxiter": 150})
            if -r.fun > bv:
                best, bv = r.x, -r.fun
        return best


def metrics(prob: Problem, probs: np.ndarray, shots: int, rng: np.random.Generator) -> dict:
    """P(feasible), feasible-weighted approximation ratio, best-of-shots feasible ratio."""
    pf = float(probs[prob.feasible].sum())
    ar = float((probs * prob.cost * prob.feasible).sum() / prob.opt)
    s = rng.choice(len(probs), size=shots, p=probs / probs.sum())
    ok = prob.feasible[s]
    best = float(prob.cost[s][ok].max() / prob.opt) if ok.any() else 0.0
    return {"p_feasible": pf, "ar_feasible": ar, "best_of_shots": best}


def classical_baseline(prob: Problem, rng: np.random.Generator, budget: int) -> float:
    """Random-restart swap local search with `budget` cost evaluations; ratio to optimum."""
    n, k = prob.n, prob.k
    adj = nx.to_numpy_array(prob.graph)

    def val(s):
        return adj[np.ix_(s, s)].sum() / 2

    best, used = 0.0, 0
    while used < budget:
        s = list(rng.choice(n, k, replace=False))
        cur = val(s)
        used += 1
        improved = True
        while improved and used < budget:
            improved = False
            for i in range(k):
                for v in range(n):
                    if v in s:
                        continue
                    t = s.copy()
                    t[i] = v
                    w = val(t)
                    used += 1
                    if w > cur:
                        s, cur, improved = t, w, True
        best = max(best, cur)
    return best / prob.opt


def random_feasible_baseline(prob: Problem, rng: np.random.Generator, shots: int) -> float:
    """Best of `shots` uniformly random size-k subsets (the null any sampler must beat)."""
    f = np.flatnonzero(prob.feasible)
    s = rng.choice(f, shots)
    return float(prob.cost[s].max() / prob.opt)
