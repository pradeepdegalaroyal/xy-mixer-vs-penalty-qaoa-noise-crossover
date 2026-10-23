import numpy as np

from xynoise.core import Qaoa, metrics, random_problem

PAR = np.array([0.3, 0.5, 0.2, 0.4])


def test_xy_preserves_feasibility_noiseless():
    p = random_problem(6, 3, 0.5, 0)
    pr = Qaoa(p, "xy", 2).run(PAR)
    assert abs(pr.sum() - 1) < 1e-9 and pr[p.feasible].sum() > 1 - 1e-9


def test_noise_leaks_out_of_subspace():
    p = random_problem(6, 3, 0.5, 0)
    pr = Qaoa(p, "xy", 2).run(PAR, 0.02, 50, np.random.default_rng(0))
    assert abs(pr.sum() - 1) < 1e-9 and pr[p.feasible].sum() < 1 - 1e-4


def test_training_and_metrics_bounded():
    p = random_problem(6, 3, 0.5, 1)
    qa = Qaoa(p, "x", 1)
    th = qa.train(2, 0)
    m = metrics(p, qa.run(th), 50, np.random.default_rng(0))
    assert 0 <= m["ar_feasible"] <= 1 and 0 <= m["p_feasible"] <= 1
