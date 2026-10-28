"""Relative EM stop: a toy history stops on time, and None keeps the old loop."""
import numpy as np

from COMIND_transformer.subtyping_em_transformer import (
    SubtypingEM,
    em_relative_stop,
    replay_lse_stop,
)


def _walk(start, steps):
    """steps are relative drops: loss <- loss * (1 - drop)."""
    h = [float(start)]
    for drop in steps:
        h.append(h[-1] * (1.0 - drop))
    return h


def test_toy_history_stops_at_the_expected_iteration():
    # 14 large steps, then three relative gains of 1e-6. min_iter=15 and
    # patience=3, so the third small step (iteration 17) is the first legal stop.
    drops = [0.1] * 14 + [1e-6] * 3 + [1e-6] * 5
    h = _walk(1000.0, drops)
    out = replay_lse_stop(h, stop_rel_tol=1e-4, stop_patience=3, min_iter=15, max_iter=80)
    assert out["n_iter"] == 17
    assert out["stop_reason"] == "converged"
    assert out["loss"] == h[17]


def test_patience_resets_after_a_large_step():
    drops = [0.1] * 20 + [1e-6] * 2 + [0.1] + [1e-6] * 3
    h = _walk(1000.0, drops)
    out = replay_lse_stop(h, stop_rel_tol=1e-4, stop_patience=3, min_iter=15, max_iter=80)
    assert out["n_iter"] == 26
    assert out["stop_reason"] == "converged"


def test_first_step_does_not_count_toward_patience():
    # Previous loss is non-finite on the first accepted step, so three small
    # steps need one extra iteration when min_iter is already satisfied.
    h = _walk(1000.0, [1e-6] * 6)
    out = replay_lse_stop(h, stop_rel_tol=1e-4, stop_patience=3, min_iter=3, max_iter=80)
    assert out["n_iter"] == 4
    assert out["stop_reason"] == "converged"


def test_stop_rel_tol_none_matches_the_old_loop():
    h = _walk(100.0, [1e-8] * 100)
    capped = replay_lse_stop(h, stop_rel_tol=None, stop_patience=3, min_iter=15, max_iter=80)
    assert capped["n_iter"] == 80
    assert capped["stop_reason"] == "max_iter"
    assert capped["loss"] == h[80]

    short = replay_lse_stop(h[:41], stop_rel_tol=None, stop_patience=3, min_iter=15, max_iter=80)
    assert short["n_iter"] == 40
    assert short["stop_reason"] == "end"
    assert short["loss"] == h[40]

    # The decision itself never stops when the tolerance is off, even if a
    # caller has already accumulated small steps.
    stop, n_small = em_relative_stop(10.0, 10.0 * (1 - 1e-8), 5, 100, None, 3, 15)
    assert stop is False
    assert n_small == 0


def test_max_iter_still_caps_a_fit_that_is_still_moving():
    h = _walk(1000.0, [0.1] * 30)
    out = replay_lse_stop(h, stop_rel_tol=1e-4, stop_patience=3, min_iter=15, max_iter=10)
    assert out["n_iter"] == 10
    assert out["stop_reason"] == "max_iter"


def test_defaults_leave_the_stop_off():
    em = SubtypingEM(K=np.eye(2), verbose=0)
    assert em.stop_rel_tol is None
    assert em.stop_patience == 3
    assert em.min_iter == 15
    assert em.n_iter_ == 0
    assert em.stop_reason_ is None
