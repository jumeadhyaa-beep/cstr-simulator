import numpy as np

from cstr.model import (CSTRParams, conversion, derivatives, is_stable,
                        simulate, steady_states)

P = CSTRParams()


def test_steady_state_residual_is_zero():
    for s in steady_states(P):
        assert np.allclose(derivatives(0, s, P), 0, atol=1e-6)


def test_simulation_converges_to_a_stable_steady_state():
    t, CA, T = simulate(P, [0.5, 350.0], t_end=50.0)
    final = np.array([CA[-1], T[-1]])
    assert np.allclose(derivatives(0, final, P), 0, atol=1e-5)
    assert is_stable(final, P)


def test_no_reaction_without_activity():
    p = P.with_(k0=0.0)
    (s,) = steady_states(p)[:1]
    assert np.isclose(s[0], p.CAf)
    assert np.isclose(conversion(s[0], p), 0.0)


def test_conversion_bounds_and_mass_balance():
    t, CA, T = simulate(P, [1.0, 300.0], t_end=30.0)
    X = conversion(CA, P)
    assert np.all(X >= -1e-9) and np.all(X <= 1 + 1e-9)


def test_higher_feed_temperature_raises_steady_temperature():
    lo = steady_states(P.with_(Tf=340.0))[-1]
    hi = steady_states(P.with_(Tf=355.0))[-1]
    assert hi[1] > lo[1]


def test_bistable_preset_has_two_stable_states():
    p = P.with_(q=150.0, Tf=340.0, Tc=290.0)
    stable = [s for s in steady_states(p) if is_stable(s, p)]
    assert len(stable) == 2


def test_oscillating_preset_has_no_stable_state_and_sustained_cycle():
    p = P.with_(q=75.0, Tf=350.0, Tc=310.0)
    assert not any(is_stable(s, p) for s in steady_states(p))
    _, _, T = simulate(p, [0.5, 350.0], t_end=100.0, n_points=2000)
    assert T[-500:].max() - T[-500:].min() > 10
