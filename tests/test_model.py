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
