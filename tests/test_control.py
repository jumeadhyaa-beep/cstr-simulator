import numpy as np

from cstr.control import PID, simulate_closed_loop
from cstr.model import CSTRParams, steady_states

OSC = CSTRParams(q=75.0, Tf=350.0, Tc=310.0)   # open loop: sustained oscillation


def test_pid_stabilizes_unstable_steady_state():
    T_sp = steady_states(OSC)[0][1]
    pid = PID(Kc=2.0, tau_i=5.0, bias=310.0)
    _, _, T, Tc = simulate_closed_loop(OSC, pid, T_sp, [0.5, 350.0], t_end=40.0)
    assert abs(T[-1] - T_sp) < 0.01
    assert np.ptp(T[-300:]) < 0.01
    assert np.isclose(Tc[-1], 310.0, atol=0.01)     # same Tc as the open-loop steady state


def test_output_respects_limits():
    pid = PID(Kc=100.0, tau_i=1.0, bias=300.0, u_min=290.0, u_max=320.0)
    _, _, _, Tc = simulate_closed_loop(OSC, pid, 380.0, [0.5, 350.0], t_end=10.0)
    assert Tc.min() >= 290.0 and Tc.max() <= 320.0


def test_integral_action_removes_offset_after_setpoint_step():
    p = CSTRParams()                                 # one stable steady state at ~324.5 K
    pid = PID(Kc=2.0, tau_i=5.0, bias=300.0)
    sp = lambda t: 324.5 if t < 10 else 330.0
    _, _, T, _ = simulate_closed_loop(p, pid, sp, [0.88, 324.5], t_end=60.0)
    assert abs(T[-1] - 330.0) < 0.05
