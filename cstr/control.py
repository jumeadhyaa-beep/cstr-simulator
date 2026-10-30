"""PID control of reactor temperature by manipulating coolant temperature Tc."""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .model import CSTRParams, derivatives


@dataclass
class PID:
    Kc: float = 2.0            # proportional gain [K coolant / K error]
    tau_i: float = 5.0         # integral time [min]; None disables the I term
    tau_d: float = 0.0         # derivative time [min]
    bias: float = 300.0        # controller output at zero error [K]
    u_min: float = 270.0       # coolant temperature limits [K]
    u_max: float = 350.0
    _integral: float = 0.0
    _prev_e: float = None

    def reset(self):
        self._integral, self._prev_e = 0.0, None

    def step(self, setpoint, measurement, dt):
        e = setpoint - measurement
        d = 0.0 if self._prev_e is None else (e - self._prev_e) / dt
        self._prev_e = e
        i_term = self._integral / self.tau_i if self.tau_i else 0.0
        u = self.bias + self.Kc * (e + i_term + self.tau_d * d)
        u_sat = min(max(u, self.u_min), self.u_max)
        # Anti-windup: only integrate while the output is not saturated
        # (or when the error would pull it back inside the limits).
        if u == u_sat or (u > u_sat) == (e < 0):
            self._integral += e * dt
        return u_sat


def simulate_closed_loop(p: CSTRParams, pid: PID, setpoint, x0, t_end=30.0, dt=0.02):
    """Simulate with PID-driven Tc (piecewise constant over each dt).

    `setpoint` is a float or a function of time. Returns t, CA, T, Tc arrays.
    """
    pid.reset()
    sp = setpoint if callable(setpoint) else (lambda t: setpoint)
    n = int(round(t_end / dt))
    t_arr = np.arange(n + 1) * dt
    CA, T, Tc = np.empty(n + 1), np.empty(n + 1), np.empty(n + 1)
    x = np.asarray(x0, dtype=float)
    CA[0], T[0] = x
    for i in range(n):
        Tc[i] = pid.step(sp(t_arr[i]), x[1], dt)
        pi = p.with_(Tc=Tc[i])
        sol = solve_ivp(lambda t, y: derivatives(t, y, pi), (0, dt), x,
                        method="LSODA", rtol=1e-8, atol=1e-10)
        x = sol.y[:, -1]
        CA[i + 1], T[i + 1] = x
    Tc[n] = Tc[n - 1]
    return t_arr, CA, T, Tc
