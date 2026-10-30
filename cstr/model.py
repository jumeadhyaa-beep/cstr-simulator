"""Non-isothermal CSTR with a first-order exothermic reaction A -> B.

States: x = [CA, T]
    dCA/dt = q/V (CAf - CA) - k(T) CA
    dT/dt  = q/V (Tf - T) + (-dH)/(rho Cp) k(T) CA + UA/(rho Cp V) (Tc - T)
    k(T)   = k0 exp(-E/(R T))
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


@dataclass(frozen=True)
class CSTRParams:
    q: float = 100.0        # feed flow rate [L/min]
    V: float = 100.0        # reactor volume [L]
    CAf: float = 1.0        # feed concentration [mol/L]
    Tf: float = 350.0       # feed temperature [K]
    Tc: float = 300.0       # coolant temperature [K]
    k0: float = 7.2e10      # pre-exponential factor [1/min]
    E_R: float = 8750.0     # activation energy / R [K]
    dH: float = -5.0e4      # heat of reaction [J/mol]
    rho: float = 1000.0     # density [g/L]
    Cp: float = 0.239       # heat capacity [J/(g K)]
    UA: float = 5.0e4       # heat transfer coefficient * area [J/(min K)]

    def with_(self, **changes) -> "CSTRParams":
        return replace(self, **changes)


def rate_constant(T, p: CSTRParams):
    return p.k0 * np.exp(-p.E_R / T)


def derivatives(t, x, p: CSTRParams):
    CA, T = x
    k = rate_constant(T, p)
    dCA = p.q / p.V * (p.CAf - CA) - k * CA
    dT = (p.q / p.V * (p.Tf - T)
          + (-p.dH) / (p.rho * p.Cp) * k * CA
          + p.UA / (p.rho * p.Cp * p.V) * (p.Tc - T))
    return [dCA, dT]


def simulate(p: CSTRParams, x0, t_end=10.0, n_points=500, params_at=None):
    """Integrate the model. `params_at(t)` may return time-varying params."""
    def rhs(t, x):
        return derivatives(t, x, params_at(t) if params_at else p)

    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(rhs, (0.0, t_end), x0, t_eval=t_eval, method="LSODA",
                    rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y[0], sol.y[1]


def conversion(CA, p: CSTRParams):
    return 1.0 - np.asarray(CA) / p.CAf


def steady_states(p: CSTRParams, n_guesses=40):
    """Find all distinct steady states by scanning initial guesses."""
    found = []
    for T0 in np.linspace(p.Tc, p.Tf + 150.0, n_guesses):
        # Start from the CA that balances mass at this temperature.
        k = rate_constant(T0, p)
        CA0 = p.q / p.V * p.CAf / (p.q / p.V + k)
        x, _, ier, _ = fsolve(lambda x: derivatives(0, x, p), [CA0, T0],
                              full_output=True, xtol=1e-12)
        if ier != 1 or not (0 <= x[0] <= p.CAf * 1.0001) or x[1] <= 0:
            continue
        if not any(np.allclose(x, s, rtol=1e-5, atol=1e-6) for s in found):
            found.append(x)
    return sorted(found, key=lambda s: s[1])


def is_stable(x, p: CSTRParams, eps=1e-6):
    """Stable if both Jacobian eigenvalues have negative real part."""
    x = np.asarray(x, dtype=float)
    J = np.zeros((2, 2))
    for j in range(2):
        d = np.zeros(2)
        d[j] = eps * max(1.0, abs(x[j]))
        J[:, j] = (np.array(derivatives(0, x + d, p))
                   - np.array(derivatives(0, x - d, p))) / (2 * d[j])
    return bool(np.all(np.linalg.eigvals(J).real < 0))
