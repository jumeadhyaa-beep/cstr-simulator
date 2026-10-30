# cstr-simulator

Simulator for a non-isothermal Continuous Stirred-Tank Reactor (CSTR) with a
first-order exothermic reaction A -> B and Arrhenius kinetics.

## Model

    dCA/dt = q/V (CAf - CA) - k(T) CA
    dT/dt  = q/V (Tf - T) + (-dH)/(rho Cp) k(T) CA + UA/(rho Cp V) (Tc - T)
    k(T)   = k0 exp(-E/(R T))

## Usage

```python
from cstr.model import CSTRParams, simulate, steady_states, is_stable, conversion

p = CSTRParams()
t, CA, T = simulate(p, x0=[0.5, 350.0], t_end=20.0)
for s in steady_states(p):
    print(s, "stable" if is_stable(s, p) else "unstable")
```

With the default parameters there is one stable steady state
(CA ~ 0.877 mol/L, T ~ 324.5 K) and two unstable ones.

## Tests

    pip install numpy scipy pytest
    python -m pytest

## Roadmap

Interactive UI, multiple-steady-state/oscillation exploration, PID control.
