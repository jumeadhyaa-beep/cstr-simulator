# cstr-simulator

[![tests](https://github.com/jumeadhyaa-beep/cstr-simulator/actions/workflows/tests.yml/badge.svg)](https://github.com/jumeadhyaa-beep/cstr-simulator/actions/workflows/tests.yml)

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

## Interactive UI

Open `web/index.html` in a browser (no build step). Sliders for q, Tf, Tc, CAf and UA
change the parameters live; plots show T(t) and the CA-T phase portrait.
Preset buttons load interesting regimes: bistability (q=150, Tf=340, Tc=290:
stable states at 314 K and 388 K; click the phase portrait to choose the start
point) and sustained oscillations (q=75, Tf=350, Tc=310).

## PID control

`cstr/control.py` provides a PID controller (anti-windup, output limits
270-350 K) that manipulates the coolant temperature Tc to hold the reactor
temperature at a setpoint:

```python
from cstr.control import PID, simulate_closed_loop
p = CSTRParams(q=75, Tf=350, Tc=310)          # open loop: sustained oscillation
t, CA, T, Tc = simulate_closed_loop(p, PID(Kc=2, tau_i=5, bias=310), 362.45, [0.5, 350])
```

With Kc=2 and tau_i=5 min the controller stabilizes the unstable steady state
(362.45 K) and removes the oscillation. The web UI has a PID checkbox with
setpoint/Kc/tau_i sliders and a preset that demonstrates this.

## Tests

    pip install numpy scipy pytest
    python -m pytest
