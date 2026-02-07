# Assumptions Registry (Non-PDF Only)

## ASSUMPTION-AA-001
- ID: ASSUMPTION-AA-001
- Scope: Atrial arrhythmia overlay only
- Statement: Atrial arrhythmia is modeled using a physiology-derived pathway: autonomic-modulated irregular atrial cycle generation, AV-node-like refractory filtering, and normalized atrial mechanical twitch dynamics (bi-exponential rise/decay with frequency-dependent contractility attenuation).

## ASSUMPTION-NUM-001
- ID: ASSUMPTION-NUM-001
- Scope: Numerical integration only
- Statement: `solve_ivp` defaults use `rtol=1e-6`, `atol=1e-9`, method `RK45`.
