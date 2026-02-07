# PDF Traceability Matrix

This table maps implemented variables/equations in `cardio_onefile.py` to PDF locations.

| name | implementation reference | PDF location |
|---|---|---|
| STW | `build_activation_signals` | p.31, timing waveform section |
| SSW | `build_activation_signals` | p.31, activity waveform section |
| ACTV clamp `BOUND(0,1,SSW)` | `build_activation_signals` | p.31, ACTV equation |
| SLV | `build_activation_signals` | p.32, ventricular stiffness |
| PP1 | `compute_algebraic_signals` | p.31, pulmonary pressure equation |
| PP2 | `compute_algebraic_signals` | p.31 |
| PP3 | `compute_algebraic_signals` | p.31 |
| PL1 | `compute_algebraic_signals` | p.31 |
| PL2 | `compute_algebraic_signals` | p.32 |
| PLA | `compute_algebraic_signals` | p.32 |
| PLV | `compute_algebraic_signals` | p.32 |
| PA1 | `compute_algebraic_signals` | p.32 |
| PA2 | `compute_algebraic_signals` | p.32 |
| PA3 | `compute_algebraic_signals` | p.32 |
| PV1 | `compute_algebraic_signals` | p.32 |
| PV2 | `compute_algebraic_signals` | p.32 |
| PRA | `compute_algebraic_signals` | p.32 |
| PRV | `compute_algebraic_signals` | p.32 |
| FLA LIMINT | `compute_baseline_derivatives` | p.32, mitral LIMINT |
| FLV LIMINT | `compute_baseline_derivatives` | p.32, aortic LIMINT |
| FRA LIMINT | `compute_baseline_derivatives` | p.32, tricuspid LIMINT |
| FRV LIMINT | `compute_baseline_derivatives` | p.32, pulmonic LIMINT |
| QP1 INTEG | `compute_baseline_derivatives` | p.31, state equations |
| FP1 INTEG | `compute_baseline_derivatives` | p.31, state equations |
| QP2 INTEG | `compute_baseline_derivatives` | p.31, state equations |
| QP3 INTEG | `compute_baseline_derivatives` | p.31, state equations |
| QL1 INTEG | `compute_baseline_derivatives` | p.31, state equations |
| FL2 INTEG | `compute_baseline_derivatives` | p.32, state equations |
| QL2 INTEG | `compute_baseline_derivatives` | p.32, state equations |
| QLA INTEG | `compute_baseline_derivatives` | p.32, state equations |
| QLV INTEG | `compute_baseline_derivatives` | p.32, state equations |
| FA1 INTEG | `compute_baseline_derivatives` | p.32, state equations |
| QA1 INTEG | `compute_baseline_derivatives` | p.32, state equations |
| QA2 INTEG | `compute_baseline_derivatives` | p.32, state equations |
| QA3 INTEG | `compute_baseline_derivatives` | p.32, state equations |
| QV1 INTEG | `compute_baseline_derivatives` | p.32, state equations |
| FV2 INTEG | `compute_baseline_derivatives` | p.32, state equations |
| QV2 INTEG | `compute_baseline_derivatives` | p.32, state equations |
| QRA INTEG | `compute_baseline_derivatives` | p.32, state equations |
| QRV INTEG | `compute_baseline_derivatives` | p.32, state equations |
| PLVM output conversion | `run_simulation` | p.33, pressure conversion to mmHg |
| PCAM proxy (= PA3) | `run_simulation` | p.33, PASM pressure output |
| Mitral P-Q: PLA-PLV vs FLA | `extract_pq_mitral` | p.32 |
| Tricuspid P-Q: PRA-PRV vs FRA | `extract_pq_tricuspid` | p.32 |

Non-PDF assumptions are intentionally isolated in `docs/assumptions.md` only.
