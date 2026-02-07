# Model Overview

## Baseline Hemodynamic Model
The baseline model in `cardio_onefile.py` is a single-file PF-1 style compartment model with ODE states for volume and inertial flows across pulmonary, left-heart, systemic, and right-heart pathways.

Key components:
- Activation waveform generation (`sig_activation_sine`, `sig_activation_square`)
- Time-varying ventricular stiffness (`left_ventricle_stiffness_cgs`)
- Algebraic pressure reconstruction from compliance and unstressed volumes
- LIMINT-style bounded inertial valve flows for mitral/aortic/tricuspid/pulmonic flows

Traceability to the source PDF is documented in:
- `docs/pdf_traceability.md`

## Physiology-Derived Atrial Arrhythmia Overlay
The arrhythmia overlay keeps baseline equations unchanged and changes only atrial activation drive through a re-derived physiology-inspired process:

1. Irregular atrial impulse timing
- Stochastic variability with low/high-frequency modulation components.

2. AV-node-like refractory filtering
- RR intervals are lower-bounded by dynamic refractory behavior.

3. Atrial mechanical twitch response
- Bi-exponential rise/decay twitch kernel for contraction waveforms.

4. Frequency-dependent atrial attenuation
- Faster local RR reduces atrial effective contraction ratio.

5. Inter-atrial timing offset
- Right atrial signal includes configurable small delay.

This behavior is summarized in:
- `docs/assumptions.md` (`ASSUMPTION-AA-001`)

## Fixed Comparison Scope
- Healthy baseline
- Atrial arrhythmia overlay
- P-Q extraction (mitral + tricuspid)
- One-page 4-panel comparison figure

Out of scope:
- VSD
- Ventricular arrhythmias
- Additional model families/variants
