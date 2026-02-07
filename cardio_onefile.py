"""
Healthy baseline + atrial arrhythmia overlay + P-Q comparison (single-file implementation).

Execution-friendly CLI:
  python cardio_onefile.py compare --duration-s 3 --dt-output-s 0.001 --random-seed 1 --output-dir out --export-format both
"""

from __future__ import annotations

# 1. Imports
import argparse
import csv
import json
import math
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import solve_ivp

# 2. Units/constants
MMHG_TO_CGS = 1332.0
CGS_TO_MMHG = 1.0 / MMHG_TO_CGS
SOLVER_REL_TOL = 1e-6
SOLVER_ABS_TOL = 1e-9

ASSUMPTION_AA_001 = "ASSUMPTION-AA-001"
ASSUMPTION_NUM_001 = "ASSUMPTION-NUM-001"


def mmhg_to_cgs_pressure(pressure_mmhg: float) -> float:
    return pressure_mmhg * MMHG_TO_CGS


def cgs_pressure_to_mmhg(pressure_cgs: float) -> float:
    return pressure_cgs * CGS_TO_MMHG


def validate_unit_round_trip() -> None:
    probe_values_mmhg = [0.0, 1.0, 12.5, -3.0, 100.0]
    for value_mmhg in probe_values_mmhg:
        value_cgs = mmhg_to_cgs_pressure(value_mmhg)
        restored_mmhg = cgs_pressure_to_mmhg(value_cgs)
        if not math.isclose(value_mmhg, restored_mmhg, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("mmHg<->CGS round-trip check failed")


# 3. PDF traceability comments
# Traceability IDs in code comments:
# - TRACE-PDF-ACTV-BOUND: p.31 ACTV=KB*BOUND(0.0,1.0,SSW)
# - TRACE-PDF-LIMINT-FLA: p.32 FLA=LIMINT((PLA-PLV-RLA*FLA)/LLA, ...)
# - TRACE-PDF-LIMINT-FLV: p.32 FLV=LIMINT((PLV-PA1-RLV*FLV)/LLV, ...)
# - TRACE-PDF-LIMINT-FRA: p.32 FRA=LIMINT((PRA-PRV-RRA*FRA)/LRA, ...)
# - TRACE-PDF-LIMINT-FRV: p.32 FRV=LIMINT((PRV-PP1-RRV*FRV)/LRV, ...)


# 4. Parameters
@dataclass(frozen=True)
class PF1Params:
    """PF-1 parameters from PDF listing (p.30-33)."""

    THI: float = field(default=800.0)
    LSI: float = field(default=2500.0)
    RSI: float = field(default=350.0)
    DLS: float = field(default=0.0)
    DRS: float = field(default=0.0)
    TSA: float = field(default=0.1)
    TS: float = field(default=0.3)
    TH: float = field(default=0.8)
    KB: float = field(default=1.0)
    SV1: float = field(default=0.9)
    SV2: float = field(default=0.25)
    TF: float = field(default=8.0)

    QP1U: float = field(default=7.8)
    PPIEDM: float = field(default=7.2)
    CP1: float = field(default=0.0001)
    QP2U: float = field(default=23.4)
    PP2EDM: float = field(default=7.0)
    CP2: float = field(default=0.0003)
    QP3U: float = field(default=210.5)
    PP3EDM: float = field(default=6.6)
    CP3: float = field(default=0.0027)
    QL1U: float = field(default=69.0)
    PLIEDM: float = field(default=4.45)
    CL1: float = field(default=0.001)
    QL2U: float = field(default=69.0)
    PL2EDM: float = field(default=3.62)
    CL2: float = field(default=0.001)
    QLAU: float = field(default=814.5)
    PLAEDM: float = field(default=3.45)
    CLA: float = field(default=0.01175)
    QLVU: float = field(default=10.0)
    PLVEDM: float = field(default=4.0)
    LD: float = field(default=45.0)
    QAIU: float = field(default=35.1)
    PAIEDM: float = field(default=64.3)
    CA1: float = field(default=0.00018)
    QA2U: float = field(default=85.0)
    PA2EDM: float = field(default=64.0)
    CA2: float = field(default=0.00025)
    QA3U: float = field(default=210.0)
    PA3EDM: float = field(default=63.0)
    CA3: float = field(default=0.00182)
    QV1U: float = field(default=909.0)
    PV1EDM: float = field(default=13.5)
    CV1: float = field(default=0.021)
    QV2U: float = field(default=1948.0)
    PV2EDM: float = field(default=7.2)
    CV2: float = field(default=0.045)
    QRAU: float = field(default=1948.0)
    PRAEDM: float = field(default=6.64)
    CRA: float = field(default=0.045)
    QRVU: float = field(default=10.0)
    PRVEDM: float = field(default=1.4)
    RD: float = field(default=72.0)

    RPW1: float = field(default=10.0)
    LP1: float = field(default=1.0)
    FP1IC: float = field(default=0.0)
    KP1: float = field(default=1.0)
    RP1: float = field(default=10.0)
    RP2: float = field(default=40.0)
    RP3: float = field(default=80.0)
    RL1: float = field(default=30.0)
    RL2: float = field(default=10.0)
    LL2: float = field(default=1.0)
    FL2IC: float = field(default=33.0)
    RLA: float = field(default=5.0)
    LLA: float = field(default=1.0)
    FLAIC: float = field(default=0.0)
    RLV: float = field(default=5.0)
    LLV: float = field(default=1.0)
    FLVIC: float = field(default=0.0)
    FA1IC: float = field(default=4.6)
    RA1: float = field(default=10.0)
    LA1: float = field(default=1.0)
    RPW2: float = field(default=10.0)
    KP2: float = field(default=1.0)
    FIS: float = field(default=0.0)
    TIS: float = field(default=3.0)
    RA2: float = field(default=160.0)
    RA3: float = field(default=1000.0)
    RV1: float = field(default=90.0)
    RV2: float = field(default=10.0)
    LV2: float = field(default=1.0)
    FV2IC: float = field(default=95.0)
    RRA: float = field(default=5.0)
    LRA: float = field(default=1.0)
    FRAIC: float = field(default=0.0)
    RRV: float = field(default=5.0)
    LRV: float = field(default=1.0)
    FRVIC: float = field(default=6.0)


@dataclass(frozen=True)
class ODESettings:
    solver_rel_tol: float = SOLVER_REL_TOL
    solver_abs_tol: float = SOLVER_ABS_TOL
    solver_max_step_s: float = float("inf")
    solver_method: str = "RK45"


# 5. Switching primitives
def zoh(time_s: float, time0_s: float, value0: float, period_s: float) -> float:
    if period_s <= 0.0:
        raise ValueError("period_s must be positive")
    if time_s < time0_s:
        return value0
    cycle_index = math.floor((time_s - time0_s) / period_s)
    return value0 + cycle_index * period_s


def rsw(condition_flag: bool, high_value: float, low_value: float) -> float:
    return high_value if condition_flag else low_value


def bound(lower_bound: float, upper_bound: float, value: float) -> float:
    if lower_bound > upper_bound:
        raise ValueError("lower_bound must be <= upper_bound")
    return max(lower_bound, min(upper_bound, value))


def realpl_derivative(state_value: float, input_value: float, tau_s: float) -> float:
    if tau_s <= 0.0:
        raise ValueError("tau_s must be positive")
    return (input_value - state_value) / tau_s


def limint_derivative(state_value: float, derivative_value: float, lower_bound: float, upper_bound: float) -> float:
    if lower_bound > upper_bound:
        raise ValueError("lower_bound must be <= upper_bound")
    if state_value <= lower_bound and derivative_value < 0.0:
        return 0.0
    if state_value >= upper_bound and derivative_value > 0.0:
        return 0.0
    return derivative_value


def run_primitive_sanity_checks() -> None:
    if bound(0.0, 1.0, -0.5) != 0.0:
        raise ValueError("BOUND sanity check failed")
    if limint_derivative(0.0, -1.0, 0.0, 1.0) != 0.0:
        raise ValueError("LIMINT sanity check failed")


# 6. Baseline equations
STATE_INDEX_MAP = {
    "state_pulmonary_artery_1_volume_ml": 0,
    "state_pulmonary_artery_1_flow_ml_per_s": 1,
    "state_pulmonary_artery_2_volume_ml": 2,
    "state_pulmonary_artery_3_volume_ml": 3,
    "state_pulmonary_vein_1_volume_ml": 4,
    "state_pulmonary_vein_2_volume_ml": 5,
    "state_pulmonary_vein_2_flow_ml_per_s": 6,
    "state_left_atrium_volume_ml": 7,
    "state_mitral_flow_ml_per_s": 8,
    "state_left_ventricle_volume_ml": 9,
    "state_aortic_flow_ml_per_s": 10,
    "state_aorta_1_volume_ml": 11,
    "state_aorta_1_flow_ml_per_s": 12,
    "state_aorta_2_volume_ml": 13,
    "state_aorta_3_volume_ml": 14,
    "state_systemic_vein_1_volume_ml": 15,
    "state_systemic_vein_2_volume_ml": 16,
    "state_systemic_vein_2_flow_ml_per_s": 17,
    "state_right_atrium_volume_ml": 18,
    "state_tricuspid_flow_ml_per_s": 19,
    "state_right_ventricle_volume_ml": 20,
    "state_pulmonic_flow_ml_per_s": 21,
}


def validate_pf1_params(params: PF1Params) -> None:
    positive_fields = [
        "THI", "LSI", "RSI", "TSA", "TS", "TH", "KB",
        "CP1", "CP2", "CP3", "CL1", "CL2", "CLA", "LD",
        "CA1", "CA2", "CA3", "CV1", "CV2", "CRA", "RD",
        "RPW1", "LP1", "KP1", "RP1", "RP2", "RP3",
        "RL1", "RL2", "LL2", "RLA", "LLA", "RLV", "LLV",
        "RA1", "LA1", "RPW2", "KP2", "RA2", "RA3",
        "RV1", "RV2", "LV2", "RRA", "LRA", "RRV", "LRV",
    ]
    for name in positive_fields:
        value = getattr(params, name)
        if value <= 0.0:
            raise ValueError(f"PF1Params.{name} must be > 0, got {value}")

    if params.TS > params.TH:
        raise ValueError(f"PF1Params.TS ({params.TS}) must be <= PF1Params.TH ({params.TH})")


def validate_ode_settings(ode_settings: ODESettings) -> None:
    if ode_settings.solver_rel_tol <= 0.0 or ode_settings.solver_abs_tol <= 0.0:
        raise ValueError("ODE tolerances must be positive")
    if ode_settings.solver_max_step_s != float("inf") and ode_settings.solver_max_step_s <= 0.0:
        raise ValueError("solver_max_step_s must be positive or inf")
    supported_methods = {"RK23", "RK45", "DOP853", "Radau", "BDF", "LSODA"}
    if ode_settings.solver_method not in supported_methods:
        raise ValueError(f"Unsupported solver method '{ode_settings.solver_method}'. "
                         f"Use one of {sorted(supported_methods)}")


def validate_simulation_outputs(time_s: np.ndarray, state_matrix: np.ndarray) -> None:
    if time_s.ndim != 1:
        raise ValueError("time_s must be 1D")
    if state_matrix.ndim != 2:
        raise ValueError("state_matrix must be 2D")
    if state_matrix.shape[1] != time_s.shape[0]:
        raise ValueError("state_matrix columns must match time samples")
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("time_s must be strictly increasing")
    if not np.isfinite(state_matrix).all():
        raise ValueError("state_matrix contains NaN or Inf")


def build_activation_signals(time_s: float, params: PF1Params) -> dict[str, float]:
    stiffness_left_systolic_cgs = params.LSI + (0.0 if time_s < params.THI else params.DLS)
    stiffness_right_systolic_cgs = params.RSI + (0.0 if time_s < params.THI else params.DRS)

    sawtooth_phase_s = time_s - zoh(time_s, 0.0, 0.0, params.TH)
    sig_activation_square = rsw(sawtooth_phase_s <= params.TS, sawtooth_phase_s, 0.0)
    sig_activation_sine = (
            params.SV1 * np.sin(np.pi * sig_activation_square / params.TS)
            - params.SV2 * np.sin(2.0 * np.pi * sig_activation_square / params.TS)
    )
    # TRACE-PDF-ACTV-BOUND
    activation_clamped_ratio = params.KB * bound(0.0, 1.0, float(sig_activation_sine))
    left_ventricle_stiffness_cgs = params.LD * (
                1.0 - activation_clamped_ratio) + stiffness_left_systolic_cgs * activation_clamped_ratio
    right_ventricle_stiffness_cgs = params.RD * (
                1.0 - activation_clamped_ratio) + stiffness_right_systolic_cgs * activation_clamped_ratio

    return {
        "sig_activation_square": float(sig_activation_square),
        "sig_activation_sine": float(sig_activation_sine),
        "left_ventricle_stiffness_cgs": float(left_ventricle_stiffness_cgs),
        "right_ventricle_stiffness_cgs": float(right_ventricle_stiffness_cgs),
    }


def compute_algebraic_signals(
        time_s: float,
        state_vector: np.ndarray,
        params: PF1Params,
) -> dict[str, float]:
    state_pulmonary_artery_1_volume_ml = state_vector[0]
    state_pulmonary_artery_1_flow_ml_per_s = state_vector[1]
    state_pulmonary_artery_2_volume_ml = state_vector[2]
    state_pulmonary_artery_3_volume_ml = state_vector[3]
    state_pulmonary_vein_1_volume_ml = state_vector[4]
    state_pulmonary_vein_2_volume_ml = state_vector[5]
    state_left_atrium_volume_ml = state_vector[7]
    state_left_ventricle_volume_ml = state_vector[9]
    state_aortic_flow_ml_per_s = state_vector[10]
    state_aorta_1_volume_ml = state_vector[11]
    state_aorta_1_flow_ml_per_s = state_vector[12]
    state_aorta_2_volume_ml = state_vector[13]
    state_aorta_3_volume_ml = state_vector[14]
    state_systemic_vein_1_volume_ml = state_vector[15]
    state_systemic_vein_2_volume_ml = state_vector[16]
    state_right_atrium_volume_ml = state_vector[18]
    state_right_ventricle_volume_ml = state_vector[20]
    state_pulmonic_flow_ml_per_s = state_vector[21]

    activation_signals = build_activation_signals(time_s, params)

    pressure_pulmonary_artery_1_cgs = (
            (state_pulmonary_artery_1_volume_ml - params.QP1U) / params.CP1
            + params.KP1 * params.RPW1 * (state_pulmonic_flow_ml_per_s - state_pulmonary_artery_1_flow_ml_per_s)
    )
    pressure_pulmonary_artery_2_cgs = (state_pulmonary_artery_2_volume_ml - params.QP2U) / params.CP2
    pressure_pulmonary_artery_3_cgs = (state_pulmonary_artery_3_volume_ml - params.QP3U) / params.CP3
    pressure_pulmonary_vein_1_cgs = (state_pulmonary_vein_1_volume_ml - params.QL1U) / params.CL1
    pressure_pulmonary_vein_2_cgs = (state_pulmonary_vein_2_volume_ml - params.QL2U) / params.CL2
    left_atrium_pressure_cgs = (state_left_atrium_volume_ml - params.QLAU) / params.CLA
    left_ventricle_pressure_cgs = (state_left_ventricle_volume_ml - params.QLVU) * activation_signals[
        "left_ventricle_stiffness_cgs"]

    pressure_aorta_1_cgs = (
            (state_aorta_1_volume_ml - params.QAIU) / params.CA1
            + params.KP2 * params.RPW2 * (state_aortic_flow_ml_per_s - state_aorta_1_flow_ml_per_s)
    )
    pressure_aorta_2_cgs = (state_aorta_2_volume_ml - params.QA2U) / params.CA2
    pressure_aorta_3_cgs = (state_aorta_3_volume_ml - params.QA3U) / params.CA3
    pressure_systemic_vein_1_cgs = (state_systemic_vein_1_volume_ml - params.QV1U) / params.CV1
    pressure_systemic_vein_2_cgs = (state_systemic_vein_2_volume_ml - params.QV2U) / params.CV2
    right_atrium_pressure_cgs = (state_right_atrium_volume_ml - params.QRAU) / params.CRA
    right_ventricle_pressure_cgs = (state_right_ventricle_volume_ml - params.QRVU) * activation_signals[
        "right_ventricle_stiffness_cgs"]

    flow_pulmonary_artery_2_ml_per_s = (pressure_pulmonary_artery_2_cgs - pressure_pulmonary_artery_3_cgs) / params.RP2
    flow_pulmonary_artery_3_ml_per_s = (pressure_pulmonary_artery_3_cgs - pressure_pulmonary_vein_1_cgs) / params.RP3
    flow_pulmonary_vein_1_ml_per_s = (pressure_pulmonary_vein_1_cgs - pressure_pulmonary_vein_2_cgs) / params.RL1
    flow_aorta_2_ml_per_s = (pressure_aorta_2_cgs - pressure_aorta_3_cgs) / params.RA2
    flow_aorta_3_ml_per_s = (pressure_aorta_3_cgs - pressure_systemic_vein_1_cgs) / params.RA3
    flow_systemic_vein_1_ml_per_s = (pressure_systemic_vein_1_cgs - pressure_systemic_vein_2_cgs) / params.RV1
    shunt_flow_ml_per_s = 0.0 if time_s < params.TIS else params.FIS

    return {
        "sig_activation_square": activation_signals["sig_activation_square"],
        "sig_activation_sine": activation_signals["sig_activation_sine"],
        "left_ventricle_stiffness_cgs": activation_signals["left_ventricle_stiffness_cgs"],
        "pressure_pulmonary_artery_1_cgs": pressure_pulmonary_artery_1_cgs,
        "pressure_pulmonary_artery_2_cgs": pressure_pulmonary_artery_2_cgs,
        "pressure_pulmonary_artery_3_cgs": pressure_pulmonary_artery_3_cgs,
        "pressure_pulmonary_vein_1_cgs": pressure_pulmonary_vein_1_cgs,
        "pressure_pulmonary_vein_2_cgs": pressure_pulmonary_vein_2_cgs,
        "left_atrium_pressure_cgs": left_atrium_pressure_cgs,
        "left_ventricle_pressure_cgs": left_ventricle_pressure_cgs,
        "pressure_aorta_1_cgs": pressure_aorta_1_cgs,
        "pressure_aorta_2_cgs": pressure_aorta_2_cgs,
        "pressure_aorta_3_cgs": pressure_aorta_3_cgs,
        "pressure_systemic_vein_1_cgs": pressure_systemic_vein_1_cgs,
        "pressure_systemic_vein_2_cgs": pressure_systemic_vein_2_cgs,
        "right_atrium_pressure_cgs": right_atrium_pressure_cgs,
        "right_ventricle_pressure_cgs": right_ventricle_pressure_cgs,
        "flow_pulmonary_artery_2_ml_per_s": flow_pulmonary_artery_2_ml_per_s,
        "flow_pulmonary_artery_3_ml_per_s": flow_pulmonary_artery_3_ml_per_s,
        "flow_pulmonary_vein_1_ml_per_s": flow_pulmonary_vein_1_ml_per_s,
        "flow_aorta_2_ml_per_s": flow_aorta_2_ml_per_s,
        "flow_aorta_3_ml_per_s": flow_aorta_3_ml_per_s,
        "flow_systemic_vein_1_ml_per_s": flow_systemic_vein_1_ml_per_s,
        "shunt_flow_ml_per_s": shunt_flow_ml_per_s,
    }


def _bounded_inertial_derivative(
        pressure_upstream_cgs: float,
        pressure_downstream_cgs: float,
        resistance_cgs: float,
        inertance_cgs: float,
        flow_state_ml_per_s: float,
        lower_limit_ml_per_s: float,
        upper_limit_ml_per_s: float,
) -> float:
    derivative_raw = (
                                 pressure_upstream_cgs - pressure_downstream_cgs - resistance_cgs * flow_state_ml_per_s) / inertance_cgs
    return limint_derivative(flow_state_ml_per_s, derivative_raw, lower_limit_ml_per_s, upper_limit_ml_per_s)


def compute_baseline_derivatives(
        time_s: float,
        state_vector: np.ndarray,
        params: PF1Params,
        atrial_activation_provider: Callable[[float], tuple[float, float]] | None = None,
) -> np.ndarray:
    """State order follows STATE_INDEX_MAP."""
    state_pulmonary_artery_1_flow_ml_per_s = state_vector[1]
    state_pulmonary_vein_2_flow_ml_per_s = state_vector[6]
    state_mitral_flow_ml_per_s = state_vector[8]
    state_aortic_flow_ml_per_s = state_vector[10]
    state_aorta_1_flow_ml_per_s = state_vector[12]
    state_systemic_vein_2_flow_ml_per_s = state_vector[17]
    state_tricuspid_flow_ml_per_s = state_vector[19]
    state_pulmonic_flow_ml_per_s = state_vector[21]

    signals = compute_algebraic_signals(time_s, state_vector, params)
    left_atrial_activation_ratio, right_atrial_activation_ratio = (
        (1.0, 1.0) if atrial_activation_provider is None else atrial_activation_provider(time_s)
    )

    der_pulmonary_artery_1_flow_ml_per_s2 = (
                                                    signals["pressure_pulmonary_artery_1_cgs"]
                                                    - signals["pressure_pulmonary_artery_2_cgs"]
                                                    - params.RPW1 * state_pulmonary_artery_1_flow_ml_per_s
                                            ) / params.LP1
    der_pulmonary_artery_1_volume_ml_per_s = state_pulmonic_flow_ml_per_s - state_pulmonary_artery_1_flow_ml_per_s
    der_pulmonary_artery_2_volume_ml_per_s = state_pulmonary_artery_1_flow_ml_per_s - signals[
        "flow_pulmonary_artery_2_ml_per_s"]
    der_pulmonary_artery_3_volume_ml_per_s = signals["flow_pulmonary_artery_2_ml_per_s"] - signals[
        "flow_pulmonary_artery_3_ml_per_s"]

    der_pulmonary_vein_1_volume_ml_per_s = signals["flow_pulmonary_artery_3_ml_per_s"] - signals[
        "flow_pulmonary_vein_1_ml_per_s"]
    der_pulmonary_vein_2_flow_ml_per_s2 = (
                                                  signals["pressure_pulmonary_vein_2_cgs"]
                                                  - signals["left_atrium_pressure_cgs"]
                                                  - params.RL2 * state_pulmonary_vein_2_flow_ml_per_s
                                          ) / params.LL2
    der_pulmonary_vein_2_volume_ml_per_s = signals[
                                               "flow_pulmonary_vein_1_ml_per_s"] - state_pulmonary_vein_2_flow_ml_per_s

    # TRACE-PDF-LIMINT-FLA
    der_mitral_flow_ml_per_s2 = _bounded_inertial_derivative(
        signals["left_atrium_pressure_cgs"] * left_atrial_activation_ratio,
        signals["left_ventricle_pressure_cgs"],
        params.RLA,
        params.LLA,
        state_mitral_flow_ml_per_s,
        0.0,
        1.0e4,
    )
    der_left_atrium_volume_ml_per_s = state_pulmonary_vein_2_flow_ml_per_s - state_mitral_flow_ml_per_s

    # TRACE-PDF-LIMINT-FLV
    der_aortic_flow_ml_per_s2 = _bounded_inertial_derivative(
        signals["left_ventricle_pressure_cgs"],
        signals["pressure_aorta_1_cgs"],
        params.RLV,
        params.LLV,
        state_aortic_flow_ml_per_s,
        0.0,
        1.0e5,
    )
    der_left_ventricle_volume_ml_per_s = state_mitral_flow_ml_per_s - state_aortic_flow_ml_per_s

    der_aorta_1_flow_ml_per_s2 = (
                                         signals["pressure_aorta_1_cgs"] - signals[
                                     "pressure_aorta_2_cgs"] - params.RA1 * state_aorta_1_flow_ml_per_s
                                 ) / params.LA1
    der_aorta_1_volume_ml_per_s = state_aortic_flow_ml_per_s - state_aorta_1_flow_ml_per_s + signals[
        "shunt_flow_ml_per_s"]
    der_aorta_2_volume_ml_per_s = state_aorta_1_flow_ml_per_s - signals["flow_aorta_2_ml_per_s"]
    der_aorta_3_volume_ml_per_s = signals["flow_aorta_2_ml_per_s"] - signals["flow_aorta_3_ml_per_s"]

    der_systemic_vein_1_volume_ml_per_s = signals["flow_aorta_3_ml_per_s"] - signals["flow_systemic_vein_1_ml_per_s"]
    der_systemic_vein_2_flow_ml_per_s2 = (
                                                 signals["pressure_systemic_vein_2_cgs"]
                                                 - signals["right_atrium_pressure_cgs"]
                                                 - params.RV2 * state_systemic_vein_2_flow_ml_per_s
                                         ) / params.LV2
    der_systemic_vein_2_volume_ml_per_s = signals["flow_systemic_vein_1_ml_per_s"] - state_systemic_vein_2_flow_ml_per_s

    # TRACE-PDF-LIMINT-FRA
    der_tricuspid_flow_ml_per_s2 = _bounded_inertial_derivative(
        signals["right_atrium_pressure_cgs"] * right_atrial_activation_ratio,
        signals["right_ventricle_pressure_cgs"],
        params.RRA,
        params.LRA,
        state_tricuspid_flow_ml_per_s,
        0.0,
        1.0e4,
    )
    der_right_atrium_volume_ml_per_s = state_systemic_vein_2_flow_ml_per_s - state_tricuspid_flow_ml_per_s

    # TRACE-PDF-LIMINT-FRV
    der_pulmonic_flow_ml_per_s2 = _bounded_inertial_derivative(
        signals["right_ventricle_pressure_cgs"],
        signals["pressure_pulmonary_artery_1_cgs"],
        params.RRV,
        params.LRV,
        state_pulmonic_flow_ml_per_s,
        0.0,
        1.0e5,
    )
    der_right_ventricle_volume_ml_per_s = state_tricuspid_flow_ml_per_s - state_pulmonic_flow_ml_per_s

    return np.array(
        [
            der_pulmonary_artery_1_volume_ml_per_s,
            der_pulmonary_artery_1_flow_ml_per_s2,
            der_pulmonary_artery_2_volume_ml_per_s,
            der_pulmonary_artery_3_volume_ml_per_s,
            der_pulmonary_vein_1_volume_ml_per_s,
            der_pulmonary_vein_2_volume_ml_per_s,
            der_pulmonary_vein_2_flow_ml_per_s2,
            der_left_atrium_volume_ml_per_s,
            der_mitral_flow_ml_per_s2,
            der_left_ventricle_volume_ml_per_s,
            der_aortic_flow_ml_per_s2,
            der_aorta_1_volume_ml_per_s,
            der_aorta_1_flow_ml_per_s2,
            der_aorta_2_volume_ml_per_s,
            der_aorta_3_volume_ml_per_s,
            der_systemic_vein_1_volume_ml_per_s,
            der_systemic_vein_2_volume_ml_per_s,
            der_systemic_vein_2_flow_ml_per_s2,
            der_right_atrium_volume_ml_per_s,
            der_tricuspid_flow_ml_per_s2,
            der_right_ventricle_volume_ml_per_s,
            der_pulmonic_flow_ml_per_s2,
        ],
        dtype=float,
    )


# 7. AF overlay
class AtrialArrhythmiaConfig:
    def __init__(
            self,
            af_atrial_amplitude_scale_left_ratio: float = 0.2,
            af_atrial_amplitude_scale_right_ratio: float = 0.2,
            af_rr_variability_ratio_std: float = 0.18,
            af_rr_lf_modulation_ratio: float = 0.08,
            af_rr_hf_modulation_ratio: float = 0.04,
            af_rr_lf_frequency_hz: float = 0.10,
            af_rr_hf_frequency_hz: float = 0.25,
            af_av_node_refractory_s: float = 0.22,
            af_av_node_refractory_variation_s: float = 0.03,
            af_atrial_twitch_rise_s: float = 0.035,
            af_atrial_twitch_decay_s: float = 0.12,
            af_right_atrial_delay_s: float = 0.015,
            af_rr_min_s: float = 0.25,
            af_rr_max_s: float = 1.50,
            af_random_seed: int = 1,
            *,
            A_LA: float | None = None,
            A_RA: float | None = None,
            jitter_sigma: float | None = None,
            jitter_clip: float | None = None,
            rr_min: float | None = None,
            rr_max: float | None = None,
            seed: int | None = None,
    ) -> None:
        self.af_atrial_amplitude_scale_left_ratio = (
            af_atrial_amplitude_scale_left_ratio if A_LA is None else A_LA
        )
        self.af_atrial_amplitude_scale_right_ratio = (
            af_atrial_amplitude_scale_right_ratio if A_RA is None else A_RA
        )
        self.af_rr_variability_ratio_std = af_rr_variability_ratio_std if jitter_sigma is None else jitter_sigma
        self.af_rr_lf_modulation_ratio = af_rr_lf_modulation_ratio
        self.af_rr_hf_modulation_ratio = af_rr_hf_modulation_ratio
        self.af_rr_lf_frequency_hz = af_rr_lf_frequency_hz
        self.af_rr_hf_frequency_hz = af_rr_hf_frequency_hz
        self.af_av_node_refractory_s = af_av_node_refractory_s
        self.af_av_node_refractory_variation_s = af_av_node_refractory_variation_s
        self.af_atrial_twitch_rise_s = af_atrial_twitch_rise_s
        self.af_atrial_twitch_decay_s = af_atrial_twitch_decay_s
        self.af_right_atrial_delay_s = af_right_atrial_delay_s
        self.af_rr_min_s = af_rr_min_s if rr_min is None else rr_min
        self.af_rr_max_s = af_rr_max_s if rr_max is None else rr_max
        self.af_random_seed = af_random_seed if seed is None else seed
        self.af_rr_jitter_ratio_clip = 1.0 if jitter_clip is None else jitter_clip

    # Backward-compatible aliases
    @property
    def A_LA(self) -> float:
        return self.af_atrial_amplitude_scale_left_ratio

    @property
    def A_RA(self) -> float:
        return self.af_atrial_amplitude_scale_right_ratio

    @property
    def jitter_sigma(self) -> float:
        return self.af_rr_variability_ratio_std

    @property
    def jitter_clip(self) -> float:
        return self.af_rr_jitter_ratio_clip

    @property
    def rr_min(self) -> float:
        return self.af_rr_min_s

    @property
    def rr_max(self) -> float:
        return self.af_rr_max_s

    @property
    def seed(self) -> int:
        return self.af_random_seed

    def to_dict(self) -> dict[str, float | int]:
        return {
            "af_atrial_amplitude_scale_left_ratio": self.af_atrial_amplitude_scale_left_ratio,
            "af_atrial_amplitude_scale_right_ratio": self.af_atrial_amplitude_scale_right_ratio,
            "af_rr_variability_ratio_std": self.af_rr_variability_ratio_std,
            "af_rr_lf_modulation_ratio": self.af_rr_lf_modulation_ratio,
            "af_rr_hf_modulation_ratio": self.af_rr_hf_modulation_ratio,
            "af_rr_lf_frequency_hz": self.af_rr_lf_frequency_hz,
            "af_rr_hf_frequency_hz": self.af_rr_hf_frequency_hz,
            "af_av_node_refractory_s": self.af_av_node_refractory_s,
            "af_av_node_refractory_variation_s": self.af_av_node_refractory_variation_s,
            "af_atrial_twitch_rise_s": self.af_atrial_twitch_rise_s,
            "af_atrial_twitch_decay_s": self.af_atrial_twitch_decay_s,
            "af_right_atrial_delay_s": self.af_right_atrial_delay_s,
            "af_rr_min_s": self.af_rr_min_s,
            "af_rr_max_s": self.af_rr_max_s,
            "af_random_seed": self.af_random_seed,
            "af_rr_jitter_ratio_clip": self.af_rr_jitter_ratio_clip,
        }


def validate_af_config(cfg: AtrialArrhythmiaConfig) -> None:
    if cfg.af_atrial_twitch_rise_s <= 0.0 or cfg.af_atrial_twitch_decay_s <= 0.0:
        raise ValueError("AF twitch time constants must be positive")
    if cfg.af_atrial_twitch_decay_s <= cfg.af_atrial_twitch_rise_s:
        raise ValueError("af_atrial_twitch_decay_s must be > af_atrial_twitch_rise_s")
    if cfg.af_rr_min_s <= 0.0 or cfg.af_rr_max_s <= 0.0:
        raise ValueError("AF rr bounds must be positive")
    if cfg.af_rr_min_s > cfg.af_rr_max_s:
        raise ValueError("af_rr_min_s must be <= af_rr_max_s")
    if cfg.af_rr_variability_ratio_std < 0.0:
        raise ValueError("af_rr_variability_ratio_std must be >= 0")


def _sample_rr_interval_s(base_rr_s: float, rng: np.random.Generator, cfg: AtrialArrhythmiaConfig) -> float:
    jitter_ratio = float(rng.normal(0.0, cfg.af_rr_variability_ratio_std))
    jitter_ratio = max(-cfg.af_rr_jitter_ratio_clip, min(cfg.af_rr_jitter_ratio_clip, jitter_ratio))
    rr_s = base_rr_s * (1.0 + jitter_ratio)
    dynamic_refractory_s = cfg.af_av_node_refractory_s + abs(
        float(rng.normal(0.0, cfg.af_av_node_refractory_variation_s)))
    rr_s = max(rr_s, dynamic_refractory_s)
    return max(cfg.af_rr_min_s, min(cfg.af_rr_max_s, rr_s))


def build_atrial_schedule(duration_s: float, baseline_rr_s: float, cfg: AtrialArrhythmiaConfig) -> tuple[
    np.ndarray, np.ndarray]:
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if baseline_rr_s <= 0.0:
        raise ValueError("baseline_rr_s must be positive")
    validate_af_config(cfg)

    rng = np.random.default_rng(cfg.af_random_seed)
    beat_start_times_s = [0.0]
    rr_intervals_s = []
    elapsed_s = 0.0
    phase_lf_rad = float(rng.uniform(0.0, 2.0 * np.pi))
    phase_hf_rad = float(rng.uniform(0.0, 2.0 * np.pi))

    while elapsed_s < duration_s:
        lf_component = cfg.af_rr_lf_modulation_ratio * np.sin(
            2.0 * np.pi * cfg.af_rr_lf_frequency_hz * elapsed_s + phase_lf_rad)
        hf_component = cfg.af_rr_hf_modulation_ratio * np.sin(
            2.0 * np.pi * cfg.af_rr_hf_frequency_hz * elapsed_s + phase_hf_rad)
        rr_drive_s = baseline_rr_s * (1.0 + float(lf_component) + float(hf_component))
        rr_s = _sample_rr_interval_s(rr_drive_s, rng, cfg)
        rr_intervals_s.append(rr_s)
        elapsed_s += rr_s
        beat_start_times_s.append(elapsed_s)

    return np.asarray(beat_start_times_s[:-1], dtype=float), np.asarray(rr_intervals_s, dtype=float)


def _normalized_atrial_twitch_kernel(delta_t_s: float, tau_rise_s: float, tau_decay_s: float) -> float:
    if delta_t_s < 0.0:
        return 0.0
    if tau_rise_s <= 0.0 or tau_decay_s <= 0.0:
        raise ValueError("atrial twitch time constants must be positive")
    if tau_decay_s <= tau_rise_s:
        raise ValueError("af_atrial_twitch_decay_s must be > af_atrial_twitch_rise_s")
    raw_value = np.exp(-delta_t_s / tau_decay_s) - np.exp(-delta_t_s / tau_rise_s)
    time_peak_s = (tau_rise_s * tau_decay_s * np.log(tau_decay_s / tau_rise_s)) / (tau_decay_s - tau_rise_s)
    peak_value = np.exp(-time_peak_s / tau_decay_s) - np.exp(-time_peak_s / tau_rise_s)
    if peak_value <= 0.0:
        return 0.0
    return max(0.0, float(raw_value / peak_value))


def _atrial_activation_from_schedule(
        time_s: float,
        beat_start_times_s: np.ndarray,
        rr_intervals_s: np.ndarray,
        atrial_amplitude_ratio: float,
        cfg: AtrialArrhythmiaConfig,
) -> float:
    if beat_start_times_s.size == 0 or time_s < beat_start_times_s[0]:
        return 0.0
    beat_index = int(np.searchsorted(beat_start_times_s, time_s, side="right") - 1)
    if beat_index < 0:
        return 0.0

    window_s = 5.0 * cfg.af_atrial_twitch_decay_s
    twitch_sum_ratio = 0.0
    idx = beat_index
    while idx >= 0:
        delta_t_s = time_s - beat_start_times_s[idx]
        if delta_t_s > window_s:
            break
        twitch_sum_ratio += _normalized_atrial_twitch_kernel(
            delta_t_s,
            cfg.af_atrial_twitch_rise_s,
            cfg.af_atrial_twitch_decay_s,
        )
        idx -= 1

    local_rr_s = rr_intervals_s[min(beat_index, rr_intervals_s.shape[0] - 1)]
    frequency_attenuation_ratio = 1.0 / (1.0 + 0.7 * max(0.0, (0.6 - local_rr_s) / 0.6))
    activation_ratio = atrial_amplitude_ratio * frequency_attenuation_ratio * twitch_sum_ratio
    return max(0.0, min(1.0, float(activation_ratio)))


def apply_atrial_arrhythmia_overlay(
        duration_s: float,
        baseline_rr_s: float,
        overlay_cfg: AtrialArrhythmiaConfig,
) -> Callable[[float], tuple[float, float]]:
    beat_start_times_s, rr_intervals_s = build_atrial_schedule(duration_s, baseline_rr_s, overlay_cfg)

    def _provider(time_s: float) -> tuple[float, float]:
        left_ratio = _atrial_activation_from_schedule(
            time_s,
            beat_start_times_s,
            rr_intervals_s,
            overlay_cfg.af_atrial_amplitude_scale_left_ratio,
            overlay_cfg,
        )
        right_ratio = _atrial_activation_from_schedule(
            time_s - overlay_cfg.af_right_atrial_delay_s,
            beat_start_times_s,
            rr_intervals_s,
            overlay_cfg.af_atrial_amplitude_scale_right_ratio,
            overlay_cfg,
        )
        return left_ratio, right_ratio

    return _provider


# 8. ODE integration
def _initial_state_vector(params: PF1Params) -> np.ndarray:
    def _volume_from_ed_pressure(volume_unstressed_ml: float, pressure_ed_mmhg: float,
                                 compliance_ml_per_cgs: float) -> float:
        return volume_unstressed_ml + pressure_ed_mmhg * MMHG_TO_CGS * compliance_ml_per_cgs

    return np.array(
        [
            _volume_from_ed_pressure(params.QP1U, params.PPIEDM, params.CP1),
            params.FP1IC,
            _volume_from_ed_pressure(params.QP2U, params.PP2EDM, params.CP2),
            _volume_from_ed_pressure(params.QP3U, params.PP3EDM, params.CP3),
            _volume_from_ed_pressure(params.QL1U, params.PLIEDM, params.CL1),
            _volume_from_ed_pressure(params.QL2U, params.PL2EDM, params.CL2),
            params.FL2IC,
            _volume_from_ed_pressure(params.QLAU, params.PLAEDM, params.CLA),
            params.FLAIC,
            params.QLVU + params.PLVEDM * MMHG_TO_CGS / params.LD,
            params.FLVIC,
            _volume_from_ed_pressure(params.QAIU, params.PAIEDM, params.CA1),
            params.FA1IC,
            _volume_from_ed_pressure(params.QA2U, params.PA2EDM, params.CA2),
            _volume_from_ed_pressure(params.QA3U, params.PA3EDM, params.CA3),
            _volume_from_ed_pressure(params.QV1U, params.PV1EDM, params.CV1),
            _volume_from_ed_pressure(params.QV2U, params.PV2EDM, params.CV2),
            params.FV2IC,
            _volume_from_ed_pressure(params.QRAU, params.PRAEDM, params.CRA),
            params.FRAIC,
            params.QRVU + params.PRVEDM * MMHG_TO_CGS / params.RD,
            params.FRVIC,
        ],
        dtype=float,
    )


def _build_time_grid_s(duration_s: float, dt_output_s: float) -> np.ndarray:
    if dt_output_s <= 0.0:
        raise ValueError("dt_output_s must be positive")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    time_s = np.arange(0.0, duration_s + 0.5 * dt_output_s, dt_output_s, dtype=float)
    if time_s[-1] > duration_s:
        time_s[-1] = duration_s
    return time_s


def run_simulation(
        params: PF1Params,
        duration_s: float,
        dt_output_s: float,
        atrial_activation_provider: Callable[[float], tuple[float, float]] | None = None,
        ode_settings: ODESettings | None = None,
) -> dict[str, np.ndarray]:
    ode_settings = ode_settings or ODESettings()
    validate_ode_settings(ode_settings)

    time_grid_s = _build_time_grid_s(duration_s, dt_output_s)

    solution = solve_ivp(
        fun=lambda t, y: compute_baseline_derivatives(t, y, params, atrial_activation_provider),
        t_span=(0.0, duration_s),
        y0=_initial_state_vector(params),
        t_eval=time_grid_s,
        method=ode_settings.solver_method,
        rtol=ode_settings.solver_rel_tol,
        atol=ode_settings.solver_abs_tol,
        max_step=ode_settings.solver_max_step_s,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    validate_simulation_outputs(solution.t, solution.y)

    n_samples = solution.t.shape[0]
    left_ventricle_pressure_mmhg = np.zeros(n_samples, dtype=float)
    arterial_compliance_pressure_mmhg = np.zeros(n_samples, dtype=float)
    left_atrium_pressure_cgs = np.zeros(n_samples, dtype=float)
    left_ventricle_pressure_cgs = np.zeros(n_samples, dtype=float)
    right_atrium_pressure_cgs = np.zeros(n_samples, dtype=float)
    right_ventricle_pressure_cgs = np.zeros(n_samples, dtype=float)
    sig_activation_square = np.zeros(n_samples, dtype=float)
    sig_activation_sine = np.zeros(n_samples, dtype=float)
    left_ventricle_stiffness_cgs = np.zeros(n_samples, dtype=float)

    for sample_index in range(n_samples):
        algebraic = compute_algebraic_signals(solution.t[sample_index], solution.y[:, sample_index], params)
        left_ventricle_pressure_mmhg[sample_index] = cgs_pressure_to_mmhg(algebraic["left_ventricle_pressure_cgs"])
        arterial_compliance_pressure_mmhg[sample_index] = cgs_pressure_to_mmhg(algebraic["pressure_aorta_3_cgs"])
        left_atrium_pressure_cgs[sample_index] = algebraic["left_atrium_pressure_cgs"]
        left_ventricle_pressure_cgs[sample_index] = algebraic["left_ventricle_pressure_cgs"]
        right_atrium_pressure_cgs[sample_index] = algebraic["right_atrium_pressure_cgs"]
        right_ventricle_pressure_cgs[sample_index] = algebraic["right_ventricle_pressure_cgs"]
        sig_activation_square[sample_index] = algebraic["sig_activation_square"]
        sig_activation_sine[sample_index] = algebraic["sig_activation_sine"]
        left_ventricle_stiffness_cgs[sample_index] = algebraic["left_ventricle_stiffness_cgs"]

    left_ventricle_volume_ml = solution.y[STATE_INDEX_MAP["state_left_ventricle_volume_ml"]]
    mitral_flow_ml_per_s = solution.y[STATE_INDEX_MAP["state_mitral_flow_ml_per_s"]]
    tricuspid_flow_ml_per_s = solution.y[STATE_INDEX_MAP["state_tricuspid_flow_ml_per_s"]]

    if not np.isfinite(left_ventricle_pressure_mmhg).all():
        raise ValueError("Computed pressure contains NaN/Inf")

    return {
        "time_s": solution.t,
        "state_matrix": solution.y,
        "sig_activation_square": sig_activation_square,
        "sig_activation_sine": sig_activation_sine,
        "left_ventricle_stiffness_cgs": left_ventricle_stiffness_cgs,
        "left_ventricle_pressure_mmhg": left_ventricle_pressure_mmhg,
        "arterial_compliance_pressure_mmhg": arterial_compliance_pressure_mmhg,
        "left_ventricle_volume_ml": left_ventricle_volume_ml,
        "left_atrium_pressure_cgs": left_atrium_pressure_cgs,
        "left_ventricle_pressure_cgs": left_ventricle_pressure_cgs,
        "right_atrium_pressure_cgs": right_atrium_pressure_cgs,
        "right_ventricle_pressure_cgs": right_ventricle_pressure_cgs,
        "mitral_flow_ml_per_s": mitral_flow_ml_per_s,
        "tricuspid_flow_ml_per_s": tricuspid_flow_ml_per_s,
        # Backward-compatible keys
        "t": solution.t,
        "y": solution.y,
        "STW": sig_activation_square,
        "SSW": sig_activation_sine,
        "SLV": left_ventricle_stiffness_cgs,
        "PLVM": left_ventricle_pressure_mmhg,
        "PCAM": arterial_compliance_pressure_mmhg,
        "QLV": left_ventricle_volume_ml,
        "PLA": left_atrium_pressure_cgs,
        "PLV": left_ventricle_pressure_cgs,
        "PRA": right_atrium_pressure_cgs,
        "PRV": right_ventricle_pressure_cgs,
        "FLA": mitral_flow_ml_per_s,
        "FRA": tricuspid_flow_ml_per_s,
    }


def simulate_healthy(
        params: PF1Params,
        tf: float,
        dt_out: float,
        solver_cfg: ODESettings | None = None,
) -> dict[str, np.ndarray]:
    return run_simulation(params, tf, dt_out, atrial_activation_provider=None, ode_settings=solver_cfg)


def simulate_af(
        params: PF1Params,
        tf: float,
        dt_out: float,
        seed: int = 1,
        overlay_cfg: AtrialArrhythmiaConfig | None = None,
        solver_cfg: ODESettings | None = None,
) -> dict[str, np.ndarray]:
    cfg = overlay_cfg or AtrialArrhythmiaConfig(af_random_seed=seed)
    provider = apply_atrial_arrhythmia_overlay(tf, params.TH, cfg)
    return run_simulation(params, tf, dt_out, atrial_activation_provider=provider, ode_settings=solver_cfg)


def simulate_compare(
        params: PF1Params,
        duration_s: float,
        dt_output_s: float,
        random_seed: int,
        solver_cfg: ODESettings | None = None,
        overlay_cfg: AtrialArrhythmiaConfig | None = None,
) -> dict[str, object]:
    healthy_outputs = simulate_healthy(params, tf=duration_s, dt_out=dt_output_s, solver_cfg=solver_cfg)
    af_cfg = overlay_cfg or AtrialArrhythmiaConfig(af_random_seed=random_seed)
    af_outputs = simulate_af(params, tf=duration_s, dt_out=dt_output_s, seed=random_seed, overlay_cfg=af_cfg,
                             solver_cfg=solver_cfg)
    if not np.array_equal(healthy_outputs["time_s"], af_outputs["time_s"]):
        raise ValueError("healthy and AF time vectors are not aligned")
    metrics = compute_summary_metrics(healthy_outputs, af_outputs)
    return {
        "time_s": healthy_outputs["time_s"],
        "healthy": healthy_outputs,
        "af": af_outputs,
        "metrics": metrics,
        "af_config": af_cfg,
    }


# 9. Post-processing (P-Q + metrics)
def extract_pq_mitral(outputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    pressure_delta_mmhg = cgs_pressure_to_mmhg(
        outputs["left_atrium_pressure_cgs"] - outputs["left_ventricle_pressure_cgs"])
    flow_ml_per_s = outputs["mitral_flow_ml_per_s"]
    return pressure_delta_mmhg, flow_ml_per_s


def extract_pq_tricuspid(outputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    pressure_delta_mmhg = cgs_pressure_to_mmhg(
        outputs["right_atrium_pressure_cgs"] - outputs["right_ventricle_pressure_cgs"])
    flow_ml_per_s = outputs["tricuspid_flow_ml_per_s"]
    return pressure_delta_mmhg, flow_ml_per_s


def extract_pq_signals(outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mitral_pressure_delta_mmhg, mitral_flow_ml_per_s = extract_pq_mitral(outputs)
    tricuspid_pressure_delta_mmhg, tricuspid_flow_ml_per_s = extract_pq_tricuspid(outputs)
    return {
        "mitral_pressure_delta_mmhg": mitral_pressure_delta_mmhg,
        "mitral_flow_ml_per_s": mitral_flow_ml_per_s,
        "tricuspid_pressure_delta_mmhg": tricuspid_pressure_delta_mmhg,
        "tricuspid_flow_ml_per_s": tricuspid_flow_ml_per_s,
    }


def _beat_to_beat_variability_index(signal_time_s: np.ndarray, volume_signal_ml: np.ndarray) -> float:
    peaks = \
    np.where((volume_signal_ml[1:-1] > volume_signal_ml[:-2]) & (volume_signal_ml[1:-1] > volume_signal_ml[2:]))[0] + 1
    if peaks.shape[0] < 3:
        return 0.0
    rr_intervals_s = np.diff(signal_time_s[peaks])
    mean_rr = float(np.mean(rr_intervals_s))
    if mean_rr <= 0.0:
        return 0.0
    return float(np.std(rr_intervals_s) / mean_rr)


def compute_summary_metrics(
        healthy_outputs: dict[str, np.ndarray],
        af_outputs: dict[str, np.ndarray],
) -> dict[str, float]:
    b2b_healthy = _beat_to_beat_variability_index(
        healthy_outputs["time_s"], healthy_outputs["left_ventricle_volume_ml"]
    )
    b2b_af = _beat_to_beat_variability_index(
        af_outputs["time_s"], af_outputs["left_ventricle_volume_ml"]
    )

    return {
        "met_mean_left_ventricle_pressure_mmhg_healthy": float(
            np.mean(healthy_outputs["left_ventricle_pressure_mmhg"])),
        "met_mean_left_ventricle_pressure_mmhg_af": float(np.mean(af_outputs["left_ventricle_pressure_mmhg"])),
        "met_mean_arterial_compliance_pressure_mmhg_healthy": float(
            np.mean(healthy_outputs["arterial_compliance_pressure_mmhg"])),
        "met_mean_arterial_compliance_pressure_mmhg_af": float(
            np.mean(af_outputs["arterial_compliance_pressure_mmhg"])),
        "met_stroke_volume_proxy_ml_healthy": float(
            np.max(healthy_outputs["left_ventricle_volume_ml"]) - np.min(healthy_outputs["left_ventricle_volume_ml"])),
        "met_stroke_volume_proxy_ml_af": float(
            np.max(af_outputs["left_ventricle_volume_ml"]) - np.min(af_outputs["left_ventricle_volume_ml"])),
        # new explicit proxy name
        "met_cycle_interval_variability_proxy_from_lv_volume_healthy": b2b_healthy,
        "met_cycle_interval_variability_proxy_from_lv_volume_af": b2b_af,
        # backward-compatible names
        "met_beat_to_beat_variability_index_healthy": b2b_healthy,
        "met_beat_to_beat_variability_index_af": b2b_af,
    }


# 10. One-page plotting
def plot_onepage_comparison(
        healthy_outputs: dict[str, np.ndarray],
        af_outputs: dict[str, np.ndarray],
        t_start_s: float = 0.0,
        t_end_s: float = 3.0,
) -> plt.Figure:
    if t_end_s <= t_start_s:
        raise ValueError("t_end_s must be > t_start_s")

    healthy_mask = (healthy_outputs["time_s"] >= t_start_s) & (healthy_outputs["time_s"] <= t_end_s)
    af_mask = (af_outputs["time_s"] >= t_start_s) & (af_outputs["time_s"] <= t_end_s)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # (a) activation
    ax = axes[0, 0]
    ax.plot(healthy_outputs["time_s"][healthy_mask], healthy_outputs["sig_activation_sine"][healthy_mask],
            label="Healthy SSW")
    ax.plot(healthy_outputs["time_s"][healthy_mask], healthy_outputs["sig_activation_square"][healthy_mask],
            label="Healthy STW")
    ax.plot(af_outputs["time_s"][af_mask], af_outputs["sig_activation_sine"][af_mask], "--", label="AF SSW")
    ax.plot(af_outputs["time_s"][af_mask], af_outputs["sig_activation_square"][af_mask], "--", label="AF STW")
    ax.set_title("(a) Activity Generator Waveforms")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Activation (ratio)")
    ax.legend(fontsize=8)

    # (b) stiffness
    ax = axes[0, 1]
    ax.plot(healthy_outputs["time_s"][healthy_mask], healthy_outputs["left_ventricle_stiffness_cgs"][healthy_mask],
            label="Healthy SLV")
    ax.plot(af_outputs["time_s"][af_mask], af_outputs["left_ventricle_stiffness_cgs"][af_mask], "--", label="AF SLV")
    ax.set_title("(b) Ventricular Stiffness")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Stiffness (cgs)")
    ax.legend(fontsize=8)

    # (c) pressures + volume
    ax = axes[1, 0]
    line1 = ax.plot(healthy_outputs["time_s"][healthy_mask],
                    healthy_outputs["left_ventricle_pressure_mmhg"][healthy_mask], label="Healthy PLVM")
    line2 = ax.plot(healthy_outputs["time_s"][healthy_mask],
                    healthy_outputs["arterial_compliance_pressure_mmhg"][healthy_mask], label="Healthy PCAM")
    line3 = ax.plot(af_outputs["time_s"][af_mask], af_outputs["left_ventricle_pressure_mmhg"][af_mask], "--",
                    label="AF PLVM")
    line4 = ax.plot(af_outputs["time_s"][af_mask], af_outputs["arterial_compliance_pressure_mmhg"][af_mask], "--",
                    label="AF PCAM")
    ax.set_title("(c) Pressures and LV Volume")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (mmHg)")

    ax2 = ax.twinx()
    line5 = ax2.plot(healthy_outputs["time_s"][healthy_mask], healthy_outputs["left_ventricle_volume_ml"][healthy_mask],
                     label="Healthy QLV")
    line6 = ax2.plot(af_outputs["time_s"][af_mask], af_outputs["left_ventricle_volume_ml"][af_mask], "--",
                     label="AF QLV")
    ax2.set_ylabel("Volume (ml)")

    lines = line1 + line2 + line3 + line4 + line5 + line6
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="best")

    # (d) PV loop
    ax = axes[1, 1]
    ax.plot(healthy_outputs["left_ventricle_volume_ml"][healthy_mask],
            healthy_outputs["left_ventricle_pressure_mmhg"][healthy_mask], label="Healthy")
    ax.plot(af_outputs["left_ventricle_volume_ml"][af_mask], af_outputs["left_ventricle_pressure_mmhg"][af_mask], "--",
            label="AF")
    ax.set_title("(d) Ventricular PV Loop")
    ax.set_xlabel("QLV (ml)")
    ax.set_ylabel("PLVM (mmHg)")
    ax.legend(fontsize=8)

    figure.tight_layout()
    return figure


# 11. Export helpers
def _write_timeseries_csv(path_csv: Path, outputs: dict[str, np.ndarray], prefix: str) -> None:
    export_columns = {
        "time_s": outputs["time_s"],
        f"sig_activation_sine_ratio_{prefix}": outputs["sig_activation_sine"],
        f"sig_activation_square_ratio_{prefix}": outputs["sig_activation_square"],
        f"left_ventricle_stiffness_cgs_{prefix}": outputs["left_ventricle_stiffness_cgs"],
        f"left_ventricle_pressure_mmhg_{prefix}": outputs["left_ventricle_pressure_mmhg"],
        f"arterial_compliance_pressure_mmhg_{prefix}": outputs["arterial_compliance_pressure_mmhg"],
        f"left_ventricle_volume_ml_{prefix}": outputs["left_ventricle_volume_ml"],
        f"mitral_flow_ml_per_s_{prefix}": outputs["mitral_flow_ml_per_s"],
        f"tricuspid_flow_ml_per_s_{prefix}": outputs["tricuspid_flow_ml_per_s"],
    }
    with path_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        headers = list(export_columns.keys())
        writer.writerow(headers)
        sample_count = len(outputs["time_s"])
        for i in range(sample_count):
            writer.writerow([float(export_columns[h][i]) for h in headers])


def save_onepage(fig: plt.Figure, outpath: str, *, format: str = "pdf", dpi: int = 300) -> None:
    path = Path(outpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format=format, dpi=dpi, bbox_inches="tight")


def load_af_config_json(path: str | None, default_seed: int) -> AtrialArrhythmiaConfig:
    if path is None:
        return AtrialArrhythmiaConfig(af_random_seed=default_seed)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = dict(payload)
    payload.setdefault("af_random_seed", default_seed)
    cfg = AtrialArrhythmiaConfig(**payload)
    validate_af_config(cfg)
    return cfg


def export_outputs(
        output_dir: Path,
        healthy_outputs: dict[str, np.ndarray] | None,
        af_outputs: dict[str, np.ndarray] | None,
        metrics: dict[str, float] | None,
        figure: plt.Figure | None,
        export_format: str,
        duration_s: float,
        dt_output_s: float,
        random_seed: int,
        solver_settings: ODESettings,
        af_config: AtrialArrhythmiaConfig | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if healthy_outputs is not None:
        _write_timeseries_csv(output_dir / "timeseries_healthy.csv", healthy_outputs, "healthy")
    if af_outputs is not None:
        _write_timeseries_csv(output_dir / "timeseries_af.csv", af_outputs, "af")

    if metrics is not None:
        with (output_dir / "summary_metrics.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric_name", "metric_value"])
            for metric_name, metric_value in metrics.items():
                writer.writerow([metric_name, metric_value])

    if figure is not None:
        if export_format in {"pdf", "both"}:
            save_onepage(figure, str(output_dir / "comparison_onepage.pdf"), format="pdf")
        if export_format in {"png", "both"}:
            save_onepage(figure, str(output_dir / "comparison_onepage.png"), format="png")

    metadata_lines = [
        f"duration_s={duration_s}",
        f"dt_output_s={dt_output_s}",
        f"random_seed={random_seed}",
        f"solver_method={solver_settings.solver_method}",
        f"solver_rel_tol={solver_settings.solver_rel_tol}",
        f"solver_abs_tol={solver_settings.solver_abs_tol}",
        f"solver_max_step_s={solver_settings.solver_max_step_s}",
        f"assumption_ids={ASSUMPTION_AA_001},{ASSUMPTION_NUM_001}",
        f"python_version={platform.python_version()}",
        f"numpy_version={np.__version__}",
        f"scipy_version={scipy.__version__}",
        f"matplotlib_version={matplotlib.__version__}",
    ]

    if af_config is not None:
        for key, value in af_config.to_dict().items():
            metadata_lines.append(f"af.{key}={value}")

    (output_dir / "run_metadata.txt").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")


# 12. CLI/main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Healthy baseline + atrial arrhythmia + P-Q comparison")
    parser.add_argument(
        "mode",
        nargs="?",
        default="compare",
        choices=["healthy", "atrial-arrhythmia", "af", "compare"],
        help="Run mode (default: compare)",
    )
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--dt-output-s", type=float, default=0.001)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="out")
    parser.add_argument("--export-format", choices=["pdf", "png", "both"], default="both")

    # New controls
    parser.add_argument("--t-start-s", type=float, default=0.0)
    parser.add_argument("--t-end-s", type=float, default=None)
    parser.add_argument("--solver-method", type=str, default="RK45")
    parser.add_argument("--rtol", type=float, default=SOLVER_REL_TOL)
    parser.add_argument("--atol", type=float, default=SOLVER_ABS_TOL)
    parser.add_argument("--max-step-s", type=float, default=float("inf"))
    parser.add_argument("--af-config-json", type=str, default=None)

    # backward-compatible aliases
    parser.add_argument("--tf", type=float, default=None)
    parser.add_argument("--dt-out", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--format", choices=["pdf", "png", "both"], default=None)

    args = parser.parse_args(argv)

    duration_s = args.tf if args.tf is not None else args.duration_s
    dt_output_s = args.dt_out if args.dt_out is not None else args.dt_output_s
    random_seed = args.seed if args.seed is not None else args.random_seed
    output_dir = Path(args.outdir if args.outdir is not None else args.output_dir)
    export_format = args.format if args.format is not None else args.export_format

    validate_unit_round_trip()
    run_primitive_sanity_checks()

    params = PF1Params()
    validate_pf1_params(params)

    ode_settings = ODESettings(
        solver_method=args.solver_method,
        solver_rel_tol=args.rtol,
        solver_abs_tol=args.atol,
        solver_max_step_s=args.max_step_s,
    )
    validate_ode_settings(ode_settings)

    try:
        af_cfg = load_af_config_json(args.af_config_json, random_seed)
    except Exception as exc:
        raise ValueError(f"Invalid AF config JSON: {exc}") from exc

    if args.mode == "healthy":
        healthy_outputs = simulate_healthy(params, tf=duration_s, dt_out=dt_output_s, solver_cfg=ode_settings)
        export_outputs(
            output_dir=output_dir,
            healthy_outputs=healthy_outputs,
            af_outputs=None,
            metrics=None,
            figure=None,
            export_format=export_format,
            duration_s=duration_s,
            dt_output_s=dt_output_s,
            random_seed=random_seed,
            solver_settings=ode_settings,
            af_config=None,
        )
        return 0

    if args.mode in {"atrial-arrhythmia", "af"}:
        af_outputs = simulate_af(
            params,
            tf=duration_s,
            dt_out=dt_output_s,
            seed=random_seed,
            overlay_cfg=af_cfg,
            solver_cfg=ode_settings
        )
        export_outputs(
            output_dir=output_dir,
            healthy_outputs=None,
            af_outputs=af_outputs,
            metrics=None,
            figure=None,
            export_format=export_format,
            duration_s=duration_s,
            dt_output_s=dt_output_s,
            random_seed=random_seed,
            solver_settings=ode_settings,
            af_config=af_cfg,
        )
        return 0

    compare_bundle = simulate_compare(
        params=params,
        duration_s=duration_s,
        dt_output_s=dt_output_s,
        random_seed=random_seed,
        solver_cfg=ode_settings,
        overlay_cfg=af_cfg,
    )
    healthy_outputs = compare_bundle["healthy"]
    af_outputs = compare_bundle["af"]
    metrics = compare_bundle["metrics"]

    t_end_plot = min(duration_s, args.t_end_s) if args.t_end_s is not None else min(3.0, duration_s)
    figure = plot_onepage_comparison(
        healthy_outputs,
        af_outputs,
        t_start_s=args.t_start_s,
        t_end_s=t_end_plot
    )
    export_outputs(
        output_dir=output_dir,
        healthy_outputs=healthy_outputs,
        af_outputs=af_outputs,
        metrics=metrics,
        figure=figure,
        export_format=export_format,
        duration_s=duration_s,
        dt_output_s=dt_output_s,
        random_seed=random_seed,
        solver_settings=ode_settings,
        af_config=af_cfg,
    )
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
