#!/usr/bin/env python
import argparse
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


@dataclass
class SegmentData:
    time_s: np.ndarray
    dt_s: np.ndarray
    current_a: np.ndarray
    voltage_v: np.ndarray
    soc_ref: np.ndarray
    ts_ms: np.ndarray


def load_data(csv_path: str) -> pd.DataFrame:
    usecols = [
        "ts_ms",
        "Battery_timestamp",
        "Battery_level",
        "Battery_voltage",
        "Battery_current_avg",
        "Battery_temperature",
        "Battery_status",
        "Battery_charge_type",
    ]
    df = pd.read_csv(csv_path, usecols=usecols)
    df = df.dropna(subset=["ts_ms", "Battery_level", "Battery_voltage", "Battery_current_avg"])
    df = df.sort_values("ts_ms").reset_index(drop=True)
    return df


def select_discharge_segment(df: pd.DataFrame) -> SegmentData:
    mask = (df["Battery_charge_type"] == 0) & (df["Battery_status"] == 3)
    df = df[mask].copy()
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError("No discharge data found with Battery_charge_type=0 and Battery_status=3.")

    level = df["Battery_level"].to_numpy()
    ts_ms = df["ts_ms"].to_numpy()

    level_diff = np.diff(level, prepend=level[0])
    non_increasing = level_diff <= 0.1
    df = df[non_increasing].copy().reset_index(drop=True)
    level = df["Battery_level"].to_numpy()
    ts_ms = df["ts_ms"].to_numpy()

    time_gap = np.diff(ts_ms, prepend=ts_ms[0]) > 60000
    level_jump = np.diff(level, prepend=level[0]) > 0.1
    segment_id = np.cumsum(time_gap | level_jump)

    counts = pd.Series(segment_id).value_counts().sort_values(ascending=False)
    best_segment = counts.index[0]
    segment = df[segment_id == best_segment].copy().reset_index(drop=True)

    ts_ms = segment["ts_ms"].to_numpy()
    time_s = (ts_ms - ts_ms[0]) / 1000.0
    dt_s = np.diff(time_s, prepend=time_s[0])
    if len(dt_s) > 1:
        dt_s[0] = dt_s[1]
    dt_s[dt_s <= 0] = np.median(dt_s[dt_s > 0]) if np.any(dt_s > 0) else 1.0

    current_a = segment["Battery_current_avg"].to_numpy() / 1000.0
    voltage_v = segment["Battery_voltage"].to_numpy() / 1000.0
    soc_ref = segment["Battery_level"].to_numpy() / 100.0

    return SegmentData(time_s=time_s, dt_s=dt_s, current_a=current_a, voltage_v=voltage_v, soc_ref=soc_ref, ts_ms=ts_ms)


def estimate_capacity_ah(current_a: np.ndarray, dt_s: np.ndarray, soc_ref: np.ndarray) -> float:
    soc_drop = soc_ref[0] - soc_ref[-1]
    if soc_drop <= 0:
        raise ValueError("SOC does not decrease in the selected segment; cannot estimate capacity.")
    charge_ah = np.sum(current_a * dt_s) / 3600.0
    return charge_ah / soc_drop


def fit_ocv_curve(soc: np.ndarray, voltage_v: np.ndarray, current_a: np.ndarray) -> UnivariateSpline:
    quantile = 0.1
    low_mask = current_a <= np.quantile(current_a, quantile)
    if low_mask.sum() < 20:
        quantile = 0.2
        low_mask = current_a <= np.quantile(current_a, quantile)

    soc_low = soc[low_mask]
    volt_low = voltage_v[low_mask]
    order = np.argsort(soc_low)
    soc_low = soc_low[order]
    volt_low = volt_low[order]

    smoothing = 0.0005 * len(soc_low)
    return UnivariateSpline(soc_low, volt_low, s=smoothing)


def simulate_voltage(
    params: np.ndarray,
    soc: np.ndarray,
    current_a: np.ndarray,
    dt_s: np.ndarray,
    ocv_func: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    r0, r1, c1, r2, c2 = params
    v1 = 0.0
    v2 = 0.0
    vt_pred = np.zeros_like(soc)
    for k in range(len(soc)):
        i = current_a[k]
        ocv = float(ocv_func(soc[k]))
        vt_pred[k] = ocv - i * r0 - v1 - v2
        if k < len(soc) - 1:
            a1 = np.exp(-dt_s[k] / (r1 * c1))
            a2 = np.exp(-dt_s[k] / (r2 * c2))
            v1 = a1 * v1 + r1 * (1 - a1) * i
            v2 = a2 * v2 + r2 * (1 - a2) * i
    return vt_pred


def fit_rc_params(
    soc: np.ndarray,
    current_a: np.ndarray,
    voltage_v: np.ndarray,
    dt_s: np.ndarray,
    ocv_func: Callable[[np.ndarray], np.ndarray],
    max_points: int = 5000,
) -> np.ndarray:
    if len(soc) > max_points:
        stride = max(1, len(soc) // max_points)
        indices = np.arange(0, len(soc), stride)
        soc = soc[indices]
        current_a = current_a[indices]
        voltage_v = voltage_v[indices]
        dt_s = dt_s[indices]

    def residuals(params: np.ndarray) -> np.ndarray:
        vt_pred = simulate_voltage(params, soc, current_a, dt_s, ocv_func)
        return vt_pred - voltage_v

    x0 = np.array([0.02, 0.01, 2000.0, 0.005, 6000.0])
    lower = np.array([1e-4, 1e-4, 10.0, 1e-4, 10.0])
    upper = np.array([0.2, 0.2, 200000.0, 0.2, 200000.0])
    result = least_squares(residuals, x0, bounds=(lower, upper), max_nfev=200)
    return result.x


def coulomb_count_soc(soc0: float, current_a: np.ndarray, dt_s: np.ndarray, capacity_ah: float) -> np.ndarray:
    charge_ah = np.cumsum(current_a * dt_s) / 3600.0
    soc = soc0 - charge_ah / capacity_ah
    return np.clip(soc, 0.0, 1.0)


def ekf_soc_estimate(
    soc0: float,
    current_a: np.ndarray,
    voltage_v: np.ndarray,
    dt_s: np.ndarray,
    capacity_ah: float,
    params: np.ndarray,
    ocv_func: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    r0, r1, c1, r2, c2 = params
    x = np.array([soc0, 0.0, 0.0])
    p = np.diag([1e-4, 1e-3, 1e-3])
    q = np.diag([1e-6, 1e-4, 1e-4])
    r = np.array([[0.0025 ** 2]])
    docv = ocv_func.derivative()

    soc_est = np.zeros_like(voltage_v)

    for k in range(len(voltage_v)):
        i = current_a[k]
        dt = dt_s[k]
        a1 = np.exp(-dt / (r1 * c1))
        a2 = np.exp(-dt / (r2 * c2))

        x_pred = np.array(
            [
                x[0] - (i * dt) / (capacity_ah * 3600.0),
                a1 * x[1] + r1 * (1 - a1) * i,
                a2 * x[2] + r2 * (1 - a2) * i,
            ]
        )
        x_pred[0] = np.clip(x_pred[0], 0.0, 1.0)

        f = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, a1, 0.0],
                [0.0, 0.0, a2],
            ]
        )
        p_pred = f @ p @ f.T + q

        h = float(ocv_func(x_pred[0])) - i * r0 - x_pred[1] - x_pred[2]
        h_jac = np.array([[float(docv(x_pred[0])), -1.0, -1.0]])

        s = h_jac @ p_pred @ h_jac.T + r
        k_gain = p_pred @ h_jac.T @ np.linalg.inv(s)

        y = voltage_v[k] - h
        x = x_pred + (k_gain.flatten() * y)
        x[0] = np.clip(x[0], 0.0, 1.0)
        p = (np.eye(3) - k_gain @ h_jac) @ p_pred

        soc_est[k] = x[0]

    return soc_est


def run_validation(csv_path: str, output_png: str) -> None:
    df = load_data(csv_path)
    segment = select_discharge_segment(df)

    capacity_ah = estimate_capacity_ah(segment.current_a, segment.dt_s, segment.soc_ref)
    ocv_func = fit_ocv_curve(segment.soc_ref, segment.voltage_v, segment.current_a)

    params = fit_rc_params(segment.soc_ref, segment.current_a, segment.voltage_v, segment.dt_s, ocv_func)

    soc_cc = coulomb_count_soc(segment.soc_ref[0], segment.current_a, segment.dt_s, capacity_ah)
    soc_ekf = ekf_soc_estimate(
        segment.soc_ref[0],
        segment.current_a,
        segment.voltage_v,
        segment.dt_s,
        capacity_ah,
        params,
        ocv_func,
    )

    rmse = np.sqrt(np.mean((soc_ekf - soc_cc) ** 2))
    mae = np.mean(np.abs(soc_ekf - soc_cc))

    print("Selected discharge segment:")
    print(f"  rows: {len(segment.soc_ref)}")
    print(f"  SOC range: {segment.soc_ref[0]:.3f} -> {segment.soc_ref[-1]:.3f}")
    print(f"  Estimated capacity: {capacity_ah:.3f} Ah")
    print("Fitted 2RC parameters:")
    print(f"  R0={params[0]:.5f} Ω, R1={params[1]:.5f} Ω, C1={params[2]:.1f} F")
    print(f"  R2={params[3]:.5f} Ω, C2={params[4]:.1f} F")
    print("SOC comparison (EKF vs coulomb counting):")
    print(f"  RMSE={rmse:.5f}, MAE={mae:.5f}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(segment.time_s / 60.0, segment.voltage_v, label="Measured V")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(segment.time_s / 60.0, soc_cc, label="SOC (Ah integration)")
    axes[1].plot(segment.time_s / 60.0, soc_ekf, label="SOC (EKF)")
    axes[1].plot(segment.time_s / 60.0, segment.soc_ref, label="SOC (Battery level)", alpha=0.6)
    axes[1].set_ylabel("SOC")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(segment.time_s / 60.0, soc_ekf - soc_cc, label="SOC error")
    axes[2].set_ylabel("EKF - Ah")
    axes[2].set_xlabel("Time (min)")
    axes[2].grid(True)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    print(f"Saved plot to {output_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate 2RC battery model using T4_clean.csv")
    parser.add_argument("--csv", default="T4_clean.csv", help="Path to T4_clean.csv")
    parser.add_argument("--output", default="validation_results.png", help="Output plot path")
    args = parser.parse_args()
    run_validation(args.csv, args.output)


if __name__ == "__main__":
    main()
