"""Build cell-level early-life features from the three MATLAB v7.3 batch files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


BATCH_FILES = {
    "Batch 1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "Batch 2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "Batch 3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

FEATURE_COLUMNS = [
    "dq_log_variance",
    "early_QD_mean",
    "early_QD_slope",
    "early_IR_mean",
    "early_Tavg_mean",
    "charge_C_mean",
]

DAY1_DELTAQ_FEATURES = ["dq_log_variance"]
DAY1_BASE_FEATURES = FEATURE_COLUMNS.copy()
DAY1_POLICY_FEATURES = FEATURE_COLUMNS + ["policy_C1", "policy_C2", "policy_switch_SOC"]
PAPER_DISCHARGE_FEATURES = [
    "log_abs_min_dq_100_10", "log_var_dq_100_10", "log_skew_dq_100_10",
    "log_kurtosis_dq_100_10", "qd_cycle_2", "qd_max_minus_cycle_2",
]
PAPER_FULL_FEATURES = [
    "log_var_dq_100_10", "log_abs_min_dq_100_10", "qd_slope_2_100",
    "qd_intercept_2_100", "qd_cycle_2", "charge_time_mean_2_6",
    "temperature_integral_2_100", "ir_min_2_100", "ir_diff_100_2",
]


def _resolve(file: h5py.File, value):
    if isinstance(value, h5py.Reference):
        if not value:
            return None
        return file[value]
    return value


def _read_numeric(file: h5py.File, value) -> np.ndarray:
    node = _resolve(file, value)
    if node is None:
        return np.array([], dtype=float)
    if isinstance(node, h5py.Dataset) and node.attrs.get("MATLAB_empty", 0):
        return np.array([], dtype=float)
    try:
        raw = node[()] if isinstance(node, h5py.Dataset) else node
        raw = np.asarray(raw)
        if raw.dtype.kind == "O":
            parts = [_read_numeric(file, item) for item in raw.ravel()]
            parts = [part for part in parts if part.size]
            return np.concatenate(parts) if parts else np.array([], dtype=float)
        return np.asarray(raw, dtype=float).ravel()
    except (TypeError, ValueError, OSError):
        return np.array([], dtype=float)


def _cell_node(file: h5py.File, batch: h5py.Group, field: str, index: int):
    return _resolve(file, batch[field][index, 0])


def _matlab_text(file: h5py.File, value) -> str:
    node = _resolve(file, value)
    if node is None:
        return "unknown"
    try:
        raw = np.asarray(node[()] if isinstance(node, h5py.Dataset) else node)
        if raw.dtype.kind in "ui":
            chars = "".join(chr(int(code)) for code in raw.ravel(order="F") if int(code))
            return chars.strip() or "unknown"
        if raw.dtype.kind == "S":
            return b"".join(raw.ravel()).decode("utf-8", errors="replace").strip() or "unknown"
        if raw.dtype.kind == "U":
            return "".join(raw.ravel(order="F")).strip() or "unknown"
    except (TypeError, ValueError, OSError):
        pass
    return "unknown"


def _cycle_signal(file: h5py.File, cycles: h5py.Group, key: str, index: int) -> np.ndarray:
    if key not in cycles:
        return np.array([], dtype=float)
    refs = cycles[key]
    if index >= refs.shape[0]:
        return np.array([], dtype=float)
    return _read_numeric(file, refs[index, 0])


def _slope(cycle: np.ndarray, value: np.ndarray) -> float:
    valid = np.isfinite(cycle) & np.isfinite(value)
    if valid.sum() < 10 or np.unique(cycle[valid]).size < 2:
        return np.nan
    return float(np.polyfit(cycle[valid], value[valid], 1)[0])


def _extract_cell(file: h5py.File, batch: h5py.Group, batch_name: str, index: int) -> dict:
    summary = _cell_node(file, batch, "summary", index)
    cycles = _cell_node(file, batch, "cycles", index)
    if not isinstance(summary, h5py.Group) or not isinstance(cycles, h5py.Group):
        raise ValueError(f"{batch_name} cell {index}: invalid HDF5 summary/cycles structure")

    cycle = _read_numeric(file, summary["cycle"])
    qd = _read_numeric(file, summary["QDischarge"])
    ir = _read_numeric(file, summary["IR"])
    tavg = _read_numeric(file, summary["Tavg"])
    tmax = _read_numeric(file, summary["Tmax"])
    tmin = _read_numeric(file, summary["Tmin"])
    chargetime = _read_numeric(file, summary["chargetime"])
    n = min(len(cycle), len(qd), len(ir), len(tavg), len(tmax), len(tmin), len(chargetime))
    cycle, qd, ir, tavg = cycle[:n], qd[:n], ir[:n], tavg[:n]
    tmax, tmin, chargetime = tmax[:n], tmin[:n], chargetime[:n]

    baseline_mask = (
        np.isfinite(cycle) & (cycle >= 2) & (cycle <= 10)
        & np.isfinite(qd) & (qd >= 0.5) & (qd <= 2.0)
    )
    baseline_qd = float(np.median(qd[baseline_mask])) if baseline_mask.any() else np.nan
    valid_qd = (
        np.isfinite(qd) & (qd > 0) & np.isfinite(baseline_qd)
        & (qd <= 1.3 * baseline_qd)
    )
    early_mask = (cycle >= 2) & (cycle <= 100)
    qd_early = qd[early_mask & valid_qd]
    qd_cycle_early = cycle[early_mask & valid_qd]
    ir_early = ir[early_mask & np.isfinite(ir) & (ir > 0)]
    temp_early = tavg[early_mask & np.isfinite(tavg) & (tavg > 0)]
    slope_mask = (cycle >= 50) & (cycle <= 100) & valid_qd
    early_slope = _slope(cycle[slope_mask], qd[slope_mask])

    # Match the paper's ΔQ(V) voltage interval: 2.0–3.5 V.
    dq_log_variance, dq_log_min, dq_log_skew, dq_log_kurtosis = (np.nan,) * 4
    if cycles["Qdlin"].shape[0] > 99 and "Vdlin" in batch:
        voltage = _read_numeric(file, _cell_node(file, batch, "Vdlin", index))
        q10 = _cycle_signal(file, cycles, "Qdlin", 9)
        q100 = _cycle_signal(file, cycles, "Qdlin", 99)
        if len(voltage) == len(q10) == len(q100) and len(voltage) > 10:
            valid = (np.isfinite(voltage) & np.isfinite(q10) & np.isfinite(q100)
                     & (voltage >= 2.0) & (voltage <= 3.5))
            delta_q = (q100[valid] - q10[valid])
            delta_q = delta_q[np.isfinite(delta_q)]
            if delta_q.size > 10:
                variance = float(np.var(delta_q, ddof=1))
                dq_log_variance = float(np.log10(max(variance, 1e-15)))
                if np.min(np.abs(delta_q)) > 0:
                    dq_log_min = float(np.log10(abs(np.min(delta_q))))
                centered = delta_q - np.mean(delta_q)
                moment2 = float(np.mean(centered ** 2))
                if moment2 > 0:
                    skew = float(np.mean(centered ** 3) / moment2 ** 1.5)
                    kurtosis = float(np.mean(centered ** 4) / moment2 ** 2)
                    if abs(skew) > 0:
                        dq_log_skew = float(np.log10(abs(skew)))
                    if abs(kurtosis) > 0:
                        dq_log_kurtosis = float(np.log10(abs(kurtosis)))

    # Average measured charging C-rate over cycles 2-100.
    cycle_c_rates = []
    if np.isfinite(baseline_qd) and baseline_qd > 0:
        n_charge_cycles = min(cycles["I"].shape[0], cycles["t"].shape[0], 100)
        for cycle_index in range(1, n_charge_cycles):
            current = _cycle_signal(file, cycles, "I", cycle_index)
            time = _cycle_signal(file, cycles, "t", cycle_index)
            if len(current) != len(time) or len(current) < 2:
                continue
            dt = np.diff(time)
            mid_current = (current[:-1] + current[1:]) / 2
            mask = (
                np.isfinite(dt) & np.isfinite(mid_current)
                & (dt > 0) & (mid_current > 0.05)
            )
            if mask.any():
                average_current = np.average(mid_current[mask], weights=dt[mask])
                # DAY 1 defines C-rate relative to the nominal cell capacity.
                cycle_c_rates.append(average_current / 1.1)

    # Paper Discharge/Full features beyond the DAY 1 feature set.
    qd_fit_mask = np.isfinite(cycle) & np.isfinite(qd) & (cycle >= 2) & (cycle <= 100)
    qd_slope_2_100, qd_intercept_2_100 = np.nan, np.nan
    if qd_fit_mask.sum() >= 2 and np.unique(cycle[qd_fit_mask]).size >= 2:
        qd_slope_2_100, qd_intercept_2_100 = np.polyfit(cycle[qd_fit_mask], qd[qd_fit_mask], 1)
    qd2_indices = np.flatnonzero(np.isfinite(cycle) & (cycle == 2) & np.isfinite(qd))
    qd_cycle_2 = float(qd[qd2_indices[0]]) if qd2_indices.size else np.nan
    qd_early = qd[qd_fit_mask]
    qd_max_minus_cycle_2 = (
        float(np.max(qd_early) - qd_cycle_2)
        if qd_early.size and np.isfinite(qd_cycle_2) else np.nan
    )
    charge_time_mask = np.isfinite(cycle) & np.isfinite(chargetime) & (cycle >= 2) & (cycle <= 6)
    charge_time_mean_2_6 = float(np.mean(chargetime[charge_time_mask])) if charge_time_mask.any() else np.nan
    ir_mask = np.isfinite(cycle) & np.isfinite(ir) & (cycle >= 2) & (cycle <= 100) & (ir != 0)
    ir_min_2_100 = float(np.min(ir[ir_mask])) if ir_mask.any() else np.nan
    ir2 = np.flatnonzero(np.isfinite(cycle) & (cycle == 2) & np.isfinite(ir))
    ir100 = np.flatnonzero(np.isfinite(cycle) & (cycle == 100) & np.isfinite(ir))
    ir_diff_100_2 = float(ir[ir100[0]] - ir[ir2[0]]) if ir2.size and ir100.size else np.nan

    temperature_integral_2_100, temperature_cycles = 0.0, 0
    for cycle_index in range(1, min(100, cycles["T"].shape[0], cycles["t"].shape[0])):
        time = _cycle_signal(file, cycles, "t", cycle_index)
        temp = _cycle_signal(file, cycles, "T", cycle_index)
        if len(time) == len(temp) and len(time) >= 2:
            valid = np.isfinite(time) & np.isfinite(temp)
            if valid.sum() >= 2:
                tv, vv = time[valid], temp[valid]
                temperature_integral_2_100 += float(
                    np.sum((vv[:-1] + vv[1:]) * 0.5 * np.diff(tv))
                )
                temperature_cycles += 1
    if temperature_cycles != 99:
        temperature_integral_2_100 = np.nan

    policy = _matlab_text(file, _cell_node(file, batch, "policy_readable", index))
    policy_match = re.match(r"\s*([0-9.]+)C\(([0-9.]+)%\)-([0-9.]+)C", policy)
    policy_c1, policy_soc, policy_c2 = (
        tuple(map(float, policy_match.groups())) if policy_match else (np.nan, np.nan, np.nan)
    )

    target_values = _read_numeric(file, _cell_node(file, batch, "cycle_life", index))
    target_values = target_values[np.isfinite(target_values)]
    cycle_life = float(target_values[0]) if target_values.size == 1 else np.nan
    return {
        "batch": batch_name,
        "cell_id": f"{batch_name.lower().replace(' ', '-')}-cell-{index:02d}",
        "local_cell_id": index,
        "cycle_life": cycle_life,
        "charging_policy": policy,
        "baseline_QD": baseline_qd,
        "early_QD_mean": float(np.mean(qd_early)) if qd_early.size else np.nan,
        "early_QD_slope": early_slope,
        "early_IR_mean": float(np.mean(ir_early)) if ir_early.size else np.nan,
        "early_Tavg_mean": float(np.mean(temp_early)) if temp_early.size else np.nan,
        "charge_C_mean": float(np.mean(cycle_c_rates)) if cycle_c_rates else np.nan,
        "dq_log_variance": dq_log_variance,
        "log_abs_min_dq_100_10": dq_log_min,
        "log_var_dq_100_10": dq_log_variance,
        "log_skew_dq_100_10": dq_log_skew,
        "log_kurtosis_dq_100_10": dq_log_kurtosis,
        "qd_slope_2_100": float(qd_slope_2_100),
        "qd_intercept_2_100": float(qd_intercept_2_100),
        "qd_cycle_2": qd_cycle_2,
        "qd_max_minus_cycle_2": qd_max_minus_cycle_2,
        "charge_time_mean_2_6": charge_time_mean_2_6,
        "temperature_integral_2_100": temperature_integral_2_100,
        "ir_min_2_100": ir_min_2_100,
        "ir_diff_100_2": ir_diff_100_2,
        "policy_C1": policy_c1,
        "policy_C2": policy_c2,
        "policy_switch_SOC": policy_soc,
        "temperature_cycles_observed": temperature_cycles,
        "early_QD_cycles": int(np.unique(qd_cycle_early).size),
        "charge_cycles_observed": len(cycle_c_rates),
    }


def build_feature_table(data_dir: str | Path, batches: dict[str, str] | None = None) -> pd.DataFrame:
    """Extract one row per cell; missing cycle_life targets remain NaN."""
    data_dir = Path(data_dir).expanduser()
    batches = batches or BATCH_FILES
    rows = []
    for batch_name, filename in batches.items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {batch_name} file: {path}")
        print(f"Extracting {batch_name}: {filename}")
        with h5py.File(path, "r") as file:
            batch = file["batch"]
            n_cells = batch["summary"].shape[0]
            for index in range(n_cells):
                rows.append(_extract_cell(file, batch, batch_name, index))
        print(f"  extracted {n_cells} cells")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing the three .mat files")
    parser.add_argument("--output", default="results/feature_table.csv")
    args = parser.parse_args()
    table = build_feature_table(args.data_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    cells = table.drop_duplicates("cell_id")
    print(f"Saved {len(table)} cell rows to {output}")
    print("Cells by batch:")
    print(cells.groupby("batch").size())
    print(f"Missing cycle_life targets: {cells['cycle_life'].isna().sum()}")


if __name__ == "__main__":
    main()
