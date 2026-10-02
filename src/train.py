"""Batch 1 -> Batch 2 cycle-life regression with policy-grouped validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.model_utils import FittedCycleLifeModel

from src.features import (
    BATCH_FILES,
    DAY1_BASE_FEATURES,
    DAY1_DELTAQ_FEATURES,
    DAY1_POLICY_FEATURES,
    FEATURE_COLUMNS,
    PAPER_DISCHARGE_FEATURES,
    PAPER_FULL_FEATURES,
    build_feature_table,
)


SEED = 42
PAPER_MAPE_TARGET_PCT = 9.1


def model_candidates():
    """Candidate set follows DAY 1, with published paper feature variants added."""
    def linear_pipeline(estimator):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", estimator),
        ])

    def tree_pipeline():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=1)),
        ])

    elastic_grid = {
        # The paper's exact MATLAB search grid is not public. Search a broad
        # log scale for the documented squared-error Elastic Net objective.
        "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "model__l1_ratio": [0.01, 0.1, 0.5, 0.9, 1.0],
    }
    forest_grid = {
        "model__max_depth": [2, 3, 4],
        "model__min_samples_leaf": [3, 5, 8],
        "model__max_features": [1.0],
    }
    ridge_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    return {
        "DummyMedian": (
            "Day1_base6",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DummyRegressor(strategy="median")),
            ]),
            [{}],
            "raw",
        ),
        "DeltaQ_LinearRegression": (
            "DeltaQ_only", linear_pipeline(LinearRegression()), [{}], "raw"
        ),
        "Day1_ElasticNet": (
            "Day1_base6", linear_pipeline(ElasticNet(max_iter=100_000, random_state=SEED)), elastic_grid, "raw"
        ),
        "Day1_Ridge": ("Day1_base6", linear_pipeline(Ridge()), ridge_grid, "raw"),
        "Day1_RandomForest": ("Day1_base6", tree_pipeline(), forest_grid, "raw"),
        "Day1_Policy_ElasticNet": (
            "Day1_plus_policy", linear_pipeline(ElasticNet(max_iter=100_000, random_state=SEED)), elastic_grid, "raw"
        ),
        "Paper_Discharge_ElasticNet": (
            "Paper_Discharge", linear_pipeline(ElasticNet(max_iter=100_000, random_state=SEED)), elastic_grid, "log10"
        ),
        "Paper_Full_ElasticNet": (
            "Paper_Full", linear_pipeline(ElasticNet(max_iter=100_000, random_state=SEED)), elastic_grid, "log10"
        ),
    }


FEATURE_SETS = {
    "DeltaQ_only": DAY1_DELTAQ_FEATURES,
    "Day1_base6": DAY1_BASE_FEATURES,
    "Day1_plus_policy": DAY1_POLICY_FEATURES,
    "Paper_Discharge": PAPER_DISCHARGE_FEATURES,
    "Paper_Full": PAPER_FULL_FEATURES,
}


def _target_to_model_scale(y, target_scale: str):
    values = np.asarray(y, dtype=float)
    if target_scale == "log10":
        if np.any(values <= 0):
            raise ValueError("Cycle life must be positive before log10 transformation.")
        return np.log10(values)
    return values


def _target_to_cycles(y_prediction, target_scale: str):
    values = np.asarray(y_prediction, dtype=float)
    return np.power(10.0, values) if target_scale == "log10" else values


def regression_metrics(y_true, y_pred):
    return {
        "mape_pct": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "mae_cycles": float(mean_absolute_error(y_true, y_pred)),
        "rmse_cycles": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _group_kfold_splits(X, y, groups, max_splits):
    n_groups = pd.Series(groups).nunique()
    n_splits = min(max_splits, n_groups)
    if n_splits < 2:
        raise ValueError("At least two charging-policy groups are required.")
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(X, y, groups))


def _nested_group_cv_predictions(estimator, param_grid, X, y, groups, target_scale="raw"):
    outer_splits = _group_kfold_splits(X, y, groups, max_splits=5)
    predictions = np.full(len(y), np.nan, dtype=float)
    for fold, (train_idx, valid_idx) in enumerate(outer_splits, start=1):
        x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        groups_train = groups.iloc[train_idx]
        inner_splits = _group_kfold_splits(
            x_train, y_train, groups_train, max_splits=3
        )
        search = GridSearchCV(
            estimator=clone(estimator),
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=inner_splits,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        search.fit(x_train, _target_to_model_scale(y_train, target_scale))
        predictions[valid_idx] = _target_to_cycles(search.predict(X.iloc[valid_idx]), target_scale)
        print(f"  outer fold {fold}/{len(outer_splits)} complete")
    if not np.isfinite(predictions).all():
        raise RuntimeError("Nested CV did not produce predictions for every cell.")
    return predictions


def _fit_group_search(estimator, param_grid, X, y, groups, target_scale="raw"):
    splits = _group_kfold_splits(X, y, groups, max_splits=3)
    search = GridSearchCV(
        estimator=clone(estimator),
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=splits,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(X, _target_to_model_scale(y, target_scale))
    return search


def run_experiment(
    feature_table: pd.DataFrame,
    results_dir: str | Path = "results",
    evaluate_batch3: bool = False,
):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    all_features = sorted({feature for cols in FEATURE_SETS.values() for feature in cols})
    missing_columns = set(all_features + ["batch", "cell_id", "cycle_life", "charging_policy"]) - set(feature_table)
    if missing_columns:
        raise ValueError(f"Feature table is missing columns: {sorted(missing_columns)}")

    known = feature_table.loc[feature_table["cycle_life"].notna()].copy()
    known["cycle_life"] = pd.to_numeric(known["cycle_life"], errors="coerce")
    batch1 = known.loc[known["batch"].eq("Batch 1")].reset_index(drop=True)
    batch2 = known.loc[known["batch"].eq("Batch 2")].reset_index(drop=True)
    batch3 = known.loc[known["batch"].eq("Batch 3")].reset_index(drop=True)
    if batch1.empty or batch2.empty:
        raise ValueError("Batch 1 training and Batch 2 test cells with known targets are required.")

    y_all = batch1["cycle_life"].astype(float)
    groups_all = batch1["charging_policy"].astype(str)

    # Hold out whole charging policies inside Batch 1.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    dev_idx, valid_idx = next(splitter.split(batch1, y_all, groups_all))
    y_dev = y_all.iloc[dev_idx]
    groups_dev = groups_all.iloc[dev_idx]
    y_valid = y_all.iloc[valid_idx]
    valid_meta = batch1.iloc[valid_idx].reset_index(drop=True)

    print(
        f"Batch 1: {len(batch1)} labeled cells; "
        f"development={len(dev_idx)}, holdout={len(valid_idx)}"
    )
    print(
        f"Batch 2: {len(batch2)} labeled test cells; "
        f"Batch 3: {len(batch3)} labeled cells"
    )
    print(
        "Batch 1 target ranges: "
        f"development {y_dev.min():.0f}-{y_dev.max():.0f}, "
        f"holdout {y_valid.min():.0f}-{y_valid.max():.0f}; "
        f"Batch 2 {batch2.cycle_life.min():.0f}-{batch2.cycle_life.max():.0f}"
    )

    candidates = model_candidates()
    cv_results, prediction_rows, fitted_all = [], [], {}
    for model_name, (feature_set_name, estimator, param_grid, target_scale) in candidates.items():
        features = FEATURE_SETS[feature_set_name]
        X_all = batch1[features]
        X_dev = X_all.iloc[dev_idx]
        X_valid = X_all.iloc[valid_idx]
        print(f"Nested grouped CV: {model_name} ({feature_set_name}, {len(features)} features)")
        oof = _nested_group_cv_predictions(
            estimator, param_grid, X_dev, y_dev, groups_dev, target_scale
        )
        row = {
            "model": model_name,
            "feature_set": feature_set_name,
            "feature_count": len(features),
            "features": ";".join(features),
            "target_scale": target_scale,
            **regression_metrics(y_dev, oof),
        }

        # Validation and test scores are reported for comparison. The held-out
        # Batch 1 validation MAPE selects the candidate; Batch 2/3 never do.
        development_search = _fit_group_search(
            estimator, param_grid, X_dev, y_dev, groups_dev, target_scale
        )
        valid_prediction = _target_to_cycles(development_search.predict(X_valid), target_scale)
        valid_metrics = regression_metrics(y_valid, valid_prediction)
        row["valid_mape_pct"] = valid_metrics["mape_pct"]
        row["valid_mae_cycles"] = valid_metrics["mae_cycles"]
        row["valid_rmse_cycles"] = valid_metrics["rmse_cycles"]

        all_search = _fit_group_search(estimator, param_grid, X_all, y_all, groups_all, target_scale)
        fitted_all[model_name] = (
            all_search.best_estimator_, features, all_search.best_params_, target_scale
        )
        row["best_params"] = json.dumps(all_search.best_params_, sort_keys=True)
        for split_name, frame in [("Batch 2 test", batch2), ("Batch 3 additional test", batch3)]:
            if split_name.startswith("Batch 3") and not evaluate_batch3:
                continue
            X_split = frame[features]
            y_split = frame["cycle_life"].astype(float)
            split_prediction = _target_to_cycles(all_search.predict(X_split), target_scale)
            split_metrics = regression_metrics(y_split, split_prediction)
            prefix = "test_batch2" if split_name == "Batch 2 test" else "test_batch3"
            row[f"{prefix}_mape_pct"] = split_metrics["mape_pct"]
            row[f"{prefix}_mae_cycles"] = split_metrics["mae_cycles"]
            row[f"{prefix}_rmse_cycles"] = split_metrics["rmse_cycles"]
            row[f"{prefix}_n"] = len(frame)

            pred_rows = frame.copy()
            pred_rows["model"] = model_name
            pred_rows["feature_set"] = feature_set_name
            pred_rows["split"] = split_name
            pred_rows["prediction"] = split_prediction
            prediction_rows.append(pred_rows)

        valid_rows = valid_meta.copy()
        valid_rows["model"] = model_name
        valid_rows["feature_set"] = feature_set_name
        valid_rows["split"] = "Batch 1 holdout"
        valid_rows["prediction"] = valid_prediction
        prediction_rows.append(valid_rows)
        cv_results.append(row)

    performance = pd.DataFrame(cv_results).sort_values("mape_pct").reset_index(drop=True)
    performance = performance.rename(columns={
        "mape_pct": "train_cv_mape_pct",
        "mae_cycles": "train_cv_mae_cycles",
        "rmse_cycles": "train_cv_rmse_cycles",
    })
    selected_name = performance.sort_values(
        ["valid_mape_pct", "train_cv_mape_pct"]
    ).iloc[0]["model"]
    performance["selected_by_batch1_valid"] = performance["model"].eq(selected_name)
    chosen = performance["selected_by_batch1_valid"]
    performance["gap_valid_minus_train_pp"] = performance["valid_mape_pct"] - performance["train_cv_mape_pct"]
    performance["gap_test_minus_valid_pp"] = performance["test_batch2_mape_pct"] - performance["valid_mape_pct"]
    performance["gap_test_minus_paper_target_pp"] = performance["test_batch2_mape_pct"] - PAPER_MAPE_TARGET_PCT
    performance["gap_batch2_minus_batch3_pp"] = performance["test_batch2_mape_pct"] - performance["test_batch3_mape_pct"]
    performance["gap_batch3_minus_paper_target_pp"] = performance["test_batch3_mape_pct"] - PAPER_MAPE_TARGET_PCT
    selected_estimator, selected_features, selected_params, selected_target_scale = fitted_all[selected_name]
    selected_model = FittedCycleLifeModel(selected_estimator, selected_target_scale)
    print(f"Selected by Batch 1 policy-group holdout: {selected_name}")

    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions["error_cycles"] = predictions["prediction"] - predictions["cycle_life"]
    predictions["absolute_error_cycles"] = predictions["error_cycles"].abs()
    predictions["absolute_percentage_error_pct"] = (
        predictions["absolute_error_cycles"] / predictions["cycle_life"] * 100
    )
    predictions.to_csv(results_dir / "test_predictions.csv", index=False)
    batch2_errors = predictions[
        predictions["split"].eq("Batch 2 test") & predictions["model"].eq(selected_name)
    ].copy()
    batch2_errors.nlargest(10, "absolute_error_cycles").to_csv(
        results_dir / "error_analysis.csv", index=False
    )
    joblib.dump(selected_model, results_dir / "selected_model.joblib")

    metadata = {
        "selected_model": selected_name,
        "selected_feature_set": performance.loc[chosen, "feature_set"].iloc[0],
        "selected_parameters": selected_params,
        "features": selected_features,
        "model_selection": "lowest Batch 1 policy-group holdout MAPE; ties use nested grouped-CV MAPE",
        "hyperparameter_tuning_objective": "mean squared error on each candidate's training target scale",
        "target_scale": selected_target_scale,
        "reported_prediction_unit": "cycle_life in cycles",
        "split": {
            "training": "Batch 1 (2017-05-12)",
            "validation": "20% policy-group holdout within Batch 1",
            "test": "Batch 2",
            "additional_test": "Batch 3" if evaluate_batch3 else None,
            "random_seed": SEED,
        },
        "validation_policies": sorted(valid_meta["charging_policy"].astype(str).unique()),
        "paper_mape_reference_pct": PAPER_MAPE_TARGET_PCT,
    }
    (results_dir / "selected_model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    performance.to_csv(results_dir / "model_performance.csv", index=False)
    print("\nCandidate performance (MAPE %, MAE/RMSE in cycles):")
    print(performance[[
        "model", "feature_set", "train_cv_mape_pct", "valid_mape_pct",
        "test_batch2_mape_pct", "test_batch3_mape_pct", "selected_by_batch1_valid",
    ]].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSelected features: {selected_features}")
    print(f"Selected parameters: {selected_params}")
    print(f"Saved results under {results_dir.resolve()}")
    return performance, predictions, selected_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="results/feature_table.csv")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="If --features does not exist, extract features from this directory of .mat files.",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--evaluate-batch3", action="store_true")
    args = parser.parse_args()

    feature_path = Path(args.features)
    if feature_path.is_file():
        features = pd.read_csv(feature_path)
    elif args.data_dir:
        features = build_feature_table(args.data_dir)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(feature_path, index=False)
    else:
        raise FileNotFoundError(
            f"Feature table not found: {feature_path}. Pass --data-dir to build it."
        )
    run_experiment(features, args.results_dir, args.evaluate_batch3)


if __name__ == "__main__":
    main()
