from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline

import protocol_decay_lab as lab


RANDOM_STATE = 42
TERMINAL_MODE_SET = {
    "terminal_global_decay",
    "multichain_relocation",
    "native_revival_or_threshold_boundary",
    "avax_side_decay_but_globally_alive",
}


def resolve_paths() -> Tuple[Path, Path, Path]:
    output_root = Path(__file__).resolve().parents[1] / "outputs"
    dataset_dir = output_root / "dataset_v2"
    model_dir = output_root / "model_v2"
    model_dir.mkdir(parents=True, exist_ok=True)
    return output_root, dataset_dir, model_dir


def rf_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_launch_dates(obs_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for csv_path in sorted(obs_dir.glob("*.csv")):
        slug = csv_path.stem
        obs = pd.read_csv(csv_path, parse_dates=["date"])
        avax = obs[obs["avax_core_tvl"].notna()][["date", "avax_core_tvl"]].rename(columns={"avax_core_tvl": "tvl"})
        if avax.empty:
            continue
        trimmed, _ = lab.trim_to_activation(avax)
        if trimmed.empty:
            continue
        rows.append({"slug": slug, "launch_date": trimmed["date"].iloc[0]})
    return pd.DataFrame(rows)


def build_curve_map(obs_dir: Path, slugs: List[str]) -> Dict[str, np.ndarray]:
    curve_map: Dict[str, np.ndarray] = {}
    for slug in slugs:
        path = obs_dir / f"{slug}.csv"
        if not path.exists():
            continue
        obs = pd.read_csv(path, parse_dates=["date"])
        avax = obs[obs["avax_core_tvl"].notna()][["date", "avax_core_tvl"]].rename(columns={"avax_core_tvl": "tvl"})
        if avax.empty:
            continue
        signature = lab.build_curve_signature(avax)
        if signature is not None:
            curve_map[slug] = signature
    return curve_map


def classify_decay_mode_v2(row: pd.Series) -> str:
    if int(row.get("core_eligible", 0) or 0) != 1:
        if row.get("core_universe_status") == "data_incomplete":
            return "data_incomplete"
        return "not_core_eligible"

    target = row.get("structural_decay_target_v2")
    current_status = row.get("current_status")
    migrated = int(row.get("migrated_candidate", 0) or 0)

    if target == 1:
        if current_status == "active_on_avax":
            return "native_revival_or_threshold_boundary"
        if migrated == 1:
            return "multichain_relocation"
        return "terminal_global_decay"

    if target == 0 and current_status == "low_tvl_on_avax" and migrated == 1:
        return "avax_side_decay_but_globally_alive"

    return "resilient_or_unproven"


def early_feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {
        "slug",
        "name",
        "category",
        "current_status",
        "structural_decay_label_v2",
        "structural_decay_target_v2",
        "history_source_mode",
        "core_universe_status",
        "decay_mode_v2",
        "stage1_terminal_target",
        "launch_date",
        "old_label",
        "target",
        "label_status",
        "target_known",
        "risk_band",
        "reason_1",
        "reason_2",
        "reason_3",
    }
    cols: List[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def live_feature_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "stage1_terminal_prob",
        "num_chains",
        "avax_only",
        "current_total_core_tvl",
        "current_avax_core_tvl",
        "current_non_avax_core_tvl",
        "current_avax_share",
        "active_chain_count_core",
        "migrated_candidate",
        "data_quality_score_v2",
        "avax_drawdown_from_peak",
        "avax_tvl_30d_change",
        "avax_tvl_90d_change",
        "avax_consecutive_decline_days",
        "avax_volatility_full",
    ]
    return [col for col in candidates if col in df.columns]


def evaluate_binary(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[Dict[str, float], np.ndarray]:
    model = rf_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(model, df[feature_cols], df[target_col], cv=cv, scoring=["roc_auc", "accuracy", "neg_brier_score"])
    probs = cross_val_predict(model, df[feature_cols], df[target_col], cv=cv, method="predict_proba")[:, 1]
    summary = {
        "sample_size": int(len(df)),
        "event_rate": float(df[target_col].mean()),
        "cv_auc_mean": float(scores["test_roc_auc"].mean()),
        "cv_auc_std": float(scores["test_roc_auc"].std()),
        "oof_auc": float(roc_auc_score(df[target_col], probs)),
        "oof_accuracy": float(accuracy_score(df[target_col], (probs >= 0.5).astype(int))),
        "oof_brier": float(brier_score_loss(df[target_col], probs)),
    }
    return summary, probs


def temporal_holdout(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> pd.DataFrame:
    model = rf_pipeline()
    rows = []
    for quantile in [0.5, 0.7, 0.8]:
        cutoff = df["launch_date"].quantile(quantile)
        train_df = df[df["launch_date"] < cutoff].copy()
        test_df = df[df["launch_date"] >= cutoff].copy()
        if train_df.empty or test_df.empty:
            continue
        if train_df[target_col].nunique() < 2 or test_df[target_col].nunique() < 2:
            continue
        model.fit(train_df[feature_cols], train_df[target_col])
        probs = model.predict_proba(test_df[feature_cols])[:, 1]
        rows.append(
            {
                "split_quantile": quantile,
                "cutoff_date": cutoff,
                "train_size": int(len(train_df)),
                "test_size": int(len(test_df)),
                "auc": float(roc_auc_score(test_df[target_col], probs)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    _, dataset_dir, model_dir = resolve_paths()
    obs_dir = dataset_dir / "observations"

    early = pd.read_csv(dataset_dir / "early_features_v2.csv")
    labels = pd.read_csv(dataset_dir / "labels_v2.csv")
    summary = pd.read_csv(dataset_dir / "features_summary_v2.csv")
    registry = pd.read_csv(dataset_dir / "registry_v2.csv")
    launch_dates = build_launch_dates(obs_dir)

    early = early.merge(registry[["slug", "avax_only", "num_chains", "old_label"]], on="slug", how="left")
    early = early.merge(launch_dates, on="slug", how="left")
    early["category_code"] = early["category"].astype("category").cat.codes.astype(float)
    early["data_quality_score"] = early["data_quality_score_v2"]
    early["label_status"] = early["structural_decay_label_v2"]
    early["target"] = early["structural_decay_target_v2"]
    early["target_known"] = early["target"].isin([0, 1]).astype(int)

    curve_map = build_curve_map(obs_dir, early["slug"].tolist())
    train_like = early[early["target"].isin([0, 1])].copy()
    train_like["target"] = train_like["target"].astype(int)
    decay_proto, resilient_proto = lab.build_archetypes(train_like, curve_map)
    early = lab.append_archetype_scores(early, curve_map, decay_proto, resilient_proto)

    mode_frame = labels.merge(
        summary[
            [
                "slug",
                "avax_peak_tvl",
                "avax_current_tvl",
                "avax_drawdown_from_peak",
                "avax_tvl_30d_change",
                "avax_tvl_90d_change",
                "avax_consecutive_decline_days",
                "avax_volatility_full",
            ]
        ],
        on="slug",
        how="left",
    ).merge(registry[["slug", "avax_only", "num_chains", "old_label"]], on="slug", how="left")
    mode_frame = mode_frame.merge(launch_dates, on="slug", how="left")
    mode_frame["decay_mode_v2"] = mode_frame.apply(classify_decay_mode_v2, axis=1)

    stage1_frame = early.merge(
        mode_frame[["slug", "decay_mode_v2", "current_status", "migrated_candidate"]],
        on="slug",
        how="left",
        suffixes=("", "_labels"),
    )

    stage1_train = stage1_frame[
        (stage1_frame["structural_decay_target_v2"] == 0)
        | (stage1_frame["decay_mode_v2"] == "terminal_global_decay")
    ].copy()
    stage1_train["target"] = (stage1_train["decay_mode_v2"] == "terminal_global_decay").astype(int)
    stage1_train["stage1_terminal_target"] = stage1_train["target"]
    stage1_features = early_feature_columns(stage1_train)

    leaderboard, weights, oof_predictions = lab.evaluate_models(stage1_train, stage1_features)
    fitted = lab.fit_full_models(stage1_train, stage1_features)
    stage1_scored = lab.score_frame(fitted, weights, stage1_frame.copy(), stage1_features)
    stage1_scored = lab.explain_protocols(stage1_scored, stage1_train)
    ensemble_oof = sum(oof_predictions[name] * weights[name] for name in weights)
    stage1_oof = stage1_train[["slug"]].copy().reset_index(drop=True)
    stage1_oof["stage1_terminal_oof_prob"] = ensemble_oof
    stage1_train = stage1_train.merge(stage1_oof, on="slug", how="left")
    stage1_scored = stage1_scored.rename(columns={"risk_score": "stage1_terminal_prob", "score_confidence": "stage1_score_confidence"})
    temporal = temporal_holdout(stage1_train, stage1_features, "stage1_terminal_target")

    stage2_frame = mode_frame.merge(
        stage1_scored[
            [
                "slug",
                "stage1_terminal_prob",
                "stage1_score_confidence",
                "risk_uncertainty",
                "risk_band",
                "reason_1",
                "reason_2",
                "reason_3",
            ]
        ],
        on="slug",
        how="left",
    )
    stage2_train = stage2_frame[stage2_frame["decay_mode_v2"].isin(TERMINAL_MODE_SET)].copy()
    stage2_train["stage2_terminal_mode_target"] = (stage2_train["decay_mode_v2"] == "terminal_global_decay").astype(int)
    stage2_features = live_feature_columns(stage2_train)
    stage2_summary, stage2_oof = evaluate_binary(stage2_train, stage2_features, "stage2_terminal_mode_target")

    stage2_model = rf_pipeline()
    stage2_model.fit(stage2_train[stage2_features], stage2_train["stage2_terminal_mode_target"])
    stage2_frame["stage2_terminal_mode_prob"] = stage2_model.predict_proba(stage2_frame[stage2_features])[:, 1]
    stage2_oof_df = stage2_train[["slug"]].copy()
    stage2_oof_df["stage2_terminal_mode_oof_prob"] = stage2_oof
    stage2_frame = stage2_frame.merge(stage2_oof_df, on="slug", how="left")

    output = stage2_frame.merge(
        stage1_scored[
            [
                "slug",
                "stage1_terminal_prob",
                "stage1_score_confidence",
                "reason_1",
                "reason_2",
                "reason_3",
            ]
        ],
        on="slug",
        how="left",
        suffixes=("", "_stage1"),
    )
    output["composite_terminal_prob"] = output["stage1_terminal_prob"] * output["stage2_terminal_mode_prob"].fillna(1.0)
    output["composite_watch_band"] = pd.cut(
        output["composite_terminal_prob"],
        bins=[-np.inf, 0.35, 0.65, np.inf],
        labels=["monitor", "elevated", "terminal_risk"],
    ).astype(str)
    non_core_mask = output["core_eligible"] != 1
    output.loc[non_core_mask, "stage1_terminal_prob"] = np.nan
    output.loc[non_core_mask, "stage2_terminal_mode_prob"] = np.nan
    output.loc[non_core_mask, "composite_terminal_prob"] = np.nan
    output.loc[output["core_universe_status"] == "boundary_excluded", "composite_watch_band"] = "not_core_eligible"
    output.loc[output["core_universe_status"] == "data_incomplete", "composite_watch_band"] = "data_incomplete"

    summary_rows = [
        {
            "stage": "stage1_v2_terminal_early_model",
            "sample_size": int(len(stage1_train)),
            "event_rate": float(stage1_train["stage1_terminal_target"].mean()),
            "cv_auc_mean": float(leaderboard.iloc[0]["cv_auc_mean"]),
            "cv_auc_std": float(leaderboard.iloc[0]["cv_auc_std"]),
            "oof_auc": float(leaderboard.iloc[0]["oof_auc"]),
            "oof_accuracy": float(leaderboard.iloc[0]["oof_accuracy"]),
            "oof_brier": float(leaderboard.iloc[0]["oof_brier"]),
            "best_model": str(leaderboard.iloc[0]["model"]),
        },
        {
            "stage": "stage2_v2_live_mode_model",
            **stage2_summary,
            "best_model": "rf",
        },
    ]
    summary_df = pd.DataFrame(summary_rows)

    mode_counts = output["decay_mode_v2"].value_counts(dropna=False).rename_axis("decay_mode_v2").reset_index(name="protocol_count")

    leaderboard.to_csv(model_dir / "stage1_model_leaderboard_v2.csv", index=False)
    temporal.to_csv(model_dir / "stage1_temporal_validation_v2.csv", index=False)
    summary_df.to_csv(model_dir / "two_stage_v2_summary.csv", index=False)
    output.to_csv(model_dir / "two_stage_v2_scores.csv", index=False)
    output.to_csv(model_dir / "decay_mode_v2_registry.csv", index=False)
    mode_counts.to_csv(model_dir / "decay_mode_v2_counts.csv", index=False)

    rf_model = fitted["rf"].named_steps["model"]
    pd.DataFrame({"feature": stage1_features, "rf_importance": rf_model.feature_importances_}).sort_values(
        "rf_importance",
        ascending=False,
    ).to_csv(model_dir / "stage1_feature_importance_v2.csv", index=False)

    run_summary = {
        "core_eligible_protocols": int((labels["core_eligible"] == 1).sum()),
        "early_feature_rows_v2": int(len(early)),
        "stage1_train_rows": int(len(stage1_train)),
        "stage2_train_rows": int(len(stage2_train)),
        "mode_counts_v2": dict(zip(mode_counts["decay_mode_v2"], mode_counts["protocol_count"])),
    }
    with open(model_dir / "retrain_v2_summary.json", "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2, ensure_ascii=False)

    print("V2 model retraining completed.")
    print()
    print("Stage 1 leaderboard:")
    print(leaderboard.to_string(index=False))
    print()
    print("Two-stage summary:")
    print(summary_df.to_string(index=False))
    print()
    print("Mode counts:")
    print(mode_counts.to_string(index=False))


if __name__ == "__main__":
    main()
