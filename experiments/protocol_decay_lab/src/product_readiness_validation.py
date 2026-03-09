from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

import protocol_decay_lab as lab


def resolve_paths() -> Tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return repo_root, output_dir


def build_launch_dates(data_dir: Path) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(data_dir.glob("tvl_*.csv")):
        slug = csv_path.stem.replace("tvl_", "")
        df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        trimmed, _ = lab.trim_to_activation(df)
        if trimmed.empty:
            continue
        rows.append(
            {
                "slug": slug,
                "launch_date": trimmed["date"].iloc[0],
                "launch_year": int(trimmed["date"].iloc[0].year),
            }
        )
    return pd.DataFrame(rows)


def temporal_validation(train_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    quantiles = [0.50, 0.60, 0.70, 0.80]
    rows = []

    for quantile in quantiles:
        cutoff = train_df["launch_date"].quantile(quantile)
        train_slice = train_df[train_df["launch_date"] < cutoff].copy()
        test_slice = train_df[train_df["launch_date"] >= cutoff].copy()
        if train_slice["target"].nunique() < 2 or test_slice["target"].nunique() < 2:
            continue

        base_models = lab.build_models()
        weights: Dict[str, float] = {}
        split_predictions: Dict[str, np.ndarray] = {}

        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=lab.RANDOM_STATE)
        for name, pipeline in base_models.items():
            fit_scores = cross_validate(
                pipeline,
                train_slice[feature_cols],
                train_slice["target"],
                cv=inner_cv,
                scoring=["roc_auc"],
                n_jobs=None,
            )
            weights[name] = float(fit_scores["test_roc_auc"].mean())

            pipeline.fit(train_slice[feature_cols], train_slice["target"])
            probs = pipeline.predict_proba(test_slice[feature_cols])[:, 1]
            split_predictions[name] = probs

            rows.append(
                {
                    "split_quantile": quantile,
                    "cutoff_date": cutoff,
                    "model": name,
                    "train_size": int(len(train_slice)),
                    "test_size": int(len(test_slice)),
                    "train_event_rate": float(train_slice["target"].mean()),
                    "test_event_rate": float(test_slice["target"].mean()),
                    "auc": float(roc_auc_score(test_slice["target"], probs)),
                    "accuracy": float(accuracy_score(test_slice["target"], (probs >= 0.5).astype(int))),
                    "brier": float(brier_score_loss(test_slice["target"], probs)),
                }
            )

        total_weight = sum(weights.values())
        normalized = {name: value / total_weight for name, value in weights.items()}
        ensemble = sum(split_predictions[name] * normalized[name] for name in normalized)
        rows.append(
            {
                "split_quantile": quantile,
                "cutoff_date": cutoff,
                "model": "ensemble",
                "train_size": int(len(train_slice)),
                "test_size": int(len(test_slice)),
                "train_event_rate": float(train_slice["target"].mean()),
                "test_event_rate": float(test_slice["target"].mean()),
                "auc": float(roc_auc_score(test_slice["target"], ensemble)),
                "accuracy": float(accuracy_score(test_slice["target"], (ensemble >= 0.5).astype(int))),
                "brier": float(brier_score_loss(test_slice["target"], ensemble)),
            }
        )

    return pd.DataFrame(rows).sort_values(["split_quantile", "model"]).reset_index(drop=True)


def enrichment_coverage_audit(train_df: pd.DataFrame, data_dir: Path, feature_cols: List[str]) -> pd.DataFrame:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=lab.RANDOM_STATE,
                ),
            ),
        ]
    )

    rows = []
    sources = {
        "combined": data_dir / "combined_features.csv",
        "onchain": data_dir / "onchain_features.csv",
    }

    for source_name, source_path in sources.items():
        source_df = pd.read_csv(source_path)
        numeric_cols = [
            col
            for col in source_df.columns
            if col not in {"slug", "label"} and pd.api.types.is_numeric_dtype(source_df[col]) and col not in train_df.columns
        ]
        merged = train_df.merge(source_df[["slug"] + numeric_cols], on="slug", how="inner")
        if merged.empty or merged["target"].nunique() < 2:
            continue

        class_min = int(merged["target"].value_counts().min())
        n_splits = min(5, class_min)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=lab.RANDOM_STATE)

        base_auc = cross_validate(model, merged[feature_cols], merged["target"], cv=cv, scoring=["roc_auc"])["test_roc_auc"].mean()
        full_cols = feature_cols + numeric_cols
        full_auc = cross_validate(model, merged[full_cols], merged["target"], cv=cv, scoring=["roc_auc"])["test_roc_auc"].mean()

        rows.append(
            {
                "source": source_name,
                "sample_size": int(len(merged)),
                "event_rate": float(merged["target"].mean()),
                "added_feature_count": int(len(numeric_cols)),
                "base_auc": float(base_auc),
                "full_auc": float(full_auc),
                "auc_delta": float(full_auc - base_auc),
            }
        )

    return pd.DataFrame(rows).sort_values("source").reset_index(drop=True)


def classify_status_review(row: pd.Series) -> Tuple[str, str, float]:
    if row["target"] == 1 and row["label"] == "alive":
        if row["avax_only"] == 0 and row["num_chains"] >= 3:
            subtype = "likely_multichain_relocation_or_avax_side_decay"
            confidence = 0.90
        elif row["avax_only"] == 1:
            subtype = "revived_native_protocol_or_threshold_mismatch"
            confidence = 0.75
        else:
            subtype = "revived_or_relocated_needs_manual_review"
            confidence = 0.70
        return "decayed_then_alive_now", subtype, confidence

    if row["target"] == 0 and row["label"] == "dead":
        if row["avax_only"] == 0 and row["num_chains"] >= 3:
            subtype = "late_avax_side_decline_or_chain_rotation"
            confidence = 0.80
        elif row["risk_score"] < 0.20:
            subtype = "late_decline_without_early_warning"
            confidence = 0.85
        else:
            subtype = "threshold_or_label_boundary_case"
            confidence = 0.60
        return "healthy_early_but_dead_now", subtype, confidence

    if row["label_status"] == "censored" and row["risk_score"] >= 0.75:
        subtype = "high_risk_censored_watchlist"
        return "insufficient_future_horizon", subtype, 0.70

    return "", "", 0.0


def build_status_review_table(scored_df: pd.DataFrame) -> pd.DataFrame:
    reviews = scored_df.copy()
    review_type = []
    review_subtype = []
    review_confidence = []

    for _, row in reviews.iterrows():
        r_type, r_subtype, r_confidence = classify_status_review(row)
        review_type.append(r_type)
        review_subtype.append(r_subtype)
        review_confidence.append(r_confidence)

    reviews["review_type"] = review_type
    reviews["review_subtype"] = review_subtype
    reviews["review_confidence"] = review_confidence
    reviews["review_priority"] = (
        reviews["review_confidence"].fillna(0.0) * 0.4
        + reviews["score_confidence"].fillna(0.0) * 0.3
        + reviews["risk_score"].fillna(0.0) * 0.3
    )
    reviews = reviews[reviews["review_type"] != ""].copy()
    reviews = reviews.sort_values("review_priority", ascending=False).reset_index(drop=True)
    return reviews


def main() -> None:
    repo_root, output_dir = resolve_paths()
    data_dir = repo_root / "avax_data"

    feature_frame, curve_map = lab.build_feature_frame(data_dir)
    feature_frame["data_quality_score"] = lab.build_data_quality_score(feature_frame)
    launch_dates = build_launch_dates(data_dir)
    feature_frame = feature_frame.merge(launch_dates, on="slug", how="left")

    train_df = feature_frame[feature_frame["target"].isin([0, 1])].copy()
    train_df["target"] = train_df["target"].astype(int)

    decay_proto, resilient_proto = lab.build_archetypes(train_df, curve_map)
    feature_frame = lab.append_archetype_scores(feature_frame, curve_map, decay_proto, resilient_proto)
    train_df = feature_frame[feature_frame["target"].isin([0, 1])].copy()
    train_df["target"] = train_df["target"].astype(int)

    feature_cols = [col for col in lab.get_feature_columns(train_df) if col not in {"launch_date", "launch_year"}]
    leaderboard, weights, _ = lab.evaluate_models(train_df, feature_cols)
    fitted_models = lab.fit_full_models(train_df, feature_cols)
    scored = lab.score_frame(fitted_models, weights, feature_frame[feature_frame["log_peak_tvl"].notna()].copy(), feature_cols)
    scored = lab.explain_protocols(scored, train_df)

    temporal = temporal_validation(train_df, feature_cols)
    enrichment = enrichment_coverage_audit(train_df, data_dir, feature_cols)
    status_review = build_status_review_table(scored)

    temporal.to_csv(output_dir / "temporal_validation.csv", index=False)
    enrichment.to_csv(output_dir / "enrichment_audit.csv", index=False)
    status_review.to_csv(output_dir / "status_review_candidates.csv", index=False)

    print("Product readiness validation completed.")
    print()
    print("Temporal validation:")
    print(temporal.to_string(index=False))
    print()
    print("Enrichment audit:")
    print(enrichment.to_string(index=False))
    print()
    print("Top status review candidates:")
    preview_cols = [
        "slug",
        "name",
        "review_type",
        "review_subtype",
        "review_priority",
        "label",
        "label_status",
        "risk_score",
        "score_confidence",
        "num_chains",
        "avax_only",
    ]
    print(status_review[preview_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
