from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EARLY_WINDOW_DAYS = 90
MIN_EARLY_POINTS = 30
PREDICT_HORIZON_DAYS = 365
SUSTAIN_DAYS = 30
ABSOLUTE_DECAY_TVL = 10_000.0
RELATIVE_DECAY_FROM_EARLY_PEAK = 0.05
RECOVERY_MULTIPLIER = 2.0
SAFE_TVL_FLOOR = 1_000.0
CURVE_POINTS = 30
RANDOM_STATE = 42


@dataclass
class ProtocolBuildResult:
    feature_row: Optional[Dict[str, float]]
    label_status: str
    target: Optional[int]
    curve_signature: Optional[np.ndarray]


def resolve_paths() -> Tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return repo_root, output_dir


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def find_activation_start(tvl: np.ndarray) -> int:
    if len(tvl) == 0:
        return 0
    for idx in range(max(1, len(tvl) - 2)):
        window = tvl[idx: idx + 3]
        if len(window) >= 2 and int((window > 0).sum()) >= 2:
            if float(window[0]) > 0.0:
                return idx
            positive_offsets = np.flatnonzero(window > 0)
            if len(positive_offsets) > 0:
                return idx + int(positive_offsets[0])
            return idx
    return 0


def trim_to_activation(df_tvl: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    df = df_tvl.sort_values("date").reset_index(drop=True).copy()
    tvl = df["tvl"].fillna(0.0).to_numpy(dtype=float)
    start_idx = find_activation_start(tvl)
    trimmed = df.iloc[start_idx:].reset_index(drop=True).copy()
    return trimmed, start_idx


def get_early_window(df_tvl: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df_tvl.empty:
        return None
    cutoff = df_tvl["date"].iloc[0] + pd.Timedelta(days=EARLY_WINDOW_DAYS)
    early = df_tvl[df_tvl["date"] <= cutoff].copy()
    if len(early) < MIN_EARLY_POINTS:
        return None
    return early


def determine_label(df_tvl: pd.DataFrame) -> Tuple[str, Optional[int]]:
    trimmed, _ = trim_to_activation(df_tvl)
    early = get_early_window(trimmed)
    if early is None:
        return "insufficient_early_window", None

    cutoff = trimmed["date"].iloc[0] + pd.Timedelta(days=EARLY_WINDOW_DAYS)
    future = trimmed[trimmed["date"] > cutoff].copy()
    if future.empty:
        return "no_future_window", None

    observed_horizon = int((future["date"].max() - cutoff).days)
    if observed_horizon < PREDICT_HORIZON_DAYS:
        return "censored", None

    horizon_end = cutoff + pd.Timedelta(days=PREDICT_HORIZON_DAYS)
    future = future[future["date"] <= horizon_end].reset_index(drop=True)

    early_peak = max(float(early["tvl"].max()), 1.0)
    decay_threshold = max(
        ABSOLUTE_DECAY_TVL,
        early_peak * RELATIVE_DECAY_FROM_EARLY_PEAK,
    )

    future_tvl = future["tvl"].fillna(0.0).to_numpy(dtype=float)
    for idx in range(0, len(future_tvl) - SUSTAIN_DAYS + 1):
        window = future_tvl[idx: idx + SUSTAIN_DAYS]
        if np.all(window < decay_threshold):
            after = future_tvl[idx + SUSTAIN_DAYS:]
            recovered = len(after) > 0 and np.quantile(after, 0.90) >= decay_threshold * RECOVERY_MULTIPLIER
            if not recovered:
                return "structural_decay", 1

    return "resilient_or_unproven", 0


def build_curve_signature(df_tvl: pd.DataFrame) -> Optional[np.ndarray]:
    trimmed, _ = trim_to_activation(df_tvl)
    early = get_early_window(trimmed)
    if early is None:
        return None

    x = (early["date"] - early["date"].iloc[0]).dt.days.to_numpy(dtype=float)
    y = np.log1p(early["tvl"].fillna(0.0).clip(lower=0.0).to_numpy(dtype=float))

    if len(x) < 2 or float(x.max()) == 0.0:
        return None

    x_target = np.linspace(0.0, float(x.max()), CURVE_POINTS)
    y_interp = np.interp(x_target, x, y)
    peak = float(y_interp.max())
    if peak > 0:
        y_interp = y_interp / peak
    return y_interp


def extract_robust_features(df_tvl: pd.DataFrame) -> Optional[Dict[str, float]]:
    trimmed, launch_trim_days = trim_to_activation(df_tvl)
    early = get_early_window(trimmed)
    if early is None:
        return None

    tvl = early["tvl"].fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    log_tvl = np.log1p(tvl)
    n_points = len(tvl)

    peak_idx = int(np.argmax(tvl))
    peak_tvl = max(float(tvl.max()), 1.0)
    end_tvl = max(float(tvl[-1]), 0.0)
    start_tvl = max(float(tvl[0]), 0.0)
    median_tvl = float(np.median(tvl))

    safe_tvl = np.maximum(tvl, SAFE_TVL_FLOOR)
    log_returns = np.diff(np.log1p(safe_tvl))
    post_peak_log = log_tvl[peak_idx:]
    rolling_7 = pd.Series(log_tvl).rolling(7, min_periods=1).mean().to_numpy(dtype=float)

    total_span_days = int((trimmed["date"].iloc[-1] - trimmed["date"].iloc[0]).days)
    early_span_days = int((early["date"].iloc[-1] - early["date"].iloc[0]).days)
    early_peak_day = int((early["date"].iloc[peak_idx] - early["date"].iloc[0]).days)

    slope_total = linregress(np.arange(n_points), log_tvl).slope if n_points >= 3 else 0.0
    slope_post = linregress(np.arange(len(post_peak_log)), post_peak_log).slope if len(post_peak_log) >= 3 else 0.0

    stable_band = np.mean(np.abs(log_tvl - np.median(log_tvl)) <= (np.std(log_tvl) + 1e-6))
    active_day_frac = float(np.mean(tvl >= ABSOLUTE_DECAY_TVL))
    nonzero_day_frac = float(np.mean(tvl > 0.0))

    return {
        "launch_trim_days": float(launch_trim_days),
        "lifecycle_days": float(total_span_days),
        "early_window_days_observed": float(early_span_days),
        "data_points": float(n_points),
        "log_start_tvl": float(np.log1p(start_tvl)),
        "log_peak_tvl": float(np.log1p(peak_tvl)),
        "log_end_tvl": float(np.log1p(end_tvl)),
        "log_median_tvl": float(np.log1p(median_tvl)),
        "retention_ratio": float((end_tvl + SAFE_TVL_FLOOR) / (peak_tvl + SAFE_TVL_FLOOR)),
        "drawdown_ratio": float((peak_tvl - end_tvl) / peak_tvl),
        "peak_day_frac": float(peak_idx / max(n_points - 1, 1)),
        "peak_day_ratio_by_span": float(early_peak_day / max(EARLY_WINDOW_DAYS, 1)),
        "auc_norm": float(log_tvl.mean() / max(log_tvl.max(), 1e-6)),
        "slope_total_log": float(slope_total),
        "slope_post_log": float(slope_post),
        "vol_log_ret": float(np.std(log_returns)) if len(log_returns) > 1 else 0.0,
        "mean_log_ret": float(np.mean(log_returns)) if len(log_returns) > 0 else 0.0,
        "downside_log_ret": float(np.std(np.minimum(log_returns, 0.0))) if len(log_returns) > 1 else 0.0,
        "crash_15_count": float((log_returns < np.log(0.85)).sum()) if len(log_returns) > 0 else 0.0,
        "rebound_15_count": float((log_returns > np.log(1.15)).sum()) if len(log_returns) > 0 else 0.0,
        "nonzero_day_frac": nonzero_day_frac,
        "active_day_frac": active_day_frac,
        "stable_band_frac": float(stable_band),
        "rolling7_end_vs_peak": float((rolling_7[-1] + 1e-6) / (rolling_7.max() + 1e-6)),
        "peak_to_median_log_gap": float(np.log1p(peak_tvl) - np.median(log_tvl)),
    }


def build_protocol_record(slug: str, df_tvl: pd.DataFrame) -> ProtocolBuildResult:
    feature_row = extract_robust_features(df_tvl)
    label_status, target = determine_label(df_tvl)
    curve_signature = build_curve_signature(df_tvl)
    if feature_row is not None:
        feature_row["slug"] = slug
    return ProtocolBuildResult(
        feature_row=feature_row,
        label_status=label_status,
        target=target,
        curve_signature=curve_signature,
    )


def build_feature_frame(data_dir: Path) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    meta = pd.read_csv(data_dir / "protocols_labeled.csv")
    meta = meta[
        [
            "slug",
            "name",
            "category",
            "label",
            "num_chains",
            "avax_only",
        ]
    ].copy()

    records: List[Dict[str, float]] = []
    curve_map: Dict[str, np.ndarray] = {}

    for csv_path in sorted(data_dir.glob("tvl_*.csv")):
        slug = csv_path.stem.replace("tvl_", "")
        df_tvl = pd.read_csv(csv_path, parse_dates=["date"])
        result = build_protocol_record(slug, df_tvl)
        row = {
            "slug": slug,
            "label_status": result.label_status,
            "target": result.target,
        }
        if result.feature_row is not None:
            row.update(result.feature_row)
        records.append(row)
        if result.curve_signature is not None:
            curve_map[slug] = result.curve_signature

    frame = pd.DataFrame(records).merge(meta, on="slug", how="left")
    frame["avax_only"] = frame["avax_only"].fillna(False).astype(int)
    frame["category"] = frame["category"].fillna("unknown")
    frame["category_code"] = frame["category"].astype("category").cat.codes.astype(float)
    frame["target_known"] = frame["target"].isin([0, 1]).astype(int)
    return frame, curve_map


def build_data_quality_score(frame: pd.DataFrame) -> pd.Series:
    coverage = (frame["early_window_days_observed"].fillna(0.0) / EARLY_WINDOW_DAYS).clip(0.0, 1.0)
    activation_penalty = 1.0 - (frame["launch_trim_days"].fillna(0.0) / 30.0).clip(0.0, 1.0)
    nonzero = frame["nonzero_day_frac"].fillna(0.0).clip(0.0, 1.0)
    active = frame["active_day_frac"].fillna(0.0).clip(0.0, 1.0)
    return (0.35 * coverage + 0.20 * activation_penalty + 0.25 * nonzero + 0.20 * active).clip(0.0, 1.0)


def build_archetypes(train_df: pd.DataFrame, curve_map: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    decay_curves = [curve_map[slug] for slug in train_df.loc[train_df["target"] == 1, "slug"] if slug in curve_map]
    resilient_curves = [curve_map[slug] for slug in train_df.loc[train_df["target"] == 0, "slug"] if slug in curve_map]
    decay_proto = np.mean(decay_curves, axis=0)
    resilient_proto = np.mean(resilient_curves, axis=0)
    return decay_proto, resilient_proto


def append_archetype_scores(
    frame: pd.DataFrame,
    curve_map: Dict[str, np.ndarray],
    decay_proto: np.ndarray,
    resilient_proto: np.ndarray,
) -> pd.DataFrame:
    distances_decay = []
    distances_resilient = []
    gaps = []

    for slug in frame["slug"]:
        curve = curve_map.get(slug)
        if curve is None:
            distances_decay.append(np.nan)
            distances_resilient.append(np.nan)
            gaps.append(np.nan)
            continue
        dist_decay = float(np.linalg.norm(curve - decay_proto))
        dist_resilient = float(np.linalg.norm(curve - resilient_proto))
        gap = dist_resilient - dist_decay
        distances_decay.append(dist_decay)
        distances_resilient.append(dist_resilient)
        gaps.append(gap)

    out = frame.copy()
    out["decay_archetype_distance"] = distances_decay
    out["resilient_archetype_distance"] = distances_resilient
    out["archetype_gap"] = gaps
    out["archetype_decay_signal"] = out["archetype_gap"].fillna(0.0).map(sigmoid)
    return out


def get_feature_columns(frame: pd.DataFrame) -> List[str]:
    excluded = {
        "slug",
        "name",
        "category",
        "label",
        "label_status",
        "target",
        "target_known",
    }
    return [col for col in frame.columns if col not in excluded]


def build_models() -> Dict[str, Pipeline]:
    return {
        "logreg": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "rf": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=600,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hgb": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_depth=4,
                        max_iter=350,
                        learning_rate=0.05,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_models(train_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, np.ndarray]]:
    X = train_df[feature_cols]
    y = train_df["target"].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    leaderboard_rows = []
    oof_predictions: Dict[str, np.ndarray] = {}
    models = build_models()

    for name, pipeline in models.items():
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=["roc_auc", "accuracy", "neg_brier_score"],
            n_jobs=None,
        )
        probs = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
        oof_predictions[name] = probs
        leaderboard_rows.append(
            {
                "model": name,
                "cv_auc_mean": float(scores["test_roc_auc"].mean()),
                "cv_auc_std": float(scores["test_roc_auc"].std()),
                "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
                "cv_accuracy_std": float(scores["test_accuracy"].std()),
                "cv_brier_mean": float((-scores["test_neg_brier_score"]).mean()),
                "oof_auc": float(roc_auc_score(y, probs)),
                "oof_accuracy": float(accuracy_score(y, (probs >= 0.5).astype(int))),
                "oof_brier": float(brier_score_loss(y, probs)),
            }
        )

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)
    weights = leaderboard.set_index("model")["cv_auc_mean"].clip(lower=0.0)
    weights = (weights / weights.sum()).to_dict()
    return leaderboard, weights, oof_predictions


def fit_full_models(train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Pipeline]:
    X = train_df[feature_cols]
    y = train_df["target"].astype(int)
    fitted = build_models()
    for pipeline in fitted.values():
        pipeline.fit(X, y)
    return fitted


def score_frame(
    fitted_models: Dict[str, Pipeline],
    weights: Dict[str, float],
    frame: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    scored = frame.copy()
    X = scored[feature_cols]

    model_probs = {}
    for name, pipeline in fitted_models.items():
        model_probs[name] = pipeline.predict_proba(X)[:, 1]
        scored[f"prob_{name}"] = model_probs[name]

    ordered_probs = np.column_stack([model_probs[name] for name in sorted(model_probs)])
    scored["risk_score"] = sum(model_probs[name] * weights[name] for name in weights)
    scored["risk_uncertainty"] = ordered_probs.std(axis=1)
    scored["score_confidence"] = (1.0 - scored["risk_uncertainty"]) * scored["data_quality_score"]
    scored["risk_band"] = pd.cut(
        scored["risk_score"],
        bins=[-np.inf, 0.45, 0.70, np.inf],
        labels=["stable_watch", "elevated", "critical"],
    ).astype(str)
    return scored


def create_reason_bank(train_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    decay = train_df[train_df["target"] == 1]
    resilient = train_df[train_df["target"] == 0]
    return {
        "retention_ratio": {"direction": -1.0, "decay_median": float(decay["retention_ratio"].median()), "resilient_median": float(resilient["retention_ratio"].median())},
        "peak_day_frac": {"direction": -1.0, "decay_median": float(decay["peak_day_frac"].median()), "resilient_median": float(resilient["peak_day_frac"].median())},
        "drawdown_ratio": {"direction": 1.0, "decay_median": float(decay["drawdown_ratio"].median()), "resilient_median": float(resilient["drawdown_ratio"].median())},
        "vol_log_ret": {"direction": 1.0, "decay_median": float(decay["vol_log_ret"].median()), "resilient_median": float(resilient["vol_log_ret"].median())},
        "active_day_frac": {"direction": -1.0, "decay_median": float(decay["active_day_frac"].median()), "resilient_median": float(resilient["active_day_frac"].median())},
        "slope_post_log": {"direction": -1.0, "decay_median": float(decay["slope_post_log"].median()), "resilient_median": float(resilient["slope_post_log"].median())},
    }


def explain_protocols(scored: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    reason_bank = create_reason_bank(train_df)
    readable = {
        "retention_ratio": "weak 90d retention",
        "peak_day_frac": "peak arrived too early",
        "drawdown_ratio": "early drawdown stayed high",
        "vol_log_ret": "log-return volatility stayed elevated",
        "active_day_frac": "active-day density stayed low",
        "slope_post_log": "post-peak slope remained negative",
    }

    out = scored.copy()
    reasons_1: List[str] = []
    reasons_2: List[str] = []
    reasons_3: List[str] = []

    for _, row in out.iterrows():
        scored_reasons: List[Tuple[float, str]] = []
        for feature, config in reason_bank.items():
            value = float(row.get(feature, np.nan))
            if np.isnan(value):
                continue
            decay_gap = config["direction"] * (value - config["decay_median"])
            resilient_gap = config["direction"] * (value - config["resilient_median"])
            signal = resilient_gap - decay_gap
            scored_reasons.append((signal, readable[feature]))
        scored_reasons.sort(reverse=True)
        labels = [text for _, text in scored_reasons[:3]]
        while len(labels) < 3:
            labels.append("")
        reasons_1.append(labels[0])
        reasons_2.append(labels[1])
        reasons_3.append(labels[2])

    out["reason_1"] = reasons_1
    out["reason_2"] = reasons_2
    out["reason_3"] = reasons_3
    return out


def export_feature_importance(
    fitted_models: Dict[str, Pipeline],
    feature_cols: List[str],
    output_dir: Path,
) -> None:
    rf_model = fitted_models["rf"].named_steps["model"]
    rf_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "rf_importance": rf_model.feature_importances_,
        }
    ).sort_values("rf_importance", ascending=False)
    rf_importance.to_csv(output_dir / "feature_importance.csv", index=False)


def export_summary(
    frame: pd.DataFrame,
    train_df: pd.DataFrame,
    leaderboard: pd.DataFrame,
    weights: Dict[str, float],
    output_dir: Path,
) -> None:
    summary = {
        "protocols_total": int(len(frame)),
        "protocols_scoreable": int(frame["log_peak_tvl"].notna().sum()),
        "trainable_protocols": int(len(train_df)),
        "target_event_rate": float(train_df["target"].mean()),
        "label_status_counts": frame["label_status"].value_counts(dropna=False).to_dict(),
        "ensemble_weights": weights,
        "best_cv_auc_model": str(leaderboard.iloc[0]["model"]),
        "best_cv_auc": float(leaderboard.iloc[0]["cv_auc_mean"]),
    }
    with open(output_dir / "experiment_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def main() -> None:
    repo_root, output_dir = resolve_paths()
    data_dir = repo_root / "avax_data"

    feature_frame, curve_map = build_feature_frame(data_dir)
    feature_frame["data_quality_score"] = build_data_quality_score(feature_frame)

    train_df = feature_frame[feature_frame["target"].isin([0, 1])].copy()
    train_df["target"] = train_df["target"].astype(int)

    decay_proto, resilient_proto = build_archetypes(train_df, curve_map)
    feature_frame = append_archetype_scores(feature_frame, curve_map, decay_proto, resilient_proto)
    train_df = feature_frame[feature_frame["target"].isin([0, 1])].copy()
    train_df["target"] = train_df["target"].astype(int)

    feature_cols = get_feature_columns(train_df)
    leaderboard, weights, _ = evaluate_models(train_df, feature_cols)
    fitted_models = fit_full_models(train_df, feature_cols)
    scored = score_frame(fitted_models, weights, feature_frame[feature_frame["log_peak_tvl"].notna()].copy(), feature_cols)
    explained = explain_protocols(scored, train_df)

    leaderboard.to_csv(output_dir / "model_leaderboard.csv", index=False)
    export_feature_importance(fitted_models, feature_cols, output_dir)
    export_summary(feature_frame, train_df, leaderboard, weights, output_dir)

    feature_frame.to_csv(output_dir / "experiment_dataset.csv", index=False)
    explained.sort_values("risk_score", ascending=False).to_csv(output_dir / "protocol_scores.csv", index=False)

    preview_cols = [
        "slug",
        "name",
        "label_status",
        "target",
        "risk_score",
        "risk_band",
        "risk_uncertainty",
        "score_confidence",
        "data_quality_score",
        "archetype_decay_signal",
        "reason_1",
        "reason_2",
        "reason_3",
    ]
    preview = explained.sort_values("risk_score", ascending=False)[preview_cols].head(15)
    print("Protocol Decay Lab completed.")
    print()
    print("Top 15 risk candidates:")
    print(preview.to_string(index=False))
    print()
    print("Leaderboard:")
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
