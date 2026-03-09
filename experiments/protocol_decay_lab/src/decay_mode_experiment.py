from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

import protocol_decay_lab as lab


PROTOCOL_URL = "https://api.llama.fi/protocol/{slug}"
TIMEOUT = 25
MAX_WORKERS = 12
AGGREGATE_KEYS = {"staking", "pool2", "borrowed"}


def resolve_paths() -> Tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return repo_root, output_dir


def fetch_protocol_meta(slug: str) -> Dict[str, object]:
    response = requests.get(PROTOCOL_URL.format(slug=slug), timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    return {
        "slug": slug,
        "parent_protocol": payload.get("parentProtocol"),
        "gecko_id": payload.get("gecko_id"),
        "chains_json": json.dumps(payload.get("chains", []), ensure_ascii=False),
        "previous_names_json": json.dumps(payload.get("previousNames", []), ensure_ascii=False),
        "current_chain_tvls_json": json.dumps(payload.get("currentChainTvls", {}), ensure_ascii=False),
    }


def load_or_fetch_registry(protocols: pd.DataFrame, output_dir: Path, refresh: bool = False) -> pd.DataFrame:
    registry_path = output_dir / "protocol_registry_live.csv"
    if registry_path.exists() and not refresh:
        return pd.read_csv(registry_path)

    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(fetch_protocol_meta, slug): slug
            for slug in protocols["slug"].drop_duplicates().tolist()
        }
        for future in as_completed(future_map):
            slug = future_map[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append({"slug": slug, "error": str(exc)})

    registry = pd.DataFrame(rows)
    registry.to_csv(registry_path, index=False)
    pd.DataFrame(failures).to_csv(output_dir / "protocol_registry_live_failures.csv", index=False)
    return registry


def parse_current_chain_metrics(chain_tvls_json: str) -> Dict[str, object]:
    try:
        chain_tvls = json.loads(chain_tvls_json) if isinstance(chain_tvls_json, str) else {}
    except Exception:
        chain_tvls = {}

    core = {}
    for key, value in chain_tvls.items():
        if key in AGGREGATE_KEYS:
            continue
        if "-" in key:
            continue
        try:
            numeric = float(value)
        except Exception:
            numeric = 0.0
        core[key] = numeric

    total_core = float(sum(core.values()))
    avax_core = float(core.get("Avalanche", 0.0))
    non_avax_core = max(total_core - avax_core, 0.0)
    active_chain_count = int(sum(v >= 10_000.0 for v in core.values()))
    dominant_chain = max(core, key=core.get) if core else ""
    dominant_share = float(core[dominant_chain] / total_core) if core and total_core > 0 else 0.0
    avax_share = float(avax_core / total_core) if total_core > 0 else 0.0

    return {
        "current_total_core_tvl": total_core,
        "current_avax_core_tvl": avax_core,
        "current_non_avax_core_tvl": non_avax_core,
        "current_avax_share": avax_share,
        "dominant_chain": dominant_chain,
        "dominant_chain_share": dominant_share,
        "active_chain_count_core": active_chain_count,
    }


def classify_decay_mode(row: pd.Series) -> Tuple[str, float]:
    if row.get("target") != 1:
        return "resilient_or_unproven", 0.0

    total_core = float(row.get("current_total_core_tvl", 0.0) or 0.0)
    avax_core = float(row.get("current_avax_core_tvl", 0.0) or 0.0)
    avax_share = float(row.get("current_avax_share", 0.0) or 0.0)
    dominant_share = float(row.get("dominant_chain_share", 0.0) or 0.0)
    active_chains = int(row.get("active_chain_count_core", 0) or 0)
    risk_score = float(row.get("risk_score", 0.0) or 0.0)
    score_confidence = float(row.get("score_confidence", 0.0) or 0.0)
    label = row.get("label")
    avax_only = int(row.get("avax_only", 0) or 0)

    if label == "dead":
        if total_core >= 100_000 and avax_share <= 0.05 and dominant_share >= 0.50 and active_chains >= 2:
            return "avax_side_decay_with_offchain_survival", 0.92
        if total_core < 100_000:
            return "terminal_global_decay", 0.90
        return "low_tvl_terminal_or_unclear", 0.65

    if label == "alive":
        if avax_only == 0 and total_core >= 100_000 and avax_share <= 0.20 and dominant_share >= 0.45 and active_chains >= 2:
            return "multichain_relocation", 0.92
        if avax_only == 1 and avax_core >= 10_000:
            return "native_revival_or_threshold_boundary", 0.82
        if total_core >= 50_000 and avax_share <= 0.50 and active_chains >= 2:
            return "avax_side_decay_but_globally_alive", 0.78
        if risk_score >= 0.85 and score_confidence >= 0.70:
            return "persistent_decay_but_not_terminal", 0.70
        return "unclear_positive_case", 0.55

    return "unclassified", 0.30


def build_base_scored_frame(data_dir: Path) -> pd.DataFrame:
    feature_frame, curve_map = lab.build_feature_frame(data_dir)
    feature_frame["data_quality_score"] = lab.build_data_quality_score(feature_frame)

    train_df = feature_frame[feature_frame["target"].isin([0, 1])].copy()
    train_df["target"] = train_df["target"].astype(int)
    decay_proto, resilient_proto = lab.build_archetypes(train_df, curve_map)
    feature_frame = lab.append_archetype_scores(feature_frame, curve_map, decay_proto, resilient_proto)

    train_df = feature_frame[feature_frame["target"].isin([0, 1])].copy()
    train_df["target"] = train_df["target"].astype(int)
    feature_cols = lab.get_feature_columns(train_df)
    leaderboard, weights, oof_predictions = lab.evaluate_models(train_df, feature_cols)
    fitted = lab.fit_full_models(train_df, feature_cols)
    scored = lab.score_frame(fitted, weights, feature_frame[feature_frame["log_peak_tvl"].notna()].copy(), feature_cols)
    scored = lab.explain_protocols(scored, train_df)
    train_indexed = train_df[["slug"]].copy().reset_index(drop=True)
    ensemble_oof = sum(oof_predictions[name] * weights[name] for name in weights)
    train_indexed["oof_risk_score"] = ensemble_oof
    scored = scored.merge(train_indexed, on="slug", how="left")
    return scored


def evaluate_subset(scored: pd.DataFrame, mode_exclusions: set) -> Dict[str, float]:
    subset = scored[(scored["target"] == 0) | (~scored["decay_mode"].isin(mode_exclusions))].copy()
    subset = subset[subset["target"].isin([0, 1])].copy()
    y = subset["target"].astype(int)
    p = subset["oof_risk_score"].astype(float)
    pred = (p >= 0.5).astype(int)
    return {
        "sample_size": int(len(subset)),
        "event_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, p)),
    }


def main() -> None:
    repo_root, output_dir = resolve_paths()
    data_dir = repo_root / "avax_data"
    protocols = pd.read_csv(data_dir / "protocols_labeled.csv")[["slug"]].drop_duplicates()

    scored = build_base_scored_frame(data_dir)
    registry = load_or_fetch_registry(protocols, output_dir, refresh=False)

    chain_metrics = registry["current_chain_tvls_json"].apply(parse_current_chain_metrics).apply(pd.Series)
    registry_enriched = pd.concat([registry, chain_metrics], axis=1)

    merged = scored.merge(registry_enriched, on="slug", how="left")
    modes = merged.apply(classify_decay_mode, axis=1, result_type="expand")
    merged["decay_mode"] = modes[0]
    merged["decay_mode_confidence"] = modes[1]

    merged.to_csv(output_dir / "decay_mode_registry.csv", index=False)
    mode_counts = merged["decay_mode"].value_counts(dropna=False).rename_axis("decay_mode").reset_index(name="protocol_count")
    mode_counts.to_csv(output_dir / "decay_mode_counts.csv", index=False)

    candidates = merged[
        merged["decay_mode"].isin(
            {
                "multichain_relocation",
                "avax_side_decay_with_offchain_survival",
                "avax_side_decay_but_globally_alive",
                "native_revival_or_threshold_boundary",
            }
        )
    ].copy()
    candidates = candidates.sort_values(
        ["decay_mode_confidence", "score_confidence", "risk_score"],
        ascending=False,
    )
    candidates.to_csv(output_dir / "decay_mode_candidates.csv", index=False)

    original_metrics = evaluate_subset(merged, set())
    refined_metrics = evaluate_subset(
        merged,
        {
            "multichain_relocation",
            "avax_side_decay_with_offchain_survival",
            "native_revival_or_threshold_boundary",
        },
    )
    summary = pd.DataFrame(
        [
            {"view": "original_structural_decay_target", **original_metrics},
            {"view": "refined_terminal_decay_view", **refined_metrics},
        ]
    )
    summary.to_csv(output_dir / "decay_mode_summary.csv", index=False)

    print("Decay mode experiment completed.")
    print()
    print("Mode counts:")
    print(mode_counts.to_string(index=False))
    print()
    print("Target view summary:")
    print(summary.to_string(index=False))
    print()
    print("Top mode candidates:")
    preview_cols = [
        "slug",
        "name",
        "decay_mode",
        "decay_mode_confidence",
        "label",
        "label_status",
        "risk_score",
        "score_confidence",
        "current_avax_core_tvl",
        "current_total_core_tvl",
        "current_avax_share",
        "dominant_chain",
    ]
    print(candidates[preview_cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
