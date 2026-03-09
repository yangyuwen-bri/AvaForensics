from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

import protocol_decay_lab as lab


PROTOCOL_URL = "https://api.llama.fi/protocol/{slug}"
TIMEOUT = 8
MAX_WORKERS = 6
FETCH_MISSING_ON_REBUILD = False
CURRENT_TVL_THRESHOLD = 10_000.0
MIGRATION_TOTAL_THRESHOLD = 100_000.0
MIGRATION_AVAX_SHARE_THRESHOLD = 0.20
AUX_SUFFIXES = {"staking", "pool2", "borrowed"}


def resolve_paths() -> Tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "dataset_v2"
    payload_dir = output_dir / "payloads"
    obs_dir = output_dir / "observations"
    payload_dir.mkdir(parents=True, exist_ok=True)
    obs_dir.mkdir(parents=True, exist_ok=True)
    return repo_root, output_dir


def normalize_key(key: str) -> Tuple[str, str]:
    parts = str(key).split("-", 1)
    base = parts[0].strip()
    suffix = parts[1].strip().lower() if len(parts) > 1 else ""
    return base, suffix


def fetch_payload(slug: str, payload_dir: Path, refresh: bool = False) -> Dict[str, object]:
    cache_path = payload_dir / f"{slug}.json"
    if cache_path.exists() and not refresh:
        with open(cache_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    response = requests.get(PROTOCOL_URL.format(slug=slug), timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    return payload


def load_cached_payloads(payload_dir: Path) -> Dict[str, Dict[str, object]]:
    cached: Dict[str, Dict[str, object]] = {}
    for path in sorted(payload_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            cached[path.stem] = json.load(handle)
    return cached


def fetch_all_payloads(slugs: List[str], payload_dir: Path, refresh: bool = False) -> Dict[str, Dict[str, object]]:
    results = load_cached_payloads(payload_dir)
    failures: List[Dict[str, str]] = []

    if not FETCH_MISSING_ON_REBUILD and not refresh:
        missing = [slug for slug in slugs if slug not in results]
        failures.extend([{"slug": slug, "error": "not_fetched_in_cache_first_mode"} for slug in missing])
        return results, failures

    targets = [slug for slug in slugs if refresh or slug not in results]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(fetch_payload, slug, payload_dir, refresh): slug
            for slug in targets
        }
        for future in as_completed(future_map):
            slug = future_map[future]
            try:
                results[slug] = future.result()
            except Exception as exc:
                failures.append({"slug": slug, "error": str(exc)})
    return results, failures


def series_to_daily(series: List[Dict[str, object]]) -> pd.Series:
    if not series:
        return pd.Series(dtype=float)
    df = pd.DataFrame(series)
    if "date" not in df.columns or "totalLiquidityUSD" not in df.columns:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"], unit="s").dt.normalize()
    df["tvl"] = pd.to_numeric(df["totalLiquidityUSD"], errors="coerce").fillna(0.0)
    return df.groupby("date")["tvl"].last().sort_index()


def parse_chain_histories(payload: Dict[str, object]) -> Dict[str, pd.Series]:
    chain_tvls = payload.get("chainTvls", {}) or {}
    parsed: Dict[str, pd.Series] = {}
    for key, value in chain_tvls.items():
        if not isinstance(value, dict) or "tvl" not in value:
            continue
        parsed[str(key)] = series_to_daily(value.get("tvl", []))
    return parsed


def parse_current_chain_core(payload: Dict[str, object]) -> Dict[str, float]:
    current = payload.get("currentChainTvls", {}) or {}
    core = {}
    for key, value in current.items():
        base, suffix = normalize_key(key)
        if suffix or str(key).lower() in AUX_SUFFIXES:
            continue
        try:
            core[base] = float(value)
        except Exception:
            core[base] = 0.0
    return core


def build_observation_frame(payload: Dict[str, object]) -> pd.DataFrame:
    parsed = parse_chain_histories(payload)
    total_series = series_to_daily(payload.get("tvl", []) or [])
    avax_core = parsed.get("Avalanche", pd.Series(dtype=float))
    avax_staking = parsed.get("Avalanche-staking", pd.Series(dtype=float))
    avax_pool2 = parsed.get("Avalanche-pool2", pd.Series(dtype=float))
    avax_borrowed = parsed.get("Avalanche-borrowed", pd.Series(dtype=float))

    avax_aux_other = pd.Series(dtype=float)
    for key, series in parsed.items():
        base, suffix = normalize_key(key)
        if base != "Avalanche" or suffix in {"", "staking", "pool2", "borrowed"}:
            continue
        avax_aux_other = avax_aux_other.add(series, fill_value=0.0)

    index = total_series.index
    for series in [avax_core, avax_staking, avax_pool2, avax_borrowed, avax_aux_other]:
        index = index.union(series.index)

    obs = pd.DataFrame(index=index.sort_values())
    obs.index.name = "date"
    obs["avax_core_tvl"] = avax_core.reindex(obs.index)
    obs["total_tvl"] = total_series.reindex(obs.index)
    obs["avax_staking_tvl"] = avax_staking.reindex(obs.index)
    obs["avax_pool2_tvl"] = avax_pool2.reindex(obs.index)
    obs["avax_borrowed_tvl"] = avax_borrowed.reindex(obs.index)
    obs["avax_aux_other_tvl"] = avax_aux_other.reindex(obs.index)
    obs["has_avax_core_history"] = obs["avax_core_tvl"].notna().astype(int)
    obs["has_total_history"] = obs["total_tvl"].notna().astype(int)
    obs["non_avax_tvl_est"] = obs["total_tvl"] - obs["avax_core_tvl"]
    obs.loc[obs["non_avax_tvl_est"] < 0, "non_avax_tvl_est"] = 0.0
    obs["avax_share_est"] = obs["avax_core_tvl"] / obs["total_tvl"]
    obs.replace([np.inf, -np.inf], np.nan, inplace=True)
    obs = obs.reset_index()
    return obs


def lifecycle_summary(obs: pd.DataFrame) -> Dict[str, float]:
    usable = obs[obs["avax_core_tvl"].notna()].copy()
    if usable.empty:
        return {}
    usable = usable.sort_values("date").reset_index(drop=True)
    tvl = usable["avax_core_tvl"].clip(lower=0.0).to_numpy(dtype=float)
    peak = float(tvl.max()) if len(tvl) else 0.0
    current = float(tvl[-1]) if len(tvl) else 0.0
    start = float(tvl[0]) if len(tvl) else 0.0
    drawdown = (peak - current) / peak if peak > 0 else 0.0
    lifespan = int((usable["date"].iloc[-1] - usable["date"].iloc[0]).days) if len(usable) > 1 else 0

    recent30 = usable[usable["date"] >= usable["date"].max() - pd.Timedelta(days=30)]
    recent90 = usable[usable["date"] >= usable["date"].max() - pd.Timedelta(days=90)]
    tvl_30d_change = (recent30["avax_core_tvl"].iloc[-1] - recent30["avax_core_tvl"].iloc[0]) / (recent30["avax_core_tvl"].iloc[0] + 1.0) if len(recent30) >= 2 else 0.0
    tvl_90d_change = (recent90["avax_core_tvl"].iloc[-1] - recent90["avax_core_tvl"].iloc[0]) / (recent90["avax_core_tvl"].iloc[0] + 1.0) if len(recent90) >= 2 else 0.0

    peak_idx = int(np.argmax(tvl)) if len(tvl) else 0
    post_peak = tvl[peak_idx:]
    consecutive_decline = int(sum(1 for idx in range(1, len(post_peak)) if post_peak[idx] < post_peak[idx - 1]))
    mean_tvl = float(tvl.mean()) if len(tvl) else 0.0
    volatility = float(tvl.std() / mean_tvl) if mean_tvl > 0 else 0.0

    return {
        "avax_peak_tvl": peak,
        "avax_current_tvl": current,
        "avax_start_tvl": start,
        "avax_drawdown_from_peak": drawdown,
        "avax_tvl_30d_change": tvl_30d_change,
        "avax_tvl_90d_change": tvl_90d_change,
        "avax_lifespan_days": lifespan,
        "avax_consecutive_decline_days": consecutive_decline,
        "avax_volatility_full": volatility,
        "avax_history_days": int(len(usable)),
    }


def compute_quality_flags(obs: pd.DataFrame, current_core: Dict[str, float]) -> Dict[str, object]:
    has_avax_history = bool(obs["has_avax_core_history"].any())
    has_total_history = bool(obs["has_total_history"].any())
    avax_days = int(obs["has_avax_core_history"].sum())
    total_days = int(obs["has_total_history"].sum())
    source_mode = "avax_core"
    if not has_avax_history and has_total_history:
        source_mode = "total_only"
    elif not has_avax_history and not has_total_history:
        source_mode = "missing"

    current_total_core = float(sum(current_core.values()))
    current_avax_core = float(current_core.get("Avalanche", 0.0))
    current_avax_share = float(current_avax_core / current_total_core) if current_total_core > 0 else 0.0
    active_chain_count_core = int(sum(value >= CURRENT_TVL_THRESHOLD for value in current_core.values()))

    quality = (
        0.55 * float(has_avax_history)
        + 0.20 * min(avax_days / 365.0, 1.0)
        + 0.15 * float(current_avax_core >= 0.0)
        + 0.10 * min(total_days / 365.0, 1.0)
    )

    return {
        "history_source_mode": source_mode,
        "has_avax_history": int(has_avax_history),
        "has_total_history": int(has_total_history),
        "avax_history_days": avax_days,
        "total_history_days": total_days,
        "current_avax_core_tvl": current_avax_core,
        "current_total_core_tvl": current_total_core,
        "current_non_avax_core_tvl": max(current_total_core - current_avax_core, 0.0),
        "current_avax_share": current_avax_share,
        "active_chain_count_core": active_chain_count_core,
        "data_quality_score_v2": round(min(quality, 1.0), 6),
    }


def classify_current_status(flags: Dict[str, object], avax_only: int) -> Dict[str, object]:
    source_mode = flags.get("history_source_mode")
    avax_tvl = flags.get("current_avax_core_tvl", 0.0)
    total_core = flags.get("current_total_core_tvl", 0.0)
    avax_share = flags.get("current_avax_share", 0.0)
    active_chains = flags.get("active_chain_count_core", 0)
    avax_tvl = 0.0 if pd.isna(avax_tvl) else float(avax_tvl)
    total_core = 0.0 if pd.isna(total_core) else float(total_core)
    avax_share = 0.0 if pd.isna(avax_share) else float(avax_share)
    active_chains = 0 if pd.isna(active_chains) else int(active_chains)

    if source_mode == "fetch_failed":
        current_status = "data_fetch_failed"
        current_tvl_band_avax = "unknown"
        low_tvl_now_avax = None
    elif source_mode == "total_only":
        current_status = "no_avax_core_history"
        current_tvl_band_avax = "unknown"
        low_tvl_now_avax = None
    elif avax_tvl >= CURRENT_TVL_THRESHOLD:
        current_status = "active_on_avax"
        current_tvl_band_avax = "ge_10k"
        low_tvl_now_avax = 0
    else:
        current_status = "low_tvl_on_avax"
        current_tvl_band_avax = "lt_10k"
        low_tvl_now_avax = 1

    migrated_candidate = int(
        source_mode == "avax_core"
        and
        avax_only == 0
        and avax_tvl < CURRENT_TVL_THRESHOLD
        and total_core >= MIGRATION_TOTAL_THRESHOLD
        and avax_share <= MIGRATION_AVAX_SHARE_THRESHOLD
        and active_chains >= 2
    )

    return {
        "current_status": current_status,
        "current_tvl_band_avax": current_tvl_band_avax,
        "low_tvl_now_avax": low_tvl_now_avax,
        "migrated_candidate": migrated_candidate,
    }


def classify_core_universe(flags: Dict[str, object]) -> Dict[str, object]:
    source_mode = flags.get("history_source_mode")
    if source_mode == "avax_core":
        return {
            "core_eligible": 1,
            "core_universe_status": "core_eligible",
            "boundary_reason": "",
        }
    if source_mode == "total_only":
        return {
            "core_eligible": 0,
            "core_universe_status": "boundary_excluded",
            "boundary_reason": "no_avax_core_history_total_only",
        }
    return {
        "core_eligible": 0,
        "core_universe_status": "data_incomplete",
        "boundary_reason": "protocol_fetch_failed",
    }


def main() -> None:
    repo_root, output_dir = resolve_paths()
    payload_dir = output_dir / "payloads"
    obs_dir = output_dir / "observations"

    old_registry = pd.read_csv(repo_root / "avax_data" / "protocols_labeled.csv")
    slugs = old_registry["slug"].drop_duplicates().tolist()
    payloads, failures = fetch_all_payloads(slugs, payload_dir, refresh=False)

    registry_rows: List[Dict[str, object]] = []
    label_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    early_rows: List[Dict[str, object]] = []
    observation_index_rows: List[Dict[str, object]] = []
    early_exclusion_rows: List[Dict[str, object]] = []
    failure_df = pd.DataFrame(failures)
    if not failure_df.empty:
        failure_df.to_csv(output_dir / "fetch_failures.csv", index=False)

    for _, meta in old_registry.iterrows():
        slug = meta["slug"]
        payload = payloads.get(slug)
        if not payload:
            flags = {
                "history_source_mode": "fetch_failed",
                "has_avax_history": 0,
                "has_total_history": 0,
                "avax_history_days": 0,
                "total_history_days": 0,
                "current_avax_core_tvl": np.nan,
                "current_total_core_tvl": np.nan,
                "current_non_avax_core_tvl": np.nan,
                "current_avax_share": np.nan,
                "active_chain_count_core": np.nan,
                "data_quality_score_v2": 0.0,
            }
            status = classify_current_status(flags, int(bool(meta["avax_only"])))
            universe = classify_core_universe(flags)
            registry_rows.append(
                {
                    "slug": slug,
                    "name": meta["name"],
                    "category": meta["category"],
                    "parent_protocol": None,
                    "gecko_id": None,
                    "chains_count_payload": np.nan,
                    "avax_only": int(bool(meta["avax_only"])),
                    "num_chains": meta["num_chains"],
                    "old_label": meta["label"],
                    "history_source_mode": "fetch_failed",
                    "has_avax_history": 0,
                    "has_total_history": 0,
                    "avax_history_days": 0,
                    "total_history_days": 0,
                    "current_avax_core_tvl": np.nan,
                    "current_total_core_tvl": np.nan,
                    "current_non_avax_core_tvl": np.nan,
                    "current_avax_share": np.nan,
                    "active_chain_count_core": np.nan,
                    "data_quality_score_v2": 0.0,
                    **universe,
                }
            )
            label_rows.append(
                {
                    "slug": slug,
                    "name": meta["name"],
                    "category": meta["category"],
                    **flags,
                    **status,
                    **universe,
                    "label_confidence_v2": 0.0,
                    "structural_decay_label_v2": "data_fetch_failed",
                    "structural_decay_target_v2": None,
                }
            )
            summary_rows.append(
                {
                    "slug": slug,
                    "name": meta["name"],
                    "category": meta["category"],
                    "avax_peak_tvl": np.nan,
                    "avax_current_tvl": np.nan,
                    "avax_start_tvl": np.nan,
                    "avax_drawdown_from_peak": np.nan,
                    "avax_tvl_30d_change": np.nan,
                    "avax_tvl_90d_change": np.nan,
                    "avax_lifespan_days": np.nan,
                    "avax_consecutive_decline_days": np.nan,
                    "avax_volatility_full": np.nan,
                    "avax_history_days": 0,
                    **flags,
                    **status,
                    **universe,
                }
            )
            observation_index_rows.append(
                {
                    "slug": slug,
                    "observation_file": "",
                    "row_count": 0,
                    "avax_history_rows": 0,
                    "total_history_rows": 0,
                    "history_source_mode": "fetch_failed",
                    **universe,
                }
            )
            continue

        obs = build_observation_frame(payload)
        obs["slug"] = slug
        obs.to_csv(obs_dir / f"{slug}.csv", index=False)

        current_core = parse_current_chain_core(payload)
        flags = compute_quality_flags(obs, current_core)
        status = classify_current_status(flags, int(bool(meta["avax_only"])))
        universe = classify_core_universe(flags)
        summary = lifecycle_summary(obs)

        registry_row = {
            "slug": slug,
            "name": payload.get("name") or meta["name"],
            "category": payload.get("category") or meta["category"],
            "parent_protocol": payload.get("parentProtocol"),
            "gecko_id": payload.get("gecko_id"),
            "chains_count_payload": len(payload.get("chains", []) or []),
            "avax_only": int(bool(meta["avax_only"])),
            "num_chains": meta["num_chains"],
            "old_label": meta["label"],
            **flags,
            **universe,
        }
        registry_rows.append(registry_row)

        label_row = {
            "slug": slug,
            "name": registry_row["name"],
            "category": registry_row["category"],
            **flags,
            **status,
            **universe,
            "label_confidence_v2": flags["data_quality_score_v2"],
        }

        avax_obs = obs[obs["avax_core_tvl"].notna()][["date", "avax_core_tvl"]].rename(columns={"avax_core_tvl": "tvl"})
        if not avax_obs.empty:
            label_status, target = lab.determine_label(avax_obs.copy())
            label_row["structural_decay_label_v2"] = label_status
            label_row["structural_decay_target_v2"] = target

            early_feats = lab.extract_robust_features(avax_obs.copy())
            if early_feats is not None:
                early_feats.update(
                    {
                        "slug": slug,
                        "name": registry_row["name"],
                        "category": registry_row["category"],
                        "current_status": status["current_status"],
                        "structural_decay_label_v2": label_status,
                        "structural_decay_target_v2": target,
                        "history_source_mode": flags["history_source_mode"],
                        "data_quality_score_v2": flags["data_quality_score_v2"],
                        "core_eligible": universe["core_eligible"],
                        "core_universe_status": universe["core_universe_status"],
                    }
                )
                early_rows.append(early_feats)
            else:
                early_exclusion_rows.append(
                    {
                        "slug": slug,
                        "name": registry_row["name"],
                        "category": registry_row["category"],
                        "history_source_mode": flags["history_source_mode"],
                        "avax_history_days": flags["avax_history_days"],
                        "current_status": status["current_status"],
                        "structural_decay_label_v2": label_status,
                        "exclusion_reason": "extract_robust_features_returned_none",
                    }
                )

        else:
            label_row["structural_decay_label_v2"] = "missing_avax_history"
            label_row["structural_decay_target_v2"] = None
            early_exclusion_rows.append(
                {
                    "slug": slug,
                    "name": registry_row["name"],
                    "category": registry_row["category"],
                    "history_source_mode": flags["history_source_mode"],
                    "avax_history_days": flags["avax_history_days"],
                    "current_status": status["current_status"],
                    "structural_decay_label_v2": "missing_avax_history",
                    "exclusion_reason": "no_avax_observation_rows",
                }
            )

        label_rows.append(label_row)

        summary_row = {
            "slug": slug,
            "name": registry_row["name"],
            "category": registry_row["category"],
            **summary,
            **flags,
            **status,
            **universe,
        }
        summary_rows.append(summary_row)

        observation_index_rows.append(
            {
                "slug": slug,
                "observation_file": str((obs_dir / f"{slug}.csv").relative_to(output_dir)),
                "row_count": int(len(obs)),
                "avax_history_rows": int(obs["has_avax_core_history"].sum()),
                "total_history_rows": int(obs["has_total_history"].sum()),
                "history_source_mode": flags["history_source_mode"],
                **universe,
            }
        )

    registry_v2 = pd.DataFrame(registry_rows).sort_values("slug")
    labels_v2 = pd.DataFrame(label_rows).sort_values("slug")
    summary_v2 = pd.DataFrame(summary_rows).sort_values("slug")
    early_v2 = pd.DataFrame(early_rows).sort_values("slug")
    observation_index = pd.DataFrame(observation_index_rows).sort_values("slug")
    early_exclusions = pd.DataFrame(early_exclusion_rows).sort_values("slug")

    registry_v2.to_csv(output_dir / "registry_v2.csv", index=False)
    labels_v2.to_csv(output_dir / "labels_v2.csv", index=False)
    summary_v2.to_csv(output_dir / "features_summary_v2.csv", index=False)
    early_v2.to_csv(output_dir / "early_features_v2.csv", index=False)
    observation_index.to_csv(output_dir / "observation_index_v2.csv", index=False)
    early_exclusions.to_csv(output_dir / "early_feature_exclusions_v2.csv", index=False)

    quarantine = labels_v2[labels_v2["core_eligible"] == 0].copy()
    quarantine.to_csv(output_dir / "quarantine_v2.csv", index=False)

    diagnostics = {
        "protocols_total": int(len(old_registry)),
        "fetch_failures": int(len(failures)),
        "cached_payload_protocols": int(len(payloads)),
        "avax_core_history_protocols": int((registry_v2["history_source_mode"] == "avax_core").sum()),
        "total_only_protocols": int((registry_v2["history_source_mode"] == "total_only").sum()),
        "missing_history_protocols": int((registry_v2["history_source_mode"] == "missing").sum()),
        "early_feature_rows_v2": int(len(early_v2)),
        "summary_rows_v2": int(len(summary_v2)),
        "migrated_candidates_v2": int(labels_v2["migrated_candidate"].fillna(0).sum()),
        "structural_decay_known_v2": int(labels_v2["structural_decay_target_v2"].isin([0, 1]).sum()),
        "core_eligible_protocols_v2": int(labels_v2["core_eligible"].fillna(0).sum()),
        "boundary_excluded_protocols_v2": int((labels_v2["core_universe_status"] == "boundary_excluded").sum()),
        "data_incomplete_protocols_v2": int((labels_v2["core_universe_status"] == "data_incomplete").sum()),
        "old_early_feature_rows": int(len(pd.read_csv(repo_root / "avax_data" / "early_features.csv"))),
        "old_summary_rows": int(len(pd.read_csv(repo_root / "avax_data" / "features_summary.csv"))),
    }
    with open(output_dir / "dataset_v2_diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, ensure_ascii=False)

    print("Dataset v2 rebuild completed.")
    print()
    print(pd.DataFrame([diagnostics]).to_string(index=False))
    print()
    print("History source modes:")
    print(registry_v2["history_source_mode"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
