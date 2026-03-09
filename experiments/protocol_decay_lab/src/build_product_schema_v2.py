from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def resolve_paths() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs" / "model_v2"


def evidence_level(row: pd.Series) -> str:
    mode = row.get("decay_mode_v2")
    status = row.get("validation_status", "")
    adapter_status = row.get("adapter_evidence_status", "")
    adapter_source_type = row.get("adapter_source_type", "")
    methodology_types = {"adapter_subgraph", "adapter_onchain_aggregate", "payload_chain_specific"}

    if mode == "multichain_relocation":
        if adapter_status == "mapped_live_code":
            return "address_registered"
        if adapter_source_type in methodology_types:
            return "methodology_backed"
        if status == "relocation_avax_side_faded":
            return "onchain_supported"
        if status == "relocation_partial_fade":
            return "weak_onchain_support"
        if status == "no_code_on_avax":
            return "address_mismatch"
        if status == "address_unavailable":
            return "address_gap"
        return "inferred_only"

    if mode == "native_revival_or_threshold_boundary":
        if status == "revival_supported":
            return "onchain_supported"
        if status in {"revival_weak_evidence", "revival_weak_evidence_long_window"}:
            return "weak_onchain_support"
        if status in {"revival_no_recent_logs", "revival_no_recent_logs_extended"}:
            return "threshold_only"
        if adapter_status == "mapped_live_code":
            return "address_registered"
        if adapter_source_type in methodology_types:
            return "methodology_backed"
        if status == "no_code_on_avax":
            return "address_mismatch"
        if status == "address_unavailable":
            return "address_gap"
        return "inferred_only"

    if mode in {"terminal_global_decay", "resilient_or_unproven"}:
        return "model_only"

    if mode == "avax_side_decay_but_globally_alive":
        if adapter_status == "mapped_live_code":
            return "address_registered"
        if adapter_source_type in methodology_types:
            return "methodology_backed"
        return "inferred_only"

    if mode in {"not_core_eligible", "data_incomplete"}:
        return "not_eligible"

    return "unknown"


def evidence_summary(row: pd.Series) -> str:
    mode = row.get("decay_mode_v2")
    status = row.get("validation_status", "")
    adapter_status = row.get("adapter_evidence_status", "")
    adapter_source_type = row.get("adapter_source_type", "")
    adapter_source_note = row.get("adapter_source_note", "")
    extended_note = row.get("extended_validation_note", "")
    methodology_types = {"adapter_subgraph", "adapter_onchain_aggregate", "payload_chain_specific"}
    adapter_note = str(adapter_source_note) if pd.notna(adapter_source_note) else ""
    extended_note = str(extended_note) if pd.notna(extended_note) else ""
    short_logs = row.get("recent_logs_last_2048_blocks")
    medium_logs = row.get("recent_logs_last_8192_blocks")

    if mode == "multichain_relocation":
        if adapter_status == "mapped_live_code":
            return "Avalanche contract address confirmed and live code detected on C-Chain."
        if adapter_source_type in methodology_types:
            return adapter_note
        if status == "address_gap":
            return "Relocation inferred from TVL and cross-chain share; Avalanche contract address unavailable."
        if status == "address_mismatch":
            return "Candidate address does not map to live Avalanche code."
        if status == "relocation_avax_side_faded":
            return "Avalanche address mapped and recent contract logs are absent."
        if status == "relocation_partial_fade":
            return "Avalanche address mapped; very low recent contract activity."
        return "Relocation remains inference-led."

    if mode == "native_revival_or_threshold_boundary":
        if adapter_status == "mapped_live_code":
            return "Avalanche contract address confirmed and live code detected on C-Chain."
        if adapter_source_type in methodology_types:
            return adapter_note
        if status == "revival_supported":
            return f"Avalanche contract active in short windows ({int(short_logs)} / {int(medium_logs)} recent logs)."
        if status == "revival_weak_evidence":
            if extended_note:
                return extended_note
            return f"Avalanche contract shows medium-window activity ({int(medium_logs)} logs) but weak very recent activity."
        if status == "revival_weak_evidence_long_window":
            return extended_note
        if status == "revival_no_recent_logs":
            return "TVL remains above threshold, but short-window Avalanche contract activity was not observed."
        if status == "revival_no_recent_logs_extended":
            return extended_note
        if status == "address_gap":
            return "Revival classification based on TVL only; Avalanche contract address unavailable."
        if status == "address_mismatch":
            return "Candidate address does not map to live Avalanche code."
        return "Revival remains inference-led."

    if mode == "terminal_global_decay":
        return "Early-window terminal decay model indicates persistent decline."

    if mode == "resilient_or_unproven":
        return "No strong terminal decay signal under current v2 definition."

    if mode == "avax_side_decay_but_globally_alive":
        if adapter_status == "mapped_live_code":
            return "Avalanche contract address confirmed and live code detected on C-Chain, while Avalanche share remains materially weaker than global TVL."
        if adapter_source_type in methodology_types:
            return adapter_note
        return "Avalanche-side weakness inferred from TVL and cross-chain distribution, while global activity persists."

    if mode == "not_core_eligible":
        return "Excluded from Avalanche core universe due to missing bare Avalanche history."

    if mode == "data_incomplete":
        return "Insufficient source data for core classification."

    return ""


def main() -> None:
    model_dir = resolve_paths()
    scores = pd.read_csv(model_dir / "two_stage_v2_scores.csv")
    evidence = pd.read_csv(model_dir / "protocol_address_evidence_best.csv")
    address_frames = []
    for path in [
        model_dir / "adapter_address_confirmed.csv",
        model_dir / "native_revival_address_confirmed.csv",
        model_dir / "avax_side_decay_address_confirmed.csv",
    ]:
        if path.exists():
            frame = pd.read_csv(path)
            if "address_source_type" not in frame.columns:
                frame["address_source_type"] = "adapter_source"
            address_frames.append(frame)
    adapter = pd.concat(address_frames, ignore_index=True) if address_frames else pd.DataFrame(columns=["slug"])

    methodology_frames = []
    for path in [
        model_dir / "adapter_methodology_confirmed.csv",
        model_dir / "native_revival_methodology_confirmed.csv",
        model_dir / "avax_side_decay_methodology_confirmed.csv",
    ]:
        if path.exists():
            methodology_frames.append(pd.read_csv(path))
    methodology = (
        pd.concat(methodology_frames, ignore_index=True)
        if methodology_frames
        else pd.DataFrame(columns=["slug"])
    )

    merged = scores.merge(
        evidence[
            [
                "slug",
                "candidate_address",
                "candidate_source",
                "code_present_on_avax",
                "recent_logs_last_2048_blocks",
                "recent_logs_last_8192_blocks",
                "validation_status",
            ]
        ],
        on="slug",
        how="left",
    )
    if not adapter.empty:
        merged = merged.merge(adapter, on="slug", how="left")
        use_adapter = merged["adapter_evidence_status"].eq("mapped_live_code")
        merged.loc[use_adapter, "candidate_address"] = merged.loc[use_adapter, "adapter_address"]
        merged.loc[use_adapter, "candidate_source"] = merged.loc[use_adapter, "address_source_type"]
        merged.loc[use_adapter, "code_present_on_avax"] = True
    if not methodology.empty:
        merged = merged.merge(methodology, on="slug", how="left")
    extended_path = model_dir / "native_revival_extended_validation.csv"
    if extended_path.exists():
        extended = pd.read_csv(extended_path)
        merged = merged.merge(
            extended[
                [
                    "slug",
                    "first_positive_blocks_back",
                    "observed_logs_until_hit",
                    "extended_validation_status",
                    "extended_validation_note",
                ]
            ],
            on="slug",
            how="left",
        )
        merged["validation_status"] = merged["extended_validation_status"].combine_first(merged["validation_status"])

    merged["evidence_level"] = merged.apply(evidence_level, axis=1)
    merged["evidence_summary"] = merged.apply(evidence_summary, axis=1)
    merged["address_registry_status"] = np.select(
        [
            merged["candidate_address"].fillna("").ne("") & merged["code_present_on_avax"].eq(True),
            merged["candidate_address"].fillna("").ne("") & merged["code_present_on_avax"].eq(False),
            merged["candidate_address"].fillna("").eq(""),
        ],
        [
            "mapped_live_code",
            "mapped_no_code",
            "missing_address",
        ],
        default="unknown",
    )

    product = merged[
        [
            "slug",
            "name",
            "core_eligible",
            "core_universe_status",
            "stage1_terminal_prob",
            "stage2_terminal_mode_prob",
            "composite_terminal_prob",
            "composite_watch_band",
            "decay_mode_v2",
            "evidence_level",
            "evidence_summary",
            "address_registry_status",
            "candidate_address",
            "candidate_source",
            "adapter_address",
            "adapter_role",
            "adapter_evidence_status",
            "adapter_source_type",
            "adapter_source_note",
            "validation_status",
            "current_avax_core_tvl",
            "current_total_core_tvl",
            "current_avax_share",
            "migrated_candidate",
            "reason_1",
            "reason_2",
            "reason_3",
        ]
    ].copy()
    product = product.rename(
        columns={
            "slug": "protocol_id",
            "name": "protocol_name",
            "stage1_terminal_prob": "terminal_risk",
            "decay_mode_v2": "decay_mode",
        }
    )
    product.to_csv(model_dir / "product_schema_v2.csv", index=False)

    summary = (
        product.groupby(["decay_mode", "evidence_level"], dropna=False)
        .size()
        .reset_index(name="protocol_count")
        .sort_values(["decay_mode", "protocol_count"], ascending=[True, False])
    )
    summary.to_csv(model_dir / "product_schema_v2_summary.csv", index=False)

    print("Product schema v2 built.")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
