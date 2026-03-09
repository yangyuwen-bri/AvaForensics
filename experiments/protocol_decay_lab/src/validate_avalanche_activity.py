from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


RPC_URL = "https://api.avax.network/ext/bc/C/rpc"
CHUNK_BLOCKS = 2_048
BLOCK_WINDOWS = [2_048, 8_192]
TOP_NATIVE_REVIVAL_WITH_ADDRESS = 15


def resolve_paths() -> Tuple[Path, Path, Path]:
    output_root = Path(__file__).resolve().parents[1] / "outputs"
    dataset_dir = output_root / "dataset_v2"
    model_dir = output_root / "model_v2"
    return output_root, dataset_dir, model_dir


def rpc_call(method: str, params: List[object]) -> object:
    response = requests.post(
        RPC_URL,
        json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(str(payload["error"]))
    return payload["result"]


def hex_block(block_number: int) -> str:
    return hex(int(block_number))


def get_latest_block() -> int:
    return int(rpc_call("eth_blockNumber", []), 16)


def count_logs(address: str, latest_block: int, blocks_back: int) -> int:
    from_block = max(latest_block - blocks_back, 0)
    total = 0
    current = from_block
    while current <= latest_block:
        chunk_end = min(current + CHUNK_BLOCKS - 1, latest_block)
        params = [
            {
                "fromBlock": hex_block(current),
                "toBlock": hex_block(chunk_end),
                "address": address,
            }
        ]
        result = rpc_call("eth_getLogs", params)
        total += len(result)
        current = chunk_end + 1
    return total


def extract_avax_address(payload: Dict[str, object]) -> Optional[str]:
    address = payload.get("address", "")
    if address:
        for part in str(address).split(","):
            chunk = part.strip()
            if "avax:" in chunk.lower():
                return chunk.split(":")[-1].strip()

    contracts = payload.get("contracts") or {}
    if isinstance(contracts, dict):
        avax = contracts.get("avax")
        if isinstance(avax, list) and avax:
            return str(avax[0])
        if isinstance(avax, str):
            return avax

    chain_tvls = payload.get("chainTvls", {}) or {}
    for key, value in chain_tvls.items():
        if "avalanche" not in str(key).lower() or not isinstance(value, dict):
            continue
        sub = value.get("contracts")
        if isinstance(sub, list) and sub:
            return str(sub[0])
        if isinstance(sub, str):
            return sub
    return None


def classify_activity_support(mode: str, logs_short: Optional[int], logs_medium: Optional[int], address: Optional[str]) -> str:
    if not address:
        return "address_unavailable"
    if logs_short is None or logs_medium is None:
        return "rpc_failed"

    if mode == "native_revival_or_threshold_boundary":
        if logs_short > 0 and logs_medium > 0:
            return "supports_native_revival"
        if logs_medium > 0:
            return "weak_native_revival_support"
        return "no_recent_avax_activity"

    if mode == "multichain_relocation":
        if logs_medium == 0:
            return "supports_avax_side_fade"
        if logs_short == 0:
            return "partial_avax_fade"
        return "active_on_avax_contract"

    return "unclassified"


def main() -> None:
    _, dataset_dir, model_dir = resolve_paths()
    payload_dir = dataset_dir / "payloads"

    scores = pd.read_csv(model_dir / "two_stage_v2_scores.csv")

    relocation = scores[scores["decay_mode_v2"] == "multichain_relocation"].copy()
    revival = scores[scores["decay_mode_v2"] == "native_revival_or_threshold_boundary"].copy()
    revival = revival.sort_values("composite_terminal_prob", ascending=False).head(TOP_NATIVE_REVIVAL_WITH_ADDRESS)

    candidate_rows: List[Dict[str, object]] = []
    for mode_name, frame in [
        ("multichain_relocation", relocation),
        ("native_revival_or_threshold_boundary", revival),
    ]:
        for _, row in frame.iterrows():
            payload_path = payload_dir / f"{row['slug']}.json"
            payload = json.load(open(payload_path, "r", encoding="utf-8")) if payload_path.exists() else {}
            address = extract_avax_address(payload)
            candidate_rows.append(
                {
                    "slug": row["slug"],
                    "name": row["name"],
                    "decay_mode_v2": mode_name,
                    "composite_terminal_prob": row["composite_terminal_prob"],
                    "current_avax_core_tvl": row["current_avax_core_tvl"],
                    "current_total_core_tvl": row["current_total_core_tvl"],
                    "current_avax_share": row["current_avax_share"],
                    "active_chain_count_core": row["active_chain_count_core"],
                    "address": address,
                }
            )

    latest_block = get_latest_block()

    rows: List[Dict[str, object]] = []
    for item in candidate_rows:
        address = item["address"]
        code_present = None
        logs = {block_count: None for block_count in BLOCK_WINDOWS}
        error = ""
        try:
            if address:
                code = rpc_call("eth_getCode", [address, "latest"])
                code_present = code not in {"0x", "0x0", None}
                if code_present:
                    for block_count in BLOCK_WINDOWS:
                        logs[block_count] = count_logs(address, latest_block, block_count)
        except Exception as exc:
            error = str(exc)

        rows.append(
            {
                **item,
                "code_present": code_present,
                "recent_logs_last_2048_blocks": logs[2048],
                "recent_logs_last_8192_blocks": logs[8192],
                "activity_validation_status": classify_activity_support(
                    item["decay_mode_v2"],
                    logs[2048],
                    logs[8192],
                    address,
                ),
                "error": error,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(model_dir / "avalanche_activity_validation.csv", index=False)

    summary = (
        result.groupby(["decay_mode_v2", "activity_validation_status"], dropna=False)
        .size()
        .reset_index(name="protocol_count")
        .sort_values(["decay_mode_v2", "protocol_count"], ascending=[True, False])
    )
    summary.to_csv(model_dir / "avalanche_activity_validation_summary.csv", index=False)

    run_meta = {
        "latest_block": latest_block,
        "windows_blocks": BLOCK_WINDOWS,
        "relocation_candidates_checked": int((result["decay_mode_v2"] == "multichain_relocation").sum()),
        "native_revival_candidates_checked": int((result["decay_mode_v2"] == "native_revival_or_threshold_boundary").sum()),
        "address_coverage_relocation": int(
            result[result["decay_mode_v2"] == "multichain_relocation"]["address"].notna().sum()
        ),
        "address_coverage_native_revival": int(
            result[result["decay_mode_v2"] == "native_revival_or_threshold_boundary"]["address"].notna().sum()
        ),
    }
    with open(model_dir / "avalanche_activity_validation_meta.json", "w", encoding="utf-8") as handle:
        json.dump(run_meta, handle, indent=2, ensure_ascii=False)

    print("Avalanche activity validation completed.")
    print()
    print(json.dumps(run_meta, indent=2, ensure_ascii=False))
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
