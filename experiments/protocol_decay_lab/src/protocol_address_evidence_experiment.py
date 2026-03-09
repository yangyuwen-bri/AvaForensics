from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


RPC_URL = "https://api.avax.network/ext/bc/C/rpc"
BLOCK_WINDOWS = [2048, 8192]
TOP_RELOCATION = 10
TOP_NATIVE_REVIVAL = 15
MAX_CANDIDATES_PER_PROTOCOL = 3


def resolve_paths() -> Tuple[Path, Path]:
    output_root = Path(__file__).resolve().parents[1] / "outputs"
    model_dir = output_root / "model_v2"
    return output_root, model_dir


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


def extract_candidates(payload: Dict[str, object]) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []
    address = payload.get("address", "")
    if address:
        for part in str(address).split(","):
            chunk = part.strip()
            if not chunk:
                continue
            if ":" in chunk:
                prefix, value = chunk.split(":", 1)
                if prefix.lower() == "avax":
                    candidates.append((value.strip(), "address_field_avax"))
                else:
                    candidates.append((value.strip(), f"address_field_{prefix.lower()}"))
            else:
                candidates.append((chunk, "address_field_bare"))

    contracts = payload.get("contracts") or {}
    if isinstance(contracts, dict):
        avax_contracts = contracts.get("avax")
        if isinstance(avax_contracts, list):
            for idx, value in enumerate(avax_contracts):
                candidates.append((str(value), f"contracts_avax_{idx}"))
        elif isinstance(avax_contracts, str):
            candidates.append((avax_contracts, "contracts_avax"))

    chain_tvls = payload.get("chainTvls", {}) or {}
    for key, value in chain_tvls.items():
        if "avalanche" not in str(key).lower() or not isinstance(value, dict):
            continue
        sub = value.get("contracts")
        if isinstance(sub, list):
            for idx, item in enumerate(sub):
                candidates.append((str(item), f"chainTvls_{key}_contracts_{idx}"))
        elif isinstance(sub, str):
            candidates.append((sub, f"chainTvls_{key}_contracts"))

    deduped: List[Tuple[str, str]] = []
    seen = set()
    for address_value, source in candidates:
        addr = str(address_value).strip()
        if not addr:
            continue
        if addr.lower() in seen:
            continue
        seen.add(addr.lower())
        deduped.append((addr, source))
    return deduped


def candidate_priority(source: str) -> int:
    if source == "address_field_avax":
        return 0
    if source.startswith("contracts_avax"):
        return 1
    if source.startswith("chainTvls_Avalanche_contracts"):
        return 2
    if source == "address_field_bare":
        return 3
    if source.startswith("address_field_"):
        return 4
    return 5


def address_code_present(address: str) -> Tuple[Optional[bool], Optional[int], str]:
    try:
        code = rpc_call("eth_getCode", [address, "latest"])
        present = code not in {"0x", "0x0", None}
        return present, len(code) if isinstance(code, str) else None, ""
    except Exception as exc:
        return None, None, str(exc)


def recent_log_count(address: str, latest_block: int, blocks_back: int) -> Tuple[Optional[int], str]:
    total = 0
    current = max(latest_block - blocks_back + 1, 0)
    try:
        while current <= latest_block:
            end = min(current + 2047, latest_block)
            logs = rpc_call(
                "eth_getLogs",
                [{"fromBlock": hex_block(current), "toBlock": hex_block(end), "address": address}],
            )
            total += len(logs)
            current = end + 1
        return total, ""
    except Exception as exc:
        return None, str(exc)


def main() -> None:
    output_root, model_dir = resolve_paths()
    dataset_dir = output_root / "dataset_v2"
    payload_dir = dataset_dir / "payloads"

    scores = pd.read_csv(model_dir / "two_stage_v2_scores.csv")
    latest_block = int(rpc_call("eth_blockNumber", []), 16)

    focus = pd.concat(
        [
            scores[scores["decay_mode_v2"] == "multichain_relocation"].sort_values("composite_terminal_prob", ascending=False).head(TOP_RELOCATION),
            scores[scores["decay_mode_v2"] == "native_revival_or_threshold_boundary"].sort_values("composite_terminal_prob", ascending=False).head(TOP_NATIVE_REVIVAL),
        ],
        ignore_index=True,
    )

    registry_rows: List[Dict[str, object]] = []
    best_rows: List[Dict[str, object]] = []
    for _, row in focus.iterrows():
        slug = row["slug"]
        payload_path = payload_dir / f"{slug}.json"
        payload = json.load(open(payload_path, "r", encoding="utf-8")) if payload_path.exists() else {}
        candidates = sorted(extract_candidates(payload), key=lambda item: (candidate_priority(item[1]), item[0]))[:MAX_CANDIDATES_PER_PROTOCOL]
        if not candidates:
            result_row = {
                "slug": slug,
                "name": row["name"],
                "decay_mode_v2": row["decay_mode_v2"],
                "candidate_address": "",
                "candidate_source": "none_found",
                "code_present_on_avax": None,
                "code_len": None,
                "recent_logs_last_2048_blocks": None,
                "recent_logs_last_8192_blocks": None,
                "validation_status": "address_unavailable",
                "error": "",
            }
            registry_rows.append(result_row)
            best_rows.append(result_row)
            continue

        slug_rows: List[Dict[str, object]] = []
        for candidate_address, candidate_source in candidates:
            code_present, code_len, code_error = address_code_present(candidate_address)
            logs_short = None
            logs_medium = None
            logs_error = ""
            if code_present:
                logs_short, err_short = recent_log_count(candidate_address, latest_block, BLOCK_WINDOWS[0])
                logs_medium, err_medium = recent_log_count(candidate_address, latest_block, BLOCK_WINDOWS[1])
                logs_error = " | ".join(part for part in [err_short, err_medium] if part)

            if not code_present:
                validation_status = "no_code_on_avax"
            elif row["decay_mode_v2"] == "native_revival_or_threshold_boundary":
                if (logs_short or 0) > 0 and (logs_medium or 0) > 0:
                    validation_status = "revival_supported"
                elif (logs_medium or 0) > 0:
                    validation_status = "revival_weak_evidence"
                else:
                    validation_status = "revival_no_recent_logs"
            else:
                if (logs_medium or 0) == 0:
                    validation_status = "relocation_avax_side_faded"
                elif (logs_short or 0) == 0:
                    validation_status = "relocation_partial_fade"
                else:
                    validation_status = "relocation_active_on_avax"

            slug_rows.append(
                {
                    "slug": slug,
                    "name": row["name"],
                    "decay_mode_v2": row["decay_mode_v2"],
                    "candidate_address": candidate_address,
                    "candidate_source": candidate_source,
                    "code_present_on_avax": code_present,
                    "code_len": code_len,
                    "recent_logs_last_2048_blocks": logs_short,
                    "recent_logs_last_8192_blocks": logs_medium,
                    "validation_status": validation_status,
                    "error": " | ".join(part for part in [code_error, logs_error] if part),
                }
            )
        registry_rows.extend(slug_rows)
        ranked_slug_rows = sorted(
            slug_rows,
            key=lambda item: (
                item["code_present_on_avax"] is not True,
                -(item["recent_logs_last_8192_blocks"] or -1),
                -(item["recent_logs_last_2048_blocks"] or -1),
                candidate_priority(item["candidate_source"]),
            ),
        )
        best_rows.append(ranked_slug_rows[0])

    registry = pd.DataFrame(registry_rows)
    registry.to_csv(model_dir / "protocol_address_evidence_registry.csv", index=False)

    best = pd.DataFrame(best_rows)
    best.to_csv(model_dir / "protocol_address_evidence_best.csv", index=False)

    summary = (
        best.groupby(["decay_mode_v2", "validation_status"], dropna=False)
        .size()
        .reset_index(name="protocol_count")
        .sort_values(["decay_mode_v2", "protocol_count"], ascending=[True, False])
    )
    summary.to_csv(model_dir / "protocol_address_evidence_summary.csv", index=False)

    print("Protocol address evidence experiment completed.")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
