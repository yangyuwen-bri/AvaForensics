from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests


RPC_URL = "https://api.avax.network/ext/bc/C/rpc"
ADAPTER_RAW_URL = "https://raw.githubusercontent.com/DefiLlama/DefiLlama-Adapters/main/projects/{module_path}"
RELOCATION_TOP = 12


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


def fetch_adapter_source(module_path: str) -> str:
    response = requests.get(ADAPTER_RAW_URL.format(module_path=module_path), timeout=30)
    response.raise_for_status()
    return response.text


def extract_avax_addresses_from_source(source: str) -> List[Tuple[str, str, int]]:
    rows: List[Tuple[str, str, int]] = []
    lines = source.splitlines()
    for idx, line in enumerate(lines):
        if "avax" not in line.lower() and "avalanche" not in line.lower():
            continue
        window = "\n".join(lines[max(0, idx - 4): min(len(lines), idx + 10)])
        for address in sorted(set(re.findall(r"0x[a-fA-F0-9]{40}", window))):
            rows.append((address, line.strip(), idx + 1))
    deduped: List[Tuple[str, str, int]] = []
    seen = set()
    for address, hint, line_no in rows:
        if address.lower() in seen:
            continue
        seen.add(address.lower())
        deduped.append((address, hint, line_no))
    return deduped


def main() -> None:
    _, dataset_dir, model_dir = resolve_paths()
    payload_dir = dataset_dir / "payloads"
    scores = pd.read_csv(model_dir / "two_stage_v2_scores.csv")
    relocation = (
        scores[scores["decay_mode_v2"] == "multichain_relocation"]
        .sort_values("composite_terminal_prob", ascending=False)
        .head(RELOCATION_TOP)
        .copy()
    )

    registry_rows: List[Dict[str, object]] = []
    best_rows: List[Dict[str, object]] = []

    for _, row in relocation.iterrows():
        slug = row["slug"]
        payload = json.load(open(payload_dir / f"{slug}.json", "r", encoding="utf-8"))
        module_path = payload.get("module")
        if not module_path:
            result_row = {
                "slug": slug,
                "name": row["name"],
                "module_path": "",
                "candidate_address": "",
                "source_hint": "",
                "source_line": None,
                "code_present_on_avax": None,
                "recent_logs_last_2048_blocks": None,
                "validation_status": "module_missing",
            }
            registry_rows.append(result_row)
            best_rows.append(result_row)
            continue

        try:
            source = fetch_adapter_source(module_path)
            candidates = extract_avax_addresses_from_source(source)
        except Exception:
            result_row = {
                "slug": slug,
                "name": row["name"],
                "module_path": module_path,
                "candidate_address": "",
                "source_hint": "",
                "source_line": None,
                "code_present_on_avax": None,
                "recent_logs_last_2048_blocks": None,
                "validation_status": "module_fetch_failed",
            }
            registry_rows.append(result_row)
            best_rows.append(result_row)
            continue

        if not candidates:
            result_row = {
                "slug": slug,
                "name": row["name"],
                "module_path": module_path,
                "candidate_address": "",
                "source_hint": "",
                "source_line": None,
                "code_present_on_avax": None,
                "recent_logs_last_2048_blocks": None,
                "validation_status": "no_avax_address_in_module",
            }
            registry_rows.append(result_row)
            best_rows.append(result_row)
            continue

        slug_rows: List[Dict[str, object]] = []
        for address, hint, line_no in candidates:
            try:
                code = rpc_call("eth_getCode", [address, "latest"])
                code_present = code not in {"0x", "0x0", None}
                status = "mapped_live_code" if code_present else "no_code_on_avax"
            except Exception:
                code_present = None
                status = "rpc_failed"

            slug_rows.append(
                {
                    "slug": slug,
                    "name": row["name"],
                    "module_path": module_path,
                    "candidate_address": address,
                    "source_hint": hint,
                    "source_line": line_no,
                    "code_present_on_avax": code_present,
                    "validation_status": status,
                }
            )

        registry_rows.extend(slug_rows)
        ranked = sorted(
            slug_rows,
            key=lambda item: (
                item["code_present_on_avax"] is not True,
                item["source_line"] or 999999,
            ),
        )
        best_rows.append(ranked[0])

    registry = pd.DataFrame(registry_rows)
    best = pd.DataFrame(best_rows)
    summary = (
        best.groupby(["validation_status"], dropna=False)
        .size()
        .reset_index(name="protocol_count")
        .sort_values("protocol_count", ascending=False)
    )

    registry.to_csv(model_dir / "adapter_address_registry.csv", index=False)
    best.to_csv(model_dir / "adapter_address_best.csv", index=False)
    summary.to_csv(model_dir / "adapter_address_summary.csv", index=False)

    print("Adapter address registry experiment completed.")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
