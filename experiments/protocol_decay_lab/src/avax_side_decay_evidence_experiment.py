from __future__ import annotations

from pathlib import Path

import pandas as pd


ADDRESS_CONFIRMATIONS = [
    {
        "slug": "superform",
        "module_path": "superform/index.js",
        "adapter_address": "0xD85ec15A9F814D6173bF1a89273bFB3964aAdaEC",
        "adapter_role": "avax_factory_from_adapter",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "adapter_source",
    },
    {
        "slug": "gamma",
        "module_path": "visor/config.js",
        "adapter_address": "0xbF145c5239B1327909f3e37CA0cF890d014105E2",
        "adapter_role": "avax_registry_from_adapter",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "adapter_source",
    },
    {
        "slug": "nested",
        "module_path": "nested/index.js",
        "adapter_address": "0x150fb0Cfa5bF3D4023bA198C725b6DCBc1577f21",
        "adapter_role": "avax_reserve_from_adapter",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "adapter_source",
    },
    {
        "slug": "bifi",
        "module_path": "bifi/index.js",
        "adapter_address": "0x446881360d6d39779D292662fca9BC85C5789dB3",
        "adapter_role": "avax_pool_from_adapter",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "adapter_source",
    },
    {
        "slug": "kine-finance",
        "module_path": "kinefinance/index.js",
        "adapter_address": "0x0ec3126390c606be63a0fa6585e68075f06679c6",
        "adapter_role": "avax_comptroller_from_adapter",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "adapter_source",
    },
    {
        "slug": "sushi-bentobox",
        "module_path": "sushiswap-bentobox/helper.js",
        "adapter_address": "0x0711b6026068f736bae6b213031fce978d48e026",
        "adapter_role": "avax_bentobox_from_adapter",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "adapter_source",
    },
]


METHODOLOGY_CONFIRMATIONS = [
    {
        "slug": "superform",
        "module_path": "superform/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter defines explicit avax support, enumerates Superform vaults from the factory and sums underlying asset balances on-chain.",
    },
    {
        "slug": "vesper",
        "module_path": "vesper/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter has a dedicated avax pool API endpoint and aggregates Avalanche pool token balances from live Vesper pool contracts.",
    },
    {
        "slug": "gamma",
        "module_path": "visor/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter includes explicit Avalanche hypervisor registries and sums token balances across on-chain hypervisor contracts.",
    },
    {
        "slug": "nested",
        "module_path": "nested/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter exposes an avax reserve contract and counts Avalanche-side assets held in the NestedReserve contract.",
    },
    {
        "slug": "bifi",
        "module_path": "bifi/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter defines explicit Avalanche lending pool contracts and sums token balances directly across Avalanche pool owners.",
    },
    {
        "slug": "kine-finance",
        "module_path": "kinefinance/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter has an explicit avax Compound-style comptroller and cether configuration, confirming Avalanche lending support in source code.",
    },
    {
        "slug": "granary-finance",
        "module_path": "the-granary/index.js",
        "adapter_source_type": "payload_chain_specific",
        "adapter_source_note": "Source payload contains dedicated Avalanche TVL and Avalanche-borrowed histories even though the current adapter module path is unresolved.",
    },
    {
        "slug": "sushi-bentobox",
        "module_path": "sushiswap-bentobox/index.js",
        "adapter_source_type": "adapter_subgraph",
        "adapter_source_note": "Adapter defines Avalanche BentoBox, Furo, Kashi and Trident subgraphs; top-level avax export intentionally zeros direct BentoBox TVL to avoid double counting child protocols.",
    },
]


def resolve_model_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs" / "model_v2"


def main() -> None:
    model_dir = resolve_model_dir()

    address_df = pd.DataFrame(ADDRESS_CONFIRMATIONS).sort_values("slug").reset_index(drop=True)
    address_df.to_csv(model_dir / "avax_side_decay_address_confirmed.csv", index=False)

    methodology_df = (
        pd.DataFrame(METHODOLOGY_CONFIRMATIONS).sort_values("slug").reset_index(drop=True)
    )
    methodology_df.to_csv(model_dir / "avax_side_decay_methodology_confirmed.csv", index=False)

    summary = pd.DataFrame(
        [
            {"artifact": "avax_side_decay_address_confirmed", "row_count": int(len(address_df))},
            {"artifact": "avax_side_decay_methodology_confirmed", "row_count": int(len(methodology_df))},
        ]
    )
    summary.to_csv(model_dir / "avax_side_decay_evidence_summary.csv", index=False)

    print("AVAX-side decay evidence registries written.")
    print()
    print(address_df.to_string(index=False))
    print()
    print(methodology_df.to_string(index=False))


if __name__ == "__main__":
    main()
