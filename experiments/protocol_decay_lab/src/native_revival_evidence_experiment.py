from __future__ import annotations

from pathlib import Path

import pandas as pd


ADDRESS_CONFIRMATIONS = [
    {
        "slug": "lydia",
        "module_path": "lydia/index.js",
        "adapter_address": "0x4c9b4e1ac6f24cde3660d5e4ef1ebf77c710c084",
        "adapter_role": "payload_explicit_avax_address",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "payload_field_confirmed",
    },
    {
        "slug": "sherpa-cash",
        "module_path": "sherpa-cash/index.js",
        "adapter_address": "0xa5e59761ebd4436fa4d20e1a27cba29fb2471fc6",
        "adapter_role": "payload_explicit_avax_address",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "payload_field_confirmed",
    },
    {
        "slug": "baguette",
        "module_path": "baguette/index.js",
        "adapter_address": "0xa1144a6a1304bd9cbb16c800f7a867508726566e",
        "adapter_role": "payload_explicit_avax_address",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "payload_field_confirmed",
    },
    {
        "slug": "joe-stek",
        "module_path": "trader-joe-stek/index.js",
        "adapter_address": "0x6e84a6216ea6dacc71ee8e6b0a5b7322eebc0fdd",
        "adapter_role": "payload_explicit_avax_address",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "payload_field_confirmed",
    },
    {
        "slug": "hurricaneswap",
        "module_path": "hurricaneswap.js",
        "adapter_address": "0x45C13620B55C35A5f539d26E88247011Eb10fDbd",
        "adapter_role": "payload_explicit_avax_address",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "payload_field_confirmed",
    },
    {
        "slug": "hakuswap",
        "module_path": "hakuswap/index.js",
        "adapter_address": "0x695Fa794d59106cEbd40ab5f5cA19F458c723829",
        "adapter_role": "payload_explicit_avax_address",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "payload_field_confirmed",
    },
    {
        "slug": "complus-network",
        "module_path": "complus/index.js",
        "adapter_address": "0x3711c397b6c8f7173391361e27e67d72f252caad",
        "adapter_role": "payload_explicit_avax_address",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "payload_field_confirmed",
    },
    {
        "slug": "penguin-finance",
        "module_path": "penguin/index.js",
        "adapter_address": "0xe896CDeaAC9615145c0cA09C8Cd5C25bced6384c",
        "adapter_role": "payload_explicit_avax_address",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "payload_field_confirmed",
    },
    {
        "slug": "cycle-finance",
        "module_path": "cyclefinance/index.js",
        "adapter_address": "0x81440c939f2c1e34fc7048e518a637205a632a74",
        "adapter_role": "payload_explicit_avax_address",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "payload_field_confirmed",
    },
    {
        "slug": "defrost",
        "module_path": "defrost/index.js",
        "adapter_address": "0x47EB6F7525C1aA999FBC9ee92715F5231eB1241D",
        "adapter_role": "payload_explicit_avax_address",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "payload_field_confirmed",
    },
    {
        "slug": "nereus-finance",
        "module_path": "nereus/index.js",
        "adapter_address": "0xfcDe4A87b8b6FA58326BB462882f1778158B02F1",
        "adapter_role": "payload_explicit_avax_address",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "payload_field_confirmed",
    },
    {
        "slug": "joe-dex",
        "module_path": "traderjoe/index.js",
        "adapter_address": "0x6e84a6216ea6dacc71ee8e6b0a5b7322eebc0fdd",
        "adapter_role": "payload_explicit_avax_address",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "payload_field_confirmed",
    },
    {
        "slug": "joe-v2",
        "module_path": "traderjoe-lb/index.js",
        "adapter_address": "0x6e84a6216ea6dacc71ee8e6b0a5b7322eebc0fdd",
        "adapter_role": "payload_explicit_avax_address",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "payload_field_confirmed",
    },
    {
        "slug": "platypus-finance",
        "module_path": "platypus-finance/index.js",
        "adapter_address": "0x22d4002028f537599be9f666d1c4fa138522f9c8",
        "adapter_role": "payload_explicit_avax_address",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "payload_field_confirmed",
    },
    {
        "slug": "swapsicle-v1",
        "module_path": "swapsicle/index.js",
        "adapter_address": "0x240248628B7B6850352764C5dFa50D1592A033A8",
        "adapter_role": "payload_explicit_avax_address",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "payload_field_confirmed",
    },
    {
        "slug": "impermax-v2",
        "module_path": "impermax/index.js",
        "adapter_address": "0xf655c8567e0f213e6c634cd2a68d992152161dc6",
        "adapter_role": "payload_explicit_avax_address",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "payload_field_confirmed",
    },
    {
        "slug": "openocean",
        "module_path": "openocean.js",
        "adapter_address": "0x042AF448582d0a3cE3CFa5b65c2675e88610B18d",
        "adapter_role": "avax_factory_from_adapter",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "adapter_source",
    },
    {
        "slug": "mushrooms-finance",
        "module_path": "mushrooms.js",
        "adapter_address": "0xa33b55d868e57b20df957ddc2f044f09f676967b",
        "adapter_role": "avax_vault_from_adapter",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "adapter_source",
    },
    {
        "slug": "stonedefi",
        "module_path": "stonedefi/index.js",
        "adapter_address": "0x9BC91eAAb1380D3a40320B1b282b6f06e2F31Acf",
        "adapter_role": "avax_vault_from_adapter",
        "adapter_evidence_status": "mapped_live_code",
        "address_source_type": "adapter_source",
    },
]


METHODOLOGY_CONFIRMATIONS = [
    {
        "slug": "aave-v2",
        "module_path": "aave/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter exposes an explicit avax v2 registry and computes TVL/borrowed balances through Aave reserve aggregation helpers.",
    },
    {
        "slug": "yieldwolf",
        "module_path": "yieldwolf/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter has an avax masterchef config and sums vault balances on-chain across YieldWolf strategy contracts.",
    },
    {
        "slug": "struct-finance",
        "module_path": "struct-finance/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter enumerates Avalanche Struct vaults and sums underlying token balances across product and yield-source contracts.",
    },
    {
        "slug": "hubble-exchange",
        "module_path": "hubble-exchange/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter defines Avalanche token-owner pairs directly and sums collateral balances on-chain.",
    },
    {
        "slug": "autofarm",
        "module_path": "autofarm.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter contains an explicit avax masterchef and sums locked farm balances across strategy contracts on-chain.",
    },
    {
        "slug": "aperture-lm",
        "module_path": "aperture/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter uses an Avalanche manager contract and aggregates strategy equity values directly from on-chain contract calls.",
    },
    {
        "slug": "abracadabra-spell",
        "module_path": "abracadabra/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter includes Avalanche BentoBox addresses and sums locked market balances directly from on-chain balanceOf calls.",
    },
    {
        "slug": "platypus-finance",
        "module_path": "platypus-finance/index.js",
        "adapter_source_type": "adapter_subgraph",
        "adapter_source_note": "Adapter fetches Avalanche pool inventories from the Platypus subgraph and unwraps token balances from returned pool assets.",
    },
    {
        "slug": "swapsicle-v1",
        "module_path": "swapsicle/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter uses Avalanche factory and staking contracts to aggregate TVL and staking balances on-chain.",
    },
    {
        "slug": "impermax-v2",
        "module_path": "impermax/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter defines explicit Avalanche factory contracts and aggregates lending-pool balances via Impermax helper calls.",
    },
    {
        "slug": "joe-dex",
        "module_path": "traderjoe/index.js",
        "adapter_source_type": "adapter_subgraph",
        "adapter_source_note": "Adapter states Avalanche liquidity is sourced from Trader Joe exchange subgraph plus staking contract balances.",
    },
    {
        "slug": "joe-v2",
        "module_path": "traderjoe-lb/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter enumerates Avalanche Liquidity Book pairs from factory contracts and sums token balances on-chain.",
    },
    {
        "slug": "opyn-gamma",
        "module_path": "opyn-gamma/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter exposes an explicit avax TVL module, confirming Avalanche is a first-class measurement target in the source adapter.",
    },
    {
        "slug": "ribbon",
        "module_path": "ribbon/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter sums totalBalance across Ribbon Theta Vaults and exports a shared EVM methodology that includes Avalanche coverage.",
    },
    {
        "slug": "radioshack",
        "module_path": "radioshack/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter defines an explicit avax factory plus Avalanche staking contracts and aggregates pair liquidity on-chain.",
    },
    {
        "slug": "gro",
        "module_path": "groprotocol/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter exports a dedicated avax TVL path that sums ERC4626 vault assets for Gro Labs on Avalanche.",
    },
    {
        "slug": "cavalre",
        "module_path": "cavel-re/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter includes an explicit avax pool list and sums underlying asset balances from the pool contract on-chain.",
    },
    {
        "slug": "stake-dao",
        "module_path": "stakedao/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter has an explicit avax export and aggregates strategy balances through Stake DAO locker and strategy endpoints.",
    },
    {
        "slug": "curve-dex",
        "module_path": "curve/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter includes avax registry handling and unwraps pool token balances directly from Curve deployment contracts.",
    },
    {
        "slug": "thetanuts-finance",
        "module_path": "thetanuts/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter documents Avalanche vault coverage and counts funds deposited into Thetanuts Avalanche vaults and products.",
    },
    {
        "slug": "homora-v2",
        "module_path": "alpha-homora/v2.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter defines explicit avax Homora V2 APIs, graph endpoints and pool configs, then aggregates collateral and safebox balances.",
    },
    {
        "slug": "iron-bank",
        "module_path": "ironbank/index.js",
        "adapter_source_type": "payload_chain_specific",
        "adapter_source_note": "Source payload contains explicit Avalanche token-level TVL history plus separate Avalanche-borrowed history, even though the current adapter path is unresolved.",
    },
    {
        "slug": "kuu-finance",
        "module_path": "kuufinance/index.js",
        "adapter_source_type": "payload_chain_specific",
        "adapter_source_note": "Source payload contains a dedicated Avalanche token and TVL history series, despite missing contract address metadata in the current payload.",
    },
    {
        "slug": "0.exchange",
        "module_path": "zerodex/index.js",
        "adapter_source_type": "payload_chain_specific",
        "adapter_source_note": "Source payload contains a dedicated Avalanche token, token-in-USD and TVL history series, even though the legacy candidate address does not map to Avalanche code.",
    },
    {
        "slug": "openocean",
        "module_path": "openocean.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter defines an explicit avax factory and staking contracts, then aggregates liquidity with the same on-chain DEX methodology used on other chains.",
    },
    {
        "slug": "mushrooms-finance",
        "module_path": "mushrooms.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter includes a dedicated avax vault list and sums ERC4626 vault assets directly from Avalanche vault contracts.",
    },
    {
        "slug": "stonedefi",
        "module_path": "stonedefi/index.js",
        "adapter_source_type": "adapter_onchain_aggregate",
        "adapter_source_note": "Adapter defines an explicit avax vault set and calculates TVL from ERC4626 totalAssets across Avalanche vault contracts.",
    },
]


def resolve_model_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs" / "model_v2"


def main() -> None:
    model_dir = resolve_model_dir()

    address_df = pd.DataFrame(ADDRESS_CONFIRMATIONS).sort_values("slug").reset_index(drop=True)
    address_df.to_csv(model_dir / "native_revival_address_confirmed.csv", index=False)

    methodology_df = (
        pd.DataFrame(METHODOLOGY_CONFIRMATIONS).sort_values("slug").reset_index(drop=True)
    )
    methodology_df.to_csv(model_dir / "native_revival_methodology_confirmed.csv", index=False)

    summary_rows = [
        {"artifact": "native_revival_address_confirmed", "row_count": int(len(address_df))},
        {"artifact": "native_revival_methodology_confirmed", "row_count": int(len(methodology_df))},
    ]
    pd.DataFrame(summary_rows).to_csv(model_dir / "native_revival_evidence_summary.csv", index=False)

    print("Native revival evidence registries written.")
    print()
    print(address_df.to_string(index=False))
    print()
    print(methodology_df.to_string(index=False))


if __name__ == "__main__":
    main()
