# AvaForensics

> AI-driven protocol health scoring for the Avalanche ecosystem.

[![Build Games 2026](https://img.shields.io/badge/Avalanche-Build%20Games%202026-E84142?style=flat-square)](https://build.avax.network/build-games)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Protocols](https://img.shields.io/badge/Protocols%20Scored-422-00D4FF?style=flat-square)](#dataset)
[![Baseline AUC](https://img.shields.io/badge/Baseline%20AUC-0.738-2ED573?style=flat-square)](#model)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

## What It Is

AvaForensics is a protocol health explorer for Avalanche. It learns from the historical TVL behavior of 422 protocols and assigns a health score to each protocol based on its first 90 days of TVL behavior.

The current MVP includes:

- A runnable Streamlit product UI
- Baseline health scoring across 422 Avalanche protocols
- Single-protocol inspection with TVL history and risk signals
- Leaderboard and ecosystem-wide browsing
- Live refresh from DeFiLlama for the selected protocol
- Avalanche official on-chain enrichment through AvaCloud Glacier Data API when available

## Why It Exists

Most Web3 tools answer contract-security questions such as "does this contract have an obvious backdoor?" AvaForensics answers a different question:

`Is this protocol behaving like projects that later decay to near-zero TVL?`

This makes it a protocol-health and decay-monitoring product, not a contract audit tool.

## MVP Snapshot

- `422` protocols scored from local research data
- `249 alive / 179 dead` labels in the current protocol registry snapshot
- `246 alive / 176 dead` protocols in the model-ready early-window dataset
- Baseline Random Forest trained on first-90-day TVL features
- Current cross-validated baseline: `AUC 0.738`, `Accuracy 0.709`
- Price enrichment coverage: `89` protocols
- Avalanche on-chain enrichment coverage: `13` protocols

## Product Surfaces

### 1. Protocol View

For any selected protocol, the UI shows:

- Health score and dead probability
- Historical TVL chart
- Top risk signals derived from early TVL behavior
- A contrast case from the opposite side of the label set
- Supporting context such as category, price coverage, and on-chain coverage

### 2. Leaderboard

The leaderboard lets users browse protocols by health score and current risk band.

### 3. Live Refresh

The live refresh layer supplements the baseline model with fresh data:

- Pulls the latest protocol TVL history from DeFiLlama
- Recomputes a refreshed score for the selected protocol
- Pulls live Avalanche chain summary data from AvaCloud Glacier when `GLACIER_API_KEY` is configured

The live refresh layer is intentionally separated from the baseline model. The baseline remains reproducible from the local research dataset, while live refresh acts as a monitoring overlay.

## Dataset

The current MVP is built on four main dataset layers:

### Protocol Registry

`avax_data/protocols_labeled.csv`

- Protocol name, slug, category
- Current TVL snapshot
- `avax_only` flag
- Current label: `alive` or `dead`

Current label rule:

- `alive`: current TVL `>= 10,000 USD`
- `dead`: current TVL `< 10,000 USD`

### Raw TVL Histories

`avax_data/tvl_{slug}.csv`

- One historical TVL series per protocol
- Used as the raw source for both model features and product charts

### Model Dataset

`avax_data/early_features.csv`

- Feature table built from the first 90 days of each protocol's TVL history
- Used to train the current baseline scoring model

### Enrichment Datasets

- `avax_data/combined_features.csv` for price/TVL divergence features
- `avax_data/onchain_features.csv` for Avalanche on-chain activity features

## Data Sources

- [DeFiLlama API](https://defillama.com/docs/api): protocol metadata and TVL history
- [AvaCloud Glacier Data API](https://developers.avacloud.io/data-api/overview): Avalanche C-Chain transaction, token, and address-level data
- CoinGecko API: token price history for price-side enrichment

## Model

The current baseline model is intentionally narrow:

- Model family: Random Forest
- Prediction target: whether a protocol later behaves like the low-TVL/dead set
- Input window: first 90 days of TVL history
- Validation: 5-fold stratified cross validation

Three of the most informative early signals are:

- TVL peak timing
- 90-day TVL retention
- Second-half ratio

This is a protocol-health baseline, not a full fraud, exploit, or rug-pull detection system.

## Avalanche Tech Used

Avalanche-specific usage in the current MVP:

- Avalanche protocol subset used as the core ecosystem focus
- AvaCloud Glacier Data API integrated for live on-chain summaries
- Avalanche contract and token addresses used to enrich selected protocol views

## Run Locally

```bash
git clone https://github.com/yangyuwen-bri/AvaForensics.git
cd AvaForensics
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open:

- `http://localhost:8501`

Optional environment setup for Avalanche live on-chain data:

```bash
cp .env.example .env
# add GLACIER_API_KEY to .env
```

Without `GLACIER_API_KEY`, the baseline product still works. Only the Avalanche live on-chain summary layer is unavailable.

Data pipeline commands now live under `scripts/`:

```bash
python scripts/fetch_tvl_history.py
python scripts/timeseries_features_model.py
python scripts/coingecko_price_features.py
python scripts/glacier_onchain_features.py
```

## Repository Map

```text
AvaForensics/
├── avaforensics/                 # MVP backend helpers
├── avax_data/                    # Local datasets and derived features
├── scripts/                      # Data ingestion and feature engineering scripts
│   ├── fetch_tvl_history.py
│   ├── timeseries_features_model.py
│   ├── coingecko_price_features.py
│   └── glacier_onchain_features.py
├── streamlit_app.py              # Product UI
├── docs/
│   ├── TECHNICAL_IMPLEMENTATION.md
│   └── SUBMISSION_CHECKLIST.md
├── requirements.txt
└── README.md
```

## Technical Notes

Technical implementation details are documented in [docs/TECHNICAL_IMPLEMENTATION.md](/Users/yuwen/AvaForensics/docs/TECHNICAL_IMPLEMENTATION.md).

A submission-oriented checklist is documented in [docs/SUBMISSION_CHECKLIST.md](/Users/yuwen/AvaForensics/docs/SUBMISSION_CHECKLIST.md).

Deployment instructions are documented in [docs/DEPLOYMENT.md](/Users/yuwen/AvaForensics/docs/DEPLOYMENT.md).

## Current Limitations

The current MVP is a strong first product version, but not yet a full production data platform.

- The protocol registry is derived primarily from DeFiLlama rather than a fully self-owned protocol registry
- The label definition is still coarse and based on current TVL thresholding
- Price enrichment covers only part of the protocol set
- Avalanche on-chain enrichment is currently a smaller subset
- Some feature distributions still need cleanup before production-grade risk scoring

## Next Direction

The next serious product phase is data infrastructure, not more UI polish:

- protocol registry cleanup
- better status and label definitions
- stronger Avalanche-native fact layer
- multi-source validation
- more reliable coverage for price and on-chain enrichment

## Build Games Context

Built for [Avalanche Build Games 2026](https://build.avax.network/build-games), Stage 2 MVP.

## License

MIT. See [LICENSE](/Users/yuwen/AvaForensics/LICENSE).
