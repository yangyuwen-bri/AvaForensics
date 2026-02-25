# AvaForensics 🛡️

> **AI-driven risk intelligence for the Avalanche ecosystem.**  
> Turning 170+ dead protocols into a living early warning system.

[![Build Games 2026](https://img.shields.io/badge/Avalanche-Build%20Games%202026-E84142?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyTDIgMjBoMjBMMTIgMnoiLz48L3N2Zz4=)](https://build.avax.network)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Data](https://img.shields.io/badge/Protocols%20Analyzed-422-00D4FF?style=flat-square)](#data)
[![Model AUC](https://img.shields.io/badge/Baseline%20AUC-0.725-2ED573?style=flat-square)](#model)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## The Problem

**90% of Web3 projects fail.** On Avalanche alone, we identified **170+ dead protocols** — projects like Blizz Finance ($1.3B peak TVL) and Yeti Finance ($740M peak TVL) that collapsed to zero.

Existing tools like RugCheck or GoPlus only scan smart contract code for backdoors. They can tell you *"does this contract have a mint function?"* — but not *"is this protocol dying?"*

**AvaForensics fills that gap.** By learning from the graveyard of dead protocols, we extract early warning signals that predict project failure months before it happens.

---

## Key Findings

After analyzing 422 Avalanche protocols (176 dead, 246 alive), three early signals consistently separate dying projects from survivors — detectable within the **first 90 days** of a protocol's life:

| Signal | Dead Projects | Alive Projects | Interpretation |
|--------|--------------|----------------|----------------|
| **TVL Peak Timing** | Day 3 (5th percentile) | Day 42 (42nd percentile) | Dying projects peak immediately after launch — classic pump-and-dump pattern |
| **90-Day TVL Retention** | 3.6% of peak | 56% of peak | Dead protocols bleed out fast; survivors hold value |
| **Second-Half Ratio** | 0.29x | 0.95x | In dying projects, the back half of their TVL history is only 29% of the front half |

All 13 extracted features pass statistical significance at **p < 0.01** (Mann-Whitney U test).

---

## Model Performance

Baseline Random Forest trained on TVL time-series features only:

```
Model: Random Forest (200 trees, balanced class weights)
Validation: 5-Fold Stratified Cross Validation
Samples: 422 protocols (41.7% dead)

AUC:      0.725 ± 0.075
Accuracy: 69.4%
```

> This is a **baseline** using only 90-day TVL curves. Token concentration and community signals are being added in the next phase.

---

## Data Pipeline

```
DeFiLlama API          Avalanche On-chain
     │                        │
     ▼                        ▼
avax_protocols.csv      (coming: token holders)
     │
     ▼
fetch_tvl_history.py   ← pulls TVL time-series for all 422 protocols
     │
     ▼
avax_data/
  ├── tvl_{slug}.csv         (individual time-series per protocol)
  ├── protocols_labeled.csv  (alive/dead labels)
  ├── features_summary.csv   (aggregate features, 422 rows × 14 cols)
  └── early_features.csv     (90-day early-window features for ML)
     │
     ▼
timeseries_features_model.py  ← feature extraction + Random Forest baseline
     │
     ▼
Health Score (0–100) + Risk Signals
```

**Data sources:**
- [DeFiLlama API](https://defillama.com/docs/api) — TVL history, protocol metadata (free, no key required)
- [Glacier API (AvaCloud)](https://glacier-api.avax.network/) — Avalanche C-Chain on-chain data
- CoinGecko API — token price history (free tier)

**Dead project labeling:** Protocols with current TVL < $10,000 are labeled `dead`. This covers 176 of 422 Avalanche protocols (41.7%).

---

## Project Structure

```
AvaForensics/
│
├── avax_data/                    # Raw and processed data
│   ├── tvl_{slug}.csv            # TVL time-series (one file per protocol)
│   ├── protocols_labeled.csv     # 422 protocols with dead/alive labels
│   ├── features_summary.csv      # Extracted features
│   └── early_features.csv        # 90-day window features for ML
│
├── explore_bq_avalanche.py       # Step 1: data source exploration
├── fetch_tvl_history.py          # Step 2: TVL history fetcher
├── timeseries_features_model.py  # Step 3: feature extraction + baseline model
├── coingecko_price_features.py   # Step 4: price data + divergence signals
├── glacier_onchain_features.py   # Step 5: Avalanche Glacier Data API integration
│
├── docs/
│   └── Web3风险预测工具市场调研报告.docx  # Market research report
│
└── README.md
```

---

## Quickstart

```bash
# Clone
git clone https://github.com/yangyuwen-bri/AvaForensics.git
cd AvaForensics

# Install dependencies
pip install pandas numpy scipy scikit-learn requests matplotlib

# Step 1: Fetch Avalanche protocol list
python fetch_tvl_history.py
# → Creates avax_data/ with TVL time-series for 422 protocols
# → Takes ~10 minutes (respects API rate limits)

# Step 2: Extract features + run baseline model
python timeseries_features_model.py
# → Outputs early_features.csv
# → Reports AUC and feature importance ranking

# Step 3: Avalanche on-chain data (requires Glacier API key)
# Get free API key at https://avacloud.io
export GLACIER_API_KEY=your_key_here
python glacier_onchain_features.py
# → Queries Avalanche C-Chain via official Glacier Data API
# → Extracts real-time on-chain activity features
```

---

## Roadmap

**Phase 1 — Data Foundation** ✅ *Complete*
- [x] Fetch and label 422 Avalanche protocols
- [x] Extract 13 TVL time-series features
- [x] Baseline model (AUC 0.725)
- [x] Identify top 3 early warning signals

**Phase 2 — Signal Enrichment** 🔄 *In Progress*
- [ ] Token holder concentration (top-10 wallet % via Glacier API)
- [ ] Price/TVL divergence signal (CoinGecko + DEX swap events)
- [ ] On-chain activity decay (daily active addresses)

**Phase 3 — Product** 📋 *Planned*
- [ ] REST API: `GET /health/{protocol_slug}` → health score + risk breakdown
- [ ] Live dashboard: real-time monitoring of all Avalanche protocols
- [ ] Alert system: email/webhook when a protocol enters danger zone
- [ ] Public leaderboard: ecosystem-wide health ranking

**Phase 4 — Expansion** 📋 *Planned*
- [ ] Historical backtesting: "Would we have caught Blizz Finance in time?"
- [ ] Early-stage scoring: contract + token distribution analysis for new launches
- [ ] Multi-chain support (Avalanche L1s / subnets first)

---

## Why Avalanche

Avalanche has one of the most transparent on-chain ecosystems in crypto — full TVL history accessible via free APIs, rich subnet activity data, and a clearly defined set of native protocols. This transparency is exactly what makes forensic analysis possible.

AvaForensics is built *specifically* for Avalanche, not a generic multi-chain tool. The dead protocol dataset (170+ projects) is Avalanche-native, the model is trained on Avalanche patterns, and the roadmap is built around Avalanche's ecosystem data APIs.

---

## Built For

**[Avalanche Build Games 2026](https://build.avax.network)** — a 6-week builder competition to accelerate the next generation of crypto entrepreneurs.

**Author:** [@yangyuwen-bri](https://github.com/yangyuwen-bri)  
**Stage:** Idea Submission (Feb 25, 2026)

---

## License

MIT — see [LICENSE](LICENSE) for details.
