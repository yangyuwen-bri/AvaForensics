# Protocol Decay Lab

This folder contains the research workflow behind the current `dataset v2` and `model v2` product build.

## Goal

Build a more product-oriented Avalanche protocol lifecycle stack that can support the product with:

- a clean AVAX core universe
- a forward-looking Stage 1 early-risk model
- a Stage 2 lifecycle interpretation layer
- an evidence layer for relocation, native revival, and AVAX-side weakness

The core prediction task shifts from a coarse current-snapshot label to a forward-looking event:

- Observe only the first 90 days of a protocol's TVL curve.
- Predict whether the protocol enters a structural decay state within the next 365 days.
- Structural decay means at least 30 consecutive days below a dynamic threshold:
  - `max($10,000, 5% of early-window peak TVL)`
  - and no meaningful recovery afterwards inside the prediction horizon.

## What Is New Here

Compared with the current project baseline, this lab adds:

- Activation-aligned windows that trim pre-launch zero padding.
- Robust log-domain features to avoid ratio explosions.
- A product-style structural-decay event label instead of a current-TVL snapshot label.
- A clean `dataset v2` built around AVAX core-eligible protocols.
- A `Stage 2` lifecycle interpretation layer for:
  - terminal decay
  - multichain relocation
  - native revival / boundary
  - AVAX-side weakness
- An evidence layer built from:
  - address mapping
  - adapter methodology recovery
  - Avalanche activity validation

## Run

From the repo root:

```bash
python3 experiments/protocol_decay_lab/src/rebuild_dataset_v2.py
python3 experiments/protocol_decay_lab/src/protocol_decay_lab.py
python3 experiments/protocol_decay_lab/src/product_readiness_validation.py
python3 experiments/protocol_decay_lab/src/decay_mode_experiment.py
python3 experiments/protocol_decay_lab/src/retrain_v2_models.py
python3 experiments/protocol_decay_lab/src/protocol_address_evidence_experiment.py
python3 experiments/protocol_decay_lab/src/validate_avalanche_activity.py
python3 experiments/protocol_decay_lab/src/native_revival_evidence_experiment.py
python3 experiments/protocol_decay_lab/src/avax_side_decay_evidence_experiment.py
python3 experiments/protocol_decay_lab/src/build_product_schema_v2.py
```

## Outputs

All outputs are written under `experiments/protocol_decay_lab/outputs/`.

The most important product-facing outputs are:

- `outputs/dataset_v2/dataset_v2_diagnostics.json`
- `outputs/dataset_v2/registry_v2.csv`
- `outputs/dataset_v2/labels_v2.csv`
- `outputs/dataset_v2/features_summary_v2.csv`
- `outputs/dataset_v2/early_features_v2.csv`
- `outputs/model_v2/stage1_model_leaderboard_v2.csv`
- `outputs/model_v2/stage1_temporal_validation_v2.csv`
- `outputs/model_v2/retrain_v2_summary.json`
- `outputs/model_v2/two_stage_v2_summary.csv`
- `outputs/model_v2/product_schema_v2.csv`
- `outputs/model_v2/product_schema_v2_summary.csv`

Evidence-layer outputs include:

- `outputs/model_v2/protocol_address_evidence_summary.csv`
- `outputs/model_v2/adapter_address_confirmed.csv`
- `outputs/model_v2/adapter_methodology_confirmed.csv`
- `outputs/model_v2/native_revival_evidence_summary.csv`
- `outputs/model_v2/avax_side_decay_evidence_summary.csv`
- `outputs/model_v2/avalanche_activity_validation_summary.csv`

These outputs are the main bridge between the research track and the current product.

## Patent Angle

This folder can support invention drafting, but it is not a patent opinion.

Candidate claimable directions to explore further:

- structural decay event labeling for DeFi liquidity trajectories
- activation-aligned liquidity curve encoding
- archetype-gated risk scoring with uncertainty and data-quality controls
