"""Backend helpers for the AvaForensics Streamlit MVP."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "avax_data"
V2_OUTPUT_DIR = BASE_DIR / "experiments" / "protocol_decay_lab" / "outputs"
V2_DATASET_DIR = V2_OUTPUT_DIR / "dataset_v2"
V2_MODEL_DIR = V2_OUTPUT_DIR / "model_v2"
DEFILLAMA_PROTOCOL_URL = "https://api.llama.fi/protocol/{slug}"
GLACIER_BASE_URL = "https://glacier-api.avax.network/v1"
AVALANCHE_CHAIN_ID = "43114"
REQUEST_TIMEOUT_SECONDS = 20
STAGE1_RANDOM_STATE = 42
SAFE_TVL_FLOOR = 1_000.0
EARLY_WINDOW_DAYS = 90
MIN_EARLY_POINTS = 30
DISPLAY_GAP_DAYS = 14

load_dotenv(BASE_DIR / ".env")

MODEL_EXCLUDE_COLUMNS = {"slug", "label", "name", "category"}
STAGE1_EXCLUDE_COLUMNS = {
    "slug",
    "name",
    "category",
    "current_status",
    "structural_decay_label_v2",
    "structural_decay_target_v2",
    "history_source_mode",
    "core_universe_status",
    "decay_mode_v2",
    "stage1_terminal_target",
    "launch_date",
    "old_label",
    "target",
    "label_status",
    "target_known",
    "risk_band",
    "reason_1",
    "reason_2",
    "reason_3",
}

SIGNAL_SPECS = [
    {
        "feature": "peak_day_frac",
        "label": "Peak Timing",
        "unit": "day",
        "window_days": 90,
        "decimals": 0,
        "direction": "high",
        "description": "Protocols that peak too early often resemble fragile early launches.",
    },
    {
        "feature": "retention_ratio",
        "label": "90-Day Retention",
        "unit": "percent",
        "window_days": None,
        "decimals": 1,
        "direction": "high",
        "description": "Stronger protocols retain more of their early AVAX-side liquidity.",
    },
    {
        "feature": "active_day_frac",
        "label": "Active-Day Density",
        "unit": "percent",
        "direction": "high",
        "decimals": 1,
        "description": "A healthier early curve spends more days above the minimum active TVL threshold.",
    },
    {
        "feature": "drawdown_ratio",
        "label": "Early Drawdown",
        "unit": "percent",
        "direction": "low",
        "decimals": 1,
        "description": "Lower early drawdown means the protocol did not collapse immediately after its first peak.",
    },
    {
        "feature": "rolling7_end_vs_peak",
        "label": "Late Window Strength",
        "unit": "ratio",
        "window_days": None,
        "decimals": 2,
        "direction": "high",
        "description": "The final stretch of the early window should still hold a meaningful fraction of the peak.",
    },
    {
        "feature": "log_peak_tvl",
        "label": "Early Peak TVL",
        "unit": "usd_log",
        "window_days": None,
        "decimals": 0,
        "direction": "high",
        "description": "Larger early capital formation correlates with longer-term survival.",
    },
]

RISK_BANDS = [
    (75, "Resilient Start"),
    (50, "Mixed Start"),
    (0, "Fragile Start"),
]

LIVE_SIGNAL_SPECS = [
    {
        "feature": "drawdown_from_peak",
        "label": "Drawdown From Peak",
        "unit": "percent",
        "direction": "low",
        "description": "Lower drawdown means the protocol is still holding more of its peak value.",
    },
    {
        "feature": "tvl_30d_change",
        "label": "30-Day TVL Change",
        "unit": "percent_signed",
        "direction": "high",
        "description": "Recent TVL momentum is one of the clearest live warnings of decay.",
    },
    {
        "feature": "tvl_90d_change",
        "label": "90-Day TVL Change",
        "unit": "percent_signed",
        "direction": "high",
        "description": "A 90-day TVL trend shows whether the protocol is stabilizing or bleeding out.",
    },
]


def _read_csv(name: str, **kwargs) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required dataset: {path}")
    return pd.read_csv(path, **kwargs)


def _optional_csv(name: str, **kwargs) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _read_v2_csv(name: str, **kwargs) -> pd.DataFrame:
    path = V2_DATASET_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required v2 dataset: {path}")
    return pd.read_csv(path, **kwargs)


def _read_v2_model_csv(name: str, **kwargs) -> pd.DataFrame:
    path = V2_MODEL_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required v2 model output: {path}")
    return pd.read_csv(path, **kwargs)


def _feature_columns(df: pd.DataFrame) -> List[str]:
    return [column for column in df.columns if column not in MODEL_EXCLUDE_COLUMNS]


def _stage1_feature_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for column in df.columns:
        if column in STAGE1_EXCLUDE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            cols.append(column)
    return cols


def _risk_band(score: float) -> str:
    for threshold, label in RISK_BANDS:
        if score >= threshold:
            return label
    return "Fragile Start"


def _currency(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def _percent(value: Optional[float], decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.{decimals}f}%"


def _signed_percent(value: Optional[float], decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:+.{decimals}f}%"


def _format_signal_value(spec: Dict[str, object], value: Optional[float]) -> str:
    unit = spec["unit"]
    if value is None or pd.isna(value):
        return "N/A"
    numeric_value = float(value)
    if unit == "day":
        return f"Day {round(numeric_value * int(spec['window_days']))}"
    if unit == "percent":
        return _percent(numeric_value, int(spec.get("decimals", 1)))
    if unit == "percent_signed":
        decimals = int(spec.get("decimals", 1))
        return _signed_percent(numeric_value, decimals)
    if unit == "ratio":
        return f"{numeric_value:.{int(spec['decimals'])}f}x"
    if unit == "usd_log":
        actual_value = np.expm1(numeric_value)
        return _currency(actual_value)
    if unit == "integer":
        return f"{int(round(numeric_value)):,}"
    return f"{numeric_value:.2f}"


def _format_delta(direction: str, row_value: float, alive_median: float, dead_median: float) -> str:
    closer_to_dead = abs(row_value - dead_median) <= abs(row_value - alive_median)
    if direction == "high":
        gap = row_value - alive_median
    else:
        gap = alive_median - row_value
    if closer_to_dead:
        return f"Closer to the terminal-decay median than the lower-risk median ({gap:.2f} vs lower-risk baseline)."
    return f"Stronger than the lower-risk median by {gap:.2f}."


def _humanize_status(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    return str(value).replace("_", " ").title()


def _display_mode_label(value: object) -> str:
    mapping = {
        "terminal_global_decay": "Terminal Decay",
        "multichain_relocation": "Likely Cross-Chain Relocation",
        "native_revival_or_threshold_boundary": "Native Revival or Boundary Case",
        "avax_side_decay_but_globally_alive": "AVAX-Side Weakness",
        "resilient_or_unproven": "No Strong Terminal Signal",
        "not_core_eligible": "Not Core Eligible",
        "data_incomplete": "Data Incomplete",
    }
    if value is None or pd.isna(value):
        return "Unknown"
    return mapping.get(str(value), _humanize_status(value))


def _display_evidence_label(value: object) -> str:
    mapping = {
        "onchain_supported": "On-Chain Supported",
        "weak_onchain_support": "Weak On-Chain Support",
        "address_registered": "Address Registered",
        "methodology_backed": "Methodology Backed",
        "threshold_only": "Threshold Only",
        "inferred_only": "Inference Only",
        "address_mismatch": "Address Mismatch",
        "address_gap": "Address Gap",
        "model_only": "Model Only",
        "not_eligible": "Not Eligible",
    }
    if value is None or pd.isna(value):
        return "Unknown"
    return mapping.get(str(value), _humanize_status(value))


def _evidence_caveat(value: object) -> str:
    mapping = {
        "onchain_supported": "Recent Avalanche activity supports this interpretation directly.",
        "weak_onchain_support": "Short-window Avalanche activity was observed, but the support is still partial.",
        "address_registered": "Avalanche contract mapping is confirmed and live code is present on C-Chain.",
        "methodology_backed": "This interpretation is supported by chain-specific adapter or payload methodology for Avalanche.",
        "threshold_only": "This is mostly a boundary reading driven by threshold status rather than stronger activity evidence.",
        "inferred_only": "This interpretation is still primarily model- and TVL-driven.",
        "address_mismatch": "The current address candidate does not map cleanly to Avalanche live code.",
        "address_gap": "Avalanche contract mapping is still missing.",
        "model_only": "This interpretation currently comes from the model layer only.",
        "not_eligible": "This protocol is outside the AVAX core scoring universe.",
    }
    if value is None or pd.isna(value):
        return "Evidence is currently limited."
    return mapping.get(str(value), "Evidence is currently limited.")


def _footprint_summary(row: pd.Series) -> str:
    avax_share = row.get("current_avax_share")
    current_avax_tvl = row.get("current_avax_core_tvl")
    current_status = str(row.get("current_status") or "")
    if pd.notna(avax_share) and float(avax_share) <= 0.01:
        return "AVAX footprint is now very thin relative to global core TVL."
    if current_status == "low_tvl_on_avax":
        return "Current AVAX-side liquidity is weak even if the protocol remains active elsewhere."
    if pd.notna(current_avax_tvl) and float(current_avax_tvl) >= 10_000_000:
        return "Current AVAX footprint is still meaningful in absolute terms."
    return "Current AVAX footprint is active, but should be read alongside cross-chain context."


def _current_avax_posture(row: pd.Series) -> str:
    avax_share = row.get("current_avax_share")
    current_status = str(row.get("current_status") or "")
    current_avax_tvl = row.get("current_avax_core_tvl")
    if current_status == "low_tvl_on_avax":
        return "Weak on Avalanche"
    if pd.notna(avax_share) and float(avax_share) < 0.01:
        return "Thin AVAX Presence"
    if pd.notna(current_avax_tvl) and float(current_avax_tvl) >= 10_000_000:
        return "Meaningful AVAX Presence"
    return "Active on Avalanche"


def _build_lifecycle_interpretation(row: pd.Series) -> Dict[str, str]:
    mode_raw = row.get("decay_mode_v2")
    evidence_raw = row.get("evidence_level")
    mode_label = _display_mode_label(mode_raw)
    evidence_label = _display_evidence_label(evidence_raw)
    evidence_summary = row.get("evidence_summary")
    if evidence_summary is None or pd.isna(evidence_summary) or not str(evidence_summary).strip():
        evidence_summary = "Evidence for this interpretation is currently limited."
    return {
        "mode": mode_label,
        "evidence_level": evidence_label,
        "summary": str(evidence_summary),
        "caveat": _evidence_caveat(evidence_raw),
        "footprint_summary": _footprint_summary(row),
    }


def _build_analyst_summary(row: pd.Series) -> Dict[str, object]:
    risk = float(row.get("dead_probability") or 0.0)
    mode_label = _display_mode_label(row.get("product_decay_mode") or row.get("decay_mode_v2"))
    evidence_label = _display_evidence_label(row.get("evidence_level"))
    posture = _current_avax_posture(row)
    if risk < 0.15:
        early_line = "Early AVAX trajectory looks resilient."
    elif risk < 0.4:
        early_line = "Early AVAX trajectory looks mixed."
    else:
        early_line = "Early AVAX trajectory carries elevated terminal-decay risk."

    if posture == "Meaningful AVAX Presence":
        footprint_line = "Current AVAX footprint is still meaningful in absolute terms."
    elif posture == "Thin AVAX Presence":
        footprint_line = "Current AVAX footprint is active but very thin relative to global core TVL."
    elif posture == "Weak on Avalanche":
        footprint_line = "Current AVAX-side liquidity is weak even if the broader protocol remains active elsewhere."
    else:
        footprint_line = "Current AVAX footprint remains active and above threshold."

    summary = f"{early_line} Lifecycle interpretation points to {mode_label.lower()}, supported at the {evidence_label.lower()} level."
    reasons = [
        early_line,
        footprint_line,
        str(row.get("evidence_summary") or "Lifecycle interpretation evidence is currently limited."),
    ]
    return {"summary": summary, "reasons": reasons}


def _find_activation_start(tvl: np.ndarray) -> int:
    if len(tvl) == 0:
        return 0
    for index in range(max(1, len(tvl) - 2)):
        window = tvl[index:index + 3]
        if len(window) >= 2 and int((window > 0).sum()) >= 2:
            if float(window[0]) > 0.0:
                return index
            positive_offsets = np.flatnonzero(window > 0)
            if len(positive_offsets) > 0:
                return index + int(positive_offsets[0])
            return index
    return 0


def _trim_to_activation(df_tvl: pd.DataFrame) -> pd.DataFrame:
    df = df_tvl.sort_values("date").reset_index(drop=True).copy()
    tvl = df["tvl"].fillna(0.0).to_numpy(dtype=float)
    start_idx = _find_activation_start(tvl)
    return df.iloc[start_idx:].reset_index(drop=True).copy()


def _break_long_gaps(df_tvl: pd.DataFrame, gap_days: int = DISPLAY_GAP_DAYS) -> tuple[pd.DataFrame, int]:
    if df_tvl.empty or len(df_tvl) < 2:
        return df_tvl.copy(), 0

    df = df_tvl.sort_values("date").reset_index(drop=True).copy()
    rows: List[Dict[str, object]] = []
    gap_breaks = 0

    for index, row in df.iterrows():
        if index > 0:
            previous_date = pd.to_datetime(df.iloc[index - 1]["date"])
            current_date = pd.to_datetime(row["date"])
            if (current_date - previous_date).days > gap_days:
                gap_breaks += 1
                rows.append(
                    {
                        "date": current_date - pd.Timedelta(seconds=1),
                        "tvl": np.nan,
                        "day_index": np.nan,
                    }
                )
        rows.append(
            {
                "date": pd.to_datetime(row["date"]),
                "tvl": row["tvl"],
                "day_index": row["day_index"] if "day_index" in df.columns else np.nan,
            }
        )

    return pd.DataFrame(rows), gap_breaks


def _prepare_display_history(df_tvl: pd.DataFrame) -> Dict[str, object]:
    if df_tvl.empty:
        return {
            "history": df_tvl.copy(),
            "activation_start": None,
            "hidden_pre_activation_points": 0,
            "gap_break_count": 0,
            "raw_points": 0,
        }

    raw = df_tvl.sort_values("date").reset_index(drop=True).copy()
    adjusted = _trim_to_activation(raw)
    display_history, gap_break_count = _break_long_gaps(adjusted)

    return {
        "history": display_history,
        "activation_start": adjusted["date"].iloc[0] if not adjusted.empty else raw["date"].iloc[0],
        "hidden_pre_activation_points": int(max(len(raw) - len(adjusted), 0)),
        "gap_break_count": int(gap_break_count),
        "raw_points": int(len(raw)),
    }


def _get_early_window(df_tvl: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df_tvl.empty:
        return None
    cutoff = df_tvl["date"].iloc[0] + pd.Timedelta(days=EARLY_WINDOW_DAYS)
    early = df_tvl[df_tvl["date"] <= cutoff].copy()
    if len(early) < MIN_EARLY_POINTS:
        return None
    return early


def _extract_stage1_v2_features(df_tvl: pd.DataFrame) -> Optional[Dict[str, float]]:
    trimmed = _trim_to_activation(df_tvl)
    early = _get_early_window(trimmed)
    if early is None:
        return None

    tvl = early["tvl"].fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    log_tvl = np.log1p(tvl)
    n_points = len(tvl)
    peak_idx = int(np.argmax(tvl))
    peak_tvl = max(float(tvl.max()), 1.0)
    end_tvl = max(float(tvl[-1]), 0.0)
    start_tvl = max(float(tvl[0]), 0.0)
    median_tvl = float(np.median(tvl))
    safe_tvl = np.maximum(tvl, SAFE_TVL_FLOOR)
    log_returns = np.diff(np.log1p(safe_tvl))
    post_peak_log = log_tvl[peak_idx:]
    rolling_7 = pd.Series(log_tvl).rolling(7, min_periods=1).mean().to_numpy(dtype=float)
    total_span_days = int((trimmed["date"].iloc[-1] - trimmed["date"].iloc[0]).days)
    early_span_days = int((early["date"].iloc[-1] - early["date"].iloc[0]).days)
    early_peak_day = int((early["date"].iloc[peak_idx] - early["date"].iloc[0]).days)
    slope_total, _, _, _, _ = stats.linregress(np.arange(n_points), log_tvl) if n_points >= 3 else (0.0, 0, 0, 0, 0)
    slope_post, _, _, _, _ = stats.linregress(np.arange(len(post_peak_log)), post_peak_log) if len(post_peak_log) >= 3 else (0.0, 0, 0, 0, 0)
    stable_band = np.mean(np.abs(log_tvl - np.median(log_tvl)) <= (np.std(log_tvl) + 1e-6))
    active_day_frac = float(np.mean(tvl >= 10_000.0))
    nonzero_day_frac = float(np.mean(tvl > 0.0))

    return {
        "launch_trim_days": float(max(len(df_tvl) - len(trimmed), 0)),
        "lifecycle_days": float(total_span_days),
        "early_window_days_observed": float(early_span_days),
        "data_points": float(n_points),
        "log_start_tvl": float(np.log1p(start_tvl)),
        "log_peak_tvl": float(np.log1p(peak_tvl)),
        "log_end_tvl": float(np.log1p(end_tvl)),
        "log_median_tvl": float(np.log1p(median_tvl)),
        "retention_ratio": float((end_tvl + SAFE_TVL_FLOOR) / (peak_tvl + SAFE_TVL_FLOOR)),
        "drawdown_ratio": float((peak_tvl - end_tvl) / peak_tvl),
        "peak_day_frac": float(peak_idx / max(n_points - 1, 1)),
        "peak_day_ratio_by_span": float(early_peak_day / max(EARLY_WINDOW_DAYS, 1)),
        "auc_norm": float(log_tvl.mean() / max(log_tvl.max(), 1e-6)),
        "slope_total_log": float(slope_total),
        "slope_post_log": float(slope_post),
        "vol_log_ret": float(np.std(log_returns)) if len(log_returns) > 1 else 0.0,
        "mean_log_ret": float(np.mean(log_returns)) if len(log_returns) > 0 else 0.0,
        "downside_log_ret": float(np.std(np.minimum(log_returns, 0.0))) if len(log_returns) > 1 else 0.0,
        "crash_15_count": float((log_returns < np.log(0.85)).sum()) if len(log_returns) > 0 else 0.0,
        "rebound_15_count": float((log_returns > np.log(1.15)).sum()) if len(log_returns) > 0 else 0.0,
        "nonzero_day_frac": nonzero_day_frac,
        "active_day_frac": active_day_frac,
        "stable_band_frac": float(stable_band),
        "rolling7_end_vs_peak": float((rolling_7[-1] + 1e-6) / (rolling_7.max() + 1e-6)),
        "peak_to_median_log_gap": float(np.log1p(peak_tvl) - np.median(log_tvl)),
    }


def _stage1_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=STAGE1_RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _signal_risk_score(
    row_value: Optional[float],
    alive_median: Optional[float],
    dead_median: Optional[float],
    direction: str,
) -> float:
    if row_value is None or pd.isna(row_value):
        return 0.5
    if alive_median is None or dead_median is None or pd.isna(alive_median) or pd.isna(dead_median):
        return 0.5
    row_value = float(row_value)
    alive_median = float(alive_median)
    dead_median = float(dead_median)
    span = max(abs(alive_median - dead_median), 1e-9)
    if direction == "high":
        position = (alive_median - row_value) / span
    else:
        position = (row_value - alive_median) / span
    return float(np.clip((position + 1) / 2, 0.0, 1.0))


def _load_tvl_history(slug: str) -> pd.DataFrame:
    v2_history_path = V2_DATASET_DIR / "observations" / f"{slug}.csv"
    if v2_history_path.exists():
        history = pd.read_csv(v2_history_path, parse_dates=["date"])
        if "avax_core_tvl" in history.columns:
            history = history[["date", "avax_core_tvl"]].rename(columns={"avax_core_tvl": "tvl"})
            history = history.dropna(subset=["tvl"]).copy()
            history["tvl"] = pd.to_numeric(history["tvl"], errors="coerce").fillna(0.0)
            history = history.sort_values("date").reset_index(drop=True)
            history["day_index"] = np.arange(len(history))
            return history

    history_path = DATA_DIR / f"tvl_{slug}.csv"
    if history_path.exists():
        history = pd.read_csv(history_path, parse_dates=["date"])
        history["date"] = pd.to_datetime(history["date"])
        history["tvl"] = pd.to_numeric(history["tvl"], errors="coerce").fillna(0.0)
        history["day_index"] = np.arange(len(history))
        return history

    protocol_payload = _request_json(DEFILLAMA_PROTOCOL_URL.format(slug=slug))
    if not protocol_payload:
        return pd.DataFrame(columns=["date", "tvl", "day_index"])
    return _extract_tvl_history_from_protocol(protocol_payload)


def _request_json(url: str, headers: Optional[Dict[str, str]] = None) -> Optional[Dict[str, object]]:
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code != 200:
            return None
        return response.json()
    except requests.RequestException:
        return None


def _extract_tvl_history_from_protocol(protocol_payload: Dict[str, object]) -> pd.DataFrame:
    chain_tvls = protocol_payload.get("chainTvls", {}) or {}
    records = None

    avalanche_history = chain_tvls.get("Avalanche")
    if isinstance(avalanche_history, dict) and avalanche_history.get("tvl"):
        records = avalanche_history["tvl"]

    if records is None:
        records = []

    if not records:
        return pd.DataFrame(columns=["date", "tvl", "day_index"])

    history = pd.DataFrame(records)
    history = history.rename(columns={"totalLiquidityUSD": "tvl"})
    if "tvl" not in history.columns or "date" not in history.columns:
        return pd.DataFrame(columns=["date", "tvl", "day_index"])

    history["date"] = pd.to_datetime(history["date"], unit="s", errors="coerce")
    history["tvl"] = pd.to_numeric(history["tvl"], errors="coerce").fillna(0.0)
    history = history.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    history["day_index"] = np.arange(len(history))
    return history


def _extract_early_features(df_tvl: pd.DataFrame, window_days: int = 90, min_data_points: int = 30) -> Optional[Dict[str, float]]:
    df = df_tvl.sort_values("date").reset_index(drop=True)
    if len(df) < min_data_points:
        return None

    start_date = df["date"].iloc[0]
    cutoff = start_date + pd.Timedelta(days=window_days)
    early = df[df["date"] <= cutoff].copy()
    if len(early) < 10:
        return None

    tvl = early["tvl"].astype(float).values
    n = len(tvl)
    t = np.arange(n)

    peak_val = float(tvl.max()) if len(tvl) else 0.0
    peak_idx = int(tvl.argmax()) if len(tvl) else 0
    final_val = float(tvl[-1]) if len(tvl) else 0.0
    first_val = float(tvl[0]) if len(tvl) and tvl[0] > 0 else 1.0
    retention_at_end = final_val / peak_val if peak_val > 0 else 0.0

    post_peak = tvl[peak_idx:]
    half_peak = peak_val * 0.5
    steps_to_half = next((i for i, value in enumerate(post_peak) if value <= half_peak), len(post_peak))
    half_life_frac = steps_to_half / (n - peak_idx + 1)

    if len(post_peak) >= 3:
        slope_post, _, _, _, _ = stats.linregress(np.arange(len(post_peak)), post_peak / (peak_val + 1))
    else:
        slope_post = 0.0

    slope_total, _, r_value, _, _ = stats.linregress(t, tvl / (peak_val + 1))
    mid = n // 2
    first_half_mean = tvl[:mid].mean() if mid else float(tvl.mean())
    second_half_mean = tvl[mid:].mean() if mid else float(tvl.mean())
    half_ratio = second_half_mean / (first_half_mean + 1)

    pct_changes = np.diff(tvl) / (tvl[:-1] + 1)
    volatility = float(pct_changes.std()) if len(pct_changes) > 1 else 0.0
    mean_pct_change = float(pct_changes.mean()) if len(pct_changes) > 1 else 0.0
    crash_count = int((pct_changes < -0.3).sum())

    growth_multiple = peak_val / (first_val + 1)

    return {
        "peak_day_frac": peak_idx / n,
        "retention_at_end": retention_at_end,
        "half_life_frac": half_life_frac,
        "slope_post_peak": slope_post,
        "slope_total": slope_total,
        "half_ratio": half_ratio,
        "r_squared": r_value ** 2,
        "volatility": volatility,
        "mean_pct_change": mean_pct_change,
        "crash_count": crash_count,
        "rise_length_frac": peak_idx / n,
        "log_growth_multiple": np.log1p(growth_multiple),
        "log_peak_tvl": np.log1p(peak_val),
        "data_points": n,
    }


def _compute_live_tvl_metrics(df_tvl: pd.DataFrame) -> Dict[str, float]:
    if df_tvl.empty:
        return {
            "peak_tvl": np.nan,
            "current_tvl": np.nan,
            "drawdown_from_peak": np.nan,
            "tvl_30d_change": np.nan,
            "tvl_90d_change": np.nan,
            "lifespan_days": np.nan,
            "consecutive_decline_days": np.nan,
            "data_points": 0,
        }

    tvl = df_tvl["tvl"].astype(float).values
    dates = pd.to_datetime(df_tvl["date"])
    peak_tvl = float(tvl.max()) if len(tvl) else np.nan
    current_tvl = float(tvl[-1]) if len(tvl) else np.nan
    drawdown = (peak_tvl - current_tvl) / peak_tvl if peak_tvl and peak_tvl > 0 else np.nan
    lifespan_days = int((dates.iloc[-1] - dates.iloc[0]).days) if len(dates) > 1 else 0

    recent_30 = df_tvl[df_tvl["date"] >= dates.iloc[-1] - pd.Timedelta(days=30)]
    recent_90 = df_tvl[df_tvl["date"] >= dates.iloc[-1] - pd.Timedelta(days=90)]
    tvl_30d_change = (
        (recent_30["tvl"].iloc[-1] - recent_30["tvl"].iloc[0]) / (recent_30["tvl"].iloc[0] + 1)
        if len(recent_30) >= 2
        else np.nan
    )
    tvl_90d_change = (
        (recent_90["tvl"].iloc[-1] - recent_90["tvl"].iloc[0]) / (recent_90["tvl"].iloc[0] + 1)
        if len(recent_90) >= 2
        else np.nan
    )

    peak_idx = int(tvl.argmax()) if len(tvl) else 0
    post_peak = tvl[peak_idx:]
    consecutive_decline = int(sum(1 for index in range(1, len(post_peak)) if post_peak[index] < post_peak[index - 1]))

    return {
        "peak_tvl": peak_tvl,
        "current_tvl": current_tvl,
        "drawdown_from_peak": drawdown,
        "tvl_30d_change": tvl_30d_change,
        "tvl_90d_change": tvl_90d_change,
        "lifespan_days": lifespan_days,
        "consecutive_decline_days": consecutive_decline,
        "data_points": int(len(df_tvl)),
    }


def _extract_contract_address(protocol_payload: Dict[str, object]) -> Optional[str]:
    address = protocol_payload.get("address", "")
    if address:
        for part in str(address).split(","):
            chunk = part.strip()
            if "avax:" in chunk.lower():
                return chunk.split(":")[-1].strip()

    contracts = protocol_payload.get("contracts") or {}
    if isinstance(contracts, dict):
        avax_contracts = contracts.get("avax")
        if isinstance(avax_contracts, list) and avax_contracts:
            return str(avax_contracts[0])
        if isinstance(avax_contracts, str):
            return avax_contracts

    chain_tvls = protocol_payload.get("chainTvls", {}) or {}
    for chain_key, chain_value in chain_tvls.items():
        if "avalanche" not in chain_key.lower() or not isinstance(chain_value, dict):
            continue
        sub_contracts = chain_value.get("contracts")
        if isinstance(sub_contracts, list) and sub_contracts:
            return str(sub_contracts[0])
        if isinstance(sub_contracts, str):
            return sub_contracts
    return None


def _glacier_headers() -> Optional[Dict[str, str]]:
    api_key = os.environ.get("GLACIER_API_KEY", "")
    if not api_key:
        try:
            import streamlit as st

            api_key = str(st.secrets.get("GLACIER_API_KEY", ""))
        except Exception:
            api_key = ""
    if not api_key:
        return None
    return {
        "Content-Type": "application/json",
        "x-glacier-api-key": api_key,
    }


def _fetch_glacier_summary(address: Optional[str]) -> Dict[str, object]:
    if not address:
        return {"available": False, "reason": "No Avalanche contract or token address found."}

    headers = _glacier_headers()
    if not headers:
        return {"available": False, "reason": "GLACIER_API_KEY is not configured."}

    metadata = _request_json(f"{GLACIER_BASE_URL}/chains/{AVALANCHE_CHAIN_ID}/addresses/{address}", headers=headers)
    native_balance = _request_json(
        f"{GLACIER_BASE_URL}/chains/{AVALANCHE_CHAIN_ID}/addresses/{address}/balances:getNative?currency=usd",
        headers=headers,
    )
    transfers = _request_json(
        f"{GLACIER_BASE_URL}/chains/{AVALANCHE_CHAIN_ID}/tokens/{address}/transfers?pageSize=25",
        headers=headers,
    )
    transactions = _request_json(
        f"{GLACIER_BASE_URL}/chains/{AVALANCHE_CHAIN_ID}/addresses/{address}/transactions?pageSize=25",
        headers=headers,
    )

    transfer_rows = transfers.get("transfers", []) if isinstance(transfers, dict) else []
    transaction_rows = transactions.get("transactions", []) if isinstance(transactions, dict) else []
    activity_rows = transfer_rows or transaction_rows

    last_activity = None
    unique_wallets = set()
    for row in activity_rows:
        timestamp = row.get("blockTimestamp")
        if timestamp is None and "nativeTransaction" in row:
            timestamp = row["nativeTransaction"].get("blockTimestamp")
        if timestamp:
            last_activity = max(last_activity or 0, int(timestamp))
        for endpoint in ("from", "to"):
            endpoint_value = row.get(endpoint)
            if endpoint_value is None and "nativeTransaction" in row:
                endpoint_value = row["nativeTransaction"].get(endpoint)
            if isinstance(endpoint_value, dict) and endpoint_value.get("address"):
                wallet = str(endpoint_value["address"])
                if wallet.lower() != address.lower():
                    unique_wallets.add(wallet)

    native_token_balance = ((native_balance or {}).get("nativeTokenBalance") or {})
    decimals = int(native_token_balance.get("decimals", 18) or 18)
    raw_balance = float(native_token_balance.get("balance", 0) or 0)
    avax_balance = raw_balance / (10 ** decimals)
    avax_price = float(((native_token_balance.get("price") or {}).get("value")) or 0)

    deployment_details = (metadata or {}).get("deploymentDetails") or {}
    return {
        "available": any([metadata, native_balance, transfers, transactions]),
        "address": address,
        "name": (metadata or {}).get("name"),
        "symbol": (metadata or {}).get("symbol"),
        "erc_type": (metadata or {}).get("ercType"),
        "logo_uri": ((metadata or {}).get("logoAsset") or {}).get("imageUri"),
        "deployer_address": deployment_details.get("deployerAddress"),
        "deployment_tx_hash": deployment_details.get("txHash"),
        "recent_transfer_count": len(transfer_rows),
        "recent_transaction_count": len(transaction_rows),
        "recent_active_wallets": len(unique_wallets),
        "last_activity_at": pd.to_datetime(last_activity, unit="s") if last_activity else None,
        "native_balance_avax": avax_balance,
        "native_balance_usd": avax_balance * avax_price if avax_price else None,
    }


def _build_live_signals(state: Dict[str, object], live_metrics: Dict[str, float]) -> List[Dict[str, object]]:
    medians = state["summary_medians"]
    signals: List[Dict[str, object]] = []
    for spec in LIVE_SIGNAL_SPECS:
        feature = spec["feature"]
        if feature not in medians.columns:
            continue
        row_value = live_metrics.get(feature)
        alive_median = medians.at["resilient", feature]
        dead_median = medians.at["terminal_decay", feature]
        risk_score = _signal_risk_score(row_value, alive_median, dead_median, str(spec["direction"]))
        signals.append(
            {
                "feature": feature,
                "label": spec["label"],
                "description": spec["description"],
                "risk_score": round(risk_score * 100, 1),
                "value": _format_signal_value(spec, row_value),
                "alive_median": _format_signal_value(spec, alive_median),
                "dead_median": _format_signal_value(spec, dead_median),
                "narrative": _format_delta(str(spec["direction"]), float(row_value), float(alive_median), float(dead_median))
                if row_value is not None and not pd.isna(row_value)
                else "Live data is insufficient for this signal.",
            }
        )
    return sorted(signals, key=lambda item: item["risk_score"], reverse=True)


@lru_cache(maxsize=1)
def load_app_state(_unused: Optional[str] = None) -> Dict[str, object]:
    early_features = _read_v2_csv("early_features_v2.csv")
    features_summary = _read_v2_csv("features_summary_v2.csv")
    labels = _read_v2_csv("labels_v2.csv")
    protocols = _read_v2_csv("registry_v2.csv")
    stage1_metrics = _read_v2_model_csv("stage1_model_leaderboard_v2.csv")
    scored = _read_v2_model_csv("two_stage_v2_scores.csv")
    product_schema = _read_v2_model_csv("product_schema_v2.csv")
    feature_importance = _read_v2_model_csv("stage1_feature_importance_v2.csv")
    combined_features = _optional_csv("combined_features.csv")
    onchain_features = _optional_csv("onchain_features.csv")

    scored = scored[scored["core_universe_status"] == "core_eligible"].copy()
    early_merge_columns = [
        column
        for column in early_features.columns
        if column
        not in {
            "slug",
            "name",
            "category",
            "current_status",
            "structural_decay_label_v2",
            "structural_decay_target_v2",
            "history_source_mode",
            "data_quality_score_v2",
            "core_eligible",
            "core_universe_status",
        }
    ]
    scored = scored.merge(
        early_features[["slug"] + early_merge_columns],
        on="slug",
        how="left",
        suffixes=("", "_early"),
    )
    scored = scored.merge(
        features_summary[
            [
                "slug",
                "avax_peak_tvl",
                "avax_current_tvl",
                "avax_drawdown_from_peak",
                "avax_tvl_30d_change",
                "avax_tvl_90d_change",
                "avax_lifespan_days",
                "avax_consecutive_decline_days",
            ]
        ],
        on="slug",
        how="left",
        suffixes=("", "_summary"),
    )
    scored = scored.merge(
        product_schema[
            [
                "protocol_id",
                "decay_mode",
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
            ]
        ].rename(columns={"protocol_id": "slug", "decay_mode": "product_decay_mode"}),
        on="slug",
        how="left",
    )
    scored["dead_probability"] = scored["stage1_terminal_prob"]
    scored["health_score"] = ((1.0 - scored["dead_probability"]) * 100).round(1)
    scored["risk_score"] = (scored["dead_probability"] * 100).round(1)
    scored["risk_band"] = scored["health_score"].map(_risk_band)
    scored["label"] = scored["current_status"]
    scored["peak_tvl"] = scored["avax_peak_tvl"]
    scored["current_tvl"] = scored["avax_current_tvl"]
    scored["drawdown_from_peak"] = scored["avax_drawdown_from_peak"]
    scored["tvl_30d_change"] = scored["avax_tvl_30d_change"]
    scored["tvl_90d_change"] = scored["avax_tvl_90d_change"]
    scored["lifespan_days"] = scored["avax_lifespan_days"]
    scored["consecutive_decline_days"] = scored["avax_consecutive_decline_days"]
    scored["snapshot_status"] = scored["current_status"].map(_humanize_status)
    scored["core_source_mode"] = scored["history_source_mode"].map(_humanize_status)
    scored["decay_mode_label"] = scored["product_decay_mode"].map(_display_mode_label)
    scored["evidence_level_label"] = scored["evidence_level"].map(_display_evidence_label)
    scored["model_ready"] = scored["stage1_terminal_prob"].notna()
    scored = scored[scored["model_ready"]].copy()

    train_frame = early_features.merge(
        scored[["slug", "decay_mode_v2"]],
        on="slug",
        how="left",
    )
    stage1_train = train_frame[
        (train_frame["structural_decay_target_v2"] == 0)
        | (train_frame["decay_mode_v2"] == "terminal_global_decay")
    ].copy()
    stage1_train["stage1_terminal_target"] = (stage1_train["decay_mode_v2"] == "terminal_global_decay").astype(int)
    feature_columns = _stage1_feature_columns(stage1_train)
    X = stage1_train[feature_columns]
    y = stage1_train["stage1_terminal_target"]
    model = _stage1_pipeline()
    model.fit(X, y)

    if not combined_features.empty:
        price_cols = [
            "slug",
            "divergence_score",
            "corr_price_tvl",
            "price_leads_tvl",
        ]
        available_price_cols = [column for column in price_cols if column in combined_features.columns]
        scored = scored.merge(
            combined_features[available_price_cols].drop_duplicates("slug"),
            on="slug",
            how="left",
        )
        scored["has_price_signal"] = scored["divergence_score"].notna()
    else:
        scored["has_price_signal"] = False

    if not onchain_features.empty:
        onchain_cols = [
            "slug",
            "onchain_retention",
            "onchain_half_ratio",
            "onchain_total_tx",
        ]
        available_onchain_cols = [column for column in onchain_cols if column in onchain_features.columns]
        scored = scored.merge(
            onchain_features[available_onchain_cols].drop_duplicates("slug"),
            on="slug",
            how="left",
        )
        scored["has_onchain_signal"] = scored["onchain_retention"].notna()
    else:
        scored["has_onchain_signal"] = False

    medians = stage1_train.groupby("stage1_terminal_target")[feature_columns].median(numeric_only=True)
    if 0 not in medians.index or 1 not in medians.index:
        raise ValueError("Both resilient and terminal-decay training rows are required for the MVP.")
    medians.index = pd.Index(["resilient" if idx == 0 else "terminal_decay" for idx in medians.index], name="stage1_terminal_target")

    summary_columns = [
        column
        for column in [
            "drawdown_from_peak",
            "tvl_30d_change",
            "tvl_90d_change",
            "lifespan_days",
            "consecutive_decline_days",
        ]
        if column in scored.columns
    ]
    summary_reference = scored[scored["structural_decay_target_v2"].isin([0, 1])].copy()
    summary_reference["summary_group"] = summary_reference["structural_decay_target_v2"].map({0: "resilient", 1: "terminal_decay"})
    summary_medians = summary_reference.groupby("summary_group")[summary_columns].median(numeric_only=True)

    leaderboard = scored.sort_values(
        ["health_score", "current_tvl", "peak_tvl"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    leaderboard.index = leaderboard.index + 1
    leaderboard["rank"] = leaderboard.index

    current_status_counts = labels["current_status"].value_counts()
    best_row = stage1_metrics.sort_values("cv_auc_mean", ascending=False).iloc[0]

    overview = {
        "protocols_analyzed": int(len(scored)),
        "active_on_avax_count": int(current_status_counts.get("active_on_avax", 0)),
        "low_tvl_on_avax_count": int(current_status_counts.get("low_tvl_on_avax", 0)),
        "boundary_excluded_count": int((labels["core_universe_status"] == "boundary_excluded").sum()),
        "data_incomplete_count": int((labels["core_universe_status"] == "data_incomplete").sum()),
        "model_not_ready_count": int(((labels["core_eligible"] == 1) & (~labels["slug"].isin(scored["slug"]))).sum()),
        "baseline_auc": float(best_row["cv_auc_mean"]),
        "baseline_auc_std": float(best_row["cv_auc_std"]),
        "baseline_accuracy": float(best_row["oof_accuracy"]),
        "price_coverage": int(scored["has_price_signal"].sum()),
        "onchain_coverage": int(scored["has_onchain_signal"].sum()),
        "healthy_count": int((scored["risk_band"] == "Resilient Start").sum()),
        "watchlist_count": int((scored["risk_band"] == "Mixed Start").sum()),
        "high_risk_count": int((scored["risk_band"] == "Fragile Start").sum()),
    }

    return {
        "overview": overview,
        "protocols": protocols,
        "scored_protocols": scored,
        "leaderboard": leaderboard,
        "feature_importance": feature_importance,
        "medians": medians,
        "summary_medians": summary_medians,
        "feature_columns": feature_columns,
        "model": model,
    }


def get_leaderboard(
    state: Dict[str, object],
    risk_band: Optional[str] = None,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    leaderboard = state["leaderboard"].copy()
    if risk_band and risk_band != "All":
        leaderboard = leaderboard[leaderboard["risk_band"] == risk_band]
    if top_n:
        leaderboard = leaderboard.head(top_n)

    columns = [
        "rank",
        "name",
        "category",
        "health_score",
        "dead_probability",
        "risk_band",
        "current_tvl",
        "peak_tvl",
        "label",
        "decay_mode_label",
        "evidence_level_label",
        "has_price_signal",
        "has_onchain_signal",
    ]
    present_columns = [column for column in columns if column in leaderboard.columns]
    table = leaderboard[present_columns].copy()
    if "current_tvl" in table.columns:
        table["current_tvl"] = table["current_tvl"].map(_currency)
    if "peak_tvl" in table.columns:
        table["peak_tvl"] = table["peak_tvl"].map(_currency)
    if "dead_probability" in table.columns:
        table["dead_probability"] = table["dead_probability"].map(lambda value: _percent(value))
    if "label" in table.columns:
        table["label"] = table["label"].map(_humanize_status)
    if "has_price_signal" in table.columns:
        table["price_signal"] = table["has_price_signal"].map({True: "Yes", False: "No"})
        table = table.drop(columns=["has_price_signal"])
    if "has_onchain_signal" in table.columns:
        table["onchain_signal"] = table["has_onchain_signal"].map({True: "Yes", False: "No"})
        table = table.drop(columns=["has_onchain_signal"])
    return table.rename(
        columns={
            "name": "Protocol",
            "category": "Category",
            "health_score": "Early Health Score",
            "dead_probability": "Stage 1 Terminal Risk",
            "risk_band": "Risk Band",
            "current_tvl": "Current TVL",
            "peak_tvl": "Peak TVL",
            "label": "Snapshot Status",
            "decay_mode_label": "Lifecycle Interpretation",
            "evidence_level_label": "Evidence Level",
        }
    )


def build_protocol_view(state: Dict[str, object], slug: str) -> Dict[str, object]:
    scored_protocols = state["scored_protocols"]
    medians = state["medians"]

    match = scored_protocols[scored_protocols["slug"] == slug]
    if match.empty:
        raise KeyError(f"Unknown protocol slug: {slug}")
    row = match.iloc[0]

    raw_history = _load_tvl_history(slug)
    display_history = _prepare_display_history(raw_history)
    history_summary = {
        "peak_tvl": _currency(row.get("peak_tvl")),
        "current_tvl": _currency(row.get("current_tvl")),
        "drawdown_from_peak": _percent(row.get("drawdown_from_peak")),
        "lifespan_days": int(row.get("lifespan_days") or 0),
        "consecutive_decline_days": int(row.get("consecutive_decline_days") or 0),
    }

    signals: List[Dict[str, object]] = []
    for spec in SIGNAL_SPECS:
        feature = spec["feature"]
        row_value = row.get(feature)
        alive_median = medians.at["resilient", feature] if feature in medians.columns else np.nan
        dead_median = medians.at["terminal_decay", feature] if feature in medians.columns else np.nan
        risk_score = _signal_risk_score(row_value, alive_median, dead_median, str(spec["direction"]))
        signals.append(
            {
                "feature": feature,
                "label": spec["label"],
                "description": spec["description"],
                "risk_score": round(risk_score * 100, 1),
                "value": _format_signal_value(spec, row_value),
                "alive_median": _format_signal_value(spec, alive_median),
                "dead_median": _format_signal_value(spec, dead_median),
                "narrative": _format_delta(str(spec["direction"]), float(row_value), float(alive_median), float(dead_median))
                if row_value is not None and not pd.isna(row_value)
                else "Signal is unavailable for this protocol.",
            }
        )

    signals = sorted(signals, key=lambda item: item["risk_score"], reverse=True)

    current_is_lower_risk = bool(row["health_score"] >= 50)
    peer_pool = scored_protocols[(scored_protocols["health_score"] < 50) if current_is_lower_risk else (scored_protocols["health_score"] >= 50)].copy()
    if peer_pool.empty:
        peer_pool = scored_protocols[scored_protocols["slug"] != slug].copy()
    peer_pool["distance"] = (peer_pool["log_peak_tvl"] - row["log_peak_tvl"]).abs()
    peer = peer_pool.sort_values(["distance", "health_score"], ascending=[True, not current_is_lower_risk]).head(1)

    comparison = None
    if not peer.empty:
        peer_row = peer.iloc[0]
        comparison = {
            "name": peer_row["name"],
            "slug": peer_row["slug"],
            "label": peer_row["label"],
            "risk_band": peer_row["risk_band"],
            "health_score": float(peer_row["health_score"]),
            "peak_tvl": _currency(peer_row.get("peak_tvl")),
            "current_tvl": _currency(peer_row.get("current_tvl")),
        }

    supporting_metrics = [
        {"label": "Terminal Decay Risk", "value": _percent(row["dead_probability"])},
        {"label": "Risk Band", "value": row["risk_band"]},
        {"label": "Category", "value": row["category"]},
        {"label": "Current AVAX Status", "value": _humanize_status(row["current_status"])},
        {"label": "Current AVAX Footprint", "value": _current_avax_posture(row)},
        {"label": "Lifecycle Interpretation", "value": row.get("decay_mode_label", "Unknown")},
        {"label": "Evidence Level", "value": row.get("evidence_level_label", "Unknown")},
        {"label": "Avalanche-Native", "value": "Yes" if bool(row.get("avax_only")) else "Multi-chain"},
        {"label": "AVAX Core TVL", "value": _currency(row.get("current_avax_core_tvl"))},
        {"label": "Total Core TVL", "value": _currency(row.get("current_total_core_tvl"))},
        {"label": "AVAX Share", "value": _percent(row.get("current_avax_share"))},
        {"label": "Address Registry", "value": _humanize_status(row.get("address_registry_status"))},
        {"label": "Price Signal Coverage", "value": "Yes" if bool(row.get("has_price_signal")) else "No"},
        {"label": "On-chain Signal Coverage", "value": "Yes" if bool(row.get("has_onchain_signal")) else "No"},
    ]

    if bool(row.get("has_price_signal")):
        supporting_metrics.extend(
            [
                {
                    "label": "Price/TVL Divergence",
                    "value": f"{float(row.get('divergence_score', 0.0)):.3f}",
                },
                {
                    "label": "Price Leads TVL",
                    "value": f"{float(row.get('price_leads_tvl', 0.0)):.3f}",
                },
            ]
        )

    if bool(row.get("has_onchain_signal")):
        supporting_metrics.extend(
            [
                {
                    "label": "On-chain Retention",
                    "value": _percent(row.get("onchain_retention")),
                },
                {
                    "label": "On-chain Total Tx",
                    "value": f"{int(row.get('onchain_total_tx') or 0):,}",
                },
            ]
        )

    return {
        "protocol": {
            "name": row["name"],
            "slug": row["slug"],
            "category": row["category"],
            "health_score": float(row["health_score"]),
            "risk_score": float(row["risk_score"]),
            "risk_band": row["risk_band"],
            "dead_probability": float(row["dead_probability"]),
            "label": row["label"],
            "current_status": row["current_status"],
            "decay_mode": row.get("product_decay_mode") or row.get("decay_mode_v2"),
            "decay_mode_label": row.get("decay_mode_label") or _display_mode_label(row.get("decay_mode_v2")),
            "evidence_level": row.get("evidence_level"),
            "evidence_level_label": row.get("evidence_level_label") or _display_evidence_label(row.get("evidence_level")),
            "evidence_summary": row.get("evidence_summary"),
            "address_registry_status": row.get("address_registry_status"),
            "current_avax_share": row.get("current_avax_share"),
            "current_avax_core_tvl": row.get("current_avax_core_tvl"),
            "current_total_core_tvl": row.get("current_total_core_tvl"),
            "current_avax_core_tvl_display": _currency(row.get("current_avax_core_tvl")),
            "current_total_core_tvl_display": _currency(row.get("current_total_core_tvl")),
            "current_avax_share_display": _percent(row.get("current_avax_share")),
            "lifecycle_interpretation": _build_lifecycle_interpretation(row),
            "avax_footprint_label": _current_avax_posture(row),
            "analyst_summary": _build_analyst_summary(row),
        },
        "history": display_history["history"],
        "history_meta": {
            "activation_start": display_history["activation_start"],
            "hidden_pre_activation_points": display_history["hidden_pre_activation_points"],
            "gap_break_count": display_history["gap_break_count"],
            "raw_points": display_history["raw_points"],
        },
        "history_summary": history_summary,
        "risk_signals": signals[:3],
        "all_signals": signals,
        "supporting_metrics": supporting_metrics,
        "comparison": comparison,
    }


def refresh_protocol_live(state: Dict[str, object], slug: str) -> Dict[str, object]:
    protocol_payload = _request_json(DEFILLAMA_PROTOCOL_URL.format(slug=slug))
    if not protocol_payload:
        return {
            "available": False,
            "slug": slug,
            "reason": "Live protocol data could not be fetched from DeFiLlama.",
        }

    raw_history = _extract_tvl_history_from_protocol(protocol_payload)
    if raw_history.empty:
        return {
            "available": False,
            "slug": slug,
            "reason": "Live Avalanche TVL history is not available for this protocol.",
        }
    display_history = _prepare_display_history(raw_history)

    scored_protocols = state["scored_protocols"]
    local_match = scored_protocols[scored_protocols["slug"] == slug]
    local_row = local_match.iloc[0] if not local_match.empty else None

    early_features = _extract_stage1_v2_features(raw_history)
    model = state["model"]
    if early_features:
        feature_vector = pd.DataFrame(
            [{column: early_features.get(column, 0.0) for column in state["feature_columns"]}]
        ).fillna(0.0)
        dead_probability = float(model.predict_proba(feature_vector)[0, 1])
        refreshed_health_score = round((1.0 - dead_probability) * 100, 1)
    else:
        dead_probability = np.nan
        refreshed_health_score = np.nan

    live_metrics = _compute_live_tvl_metrics(raw_history)
    live_signals = _build_live_signals(state, live_metrics)
    valid_live_signal_scores = [signal["risk_score"] / 100 for signal in live_signals if signal["risk_score"] is not None]
    live_monitor_score = round((1.0 - float(np.mean(valid_live_signal_scores))) * 100, 1) if valid_live_signal_scores else np.nan

    contract_address = _extract_contract_address(protocol_payload)
    glacier_summary = _fetch_glacier_summary(contract_address)

    local_current_tvl = float(local_row.get("current_tvl")) if local_row is not None and pd.notna(local_row.get("current_tvl")) else np.nan
    live_current_tvl = live_metrics.get("current_tvl")
    current_tvl_delta = live_current_tvl - local_current_tvl if pd.notna(local_current_tvl) and pd.notna(live_current_tvl) else np.nan
    current_tvl_delta_pct = (
        current_tvl_delta / local_current_tvl if pd.notna(current_tvl_delta) and local_current_tvl not in (0, np.nan) else np.nan
    )

    return {
        "available": True,
        "slug": slug,
        "protocol_name": protocol_payload.get("name") or (local_row["name"] if local_row is not None else slug),
        "history": display_history["history"],
        "history_meta": {
            "activation_start": display_history["activation_start"],
            "hidden_pre_activation_points": display_history["hidden_pre_activation_points"],
            "gap_break_count": display_history["gap_break_count"],
            "raw_points": display_history["raw_points"],
        },
        "baseline": {
            "refreshed_health_score": refreshed_health_score,
            "dead_probability": dead_probability,
            "risk_band": _risk_band(refreshed_health_score) if not pd.isna(refreshed_health_score) else "Unavailable",
            "last_live_date": raw_history["date"].max(),
            "data_points": int(len(raw_history)),
        },
        "live_metrics": {
            **live_metrics,
            "current_tvl_delta": current_tvl_delta,
            "current_tvl_delta_pct": current_tvl_delta_pct,
            "monitor_score": live_monitor_score,
            "monitor_band": _risk_band(live_monitor_score) if not pd.isna(live_monitor_score) else "Unavailable",
        },
        "live_signals": live_signals[:3],
        "all_live_signals": live_signals,
        "glacier": glacier_summary,
        "sources": {
            "defillama_live": True,
            "glacier_live": bool(glacier_summary.get("available")),
            "contract_address": contract_address,
        },
    }
