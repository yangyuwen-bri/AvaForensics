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
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "avax_data"
DEFILLAMA_PROTOCOL_URL = "https://api.llama.fi/protocol/{slug}"
GLACIER_BASE_URL = "https://glacier-api.avax.network/v1"
AVALANCHE_CHAIN_ID = "43114"
REQUEST_TIMEOUT_SECONDS = 20

load_dotenv(BASE_DIR / ".env")

MODEL_EXCLUDE_COLUMNS = {"slug", "label", "name", "category"}

SIGNAL_SPECS = [
    {
        "feature": "peak_day_frac",
        "label": "TVL Peak Timing",
        "unit": "day",
        "window_days": 90,
        "decimals": 0,
        "direction": "high",
        "description": "Projects that peak too early often behave like pump-and-dump launches.",
    },
    {
        "feature": "retention_at_end",
        "label": "90-Day TVL Retention",
        "unit": "percent",
        "window_days": None,
        "decimals": 1,
        "direction": "high",
        "description": "Healthy protocols keep more of their early TVL instead of bleeding out.",
    },
    {
        "feature": "half_ratio",
        "label": "Second-Half Ratio",
        "unit": "ratio",
        "window_days": None,
        "decimals": 2,
        "direction": "high",
        "description": "The back half of a healthy protocol's early history should stay close to the front half.",
    },
    {
        "feature": "half_life_frac",
        "label": "Post-Peak Half-Life",
        "unit": "percent",
        "window_days": None,
        "decimals": 1,
        "direction": "high",
        "description": "Longer half-life means the protocol kept value after its early peak.",
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
    (75, "Healthy"),
    (50, "Watchlist"),
    (0, "High Risk"),
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


def _feature_columns(df: pd.DataFrame) -> List[str]:
    return [column for column in df.columns if column not in MODEL_EXCLUDE_COLUMNS]


def _risk_band(score: float) -> str:
    for threshold, label in RISK_BANDS:
        if score >= threshold:
            return label
    return "High Risk"


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
        return f"Closer to dead-project median than alive-project median ({gap:.2f} vs alive baseline)."
    return f"Stronger than the alive-project median by {gap:.2f}."


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

    for chain_key, chain_value in chain_tvls.items():
        if "avalanche" in chain_key.lower() and isinstance(chain_value, dict) and chain_value.get("tvl"):
            records = chain_value["tvl"]
            break

    if records is None:
        if isinstance(protocol_payload.get("tvl"), list):
            records = protocol_payload["tvl"]
        else:
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
        alive_median = medians.at["alive", feature]
        dead_median = medians.at["dead", feature]
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
def load_app_state() -> Dict[str, object]:
    early_features = _read_csv("early_features.csv")
    features_summary = _read_csv("features_summary.csv")
    protocols = _read_csv("protocols_labeled.csv")
    feature_importance = _optional_csv("feature_importance.csv")
    combined_features = _optional_csv("combined_features.csv")
    onchain_features = _optional_csv("onchain_features.csv")

    feature_columns = _feature_columns(early_features)
    X = early_features[feature_columns].fillna(0)
    y = (early_features["label"] == "dead").astype(int)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    probabilities = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    model.fit(X, y)

    scored = early_features.copy()
    scored["dead_probability"] = probabilities
    scored["health_score"] = ((1.0 - scored["dead_probability"]) * 100).round(1)
    scored["risk_score"] = (scored["dead_probability"] * 100).round(1)
    scored["risk_band"] = scored["health_score"].map(_risk_band)

    scored = scored.merge(
        features_summary[
            [
                "slug",
                "peak_tvl",
                "current_tvl",
                "drawdown_from_peak",
                "lifespan_days",
                "consecutive_decline_days",
                "avax_only",
            ]
        ],
        on="slug",
        how="left",
    )

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

    if feature_importance.empty:
        feature_importance = pd.DataFrame(
            {
                "特征": feature_columns,
                "重要性": model.feature_importances_,
            }
        ).sort_values("重要性", ascending=False)

    medians = early_features.groupby("label")[feature_columns].median(numeric_only=True)
    if "alive" not in medians.index or "dead" not in medians.index:
        raise ValueError("Both alive and dead labels are required for the MVP.")

    summary_columns = [
        column
        for column in [
            "drawdown_from_peak",
            "tvl_30d_change",
            "tvl_90d_change",
            "lifespan_days",
            "consecutive_decline_days",
        ]
        if column in features_summary.columns
    ]
    summary_medians = features_summary.groupby("label")[summary_columns].median(numeric_only=True)

    leaderboard = scored.sort_values(
        ["health_score", "current_tvl", "peak_tvl"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    leaderboard.index = leaderboard.index + 1
    leaderboard["rank"] = leaderboard.index

    top_dead_case = (
        scored[scored["label"] == "dead"]
        .sort_values(["peak_tvl", "health_score"], ascending=[False, True])
        .head(1)
    )
    top_alive_case = (
        scored[scored["label"] == "alive"]
        .sort_values(["health_score", "peak_tvl"], ascending=[False, False])
        .head(1)
    )

    overview = {
        "protocols_analyzed": int(len(scored)),
        "dead_protocols": int((scored["label"] == "dead").sum()),
        "alive_protocols": int((scored["label"] == "alive").sum()),
        "baseline_auc": float(auc_scores.mean()),
        "baseline_auc_std": float(auc_scores.std()),
        "baseline_accuracy": float(acc_scores.mean()),
        "price_coverage": int(scored["has_price_signal"].sum()),
        "onchain_coverage": int(scored["has_onchain_signal"].sum()),
        "healthy_count": int((scored["risk_band"] == "Healthy").sum()),
        "watchlist_count": int((scored["risk_band"] == "Watchlist").sum()),
        "high_risk_count": int((scored["risk_band"] == "High Risk").sum()),
        "top_alive_slug": top_alive_case["slug"].iloc[0] if not top_alive_case.empty else None,
        "top_dead_slug": top_dead_case["slug"].iloc[0] if not top_dead_case.empty else None,
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
        "risk_band",
        "current_tvl",
        "peak_tvl",
        "label",
        "has_price_signal",
        "has_onchain_signal",
    ]
    present_columns = [column for column in columns if column in leaderboard.columns]
    table = leaderboard[present_columns].copy()
    if "current_tvl" in table.columns:
        table["current_tvl"] = table["current_tvl"].map(_currency)
    if "peak_tvl" in table.columns:
        table["peak_tvl"] = table["peak_tvl"].map(_currency)
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
            "health_score": "Health Score",
            "risk_band": "Risk Band",
            "current_tvl": "Current TVL",
            "peak_tvl": "Peak TVL",
            "label": "Current Label",
        }
    )


def build_protocol_view(state: Dict[str, object], slug: str) -> Dict[str, object]:
    scored_protocols = state["scored_protocols"]
    medians = state["medians"]

    match = scored_protocols[scored_protocols["slug"] == slug]
    if match.empty:
        raise KeyError(f"Unknown protocol slug: {slug}")
    row = match.iloc[0]

    history = _load_tvl_history(slug)
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
        alive_median = medians.at["alive", feature] if feature in medians.columns else np.nan
        dead_median = medians.at["dead", feature] if feature in medians.columns else np.nan
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
                "narrative": _format_delta(str(spec["direction"]), float(row_value), float(alive_median), float(dead_median)),
            }
        )

    signals = sorted(signals, key=lambda item: item["risk_score"], reverse=True)

    peer_pool = scored_protocols[scored_protocols["label"] != row["label"]].copy()
    peer_pool["distance"] = (peer_pool["log_peak_tvl"] - row["log_peak_tvl"]).abs()
    peer = peer_pool.sort_values(["distance", "health_score"], ascending=[True, row["label"] == "dead"]).head(1)

    comparison = None
    if not peer.empty:
        peer_row = peer.iloc[0]
        comparison = {
            "name": peer_row["name"],
            "slug": peer_row["slug"],
            "label": peer_row["label"],
            "health_score": float(peer_row["health_score"]),
            "peak_tvl": _currency(peer_row.get("peak_tvl")),
            "current_tvl": _currency(peer_row.get("current_tvl")),
        }

    supporting_metrics = [
        {"label": "Dead Probability", "value": _percent(row["dead_probability"])},
        {"label": "Risk Band", "value": row["risk_band"]},
        {"label": "Category", "value": row["category"]},
        {"label": "Current Label", "value": row["label"].title()},
        {"label": "Avalanche-Native", "value": "Yes" if bool(row.get("avax_only")) else "Multi-chain"},
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
        },
        "history": history,
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

    history = _extract_tvl_history_from_protocol(protocol_payload)
    if history.empty:
        return {
            "available": False,
            "slug": slug,
            "reason": "Live Avalanche TVL history is not available for this protocol.",
        }

    scored_protocols = state["scored_protocols"]
    local_match = scored_protocols[scored_protocols["slug"] == slug]
    local_row = local_match.iloc[0] if not local_match.empty else None

    early_features = _extract_early_features(history)
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

    live_metrics = _compute_live_tvl_metrics(history)
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
        "history": history,
        "baseline": {
            "refreshed_health_score": refreshed_health_score,
            "dead_probability": dead_probability,
            "risk_band": _risk_band(refreshed_health_score) if not pd.isna(refreshed_health_score) else "Unavailable",
            "last_live_date": history["date"].max(),
            "data_points": int(len(history)),
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
