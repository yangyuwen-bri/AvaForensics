"""
接入 CoinGecko 价格数据
目标：
  - 拉取每个协议对应代币的历史价格
  - 提取价格早期特征（前90天）
  - 关键：找出 TVL 和价格的"背离信号"
  - 合并到已有特征表，重跑模型看 AUC 提升多少
"""

import requests
import pandas as pd
import numpy as np
import time
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── 配置 ──────────────────────────────────────────────
DATA_DIR          = Path("avax_data")
EARLY_WINDOW_DAYS = 90
REQUEST_DELAY     = 1.5   # CoinGecko 免费版限速 ~30次/分钟，保守用1.5秒

# ── Step 1: 通过 DeFiLlama 拿到每个协议的 coingecko ID ──
# DeFiLlama 的 protocol 详情接口里直接含有 gecko_id，省去手动匹配

def get_gecko_id(slug: str):
    """从 DeFiLlama 拿 coingecko_id"""
    url = f"https://api.llama.fi/protocol/{slug}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data.get("gecko_id") or None
    except:
        return None


def fetch_price_history(gecko_id: str, days: int = 365):
    """
    调用 CoinGecko /coins/{id}/market_chart
    返回 DataFrame(date, price, volume, market_cap)
    免费版最多拉 365 天
    """
    url = f"https://api.coingecko.com/api/v3/coins/{gecko_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 429:
            print("  ⚠️  限速，等待60秒...")
            time.sleep(60)
            resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()

        prices     = data.get("prices", [])
        volumes    = data.get("total_volumes", [])
        market_cap = data.get("market_caps", [])

        if not prices:
            return None

        df = pd.DataFrame(prices, columns=["ts", "price"])
        df["date"]       = pd.to_datetime(df["ts"], unit="ms")
        df["volume"]     = [v[1] for v in volumes]     if volumes     else 0
        df["market_cap"] = [m[1] for m in market_cap]  if market_cap else 0
        df = df.drop(columns=["ts"]).sort_values("date").reset_index(drop=True)
        return df

    except Exception as e:
        print(f"  ⚠️  {gecko_id} 请求失败: {e}")
        return None


def extract_price_features(df_price: pd.DataFrame,
                            df_tvl:   pd.DataFrame,
                            window_days: int = 90):
    """
    从价格时序提取早期特征，并计算价格/TVL背离度
    """
    # 截取早期窗口
    df_p = df_price.sort_values("date").reset_index(drop=True)
    start = df_p["date"].iloc[0]
    cutoff = start + pd.Timedelta(days=window_days)
    early_p = df_p[df_p["date"] <= cutoff].copy()

    if len(early_p) < 10:
        return None

    price  = early_p["price"].values
    volume = early_p["volume"].values
    n      = len(price)

    # ── 价格基础特征 ──────────────────────────────────
    peak_price   = price.max()
    peak_idx     = price.argmax()
    final_price  = price[-1]
    first_price  = price[0] if price[0] > 0 else 1e-10

    price_peak_day_frac    = peak_idx / n
    price_retention_at_end = final_price / (peak_price + 1e-10)

    # 全窗口价格趋势（线性斜率）
    slope_price, _, _, _, _ = stats.linregress(
        np.arange(n), price / (peak_price + 1e-10)
    )

    # 价格波动性
    pct_changes     = np.diff(price) / (price[:-1] + 1e-10)
    price_volatility = pct_changes.std() if len(pct_changes) > 1 else 0
    price_crash_count = int((pct_changes < -0.2).sum())  # 单日跌超20%

    # 成交量衰减（后半段 vs 前半段）
    mid = n // 2
    vol_ratio = volume[mid:].mean() / (volume[:mid].mean() + 1e-10)

    # ── 核心：价格 vs TVL 背离度 ─────────────────────
    # 把价格和TVL对齐到同一时间轴，计算相关性和背离
    divergence_score    = None
    price_leads_tvl     = None
    corr_price_tvl      = None

    if df_tvl is not None and len(df_tvl) > 10:
        df_t = df_tvl.sort_values("date").reset_index(drop=True)
        early_t = df_t[df_t["date"] <= cutoff].copy()

        if len(early_t) >= 10:
            # 按周聚合，减少噪音
            early_p["week"] = (early_p["date"] - start).dt.days // 7
            early_t["week"] = (early_t["date"] - start).dt.days // 7

            p_weekly = early_p.groupby("week")["price"].mean()
            t_weekly = early_t.groupby("week")["tvl"].mean()

            common_weeks = p_weekly.index.intersection(t_weekly.index)
            if len(common_weeks) >= 4:
                p_vals = p_weekly[common_weeks].values
                t_vals = t_weekly[common_weeks].values

                # 归一化到 0-1
                p_norm = (p_vals - p_vals.min()) / (p_vals.ptp() + 1e-10)
                t_norm = (t_vals - t_vals.min()) / (t_vals.ptp() + 1e-10)

                # 相关性（高相关=同步变动，低相关=背离）
                if p_vals.std() > 0 and t_vals.std() > 0:
                    corr_price_tvl = float(np.corrcoef(p_norm, t_norm)[0, 1])
                else:
                    corr_price_tvl = 0.0

                # 背离度：价格下跌但TVL还在（出货信号）
                # 或 TVL下跌但价格还撑着（拉盘出货）
                divergence_score = float(np.mean(np.abs(p_norm - t_norm)))

                # 价格是否领先TVL下跌（滞后相关）
                if len(p_norm) >= 3:
                    # 价格(t-1) 和 TVL(t) 的相关性
                    corr_lead = float(np.corrcoef(p_norm[:-1], t_norm[1:])[0, 1])
                    price_leads_tvl = corr_lead
                else:
                    price_leads_tvl = 0.0

    return {
        "price_peak_day_frac":    price_peak_day_frac,     # 价格峰值出现早晚
        "price_retention_at_end": price_retention_at_end,  # 窗口末价格留存率
        "slope_price":            slope_price,              # 价格整体趋势
        "price_volatility":       price_volatility,         # 价格波动性
        "price_crash_count":      price_crash_count,        # 暴跌次数
        "volume_decay_ratio":     vol_ratio,                # 成交量衰减
        "corr_price_tvl":         corr_price_tvl,           # 价格TVL相关性
        "divergence_score":       divergence_score,         # 背离度（关键特征）
        "price_leads_tvl":        price_leads_tvl,          # 价格领先TVL的程度
    }


# ── 主流程 ────────────────────────────────────────────
print("=" * 55)
print("Step 1: 获取 CoinGecko ID")
print("=" * 55)

df_meta = pd.read_csv(DATA_DIR / "protocols_labeled.csv")
label_map = dict(zip(df_meta["slug"], df_meta["label"]))

# 加载已有早期特征
df_early = pd.read_csv(DATA_DIR / "early_features.csv")
slugs_to_fetch = df_early["slug"].tolist()

gecko_map = {}
for i, slug in enumerate(slugs_to_fetch):
    gecko_id = get_gecko_id(slug)
    if gecko_id:
        gecko_map[slug] = gecko_id
    if (i + 1) % 20 == 0:
        print(f"  进度: {i+1}/{len(slugs_to_fetch)}, 找到gecko_id: {len(gecko_map)}")
    time.sleep(0.3)

print(f"\n✅ 找到 gecko_id 的项目: {len(gecko_map)} / {len(slugs_to_fetch)}")

# 保存映射关系
pd.DataFrame([
    {"slug": k, "gecko_id": v} for k, v in gecko_map.items()
]).to_csv(DATA_DIR / "gecko_id_map.csv", index=False)
print(f"💾 gecko_id_map.csv 已保存\n")


print("=" * 55)
print("Step 2: 拉取历史价格 + 提取特征")
print("=" * 55)

price_features = []
failed_price = []

for i, (slug, gecko_id) in enumerate(gecko_map.items()):
    print(f"[{i+1}/{len(gecko_map)}] {slug} ({gecko_id}) ...", end=" ")

    # 拉价格数据
    df_price = fetch_price_history(gecko_id, days=365)
    if df_price is None or len(df_price) < 10:
        print("❌ 价格数据不足")
        failed_price.append(slug)
        time.sleep(REQUEST_DELAY)
        continue

    # 保存原始价格
    df_price["slug"] = slug
    df_price.to_csv(DATA_DIR / f"price_{slug}.csv", index=False)

    # 加载对应的TVL时序
    tvl_path = DATA_DIR / f"tvl_{slug}.csv"
    df_tvl = pd.read_csv(tvl_path, parse_dates=["date"]) if tvl_path.exists() else None

    # 提取价格特征
    feats = extract_price_features(df_price, df_tvl, window_days=EARLY_WINDOW_DAYS)
    if feats is None:
        print("❌ 特征提取失败")
        failed_price.append(slug)
        time.sleep(REQUEST_DELAY)
        continue

    feats["slug"]  = slug
    feats["label"] = label_map.get(slug, "unknown")
    price_features.append(feats)
    print(f"✅ 背离度={feats.get('divergence_score', 'N/A'):.3f}" if feats.get('divergence_score') else "✅")

    time.sleep(REQUEST_DELAY)

print(f"\n成功: {len(price_features)}  失败: {len(failed_price)}")


print("\n" + "=" * 55)
print("Step 3: 合并特征 + 重跑模型")
print("=" * 55)

if not price_features:
    print("❌ 没有价格特征数据，退出")
    exit()

df_price_feats = pd.DataFrame(price_features)

# 合并到早期TVL特征
df_combined = df_early.merge(
    df_price_feats.drop(columns=["label"]),
    on="slug", how="inner"
)
print(f"合并后样本数: {len(df_combined)}")
print(f"标签分布: {df_combined['label'].value_counts().to_dict()}")

df_combined.to_csv(DATA_DIR / "combined_features.csv", index=False)
print(f"💾 combined_features.csv 已保存\n")


# ── 对比实验：TVL特征 alone vs TVL+价格特征 ──────────
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y  = (df_combined["label"] == "dead").astype(int)

tvl_cols = [
    "peak_day_frac", "retention_at_end", "half_ratio",
    "half_life_frac", "log_growth_multiple", "volatility",
    "slope_total", "r_squared", "rise_length_frac",
    "log_peak_tvl", "crash_count", "mean_pct_change", "data_points"
]
price_cols = [
    "price_peak_day_frac", "price_retention_at_end", "slope_price",
    "price_volatility", "price_crash_count", "volume_decay_ratio",
    "corr_price_tvl", "divergence_score", "price_leads_tvl"
]

# 只用存在的列
tvl_cols   = [c for c in tvl_cols   if c in df_combined.columns]
price_cols = [c for c in price_cols if c in df_combined.columns]

print("── 模型对比（随机森林，5折AUC）──")
for label, cols in [
    ("TVL特征（baseline）",    tvl_cols),
    ("价格特征（单独）",        price_cols),
    ("TVL + 价格特征（合并）",  tvl_cols + price_cols),
]:
    X = df_combined[cols].fillna(0)
    scores = cross_val_score(
        RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        X, y, cv=cv, scoring="roc_auc"
    )
    print(f"  {label:30s} AUC: {scores.mean():.3f} ± {scores.std():.3f}")


# ── 背离特征专项分析 ─────────────────────────────────
print("\n── 背离度（divergence_score）分析 ──")
if "divergence_score" in df_combined.columns:
    alive = df_combined[df_combined["label"] == "alive"]["divergence_score"].dropna()
    dead  = df_combined[df_combined["label"] == "dead"]["divergence_score"].dropna()
    print(f"  alive 中位数: {alive.median():.3f}")
    print(f"  dead  中位数: {dead.median():.3f}")
    stat, pval = stats.mannwhitneyu(alive, dead, alternative="two-sided")
    print(f"  p值: {pval:.4f}  {'✅ 显著' if pval < 0.05 else '❌ 不显著'}")

print("\n── 价格领先TVL（price_leads_tvl）分析 ──")
if "price_leads_tvl" in df_combined.columns:
    alive = df_combined[df_combined["label"] == "alive"]["price_leads_tvl"].dropna()
    dead  = df_combined[df_combined["label"] == "dead"]["price_leads_tvl"].dropna()
    print(f"  alive 中位数: {alive.median():.3f}")
    print(f"  dead  中位数: {dead.median():.3f}")
    stat, pval = stats.mannwhitneyu(alive, dead, alternative="two-sided")
    print(f"  p值: {pval:.4f}  {'✅ 显著' if pval < 0.05 else '❌ 不显著'}")

print("\n" + "=" * 55)
print("把三行 AUC 对比结果发给我，看价格数据带来多少提升")
print("=" * 55)