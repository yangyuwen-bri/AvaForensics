"""
拉取 Avalanche 协议的历史 TVL 时序数据
目标：
  - 过滤掉 CEX 等噪音
  - 拉取每个协议的历史 TVL 曲线
  - 区分"存活"vs"死亡"项目
  - 保存供后续特征工程使用
"""

import requests
import pandas as pd
import time
import json
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────
OUTPUT_DIR = Path("avax_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# 过滤掉这些 category（CEX、RWA等与链上DeFi无关）
EXCLUDE_CATEGORIES = {
    "CEX", "RWA", "Cross Chain Bridge", "Payments",
    "Bridge",  # 跨链桥TVL大多是多链共享，不反映Avalanche本身
}

# TVL 低于此值视为"实际死亡"（美元）
DEAD_TVL_THRESHOLD = 10_000

# 拉取时序时的请求间隔（秒），避免被限流
REQUEST_DELAY = 0.5

# ── 加载协议列表 ──────────────────────────────────────
df = pd.read_csv("avax_protocols.csv")

# 过滤 CEX 等噪音
df_clean = df[~df["category"].isin(EXCLUDE_CATEGORIES)].copy()
print(f"过滤后协议数: {len(df_clean)}  (原始: {len(df)})")

# 拆分：存活 vs 死亡
df_alive = df_clean[df_clean["tvl_usd"] >= DEAD_TVL_THRESHOLD].copy()
df_dead  = df_clean[df_clean["tvl_usd"] <  DEAD_TVL_THRESHOLD].copy()

print(f"  存活项目 (TVL >= ${DEAD_TVL_THRESHOLD:,}): {len(df_alive)}")
print(f"  死亡/僵尸项目 (TVL < ${DEAD_TVL_THRESHOLD:,}): {len(df_dead)}")
print()

# 给项目打标签
df_clean["label"] = (df_clean["tvl_usd"] >= DEAD_TVL_THRESHOLD).map(
    {True: "alive", False: "dead"}
)
df_clean.to_csv(OUTPUT_DIR / "protocols_labeled.csv", index=False)
print(f"💾 已保存带标签的协议列表 → {OUTPUT_DIR}/protocols_labeled.csv")

# ── 拉取历史 TVL 时序 ─────────────────────────────────
def fetch_tvl_history(slug: str):
    """
    调用 DeFiLlama API 拉取单个协议的历史 TVL
    返回 DataFrame(date, tvl) 或 None（失败时）
    """
    url = f"https://api.llama.fi/protocol/{slug}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()

        # tvlByChain 里有按链分开的数据，优先取 Avalanche 的
        chain_tvls = data.get("chainTvls", {})
        avax_key = None
        for key in chain_tvls:
            if "avalanche" in key.lower():
                avax_key = key
                break

        if avax_key and chain_tvls[avax_key].get("tvl"):
            records = chain_tvls[avax_key]["tvl"]
        elif data.get("tvl"):
            # 没有按链分开的就用总 TVL
            records = data["tvl"]
        else:
            return None

        df_tvl = pd.DataFrame(records)  # columns: date(timestamp), totalLiquidityUSD
        df_tvl = df_tvl.rename(columns={"totalLiquidityUSD": "tvl"})
        df_tvl["date"] = pd.to_datetime(df_tvl["date"], unit="s")
        df_tvl = df_tvl.sort_values("date").reset_index(drop=True)
        return df_tvl

    except Exception as e:
        print(f"  ⚠️  {slug} 请求失败: {e}")
        return None


def compute_features(slug: str, df_tvl: pd.DataFrame) -> dict:
    """
    从时序 TVL 数据提取项目健康度特征
    """
    tvl = df_tvl["tvl"].values
    dates = df_tvl["date"].values

    peak_tvl = float(tvl.max()) if len(tvl) > 0 else 0
    current_tvl = float(tvl[-1]) if len(tvl) > 0 else 0
    lifespan_days = int((dates[-1] - dates[0]) / 1e9 / 86400) if len(dates) > 1 else 0

    # 从峰值跌落多少
    drawdown_from_peak = (
        (peak_tvl - current_tvl) / peak_tvl if peak_tvl > 0 else 0
    )

    # 最近30天 TVL 变化率
    recent = df_tvl[df_tvl["date"] >= df_tvl["date"].max() - pd.Timedelta(days=30)]
    if len(recent) >= 2:
        tvl_30d_change = (recent["tvl"].iloc[-1] - recent["tvl"].iloc[0]) / (recent["tvl"].iloc[0] + 1)
    else:
        tvl_30d_change = 0.0

    # 最近90天 TVL 变化率
    recent90 = df_tvl[df_tvl["date"] >= df_tvl["date"].max() - pd.Timedelta(days=90)]
    if len(recent90) >= 2:
        tvl_90d_change = (recent90["tvl"].iloc[-1] - recent90["tvl"].iloc[0]) / (recent90["tvl"].iloc[0] + 1)
    else:
        tvl_90d_change = 0.0

    # TVL 波动性（标准差/均值）
    mean_tvl = float(tvl.mean()) if len(tvl) > 0 else 0
    std_tvl  = float(tvl.std())  if len(tvl) > 1 else 0
    volatility = std_tvl / mean_tvl if mean_tvl > 0 else 0

    # 连续下跌天数（从最高点之后）
    peak_idx = tvl.argmax()
    post_peak = tvl[peak_idx:]
    consecutive_decline = int(sum(1 for i in range(1, len(post_peak)) if post_peak[i] < post_peak[i-1]))

    return {
        "slug": slug,
        "peak_tvl": peak_tvl,
        "current_tvl": current_tvl,
        "drawdown_from_peak": drawdown_from_peak,
        "tvl_30d_change": tvl_30d_change,
        "tvl_90d_change": tvl_90d_change,
        "volatility": volatility,
        "lifespan_days": lifespan_days,
        "consecutive_decline_days": consecutive_decline,
        "data_points": len(df_tvl),
    }


# ── 主循环：拉所有协议的时序数据 ──────────────────────
print("\n开始拉取历史 TVL 时序（可能需要几分钟）...")
print("-" * 50)

all_features = []
saved_count = 0
failed = []

# 优先拉 avax_only 的项目，再拉多链的
df_priority = pd.concat([
    df_clean[df_clean["avax_only"] == True],
    df_clean[df_clean["avax_only"] == False],
]).reset_index(drop=True)

for i, row in df_priority.iterrows():
    slug = row["slug"]
    label = row["label"]
    print(f"[{i+1}/{len(df_priority)}] {row['name']} ({label}) ...", end=" ")

    df_tvl = fetch_tvl_history(slug)

    if df_tvl is None or len(df_tvl) < 7:
        print("❌ 数据不足，跳过")
        failed.append(slug)
        time.sleep(REQUEST_DELAY)
        continue

    # 保存原始时序
    df_tvl["slug"] = slug
    df_tvl["label"] = label
    df_tvl.to_csv(OUTPUT_DIR / f"tvl_{slug}.csv", index=False)
    saved_count += 1

    # 提取特征
    features = compute_features(slug, df_tvl)
    features["label"] = label
    features["name"] = row["name"]
    features["category"] = row["category"]
    features["avax_only"] = row["avax_only"]
    all_features.append(features)

    peak = features["peak_tvl"]
    dd = features["drawdown_from_peak"]
    print(f"✅ 峰值TVL=${peak:,.0f}, 跌幅={dd:.1%}")

    time.sleep(REQUEST_DELAY)

# ── 汇总特征表 ────────────────────────────────────────
print("\n" + "=" * 50)
print(f"完成！成功: {saved_count}，失败: {len(failed)}")

if all_features:
    df_features = pd.DataFrame(all_features)
    df_features.to_csv(OUTPUT_DIR / "features_summary.csv", index=False)
    print(f"💾 特征汇总 → {OUTPUT_DIR}/features_summary.csv")

    print("\n── 死亡 vs 存活 项目的特征对比 ──")
    cols = ["peak_tvl", "drawdown_from_peak", "tvl_30d_change",
            "tvl_90d_change", "lifespan_days", "volatility"]
    print(df_features.groupby("label")[cols].median().round(3).to_string())

    print("\n── 死亡项目 category 分布 ──")
    print(df_features[df_features["label"] == "dead"]["category"].value_counts().head(10))

if failed:
    with open(OUTPUT_DIR / "failed_slugs.txt", "w") as f:
        f.write("\n".join(failed))
    print(f"\n⚠️  失败的 slug 已保存 → {OUTPUT_DIR}/failed_slugs.txt")

print("\n下一步：把 features_summary.csv 发给我，开始特征工程和模型训练！")