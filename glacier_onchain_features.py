"""
功能：
  1. 通过合约地址拉取链上交易量历史（每日tx数）
  2. 提取"链上活跃度衰减"特征
  3. 合并进已有特征表，形成三维数据：TVL + 价格 + 链上活跃度
  4. 重跑模型，看链上数据带来多少提升

使用前准备：
  1. 去 https://avacloud.io 注册免费账号
  2. 创建 API Key
  3. 在下方填入你的 API Key，或设置环境变量 GLACIER_API_KEY

Glacier API 文档：https://glacier-api.avax.network/api
"""

import os
import time
from dotenv import load_dotenv
load_dotenv()  # 从 .env 文件加载环境变量
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── 配置 ──────────────────────────────────────────────────────
DATA_DIR = Path("avax_data")

# 从环境变量读取，或者直接填在这里
GLACIER_API_KEY = os.environ.get("GLACIER_API_KEY", "")

GLACIER_BASE    = "https://glacier-api.avax.network/v1"
AVALANCHE_CHAIN = "43114"   # Avalanche C-Chain mainnet chain ID
REQUEST_DELAY   = 0.5       # 免费版限速，保守处理

HEADERS = {
    "Content-Type": "application/json",
    "x-glacier-api-key": GLACIER_API_KEY,
}

EARLY_WINDOW_DAYS = 90


# ── Glacier API 封装 ──────────────────────────────────────────

def check_api_key():
    """验证 API Key 是否有效"""
    url = f"{GLACIER_BASE}/chains"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    if resp.status_code == 401:
        print("❌ API Key 无效，请去 https://avacloud.io 注册并创建 Key")
        return False
    elif resp.status_code == 200:
        chains = resp.json().get("chains", [])
        avax = [c for c in chains if c.get("chainId") == AVALANCHE_CHAIN]
        print(f"✅ API Key 有效  |  找到 Avalanche C-Chain: {bool(avax)}")
        return True
    else:
        print(f"⚠️  API 返回 {resp.status_code}: {resp.text[:200]}")
        return False


def get_contract_transactions(
    contract_address: str,
    page_size: int = 100,
    max_pages: int = 10
) -> pd.DataFrame:
    """
    拉取合约地址的历史交易记录
    Glacier API 返回格式: {nativeTransaction: {...}, erc20Transfers: [...]}
    """
    url = f"{GLACIER_BASE}/chains/{AVALANCHE_CHAIN}/addresses/{contract_address}/transactions"
    params = {"pageSize": page_size}
    all_txs = []

    for page in range(max_pages):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if resp.status_code == 429:
                print("  ⚠️  限速，等30秒...")
                time.sleep(30)
                continue
            if resp.status_code == 404:
                return None
            if resp.status_code != 200:
                return None

            data = resp.json()
            txs  = data.get("transactions", [])
            all_txs.extend(txs)

            next_token = data.get("nextPageToken")
            if not next_token or len(txs) == 0:
                break
            params["pageToken"] = next_token
            time.sleep(REQUEST_DELAY)

        except Exception as e:
            print(f"  ⚠️  请求失败: {e}")
            break

    if not all_txs:
        return None

    rows = []
    for tx in all_txs:
        # Glacier API 实际结构：{nativeTransaction: {...}, erc20Transfers: [...]}
        native = tx.get("nativeTransaction", tx)  # 兼容两种结构
        ts = native.get("blockTimestamp") or native.get("timestamp", 0)
        rows.append({
            "timestamp": ts,
            "tx_hash":   native.get("txHash", ""),
            "from_addr": native.get("from", {}).get("address", ""),
            "to_addr":   native.get("to", {}).get("address", ""),
            "value":     float(native.get("value", 0)),
            # 同时记录是否包含 erc20 活动
            "has_erc20": len(tx.get("erc20Transfers", [])) > 0,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.date
    return df


def get_erc20_transfers(
    contract_address: str,
    page_size: int = 100,
    max_pages: int = 5
) -> pd.DataFrame:
    """
    拉取 ERC20 代币转账记录（对 DeFi 协议更有意义）
    """
    url = f"{GLACIER_BASE}/chains/{AVALANCHE_CHAIN}/addresses/{contract_address}/transactions:listErc20Transfers"
    params = {"pageSize": page_size}
    all_transfers = []

    for _ in range(max_pages):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if resp.status_code not in (200, 404):
                break
            if resp.status_code == 404:
                return None

            data = resp.json()
            transfers = data.get("erc20Transfers", [])
            all_transfers.extend(transfers)

            next_token = data.get("nextPageToken")
            if not next_token:
                break
            params["pageToken"] = next_token
            time.sleep(REQUEST_DELAY)
        except:
            break

    if not all_transfers:
        return None

    rows = []
    for t in all_transfers:
        rows.append({
            "timestamp": t.get("blockTimestamp", 0),
            "from_addr": t.get("from", {}).get("address", ""),
            "to_addr":   t.get("to", {}).get("address", ""),
            "value":     float(t.get("value", 0)),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.date
    return df


def get_token_holders_count(contract_address: str) -> int:
    """
    获取代币持有者数量（ERC20）
    持有者数越少 → 越集中 → 风险越高
    """
    url = f"{GLACIER_BASE}/chains/{AVALANCHE_CHAIN}/tokens/{contract_address}/holders"
    params = {"pageSize": 1}
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if resp.status_code == 200:
            # nextPageToken存在说明有很多持有者
            data = resp.json()
            holders = data.get("holders", [])
            # 用第一页估算：如果只有几个持有者，风险极高
            return len(holders)
    except:
        pass
    return None


# ── 从链上交易记录提取早期特征 ────────────────────────────────

def extract_onchain_features(df_tx: pd.DataFrame, window_days: int = 90) -> dict:
    """
    从交易记录提取早期活跃度特征
    只用前 window_days 天的数据
    """
    if df_tx is None or len(df_tx) < 5:
        return None

    # 按日期聚合交易量
    df_tx["date"] = pd.to_datetime(df_tx["date"])
    daily = df_tx.groupby("date").size().reset_index(name="tx_count")
    daily = daily.sort_values("date").reset_index(drop=True)

    start = daily["date"].iloc[0]
    cutoff = start + pd.Timedelta(days=window_days)
    early = daily[daily["date"] <= cutoff].copy()

    if len(early) < 5:
        return None

    tx = early["tx_count"].values
    n  = len(tx)

    # 峰值活跃度及其时机
    peak_idx     = tx.argmax()
    peak_tx      = tx.max()
    peak_day_frac = peak_idx / n

    # 活跃度留存（窗口末尾 vs 峰值）
    retention = tx[-1] / (peak_tx + 1)

    # 前半段 vs 后半段（活跃度是否在衰减）
    mid = n // 2
    half_ratio = tx[mid:].mean() / (tx[:mid].mean() + 1)

    # 整体趋势斜率
    slope, _, _, _, _ = stats.linregress(np.arange(n), tx / (peak_tx + 1))

    # 归零天数（单日tx=0的比例）
    zero_day_frac = (tx == 0).sum() / n

    # 唯一活跃地址（如果数据里有的话）
    unique_addrs = None
    if "from_addr" in df_tx.columns:
        early_raw = df_tx[df_tx["date"] <= cutoff]
        unique_addrs = early_raw["from_addr"].nunique()

    return {
        "onchain_peak_day_frac":  peak_day_frac,   # 链上活跃峰值出现时机
        "onchain_retention":      retention,        # 活跃度留存率
        "onchain_half_ratio":     half_ratio,       # 后半段/前半段活跃度
        "onchain_slope":          slope,            # 活跃度趋势
        "onchain_zero_day_frac":  zero_day_frac,    # 归零天比例
        "onchain_unique_addrs":   unique_addrs or 0,
        "onchain_total_tx":       int(tx.sum()),
    }


# ── 合约地址映射（需要从DeFiLlama协议详情里获取） ──────────────

def get_contract_address_from_defillama(slug: str) -> str:
    """
    从 DeFiLlama 协议详情里提取 Avalanche 主合约地址
    优先级：address字段 > chainTvls.avalanche 合约 > contracts字段
    """
    url = f"https://api.llama.fi/protocol/{slug}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()

        # 1. 直接的 address 字段（格式 "avax:0x..."）
        address = data.get("address", "")
        if address:
            for part in address.split(","):
                part = part.strip()
                if "avax:" in part.lower():
                    return part.split(":")[-1].strip()

        # 2. 从 contracts 字段找
        contracts = data.get("contracts", {})
        if "avax" in contracts:
            addrs = contracts["avax"]
            if isinstance(addrs, list) and addrs:
                return addrs[0]
            if isinstance(addrs, str):
                return addrs

        # 3. 从 chainTvls 里找 Avalanche 合约
        chain_tvls = data.get("chainTvls", {})
        for chain_key in ["Avalanche", "avalanche", "AVAX"]:
            if chain_key in chain_tvls:
                chain_data = chain_tvls[chain_key]
                # 有些协议在这里存合约
                if isinstance(chain_data, dict):
                    sub_contracts = chain_data.get("contracts", [])
                    if sub_contracts and isinstance(sub_contracts, list):
                        return sub_contracts[0]

    except:
        pass
    return None


# ── 主流程 ────────────────────────────────────────────────────

print("=" * 55)
print("AvaForensics — Glacier API 链上数据接入")
print("=" * 55)

# Step 0: 验证 API Key
if GLACIER_API_KEY == "your_api_key_here":
    print()
    print("⚠️  请先设置 API Key！")
    print()
    print("步骤：")
    print("  1. 访问 https://avacloud.io")
    print("  2. 注册免费账号 → 创建 Project → 生成 API Key")
    print("  3. 方式A：直接修改本文件第22行的 GLACIER_API_KEY")
    print("     方式B：export GLACIER_API_KEY=你的key  然后重新运行")
    print()
    print("拿到 Key 后重新运行此脚本")
    exit()

print("\nStep 0: 验证 API Key...")
if not check_api_key():
    exit()

# Step 1: 获取合约地址
print("\nStep 1: 从 DeFiLlama 获取合约地址...")
df_meta = pd.read_csv(DATA_DIR / "protocols_labeled.csv")
label_map = dict(zip(df_meta["slug"], df_meta["label"]))

# 优先处理有价值的项目（dead项目峰值TVL > 100万）
df_early = pd.read_csv(DATA_DIR / "early_features.csv")
priority_slugs = df_early[
    df_early["log_peak_tvl"] > np.log1p(1e6)
]["slug"].tolist()[:50]  # 先跑前50个，验证流程

print(f"优先处理 {len(priority_slugs)} 个高峰值项目")

address_map = {}
for i, slug in enumerate(priority_slugs):
    addr = get_contract_address_from_defillama(slug)
    if addr:
        address_map[slug] = addr
    if (i + 1) % 10 == 0:
        print(f"  进度: {i+1}/{len(priority_slugs)}, 找到地址: {len(address_map)}")
    time.sleep(0.3)

print(f"✅ 找到合约地址: {len(address_map)} / {len(priority_slugs)}")
pd.DataFrame([
    {"slug": k, "contract_address": v} for k, v in address_map.items()
]).to_csv(DATA_DIR / "contract_addresses.csv", index=False)


# Step 2: 拉链上交易数据 + 提取特征
print("\nStep 2: 拉取链上交易记录 + 提取活跃度特征...")
onchain_features = []
failed = []

for i, (slug, addr) in enumerate(address_map.items()):
    print(f"[{i+1}/{len(address_map)}] {slug[:30]:30s} ({addr[:10]}...)", end=" ")

    # 先试 ERC20 转账（对 DeFi 更有意义）
    df_tx = get_erc20_transfers(addr, max_pages=3)

    # 如果没有 ERC20 转账，用普通 tx
    if df_tx is None or len(df_tx) < 5:
        df_tx = get_contract_transactions(addr, max_pages=3)

    feats = extract_onchain_features(df_tx, window_days=EARLY_WINDOW_DAYS)

    if feats is None:
        print("❌ 数据不足")
        failed.append(slug)
        time.sleep(REQUEST_DELAY)
        continue

    feats["slug"]  = slug
    feats["label"] = label_map.get(slug, "unknown")
    onchain_features.append(feats)
    print(f"✅ total_tx={feats['onchain_total_tx']:,}  retention={feats['onchain_retention']:.3f}")
    time.sleep(REQUEST_DELAY)

print(f"\n成功: {len(onchain_features)}  失败: {len(failed)}")


# Step 3: 特征分析 + 合并模型
if not onchain_features:
    print("❌ 没有链上特征数据，检查 API Key 和合约地址")
    exit()

df_onchain = pd.DataFrame(onchain_features)
df_onchain.to_csv(DATA_DIR / "onchain_features.csv", index=False)
print(f"💾 onchain_features.csv 已保存\n")

# 区分力分析
print("=" * 55)
print("链上特征区分力分析")
print("=" * 55)
alive = df_onchain[df_onchain["label"] == "alive"]
dead  = df_onchain[df_onchain["label"] == "dead"]

onchain_cols = [c for c in df_onchain.columns
                if c.startswith("onchain_") and c not in ["slug", "label"]]

for col in onchain_cols:
    a = alive[col].dropna()
    d = dead[col].dropna()
    if len(a) < 3 or len(d) < 3:
        continue
    _, pval = stats.mannwhitneyu(a, d, alternative="two-sided")
    sig = "✅" if pval < 0.05 else "❌"
    print(f"  {col:35s}  alive={a.median():.3f}  dead={d.median():.3f}  p={pval:.4f} {sig}")

# 合并 TVL + 链上 特征跑模型
print("\n" + "=" * 55)
print("模型对比：TVL alone vs TVL + 链上活跃度")
print("=" * 55)

df_tvl_feats = pd.read_csv(DATA_DIR / "early_features.csv")
df_combined  = df_tvl_feats.merge(
    df_onchain.drop(columns=["label"]), on="slug", how="inner"
)
print(f"合并后样本: {len(df_combined)}  (dead: {(df_combined['label']=='dead').sum()})")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

y  = (df_combined["label"] == "dead").astype(int)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

tvl_cols = [
    "peak_day_frac", "retention_at_end", "half_ratio", "half_life_frac",
    "log_growth_multiple", "volatility", "slope_total", "r_squared",
    "rise_length_frac", "log_peak_tvl", "crash_count", "mean_pct_change"
]
tvl_cols     = [c for c in tvl_cols if c in df_combined.columns]
onchain_cols = [c for c in onchain_cols if c in df_combined.columns]

for label, cols in [
    ("TVL特征（baseline）         ", tvl_cols),
    ("链上活跃度（单独）            ", onchain_cols),
    ("TVL + 链上活跃度（合并）      ", tvl_cols + onchain_cols),
]:
    X      = df_combined[cols].fillna(0)
    scores = cross_val_score(
        RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        X, y, cv=cv, scoring="roc_auc"
    )
    print(f"  {label}  AUC: {scores.mean():.3f} ± {scores.std():.3f}")

print("\n" + "=" * 55)
print("链上组件接入完成 ✅")
print("这是 AvaForensics 的 Avalanche on-chain component")
print("=" * 55)