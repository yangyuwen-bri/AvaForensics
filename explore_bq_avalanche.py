"""
探索 BigQuery 中 Avalanche 相关数据集
策略：
  1. 先验证 BQ 连接是否正常（查自己项目）
  2. 直接尝试已知的 Avalanche 公链数据集（指定正确 location）
  3. DeFiLlama 作为可靠备用
"""

from google.cloud import bigquery
import pandas as pd

PROJECT_ID = "getpatent-486217"
client = bigquery.Client(project=PROJECT_ID)

# ─────────────────────────────────────────
# 1. 验证 BQ 连接：查自己项目有哪些数据集
# ─────────────────────────────────────────
print("=" * 60)
print("【1】验证 BigQuery 连接（列出自己项目的数据集）")
print("=" * 60)

try:
    datasets = list(client.list_datasets())
    if datasets:
        print(f"✅ 连接成功，项目 {PROJECT_ID} 下有以下数据集：")
        for ds in datasets:
            print(f"   - {ds.dataset_id}")
    else:
        print(f"✅ 连接成功，项目 {PROJECT_ID} 下暂无数据集（这是正常的）")
except Exception as e:
    print(f"❌ BQ 连接失败: {e}")

# ─────────────────────────────────────────
# 2. 直接查询 crypto_ethereum（指定 EU 或 US location 都试试）
#    注意：bigquery-public-data.crypto_ethereum 实际在 US multi-region
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("【2】探测 crypto_ethereum（EVM 参考，确认 BQ 公共数据可达性）")
print("=" * 60)

# 用 SELECT 1 FROM 表 LIMIT 0 来测试可达性，不扫描数据
eth_test_query = """
SELECT COUNT(*) as block_count
FROM `bigquery-public-data.crypto_ethereum.blocks`
WHERE DATE(timestamp) = '2024-01-01'
"""
try:
    job_config = bigquery.QueryJobConfig(
        location="US",
        maximum_bytes_billed=10 * 1024 * 1024 * 1024  # 10GB 上限保护
    )
    df = client.query(eth_test_query, job_config=job_config).to_dataframe()
    print(f"✅ crypto_ethereum 可访问！2024-01-01 共 {df['block_count'].iloc[0]:,} 个区块")
except Exception as e:
    print(f"❌ crypto_ethereum 不可访问: {str(e)[:200]}")

# ─────────────────────────────────────────
# 3. 直接尝试 Google 官方区块链数据集（goog_blockchain_*）
#    这些在 US 多区域，需要通过 Analytics Hub 订阅
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("【3】直接查询 Google 官方区块链数据集")
print("=" * 60)

# goog_blockchain_avalanche_mainnet_us 是 Google 官方数据集
candidates = {
    "Avalanche (Google官方)": {
        "query": "SELECT COUNT(*) as cnt FROM `bigquery-public-data.goog_blockchain_avalanche_mainnet_us.transactions` WHERE DATE(block_timestamp) = '2024-01-01'",
        "location": "US",
    },
    "Ethereum (Google官方)": {
        "query": "SELECT COUNT(*) as cnt FROM `bigquery-public-data.goog_blockchain_ethereum_mainnet_us.transactions` WHERE DATE(block_timestamp) = '2024-01-01'",
        "location": "US",
    },
    "crypto_avalanche": {
        "query": "SELECT table_name FROM `bigquery-public-data.crypto_avalanche`.INFORMATION_SCHEMA.TABLES LIMIT 10",
        "location": "US",
    },
}

for name, cfg in candidates.items():
    try:
        job_config = bigquery.QueryJobConfig(
            location=cfg["location"],
            maximum_bytes_billed=10 * 1024 * 1024 * 1024
        )
        df = client.query(cfg["query"], job_config=job_config).to_dataframe()
        print(f"\n✅ {name} 可访问！")
        print(df.to_string(index=False))
    except Exception as e:
        msg = str(e)[:150]
        if "403" in msg:
            print(f"\n❌ {name}: 权限不足（需要申请访问）")
        elif "404" in msg:
            print(f"\n❌ {name}: 数据集不存在")
        else:
            print(f"\n❌ {name}: {msg}")

# ─────────────────────────────────────────
# 4. DeFiLlama —— 已验证可用，补充协议详情
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("【4】DeFiLlama：Avalanche 生态协议详情")
print("=" * 60)

import requests

try:
    resp = requests.get("https://api.llama.fi/protocols", timeout=15)
    protocols = resp.json()

    avax_protocols = [
        p for p in protocols
        if "Avalanche" in p.get("chains", [])
    ]

    df_avax = pd.DataFrame([{
        "name": p["name"],
        "category": p.get("category", ""),
        "tvl_usd": p.get("tvl", 0),
        "num_chains": len(p.get("chains", [])),
        "avax_only": len(p.get("chains", [])) == 1,  # 是否 Avalanche 独占
        "slug": p.get("slug", ""),
    } for p in avax_protocols])

    df_avax = df_avax.sort_values("tvl_usd", ascending=False)

    # Avalanche 原生（只在 Avalanche 上）的协议
    df_native = df_avax[df_avax["avax_only"] == True].copy()
    print(f"\n✅ Avalanche 原生协议（仅在 Avalanche 上部署）: {len(df_native)} 个")
    print(df_native.head(20)[["name", "category", "tvl_usd"]].to_string(index=False))

    print(f"\n✅ 共 {len(df_avax)} 个在 Avalanche 上有部署的协议")
    print("\n-- TVL 前20名（含多链协议）--")
    print(df_avax.head(20)[["name", "category", "tvl_usd", "num_chains"]].to_string(index=False))

    print("\n-- 按 category 分布 --")
    print(df_avax["category"].value_counts().head(15).to_string())

    df_avax.to_csv("avax_protocols.csv", index=False)
    print("\n💾 已保存到 avax_protocols.csv")

except Exception as e:
    print(f"DeFiLlama 请求失败: {e}")

print("\n" + "=" * 60)
print("探索完成！")
print("=" * 60)
