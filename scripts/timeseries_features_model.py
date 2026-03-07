"""
时序特征提取 + Baseline 模型
核心逻辑：
  - 只用项目"早期窗口"（前N天）的TVL曲线提取特征
  - 预测项目最终是否死亡
  - 跑 baseline 看准确率，评估哪些早期信号最有预测力
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── 配置 ──────────────────────────────────────────────
DATA_DIR = Path("avax_data")
EARLY_WINDOW_DAYS = 90   # 只看前90天
MIN_DATA_POINTS   = 30   # 至少要有30个数据点才纳入分析
PREDICT_HORIZON   = 365  # 预测1年后是否死亡

# ── 从单条时序提取早期特征 ────────────────────────────
def extract_early_features(df_tvl: pd.DataFrame, window_days: int = 90):
    """
    输入：完整的TVL时序 DataFrame（含 date, tvl 列）
    输出：只用前 window_days 天数据计算出的特征字典
    """
    df = df_tvl.sort_values("date").reset_index(drop=True)
    if len(df) < MIN_DATA_POINTS:
        return None

    # 截取早期窗口
    start_date = df["date"].iloc[0]
    cutoff     = start_date + pd.Timedelta(days=window_days)
    early      = df[df["date"] <= cutoff].copy()

    if len(early) < 10:
        return None

    tvl    = early["tvl"].values
    n      = len(tvl)
    t      = np.arange(n)  # 时间轴（索引）

    # ── 1. 峰值相关 ──────────────────────────────────
    peak_val   = tvl.max()
    peak_idx   = tvl.argmax()
    peak_day   = peak_idx  # 峰值出现在第几个数据点（近似天数）
    peak_frac  = peak_idx / n  # 峰值出现在窗口的哪个位置（0=最开始, 1=最末尾）

    # 最终值
    final_val  = tvl[-1]
    first_val  = tvl[0] if tvl[0] > 0 else 1

    # 相对峰值还剩多少
    retention_at_end = final_val / peak_val if peak_val > 0 else 0

    # ── 2. 衰退速度 ──────────────────────────────────
    post_peak  = tvl[peak_idx:]

    # 从峰值跌到50%用了多少步
    half_peak  = peak_val * 0.5
    steps_to_half = next(
        (i for i, v in enumerate(post_peak) if v <= half_peak),
        len(post_peak)  # 如果没跌到，就用窗口长度
    )
    half_life_frac = steps_to_half / (n - peak_idx + 1)  # 归一化

    # 峰值后均匀下跌速度（线性拟合斜率）
    if len(post_peak) >= 3:
        slope_post, _, _, _, _ = stats.linregress(
            np.arange(len(post_peak)), post_peak / (peak_val + 1)
        )
    else:
        slope_post = 0.0

    # ── 3. 整体增长/衰退形态 ─────────────────────────
    # 全窗口线性趋势
    slope_total, intercept, r_val, _, _ = stats.linregress(t, tvl / (peak_val + 1))
    r_squared = r_val ** 2

    # 前半段 vs 后半段的均值对比（判断整体是在上涨还是下跌）
    mid = n // 2
    first_half_mean  = tvl[:mid].mean()
    second_half_mean = tvl[mid:].mean()
    half_ratio = second_half_mean / (first_half_mean + 1)  # >1 上涨，<1 下跌

    # ── 4. 波动性 ────────────────────────────────────
    # 日涨跌幅
    pct_changes = np.diff(tvl) / (tvl[:-1] + 1)
    volatility      = pct_changes.std() if len(pct_changes) > 1 else 0
    mean_pct_change = pct_changes.mean() if len(pct_changes) > 1 else 0

    # 极端暴跌次数（单日跌超30%）
    crash_count = int((pct_changes < -0.3).sum())

    # ── 5. 增长阶段特征 ──────────────────────────────
    # 上涨期长度（峰值前）
    rise_length_frac = peak_idx / n

    # 初始增速（第一个数据点到峰值的增长倍数）
    growth_multiple = peak_val / (first_val + 1)
    log_growth      = np.log1p(growth_multiple)

    # ── 6. 规模特征（对数变换） ───────────────────────
    log_peak_tvl  = np.log1p(peak_val)
    log_final_tvl = np.log1p(final_val)

    return {
        # 峰值相关
        "peak_day_frac":      peak_frac,          # 峰值出现早晚（越早=越可能是庞氏）
        "retention_at_end":   retention_at_end,   # 窗口末尾还剩峰值的多少
        "half_life_frac":     half_life_frac,      # 跌到一半峰值的速度（越慢越好）

        # 衰退斜率
        "slope_post_peak":    slope_post,          # 峰值后下跌斜率（负=下跌）
        "slope_total":        slope_total,         # 全窗口趋势

        # 形态
        "half_ratio":         half_ratio,          # 后半段/前半段均值（>1=整体上涨）
        "r_squared":          r_squared,           # 趋势拟合优度

        # 波动性
        "volatility":         volatility,          # 日涨跌幅标准差
        "mean_pct_change":    mean_pct_change,     # 平均日涨跌幅
        "crash_count":        crash_count,         # 暴跌次数

        # 增长质量
        "rise_length_frac":   rise_length_frac,    # 上涨期占比
        "log_growth_multiple": log_growth,         # 初始增长倍数（对数）

        # 规模
        "log_peak_tvl":       log_peak_tvl,        # 峰值TVL（对数）
        "data_points":        n,
    }


# ── 加载所有时序文件，提取特征 ────────────────────────
print("加载时序数据并提取早期特征...")
print(f"窗口: 前 {EARLY_WINDOW_DAYS} 天  |  最小数据点: {MIN_DATA_POINTS}")
print("-" * 50)

# 加载 label 信息
df_meta = pd.read_csv(DATA_DIR / "protocols_labeled.csv")
label_map = dict(zip(df_meta["slug"], df_meta["label"]))
name_map  = dict(zip(df_meta["slug"], df_meta["name"]))
cat_map   = dict(zip(df_meta["slug"], df_meta["category"]))

records = []
skipped = 0

for fpath in sorted(DATA_DIR.glob("tvl_*.csv")):
    slug = fpath.stem.replace("tvl_", "")
    if slug not in label_map:
        continue

    df_tvl = pd.read_csv(fpath, parse_dates=["date"])
    feats = extract_early_features(df_tvl, window_days=EARLY_WINDOW_DAYS)

    if feats is None:
        skipped += 1
        continue

    feats["slug"]     = slug
    feats["label"]    = label_map[slug]
    feats["name"]     = name_map.get(slug, slug)
    feats["category"] = cat_map.get(slug, "")
    records.append(feats)

df_feats = pd.DataFrame(records)
print(f"成功提取: {len(df_feats)} 个项目  |  跳过(数据不足): {skipped}")
print(f"标签分布: {df_feats['label'].value_counts().to_dict()}")

df_feats.to_csv(DATA_DIR / "early_features.csv", index=False)
print(f"💾 已保存 → {DATA_DIR}/early_features.csv\n")

# ── 特征区分力分析 ────────────────────────────────────
print("=" * 50)
print("特征区分力（Mann-Whitney U 检验）")
print("=" * 50)

feature_cols = [c for c in df_feats.columns
                if c not in ["slug", "label", "name", "category"]]

alive = df_feats[df_feats["label"] == "alive"]
dead  = df_feats[df_feats["label"] == "dead"]

rows = []
for col in feature_cols:
    a = alive[col].dropna()
    d = dead[col].dropna()
    if len(a) < 5 or len(d) < 5:
        continue
    stat, pval = stats.mannwhitneyu(a, d, alternative="two-sided")
    rows.append({
        "特征": col,
        "alive中位数": round(a.median(), 4),
        "dead中位数":  round(d.median(), 4),
        "p值": pval,
        "显著": "✅" if pval < 0.01 else ("⚠️" if pval < 0.05 else "❌")
    })

df_rank = pd.DataFrame(rows).sort_values("p值")
print(df_rank.to_string(index=False))

# ── Baseline 模型（随机森林） ─────────────────────────
print("\n" + "=" * 50)
print("Baseline 模型：随机森林（只用早期特征）")
print("=" * 50)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import sklearn

X = df_feats[feature_cols].fillna(0)
y = (df_feats["label"] == "dead").astype(int)

print(f"样本数: {len(X)}  |  特征数: {len(feature_cols)}")
print(f"死亡项目占比: {y.mean():.1%}\n")

# 5折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in [
    ("随机森林",        RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
    ("梯度提升(GBDT)", GradientBoostingClassifier(n_estimators=200, random_state=42)),
]:
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"{name}:")
    print(f"  AUC:      {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
    print(f"  Accuracy: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")
    print()

# 用完整数据训练一次，看特征重要性
rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
rf.fit(X, y)

print("── 特征重要性（随机森林）──")
importance_df = pd.DataFrame({
    "特征": feature_cols,
    "重要性": rf.feature_importances_
}).sort_values("重要性", ascending=False)
print(importance_df.to_string(index=False))

# ── 保存特征重要性 ────────────────────────────────────
importance_df.to_csv(DATA_DIR / "feature_importance.csv", index=False)
print(f"\n💾 特征重要性 → {DATA_DIR}/feature_importance.csv")

print("\n" + "=" * 50)
print("下一步：把终端输出（特别是AUC分数和特征重要性）发给我")
print("我们根据结果决定：补充哪些新特征 / 调整预测窗口 / 优化模型")
print("=" * 50)