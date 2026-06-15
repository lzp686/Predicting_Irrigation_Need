"""
PS-S6E4 | 公式融合版 EoS Voting
目标：在 0.98114 基础上提分
核心改动：不一致行用公式裁决，替代 Aux (0.97254)
"""

import os
import numpy as np
import pandas as pd
from scipy.special import softmax

# ===== 路径配置 =====
PPS  = './data/'
PD74 = './data/pd74/'
PD85 = './data/pd85/'
OUT  = './output/'
os.makedirs(OUT, exist_ok=True)

# ===== 公式系数 =====
FEATURE_NAMES = [
    'soil_lt_25', 'temp_gt_30', 'rain_lt_300', 'wind_gt_10',
    'Crop_Growth_Stage_Flowering', 'Crop_Growth_Stage_Harvest',
    'Crop_Growth_Stage_Sowing', 'Crop_Growth_Stage_Vegetative',
    'Mulching_Used_No', 'Mulching_Used_Yes',
]
INTERCEPTS = np.array([16.3173, 4.6524, -20.9697])
COEFS = np.array([
    [-11.0237, -5.8559, -10.8500, -5.8284, -5.4155,  5.5073,  5.2299, -5.4617, -3.0014,  2.8613],
    [  0.3290, -0.0204,  0.1542,  0.0841,  0.3586, -0.1348, -0.3547,  0.3334,  0.1883,  0.0142],
    [ 10.6947,  5.8763, 10.6958,  5.7444,  5.0569, -5.3725, -4.8752,  5.1283,  2.8131, -2.8755],
])
LABEL_MAP = {0: 'Low', 1: 'Medium', 2: 'High'}

def build_features(df):
    d = df.copy()
    d['soil_lt_25'] = (d['Soil_Moisture'] < 25).astype(int)
    d['temp_gt_30']  = (d['Temperature_C'] > 30).astype(int)
    d['rain_lt_300'] = (d['Rainfall_mm'] < 300).astype(int)
    d['wind_gt_10']  = (d['Wind_Speed_kmh'] > 10).astype(int)
    cgs = pd.get_dummies(d['Crop_Growth_Stage'], prefix='Crop_Growth_Stage')
    mul = pd.get_dummies(d['Mulching_Used'], prefix='Mulching_Used')
    X = pd.concat([d[['soil_lt_25','temp_gt_30','rain_lt_300','wind_gt_10']], cgs, mul], axis=1)
    for col in FEATURE_NAMES:
        if col not in X.columns:
            X[col] = 0
    return X[FEATURE_NAMES]

def formula_predict(X_matrix):
    logits = X_matrix @ COEFS.T + INTERCEPTS
    probs = softmax(logits, axis=1)
    return np.argmax(probs, axis=1), probs.max(axis=1)

# ===== 读取数据 =====
print("Loading data...")
test  = pd.read_csv('./data/test.csv')
sub   = pd.read_csv(PPS + 'sample_submission.csv')

df74  = pd.read_csv(PD85 + '5(4) - 0.98074.csv').rename(columns={'Irrigation_Need': 'df74'})
df72  = pd.read_csv(PD85 + '5(9) - 0.98072.csv').rename(columns={'Irrigation_Need': 'df72'})
dfAux = pd.read_csv(PD85 + 'Aux - 0.97254.csv').rename(columns={'Irrigation_Need': 'Aux1'})

# ===== 公式预测测试集 =====
print("Running formula predictions...")
X_test = build_features(test)
formula_preds_idx, formula_conf = formula_predict(X_test.values.astype(float))
formula_preds = [LABEL_MAP[i] for i in formula_preds_idx]

# ===== 融合逻辑 =====
print("Merging predictions...")
dfs = pd.merge(df74, df72, on='id')
dfs = pd.merge(dfs, dfAux, on='id')
dfs['formula_pred'] = formula_preds
dfs['formula_conf'] = formula_conf

CONF_THRESHOLD = 0.90  # 可尝试 0.85, 0.90, 0.95

def fuse(row):
    if row['df74'] == row['df72']:
        return row['df74']  # 一致：直接采用
    # 不一致：公式裁决
    if row['formula_conf'] >= CONF_THRESHOLD:
        return row['formula_pred']
    # 公式也不确定：Aux 三票制
    votes = [row['df74'], row['df72'], row['Aux1']]
    return max(set(votes), key=votes.count)

dfs['Irrigation_Need'] = dfs.apply(fuse, axis=1)

# ===== 统计 =====
agree = (dfs['df74'] == dfs['df72'])
disagree = ~agree
formula_decided = disagree & (dfs['formula_conf'] >= CONF_THRESHOLD)
aux_decided     = disagree & (dfs['formula_conf'] < CONF_THRESHOLD)

print(f"\n=== 决策分布 ===")
print(f"一致行（直接采用）:   {agree.sum():>7,} 行 ({agree.mean()*100:.2f}%)")
print(f"不一致行合计:          {disagree.sum():>7,} 行 ({disagree.mean()*100:.2f}%)")
print(f"  └ 公式裁决（conf≥{CONF_THRESHOLD}）: {formula_decided.sum():>6,} 行")
print(f"  └ Aux 三票 fallback: {aux_decided.sum():>6,} 行")

print(f"\n=== 最终预测分布 ===")
print(dfs['Irrigation_Need'].value_counts())

# 与原始 0.98074 对比
agree_74 = (dfs['Irrigation_Need'] == dfs['df74'])
print(f"\n与 0.98074 一致率: {agree_74.mean()*100:.4f}% ({agree_74.sum()} 行)")

# 不一致行中，公式和 df74 各持几票？
in_disagree = dfs[disagree]
formula_vs_74 = (in_disagree['formula_pred'] == in_disagree['df74']).sum()
formula_vs_72 = (in_disagree['formula_pred'] == in_disagree['df72']).sum()
print(f"\n不一致行中：")
print(f"  公式同意 df74: {formula_vs_74} / {disagree.sum()}")
print(f"  公式同意 df72: {formula_vs_72} / {disagree.sum()}")

# ===== 保存 =====
sub2 = sub.copy()
sub2['Irrigation_Need'] = dfs['Irrigation_Need'].values
sub2.to_csv(OUT + 'submission_formula_fused.csv', index=False)
print(f"\nSaved: {OUT}submission_formula_fused.csv")

# ===== 多个阈值对比（帮助选参数） =====
print("\n=== 不同公式置信度阈值对比 ===")
for thresh in [0.80, 0.85, 0.90, 0.95, 0.99]:
    def fuse_t(row, t=thresh):
        if row['df74'] == row['df72']: return row['df74']
        if row['formula_conf'] >= t:   return row['formula_pred']
        votes = [row['df74'], row['df72'], row['Aux1']]
        return max(set(votes), key=votes.count)
    pred_t = dfs.apply(fuse_t, axis=1)
    diff_from_base = (pred_t != dfs['df74']).sum()
    print(f"  thresh={thresh:.2f}: 改变了 {diff_from_base:5,} 行 vs 0.98074")