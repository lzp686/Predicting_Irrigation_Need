import pandas as pd
import numpy as np
from scipy.special import softmax

# ===== 公式 =====
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
sub_high = pd.read_csv('./output/submission_part2_0.981.csv')
df74     = pd.read_csv('./data/pd74/5(4) - 0.98074.csv')
test     = pd.read_csv('./data/test.csv')

diff_mask = sub_high['Irrigation_Need'].values != df74['Irrigation_Need'].values
diff_idx  = np.where(diff_mask)[0]
df_diff   = test.iloc[diff_idx].copy().reset_index(drop=True)
df_diff['pred_high'] = sub_high['Irrigation_Need'].values[diff_idx]
df_diff['pred_74']   = df74['Irrigation_Need'].values[diff_idx]

# ===== 公式预测这 140 行 =====
X = build_features(df_diff)
pred_idx, conf = formula_predict(X.values.astype(float))
df_diff['formula'] = [LABEL_MAP[i] for i in pred_idx]
df_diff['formula_conf'] = conf

# ===== 三方对比 =====
print("=== 公式 vs 你(0.981) vs df74 ===")
print(f"\n公式预测分布:\n{df_diff['formula'].value_counts()}")

print(f"\n=== 三方一致性 ===")
agree_all   = (df_diff['pred_high'] == df_diff['formula']).sum()
agree_74    = (df_diff['pred_74']   == df_diff['formula']).sum()
print(f"公式同意你(0.981): {agree_all} / 140")
print(f"公式同意df74:      {agree_74} / 140")

print(f"\n=== 按分歧方向细看 ===")
for a, b in [('Low','Medium'), ('Medium','High'), ('Medium','Low')]:
    mask = (df_diff['pred_high']==a) & (df_diff['pred_74']==b)
    sub  = df_diff[mask]
    if len(sub) == 0: continue
    f_you = (sub['formula'] == a).sum()
    f_74  = (sub['formula'] == b).sum()
    f_3rd = len(sub) - f_you - f_74
    print(f"\n你={a}, df74={b} ({len(sub)}行):")
    print(f"  公式同意你:   {f_you}")
    print(f"  公式同意df74: {f_74}")
    print(f"  公式第三意见: {f_3rd}")
    print(f"  公式置信度: mean={sub['formula_conf'].mean():.3f}, min={sub['formula_conf'].min():.3f}")