import pandas as pd
import numpy as np

sub_high = pd.read_csv('./output/submission_part2_0.981.csv')
df74     = pd.read_csv('./data/pd74/5(4) - 0.98074.csv')
test     = pd.read_csv('./data/test.csv')

diff_mask = sub_high['Irrigation_Need'].values != df74['Irrigation_Need'].values
diff_idx  = np.where(diff_mask)[0]

df_diff = test.iloc[diff_idx].copy()
df_diff['pred_high'] = sub_high['Irrigation_Need'].values[diff_idx]
df_diff['pred_74']   = df74['Irrigation_Need'].values[diff_idx]

print(f"=== 140行不一致的特征分布 ===")
print(f"\n你(0.981) 预测分布:\n{df_diff['pred_high'].value_counts()}")
print(f"\ndf74 预测分布:\n{df_diff['pred_74'].value_counts()}")

print(f"\n=== 分歧方向 ===")
for a, b in [('Low','Medium'),('Low','High'),('Medium','High'),
             ('Medium','Low'),('High','Low'),('High','Medium')]:
    n = ((df_diff['pred_high']==a) & (df_diff['pred_74']==b)).sum()
    if n > 0:
        print(f"  你={a}, df74={b}: {n}行")

print(f"\n=== 关键特征统计（不一致行 vs 全体）===")
for col in ['Soil_Moisture','Temperature_C','Rainfall_mm','Wind_Speed_kmh']:
    print(f"\n{col}:")
    print(f"  不一致行: mean={df_diff[col].mean():.2f}, std={df_diff[col].std():.2f}")
    print(f"  全体测试: mean={test[col].mean():.2f}, std={test[col].std():.2f}")

print(f"\n=== Crop_Growth_Stage 分布 ===")
print(df_diff['Crop_Growth_Stage'].value_counts())
print(f"\n全体:")
print(test['Crop_Growth_Stage'].value_counts(normalize=True).round(4))

print(f"\n=== Mulching_Used 分布 ===")
print(df_diff['Mulching_Used'].value_counts())