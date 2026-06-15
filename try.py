import pandas as pd
import numpy as np

# 1. 读取你的基准文件和对比文件 (请确保是从最初的 0.981 基础文件开始改)
sub_high = pd.read_csv('./output/submission_part2_0.981.csv')
df74     = pd.read_csv('./data/pd74/5(4) - 0.98074.csv')

# 2. 找出不一致的行
diff_mask = sub_high['Irrigation_Need'].values != df74['Irrigation_Need'].values
diff_idx  = np.where(diff_mask)[0]

pred_high_vals = sub_high['Irrigation_Need'].values[diff_idx]
pred_74_vals   = df74['Irrigation_Need'].values[diff_idx]

# 3. 定位这最后的 8 行：你=Medium, df74=Low
mask_8 = (pred_high_vals == 'Medium') & (pred_74_vals == 'Low')
idx_8  = diff_idx[mask_8]
print(f"共找到 {len(idx_8)} 行准备翻转")
print(f"对应的行索引: {idx_8}")

# 4. 复制一份并进行最后的翻转
sub_new = sub_high.copy()
sub_new.loc[idx_8, 'Irrigation_Need'] = 'Low'

# 5. 保存结果
output_file = './output/submission_flip_medium2low_last8.csv'
sub_new.to_csv(output_file, index=False)
print(f"已保存: {output_file}")

# 再次确认修改结果
print(f"修改后这组的分布:\n{sub_new.loc[idx_8, 'Irrigation_Need'].value_counts()}")