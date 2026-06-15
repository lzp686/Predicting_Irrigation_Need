"""
PS-S6E4 EoS Voting (0.981) - 本地版
前置条件：需要从 Kaggle 下载以下文件

数据集 ps-s6e4-07 (nina2025/ps-s6e4-07):
  - 0.97971.a.csv
  - 0.97971.b.csv
  - 0.97971.c.csv
  - 0.97971.d.csv
  - 0.97971.x.csv
  - 0.98010.csv

数据集 ps-s6e4-74 (nina2025/ps-s6e4-74):
  - 5(8) - 0.98057.csv
  - 5(4) - 0.98074.csv

数据集 ps-s6e4-85 (nina2025/ps-s6e4-85):
  - 5(4) - 0.98074.csv
  - 5(9) - 0.98072.csv
  - Aux - 0.97254.csv

竞赛数据 (playground-series-s6e4):
  - sample_submission.csv

下载命令：
  kaggle datasets download nina2025/ps-s6e4-07 -p ./data/pd7
  kaggle datasets download nina2025/ps-s6e4-74 -p ./data/pd74
  kaggle datasets download nina2025/ps-s6e4-85 -p ./data/pd85
  kaggle competitions download -c playground-series-s6e4 -p ./data/pps --file sample_submission.csv
"""

import os
import pandas as pd

# ===== 路径配置，改成你自己的路径 =====
PPS  = './data/'    # 竞赛数据
PD7  = './data/pd7/'    # nina2025/ps-s6e4-07
PD74 = './data/pd74/'   # nina2025/ps-s6e4-74
PD85 = './data/pd85/'   # nina2025/ps-s6e4-85
OUT  = './output/'
os.makedirs(OUT, exist_ok=True)

# ===== 辅助函数 =====
def merge(list_df):
    dfs = pd.merge(list_df[0], list_df[1], on='id')
    for i in range(2, len(list_df)):
        dfs = pd.merge(dfs, list_df[i], on='id')
    return dfs

def microEDA(x, list_names_df):
    lwh = []
    for _p in list_names_df:
        wh = x[_p][0:1]
        if wh == 'M': wh = '_'
        lwh.append(f'{wh} ')
    return ''.join(lwh)

def left4(x):
    return '=' if x['7971a'] == x['7971b'] == x['7971c'] == x['7971d'] else '!='

def most_common_element(x):
    preds = x['mEDA_with_Aux'].split(' ')
    preds = [p for p in preds if p]
    mce = max(set(preds), key=preds.count)
    if mce == 'L': return 'Low'
    if mce == 'H': return 'High'
    return 'Medium'

# ===== 读取 sample_submission =====
sub = pd.read_csv(PPS + 'sample_submission.csv')
print(f"sample_submission shape: {sub.shape}")

# ===== Part 1: 构建 transfer2 (LB≈0.98072) =====
print("\n--- Part 1: Building transfer2 ---")

df1 = pd.read_csv(PD7 + '0.97971.a.csv').rename(columns={'Irrigation_Need': '7971a'})
df2 = pd.read_csv(PD7 + '0.97971.b.csv').rename(columns={'Irrigation_Need': '7971b'})
df3 = pd.read_csv(PD7 + '0.97971.c.csv').rename(columns={'Irrigation_Need': '7971c'})
df4 = pd.read_csv(PD7 + '0.97971.d.csv').rename(columns={'Irrigation_Need': '7971d'})
df5 = pd.read_csv(PD7 + '0.97971.x.csv').rename(columns={'Irrigation_Need': '7971'})
df6 = pd.read_csv(PD74 + '5(8) - 0.98057.csv').rename(columns={'Irrigation_Need': '8057'})

dfs = merge([df1, df2, df3, df4, df5, df6])
dfs['left4'] = dfs.apply(left4, axis=1)
print(f"left4 value counts:\n{dfs['left4'].value_counts()}")

def transfer2(x):
    if x['left4'] == '!=': return x['7971']
    else:                   return x['8057']

dfs['transfer2'] = dfs.apply(transfer2, axis=1)
sub['Irrigation_Need'] = dfs['transfer2']
print(f"transfer2 done. Value counts:\n{sub['Irrigation_Need'].value_counts()}")

# ===== 🛢️ Barrel Slice: df74 前半段 + sub 后半段 (LB≈0.98088) =====
print("\n--- Barrel Slice (v9, LB≈0.98088) ---")

df74 = pd.read_csv(PD74 + '5(4) - 0.98074.csv')
cut = 137_400
df_barrel = pd.concat([
    df74.iloc[0:cut].reset_index(drop=True),
    sub.iloc[cut:].reset_index(drop=True)
], axis=0).reset_index(drop=True)

df_barrel.to_csv(OUT + 'submission_barrel_0.98088.csv', index=False)
print(f"Saved: {OUT}submission_barrel_0.98088.csv  shape={df_barrel.shape}")

# ===== Part 2: Aux-assisted correction (LB≈0.981) =====
print("\n--- Part 2: Aux-assisted correction ---")

df74_2  = pd.read_csv(PD85 + '5(4) - 0.98074.csv').rename(columns={'Irrigation_Need': 'df74'})
df72    = pd.read_csv(PD85 + '5(9) - 0.98072.csv').rename(columns={'Irrigation_Need': 'df72'})
dfAux1  = pd.read_csv(PD85 + 'Aux - 0.97254.csv').rename(columns={'Irrigation_Need': 'Aux1'})

dfs2 = pd.merge(df74_2, df72, on='id')
dfs2['mEDA'] = dfs2.apply(lambda x: microEDA(x, ['df74', 'df72']), axis=1)
print(f"mEDA distribution:\n{dfs2['mEDA'].value_counts()}")

# 找出 df74 与 df72 不一致的行
dfLMH = dfs2[dfs2['df74'] != dfs2['df72']].copy()
print(f"\nDiffering rows: {dfLMH.shape[0]}")

# 加入 Aux 辅助文件做多数票
dfLMH = pd.merge(dfLMH, dfAux1, on='id')
dfLMH['mEDA_with_Aux'] = dfLMH.apply(
    lambda x: microEDA(x, ['df74', 'df72', 'Aux1']), axis=1
)
dfLMH['help_Aux'] = dfLMH.apply(most_common_element, axis=1)

print(f"\nAux correction distribution:\n{dfLMH['mEDA_with_Aux'].value_counts()}")

# 合并回主表
dfs2 = pd.merge(dfs2, dfLMH[['id', 'help_Aux']], on='id', how='left')

def apply_help_Aux(x):
    if x['df74'] == x['df72']: return x['df74']
    else:                       return x['help_Aux']

dfs2['Irrigation_Need'] = dfs2.apply(apply_help_Aux, axis=1)

sub2 = pd.read_csv(PPS + 'sample_submission.csv')
sub2['Irrigation_Need'] = dfs2['Irrigation_Need']
sub2.to_csv(OUT + 'submission_part2_0.981.csv', index=False)
print(f"\nSaved: {OUT}submission_part2_0.981.csv")
print(f"Final distribution:\n{sub2['Irrigation_Need'].value_counts()}")

print("\n===== Done! =====")
print(f"  {OUT}submission_barrel_0.98088.csv  <- 推荐提交（Part1结果）")
print(f"  {OUT}submission_part2_0.981.csv     <- Part2结果（依赖 pd85 数据集）")