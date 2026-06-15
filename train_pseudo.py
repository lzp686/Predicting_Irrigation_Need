import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# =========================================================
# === 1. 路径配置 (请修改为你的实际路径) ===
# =========================================================
DATA_DIR = './data/'
NPY_DIR = './data/npy/'

TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
# 你的 0.98114 终极高分预测文件 (用于提供伪标签基准)
PSEUDO_LABEL_PATH = os.path.join(DATA_DIR, 'csv', 'eos11_0.98151.csv')

os.makedirs(NPY_DIR, exist_ok=True)

# =========================================================
# === 2. 数据读取与目标编码 ===
# =========================================================
print(">>> [1/5] 读取数据...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
pseudo_sub = pd.read_csv(PSEUDO_LABEL_PATH)

# 将伪标签拼接给 test
test['Irrigation_Need'] = pseudo_sub['Irrigation_Need']

target_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
train['target'] = train['Irrigation_Need'].map(target_mapping)
test['target'] = test['Irrigation_Need'].map(target_mapping)

# =========================================================
# === 3. 【核心】应用完美公式为测试集进行“置信度体检” ===
# =========================================================
print(">>> [2/5] 使用原始完美公式计算测试集置信度，执行去毒过滤...")
# 构造公式所需的二值化特征
test['soil_lt_25'] = (test['Soil_Moisture'] < 25).astype(int)
test['temp_gt_30'] = (test['Temperature_C'] > 30).astype(int)  # 注意这里是 Temperature_C
test['rain_lt_300'] = (test['Rainfall_mm'] < 300).astype(int)
test['wind_gt_10'] = (test['Wind_Speed_kmh'] > 10).astype(int)

cgs_F = (test['Crop_Growth_Stage'] == 'Flowering').astype(int)
cgs_H = (test['Crop_Growth_Stage'] == 'Harvest').astype(int)
cgs_S = (test['Crop_Growth_Stage'] == 'Sowing').astype(int)
cgs_V = (test['Crop_Growth_Stage'] == 'Vegetative').astype(int)

mulch_N = (test['Mulching_Used'] == 'No').astype(int)
mulch_Y = (test['Mulching_Used'] == 'Yes').astype(int)

# 计算 Logits
logit_low = 16.3173 - 11.0237 * test['soil_lt_25'] - 5.8559 * test['temp_gt_30'] \
            - 10.8500 * test['rain_lt_300'] - 5.8284 * test['wind_gt_10'] \
            - 5.4155 * cgs_F + 5.5073 * cgs_H + 5.2299 * cgs_S - 5.4617 * cgs_V \
            - 3.0014 * mulch_N + 2.8613 * mulch_Y

logit_med = 4.6524 + 0.3290 * test['soil_lt_25'] - 0.0204 * test['temp_gt_30'] \
            + 0.1542 * test['rain_lt_300'] + 0.0841 * test['wind_gt_10'] \
            + 0.3586 * cgs_F - 0.1348 * cgs_H - 0.3547 * cgs_S + 0.3334 * cgs_V \
            + 0.1883 * mulch_N + 0.0142 * mulch_Y

logit_high = -20.9697 + 10.6947 * test['soil_lt_25'] + 5.8763 * test['temp_gt_30'] \
             + 10.6958 * test['rain_lt_300'] + 5.7444 * test['wind_gt_10'] \
             + 5.0569 * cgs_F - 5.3725 * cgs_H - 4.8752 * cgs_S + 5.1283 * cgs_V \
             + 2.8131 * mulch_N - 2.8755 * mulch_Y

# 使用 Softmax 计算最终概率
exp_low = np.exp(logit_low)
exp_med = np.exp(logit_med)
exp_high = np.exp(logit_high)
sum_exp = exp_low + exp_med + exp_high

test['prob_Low'] = exp_low / sum_exp
test['prob_Medium'] = exp_med / sum_exp
test['prob_High'] = exp_high / sum_exp

# 提取公式算出的最大概率 (置信度)
test['formula_max_prob'] = test[['prob_Low', 'prob_Medium', 'prob_High']].max(axis=1)

# === 去毒操作：只保留置信度 > 0.90 的强力样本 ===
CONFIDENCE_THRESHOLD = 0.90
confident_mask = test['formula_max_prob'] > CONFIDENCE_THRESHOLD
confident_test = test[confident_mask].copy()

print(f"   - 原始伪标签总数: {len(test)}")
print(f"   - 过滤后高置信度伪标签数: {len(confident_test)} (剔除了 {len(test) - len(confident_test)} 个有毒边界样本)")

# 清理辅助列
drop_cols = ['id', 'Irrigation_Need', 'target', 'soil_lt_25', 'temp_gt_30', 'rain_lt_300', 'wind_gt_10', 'prob_Low', 'prob_Medium', 'prob_High', 'formula_max_prob']
features = [c for c in train.columns if c not in drop_cols]

# =========================================================
# === 4. 原版特征工程对齐 ===
# =========================================================
print(">>> [3/5] 执行类别特征编码...")
cat_cols = train[features].select_dtypes(include=['object']).columns.tolist()

for col in cat_cols:
    le = LabelEncoder()
    train_vals = train[col].fillna('nan').astype(str)
    test_vals = test[col].fillna('nan').astype(str)
    confident_vals = confident_test[col].fillna('nan').astype(str)
    
    # 结合 train 和 test 整体 fit
    le.fit(pd.concat([train_vals, test_vals]))
    
    # 第一步：先用 LabelEncoder 转换并赋值 (此时是 NumPy 默认的 int 类型)
    train[col] = le.transform(train_vals)
    test[col] = le.transform(test_vals)
    confident_test[col] = le.transform(confident_vals)
    
    # 第二步：再使用 Pandas 将整列转换为 category 类型
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')
    confident_test[col] = confident_test[col].astype('category')

X = train[features]
y = train['target']
X_test = test[features] # 用于最后生成 test_preds
X_confident_pseudo = confident_test[features] # 用于混入训练集
y_confident_pseudo = confident_test['target']

# =========================================================
# === 5. 严格对齐原版模型参数 ===
# =========================================================
RANDOM_STATE = 2024 
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

lgb_params = {
    'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
    'learning_rate': 0.05, 'n_estimators': 2000, 'colsample_bytree': 0.7,
    'subsample': 0.7, 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1
}

xgb_params = {
    'objective': 'multi:softprob', 'num_class': 3, 'eval_metric': 'mlogloss',
    'learning_rate': 0.05, 'n_estimators': 2000, 'colsample_bytree': 0.7,
    'subsample': 0.7, 'random_state': RANDOM_STATE, 'n_jobs': -1,
    'enable_categorical': True, 'tree_method': 'hist'
}

# =========================================================
# === 6. 开始交叉验证：折内【安全】注入纯净伪标签 ===
# =========================================================
oof_lgb = np.zeros((len(train), 3))
test_preds_lgb = np.zeros((len(test), 3))

oof_xgb = np.zeros((len(train), 3))
test_preds_xgb = np.zeros((len(test), 3))

print(f">>> [4/5] 开始 {n_splits} 折交叉验证训练 (结合去毒版伪标签)...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n========== Fold {fold + 1} ==========")
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    
    # 将过滤后的【高置信度伪标签】追加到本折的训练集中
    X_train_combined = pd.concat([X_train_fold, X_confident_pseudo], axis=0).reset_index(drop=True)
    y_train_combined = pd.concat([y_train_fold, y_confident_pseudo], axis=0).reset_index(drop=True)
    
    # 彻底打乱混合后的数据
    shuffle_idx = np.random.RandomState(RANDOM_STATE + fold).permutation(len(X_train_combined))
    X_train_combined = X_train_combined.iloc[shuffle_idx]
    y_train_combined = y_train_combined.iloc[shuffle_idx]
    
    print(f"训练集规模: {len(X_train_combined)} (原训练集 {len(X_train_fold)} + 去毒伪标签 {len(X_confident_pseudo)})")
    
    print("➡️ Training LightGBM...")
    lgb_model = LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train_combined, y_train_combined, eval_set=[(X_val_fold, y_val_fold)], callbacks=[])
    
    oof_lgb[val_idx] = lgb_model.predict_proba(X_val_fold)
    test_preds_lgb += lgb_model.predict_proba(X_test) / n_splits
    
    print("➡️ Training XGBoost...")
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(X_train_combined, y_train_combined, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
    
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val_fold)
    test_preds_xgb += xgb_model.predict_proba(X_test) / n_splits

# =========================================================
# === 7. 评估与输出 ===
# =========================================================
print("\n>>> [5/5] 训练完成！评估最终纯净 OOF 准确率...")
lgb_acc = np.mean(np.argmax(oof_lgb, axis=1) == y.values)
xgb_acc = np.mean(np.argmax(oof_xgb, axis=1) == y.values)
print(f"🌟 LightGBM 终极去毒版 OOF Accuracy: {lgb_acc:.6f}")
print(f"🌟 XGBoost  终极去毒版 OOF Accuracy: {xgb_acc:.6f}")

np.save(os.path.join(NPY_DIR, 'oof_preds_lgb.npy'), oof_lgb)
np.save(os.path.join(NPY_DIR, 'test_preds_lgb.npy'), test_preds_lgb)
np.save(os.path.join(NPY_DIR, 'oof_preds_xgb.npy'), oof_xgb)
np.save(os.path.join(NPY_DIR, 'test_preds_xgb.npy'), test_preds_xgb)

print("\n✅ 去毒伪标签概率矩阵生成完毕！现在回到 lgb.py 再做一次 Optuna 融合！")