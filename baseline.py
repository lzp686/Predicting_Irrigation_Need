# ============================================================
# Kaggle Playground S6E4 - Predicting Irrigation Need
# Local Baseline with LightGBM + 5-Fold CV
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置：修改 DATA_DIR 为你的数据路径
# ============================================================
DATA_DIR   = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET   = None   # 自动检测，无需修改
N_FOLDS  = 5
SEED     = 42

# ============================================================
# 1. 读取数据
# ============================================================
print("=" * 50)
print("1. 读取数据")
print("=" * 50)

train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")
sub   = pd.read_csv(DATA_DIR / "sample_submission.csv")

print(f"Train: {train.shape}")
print(f"Test:  {test.shape}")
print(f"\n列名: {train.columns.tolist()}")

# 自动检测目标列（sample_submission 中不是 id 的那列）
sub_cols = sub.columns.tolist()
id_col   = sub_cols[0]
TARGET   = sub_cols[1]
print(f"\n自动检测 → ID列: '{id_col}', 目标列: '{TARGET}'")

# ============================================================
# 2. EDA 概览
# ============================================================
print("\n" + "=" * 50)
print("2. EDA 概览")
print("=" * 50)

print("\n--- 数据类型 ---")
print(train.dtypes)

print("\n--- 缺失值 ---")
missing = train.isnull().sum()
print(missing[missing > 0] if missing.any() else "无缺失值 ✓")

print(f"\n--- 目标变量分布 ---")
print(train[TARGET].value_counts())

# ============================================================
# 3. 预处理
# ============================================================
print("\n" + "=" * 50)
print("3. 预处理")
print("=" * 50)

# 保存 id，从特征中删除
train_ids = train[id_col]
test_ids  = test[id_col]

X = train.drop(columns=[id_col, TARGET])
y = train[TARGET].copy()
X_test = test.drop(columns=[id_col])

# 类别特征编码
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"数值特征 ({len(num_cols)}): {num_cols}")
print(f"类别特征 ({len(cat_cols)}): {cat_cols}")

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col]      = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    le_dict[col] = le

# 目标变量编码（若为字符串）
le_target = None
if y.dtype == 'object':
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    print(f"\n目标类别: {le_target.classes_}")

n_classes  = len(np.unique(y))
IS_BINARY  = (n_classes == 2)
print(f"\n类别数: {n_classes} → {'二分类' if IS_BINARY else '多分类'}")

# ============================================================
# 4. LightGBM 5折交叉验证
# ============================================================
print("\n" + "=" * 50)
print("4. LightGBM 训练（5折 CV）")
print("=" * 50)

lgb_params = {
    'objective':        'binary' if IS_BINARY else 'multiclass',
    'metric':           'binary_logloss' if IS_BINARY else 'multi_logloss',
    'num_class':        1 if IS_BINARY else n_classes,
    'n_estimators':     1000,
    'learning_rate':    0.05,
    'num_leaves':       63,
    'max_depth':        -1,
    'min_child_samples': 20,
    'subsample':        0.8,
    'subsample_freq':   1,
    'colsample_bytree': 0.8,
    'reg_alpha':        0.1,
    'reg_lambda':       1.0,
    'random_state':     SEED,
    'n_jobs':           -1,
    'verbose':          -1,
}

# OOF & 测试集预测矩阵
if IS_BINARY:
    oof_proba  = np.zeros(len(X))
    test_proba = np.zeros(len(X_test))
else:
    oof_proba  = np.zeros((len(X), n_classes))
    test_proba = np.zeros((len(X_test), n_classes))

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_scores  = []
best_models  = []

for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"\n  Fold {fold+1}/{N_FOLDS}")
    
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx],      y[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ]
    )

    if IS_BINARY:
        val_pred           = model.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = val_pred
        test_proba        += model.predict_proba(X_test)[:, 1] / N_FOLDS
    else:
        val_pred           = model.predict_proba(X_val)
        oof_proba[val_idx] = val_pred
        test_proba        += model.predict_proba(X_test) / N_FOLDS

    score = log_loss(y_val, val_pred)
    acc   = accuracy_score(y_val, model.predict(X_val))
    fold_scores.append(score)
    best_models.append(model)
    print(f"  → Log Loss: {score:.5f} | Accuracy: {acc:.5f} | 最佳轮次: {model.best_iteration_}")

print(f"\n{'='*50}")
print(f"OOF Log Loss: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")

# OOF Accuracy
oof_pred_label = (oof_proba > 0.5).astype(int) if IS_BINARY else np.argmax(oof_proba, axis=1)
print(f"OOF Accuracy: {accuracy_score(y, oof_pred_label):.5f}")

# ============================================================
# 5. 特征重要性
# ============================================================
print("\n" + "=" * 50)
print("5. 特征重要性")
print("=" * 50)

# 用最后一个 fold 的模型
last_model = best_models[-1]
feat_imp = pd.DataFrame({
    'feature':    X.columns,
    'importance': last_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feat_imp.to_string(index=False))

plt.figure(figsize=(10, max(4, len(feat_imp) * 0.35)))
sns.barplot(data=feat_imp.head(20), x='importance', y='feature', palette='viridis')
plt.title('Top 20 Feature Importances (LightGBM)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150)
plt.show()
print(f"\n特征重要性图已保存 → output/feature_importance.png")

# ============================================================
# 6. 生成提交文件
# ============================================================
print("\n" + "=" * 50)
print("6. 生成提交文件")
print("=" * 50)

if IS_BINARY:
    final_pred = (test_proba > 0.5).astype(int)
else:
    final_pred = np.argmax(test_proba, axis=1)

# 若目标列原来是字符串，反编码回来
if le_target is not None:
    final_pred = le_target.inverse_transform(final_pred)

sub[TARGET] = final_pred
sub.to_csv(OUTPUT_DIR / "submission.csv", index=False)

print(f"提交文件已保存 → output/submission.csv")
print(f"\n预测分布:\n{pd.Series(final_pred).value_counts()}")
print("\n前5行预览:")
print(sub.head())