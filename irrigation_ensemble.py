"""
Playground Series S6E4 — Irrigation Need Prediction
LightGBM + XGBoost + CatBoost Ensemble (本地运行版)

使用方法:
  1. 安装依赖: pip install lightgbm xgboost catboost scikit-learn pandas numpy optuna
  2. 下载比赛数据: kaggle competitions download -c playground-series-s6e4
  3. 下载原始数据: kaggle datasets download -d miadul/irrigation-water-requirement-prediction-dataset
  4. 修改下方 DATA_DIR / ORIG_CSV 路径
  5. python irrigation_ensemble.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

# ─────────────────────────────────────────────
#  ★ 修改这里：你的数据路径
# ─────────────────────────────────────────────
DATA_DIR = Path("./data")          # 存放 train.csv / test.csv 的文件夹
ORIG_CSV = Path("./data/irrigation_prediction.csv")  # 原始数据集 CSV
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED     = 42
N_FOLDS  = 5
TARGET   = "Irrigation_Need"


# ══════════════════════════════════════════════
#  1. 读取 & 清洗数据
# ══════════════════════════════════════════════
def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")
    orig  = pd.read_csv(ORIG_CSV)

    # 修复异常值
    for c in ["Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm"]:
        train[c] = pd.to_numeric(train[c], errors="coerce")

    # 删除 Soil_Type 以数字开头的坏行
    bad_mask = train["Soil_Type"].str.match(r"^\d", na=False)
    train = train[~bad_mask].reset_index(drop=True)

    test_ids = test["id"].copy()
    train = train.drop(columns="id")
    test  = test.drop(columns="id")

    # 合并原始数据集（重要！）
    train = pd.concat([train, orig], axis=0, ignore_index=True)
    train = train.dropna(subset=[TARGET]).reset_index(drop=True)

    print(f"训练集: {train.shape}  测试集: {test.shape}")
    return train, test, test_ids


# ══════════════════════════════════════════════
#  2. 特征工程（97 特征）
# ══════════════════════════════════════════════
def feature_engineering(X_train, X_test):
    NUM_COLS = X_train.select_dtypes(include="number").columns.tolist()
    CAT_COLS = X_train.select_dtypes(include="object").columns.tolist()

    # ── Tier 1: 领域交叉特征 ──
    def add_tier1(df):
        df = df.copy()
        df["ET_index"]      = df["Temperature_C"] * df["Sunlight_Hours"] / (df["Humidity"] + 1)
        df["water_supply"]  = df["Rainfall_mm"] + df["Previous_Irrigation_mm"]
        df["drought_index"] = (
            np.maximum(0, 800 - df["Rainfall_mm"])
            + np.maximum(0, 40 - df["Soil_Moisture"]) * 10
        ) / (df["Humidity"] + 1)
        df["water_stress"]  = df["ET_index"] / (df["water_supply"] + 1)
        df["moist_x_temp"]  = df["Soil_Moisture"] * df["Temperature_C"]
        df["rain_per_area"] = df["Rainfall_mm"] / (df["Field_Area_hectare"] + 0.1)
        return df

    X_train = add_tier1(X_train)
    X_test  = add_tier1(X_test)

    # ── Tier 2: 分组聚合特征 ──
    agg_nums = ["Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh", "Humidity"]
    agg_cats = ["Crop_Growth_Stage", "Crop_Type", "Region", "Season"]

    for cat in agg_cats:
        for num in agg_nums:
            grp      = X_train.groupby(cat)[num]
            mean_map = grp.mean()
            std_map  = grp.std().fillna(0)

            mean_col = f"{num}_mean_by_{cat}"
            std_col  = f"{num}_std_by_{cat}"
            diff_col = f"{num}_diff_from_{cat}_mean"

            X_train[mean_col] = X_train[cat].map(mean_map)
            X_test[mean_col]  = X_test[cat].map(mean_map).fillna(X_train[num].mean())

            X_train[std_col]  = X_train[cat].map(std_map)
            X_test[std_col]   = X_test[cat].map(std_map).fillna(X_train[num].std())

            X_train[diff_col] = X_train[num] - X_train[mean_col]
            X_test[diff_col]  = X_test[num]  - X_test[mean_col]

    # ── Tier 3: 频率编码 + 比率特征 ──
    for col in CAT_COLS:
        freq_map = X_train[col].value_counts(normalize=True)
        X_train[f"{col}_freq"] = X_train[col].map(freq_map)
        X_test[f"{col}_freq"]  = X_test[col].map(freq_map).fillna(0)

    X_train["moisture_humidity_ratio"] = X_train["Soil_Moisture"] / (X_train["Humidity"] + 1)
    X_test["moisture_humidity_ratio"]  = X_test["Soil_Moisture"]  / (X_test["Humidity"] + 1)

    X_train["temp_wind_product"] = X_train["Temperature_C"] * X_train["Wind_Speed_kmh"]
    X_test["temp_wind_product"]  = X_test["Temperature_C"]  * X_test["Wind_Speed_kmh"]

    rain_median = X_train["Rainfall_mm"].median()
    X_train["rain_deficit"] = rain_median - X_train["Rainfall_mm"]
    X_test["rain_deficit"]  = rain_median - X_test["Rainfall_mm"]

    X_train["soil_conductivity_ratio"] = X_train["Soil_Moisture"] / (X_train["Electrical_Conductivity"] + 0.1)
    X_test["soil_conductivity_ratio"]  = X_test["Soil_Moisture"]  / (X_test["Electrical_Conductivity"] + 0.1)

    print(f"特征工程完成: {X_train.shape[1]} 个特征")
    return X_train, X_test, CAT_COLS


# ══════════════════════════════════════════════
#  3. LightGBM
# ══════════════════════════════════════════════
def train_lgb(X_train, y_encoded, X_test):
    print("\n" + "="*50)
    print("训练 LightGBM ...")

    params = {
        "objective":        "multiclass",
        "num_class":        3,
        "metric":           "multi_logloss",
        "learning_rate":    0.05,
        "num_leaves":       127,
        "max_depth":        -1,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "class_weight":     "balanced",
        "n_jobs":           -1,
        "random_state":     SEED,
        "verbose":          -1,
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros((len(X_train), 3))
    test_preds = np.zeros((len(X_test), 3))
    fold_scores = []

    # Label-encode 类别列（LGB 也可以用 category 类型）
    X_tr = X_train.copy()
    X_te = X_test.copy()
    for col in X_tr.select_dtypes("object").columns:
        X_tr[col] = X_tr[col].astype("category")
        X_te[col] = X_te[col].astype("category")

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr, y_encoded), 1):
        X_trn, X_val = X_tr.iloc[trn_idx], X_tr.iloc[val_idx]
        y_trn, y_val = y_encoded[trn_idx],  y_encoded[val_idx]

        dtrain = lgb.Dataset(X_trn, label=y_trn)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params, dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
        )

        oof[val_idx]  = model.predict(X_val)
        test_preds   += model.predict(X_te) / N_FOLDS

        score = balanced_accuracy_score(y_val, oof[val_idx].argmax(axis=1))
        fold_scores.append(score)
        print(f"  Fold {fold}: BA = {score:.6f}  best_iter={model.best_iteration}")

    oof_score = balanced_accuracy_score(y_encoded, oof.argmax(axis=1))
    print(f"\nLightGBM OOF BA: {oof_score:.6f}  (±{np.std(fold_scores):.6f})")
    return oof, test_preds


# ══════════════════════════════════════════════
#  4. XGBoost
# ══════════════════════════════════════════════
def train_xgb(X_train, y_encoded, X_test, CAT_COLS):
    print("\n" + "="*50)
    print("训练 XGBoost ...")

    # Label-encode 类别列
    X_tr = X_train.copy()
    X_te = X_test.copy()
    le_dict = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        le.fit(pd.concat([X_tr[col], X_te[col]], axis=0).astype(str))
        X_tr[col] = le.transform(X_tr[col].astype(str))
        X_te[col] = le.transform(X_te[col].astype(str))
        le_dict[col] = le

    params = {
        "objective":         "multi:softprob",
        "num_class":         3,
        "eval_metric":       "mlogloss",
        "learning_rate":     0.05,
        "max_depth":         8,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "min_child_weight":  50,
        "tree_method":       "hist",   # CPU 版本用 hist
        "random_state":      SEED,
        "n_jobs":            -1,
        "verbosity":         0,
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros((len(X_tr), 3))
    test_preds = np.zeros((len(X_te), 3))
    fold_scores = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr, y_encoded), 1):
        X_trn, X_val = X_tr.iloc[trn_idx], X_tr.iloc[val_idx]
        y_trn, y_val = y_encoded[trn_idx],  y_encoded[val_idx]

        sw_trn = compute_sample_weight("balanced", y_trn)
        sw_val = compute_sample_weight("balanced", y_val)

        dtrain = xgb.DMatrix(X_trn, label=y_trn, weight=sw_trn)
        dval   = xgb.DMatrix(X_val, label=y_val, weight=sw_val)
        dtest  = xgb.DMatrix(X_te)

        model = xgb.train(
            params, dtrain,
            num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )

        oof[val_idx]  = model.predict(dval)
        test_preds   += model.predict(dtest) / N_FOLDS

        score = balanced_accuracy_score(y_val, oof[val_idx].argmax(axis=1))
        fold_scores.append(score)
        print(f"  Fold {fold}: BA = {score:.6f}  best_iter={model.best_iteration}")

    oof_score = balanced_accuracy_score(y_encoded, oof.argmax(axis=1))
    print(f"\nXGBoost OOF BA: {oof_score:.6f}  (±{np.std(fold_scores):.6f})")
    return oof, test_preds


# ══════════════════════════════════════════════
#  5. CatBoost
# ══════════════════════════════════════════════
def train_cat(X_train, y_encoded, X_test, CAT_COLS):
    print("\n" + "="*50)
    print("训练 CatBoost ...")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros((len(X_train), 3))
    test_preds = np.zeros((len(X_test), 3))
    fold_scores = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_encoded), 1):
        X_trn, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
        y_trn, y_val = y_encoded[trn_idx],     y_encoded[val_idx]

        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            bootstrap_type="Bernoulli",
            subsample=0.8,
            auto_class_weights="Balanced",
            cat_features=CAT_COLS,
            eval_metric="TotalF1:average=Macro",
            random_seed=SEED,
            verbose=0,
            early_stopping_rounds=100,
            task_type="GPU",   # 有 GPU 改成 "GPU"
        )

        model.fit(X_trn, y_trn, eval_set=(X_val, y_val))

        oof[val_idx]  = model.predict_proba(X_val)
        test_preds   += model.predict_proba(X_test) / N_FOLDS

        score = balanced_accuracy_score(y_val, oof[val_idx].argmax(axis=1))
        fold_scores.append(score)
        print(f"  Fold {fold}: BA = {score:.6f}  best_iter={model.best_iteration_}")

    oof_score = balanced_accuracy_score(y_encoded, oof.argmax(axis=1))
    print(f"\nCatBoost OOF BA: {oof_score:.6f}  (±{np.std(fold_scores):.6f})")
    return oof, test_preds


# ══════════════════════════════════════════════
#  6. 集成：网格搜索最优权重
# ══════════════════════════════════════════════
def find_best_weights(oof_lgb, oof_xgb, oof_cb, y_encoded):
    print("\n" + "="*50)
    print("网格搜索最优集成权重 ...")

    best_ba = 0
    best_w  = (1/3, 1/3, 1/3)

    for w_lgb in np.arange(0.0, 1.01, 0.05):
        for w_xgb in np.arange(0.0, 1.01 - w_lgb, 0.05):
            w_cb = round(1.0 - w_lgb - w_xgb, 2)
            if w_cb < 0:
                continue
            ens = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb
            ba  = balanced_accuracy_score(y_encoded, ens.argmax(axis=1))
            if ba > best_ba:
                best_ba = ba
                best_w  = (round(w_lgb, 2), round(w_xgb, 2), w_cb)

    print(f"最优权重: LGB={best_w[0]}, XGB={best_w[1]}, CB={best_w[2]}")
    print(f"集成 OOF BA: {best_ba:.6f}")
    return best_w, best_ba


# ══════════════════════════════════════════════
#  7. Threshold 优化
# ══════════════════════════════════════════════
def optimize_threshold(oof_ens, y_encoded, base_ba):
    print("\n" + "="*50)
    print("Threshold 优化 ...")

    best_ba   = base_ba
    best_h    = 1.0
    best_m    = 1.0

    for h_mult in np.arange(1.0, 5.01, 0.1):
        for m_mult in np.arange(0.4, 1.21, 0.05):
            adj = oof_ens.copy()
            adj[:, 0] *= h_mult   # High
            adj[:, 2] *= m_mult   # Medium
            ba = balanced_accuracy_score(y_encoded, adj.argmax(axis=1))
            if ba > best_ba:
                best_ba = ba
                best_h  = round(h_mult, 2)
                best_m  = round(m_mult, 2)

    print(f"最优 Threshold: High×{best_h}, Medium×{best_m}")
    print(f"优化后 OOF BA: {best_ba:.6f}")
    return best_h, best_m, best_ba


# ══════════════════════════════════════════════
#  8. 主流程
# ══════════════════════════════════════════════
def main():
    # 读数据
    train, test, test_ids = load_data()

    # 分离特征 / 标签
    y_train  = train[TARGET].copy()
    X_train  = train.drop(columns=TARGET)
    X_test   = test.copy()

    CAT_COLS = X_train.select_dtypes(include="object").columns.tolist()

    # 特征工程
    X_train, X_test, CAT_COLS = feature_engineering(X_train, X_test)

    # 编码标签
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y_train)
    class_names = le_target.classes_
    print(f"类别: {list(class_names)}")

    # 训练三个模型
    oof_lgb,  test_lgb  = train_lgb(X_train, y_encoded, X_test)
    oof_xgb,  test_xgb  = train_xgb(X_train, y_encoded, X_test, CAT_COLS)
    oof_cb,   test_cb   = train_cat(X_train, y_encoded, X_test, CAT_COLS)

    # 保存 OOF（方便后续 stacking）
    np.save(OUTPUT_DIR / "oof_lgb.npy",  oof_lgb)
    np.save(OUTPUT_DIR / "oof_xgb.npy",  oof_xgb)
    np.save(OUTPUT_DIR / "oof_cb.npy",   oof_cb)
    np.save(OUTPUT_DIR / "test_lgb.npy", test_lgb)
    np.save(OUTPUT_DIR / "test_xgb.npy", test_xgb)
    np.save(OUTPUT_DIR / "test_cb.npy",  test_cb)

    # 搜索最优权重
    best_w, ens_ba = find_best_weights(oof_lgb, oof_xgb, oof_cb, y_encoded)
    w_lgb, w_xgb, w_cb = best_w

    oof_ens  = w_lgb * oof_lgb  + w_xgb * oof_xgb  + w_cb * oof_cb
    test_ens = w_lgb * test_lgb + w_xgb * test_xgb + w_cb * test_cb

    # Threshold 优化
    best_h, best_m, final_ba = optimize_threshold(oof_ens, y_encoded, ens_ba)

    # 生成提交文件
    # 方案1：默认 argmax
    sub1 = pd.DataFrame({
        "id":              test_ids,
        "Irrigation_Need": le_target.inverse_transform(test_ens.argmax(axis=1)),
    })
    sub1.to_csv(OUTPUT_DIR / "submission_default.csv", index=False)

    # 方案2：threshold 优化后
    adj_test = test_ens.copy()
    adj_test[:, 0] *= best_h
    adj_test[:, 2] *= best_m
    sub2 = pd.DataFrame({
        "id":              test_ids,
        "Irrigation_Need": le_target.inverse_transform(adj_test.argmax(axis=1)),
    })
    sub2.to_csv(OUTPUT_DIR / "submission_threshold.csv", index=False)

    # 最终汇总
    print("\n" + "="*60)
    print("  最终结果汇总")
    print("="*60)
    print(f"  LightGBM OOF BA:          {balanced_accuracy_score(y_encoded, oof_lgb.argmax(1)):.6f}")
    print(f"  XGBoost  OOF BA:          {balanced_accuracy_score(y_encoded, oof_xgb.argmax(1)):.6f}")
    print(f"  CatBoost OOF BA:          {balanced_accuracy_score(y_encoded, oof_cb.argmax(1)):.6f}")
    print(f"  集成 (weights={best_w}): {ens_ba:.6f}")
    print(f"  集成 + Threshold:         {final_ba:.6f}")
    print(f"\n  提交文件:")
    print(f"    output/submission_default.csv   <- 集成默认版")
    print(f"    output/submission_threshold.csv <- threshold优化版 (推荐提交)")


if __name__ == "__main__":
    main()
