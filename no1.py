"""
Playground Series S6E4 — Irrigation Need Prediction
优化版：利用原始数据精确公式 + LightGBM + XGBoost + CatBoost 集成

核心改进：
  1. 原始数据用规则公式直接打标（BA=1），作为"完美锚点"加入训练
  2. 将规则公式的中间变量作为强特征加入模型
  3. 测试集预测时用规则公式概率与模型概率 blend
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
DATA_DIR   = Path("./data")
ORIG_CSV   = Path("./data/irrigation_prediction.csv")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED    = 42
N_FOLDS = 5
TARGET  = "Irrigation_Need"


# ══════════════════════════════════════════════
#  规则公式：原始数据的完美分类器
# ══════════════════════════════════════════════
def apply_rule_formula(df):
    """
    根据社区发现的精确公式计算 High/Low score 及最终标签。
    对原始数据 BA=1，对比赛数据作为强特征使用。
    """
    df = df.copy()

    # 中间二值特征
    df["soil_lt_25"]  = (df["Soil_Moisture"]    < 25).astype(int)
    df["temp_gt_30"]  = (df["Temperature_C"]    > 30).astype(int)
    df["rain_lt_300"] = (df["Rainfall_mm"]       < 300).astype(int)
    df["wind_gt_10"]  = (df["Wind_Speed_kmh"]   > 10).astype(int)
    df["is_harvest"]  = (df["Crop_Growth_Stage"] == "Harvest").astype(int)
    df["is_sowing"]   = (df["Crop_Growth_Stage"] == "Sowing").astype(int)
    df["is_flower"]   = (df["Crop_Growth_Stage"] == "Flowering").astype(int)
    df["is_veg"]      = (df["Crop_Growth_Stage"] == "Vegetative").astype(int)
    df["mulch_yes"]   = (df["Mulching_Used"]     == "Yes").astype(int)
    df["mulch_no"]    = (df["Mulching_Used"]     == "No").astype(int)

    # High / Low 得分
    df["rule_high_score"] = (
        2 * df["soil_lt_25"]  +
        2 * df["rain_lt_300"] +
        1 * df["temp_gt_30"]  +
        1 * df["wind_gt_10"]
    )
    df["rule_low_score"] = (
        2 * df["is_harvest"] +
        2 * df["is_sowing"]  +
        1 * df["mulch_yes"]
    )
    df["rule_score"] = df["rule_high_score"] - df["rule_low_score"]

    # 规则标签（0=High,1=Low,2=Medium — 与 LabelEncoder 对齐后再用）
    def score_to_label(s):
        if s <= 0:   return "Low"
        elif s <= 3: return "Medium"
        else:        return "High"
    df["rule_label"] = df["rule_score"].apply(score_to_label)

    # Logit 概率（精确公式）
    def logit_probs(row):
        s, t, r, w = row["soil_lt_25"], row["temp_gt_30"], row["rain_lt_300"], row["wind_gt_10"]
        cgs_fl = row["is_flower"]; cgs_ha = row["is_harvest"]
        cgs_so = row["is_sowing"]; cgs_ve = row["is_veg"]
        mu_no  = row["mulch_no"];  mu_ye  = row["mulch_yes"]

        lo_low = (16.3173 - 11.0237*s - 5.8559*t - 10.8500*r - 5.8284*w
                  - 5.4155*cgs_fl + 5.5073*cgs_ha + 5.2299*cgs_so - 5.4617*cgs_ve
                  - 3.0014*mu_no + 2.8613*mu_ye)
        lo_med = (4.6524  + 0.3290*s  - 0.0204*t +  0.1542*r + 0.0841*w
                  + 0.3586*cgs_fl - 0.1348*cgs_ha - 0.3547*cgs_so + 0.3334*cgs_ve
                  + 0.1883*mu_no + 0.0142*mu_ye)
        lo_hi  = (-20.9697 + 10.6947*s + 5.8763*t + 10.6958*r + 5.7444*w
                  + 5.0569*cgs_fl - 5.3725*cgs_ha - 4.8752*cgs_so + 5.1283*cgs_ve
                  + 2.8131*mu_no - 2.8755*mu_ye)

        exp_lo, exp_me, exp_hi = np.exp(lo_low), np.exp(lo_med), np.exp(lo_hi)
        denom = exp_lo + exp_me + exp_hi
        # 顺序对应 LabelEncoder: High=0, Low=1, Medium=2
        return exp_hi/denom, exp_lo/denom, exp_me/denom

    probs = df.apply(logit_probs, axis=1, result_type="expand")
    probs.columns = ["rule_prob_high", "rule_prob_low", "rule_prob_med"]
    df = pd.concat([df, probs], axis=1)

    return df


# ══════════════════════════════════════════════
#  读取 & 清洗数据
# ══════════════════════════════════════════════
def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")
    orig  = pd.read_csv(ORIG_CSV)

    for c in ["Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm"]:
        train[c] = pd.to_numeric(train[c], errors="coerce")

    bad_mask = train["Soil_Type"].str.match(r"^\d", na=False)
    train = train[~bad_mask].reset_index(drop=True)

    test_ids = test["id"].copy()
    train = train.drop(columns="id")
    test  = test.drop(columns="id")

    # ── 关键改进1：对原始数据用规则公式直接打标，不依赖原始 TARGET 列噪声 ──
    orig_with_rule = apply_rule_formula(orig)
    # 用规则标签覆盖原始标签（规则在原始数据上 BA=1）
    orig[TARGET] = orig_with_rule["rule_label"]

    train = pd.concat([train, orig], axis=0, ignore_index=True)
    train = train.dropna(subset=[TARGET]).reset_index(drop=True)

    print(f"训练集: {train.shape}  测试集: {test.shape}")
    return train, test, test_ids


# ══════════════════════════════════════════════
#  特征工程
# ══════════════════════════════════════════════
def feature_engineering(X_train, X_test):
    NUM_COLS = X_train.select_dtypes(include="number").columns.tolist()
    CAT_COLS = X_train.select_dtypes(include="object").columns.tolist()

    # ── 关键改进2：先加入规则特征（最强信号！）──
    X_train = apply_rule_formula(X_train)
    X_test  = apply_rule_formula(X_test)

    # 规则特征列（已在 apply_rule_formula 中加入）
    rule_feat_cols = [
        "soil_lt_25", "temp_gt_30", "rain_lt_300", "wind_gt_10",
        "is_harvest", "is_sowing", "is_flower", "is_veg",
        "mulch_yes", "mulch_no",
        "rule_high_score", "rule_low_score", "rule_score",
        "rule_prob_high", "rule_prob_low", "rule_prob_med",
    ]
    # rule_label 是字符串，不直接作为特征（已编码进 rule_prob_* 了）
    X_train = X_train.drop(columns=["rule_label"])
    X_test  = X_test.drop(columns=["rule_label"])

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

    # ── 关键改进3：规则score的交叉特征 ──
    X_train["rule_score_x_moisture"] = X_train["rule_score"] * X_train["Soil_Moisture"]
    X_test["rule_score_x_moisture"]  = X_test["rule_score"]  * X_test["Soil_Moisture"]
    X_train["rule_score_x_rain"]     = X_train["rule_score"] * X_train["Rainfall_mm"]
    X_test["rule_score_x_rain"]      = X_test["rule_score"]  * X_test["Rainfall_mm"]

    print(f"特征工程完成: {X_train.shape[1]} 个特征")
    return X_train, X_test, CAT_COLS


# ══════════════════════════════════════════════
#  LightGBM
# ══════════════════════════════════════════════
def train_lgb(X_train, y_encoded, X_test):
    print("\n" + "="*50)
    print("训练 LightGBM ...")

    params = {
        "objective":         "multiclass",
        "num_class":         3,
        "metric":            "multi_logloss",
        "learning_rate":     0.03,
        "num_leaves":        127,
        "max_depth":         -1,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "min_child_samples": 30,
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
        "class_weight":      "balanced",
        "n_jobs":            -1,
        "random_state":      SEED,
        "verbose":           -1,
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros((len(X_train), 3))
    test_preds = np.zeros((len(X_test), 3))
    fold_scores = []

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
            num_boost_round=3000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(-1)],
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
#  XGBoost
# ══════════════════════════════════════════════
def train_xgb(X_train, y_encoded, X_test, CAT_COLS):
    print("\n" + "="*50)
    print("训练 XGBoost ...")

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
        "objective":        "multi:softprob",
        "num_class":        3,
        "eval_metric":      "mlogloss",
        "learning_rate":    0.03,
        "max_depth":        8,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 30,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "tree_method":      "hist",
        "random_state":     SEED,
        "n_jobs":           -1,
        "verbosity":        0,
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
            num_boost_round=3000,
            evals=[(dval, "val")],
            early_stopping_rounds=150,
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
#  CatBoost
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
            iterations=3000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=3,
            bootstrap_type="Bernoulli",
            subsample=0.8,
            auto_class_weights="Balanced",
            cat_features=CAT_COLS,
            eval_metric="TotalF1:average=Macro",
            random_seed=SEED,
            verbose=0,
            early_stopping_rounds=150,
            task_type="CPU",
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
#  规则公式直接预测（对比赛数据作为第4个"模型"）
# ══════════════════════════════════════════════
def get_rule_preds(X_df, le_target):
    """
    从已经过 apply_rule_formula 的 DataFrame 中提取规则概率。
    列顺序对应 LabelEncoder: High=0, Low=1, Medium=2
    """
    # rule_prob_high/low/med 已在 feature_engineering 中计算
    probs = X_df[["rule_prob_high", "rule_prob_low", "rule_prob_med"]].values
    return probs


# ══════════════════════════════════════════════
#  集成：网格搜索最优权重（4个来源）
# ══════════════════════════════════════════════
def find_best_weights_4(oof_lgb, oof_xgb, oof_cb, oof_rule, y_encoded):
    print("\n" + "="*50)
    print("网格搜索最优集成权重 (LGB + XGB + CB + Rule) ...")

    best_ba = 0
    best_w  = (0.25, 0.25, 0.25, 0.25)

    step = 0.05
    for w_lgb in np.arange(0.0, 1.01, step):
        for w_xgb in np.arange(0.0, 1.01 - w_lgb, step):
            for w_cb in np.arange(0.0, 1.01 - w_lgb - w_xgb, step):
                w_rule = round(1.0 - w_lgb - w_xgb - w_cb, 2)
                if w_rule < 0:
                    continue
                ens = (w_lgb * oof_lgb + w_xgb * oof_xgb
                       + w_cb * oof_cb + w_rule * oof_rule)
                ba  = balanced_accuracy_score(y_encoded, ens.argmax(axis=1))
                if ba > best_ba:
                    best_ba = ba
                    best_w  = (round(w_lgb,2), round(w_xgb,2),
                               round(w_cb,2),  w_rule)

    print(f"最优权重: LGB={best_w[0]}, XGB={best_w[1]}, CB={best_w[2]}, Rule={best_w[3]}")
    print(f"集成 OOF BA: {best_ba:.6f}")
    return best_w, best_ba


# ══════════════════════════════════════════════
#  Threshold 优化
# ══════════════════════════════════════════════
def optimize_threshold(oof_ens, y_encoded, base_ba):
    print("\n" + "="*50)
    print("Threshold 优化 ...")

    best_ba = base_ba
    best_h  = 1.0
    best_m  = 1.0

    for h_mult in np.arange(1.0, 5.01, 0.1):
        for m_mult in np.arange(0.4, 1.21, 0.05):
            adj = oof_ens.copy()
            adj[:, 0] *= h_mult   # High (index 0)
            adj[:, 2] *= m_mult   # Medium (index 2)
            ba = balanced_accuracy_score(y_encoded, adj.argmax(axis=1))
            if ba > best_ba:
                best_ba = ba
                best_h  = round(h_mult, 2)
                best_m  = round(m_mult, 2)

    print(f"最优 Threshold: High×{best_h}, Medium×{best_m}")
    print(f"优化后 OOF BA: {best_ba:.6f}")
    return best_h, best_m, best_ba


# ══════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════
def main():
    train, test, test_ids = load_data()

    y_train  = train[TARGET].copy()
    X_train  = train.drop(columns=TARGET)
    X_test   = test.copy()

    CAT_COLS = X_train.select_dtypes(include="object").columns.tolist()

    X_train, X_test, CAT_COLS = feature_engineering(X_train, X_test)

    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y_train)
    class_names = le_target.classes_
    print(f"类别顺序 (LabelEncoder): {list(class_names)}")
    # 确认顺序：应该是 ['High', 'Low', 'Medium']
    # rule_prob_high=index0, rule_prob_low=index1, rule_prob_med=index2

    # 训练三个模型
    oof_lgb,  test_lgb  = train_lgb(X_train, y_encoded, X_test)
    oof_xgb,  test_xgb  = train_xgb(X_train, y_encoded, X_test, CAT_COLS)
    oof_cb,   test_cb   = train_cat(X_train, y_encoded, X_test, CAT_COLS)

    # 规则公式直接预测（OOF = 全训练集的规则概率）
    oof_rule  = get_rule_preds(X_train, le_target)
    test_rule = get_rule_preds(X_test,  le_target)
    rule_oof_ba = balanced_accuracy_score(y_encoded, oof_rule.argmax(axis=1))
    print(f"\n规则公式 OOF BA (训练集): {rule_oof_ba:.6f}")

    # 保存 OOF
    for name, arr in [("lgb", oof_lgb), ("xgb", oof_xgb), ("cb", oof_cb), ("rule", oof_rule),
                      ("test_lgb", test_lgb), ("test_xgb", test_xgb),
                      ("test_cb", test_cb), ("test_rule", test_rule)]:
        np.save(OUTPUT_DIR / f"oof_{name}.npy", arr)

    # 搜索最优权重（含 Rule）
    best_w, ens_ba = find_best_weights_4(oof_lgb, oof_xgb, oof_cb, oof_rule, y_encoded)
    w_lgb, w_xgb, w_cb, w_rule = best_w

    oof_ens  = w_lgb*oof_lgb  + w_xgb*oof_xgb  + w_cb*oof_cb  + w_rule*oof_rule
    test_ens = w_lgb*test_lgb + w_xgb*test_xgb + w_cb*test_cb + w_rule*test_rule

    # Threshold 优化
    best_h, best_m, final_ba = optimize_threshold(oof_ens, y_encoded, ens_ba)

    # ── 关键改进4：纯规则公式提交（作为对照，可能在 LB 上很强）──
    test_rule_labels = le_target.inverse_transform(test_rule.argmax(axis=1))
    sub_rule = pd.DataFrame({"id": test_ids, "Irrigation_Need": test_rule_labels})
    sub_rule.to_csv(OUTPUT_DIR / "submission_rule_only.csv", index=False)

    # 集成默认版
    sub1 = pd.DataFrame({
        "id":              test_ids,
        "Irrigation_Need": le_target.inverse_transform(test_ens.argmax(axis=1)),
    })
    sub1.to_csv(OUTPUT_DIR / "submission_default.csv", index=False)

    # 集成 + threshold 优化版
    adj_test = test_ens.copy()
    adj_test[:, 0] *= best_h
    adj_test[:, 2] *= best_m
    sub2 = pd.DataFrame({
        "id":              test_ids,
        "Irrigation_Need": le_target.inverse_transform(adj_test.argmax(axis=1)),
    })
    sub2.to_csv(OUTPUT_DIR / "submission_threshold.csv", index=False)

    print("\n" + "="*60)
    print("  最终结果汇总")
    print("="*60)
    print(f"  规则公式  OOF BA:               {rule_oof_ba:.6f}")
    print(f"  LightGBM  OOF BA:               {balanced_accuracy_score(y_encoded, oof_lgb.argmax(1)):.6f}")
    print(f"  XGBoost   OOF BA:               {balanced_accuracy_score(y_encoded, oof_xgb.argmax(1)):.6f}")
    print(f"  CatBoost  OOF BA:               {balanced_accuracy_score(y_encoded, oof_cb.argmax(1)):.6f}")
    print(f"  集成 (weights={best_w}): {ens_ba:.6f}")
    print(f"  集成 + Threshold:               {final_ba:.6f}")
    print(f"\n  提交文件:")
    print(f"    submission_rule_only.csv  <- 纯规则公式（强基线，先试试）")
    print(f"    submission_default.csv    <- 集成默认版")
    print(f"    submission_threshold.csv  <- threshold优化版 (推荐)")


if __name__ == "__main__":
    main()