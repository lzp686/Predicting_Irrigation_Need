"""
Playground Series S6E4 — Formula Boost
基于 Discussion 687460 发现的原始数据生成公式

策略 A：直接用公式预测测试集 → 可能直接到 0.98+
策略 B：把公式概率作为新特征融入集成模型 → 稳健提升
"""

import warnings; warnings.filterwarnings("ignore")
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

# ★ 修改这里
DATA_DIR   = Path("./data")
ORIG_CSV   = Path("./data/irrigation_prediction.csv")
OUTPUT_DIR = Path("./output"); OUTPUT_DIR.mkdir(exist_ok=True)
SEED, N_FOLDS, TARGET = 42, 5, "Irrigation_Need"


# ══════════════════════════════════════════════
#  核心公式
# ══════════════════════════════════════════════

def apply_formula_simple(df):
    """简化规则版 — 对原始数据 BA = 1.0"""
    h = (2*(df["Soil_Moisture"]<25) + 2*(df["Rainfall_mm"]<300)
         + (df["Temperature_C"]>30) + (df["Wind_Speed_kmh"]>10)).astype(int)
    l = (2*df["Crop_Growth_Stage"].isin(["Harvest"]).astype(int)
         + 2*df["Crop_Growth_Stage"].isin(["Sowing"]).astype(int)
         + (df["Mulching_Used"]=="Yes").astype(int))
    score = h - l
    return pd.cut(score, bins=[-np.inf, 0, 3, np.inf],
                  labels=["Low","Medium","High"]).astype(str)


def apply_formula_logistic(df):
    """精确 Logistic 公式，返回概率矩阵 列=[Low, Medium, High]"""
    s,t,r,w   = [(df[c]>v if op else df[c]<v).astype(float)
                 for c,v,op in [("Soil_Moisture",25,False),("Temperature_C",30,True),
                                ("Rainfall_mm",300,False),("Wind_Speed_kmh",10,True)]]
    fl = (df["Crop_Growth_Stage"]=="Flowering").astype(float)
    ha = (df["Crop_Growth_Stage"]=="Harvest").astype(float)
    so = (df["Crop_Growth_Stage"]=="Sowing").astype(float)
    ve = (df["Crop_Growth_Stage"]=="Vegetative").astype(float)
    mn = (df["Mulching_Used"]=="No").astype(float)
    my = (df["Mulching_Used"]=="Yes").astype(float)

    lo = 16.3173 -11.0237*s -5.8559*t -10.8500*r -5.8284*w -5.4155*fl +5.5073*ha +5.2299*so -5.4617*ve -3.0014*mn +2.8613*my
    me =  4.6524  +0.3290*s -0.0204*t  +0.1542*r +0.0841*w +0.3586*fl -0.1348*ha -0.3547*so +0.3334*ve +0.1883*mn +0.0142*my
    hi =-20.9697 +10.6947*s +5.8763*t +10.6958*r +5.7444*w +5.0569*fl -5.3725*ha -4.8752*so +5.1283*ve +2.8131*mn -2.8755*my

    logits = np.stack([lo, me, hi], axis=1)
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


# ══════════════════════════════════════════════
#  数据读取
# ══════════════════════════════════════════════
def load_data():
    train = pd.read_csv(DATA_DIR/"train.csv")
    test  = pd.read_csv(DATA_DIR/"test.csv")
    orig  = pd.read_csv(ORIG_CSV)
    for c in ["Wind_Speed_kmh","Field_Area_hectare","Previous_Irrigation_mm"]:
        train[c] = pd.to_numeric(train[c], errors="coerce")
    train = train[~train["Soil_Type"].str.match(r"^\d",na=False)].reset_index(drop=True)
    test_ids = test["id"].copy()
    train, test = train.drop(columns="id"), test.drop(columns="id")
    train = pd.concat([train, orig], ignore_index=True).dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"训练集: {train.shape}  测试集: {test.shape}")
    return train, test, test_ids


# ══════════════════════════════════════════════
#  策略 A：直接公式预测
# ══════════════════════════════════════════════
def strategy_a(test, test_ids):
    print("\n"+"="*50)
    print("策略 A：直接用公式预测测试集")
    orig = pd.read_csv(ORIG_CSV)
    print(f"公式在原始数据上 BA = {balanced_accuracy_score(orig[TARGET], apply_formula_simple(orig)):.6f}")
    preds = apply_formula_simple(test)
    print(f"测试集分布:\n{preds.value_counts()}")
    pd.DataFrame({"id":test_ids,"Irrigation_Need":preds}).to_csv(
        OUTPUT_DIR/"submission_formula_direct.csv", index=False)
    print("→ 保存: output/submission_formula_direct.csv")


# ══════════════════════════════════════════════
#  特征工程（含公式特征）
# ══════════════════════════════════════════════
def feature_engineering(Xtr, Xte):
    CAT = Xtr.select_dtypes("object").columns.tolist()

    def tier1(d):
        d = d.copy()
        d["ET_index"]      = d["Temperature_C"]*d["Sunlight_Hours"]/(d["Humidity"]+1)
        d["water_supply"]  = d["Rainfall_mm"]+d["Previous_Irrigation_mm"]
        d["drought_index"] = (np.maximum(0,800-d["Rainfall_mm"])+np.maximum(0,40-d["Soil_Moisture"])*10)/(d["Humidity"]+1)
        d["water_stress"]  = d["ET_index"]/(d["water_supply"]+1)
        d["moist_x_temp"]  = d["Soil_Moisture"]*d["Temperature_C"]
        d["rain_per_area"] = d["Rainfall_mm"]/(d["Field_Area_hectare"]+0.1)
        return d

    Xtr, Xte = tier1(Xtr), tier1(Xte)

    for cat in ["Crop_Growth_Stage","Crop_Type","Region","Season"]:
        for num in ["Soil_Moisture","Rainfall_mm","Temperature_C","Wind_Speed_kmh","Humidity"]:
            mm = Xtr.groupby(cat)[num].mean()
            sm = Xtr.groupby(cat)[num].std().fillna(0)
            for df, ref in [(Xtr, Xtr), (Xte, Xtr)]:
                df[f"{num}_mean_{cat}"] = df[cat].map(mm) if df is Xtr else df[cat].map(mm).fillna(ref[num].mean())
                df[f"{num}_std_{cat}"]  = df[cat].map(sm) if df is Xtr else df[cat].map(sm).fillna(ref[num].std())
                df[f"{num}_diff_{cat}"] = df[num] - df[f"{num}_mean_{cat}"]

    for c in CAT:
        fm = Xtr[c].value_counts(normalize=True)
        Xtr[f"{c}_freq"] = Xtr[c].map(fm)
        Xte[f"{c}_freq"] = Xte[c].map(fm).fillna(0)

    Xtr["mhr"] = Xtr["Soil_Moisture"]/(Xtr["Humidity"]+1)
    Xte["mhr"] = Xte["Soil_Moisture"]/(Xte["Humidity"]+1)
    Xtr["twp"] = Xtr["Temperature_C"]*Xtr["Wind_Speed_kmh"]
    Xte["twp"] = Xte["Temperature_C"]*Xte["Wind_Speed_kmh"]
    rm = Xtr["Rainfall_mm"].median()
    Xtr["rain_deficit"] = rm - Xtr["Rainfall_mm"]
    Xte["rain_deficit"] = rm - Xte["Rainfall_mm"]
    Xtr["scr"] = Xtr["Soil_Moisture"]/(Xtr["Electrical_Conductivity"]+0.1)
    Xte["scr"] = Xte["Soil_Moisture"]/(Xte["Electrical_Conductivity"]+0.1)

    # ★ 公式特征（最关键）
    for df in [Xtr, Xte]:
        probs = apply_formula_logistic(df)
        df["formula_p_low"]    = probs[:,0]
        df["formula_p_medium"] = probs[:,1]
        df["formula_p_high"]   = probs[:,2]
        h = (2*(df["Soil_Moisture"]<25)+2*(df["Rainfall_mm"]<300)
             +(df["Temperature_C"]>30)+(df["Wind_Speed_kmh"]>10)).astype(int)
        l = (2*df["Crop_Growth_Stage"].isin(["Harvest","Sowing"]).astype(int)
             +(df["Mulching_Used"]=="Yes").astype(int))
        df["formula_high_score"] = h
        df["formula_low_score"]  = l
        df["formula_net_score"]  = h - l

    print(f"特征总数: {Xtr.shape[1]}（含 6 个公式特征）")
    return Xtr, Xte, CAT


# ══════════════════════════════════════════════
#  模型训练
# ══════════════════════════════════════════════
def train_lgb(Xtr, y, Xte):
    print("\n训练 LightGBM ...")
    p = {"objective":"multiclass","num_class":3,"metric":"multi_logloss","learning_rate":0.05,
         "num_leaves":127,"subsample":0.8,"colsample_bytree":0.8,"min_child_samples":50,
         "class_weight":"balanced","n_jobs":-1,"random_state":SEED,"verbose":-1}
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof, tp = np.zeros((len(Xtr),3)), np.zeros((len(Xte),3))
    X2, T2 = Xtr.copy(), Xte.copy()
    for c in X2.select_dtypes("object"):
        X2[c]=X2[c].astype("category"); T2[c]=T2[c].astype("category")
    for f,(ti,vi) in enumerate(skf.split(X2,y),1):
        m = lgb.train(p, lgb.Dataset(X2.iloc[ti],y[ti]), 2000,
                      valid_sets=[lgb.Dataset(X2.iloc[vi],y[vi])],
                      callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(-1)])
        oof[vi]=m.predict(X2.iloc[vi]); tp+=m.predict(T2)/N_FOLDS
        print(f"  Fold {f}: {balanced_accuracy_score(y[vi],oof[vi].argmax(1)):.6f}")
    print(f"LGB OOF: {balanced_accuracy_score(y,oof.argmax(1)):.6f}")
    return oof, tp


def train_xgb(Xtr, y, Xte, CAT):
    print("\n训练 XGBoost ...")
    X2, T2 = Xtr.copy(), Xte.copy()
    for c in CAT:
        le=LabelEncoder().fit(pd.concat([X2[c],T2[c]]).astype(str))
        X2[c]=le.transform(X2[c].astype(str)); T2[c]=le.transform(T2[c].astype(str))
    p = {"objective":"multi:softprob","num_class":3,"eval_metric":"mlogloss","learning_rate":0.05,
         "max_depth":8,"subsample":0.8,"colsample_bytree":0.8,"min_child_weight":50,
         "tree_method":"hist","random_state":SEED,"n_jobs":-1,"verbosity":0}
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof, tp = np.zeros((len(X2),3)), np.zeros((len(T2),3))
    dtest = xgb.DMatrix(T2)
    for f,(ti,vi) in enumerate(skf.split(X2,y),1):
        sw = compute_sample_weight("balanced",y[ti])
        m = xgb.train(p, xgb.DMatrix(X2.iloc[ti],y[ti],weight=sw), 2000,
                      evals=[(xgb.DMatrix(X2.iloc[vi],y[vi]),"v")],
                      early_stopping_rounds=100, verbose_eval=False)
        oof[vi]=m.predict(xgb.DMatrix(X2.iloc[vi])); tp+=m.predict(dtest)/N_FOLDS
        print(f"  Fold {f}: {balanced_accuracy_score(y[vi],oof[vi].argmax(1)):.6f}")
    print(f"XGB OOF: {balanced_accuracy_score(y,oof.argmax(1)):.6f}")
    return oof, tp


def train_cat(Xtr, y, Xte, CAT):
    print("\n训练 CatBoost ...")
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof, tp = np.zeros((len(Xtr),3)), np.zeros((len(Xte),3))
    for f,(ti,vi) in enumerate(skf.split(Xtr,y),1):
        m = CatBoostClassifier(iterations=2000,learning_rate=0.05,depth=8,l2_leaf_reg=3,
                               bootstrap_type="Bernoulli",subsample=0.8,auto_class_weights="Balanced",
                               cat_features=CAT,random_seed=SEED,verbose=0,
                               early_stopping_rounds=100,task_type="CPU")
        m.fit(Xtr.iloc[ti],y[ti],eval_set=(Xtr.iloc[vi],y[vi]))
        oof[vi]=m.predict_proba(Xtr.iloc[vi]); tp+=m.predict_proba(Xte)/N_FOLDS
        print(f"  Fold {f}: {balanced_accuracy_score(y[vi],oof[vi].argmax(1)):.6f}")
    print(f"CB OOF: {balanced_accuracy_score(y,oof.argmax(1)):.6f}")
    return oof, tp


def find_best_weights(ol, ox, oc, y):
    print("\n搜索集成权重 ...")
    best, bw = 0, (1/3,1/3,1/3)
    for wl in np.arange(0,1.01,0.05):
        for wx in np.arange(0,1.01-wl,0.05):
            wc=round(1-wl-wx,2)
            if wc<0: continue
            ba=balanced_accuracy_score(y,(wl*ol+wx*ox+wc*oc).argmax(1))
            if ba>best: best,bw=ba,(round(wl,2),round(wx,2),wc)
    print(f"最优: LGB={bw[0]}, XGB={bw[1]}, CB={bw[2]}  BA={best:.6f}")
    return bw, best


def optimize_threshold(ens, y, base):
    print("\nThreshold 优化 ...")
    best,bh,bm=base,1.0,1.0
    for h in np.arange(1.0,5.01,0.1):
        for m in np.arange(0.4,1.21,0.05):
            a=ens.copy(); a[:,0]*=h; a[:,2]*=m
            ba=balanced_accuracy_score(y,a.argmax(1))
            if ba>best: best,bh,bm=ba,round(h,2),round(m,2)
    print(f"最优: High×{bh}, Med×{bm}  BA={best:.6f}")
    return bh, bm, best


# ══════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════
def main():
    train, test, test_ids = load_data()

    # 策略 A
    strategy_a(test, test_ids)

    # 策略 B
    print("\n"+"="*50+"  策略 B：公式特征 + 集成模型")
    y_raw = train[TARGET].copy()
    Xtr   = train.drop(columns=TARGET)
    Xte   = test.copy()
    CAT   = Xtr.select_dtypes("object").columns.tolist()
    Xtr, Xte, CAT = feature_engineering(Xtr, Xte)
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    print(f"类别顺序: {list(le.classes_)}")

    ol, tl = train_lgb(Xtr, y, Xte)
    ox, tx = train_xgb(Xtr, y, Xte, CAT)
    oc, tc = train_cat(Xtr, y, Xte, CAT)

    bw, ens_ba = find_best_weights(ol, ox, oc, y)
    wl,wx,wc   = bw
    ens_t = wl*tl + wx*tx + wc*tc
    bh, bm, final_ba = optimize_threshold(wl*ol+wx*ox+wc*oc, y, ens_ba)

    adj=ens_t.copy(); adj[:,0]*=bh; adj[:,2]*=bm
    pd.DataFrame({"id":test_ids,"Irrigation_Need":le.inverse_transform(adj.argmax(1))}).to_csv(
        OUTPUT_DIR/"submission_formula_ensemble.csv", index=False)

    print("\n"+"="*60)
    print(f"  策略 A: output/submission_formula_direct.csv")
    print(f"  策略 B: output/submission_formula_ensemble.csv  OOF={final_ba:.6f}")
    print("\n  建议: 先提交 A，LB>=0.98 直接用 A；否则用 B")


if __name__ == "__main__":
    main()