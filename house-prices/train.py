import warnings, time
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet, HuberRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import features as F

warnings.filterwarnings("ignore")
SEED = 42
NFOLDS = 10

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def make_matrices():
    train, test = F.load_raw()
    train = F.drop_outliers(train)
    d, y, ntr, ids = F.build(train, test)

    cat_cols = list(d.select_dtypes(object).columns)

    X_oh = pd.get_dummies(d, columns=cat_cols, drop_first=False).astype(float)

    X_lin = pd.get_dummies(F.skew_fix(d), columns=cat_cols, drop_first=False).astype(float)

    keep = X_lin.columns[(X_lin.iloc[:ntr] != 0).sum(axis=0) >= 10]
    X_lin = X_lin[keep]

    X_cat = d.copy()

    return (X_oh.iloc[:ntr].values, X_oh.iloc[ntr:].values,
            X_lin.iloc[:ntr].values, X_lin.iloc[ntr:].values,
            X_cat.iloc[:ntr].reset_index(drop=True), X_cat.iloc[ntr:].reset_index(drop=True),
            cat_cols, y, test["Id"].values)

Xt_tr, Xt_te, Xl_tr, Xl_te, Xc_tr, Xc_te, CAT_COLS, y, test_ids = make_matrices()
print(f"agac matrisi {Xt_tr.shape} | lineer matris {Xl_tr.shape}")

def lin(model):
    return make_pipeline(RobustScaler(), model)

MODELS = {

    "lasso":    ("lin", lin(Lasso(alpha=0.0005, max_iter=50000, random_state=SEED))),
    "ridge":    ("lin", lin(Ridge(alpha=13.0, random_state=SEED))),
    "enet":     ("lin", lin(ElasticNet(alpha=0.0007, l1_ratio=0.85, max_iter=50000, random_state=SEED))),
    "krr":      ("lin", lin(KernelRidge(alpha=0.5, kernel="polynomial", degree=2, coef0=2.0))),
    "svr":      ("lin", lin(SVR(C=22, epsilon=0.009, gamma=0.0004))),
    "huber":    ("lin", lin(HuberRegressor(alpha=0.0005, epsilon=1.35, max_iter=3000))),

    "gbr":      ("tree", GradientBoostingRegressor(
                    n_estimators=3000, learning_rate=0.02, max_depth=4, max_features="sqrt",
                    min_samples_leaf=15, min_samples_split=10, loss="huber",
                    subsample=0.8, random_state=SEED)),
    "xgb":      ("tree", XGBRegressor(
                    n_estimators=3500, learning_rate=0.015, max_depth=3, min_child_weight=1,
                    subsample=0.7, colsample_bytree=0.4, reg_alpha=0.0005, reg_lambda=1.0,
                    gamma=0.0, n_jobs=-1, random_state=SEED, tree_method="hist")),
    "lgb":      ("tree", LGBMRegressor(
                    n_estimators=4000, learning_rate=0.01, num_leaves=6, max_bin=200,
                    min_child_samples=12, min_child_weight=0.001, feature_fraction=0.3,
                    bagging_fraction=0.75, bagging_freq=1, reg_alpha=0.1, reg_lambda=0.5,
                    n_jobs=-1, random_state=SEED, verbose=-1)),
    "et":       ("tree", ExtraTreesRegressor(
                    n_estimators=800, max_features=0.4, min_samples_leaf=2,
                    n_jobs=-1, random_state=SEED)),
    "cat":      ("cat", CatBoostRegressor(
                    iterations=4000, learning_rate=0.02, depth=5, l2_leaf_reg=3.0,
                    loss_function="RMSE", random_seed=SEED, verbose=0, allow_writing_files=False)),
}

def get_data(kind):
    if kind == "lin":
        return Xl_tr, Xl_te
    if kind == "tree":
        return Xt_tr, Xt_te
    return Xc_tr, Xc_te

def run_cv(name, kind, model):
    Xtr, Xte = get_data(kind)
    oof = np.zeros(len(y))
    pred = np.zeros(len(test_ids))
    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    t0 = time.time()
    for tr_idx, va_idx in kf.split(Xtr):
        from sklearn.base import clone
        m = clone(model)
        if kind == "cat":
            m.fit(Xtr.iloc[tr_idx], y[tr_idx], cat_features=CAT_COLS)
            oof[va_idx] = m.predict(Xtr.iloc[va_idx])
            pred += m.predict(Xte) / NFOLDS
        else:
            m.fit(Xtr[tr_idx], y[tr_idx])
            oof[va_idx] = m.predict(Xtr[va_idx])
            pred += m.predict(Xte) / NFOLDS
    s = rmse(oof, y)
    print(f"  {name:6s} CV RMSE(log) = {s:.5f}   [{time.time()-t0:5.1f}s]")
    return oof, pred, s

if __name__ == "__main__":
    print(f"\n{NFOLDS}-kat capraz dogrulama:")
    oofs, preds, scores = {}, {}, {}
    for name, (kind, model) in MODELS.items():
        o, p, s = run_cv(name, kind, model)
        oofs[name], preds[name], scores[name] = o, p, s

    names = list(oofs)
    OOF = np.column_stack([oofs[n] for n in names])
    TEST = np.column_stack([preds[n] for n in names])
    np.savez("oof_cache.npz", OOF=OOF, TEST=TEST, y=y, names=np.array(names), ids=test_ids)

    w, _ = nnls(OOF, y)
    w = w / w.sum()
    blend_oof = OOF @ w
    print(f"\nAgirliklar: " + ", ".join(f"{n}={wi:.3f}" for n, wi in zip(names, w) if wi > 1e-4))
    print(f"NNLS harman CV RMSE = {rmse(blend_oof, y):.5f}")

    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED + 1)
    nested = np.zeros(len(y))
    for tr_i, va_i in kf.split(OOF):
        ww, _ = nnls(OOF[tr_i], y[tr_i])
        nested[va_i] = OOF[va_i] @ (ww / max(ww.sum(), 1e-9))
    print(f"Ic ice (dogrulanmis) harman CV RMSE = {rmse(nested, y):.5f}")

    final_log = TEST @ w
    sub = pd.DataFrame({"Id": test_ids, "SalePrice": np.expm1(final_log)})
    sub.to_csv("submission.csv", index=False)
    print(f"\nsubmission.csv yazildi -> {sub.shape}, fiyat araligi "
          f"{sub.SalePrice.min():,.0f} - {sub.SalePrice.max():,.0f}")
