import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.special import boxcox1p

DATA_DIR = "house-prices-advanced-regression-techniques"

QUAL = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_MAPS = {
    "ExterQual": QUAL, "ExterCond": QUAL, "BsmtQual": QUAL, "BsmtCond": QUAL,
    "HeatingQC": QUAL, "KitchenQual": QUAL, "FireplaceQu": QUAL,
    "GarageQual": QUAL, "GarageCond": QUAL, "PoolQC": QUAL,
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "Fence": {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
    "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "CentralAir": {"N": 0, "Y": 1},
    "Street": {"Grvl": 0, "Pave": 1},
    "Alley": {"None": 0, "Grvl": 1, "Pave": 2},
    "Utilities": {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3},
}

NONE_COLS = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
             "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish",
             "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature",
             "MasVnrType"]

ZERO_COLS = ["MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
             "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea"]

def load_raw():
    train = pd.read_csv(f"{DATA_DIR}/train.csv", keep_default_na=False, na_values=[""])
    test = pd.read_csv(f"{DATA_DIR}/test.csv", keep_default_na=False, na_values=[""])

    num_cols = ["LotFrontage", "MasVnrArea", "GarageYrBlt", "BsmtFinSF1", "BsmtFinSF2",
                "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath",
                "GarageCars", "GarageArea"]
    for df in (train, test):
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.replace("NA", np.nan, inplace=True)
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return train, test

def drop_outliers(train):
    """Klasik Ames aykirilari: cok buyuk ama cok ucuz iki ev (De Cock'un uyarisi)."""
    mask = (train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)
    return train.loc[~mask].reset_index(drop=True)

def build(train, test):
    """Train+test birlestirip temizler, oznitelik uretir. (X_all, n_train) doner."""
    y = np.log1p(train["SalePrice"].values)
    ntr = len(train)
    all_df = pd.concat([train.drop(columns=["SalePrice"]), test], ignore_index=True)
    ids = all_df["Id"].values
    all_df = all_df.drop(columns=["Id"])

    for c in NONE_COLS:
        all_df[c] = all_df[c].fillna("None")
    for c in ZERO_COLS:
        all_df[c] = all_df[c].fillna(0)

    all_df["LotFrontage"] = all_df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda s: s.fillna(s.median()))
    all_df["LotFrontage"] = all_df["LotFrontage"].fillna(all_df["LotFrontage"].median())
    all_df["GarageYrBlt"] = all_df["GarageYrBlt"].fillna(all_df["YearBuilt"])

    all_df["MSZoning"] = all_df.groupby("MSSubClass")["MSZoning"].transform(
        lambda s: s.fillna(s.mode()[0] if not s.mode().empty else "RL"))
    all_df["Functional"] = all_df["Functional"].fillna("Typ")
    all_df["Utilities"] = all_df["Utilities"].fillna("AllPub")
    for c in ["Electrical", "KitchenQual", "Exterior1st", "Exterior2nd", "SaleType"]:
        all_df[c] = all_df[c].fillna(all_df[c].mode()[0])
    for c in all_df.columns:
        if all_df[c].dtype == object:
            all_df[c] = all_df[c].fillna("None")
        else:
            all_df[c] = all_df[c].fillna(0)

    for c, m in ORDINAL_MAPS.items():
        all_df[c] = all_df[c].map(m).fillna(0).astype(int)

    d = all_df.copy()
    d["TotalSF"] = d["TotalBsmtSF"] + d["1stFlrSF"] + d["2ndFlrSF"]
    d["TotalFinSF"] = d["GrLivArea"] + d["BsmtFinSF1"] + d["BsmtFinSF2"]
    d["TotalBath"] = d["FullBath"] + 0.5 * d["HalfBath"] + d["BsmtFullBath"] + 0.5 * d["BsmtHalfBath"]
    d["TotalPorchSF"] = (d["OpenPorchSF"] + d["EnclosedPorch"] + d["3SsnPorch"]
                         + d["ScreenPorch"] + d["WoodDeckSF"])
    d["Age"] = d["YrSold"] - d["YearBuilt"]
    d["RemodAge"] = d["YrSold"] - d["YearRemodAdd"]
    d["GarageAge"] = d["YrSold"] - d["GarageYrBlt"]
    d["IsRemodeled"] = (d["YearRemodAdd"] != d["YearBuilt"]).astype(int)
    d["IsNew"] = (d["YrSold"] == d["YearBuilt"]).astype(int)
    d["HasPool"] = (d["PoolArea"] > 0).astype(int)
    d["Has2ndFloor"] = (d["2ndFlrSF"] > 0).astype(int)
    d["HasGarage"] = (d["GarageArea"] > 0).astype(int)
    d["HasBsmt"] = (d["TotalBsmtSF"] > 0).astype(int)
    d["HasFireplace"] = (d["Fireplaces"] > 0).astype(int)
    d["HasPorch"] = (d["TotalPorchSF"] > 0).astype(int)

    d["OverallGrade"] = d["OverallQual"] * d["OverallCond"]
    d["QualxSF"] = d["OverallQual"] * d["TotalSF"]
    d["QualxLiv"] = d["OverallQual"] * d["GrLivArea"]
    d["ExterGrade"] = d["ExterQual"] * d["ExterCond"]
    d["KitchenGrade"] = d["KitchenQual"] * d["KitchenAbvGr"]
    d["GarageGrade"] = d["GarageQual"] * d["GarageCars"]
    d["BsmtGrade"] = d["BsmtQual"] * d["TotalBsmtSF"]
    d["QualSum"] = (d["ExterQual"] + d["BsmtQual"] + d["HeatingQC"] + d["KitchenQual"]
                    + d["FireplaceQu"] + d["GarageQual"])
    d["LivAreaPerRoom"] = d["GrLivArea"] / d["TotRmsAbvGrd"].clip(lower=1)
    d["SFPerLot"] = d["TotalSF"] / d["LotArea"].clip(lower=1)
    d["SpaciousRatio"] = d["GrLivArea"] / d["TotalSF"].clip(lower=1)

    d["MSSubClass"] = d["MSSubClass"].astype(str)
    d["MoSold_cat"] = d["MoSold"].astype(str)
    d["YrSold_cat"] = d["YrSold"].astype(str)
    d = d.copy()
    return d, y, ntr, ids

def skew_fix(df, thresh=0.5, lam=0.15):
    """Sayisal sutunlardaki carpikligi Box-Cox(1p) ile duzeltir (lineer modeller icin)."""
    out = df.copy()
    num = out.select_dtypes(exclude=object).columns
    sk = out[num].apply(lambda s: skew(s.dropna()))
    for c in sk[sk.abs() > thresh].index:
        if out[c].min() >= 0:
            out[c] = boxcox1p(out[c], lam)
    return out
