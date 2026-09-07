import numpy as np, pandas as pd, re, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

train = pd.read_csv("train.csv"); test = pd.read_csv("test.csv")
y = train["Survived"].values
full = pd.concat([train.drop(columns="Survived"), test], ignore_index=True)

full["Title"] = full["Name"].str.extract(r",\s*([^\.]+)\.")[0].str.strip()
full["Title"] = full["Title"].replace(
    {"Mlle":"Miss","Ms":"Miss","Mme":"Mrs","Lady":"Rare","Countess":"Rare","the Countess":"Rare",
     "Capt":"Rare","Col":"Rare","Don":"Rare","Dona":"Rare","Dr":"Rare","Major":"Rare",
     "Rev":"Rare","Sir":"Rare","Jonkheer":"Rare"})

full["FamilySize"] = full["SibSp"] + full["Parch"] + 1
full["IsAlone"] = (full["FamilySize"] == 1).astype(int)
full["Surname"] = full["Name"].str.split(",").str[0].str.strip().str.lower()
full["TicketPrefix"] = full["Ticket"].str.replace(r"[\./\s]", "", regex=True).str[:-1]
full["TicketGroup"] = full.groupby("Ticket")["Ticket"].transform("size")

full["Embarked"] = full["Embarked"].fillna("S")
full["Fare"] = full["Fare"].fillna(full.groupby("Pclass")["Fare"].transform("median"))
full["FarePerPerson"] = full["Fare"] / full["TicketGroup"]
full["Age"] = full["Age"].fillna(full.groupby(["Title","Pclass"])["Age"].transform("median"))
full["Age"] = full["Age"].fillna(full["Age"].median())
full["Deck"] = full["Cabin"].fillna("U").str[0].replace({"T":"U"})
full["HasCabin"] = (~full["Cabin"].isna()).astype(int)

full["Child"] = (full["Age"] < 14).astype(int)
full["LogFare"] = np.log1p(full["Fare"])
full["AgeBin"] = pd.qcut(full["Age"], 6, labels=False, duplicates="drop")
full["FareBin"] = pd.qcut(full["Fare"], 8, labels=False, duplicates="drop")

full["GroupId"] = full["Surname"] + "_" + full["Pclass"].astype(str) + "_" + full["TicketPrefix"].fillna("")
lab = pd.Series(np.nan, index=full.index); lab.iloc[:len(train)] = y
g = lab.groupby(full["GroupId"])
full["FamSurv"] = ((g.transform("sum") - lab.fillna(0)) / (g.transform("count") - lab.notna().astype(int))).fillna(-1)
full["FamKnown"] = (full["FamSurv"] >= 0).astype(int)

cols = ["Pclass","Sex","Age","SibSp","Parch","LogFare","FarePerPerson","Embarked","Title",
        "FamilySize","IsAlone","TicketGroup","Deck","HasCabin","Child","AgeBin","FareBin",
        "FamSurv","FamKnown"]
X = pd.get_dummies(full[cols], columns=["Sex","Embarked","Title","Deck"], drop_first=True).astype(float)
Xtr, Xte = X.iloc[:len(train)].values, X.iloc[len(train):].values

models = [
    ("gb", GradientBoostingClassifier(n_estimators=400, learning_rate=0.03, max_depth=3,
                                      subsample=0.85, random_state=0)),
    ("rf", RandomForestClassifier(n_estimators=800, max_depth=7, min_samples_leaf=3,
                                  max_features="sqrt", random_state=0, n_jobs=-1)),
    ("et", ExtraTreesClassifier(n_estimators=800, max_depth=9, min_samples_leaf=2,
                                random_state=0, n_jobs=-1)),
    ("lr", make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))),
]
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
for n, m in models:
    s = cross_val_score(m, Xtr, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"{n:>3}: {s.mean():.4f} +/- {s.std():.4f}")

vote = VotingClassifier(models, voting="soft")
s = cross_val_score(vote, Xtr, y, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"ens: {s.mean():.4f} +/- {s.std():.4f}")

vote.fit(Xtr, y)
pred = vote.predict(Xte).astype(int)
pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pred}).to_csv("submission_model.csv", index=False)
print(f"\nsubmission_model.csv yazildi: {len(pred)} satir, hayatta kalan {pred.sum()}")
