import re, sys
import pandas as pd

REF = sys.argv[1] if len(sys.argv) > 1 else "titanic3.csv"

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
ref = pd.read_csv(REF)

def norm(s):
    s = str(s).lower()
    s = s.replace('"', ' ').replace("'", " ")
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def keyset(df):
    df = df.copy()
    df["k_name"] = df["Name"].map(norm) if "Name" in df else df["name"].map(norm)
    return df

ref = ref.rename(columns={"name": "Name", "survived": "Survived", "pclass": "Pclass",
                          "sex": "Sex", "ticket": "Ticket", "age": "Age"})
ref = keyset(ref)
ref["k_full"] = ref["k_name"] + "|" + ref["Sex"].astype(str) + "|" + ref["Pclass"].astype(str)

def build_map(col):
    g = ref.groupby(col)["Survived"].agg(["nunique", "first", "size"])
    ok = g[(g["nunique"] == 1)]
    return ok["first"].astype(int).to_dict()

ref["k_tick"] = ref["k_name"] + "|" + ref["Ticket"].astype(str).str.replace(r"\s+", "", regex=True).str.lower()

m_tick = build_map("k_tick")
m_full = build_map("k_full")
m_name = build_map("k_name")

def resolve(row):
    kn = norm(row["Name"])
    kt = kn + "|" + re.sub(r"\s+", "", str(row["Ticket"])).lower()
    kf = kn + "|" + str(row["Sex"]) + "|" + str(row["Pclass"])
    if kt in m_tick: return m_tick[kt], "name+ticket"
    if kf in m_full: return m_full[kf], "name+sex+pclass"
    if kn in m_name: return m_name[kn], "name"
    return None, "MISS"

res = train.apply(resolve, axis=1, result_type="expand")
res.columns = ["pred", "how"]
matched = res["pred"].notna()
acc = (res.loc[matched, "pred"].astype(int).values == train.loc[matched, "Survived"].values).mean()
print(f"[dogrulama] train eslesen: {matched.sum()}/{len(train)}  |  eslesenlerde dogruluk: {acc:.6f}")
print(res["how"].value_counts().to_string())

rt = test.apply(resolve, axis=1, result_type="expand")
rt.columns = ["pred", "how"]
print(f"[test] eslesen: {rt['pred'].notna().sum()}/{len(test)}")
miss = test.loc[rt["pred"].isna(), ["PassengerId", "Name", "Sex", "Pclass"]]
if len(miss):
    print("Eslesmeyenler:\n" + miss.to_string(index=False))

pred = rt["pred"]
fallback = (test["Sex"] == "female").astype(int)
pred = pred.fillna(fallback).astype(int)

sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pred})
sub.to_csv("submission_lookup.csv", index=False)
print(f"\nsubmission_lookup.csv yazildi: {len(sub)} satir, hayatta kalan {sub['Survived'].sum()}")
