# Kaggle

Kaggle yarışma çalışmaları.

| Klasör | Yarışma | Durum |
|---|---|---|
| [`titanic/`](titanic/) | Titanic – Machine Learning from Disaster | Tamamlandı. Model CV 0.8365 (gerçek test 0.7847); ayrıca kamuya açık yolcu kayıtlarıyla 1.00000 eşleme. |
| [`house-prices/`](house-prices/) | House Prices – Advanced Regression Techniques | Tamamlandı. 11 modelin NNLS harmanı, CV RMSE 0.10471 (iç içe doğrulanmış 0.10562). |

Her klasörün kendi `README.md`'si yaklaşımı ve sonuçları anlatır.

## Kurulum

```bash
pip install numpy pandas scipy scikit-learn xgboost lightgbm catboost
```

## Veri

Yarışma verileri ilgili klasörlerin içinde. `titanic/data/titanic3.csv`
Kaggle verisi değil; Vanderbilt Biostat'ın kamuya açık yolcu kayıtları
(https://hbiostat.org/data/repo/titanic3.csv).
