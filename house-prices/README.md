# House Prices – Advanced Regression Techniques

Ames (Iowa) konut veri setinde 79 açıklayıcı değişkenden ev satış fiyatı tahmini.
Kaggle metriği: `log(tahmin)` ile `log(gerçek)` arasındaki RMSE.

**Sonuç: 10-kat çapraz doğrulamada harman RMSE = 0.10471** (iç içe doğrulanmış: 0.10562).

---

## Yaklaşım

Üç ayrı temsil üretilip her model kendi sevdiği matrisle eğitiliyor, ardından
11 modelin katlar-dışı (out-of-fold) tahminleri negatif olmayan en küçük
karelerle harmanlanıyor.

### 1. Veri temizliği (`features.py`)

Bu veri setinin en kritik inceliği: `NA` çoğu sütunda **eksik veri değil, "yok"
anlamına gelen gerçek bir kategori** (havuz yok, bodrum yok, şömine yok).
`pandas` bunu varsayılan olarak `NaN`'a çevirdiği için `keep_default_na=False`
ile okunuyor, sayısal sütunlar ayrıca zorla `NaN`'a dönüştürülüyor.

| Sütun grubu | Doldurma stratejisi |
|---|---|
| `Alley`, `PoolQC`, `Fence`, `Bsmt*`, `Garage*`, `FireplaceQu`, `MasVnrType` | `"None"` (yapı gerçekten yok) |
| `MasVnrArea`, `BsmtFinSF*`, `GarageCars/Area` | `0` |
| `LotFrontage` | Mahalle medyanı (parsel genişliği mahalleye göre düzenli) |
| `GarageYrBlt` | `YearBuilt` |
| `MSZoning` | Aynı `MSSubClass` içindeki mod |
| `Functional` | `"Typ"` (veri sözlüğünün belirttiği varsayılan) |

**Aykırı değerler:** De Cock'un makalesinde uyardığı iki ev atılıyor —
`GrLivArea > 4000` ama `SalePrice < 300000` (kısmi satışlar, gerçek piyasa
fiyatı değil).

### 2. Öznitelik mühendisliği

**Sıralı kodlama (20 sütun).** `Ex/Gd/TA/Fa/Po/None` gibi kalite ölçekleri
tek yönlü kukla değişkene değil `5..0` sayısına çevriliyor — sıralama bilgisi
korunuyor. Aynısı `BsmtExposure`, `BsmtFinType1/2`, `GarageFinish`,
`Functional`, `Fence`, `LotShape`, `LandSlope`, `PavedDrive` için.

**Türetilmiş öznitelikler (~25 adet):**

- *Toplamlar:* `TotalSF` (bodrum + 1. kat + 2. kat), `TotalFinSF`, `TotalBath`
  (yarım banyolar 0.5 ağırlıkla), `TotalPorchSF`
- *Yaş:* `Age`, `RemodAge`, `GarageAge`, `IsRemodeled`, `IsNew`
- *Varlık bayrakları:* `HasPool`, `Has2ndFloor`, `HasGarage`, `HasBsmt`,
  `HasFireplace`, `HasPorch`
- *Kalite × büyüklük etkileşimleri:* `QualxSF`, `QualxLiv`, `OverallGrade`,
  `BsmtGrade`, `GarageGrade`, `QualSum` — fiyatın en güçlü sinyali, çünkü aynı
  metrekare farklı kalitede çok farklı fiyatlanıyor
- *Oranlar:* `LivAreaPerRoom`, `SFPerLot`, `SpaciousRatio`

**Çarpıklık düzeltmesi.** Lineer modeller için `|skew| > 0.5` olan sayısal
sütunlara Box-Cox(1p), λ=0.15 uygulanıyor. Hedef değişken `log1p(SalePrice)`
olarak modelleniyor — bu zaten yarışma metriğiyle birebir örtüşüyor.

### 3. Üç matris

| Matris | İçerik | Kullanan modeller |
|---|---|---|
| Ağaç (1458 × 278) | One-hot, çarpıklık düzeltmesi yok | GBR, XGBoost, LightGBM, ExtraTrees |
| Lineer (1458 × 222) | Box-Cox + one-hot + `RobustScaler`; 10'dan az örnekte görülen seyrek kuklalar atılmış | Lasso, Ridge, ElasticNet, KernelRidge, SVR, Huber |
| Kategorik | Kategorikler metin olarak | CatBoost (yerel kategorik desteği) |

### 4. Modeller ve harman (`train.py`)

10-kat `KFold` ile her modelin OOF tahminleri üretiliyor, test tahmini katların
ortalaması olarak alınıyor. Ağırlıklar OOF matrisi üzerinde **NNLS** ile
çözülüyor (negatif ağırlık yok → aşırı uyumu sınırlar).

| Model | CV RMSE (log) | Harman ağırlığı |
|---|---|---|
| Lasso | 0.10883 | **0.296** |
| ElasticNet | 0.10898 | – |
| SVR | 0.10952 | **0.225** |
| XGBoost | 0.10979 | **0.296** |
| KernelRidge | 0.11084 | – |
| Ridge | 0.11109 | – |
| CatBoost | 0.11154 | **0.076** |
| GradientBoosting (huber) | 0.11195 | **0.023** |
| LightGBM | 0.11518 | – |
| Huber | 0.11530 | **0.084** |
| ExtraTrees | 0.12335 | – |
| **NNLS harman** | **0.10471** | |
| İç içe doğrulanmış harman | 0.10562 | |

NNLS ağırlıkları tüm OOF üzerinde bulunduğu için harman skoru bir miktar
iyimser. Gerçekçi genelleme tahmini **iç içe doğrulanmış 0.10562** — ağırlıkların
kendisi de dış katmanda ayrı bir `KFold` ile yeniden öğrenilerek ölçülüyor.

Dikkat çekici nokta: en iyi tekil model bir **lineer** model (Lasso, 0.10883),
gradyan artırma değil. 1458 satır ve 278 sütunla veri, düzenlileştirilmiş
lineer modelin tercih edeceği rejimde. Harmanın kazancı da buradan geliyor —
lineer modeller (Lasso + SVR + Huber = 0.605 ağırlık) ile ağaçlar (XGB + Cat +
GBR = 0.395) farklı hatalar yapıyor.

---

## Çalıştırma

```bash
python train.py
```

Toplam ~35 dakika (CatBoost'un 10-kat çaprazı tek başına ~23 dakika).
Çıktı: `submission.csv` (1459 satır) ve `oof_cache.npz` (yeniden eğitmeden
harman denemeleri için OOF/test matrisleri).

### Dosyalar

```
features.py    Veri yükleme, temizlik, öznitelik üretimi, çarpıklık düzeltmesi
train.py       Model tanımları, 10-kat OOF döngüsü, NNLS harmanı, submission
submission.csv Kaggle'a yüklenmeye hazır tahminler
oof_cache.npz  Önbelleğe alınmış OOF/test tahminleri
```

### Bağımlılıklar

`numpy` `pandas` `scipy` `scikit-learn` `xgboost` `lightgbm` `catboost`

---

## Leaderboard hakkında dürüst not

Bu yarışmanın liderlik tablosunda 0.00000 skorlar var. Ames veri seti
[kamuya açık](https://jse.amstat.org/v19n3/decock.pdf) olduğundan, test
setindeki evleri orijinal kaynakla eşleştirip gerçek fiyatları yapıştırmak
mümkün. Bu çözüm bunu yapmıyor; skorlar tamamen modelden geliyor.

## Sonraki adımlar

- **Çok tohumlu ortalama** — aynı modelleri 5 farklı `random_state` ile eğitip
  ortalamak, varyansı düşürür (~0.001-0.002)
- **`Neighborhood` için hedef kodlama** — kat içinde hesaplanmalı, aksi halde
  sızıntı olur
- **Pahalı ev kalibrasyonu** — model üst uçta sistematik olarak düşük tahmin
  ediyor; tahmin dağılımının uçlarına hafif ölçekleme
- **Pseudo-labeling** — test tahminlerinin en güvenilir kısmını eğitime katmak
