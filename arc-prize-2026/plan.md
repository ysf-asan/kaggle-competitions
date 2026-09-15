# ARC-AGI-2 2026 — Mimari Kararı ve Uygulama Planı

Kaynak: Kaggle yarışma sayfası (overview / rules / code / models / discussion /
leaderboard), NVARC açık kaynak deposu ve public notebook kaynak kodu.
İnceleme tarihi 8 Eylül 2026. Hedef: **35-40 public LB bandı.**

---

## 0. Doğrulanmış kısıtlar

Yarışma id 133469. Kaggle API'sinden birebir çekilen değerler:

| | Değer |
|---|---|
| Günlük gönderim | **1** |
| Final için seçilebilecek gönderim | 2 |
| Takım büyüklüğü | maks **5** (Paper Track'te 8 — karıştırma) |
| CPU / GPU çalışma süresi | ≤ 720 dk (12 saat) |
| İnternet | Kapalı |
| Harici veri | Serbest, ücretsiz ve herkese açık olmak şartıyla (pretrained model dahil) |
| Çıktı | `submission.json` |
| Donanım | L4 x4, 96 GB VRAM — sadece bu yarışmaya bağlı notebook'larda |
| Public LB | Test verisinin %50'si |
| Entry + Team Merger | 26 Ekim 2026 |
| Son gönderim | 2 Kasım 2026 |

Bugünden son gönderime **55 gün / 55 submission** hakkı kalıyor. Henüz kayıt
olunmadı — "Join Competition" butonu aktif. **İlk yapılacak iş bu.**

Ödül yapısı: Progress Prizes 275k$ (8 sıra), Grand/Innovation Prize 275k$
(writeup rubriği), Bonus 150k$ (%85 eşiği). Ödüle hak kazanan çözümünü açık
kaynak yapmazsa diskalifiye.

### Veri dosyaları (yerel doğrulama yapıldı)

| Dosya | Task | Test çıktısı |
|---|---|---|
| `arc-agi_training_challenges.json` | 1000 | 1076 |
| `arc-agi_evaluation_challenges.json` | **120** | 172 |
| `arc-agi_test_challenges.json` | 240 | 259 |

Kritik: `test_challenges.json` içindeki 240 task'ın **240'ı da training setinin
alt kümesi**. Yani yerel test dosyası sadece bir yer tutucu; rerun sırasında 240
gizli task ile değiştiriliyor. Gerçek doğrulama seti elimizde sadece **120
task** — 1 task yaklaşık %0.83. Gürültü yüksek, tek task farkına anlam yükleme.

---

## 1. Leaderboard anatomisi

`arc3.huikang.dev/leaderboard/?comp=arc2` üzerinden submission başına runtime
dahil geçmiş:

```
 1. rabbithole    76.67   81 sub   12h 18m/koşu   (21.5 -> 34 -> 38 -> 44 -> 48 -> 70 -> 76 kademeli)
 2. nvbanana      72.08  110 sub    9h 52m/koşu   (CPMP + Darragh, NVARC ekibi)
 3. Tufa Labs     40.83   17 sub   12h 01m/koşu   (32.5 -> 32.9 -> 40.8, iki haftada)
 4. Yi-Chia Chen  39.58    7 sub   11h 22m/koşu   <- SADECE 7 GÖNDERİMLE
 5. Junhua Yang   37.22  160 sub   11h 11m/koşu
 6. Nguyen        35.97   44 sub    5h 44m/koşu   <- BÜTÇENİN YARISINI KULLANIYOR
 7. AI Winter     35.42   57 sub   11h 48m/koşu
 8. Kha Vo        34.86   76 sub   11h 32m/koşu   <- ödül burada bitiyor
 9. Bong          34.44  127 sub   11h 06m/koşu
10-49.            32.6-33.89       ~10h 40m       <- public notebook kümesi
```

Üç okuma:

1. **33.89 bandı hazır notebook.** 40 takım aynı kodu çalıştırıyor. Oradan
   çıkmak zorundayız, orada kalmak hiçbir şey ifade etmiyor.
2. **Yi-Chia Chen 7 gönderimde 39.58 yaptı.** Yani 35-40 bandı yüzlerce
   deneme gerektiren bir grind değil; doğru fikir + birkaç koşu.
3. **Nguyen 5h44'te 35.97 yapıyor.** 12 saatlik bütçenin yarısını kullanıyor.
   Bütçede ciddi boşluk var — bu doğrudan bir kaldıraç.

CPMP (2. sıra) forumda: "2nd team has clearly moved beyond what could be done
with knob fitting on our last year solution." Yani hazır çözümün
hiperparametrelerini oynatmak bir tavana çarpıyor; o tavan yaklaşık **35**.

---

## 2. Public 33.89 notebook'unun mimarisi

`koushikrudra/failed-in-aimo` (489 oy, GOLD). Kaynak kodu
`reference/failed-in-aimo-v1.py` altına indirildi. Bu, NVARC'ın 2025 kazanan
çözümünün ARChitects bileşeninin 2026'ya taşınmış hali.

**Model:** `ivan/qwen3_4b_grids15_sft139` (bfloat16). Tabanı
**Qwen3-4B-Thinking-2507**, 16 ek grid token'lı tokenizer ile; NVARC tarafından
32 GPU'da 12.716 adım SFT edilmiş.

**Boru hattı (task başına, 4 GPU worker paralel):**

1. **Test-time training.** Task'ın train çiftlerinden 16 augmentation üret
   (dihedral × renk permütasyonu), LoRA r=256 (tüm linear + `embed_tokens` +
   `lm_head`), rslora, lr 5e-5, 1 epoch, bf16, seq 8192. Her task için LoRA
   sıfırlanıyor.
2. **Decode.** `inference_turbo_dfs` — olasılık eşikli DFS token araması
   (`max_score = -log(0.2)`), 16 augmentation üzerinden, 4'lü batch'ler halinde.
3. **Yeniden puanlama.** Her aday grid'i 8 augmentation altında modele geri
   verip toplam log-olasılığını hesapla (`calc_scores`).
4. **Seçim.** `score_kgmon` — aynı grid'i üreten augmentation sayısı eksi
   ortalama augmentation skoru. En iyi 2 aday `attempt_1` / `attempt_2`.

**Zamanlama:** task başına 1200 s tavan, global 12 saat eksi 10 dakika.

**Kritik operasyonel numara:** notebook rerun modunda değilken sadece **4 eval
task'ı** işliyor. Bu yüzden "commit" koşusu 26 dakika sürüyor. Gerçek 240 task
sadece submission rerun'ında çalışıp ~10-11 saat alıyor.

---

## 3. İki kritik bulgu

### 3.1 Yerel eval seti kontamine

NVARC'ın SFT veri karışımı (`SDG/scripts/build_datasets.py`, satır 249):

```python
ds = convert_arc_to_messages("external/ARC-AGI-2/data/evaluation/*.json", num_samples=6)
ds.save_to_disk(f"{output_path}/arc2_evaluation6")
```

**ARC-AGI-2 public evaluation setinin 120 task'ı, 6 augmentation ile
`qwen3_4b_grids15_sft139` modelinin eğitim verisinde.** Yani elimizdeki tek
doğrulama seti, kullanacağımız base modelin gördüğü veri.

Sonucu forumda görünüyor: Bitterbot ekibi "11.67% LB vs ~36% local" raporladı.
Yerel skora bakıp karar veren herkes bu duvara çarpıyor.

**Doğrulama stratejisi buna göre kurulacak:**

- Yerel 120 eval seti → sadece **eşleştirilmiş karşılaştırma** için. Model
  ağırlığına dokunmayan değişiklikler (seçim algoritması, zaman bütçesi,
  ensemble) için deltalar taşınır. Mutlak skor asla raporlanmaz.
- **ARC-AGI-1 evaluation (400 task)** → temiz holdout. Karışımda yok
  (`rearc` ARC-AGI-1 *training*'i kapsıyor, evaluation'ı değil). Kolay bir set
  ama regresyon dedektörü olarak sağlam. `arcprize/ARC-AGI` reposundan alınır,
  harici veri kuralına uygun.
- **Leaderboard** → model ağırlığına dokunan her değişiklik için tek geçerli
  sinyal. Günde 1, dikkatli harcanacak.

### 3.2 Quota ile submission ayrı

Junhua Yang 160 gönderim × ~12 saat × 4 GPU yaptı. Haftalık 30 saatlik Kaggle
quota'sıyla bu imkânsız. Yani **submission rerun'ları quota yakmıyor**; sadece
"Save & Run All" (commit) koşusu yakıyor, L4x4'te 2× hızda.

Bu, geliştirme döngüsünü belirliyor:

- Commit koşusunu küçük tut (public notebook'un yaptığı gibi 4-20 task).
  30-60 dk × 2 = quota'dan 1-2 saat.
- Ağır doğrulamayı günde 1 submission ile LB'den al.
- Haftada ~15 saat efektif L4x4 quota → ~15-20 commit. Yeterli.

---

## 4. Seçenekler — teker teker

Beklenen kazanç tahminleri 33.9 tabanı üzerinden.

### A. Uyarlanabilir hesap bütçesi — **+1 ila +3, düşük risk, düşük maliyet**

Şu an her task'a sabit bütçe veriliyor: 16 TTT augmentation, 1 epoch, 8 decode
batch'i. 240 task / 4 GPU / 12 saat = task başına ~12 dk ortalama; tavan 1200 s.
Kolay task'lar erken bitiyor, artan zaman **çöpe gidiyor**. Nguyen'in 5h44'te
35.97 yapması bu boşluğun kanıtı.

Yapılacak: iki geçişli planlama.
- **Geçiş 1:** 240 task, ucuz bütçe (~5 dk). Her task için güven ölçüsü çıkar —
  augmentation'lar arası mutabakat, en iyi ile ikinci aday arasındaki skor farkı,
  DFS'in kaç farklı grid ürettiği.
- **Geçiş 2:** en düşük güvenli %35-40'lık dilime kalan tüm zamanı dağıt —
  2-3× TTT epoch, 32 augmentation, daha derin DFS, daha gevşek eşik.

Bu, 2025 teknik raporunun yılın teması dediği "feedback sinyaliyle yönlendirilen
iyileştirme döngüsü"nün en ucuz hali. Model ağırlığına dokunmuyor → yerel eval
ile ölçülebilir.

### B. TRM ensemble — **+1 ila +3, orta risk, düşük maliyet**

NVARC'ın çözümü iki bileşenliydi; public notebook sadece birincisini içeriyor.
İkincisi Tiny Recursive Model — 7M parametre, pretrain yok, özyinelemeli latent
akıl yürütme. Tamamen farklı bir tümevarım önyargısı, dolayısıyla **farklı
task'ları çözüyor**.

Ağırlıklar public: `cpmpml/arc-prize-trm-031` Kaggle dataset'i (`step_220708`).
Referans notebook `cpmpml/arc2-trm-v31`, tek başına LB **10.0** (pass@2 %10.1).
Çalışma maliyeti dakikalar mertebesinde.

**Risk:** naif birleştirme skoru düşürüyor — `christopherdaleman/arc-2026-nvarc-trm-evidence-cost-v1`
notebook'u NVARC+TRM ile 31.11, yani 33.89'un altında. Doğru yapılışı: TRM
adaylarını **Qwen scorer'ına ver**, ortak para birimiyle sırala. TRM adayı
yüksek güvenli bir LLM adayını asla itmesin; sadece `attempt_2` doldursun veya
LLM'in hiç aday üretemediği task'ta devreye girsin.

### C. CPU'da doğrulanmış program araması — **+0.5 ila +2, düşük risk, sıfır GPU maliyeti**

L4x4 makinesinde GPU'lar çalışırken onlarca vCPU boşta. Paralel CPU
süreçlerinde klasik DSL/arama çözücüsü (icecuber tarzı) koştur. **Sadece tüm
train çiftlerini birebir üreten programlar kabul edilir** — sert doğrulayıcı,
dolayısıyla çok yüksek kesinlik. Bulunduğunda `attempt_1`'i ezer.

ARC-AGI-2 bu tür aramaya ARC-AGI-1'den çok daha dirençli; task'ların belki
%2-5'i. Ama GPU maliyeti sıfır.

**Bağlam uzunluğu — ölçüldü, büyük sorun değil.** `max_seq_length=8192` ve
`cut_to_len` sığmayan train çiftlerini atıyor. Hücre başına 1 token varsayımıyla
yerelde hesaplandı:

```
eval-120     medyan 2390 tok, p90 4164, maks 8474  ->  8192'yi aşan: 1 task (%1)
train-1000   medyan 1020 tok, p90 2964, maks 8474  ->  8192'yi aşan: 1 task (%0)
```

Yani kesilme kaybı ihmal edilebilir. Ama başka bir şey görünüyor: **eval
task'ları training task'larından medyanda 2.3× daha uzun** (2390 vs 1020).
Base modelin SFT dağılımı belirgin biçimde daha küçük grid'lerden oluşuyor.
Uzun task'larda başarı oranını ayrı raporla — muhtemelen orada belirgin bir
düşüş var ve geçiş 2'nin bütçesi öncelikle oraya gitmeli.

### D. Aday seçimi / sıralama — **+0.5 ila +2, düşük risk, düşük maliyet**

Şu an `score_kgmon` sabit. Elde zaten zengin öznitelik var: `beam_score`, 8
augmentation skorunun ortalaması/minimumu/varyansı, aynı grid'i üreten farklı
augmentation sayısı. Eklenebilecekler: çıktı boyutunun train çiftlerinden
çıkarılan boyut kuralına uyması, renk kümesi tutarlılığı, aday grid'in train
çıktılarıyla yapısal benzerliği.

İki hakkımız var — `attempt_2`, `attempt_1`'in biraz farklı bir versiyonu
olmamalı. **Mod çeşitliliği zorla:** attempt_2, attempt_1'den yeterince farklı
bir kümeden gelsin. Notebook'ta zaten `benchmark_selection_algos` var, üzerine
kurulur.

### E. Base modelin yeniden SFT'lenmesi — **+2 ila +5, yüksek maliyet, ÖNERİLMİYOR**

NVARC 103k sentetik puzzle + 3.2M augmented alt küme yayınladı
(`sorokin/nvarc-synthetic-puzzles`, `sorokin/nvarc-augmented-puzzles`). Ama
`qwen3_4b_grids15_sft139` zaten bunların üstünde eğitildi. Yeni veri üretme
fikri olmadan marjinal. Ayrıca 4 node × 8 GPU'luk bir iş — kiralık GPU'da
dört haneli dolar. **Bu bütçede ana hat olamaz.**

### F. Program sentezi / araç kullanan ajan — **belirsiz ama çok yüksek tavan, yüksek maliyet**

Zirvedeki iki takımın yaptığı bu. Forumda cm391'in işaret ettiği ipucu:
NVIDIA `nvidia/Nemotron-SFT-ARC-AGI-v1` datasetini yayınladı (CC-BY 4.0) —
ARC puzzle'larını çözen çok turlu ajan izleri; araçlar: Python executor,
doğal dil grid manipülatörü, çözüm gönderme. Model bir `transform(grid)`
fonksiyonu yazıyor, çalıştırılıyor, ground truth ile birebir eşleşenler
tutulmuş. 67.5k (geniş) / 207.5k (çok çözümlü) örnek, 18.4 GB. Kardeş dataset
`nvidia/Nemotron-RL-ARC-AGI-v1` de var.

**Sıçramanın kaynağı büyük olasılıkla bu.** 2. takımın izlediği yol: 21.5 → 34
→ 38 → 44 → 48 → 70 → 76.

Ama: 8-14B modelin SFT'i gerekiyor, elde hazır güvenilir checkpoint yok
(HuggingFace'te Trelis'in Qwen3-4B / gpt-oss-20B denemeleri var ama indirme
sayıları tek haneli, kalite belirsiz). Kiralık GPU'da 2-4 hafta ve 300-1500$.
Ayrıca çıkarım maliyeti TTT ile çakışıyor — TTT task başına sıralı fine-tune
istiyor, program sentezi ise toplu throughput.

**Karar: ana hat değil, paralel/stretch hat.** Ana hat 35-38'i garantiliyorsa,
bu 40+'ın tek gerçekçi kapısı. Ve Grand Prize rubriğinde (Novelty, Progress)
en değerli parça.

---

## 5. Karar

### Ana hat — hedef 36-38

```
NVARC/ARChitects tabanı (Qwen3-4B-Thinking + task başına LoRA TTT)
  + A: iki geçişli uyarlanabilir hesap bütçesi
  + D: öznitelik tabanlı aday sıralama + attempt_2 çeşitlilik zorlaması
  + C: boşta CPU'da doğrulanmış DSL araması (uzun grid kurtarma dahil)
  + B: TRM ensemble, Qwen scorer'ı ortak para birimi olarak
```

Hiçbiri model ağırlığına dokunmuyor → hepsi yerel 120 task üzerinde
**eşleştirilmiş** olarak ölçülebilir, kontaminasyon deltayı bozmaz.

### Stretch hat — hedef 40-42

Nemotron SFT izlerinin filtrelenmiş 20-30k'lık alt kümesiyle Qwen3-4B-Thinking
üzerinde LoRA SFT (kiralık tek A100, ~100-300$). Çıkarımda **üçüncü aday
üreteci** olarak, sert doğrulama ile: ürettiği Python programı tüm train
çiftlerini birebir üretmiyorsa aday atılır. Doğrulanmış olduğu için skoru
düşüremez, sadece yükseltebilir.

Zaman bölüşümü: saat 0-8 TTT hattı, saat 8-11 çözülmemiş task'larda program
sentezi. (GPU'ları bölmek yerine zamanı bölmek daha temiz — vLLM ile TTT aynı
anda aynı kartta iyi geçinmiyor.)

### Neden bu seçim

- Ana hattın dört bileşeni de **bağımsız** ve **birikimli**; biri tutmazsa
  diğerleri ayakta kalır.
- Hiçbiri yeniden pretrain gerektirmiyor — yerel donanımla (RTX 3050 Ti 4 GB,
  16 GB RAM) uyumlu, tüm ağır iş Kaggle'da.
- Hepsi 2025 teknik raporunun "refinement loop" temasına oturuyor; writeup'ta
  Theory ve Progress kriterleri için anlatılacak bir hikâye var.
- Stretch hat başarısız olsa bile ana hat ödül bandına (8. sıra 34.86) yakın
  duruyor ve madalya bandını rahat geçiyor.

---

## 6. Sekiz haftalık plan

Bugün 8 Eylül. Entry deadline 26 Ekim, son gönderim 2 Kasım, writeup 9 Kasım.

**Hafta 1 (8-14 Eylül) — zemini kilitle**
- [ ] ARC-AGI-2, ARC-AGI-3 ve Paper Track'e kayıt + kural kabulü. 26 Ekim'i
      bekleme, hiçbir faydası yok.
- [ ] `failed-in-aimo` notebook'unu fork'la, olduğu gibi submit et, 33.89'u
      doğrula. Bu bizim referans noktamız.
- [ ] Yerel altyapı: grid görselleştirici, 120 eval task'ı üzerinde
      task × çözüldü/çözülmedi tablosu, eşleştirilmiş karşılaştırma scripti.
- [ ] ARC-AGI-1 evaluation (400 task) setini temiz holdout olarak indir,
      Kaggle dataset'i olarak yükle.
- [ ] Task uzunluğu × başarı tablosu çıkar (uzunluk hesabı yapıldı: eval medyanı
      2390 token, training medyanı 1020 — başarının uzunlukla nasıl düştüğünü ölç).

**Hafta 2 (15-21 Eylül) — ucuz kazanımlar**
- [x] A: iki geçişli bütçe kuruldu (`src/arc_confidence.py`, `src/starter.py`).
      Güven ölçüsü = kapsama × (mutabakat, marj); 1. geçişin hiç ulaşamadığı
      task'lar 2. geçiş kuyruğunun başında.
- [x] D: çıktı şekli önceliği (`src/arc_decoder.py`). Eleme değil yeniden
      sıralama — hiçbir aday kaybolmuyor. `benchmark_selection_algos` artık
      önceliğin açık/kapalı halini eşleştirilmiş karşılaştırıyor.
- [ ] Commit koşusu ile duman testi, sonra LB'de ölç.
- Beklenen: 33.9 → 35-36.

**Hafta 3 (22-28 Eylül) — çeşitlilik**
- [ ] C: CPU DSL çözücüsü, sert doğrulayıcı, GPU worker'larıyla paralel süreç.
      Uzun-grid task'ları öncelikle hedefle.
- [ ] B: TRM checkpoint'ini yükle, ayrı bir aday üreteci olarak çalıştır,
      Qwen scorer'ıyla ortak sıralamaya sok.
- Beklenen: 36-38.

**Hafta 4-6 (29 Eylül - 19 Ekim) — stretch hattı**
- [ ] Nemotron SFT datasetini incele, birebir eşleşen izleri filtrele,
      20-30k'lık alt küme çıkar.
- [ ] Kiralık A100'de Qwen3-4B-Thinking üzerine LoRA SFT.
- [ ] Kaggle'da offline Python executor sandbox'ı + doğrulama döngüsü.
- [ ] Zaman bölüşümlü entegrasyon, LB'de ölç.
- Karar noktası **19 Ekim**: stretch hat LB'de ana hattı geçmediyse bırak,
      ana hattı cilala.

**Hafta 7 (20-26 Ekim) — kilitleme**
- [ ] **26 Ekim entry deadline** — kayıt ve takım durumu kesinleşmiş olmalı.
- [ ] Zaman güvenliği: her aşamada `end_time` kontrolü, timeout'ta kısmi
      submission yazma. CPMP'nin 8 gün önce yaşadığı gibi altyapı kaynaklı
      timeout'lar oluyor; 12 saati son dakikaya kadar doldurma, ~11h hedefle.

**Hafta 8 (27 Ekim - 2 Kasım) — son gönderimler**
- [ ] Son ayarlar, 2 final submission seçimi. Birini muhafazakâr (kanıtlanmış,
      düşük varyans), diğerini agresif seç.
- [ ] Rerun kuyruğu deadline'a yakın tıkanıyor — son submission'ı 31 Ekim'e kadar
      bitir.

**9 Kasım'a kadar — writeup**
- [ ] Paper Track + Grand Prize writeup. Rubriğin 6 kriterinden sadece 1'i skor.
      Kalan 5'i (Universality, Progress, Theory, Completeness, Novelty) fikrin
      kalitesi. Hafta 1-3'te çıkarılan hata analizi tablosu bu metnin hammaddesi.
- [ ] Public notebook + kapak görseli + maks 1500 kelime.

---

## 7. Riskler

| Risk | Azaltma |
|---|---|
| Yerel eval kontamine, yanlış yöne koşma | Mutlak skor kullanma; eşleştirilmiş delta + ARC-AGI-1 holdout + LB |
| Kaggle altyapı timeout'u (CPMP'de oldu) | ~11h hedefle, her aşamada kısmi çıktı yaz, submission iadesi talep et |
| Naif TRM birleştirmesi skoru düşürür | TRM sadece attempt_2 veya LLM'in boş kaldığı yerde; ortak scorer |
| Stretch hat zamanı yer | 19 Ekim'de sert karar noktası |
| Public → private LB kayması | Final'de 2 gönderimden biri düşük varyanslı |
| Quota | Commit koşularını 4-20 task'ta tut, ağır doğrulamayı LB'ye bırak |
| Disk (26 GB boş) | 18 GB'lık Nemotron dataseti yerelde tutulmaz; doğrudan Kaggle/kiralık makinede işlenir |

## 8. Donanım gerçeği

Yerel makine: RTX 3050 Ti Laptop **4 GB VRAM**, 16 GB RAM, 26 GB boş disk.
Bu makinede hiçbir model eğitilemez, 4B model çıkarımı bile yapılamaz. Yerel
makinenin rolü: veri analizi, görselleştirme, CPU DSL çözücüsü geliştirme,
notebook yazımı. Tüm GPU işi Kaggle'da (ana hat) veya kiralık makinede
(stretch hat).

## 9. Referans linkler

- Public 33.89 notebook: `kaggle.com/code/koushikrudra/failed-in-aimo`
- Base model: `kaggle.com/models/ivan/qwen3_4b_grids15_sft139`
- TRM ağırlıkları: `kaggle.com/datasets/cpmpml/arc-prize-trm-031`
- TRM notebook: `kaggle.com/code/cpmpml/arc2-trm-v31`
- NVARC kaynak kodu: `github.com/1ytic/NVARC`
- Sentetik puzzle'lar: `kaggle.com/datasets/sorokin/nvarc-synthetic-puzzles`
- Nemotron SFT: `huggingface.co/datasets/nvidia/Nemotron-SFT-ARC-AGI-v1`
- Nemotron RL: `huggingface.co/datasets/nvidia/Nemotron-RL-ARC-AGI-v1`
- Runtime'lı LB geçmişi: `arc3.huikang.dev/leaderboard/?comp=arc2`
- 2025 teknik raporu: arXiv 2601.10904
