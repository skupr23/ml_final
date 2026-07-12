# Walmart Recruiting - Store Sales Forecasting

**Kaggle competition:** [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting)
**ამოცანა:** Time-Series Forecasting — 45 Walmart-ის მაღაზიის, 81 დეპარტამენტის კვირეული გაყიდვების პროგნოზირება, დღესასწაულების ეფექტების გათვალისწინებით.

---

## 1. პროექტის მიმოხილვა

პროექტის მიზანი იყო დროითი მწკრივების (time-series) პროგნოზირებისთვის არსებული სხვადასხვა არქიტექტურის — tree-based, classical statistical, და deep learning — შესწავლა, იმპლემენტაცია და შედარება ერთმანეთთან, ერთსა და იმავე მონაცემებზე, ერთი და იგივე შეფასების სქემით.

დავტესტეთ და დავლოგეთ შემდეგი არქიტექტურები:

| კატეგორია | მოდელი |
|---|---|
| Tree-Based | LightGBM, XGBoost |
| Ensemble | LightGBM + XGBoost (weighted blend) |
| Classical Statistical | ARIMA, SARIMA |
| Foundation/Local-fit Statistical | Prophet |
| Deep Learning | DLinear, N-BEATS, PatchTST |

ყველა ექსპერიმენტი დალოგილია **MLflow/DagsHub**-ზე, ცალკე experiment თითოეული არქიტექტურისთვის, ხოლო ყველა run შესაბამისობაშია დავალების მოთხოვნილ სტრუქტურასთან (`{Model}_Cleaning`, `{Model}_Feature_Selection`, `{Model}_Validation`, `{Model}_Final_Model`).

---

## 2. მონაცემები

- **train.csv** — 421,570 რიგი, 45 მაღაზია × 81 დეპარტამენტი = 3,331 უნიკალური Store-Dept წყვილი, კვირეული გაყიდვები 2010-02-05-დან 2012-10-26-მდე.
- **test.csv** — 115,064 რიგი, 39 კვირის პროგნოზის ჰორიზონტი.
- **features.csv** — Temperature, Fuel_Price, CPI, Unemployment, 5 MarkDown სვეტი, IsHoliday.
- **stores.csv** — Store Type და Size.

**მთავარი დაკვირვებები EDA-დან:**
- მონაცემები არის unbalanced panel — ყველა დეპარტამენტი ყველა მაღაზიაში არ არსებობს.
- ძლიერი წლიური სეზონურობა — 52-კვირიანი lag ერთ-ერთი ყველაზე ძლიერი პრედიქტორია (ეს დადასტურდა ჩვენი baseline-ის შედეგებითაც, იხ. ქვემოთ).
- Holiday კვირები წონდება 5x სტანდარტული კვირის მიმართ (WMAE-ს განმარტება).
- MarkDown სვეტებში ბევრი missing მნიშვნელობაა — დავამუშავეთ როგორც missing indicator + fillna(0), არა უბრალო წაშლა.

---

## 3. Feature Engineering (`feature_engineering.ipynb`)

გავაერთიანეთ train/features/stores, შემდეგ ავაშენეთ:

- **MarkDown დამუშავება:** `{col}_missing` ინდიკატორები + fillna(0), `markdown_total`, `markdown_count`, `has_any_markdown`.
- **CPI/Unemployment:** per-Store forward-fill, train+test გაერთიანებულზე, რომ test-მა მემკვიდრეობით მიიღოს ბოლო ცნობილი მნიშვნელობა.
- **Holiday type:** ცალკე კატეგორია SuperBowl / LaborDay / Thanksgiving / Christmas / Normal-ისთვის (არა მხოლოდ ბინარული IsHoliday), რადგან სხვადასხვა holiday განსხვავებულად მოქმედებს გაყიდვებზე.
- **კალენდარული features:** year, month, week_of_year, week_sin/cos (ციკლური კოდირება).
- **Lag features:** ზუსტი თარიღით აგებული (არა row-based shift, რომელიც shift-ს გამოიწვევდა გაფურჩქვნილ (gapped) სერიებზე) — lag_1, 2, 4, 13, 26, 51, 52, 53.
- **Rolling stats:** rolling_mean_4/13/26, rolling_median_8, ყოველთვის `shift(1)`-ის შემდეგ, რომ არ მომხდარიყო target leakage.
- **Fallback აგრეგატები:** dept/store/store-dept median, გამოთვლილი მხოლოდ train-ზე, test-ში unseen კომბინაციებისთვის.

**⚠️ მნიშვნელოვანი აღმოჩენა (feature availability bug):** საწყის ვერსიაში `lag_1/2/4/13/26` ვიყენებდით feature-ებად, თუმცა აღმოვაჩინეთ, რომ ეს lag-ები რეალურ Kaggle test set-ზე NaN-ია უმეტეს რიგში — რადგან 1-26 კვირით უკან საჭირო გაყიდვების მონაცემი თავად test-ის პერიოდშივე ხვდება, სადაც რეალური გაყიდვები არ გვაქვს (არ ვიყენებთ autoregression-ს). ეს იწვევდა train/production მისმატებას: validation WMAE ძალიან კარგი გამოდიოდა (~1300), მაგრამ რეალურ Kaggle submission-ზე მივიღეთ **5961** — რადგან internal validation split თავად train-ის შიგნით ხდება, სადაც ეს lag-ები ყოველთვის ხელმისაწვდომია. გამოსწორების შემდეგ (მოკლე lag-ების გამორიცხვა feature ნაკრებიდან) მივიღეთ რეალისტური, production-თან თანმიმდევრული შედეგები.

---

## 4. მეთოდოლოგია

- **შეფასების მეტრიკა:** Weighted Mean Absolute Error (WMAE), holiday კვირები წონდება 5x.
- **Validation სქემა:** ქრონოლოგიური holdout — ბოლო 39 კვირა (იგივე ჰორიზონტი, რაც Kaggle test set-ს აქვს), **არა** random k-fold, რადგან random split დროით მწკრივში leakage-ს გამოიწვევდა (მომავალზე დატრენინგება წარსულის პროგნოზირებისთვის).
- **Baseline:** 52-კვირიანი seasonal naive ("რა იყო გასულ წელს ამავე კვირას"), fallback-ით store-dept → dept → global median-ზე.
- **Feature Selection სტრატეგია მოდელის ტიპის მიხედვით** (დეტალურად ქვემოთ თითოეულ მოდელზე).
- **Model Registry:** ყველა მოდელისთვის საბოლოო pipeline დალოგილია MLflow-ში, მაგრამ **მხოლოდ საუკეთესო საერთო არქიტექტურაა რეგისტრირებული** Model Registry-ში (`REGISTER_AS_GLOBAL_BEST` flag თითოეულ notebook-ში).

---

## 5. მოდელების შედეგები

### 5.1 Baseline — 52-Week Seasonal Naive

სრულ dataset-ზე (val split, 118,540 row): **WMAE = 1789.91**

ეს არის ბაზისი, რომელსაც ყველა შემდეგი მოდელი უნდა სჯობნიდეს.

### 5.2 LightGBM

- **Feature selection:** ერთჯერადი importance-based prune (ტრეინინგი ყველა feature-ზე → ჩამოშლა 0-importance feature-ების → ხელახალი ტრეინინგი). დავაკლეთ 6 feature 42-დან.
- **Validation WMAE:** 1384.82 (holiday: 1464.62, non-holiday: 1363.67)
- **გაუმჯობესება baseline-თან შედარებით:** 405.10

### 5.3 XGBoost

- იგივე feature selection ლოგიკა, `tree_method='hist'`, GPU-ზე დატრენინგებული.
- **Validation WMAE:** 1361.86 (holiday: 1446.32, non-holiday: 1339.48)
- **გაუმჯობესება baseline-თან შედარებით:** 428.05
- XGBoost ოდნავ სჯობს LightGBM-ს — შესაძლო ახსნა: `max_depth=8` მეტ მოქნილობას აძლევს მოდელს, ვიდრე LightGBM-ის default `num_leaves=64`.

### 5.4 Ensemble (LightGBM + XGBoost)

Weighted blend, წონები მოძებნილია grid search-ით (0-დან 1-მდე, ნაბიჯი 0.05):

- **საუკეთესო წონები:** XGBoost 0.70 / LightGBM 0.30
- **Ensemble WMAE:** 1354.77
- **გაუმჯობესება baseline-თან შედარებით:** 435.14
- **გაუმჯობესება საუკეთესო ცალკეულ მოდელთან შედარებით (XGBoost):** 5.93

Ensemble მცირედით სჯობს ორივე ცალკეულ tree-based მოდელს — მოსალოდნელი შედეგია, რადგან LightGBM და XGBoost ოდნავ განსხვავებულ შეცდომებს უშვებენ.

### 5.5 Prophet

- **მიდგომა:** ცალკე Prophet მოდელი თითოეული Store-Dept წყვილისთვის (3,167 მოდელი, 41 fallback მოკლე ისტორიის მქონე წყვილებისთვის).
- **Feature selection:** Prophet-ის საკუთარი trend+seasonality decomposition-ის გამო, გარეშე regressor-ები არ გამოვიყენეთ; holiday ეფექტები დამატებულია Prophet-ის სპეციალური `holidays` პარამეტრით (კონკურსის რეალური თარიღები, არა Prophet-ის ჩაშენებული US calendar).
- **Validation WMAE:** 1905.42 (holiday: 1896.99, non-holiday: 1907.66)
- **შედეგი baseline-თან შედარებით:** **-115.51** (Prophet უარესია baseline-ზე)
- **მიზეზი:** per-pair WMAE-ის განაწილება ძალიან ფართოა (median 1040, საშუალო 1931, max 67,959) — ცალკეული Prophet მოდელები, განსაკუთრებით მცირე/noisy სერიებზე, ცუდად ერგებიან. ეს არის საინტერესო თეორიული დასკვნა: local per-series მოდელები, ცალკე დატრენინგებული სუსტი ისტორიის მქონე სერიებზე, ვერ სარგებლობენ tree-based მოდელების global-ი სწავლის უპირატესობით.

### 5.6 DLinear

- **არქიტექტურა:** trend/seasonal decomposition + 2 წრფივი შრე (linear layer).
- **Feature selection:** სუფთა univariate — მხოლოდ Weekly_Sales history, არავითარი გარეშე covariate, ორიგინალი DLinear paper-ის დიზაინის შესაბამისად.
- **Validation WMAE:** 1814.32 (holiday: 2122.09, non-holiday: 1732.75)
- **შედეგი baseline-თან შედარებით:** **-24.41** (მცირედით უარესი baseline-ზე)
- **მიზეზი:** DLinear-ს არ აქვს holiday/calendar-ის ცოდნა (მისი univariate ბუნების გამო) — Holiday WMAE (2122) ბევრად უარესია non-holiday-ზე (1733), რაც ადასტურებს, რომ 52-week baseline-ის implicit seasonal ცოდნა (რაც წინა წლის იმავე კვირას იმეორებს) აჯობებს DLinear-ის სუფთა linear decomposition-ს, სულ მცირე ისეთ dataset-ზე, სადაც holiday spikes დომინანტურია.

### 5.7 N-BEATS

- **Cross-validation:** 3 კონფიგურაცია (`small_underfit`, `balanced`, `deep_regularized`), 3-fold expanding window CV.
- **საუკეთესო კონფიგურაცია:** `balanced` (hidden_size=256, 2 stacks, 2 blocks, CV mean WMAE 2543.29)
- **Validation WMAE (39-week holdout):** 1758.13 (holiday: 1769.76, non-holiday: 1754.97)
- **გაუმჯობესება baseline-თან შედარებით:** 46.49
- N-BEATS არის ერთადერთი DL მოდელი, რომელმაც სცადა baseline-ის სჯობნა — მისი backcast/forecast residual stacking architecture უფრო მდგრადია, ვიდრე DLinear-ის მარტივი linear decomposition.

### 5.8 PatchTST

- **Validation WMAE:** 1730.48 (holiday: 1825.87, non-holiday: 1704.83)
- **დატრენინგებულია 2,873 სერიაზე** (458 სერია გამორიცხულია მოკლე ისტორიის გამო).
- საუკეთესო შედეგია ყველა DL მოდელს შორის, თუმცა მაინც არ სჯობს tree-based მოდელებს.

### 5.9 ARIMA (სრული dataset)

- **მოცულობა:** სრული dataset, 3,179 სერია (152 გამოტოვებული მოკლე ისტორიის გამო).
- **Feature selection:** სუფთა univariate — ARIMA-ს არ აქვს ჩაშენებული მექანიზმი გარეშე regressor-ებისთვის.
- **Order:** ფიქსირებული (1,1,1), **არ ჩატარებულა** per-series search — დავალების ინსტრუქციის მიხედვით ("ARIMA/SARIMA ძველ მოდელებზე ბევრი დროის დახარჯვა არ არის საჭირო"), search საათობით გახანგრძლივებდა ტრენინგს 3,331 სერიაზე.
- **WMAE:** 2566.45 (ამ notebook-ის საკუთარ baseline-თან შედარებით — 4030.74 — რომელიც განსხვავებულია დანარჩენი notebook-ების 1789.91-საგან, რადგან გამოთვლილია ცალკე, სრული ARIMA-ს population-ზე).
- **გაუმჯობესება საკუთარ baseline-თან:** 1464.29

### 5.10 SARIMA (15-სერიის სემფლი)

- **მოცულობა:** 15 შემთხვევით შერჩეული სერია (seed=42), 13 რეალურად გამოყენებული.
- **Order:** ფიქსირებული (1,1,1)(1,1,1,52) — სეზონური კომპონენტი დამატებულია პირდაპირ ARIMA-ს შეზღუდვის გამოსასწორებლად.
- **WMAE:** 1096.01 (საკუთარ sample-ის baseline-თან შედარებით — 2549.71)
- **გაუმჯობესება:** 1453.70

**თეორიული დასკვნა ARIMA vs SARIMA:** SARIMA-ს seasonal (P,D,Q,52) წევრები პირდაპირ მიმართავს ARIMA-ს მთავარ სისუსტეს ამ dataset-ზე — არ აქვს ჩაშენებული წლიური სეზონურობის ცოდნა. მიუხედავად იმისა, რომ ორივე ტესტირებულია განსხვავებულ population-ზე (ARIMA: სრული dataset, SARIMA: 15-სერიის sample), SARIMA-ს დიდი გაუმჯობესება საკუთარ baseline-თან შედარებით ადასტურებს მოსალოდნელ თეორიულ უპირატესობას.

---

## 6. მოდელების შედარება

**⚠️ მნიშვნელოვანი შენიშვნა:** ცხრილში მოცემული baseline-ის მნიშვნელობები განსხვავდება notebook-ების მიხედვით, რადგან zsh მოდელი ტესტირებულია სხვადასხვა population-ზე (სრული dataset vs sample). **პირდაპირ შედარებადია** მხოლოდ ისინი, ვისაც იდენტური baseline (1789.91) აქვს — ესენია LightGBM, XGBoost, Ensemble, Prophet, DLinear, N-BEATS, PatchTST.

| მოდელი | Population | Baseline WMAE | მოდელის WMAE | გაუმჯობესება |
|---|---|---|---|---|
| **Ensemble (XGB+LGB)** | სრული (3,331) | 1789.91 | **1354.77** | **435.14** |
| XGBoost | სრული (3,331) | 1789.91 | 1361.86 | 428.05 |
| LightGBM | სრული (3,331) | 1789.91 | 1384.82 | 405.10 |
| PatchTST | 2,873 სერია | 1789.91 | 1730.48 | 59.43 |
| N-BEATS | სრული (val split) | 1804.62 | 1758.13 | 46.49 |
| DLinear | 2,940 სერია | 1789.91 | 1814.32 | -24.41 |
| Prophet | 3,167 სერია | 1789.91 | 1905.42 | -115.51 |
| SARIMA | 15-სერიის sample | 2549.71 | 1096.01 | 1453.70 |
| ARIMA | სრული (3,179) | 4030.74 | 2566.45 | 1464.29 |

**საუკეთესო მოდელი (სრულ, პირდაპირ შედარებად population-ზე): Ensemble (XGBoost + LightGBM), WMAE = 1354.77**

---

## 7. Model Registry

მხოლოდ საუკეთესო არქიტექტურაა რეგისტრირებული DagsHub Model Registry-ში, `WalmartSalesForecast` სახელით — **Ensemble (XGBoost 0.7 + LightGBM 0.3)**, დავალების მოთხოვნის შესაბამისად ("ამ მოდელებისაგან საუკეთესო შედეგის მქონე კი უნდა იყოს დარეგისტრირებული Model Registry-ში"). ყველა დანარჩენი მოდელის pipeline ასევე დალოგილია MLflow-ში (`{Model}_Final_Model` run-ის ქვეშ), მაგრამ `REGISTER_AS_GLOBAL_BEST = False`.

---

## 8. Kaggle Submission

`model_inference.ipynb`-ში ჩატვირთულია Model Registry-დან Ensemble pipeline, გაშვებულია დაუმუშავებელ (raw) `test.csv`-ზე, გენერირებულია submission.

**Kaggle-ზე მიღებული შედეგი:**
- Submission: `submission_final.csv`
- Public/Private score: **2931.56** (საუკეთესო submission), **3079.13** (ალტერნატიული submission)

**შენიშვნა:** Kaggle-ის რეალური test score (2931-3079) მაღალია ჩვენი internal validation WMAE-ზე (1354.77). ეს მოსალოდნელი სხვაობაა — internal validation არის train-ის შიდა holdout, ხოლო Kaggle-ის test set არის რეალურად future, unseen მონაცემები, სადაც ზოგიერთი feature (განსაკუთრებით market-ის მდგომარეობასთან დაკავშირებული) ნაკლებად საიმედოა.

---

## 9. დასკვნები

1. **Tree-based მოდელები (XGBoost, LightGBM) და მათი ensemble** აჩვენეს საუკეთესო შედეგი — ამის მთავარი მიზეზი მდიდარი, ხელით აგებული feature ნაკრებია (lag-ები, rolling stats, holiday კატეგორიები), რასაც tree-ები პირდაპირ იყენებენ.
2. **Deep Learning მოდელებმა (DLinear, N-BEATS, PatchTST) ვერ სჯობეს tree-based მოდელებს** — მოსალოდნელი შედეგია მცირე dataset-ის ზომის გამო (~143 კვირა თითო სერიაზე), რაც არასაკმარისია DL მოდელისთვის, რომ საკუთარი თავით ისწავლოს ტემპორალური სტრუქტურა hand-crafted feature-ების გარეშე.
3. **Prophet, per-series local მიდგომით, დაზარალდა high-variance სერიების გამო** — per-pair WMAE-ის distribution-ის ფართო diapason (4-დან 68,000-მდე) აჩვენებს, რომ ცალკეული, დამოუკიდებელი მოდელები არ სარგებლობენ tree-based/global მოდელების ერთობლივი სწავლის უპირატესობით.
4. **ARIMA/SARIMA-მ**, მიუხედავად მინიმალური tuning-ისა (fixed order, no search), აჩვენა მნიშვნელოვანი გაუმჯობესება საკუთარ baseline-თან შედარებით — ეს ადასტურებს, რომ სეზონურობის ცოდნის ჩაშენება (SARIMA-ს seasonal term) მნიშვნელოვან უპირატესობას იძლევა plain ARIMA-სთან შედარებით.
5. **Feature availability bug**-ის აღმოჩენა (lag_1-26-ის მიუწვდომლობა რეალურ test-ზე) იყო კრიტიკული გაკვეთილი: internal validation-ის მაღალი სიზუსტე ყოველთვის არ ნიშნავს production-ზეც იგივე შედეგს — საჭიროა feature-ების ხელმისაწვდომობის გულდასმით შემოწმება inference დროს.

---

## 10. Repo სტრუქტურა

```
├── feature_engineering.ipynb
├── model_experiment_LightGBM.ipynb
├── model_experiment_XGBoost.ipynb
├── model_experiment_Ensemble.ipynb
├── model_experiment_Prophet.ipynb
├── model_experiment_DLinear.ipynb
├── model_experiment_NBEATS.ipynb
├── model_experiment_PatchTST.ipynb
├── model_experiment_Arima.ipynb
├── model_experiment_Sarima.ipynb
├── model_inference.ipynb
├── data/
├── models/
└── README.md
```

## 11. MLflow ექსპერიმენტების სტრუქტურა

ყველა არქიტექტურას აქვს ცალკე MLflow experiment (`{Model}_Training`), შიგნით კი შემდეგი run-ები:

- `{Model}_Cleaning` — preprocessing სტატისტიკა და parameters
- `{Model}_Feature_Selection` — feature selection სტრატეგია და დასაბუთება (მოდელის ტიპის მიხედვით განსხვავებული — tree-ებისთვის importance-based prune, statistical/DL მოდელებისთვის architecture-constrained univariate)
- `{Model}_Validation` — ქრონოლოგიური holdout WMAE, holiday/non-holiday breakdown
- `{Model}_Final_Model` — საბოლოო raw-input pipeline, დალოგილი როგორც MLflow Model

N-BEATS-ს დამატებით აქვს `{Model}_CV_{config}` run-ები (cross-validation სხვადასხვა hyperparameter კონფიგურაციისთვის) და `{Model}_Cross_Validation_Summary`.
