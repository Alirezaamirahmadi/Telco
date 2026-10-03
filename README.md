# Telco

## 1. Project Overview

این پروژه، مرحله‌ی **From ML Model → ML System** است. برخلاف پروژه‌ی قبلی (Bank Marketing) که کل کار داخل چند Notebook انجام می‌شد، در این پروژه هدف این است که یک مدل Machine Learning به یک **سیستم قابل اجرا، قابل تکرار، تست‌پذیر و قابل نگهداری** تبدیل شود.

**چیزهایی که نسبت به مرحله‌ی قبل تازه به کار گرفته شده‌اند:**
- جدا کردن کد اصلی از Notebook و انتقال آن به ماژول‌های مستقل در `src/`
- متمرکز کردن تنظیمات پروژه (مسیرها، Random Seed، اندازه‌ی Split و پارامترهای مدل) در `src/config.py`
- ساخت اسکریپت‌های مستقل برای Train، Evaluate و Predict، بدون نیاز به باز کردن Notebook
- ذخیره‌ی کامل Inference Pipeline (Preprocessing + Model) به‌جای ذخیره‌ی تنها مدل
- نوشتن Unit Test با `pytest` برای Data، Preprocessing و Prediction
- استفاده از `logging` برای ثبت مراحل اصلی اجرا
- تعریف Validation مشخص برای داده‌ی ورودی Prediction و خطاهای قابل پیش‌بینی

Notebook فقط برای **EDA، مقایسه‌ی مدل‌ها و انتخاب مدل نهایی** استفاده می‌شود و اجرای اصلی سیستم به Notebook وابسته نیست.

## 2. Problem Definition

- **Problem:** پیش‌بینی اینکه آیا یک مشتری شرکت مخابراتی، سرویس خود را قطع می‌کند (Churn) یا نه.
- **Target:** ستون `Churn` (مقادیر `Yes` / `No` که در کد به `1` / `0` تبدیل می‌شود).
- **نوع مسئله:** Binary Classification.
- **واحد هر Sample:** یک مشتری، با اطلاعات قرارداد، سرویس‌ها و صورت‌حساب او.

## 3. Dataset

- **نام:** Telco Customer Churn (فایل `WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **منبع:** Kaggle – Telco Customer Churn
- **حجم:** 7,043 رکورد و 21 ستون شامل `customerID`، 19 ویژگی مورد استفاده برای مدل و ستون هدف `Churn`
- **ترکیب Featureها:** عددی (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`) و Categorical (نوع قرارداد، نوع اینترنت، روش پرداخت و غیره)
- **نکته‌ی کیفیت داده:** ستون `TotalCharges` در فایل خام به‌صورت رشته ذخیره شده و 11 مقدار خالی/نامعتبر دارد؛ در `src/data.py` با `pd.to_numeric(..., errors="coerce")` به عدد تبدیل می‌شود و مقادیر نامعتبر به `NaN` تبدیل می‌شوند. سپس Imputer داخل Pipeline این مقادیر را مدیریت می‌کند.
- ستون `customerID` فقط شناسه است و قبل از مدل‌سازی حذف می‌شود.
- **Class Distribution:** 5,174 مشتری با `Churn=No` و 1,869 مشتری با `Churn=Yes` (تقریباً 73.5٪ در برابر 26.5٪).

## 4. Project Structure

```text
Telco/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── prediction_input.csv
├── notebooks/
│   └── exploration.ipynb
├── models/
│   ├── telco_churn_v1.joblib       # Versioned model artifact
│   └── metadata_v1.json            # Model metadata and metrics
├── outputs/
│   └── training_runs.json          # Experiment tracking
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── tracking.py
│   └── api.py                      # FastAPI application
├── tests/
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_prediction.py
│   └── test_api.py
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

مسیرها در `src/config.py` با `pathlib.Path` نسبت به ریشه‌ی پروژه ساخته می‌شوند؛ بنابراین کد به مسیر خاص کامپیوتر وابسته نیست.

## 5. Installation

```bash
git clone <repository-url>
cd Telco

python -m venv venv

# Windows PowerShell
venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

## 6. Configuration

تمام تنظیمات مهم پروژه در `src/config.py` متمرکز شده‌اند:

| متغیر | مقدار | توضیح |
|---|---:|---|
| `RANDOM_STATE` | `42` | Seed برای تکرارپذیری |
| `TEST_SIZE` | `0.20` | سهم Test از کل داده |
| `VALIDATION_SIZE` | `0.25` | سهم Validation از داده‌ی باقی‌مانده؛ نتیجه حدوداً Train=60٪، Validation=20٪، Test=20٪ |
| `DATA_PATH` | `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` | مسیر دیتاست |
| `MODEL_PATH` | `models/telco_churn_v1.joblib` | مسیر Model Artifact |
| `TARGET_COLUMN` | `Churn` | ستون هدف |
| `ID_COLUMN` | `customerID` | شناسه‌ای که وارد مدل نمی‌شود |
| `MODEL_PARAMS` | `max_iter=1000`, `random_state=42` | پارامترهای Logistic Regression |
| `MODEL_VERSION` | `v1` | نسخه مدل |
| `METADATA_PATH` | `models/metadata_v1.json` | Metadata مدل |
| `TRACKING_PATH` | `outputs/training_runs.json` | Experiment Tracking | |

منطق Split فقط در `src/preprocessing.py` و داخل تابع `split_data()` قرار دارد و تمام مقادیر Split از `config.py` خوانده می‌شوند.

## 7. Training

```bash
python -m src.train
```

این دستور:
1. Dataset را Load و Validate می‌کند.
2. `TotalCharges` را عددی و `Churn` را به 0/1 تبدیل می‌کند.
3. `customerID` را حذف می‌کند.
4. داده را با `stratify` به Train، Validation و Test تقسیم می‌کند.
5. Preprocessing Pipeline را می‌سازد:
   - Numeric: Median Imputation + StandardScaler
   - Categorical: Most-Frequent Imputation + OneHotEncoder
6. مدل `LogisticRegression` را روی Train آموزش می‌دهد.
7. کل Pipeline شامل Preprocessing + Model را با `joblib` ذخیره می‌کند.
8. Metricهای Test را در Metadata مدل و Experiment Tracking ثبت می‌کند.

اجرای Training به Notebook وابسته نیست.

## 8. Evaluation

```bash
python -m src.evaluate
```

این اسکریپت مستقل از Training اجرا می‌شود:
- Pipeline نسخه‌بندی‌شده را با `joblib.load` بارگذاری می‌کند.
- Dataset را Load، Validate و Clean می‌کند.
- همان Train/Validation/Test Split را با همان `RANDOM_STATE` بازسازی می‌کند.
- Metricهای Test را روی داده‌ی Test محاسبه می‌کند.

Metricهای گزارش‌شده شامل Accuracy، Precision، Recall، F1 و ROC-AUC هستند.

## 9. Prediction

```bash
python -m src.predict --input data/prediction_input.csv
```

`src/predict.py` برای Prediction روی داده‌ی جدید طراحی شده و مراحل زیر را انجام می‌دهد:

1. بررسی می‌کند فایل ورودی وجود داشته باشد.
2. بررسی می‌کند فایل خالی نباشد.
3. وجود تمام ستون‌های مورد نیاز Featureها را بررسی می‌کند.
4. وجود ستون‌های غیرمنتظره را بررسی می‌کند.
5. بررسی می‌کند مقادیر `TotalCharges` قابل تبدیل به عدد باشند.
6. در صورت وجود، `customerID` را حذف می‌کند.
7. Pipeline ذخیره‌شده را Load می‌کند.
8. Prediction را انجام می‌دهد و نتیجه را به `Yes` / `No` تبدیل می‌کند.

نمونه‌ی موجود در `data/prediction_input.csv` شامل 5 مشتری است و خروجی Notebook/اجرای Prediction برای آن:

| customerID | Churn_Prediction |
|---|---|
| 7590-VHVEG | Yes |
| 5575-GNVDE | No |
| 3668-QPYBK | No |
| 7795-CFOCW | No |
| 9237-HQITU | Yes |

## 10. Testing

```bash
pytest
```

سه Test اصلی وجود دارد:

| فایل | بررسی |
|---|---|
| `tests/test_data.py` | ستون‌های ضروری، Cleaning، تبدیل Target به 0/1، حذف `customerID` و عددی بودن `TotalCharges` |
| `tests/test_preprocessing.py` | اجرای درست Preprocessing روی Train/Validation و نبودن `NaN` در خروجی |
| `tests/test_prediction.py` | آموزش Pipeline روی Training Data، ذخیره/Load با `joblib` و Prediction روی داده‌ی دیده‌نشده |

در تست Prediction، Preprocessing فقط روی Training Data `fit` می‌شود و داده‌ی Test فقط برای `transform/predict` استفاده می‌شود.

## 11. Model

### Model Selection در Notebook

در `notebooks/exploration.ipynb` مدل‌ها با استفاده از F1 روی Validation مقایسه شدند:

| Model | Validation F1 |
|---|---:|
| Logistic Regression | 0.5908 |
| Tuned Random Forest | 0.5604 |
| Decision Tree | 0.5370 |
| Random Forest | 0.5239 |
| Dummy | 0.0000 |

بر اساس این مقایسه، **Logistic Regression** به‌عنوان مدل نهایی انتخاب شد و همین مدل در `src/train.py` استفاده می‌شود.

### Final Model

- **Algorithm:** Logistic Regression
- **Parameters:** `max_iter=1000`, `random_state=42`
- **Stored artifact:** یک `sklearn.pipeline.Pipeline` کامل شامل Preprocessing و Model
- **Format:** `joblib`
- **Path:** `models/telco_churn_v1.joblib`
- **Metadata:** `models/metadata_v1.json`

## 12. Metrics

اعداد زیر از خروجی نهایی Notebook و اجرای Pipeline پروژه به‌دست آمده‌اند:

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.8034 | 0.8027 |
| Precision | 0.6601 | 0.6509 |
| Recall | 0.5348 | 0.5535 |
| F1 | 0.5908 | 0.5983 |
| ROC-AUC | 0.8360 | 0.8428 |

**Final Test Result:**
- Accuracy: **0.8027**
- Precision: **0.6509**
- Recall: **0.5535**
- F1: **0.5983**
- ROC-AUC: **0.8428**

این Metricها مربوط به Test Set هستند که در زمان Training برای انتخاب مدل استفاده نشده است.

## 13. Data Leakage Prevention

- Split قبل از Fit کردن Preprocessing انجام می‌شود.
- Imputer، Scaler و Encoder فقط روی Training Data `fit` می‌شوند.
- Validation و Test فقط با `transform` پردازش می‌شوند.
- تبدیل `TotalCharges` به عدد یک تبدیل نوع داده است و از اطلاعات سایر Samples استفاده نمی‌کند.
- در Notebook، `GridSearchCV` برای Random Forest فقط روی Training Data و با Cross-Validation انجام شده است.
- Test Set برای ارزیابی نهایی مدل استفاده شده و در انتخاب مدل از آن استفاده نشده است.
- در `test_prediction.py` نیز Pipeline روی Training Data آموزش داده می‌شود و سپس روی داده‌ی دیده‌نشده Prediction انجام می‌دهد.

## 14. از Clone تا اولین Prediction

برای اجرای کامل پروژه بدون Notebook:

```bash
git clone <repository-url>
cd Telco

python -m venv venv

# Windows PowerShell
venv\Scripts\Activate.ps1

pip install -r requirements.txt

# اجرای تست‌ها
pytest

# آموزش مدل
python -m src.train

# ارزیابی مستقل
python -m src.evaluate

# اولین Prediction
python -m src.predict --input data/prediction_input.csv
```

هیچ‌کدام از مراحل بالا به Jupyter وابسته نیستند. `notebooks/exploration.ipynb` فقط برای EDA، تحلیل و انتخاب مدل استفاده می‌شود.

## 15. Model Versioning

مدل به‌صورت Versioned Artifact ذخیره می‌شود:

```text
models/
├── telco_churn_v1.joblib
└── metadata_v1.json
```

`metadata_v1.json` شامل نسخه مدل، تاریخ آموزش، نسخه دیتاست، نوع مدل، Featureها، Metricها، Random State و Hyperparameters است.

برای تغییر نسخه مدل، مقدار `MODEL_VERSION` در `src/config.py` تغییر داده می‌شود تا Artifact و Metadata جدید با نام نسخه‌ی جدید ذخیره شوند.

## 16. Experiment Tracking

هر اجرای:

```bash
python -m src.train
```

یک Run جدید در:

```text
outputs/training_runs.json
```

ثبت می‌کند. اطلاعات Run شامل `run_id`، زمان اجرا، نوع مدل، Hyperparameters، اندازه Train/Validation/Test، Metricها و Model Version است.

## 17. FastAPI

اجرای API:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Health

```http
GET /health
```

نمونه پاسخ:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "v1"
}
```

اگر Model Artifact قابل Load نباشد، `model_loaded` برابر `false` و `status` برابر `error` خواهد بود.

### Prediction

```http
POST /predict
Content-Type: application/json
```

نمونه Request:

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.0,
  "TotalCharges": 840.0
}
```

نمونه Response:

```json
{
  "prediction": 0,
  "probability": 0.3009071176019144,
  "model_version": "v1"
}
```

API با Pydantic ورودی را Validation می‌کند و Missing Field، Type اشتباه، مقدار Categorical نامعتبر، مقدار Numeric نامعتبر و Extra Field را رد می‌کند.

## 18. API Testing

تست‌های API در `tests/test_api.py` قرار دارند و بدون اجرای دستی Server با `FastAPI TestClient` اجرا می‌شوند:

```bash
pytest -v
```

موارد اصلی:
- `test_health`
- `test_valid_prediction`
- `test_invalid_input`
- `test_model_loaded`

## 19. Docker

ساخت Image:

```bash
docker build -t telco-churn-api .
```

اجرای Container:

```bash
docker run --rm -p 8000:8000 telco-churn-api
```

API سپس روی:

```text
http://127.0.0.1:8000
```

در دسترس خواهد بود.

Docker به مسیرهای لوکال ویندوز وابسته نیست و Model Artifact و کد لازم را داخل Image قرار می‌دهد.

## 20. Continuous Integration

فایل:

```text
.github/workflows/ci.yml
```

با هر `push` و `pull_request`:
1. Repository را Checkout می‌کند.
2. Python 3.14 را آماده می‌کند.
3. Dependencies را نصب می‌کند.
4. `pytest -v` را اجرا می‌کند.
5. در صورت موفقیت تست‌ها، Docker Image را Build می‌کند.

## 21. Limitations

- مدل نهایی Logistic Regression است و منطق مقایسه‌ی چند مدل در Notebook انجام شده؛ در `src/train.py` فقط مدل نهایی اجرا می‌شود.
- Recall مدل روی Test برابر **0.5535** است؛ بنابراین بخشی از مشتریان Churn در این مدل شناسایی نمی‌شوند.
- Dataset مربوط به یک مجموعه‌داده‌ی مشخص از مشتریان یک شرکت مخابراتی است و تعمیم مستقیم مدل به بازار یا شرکت دیگر نیازمند اعتبارسنجی و احتمالاً آموزش مجدد است.
- عملکرد مدل به کیفیت و ساختار ورودی وابسته است؛ `src/predict.py` برای جلوگیری از ورودی نامعتبر، وجود ستون‌های لازم، ساختار غیرمنتظره و مقدار غیرعددی در `TotalCharges` را بررسی می‌کند، اما تغییر اساسی در Schema دیتاست نیازمند به‌روزرسانی Pipeline است.
- مدل و Threshold فعلی با هدف استفاده‌ی پایه از F1 انتخاب شده‌اند و برای کاربرد عملی ممکن است نیاز به Cost-Sensitive Learning، Threshold Tuning یا Calibration وجود داشته باشد.

## References

- Dataset: Kaggle – Telco Customer Churn
- `pandas`: Data Loading and Manipulation
- `scikit-learn`: Preprocessing, Pipeline, Model Training and Evaluation
- `joblib`: Model Artifact Serialization
- `pytest`: Automated Testing
