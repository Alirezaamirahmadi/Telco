from pathlib import Path


# مسیر اصلی پروژه
BASE_DIR = Path(__file__).resolve().parent.parent


# مسیر دیتاست
DATA_PATH = (
    BASE_DIR
    / "data"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


# Model Versioning
MODEL_VERSION = "v1"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / f"telco_churn_{MODEL_VERSION}.joblib"
)

METADATA_PATH = (
    BASE_DIR
    / "models"
    / f"metadata_{MODEL_VERSION}.json"
)


# Experiment Tracking
TRACKING_PATH = (
    BASE_DIR
    / "outputs"
    / "training_runs.json"
)


# مسیر ذخیره خروجی‌ها
OUTPUTS_DIR = BASE_DIR / "outputs"


# تنظیمات داده
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25


# ستون‌های اصلی
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


# تنظیمات مدل نهایی
MODEL_TYPE = "LogisticRegression"

MODEL_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}