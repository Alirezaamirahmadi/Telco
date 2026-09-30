import logging
from pathlib import Path

import pandas as pd

from .config import DATA_PATH, TARGET_COLUMN, ID_COLUMN


# ساخت Logger برای ثبت مراحل مهم اجرای برنامه
logger = logging.getLogger(__name__)


# ستون‌هایی که وجود آن‌ها برای اجرای پروژه ضروری است
REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def load_data(path: Path = DATA_PATH):
    """بارگذاری Dataset از مسیر مشخص‌شده."""

    # بررسی وجود فایل Dataset قبل از خواندن
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    logger.info("Loading dataset from: %s", path)

    # خواندن Dataset با Pandas
    df = pd.read_csv(path)

    logger.info(
        "Dataset loaded successfully: %d rows, %d columns",
        df.shape[0],
        df.shape[1],
    )

    return df


def validate_data(df):
    """بررسی وجود ستون‌های موردنیاز و معتبر بودن Dataset."""

    # پیدا کردن ستون‌هایی که در Dataset وجود ندارند
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    # اگر ستونی وجود نداشته باشد، اجرای برنامه متوقف می‌شود
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # جلوگیری از اجرای Pipeline روی Dataset خالی
    if df.empty:
        raise ValueError("Dataset is empty.")

    logger.info("Dataset validation passed.")

    return df


def clean_data(df):
    """انجام Cleaning اولیه و قطعی روی Dataset."""

    # برای جلوگیری از تغییر مستقیم داده اصلی، یک کپی می‌سازیم
    df = df.copy()

    # تبدیل TotalCharges از object به مقدار عددی
    # مقادیر خالی یا نامعتبر به NaN تبدیل می‌شوند
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    # تبدیل Target از Yes/No به 1/0
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({
        "No": 0,
        "Yes": 1,
    })

    # بررسی وجود مقدار نامعتبر در Target
    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Target column contains invalid values."
        )

    # حذف CustomerID چون فقط شناسه مشتری است
    # و نباید به عنوان Feature وارد مدل شود
    df = df.drop(columns=[ID_COLUMN])

    logger.info("Data cleaning completed.")

    return df