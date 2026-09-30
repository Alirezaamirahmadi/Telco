import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd

from .config import MODEL_PATH
from .data import REQUIRED_COLUMNS


# تنظیمات Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def load_model():
    """بارگذاری Pipeline ذخیره‌شده."""

    # بررسی وجود فایل مدل
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    logger.info(
        "Loading model from: %s",
        MODEL_PATH,
    )

    return joblib.load(MODEL_PATH)


def validate_input_data(data):
    """Comprehensive validation of input data for prediction."""

    # بررسی خالی نبودن DataFrame
    if data.empty:
        raise ValueError(
            "Input data is empty."
        )

    # ستون‌هایی که مدل برای Prediction نیاز دارد
    required_input_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in {"Churn", "customerID"}
    ]

    # بررسی وجود ستون‌های ضروری
    missing_columns = [
        column
        for column in required_input_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required input columns: {missing_columns}"
        )

    # customerID اختیاری است چون در مدل استفاده نمی‌شود
    allowed_columns = set(required_input_columns)
    allowed_columns.add("customerID")

    # بررسی وجود ستون‌های غیرمنتظره
    unexpected_columns = [
        column
        for column in data.columns
        if column not in allowed_columns
    ]

    if unexpected_columns:
        raise ValueError(
            f"Unexpected input columns: {unexpected_columns}"
        )

    # بررسی قابل تبدیل بودن TotalCharges به عدد
    total_charges = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce",
    )

    # مقدار خالی مجاز است و بعداً توسط Pipeline مدیریت می‌شود.
    # اما مقدار غیرخالی و غیرقابل تبدیل نامعتبر است.
    invalid_total_charges = (
        data["TotalCharges"].notna()
        & total_charges.isna()
    )

    if invalid_total_charges.any():
        raise ValueError(
            "TotalCharges contains non-numeric values."
        )

    logger.info(
        "Input data validation passed."
    )

    return data


def prepare_input_data(data):
    """Preparing input data before prediction."""

    # ساخت کپی برای جلوگیری از تغییر داده اصلی
    data = data.copy()

    # CustomerID در مدل استفاده نمی‌شود
    if "customerID" in data.columns:
        data = data.drop(
            columns=["customerID"]
        )

    # تبدیل TotalCharges به عدد
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce",
    )

    return data


def predict(input_path):
    """انجام Prediction روی داده جدید."""

    # بررسی وجود فایل ورودی
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    logger.info(
        "Loading input data from: %s",
        input_path,
    )

    # خواندن فایل CSV
    data = pd.read_csv(input_path)

    # اعتبارسنجی کامل داده ورودی
    data = validate_input_data(data)

    # آماده‌سازی داده برای مدل
    prepared_data = prepare_input_data(
        data
    )

    # بارگذاری Pipeline ذخیره‌شده
    model = load_model()

    # انجام Prediction
    predictions = model.predict(
        prepared_data
    )

    # تبدیل خروجی عددی به Yes / No
    prediction_labels = [
        "Yes" if prediction == 1 else "No"
        for prediction in predictions
    ]

    # ساخت خروجی
    result = data.copy()

    result["Churn_Prediction"] = (
        prediction_labels
    )

    logger.info(
        "Prediction completed for %d records.",
        len(result),
    )

    return result


def main():
    """Running Prediction via the command line."""

    parser = argparse.ArgumentParser(
        description="Predict customer churn."
    )
    # مسیر فایل CSV ورودی
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file.",
    )

    args = parser.parse_args()

    # تبدیل مسیر ورودی به Path
    input_path = Path(
        args.input
    )

    # انجام Prediction
    result = predict(
        input_path
    )

    # نمایش نتیجه
    print(result)


if __name__ == "__main__":
    main()