import json
import logging

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .config import (
    MODEL_PATH,
    MODEL_PARAMS,
    OUTPUTS_DIR,
    TARGET_COLUMN,
)
from .data import load_data, validate_data, clean_data
from .preprocessing import create_preprocessor, split_data


# تنظیمات اولیه Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def evaluate_model(model, X, y):
    """Calculation of model evaluation metrics."""

    # پیش‌بینی کلاس‌ها
    predictions = model.predict(X)

    # احتمال تعلق نمونه به کلاس Churn
    probabilities = model.predict_proba(X)[:, 1]

    # محاسبه معیارهای ارزیابی
    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),
    }

    return metrics


def main():
    """Full execution of the model training process."""

    logger.info("Training process started.")

    # ایجاد پوشه‌های موردنیاز در صورت نبودن آن‌ها
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------
    # 1. Load Data
    # -------------------------

    logger.info("Loading dataset.")

    df = load_data()

    logger.info(
        "Raw dataset shape: %s",
        df.shape,
    )

    # -------------------------
    # 2. Validate Data
    # -------------------------

    logger.info("Validating dataset.")

    df = validate_data(df)

    # -------------------------
    # 3. Clean Data
    # -------------------------

    logger.info("Cleaning dataset.")

    df = clean_data(df)

    # -------------------------
    # 4. Split Data
    # -------------------------

    # تقسیم داده فقط از طریق تابع مرکزی split_data انجام می‌شود
    # بنابراین تنظیمات Split فقط در config.py قرار دارند
    logger.info("Splitting dataset.")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(df)

    logger.info(
        "Train size: %d",
        len(X_train),
    )

    logger.info(
        "Validation size: %d",
        len(X_val),
    )

    logger.info(
        "Test size: %d",
        len(X_test),
    )

    # -------------------------
    # 5. Create Preprocessing
    # -------------------------

    logger.info("Creating preprocessing pipeline.")

    # Preprocessing فقط با Train ساخته می‌شود
    # تا اطلاعات Validation و Test وارد فرآیند Fit نشوند
    preprocessor = create_preprocessor(X_train)

    # -------------------------
    # 6. Create Model
    # -------------------------

    logger.info("Creating Logistic Regression model.")

    model = LogisticRegression(
        **MODEL_PARAMS
    )

    # -------------------------
    # 7. Create Full Pipeline
    # -------------------------

    # ترکیب Preprocessing و Model در یک Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    # -------------------------
    # 8. Train Model
    # -------------------------

    logger.info("Model training started.")

    pipeline.fit(
        X_train,
        y_train,
    )

    logger.info("Model training completed.")

    # -------------------------
    # 9. Validation Evaluation
    # -------------------------

    logger.info("Evaluating model on Validation Set.")

    validation_metrics = evaluate_model(
        pipeline,
        X_val,
        y_val,
    )
    logger.info(
        "Validation metrics: %s",
        validation_metrics,
    )

    # -------------------------
    # 10. Final Test Evaluation
    # -------------------------

    # Test Set فقط برای ارزیابی نهایی استفاده می‌شود
    # و در Training یا Model Selection دخالتی ندارد
    logger.info("Evaluating model on Test Set.")

    test_metrics = evaluate_model(
        pipeline,
        X_test,
        y_test,
    )

    logger.info(
        "Test metrics: %s",
        test_metrics,
    )

    # -------------------------
    # 11. Save Full Pipeline
    # -------------------------

    # کل Pipeline ذخیره می‌شود:
    # Preprocessing + Model
    joblib.dump(
        pipeline,
        MODEL_PATH,
    )

    logger.info(
        "Model pipeline saved to: %s",
        MODEL_PATH,
    )

    # -------------------------
    # 12. Save Metrics
    # -------------------------

    metrics = {
        "model": "Logistic Regression",
        "validation": validation_metrics,
        "test": test_metrics,
    }

    metrics_path = OUTPUTS_DIR / "metrics.json"

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    logger.info(
        "Metrics saved to: %s",
        metrics_path,
    )

    logger.info(
        "Training process completed successfully."
    )


if __name__ == "__main__":
    main()