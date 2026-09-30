import logging

import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from .config import MODEL_PATH
from .data import load_data, validate_data, clean_data
from .preprocessing import split_data


# تنظیمات Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """ارزیابی مدل ذخیره‌شده روی Test Set."""

    logger.info("Evaluation process started.")

    # بررسی وجود مدل ذخیره‌شده
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    # -------------------------
    # 1. Load Data
    # -------------------------

    df = load_data()

    # -------------------------
    # 2. Validate Data
    # -------------------------

    df = validate_data(df)

    # -------------------------
    # 3. Clean Data
    # -------------------------

    df = clean_data(df)

    # -------------------------
    # 4. Split Data
    # -------------------------

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(df)

    logger.info(
        "Test set contains %d records.",
        len(X_test),
    )

    # -------------------------
    # 5. Load Saved Pipeline
    # -------------------------

    logger.info(
        "Loading model from: %s",
        MODEL_PATH,
    )

    pipeline = joblib.load(MODEL_PATH)

    # -------------------------
    # 6. Predict on Test Set
    # -------------------------

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # -------------------------
    # 7. Calculate Metrics
    # -------------------------

    metrics = {
        "Accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    # -------------------------
    # 8. Report Results
    # -------------------------

    logger.info("Test metrics:")

    for metric_name, metric_value in metrics.items():
        logger.info(
            "%s: %.4f",
            metric_name,
            metric_value,
        )

    logger.info("Evaluation process completed.")

    return metrics


if __name__ == "__main__":
    main()