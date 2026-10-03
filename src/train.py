import json
import logging
from datetime import datetime, timezone

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
    DATA_PATH,
    METADATA_PATH,
    MODEL_PATH,
    MODEL_PARAMS,
    MODEL_TYPE,
    MODEL_VERSION,
    RANDOM_STATE,
    TRACKING_PATH,
)
from .data import load_data, validate_data, clean_data
from .preprocessing import split_data, create_preprocessor
from .tracking import log_training_run


# تنظیم Logging برای نمایش اطلاعات مهم Training
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def calculate_metrics(y_true, predictions, probabilities):
    """Calculating model evaluation metrics."""

    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }

    return metrics


def save_model_metadata(features, metrics):
    """Store metadata related to the model version.
    The metadata contains the information necessary to identify and
    reproduce the model artifact.
    """

    metadata = {
        "model_version": MODEL_VERSION,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "dataset_version": DATA_PATH.name,
        "model_type": MODEL_TYPE,
        "features": features,
        "metrics": metrics,
        "random_state": RANDOM_STATE,
        "hyperparameters": MODEL_PARAMS,
    }

    # اطمینان از وجود پوشه models
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ذخیره Metadata در فایل JSON
    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    logger.info("Model metadata saved to: %s", METADATA_PATH)


def train():
    """
Execution of the complete training process.

    Steps:
    1. Load Dataset
    2. Validate Dataset
    3. Clean Dataset
    4. Split Dataset
    5. Create Preprocessing Pipeline
    6. Train Model
    7. Evaluate Model
    8. Save Versioned Model
    9. Save Model Metadata
    10. Register Training Run
    """

    logger.info("Starting training process.")

    # ---------------------------------------------------------
    # 1. Load Dataset
    # ---------------------------------------------------------
    df = load_data(DATA_PATH)

    # ---------------------------------------------------------
    # 2. Validate Dataset
    # ---------------------------------------------------------
    df = validate_data(df)

    # ---------------------------------------------------------
    # 3. Clean Dataset
    # ---------------------------------------------------------
    df = clean_data(df)

    # ---------------------------------------------------------
    # 4. Split Dataset
    # ---------------------------------------------------------
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(df)

    logger.info(
        "Dataset split completed: train=%d, validation=%d, test=%d",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    # ---------------------------------------------------------
    # 5. Create Preprocessing Pipeline
    # ---------------------------------------------------------
    preprocessor = create_preprocessor(X_train)

    # ---------------------------------------------------------
    # 6. Create ML Pipeline
    # ---------------------------------------------------------
    model = LogisticRegression(**MODEL_PARAMS)

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    # ---------------------------------------------------------
    # 7. Train Model
    # ---------------------------------------------------------
    logger.info("Training %s model.", MODEL_TYPE)

    pipeline.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 8. Evaluate Model on Unseen Test Data
    # ---------------------------------------------------------
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(
        y_test,
        predictions,
        probabilities,
    )

    logger.info(
        "Test metrics - Accuracy: %.4f, Precision: %.4f, "
        "Recall: %.4f, F1: %.4f, ROC-AUC: %.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
    )

    # ---------------------------------------------------------
    # 9. Save Versioned Model Artifact
    # ---------------------------------------------------------
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)

    logger.info(
        "Model version %s saved to: %s",
        MODEL_VERSION,
        MODEL_PATH,
    )

    # ---------------------------------------------------------
    # 10. Save Model Metadata
    # ---------------------------------------------------------
    save_model_metadata(
        features=X_train.columns.tolist(),
        metrics=metrics,
    )

    # ---------------------------------------------------------
    # 11. Register Training Run
    # ---------------------------------------------------------
    run = log_training_run(
        tracking_path=TRACKING_PATH,
        model_type=MODEL_TYPE,
        hyperparameters=MODEL_PARAMS,
        train_size=len(X_train),
        validation_size=len(X_val),
        test_size=len(X_test),
        metrics=metrics,
        model_version=MODEL_VERSION,
    )

    logger.info(
        "Training run registered successfully. Run ID: %s",
        run["run_id"],
    )

    logger.info("Training process completed successfully.")

    return pipeline, metrics, run


if __name__ == "__main__":
    train()