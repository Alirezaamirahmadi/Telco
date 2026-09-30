import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import DATA_PATH, TARGET_COLUMN
from src.data import load_data, clean_data
from src.preprocessing import create_preprocessor, split_data


def test_saved_pipeline_prediction(tmp_path):
    """Evaluating storage, loading, and prediction on unseen data."""

    # -------------------------
    # 1. Load and Clean Data
    # -------------------------

    df = load_data(DATA_PATH)
    df = clean_data(df)

    # -------------------------
    # 2. Split Data
    # -------------------------

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(df)

    # -------------------------
    # 3. Create Preprocessing
    # -------------------------

    # Preprocessing فقط روی Training Data ساخته می‌شود
    preprocessor = create_preprocessor(X_train)

    # -------------------------
    # 4. Create Pipeline
    # -------------------------

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    # -------------------------
    # 5. Train on Training Data
    # -------------------------

    pipeline.fit(
        X_train,
        y_train,
    )

    # -------------------------
    # 6. Save Pipeline
    # -------------------------

    model_path = tmp_path / "test_model.joblib"

    joblib.dump(
        pipeline,
        model_path,
    )

    # -------------------------
    # 7. Load Pipeline
    # -------------------------

    loaded_pipeline = joblib.load(
        model_path
    )

    # -------------------------
    # 8. Predict on Unseen Data
    # -------------------------

    # Validation Data هنگام Training دیده نشده است
    predictions = loaded_pipeline.predict(
        X_val
    )

    # -------------------------
    # 9. Assertions
    # -------------------------

    # تعداد Predictionها باید با تعداد نمونه‌ها برابر باشد
    assert len(predictions) == len(X_val)

    # خروجی مدل باید فقط شامل 0 و 1 باشد
    assert set(predictions).issubset({0, 1})