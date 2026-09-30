import numpy as np

from src.config import DATA_PATH
from src.data import load_data, clean_data
from src.preprocessing import split_data, create_preprocessor


def test_preprocessing_pipeline():
    """Testing Preprocessing Pipeline."""

    # بارگذاری و Cleaning داده
    df = load_data(DATA_PATH)
    df = clean_data(df)

    # تقسیم داده به Train، Validation و Test
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(df)

    # ساخت Preprocessor فقط بر اساس داده‌های Train
    preprocessor = create_preprocessor(X_train)

    # Fit و Transform روی Train
    X_train_transformed = preprocessor.fit_transform(X_train)

    # فقط Transform روی Validation
    X_val_transformed = preprocessor.transform(X_val)

    # تعداد ردیف‌ها باید درست باشد
    assert X_train_transformed.shape[0] == len(X_train)
    assert X_val_transformed.shape[0] == len(X_val)

    # تعداد Featureها در Train و Validation باید یکسان باشد
    assert X_train_transformed.shape[1] == X_val_transformed.shape[1]

    # نباید مقدار NaN در خروجی وجود داشته باشد
    assert not np.isnan(X_train_transformed).any()
    assert not np.isnan(X_val_transformed).any()