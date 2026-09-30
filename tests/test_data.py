from src.config import DATA_PATH, TARGET_COLUMN
from src.data import load_data, validate_data, clean_data


def test_required_columns():
    """Checking for the presence of essential dataset columns."""

    # بارگذاری Dataset
    df = load_data(DATA_PATH)

    # بررسی معتبر بودن Dataset
    df = validate_data(df)

    # اطمینان از وجود Target
    assert TARGET_COLUMN in df.columns


def test_clean_data():
    """Review of data cleaning."""

    # بارگذاری و Cleaning داده
    df = load_data(DATA_PATH)
    df = clean_data(df)

    # Target باید فقط شامل 0 و 1 باشد
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})

    # customerID نباید بعد از Cleaning وجود داشته باشد
    assert "customerID" not in df.columns

    # TotalCharges باید عددی شده باشد
    assert df["TotalCharges"].dtype.kind in "fi"