from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE
from .config import TARGET_COLUMN


def get_feature_types(X):
    """Identifying numerical and categorical features."""

    # پیدا کردن ستون‌های عددی
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # پیدا کردن ستون‌های دسته‌ای
    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    return numeric_features, categorical_features


def create_preprocessor(X):
    """Building the preprocessing pipeline."""

    # تعیین نوع ویژگی‌ها
    numeric_features, categorical_features = get_feature_types(X)

    # Pipeline مربوط به ویژگی‌های عددی
    numeric_pipeline = Pipeline([
        # پر کردن مقادیر گمشده با Median
        ("imputer", SimpleImputer(strategy="median")),

        # استانداردسازی ویژگی‌های عددی
        ("scaler", StandardScaler()),
    ])

    # Pipeline مربوط به ویژگی‌های دسته‌ای
    categorical_pipeline = Pipeline([
        # پر کردن مقادیر گمشده با پرتکرارترین مقدار
        ("imputer", SimpleImputer(strategy="most_frequent")),

        # تبدیل ویژگی‌های دسته‌ای به One-Hot Encoding
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )),
    ])

    # ترکیب Pipelineهای عددی و دسته‌ای
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return preprocessor


def split_data(df):
    """Splitting the dataset into train, validation, and test sets."""

    # جدا کردن Featureها از Target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # جدا کردن Test Set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # تقسیم باقی داده به Train و Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )