import json
from datetime import datetime, timezone
from pathlib import Path

# ساخت یک شناسه یکتا برای هر اجرای آموزش مدل
def create_run_id():
    """Create a unique identifier for a training run."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


# ثبت اطلاعات مربوط به هر اجرای آموزش مدل
def log_training_run(
    tracking_path: Path,
    model_type,
    hyperparameters,
    train_size,
    validation_size,
    test_size,
    metrics,
    model_version,
):
    """Append a training run to the experiment tracking file."""

    # اطمینان از وجود پوشه خروجی
    tracking_path.parent.mkdir(parents=True, exist_ok=True)


    # اگر فایل Tracking وجود داشته باشد،
    # اطلاعات Runهای قبلی را می‌خوانیم
    if tracking_path.exists():
        with tracking_path.open("r", encoding="utf-8") as file:
            runs = json.load(file)
    else:
        # در اولین اجرا یک لیست خالی ایجاد می‌کنیم
        runs = []

    # اطلاعات Training Run جدید
    run = {
        "run_id": create_run_id(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "train_size": train_size,
        "validation_size": validation_size,
        "test_size": test_size,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "model_version": model_version,
    }

    # اضافه کردن Training Run جدید به Runهای قبلی
    runs.append(run)

    # ذخیره تمام Training Runها در فایل JSON
    with tracking_path.open("w", encoding="utf-8") as file:
        json.dump(runs, file, indent=4)

    # برگرداندن اطلاعات Run ثبت‌شد
    return run