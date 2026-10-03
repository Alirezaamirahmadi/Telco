import logging
from typing import Literal

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .config import MODEL_PATH, MODEL_VERSION


# تنظیمات Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# -------------------------
# FastAPI Application
# -------------------------

app = FastAPI(
    title="Telco Customer Churn API",
    version=MODEL_VERSION,
)


# -------------------------
# Model Loading
# -------------------------

model = None
model_loaded = False


def load_model():
    """Loading the model artifact upon API startup."""

    global model
    global model_loaded

    logger.info(
        "Loading model from: %s",
        MODEL_PATH,
    )

    if not MODEL_PATH.exists():
        logger.error(
            "Model artifact not found: %s",
            MODEL_PATH,
        )
        model_loaded = False
        return

    try:
        model = joblib.load(MODEL_PATH)
        model_loaded = True

        logger.info(
            "Model %s loaded successfully.",
            MODEL_VERSION,
        )

    except Exception:
        model = None
        model_loaded = False

        logger.exception(
            "Failed to load model."
        )


# بارگذاری مدل هنگام import شدن API
load_model()


# -------------------------
# Input Schema
# -------------------------

class PredictionRequest(BaseModel):
    """Prediction API input structure."""

    model_config = ConfigDict(
        extra="forbid"
    )

    gender: Literal["Male", "Female"]

    SeniorCitizen: int = Field(
        ge=0,
        le=1,
    )

    Partner: Literal["Yes", "No"]

    Dependents: Literal["Yes", "No"]

    tenure: int = Field(
        ge=0,
    )

    PhoneService: Literal["Yes", "No"]

    MultipleLines: Literal[
        "Yes",
        "No",
        "No phone service",
    ]

    InternetService: Literal[
        "DSL",
        "Fiber optic",
        "No",
    ]

    OnlineSecurity: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    OnlineBackup: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    DeviceProtection: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    TechSupport: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    StreamingTV: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    StreamingMovies: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year",
    ]

    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    MonthlyCharges: float = Field(
        ge=0,
    )

    TotalCharges: float = Field(
        ge=0,
    )


# -------------------------
# Health Endpoint
# -------------------------

@app.get("/health")
def health():
    """Checking the status of the API and Model."""

    logger.info(
        "Health check requested."
    )

    return {
        "status": "ok" if model_loaded else "error",
        "model_loaded": model_loaded,
        "model_version": MODEL_VERSION,
    }


# -------------------------
# Prediction Endpoint
# -------------------------

@app.post("/predict")
def predict(request: PredictionRequest):
    """Performing a prediction for a customer."""

    logger.info(
        "Prediction request received."
    )

    # اگر مدل Load نشده باشد،
    # API نباید Prediction انجام دهد.
    if not model_loaded or model is None:
        logger.error(
            "Prediction rejected because model is not loaded."
        )

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    try:
        # تبدیل Request به DataFrame
        input_data = pd.DataFrame(
            [request.model_dump()]
        )

        # انجام Prediction
        prediction = int(
            model.predict(input_data)[0]
        )

        # محاسبه Probability مربوط به Churn = Yes
        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        logger.info(
            "Prediction completed successfully."
        )

        return {
            "prediction": prediction,
            "probability": probability,
            "model_version": MODEL_VERSION,
        }

    except Exception:
        logger.exception(
            "Prediction failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        )