from fastapi.testclient import TestClient

from src.api import app, model_loaded


client = TestClient(app)


VALID_PAYLOAD = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
}


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()
    assert body["model_loaded"] is model_loaded
    assert body["model_version"] == "v1"
    assert body["status"] == ("ok" if model_loaded else "error")


def test_valid_prediction():
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"] == "v1"


def test_invalid_categorical_input():
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["InternetService"] = "InvalidService"

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_missing_field():
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload.pop("gender")

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_wrong_type():
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["tenure"] = "not-an-integer"

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_invalid_numeric_input():
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["MonthlyCharges"] = -1

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_empty_request():
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_extra_field():
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["unexpected_field"] = "value"

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_model_loaded():
    assert model_loaded is True
