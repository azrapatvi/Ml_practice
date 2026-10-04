import os
import joblib

from src.models.predict import predict_delivery


def test_model_exists():

    model_path = "models/model.pkl"

    assert os.path.exists(model_path)


def test_model_can_be_loaded():

    model = joblib.load("models/model.pkl")

    assert model is not None


def test_prediction():

    model = joblib.load("models/model.pkl")

    input_data = {
        "delivery_person_age": 30,
        "delivery_person_ratings": 4.7,
        "weatherconditions": "Sunny",
        "road_traffic_density": "High",
        "vehicle_condition": 2,
        "type_of_order": "Meal",
        "type_of_vehicle": "motorcycle",
        "multiple_deliveries": 1,
        "festival": 0,
        "city": "Metropolitian",
        "time_orderd_hr": 12,
        "time_orderd_mins": 30,
        "time_order_picked_hr": 12,
        "time_order_picked_mins": 40,
        "orderd_day_of_week": 2,
        "orderd_month": 8,
        "orderd_is_weekend": 0,
        "distance_km": 5.2
    }

    prediction = predict_delivery(model, input_data)

    assert prediction is not None
    assert isinstance(prediction, float)
    assert prediction > 0