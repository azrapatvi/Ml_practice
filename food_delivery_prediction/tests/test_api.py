from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home_page():

    response = client.get("/")

    assert response.status_code == 200


def test_prediction_api():

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

    response = client.post(
        "/predict",
        json=input_data
    )

    assert response.status_code == 200

    result = response.json()

    assert "predicted_delivery_time" in result
    assert "unit" in result

    assert result["predicted_delivery_time"] > 0