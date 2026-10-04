from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.models.predict import load_model, predict_delivery


app = FastAPI(
    title="Food Delivery Prediction API",
    version="1.0.0"
)


templates = Jinja2Templates(directory="app/templates")


# Load trained model
model = load_model("models/model.pkl")


class DeliveryInput(BaseModel):

    delivery_person_age: float
    delivery_person_ratings: float

    weatherconditions: str
    road_traffic_density: str

    vehicle_condition: int

    type_of_order: str
    type_of_vehicle: str

    multiple_deliveries: int
    festival: int

    city: str

    time_orderd_hr: int
    time_orderd_mins: int

    time_order_picked_hr: int
    time_order_picked_mins: int

    orderd_day_of_week: int
    orderd_month: int
    orderd_is_weekend: int

    distance_km: float


@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/predict")
def predict(data: DeliveryInput):

    input_data = data.model_dump()

    prediction = predict_delivery(
        model,
        input_data
    )

    return {
        "predicted_delivery_time": round(prediction, 2),
        "unit": "minutes"
    }
