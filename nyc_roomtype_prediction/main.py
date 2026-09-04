from fastapi import FastAPI,Form
import pandas as pd
import joblib

best_pipeline=joblib.load("best_pipeline.pkl")

app=FastAPI()


@app.post("/predict")
def predict(
    neighbourhood_group:str=Form(...),
    neighbourhood: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    price: float = Form(...),
    minimum_nights: float = Form(...),
    number_of_reviews: float = Form(...),
    reviews_per_month: float = Form(...),
    calculated_host_listings_count: float = Form(...),
    availability_365: float = Form(...)
):
    new_df=pd.DataFrame([{
        "neighbourhood_group":neighbourhood_group,
        "neighbourhood":neighbourhood,
        "latitude":latitude,
        "longitude":longitude,
        "price":price,
        "minimum_nights":minimum_nights,
        "number_of_reviews":number_of_reviews,
        "reviews_per_month":reviews_per_month,
        "calculated_host_listings_count":calculated_host_listings_count,
        "availability_365":availability_365
    }])

    prediction=best_pipeline.predict(new_df)
    prediction_probability=best_pipeline.predict_proba(new_df)

    return {
        "predicted_room_type": prediction[0],
        "predicted_probability":prediction_probability[0].tolist()
    }

from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="static", html=True), name="static")
