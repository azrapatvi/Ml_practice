import joblib
import pandas as pd


def load_model(model_path):
    model = joblib.load(model_path)
    return model


def predict_delivery(model, data):

    data = pd.DataFrame([data])

    data = data[model.feature_names_in_]

    prediction = model.predict(data)

    return float(prediction[0])