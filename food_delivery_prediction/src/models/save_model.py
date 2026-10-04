import os
import joblib


def save_model(model, path="models/model.pkl"):

    # Create directory if it does not exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save model
    joblib.dump(model, path)

    print(f"Model saved successfully at: {path}")