from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def evaluate_model(model_pipeline, X_test, y_test):

    # Make predictions
    predictions = model_pipeline.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    # Print results
    print("----------------------------------------")
    print("MODEL EVALUATION")
    print("----------------------------------------")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    # Return metrics
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }