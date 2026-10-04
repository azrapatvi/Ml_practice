from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.pipeline import create_preprocessor
from src.evaluation.evaluate import evaluate_model


def train_model(df):

    X = df.drop('time_taken_mins', axis=1)
    y = df['time_taken_mins']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    preprocessor = create_preprocessor()

    xgb_model = XGBRegressor(
        colsample_bytree=0.8,
        learning_rate=0.01,
        max_depth=9,
        n_estimators=600,
        subsample=0.8,
        random_state=42
    )

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", xgb_model)
    ])

    # Training
    model_pipeline.fit(X_train, y_train)

    # Evaluation
    metrics = evaluate_model(
        model_pipeline,
        X_test,
        y_test
    )

    return model_pipeline, metrics