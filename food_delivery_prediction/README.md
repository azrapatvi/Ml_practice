# Food Delivery Time Prediction

An end-to-end Machine Learning application that predicts food delivery time based on delivery partner details, weather, traffic, vehicle condition, order information, location, time, and delivery distance.

The project includes a complete ML pipeline from data preprocessing and feature engineering to model training, evaluation, model persistence, automated testing, and a FastAPI-based prediction application.

---

## Project Overview

Food delivery time depends on several factors such as traffic density, weather conditions, delivery distance, vehicle condition, delivery partner ratings, order type, and time of ordering.

This project uses historical food delivery data to build a regression model that predicts the expected delivery time in minutes.

### Key Features

* Data ingestion and preprocessing
* Data cleaning
* Feature engineering
* Machine Learning regression model
* Model evaluation using MAE, RMSE, and R²
* Saved trained model using Joblib
* FastAPI REST API
* Web-based prediction interface
* Pydantic request validation
* Automated unit and API testing using Pytest
* Modular ML project structure

---

## Machine Learning Workflow

```text
Raw Dataset
     |
     v
Data Ingestion
     |
     v
Data Cleaning
     |
     v
Feature Engineering
     |
     v
Train / Test Split
     |
     v
Model Training
     |
     v
Model Evaluation
     |
     v
Model Saving
     |
     v
FastAPI Prediction API
     |
     v
Web Interface
```

---

## Model Performance

The current trained model achieved the following results on the test dataset:

| Metric |        Score |
| ------ | -----------: |
| MAE    | 3.11 minutes |
| RMSE   | 3.89 minutes |
| R²     |       0.8273 |

### Interpretation

* **MAE = 3.11**: On average, the prediction differs from the actual delivery time by approximately 3.11 minutes.
* **RMSE = 3.89**: The RMSE indicates the overall prediction error while giving more weight to larger errors.
* **R² = 0.8273**: The model explains approximately 82.73% of the variation in delivery time in the evaluation dataset.

---

## Input Features

The prediction API accepts the following features:

| Feature                   | Description                                |
| ------------------------- | ------------------------------------------ |
| `delivery_person_age`     | Age of the delivery partner                |
| `delivery_person_ratings` | Delivery partner rating                    |
| `weatherconditions`       | Weather condition                          |
| `road_traffic_density`    | Traffic density                            |
| `vehicle_condition`       | Condition of the delivery vehicle          |
| `type_of_order`           | Type of food order                         |
| `type_of_vehicle`         | Vehicle used for delivery                  |
| `multiple_deliveries`     | Number of deliveries handled together      |
| `festival`                | Whether the order occurs during a festival |
| `city`                    | City category                              |
| `time_orderd_hr`          | Order hour                                 |
| `time_orderd_mins`        | Order minute                               |
| `time_order_picked_hr`    | Pickup hour                                |
| `time_order_picked_mins`  | Pickup minute                              |
| `orderd_day_of_week`      | Day of the week                            |
| `orderd_month`            | Month                                      |
| `orderd_is_weekend`       | Whether the order was placed on a weekend  |
| `distance_km`             | Delivery distance in kilometers            |

---

## Project Structure

```text
food-delivery-prediction/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── templates/
│       └── index.html
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── models/
│   └── model.pkl
│
├── notebooks/
│   └── experimentation.ipynb
│
├── scripts/
│   ├── __init__.py
│   └── train_pipeline.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   └── data_cleaning.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── save_model.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py
│   │
│   └── pipeline.py
│
├── tests/
│   ├── __init__.py
│   ├── test_model.py
│   └── test_api.py
│
├── requirements.txt
├── .gitignore
├── Dockerfile
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd food-delivery-prediction
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
```

### 3. Activate the virtual environment

```powershell
.\venv\Scripts\activate
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Train the Model

The complete training pipeline can be executed using:

```powershell
python -m scripts.train_pipeline
```

The pipeline performs:

```text
Data Ingestion
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Saving
```

The trained model is saved to:

```text
models/model.pkl
```

---

## Run the FastAPI Application

Start the application with:

```powershell
uvicorn app.main:app --reload
```

The application will be available locally at:

```text
http://127.0.0.1:8000
```

The web interface allows users to enter delivery information and receive a predicted delivery time.

---

## API Endpoint

### Prediction

```text
POST /predict
```

Example request:

```json
{
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
```

Example response:

```json
{
    "predicted_delivery_time": 21.74,
    "unit": "minutes"
}
```

---

## API Documentation

FastAPI automatically provides interactive API documentation.

After starting the application, open:

```text
/docs
```

This provides an interactive Swagger UI where the `/predict` endpoint can be tested directly.

---

## Testing

The project includes automated tests using Pytest.

Run all tests:

```powershell
python -m pytest
```

Current test coverage includes:

* Model file existence
* Model loading
* Model prediction
* FastAPI home page
* FastAPI prediction endpoint

Current result:

```text
5 passed
```

---

## Technologies Used

### Programming

* Python

### Data Science

* Pandas
* NumPy
* Scikit-learn

### Machine Learning

* Regression
* Feature Engineering
* Model Evaluation
* Joblib

### Backend

* FastAPI
* Pydantic
* Uvicorn

### Testing

* Pytest
* FastAPI TestClient

### Frontend

* HTML
* CSS
* JavaScript
* Jinja2

### Development

* Jupyter Notebook
* Git
* GitHub

---

## MLOps Practices

This project follows a modular machine learning workflow rather than keeping the complete process inside a single notebook.

### Implemented

* Modular data processing
* Separate feature engineering
* Separate model training
* Model persistence
* Prediction module
* Automated model tests
* Automated API tests
* Dependency management using `requirements.txt`
* Git-based version control

### Planned / Optional Extensions

* Model monitoring
* Experiment tracking
* Model registry
* Docker containerization
* CI/CD automation
* Cloud deployment
* Data and model versioning

---

## Why This Project?

This project demonstrates the complete lifecycle of a machine learning application:

```text
Data
 ↓
Preprocessing
 ↓
Feature Engineering
 ↓
Machine Learning
 ↓
Evaluation
 ↓
Model Persistence
 ↓
API
 ↓
Application
 ↓
Automated Testing
```

It demonstrates practical skills required for **Data Scientist, Machine Learning Engineer, and AI Engineer** roles.

---


