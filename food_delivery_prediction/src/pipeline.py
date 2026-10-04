from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def create_preprocessor():

    # Numerical columns → Impute + Scale
    num_cols = [
        'delivery_person_age',
        'delivery_person_ratings',
        'time_order_picked_hr',
        'time_order_picked_mins',
        'time_orderd_hr',
        'time_orderd_mins',
        'orderd_day_of_week',
        'orderd_month',
        'orderd_is_weekend',
        'distance_km'
    ]

    # Categorical columns → Impute + One Hot Encoding
    cat_cols_ohe = [
        'weatherconditions',
        'type_of_order',
        'type_of_vehicle',
        'city'
    ]

    # Categorical column → Impute + Ordinal Encoding
    cat_cols_ordinal = [
        'road_traffic_density'
    ]

    # Numerical columns → Impute only
    no_encoding_int = [
        'vehicle_condition',
        'multiple_deliveries',
        'festival'
    ]

    # Numerical pipeline
    num_impute_scaling_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy='median')),
        ("scaling", StandardScaler())
    ])

    # One-hot pipeline
    cat_impute_ohe_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy='most_frequent')),
        ("ohe", OneHotEncoder(handle_unknown='ignore'))
    ])

    # Ordinal pipeline
    cat_impute_ordinal_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy='most_frequent')),
        ("ordinal", OrdinalEncoder(
            categories=[['Low', 'Medium', 'High', 'Jam']]
        ))
    ])

    # Numerical columns → Impute only
    num_impute_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy='median'))
    ])

    # Combine everything
    preprocessor = ColumnTransformer([
        ("num", num_impute_scaling_pipeline, num_cols),
        ("ohe", cat_impute_ohe_pipeline, cat_cols_ohe),
        ("ordinal", cat_impute_ordinal_pipeline, cat_cols_ordinal),
        ("no_encoding", num_impute_pipeline, no_encoding_int)
    ])

    return preprocessor