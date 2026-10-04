import pandas as pd
import numpy as np

def clean_data(df):
    drop_cols=['ID','Delivery_person_ID']

    df=df.replace(r"^\s*NaN\s*$", np.nan, regex=True)
    df.drop(drop_cols,axis=1,inplace=True)

    df.columns=df.columns.str.lower().str.strip()

    ## Convert Columns to Numeric
    df['delivery_person_age'] = pd.to_numeric(
        df['delivery_person_age'],
        errors='coerce'
    )
    df['delivery_person_ratings'] = pd.to_numeric(
        df['delivery_person_ratings'],
        errors='coerce'
    )
    df['multiple_deliveries'] = pd.to_numeric(
        df['multiple_deliveries'],
        errors='coerce'
    )

    ## Convert Columns to Date time
    df["order_date"] = pd.to_datetime(
        df["order_date"],
        format="%d-%m-%Y",
        errors="coerce"
    )
    df["time_orderd"] = pd.to_datetime(
        df["time_orderd"],
        format="%H:%M:%S",
        errors="coerce"
    )
    df["time_order_picked"] = pd.to_datetime(
        df["time_order_picked"],
        format="%H:%M:%S",
        errors="coerce"
    )

    cat_cols=df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

    for i in cat_cols:
        df[i] = df[i].str.strip()

    df['weatherconditions'] = df['weatherconditions'].replace('conditions NaN', np.nan)

    df['delivery_person_ratings'] = df['delivery_person_ratings'].clip(upper=5)
    


    return df

