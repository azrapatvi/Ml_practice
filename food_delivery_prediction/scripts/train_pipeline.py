import os

from src.data.data_ingestion import download_data, load_data
from src.data.data_cleaning import clean_data
from src.features.feature_engineering import feature_engineering

from src.models.train import train_model
from src.evaluation.evaluate import evaluate_model
from src.models.save_model import save_model

def main():

    #data ingestion
    path=download_data()
    full_path=os.path.join(path,'train.csv')
    df=load_data(full_path)
    print("data loaded successfully\n")

    #data cleaning
    df=clean_data(df)
    print("data cleaned")
    print(df.shape)

    # feature engineering
    df=feature_engineering(df)
    print("done with feature engineering: \n",df.shape)

    #train
    model_pipeline,metrics=train_model(df)

    print("\nFinal Metrics:")
    print(metrics)

    save_model(model_pipeline, "models/model.pkl")
    print("ALL STEPS ARE DONE!!")   


if __name__ == "__main__":
    main()