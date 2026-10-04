import pandas as pd
import numpy as np
import kagglehub
import os

def download_data():
    # Download latest version
    path = kagglehub.dataset_download("gauravmalik26/food-delivery-dataset")
    print("Path to dataset files:", path)
    return path

def load_data(df_path):

    df=pd.read_csv(df_path)
    return df

if __name__=="__main__":

    path=download_data()
    full_path=os.path.join(path,'train.csv')

    df=load_data(full_path)

    print(df.head())