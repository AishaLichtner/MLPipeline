# Import necessary libraries
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def load_data(path="C:\Users\aisha\MLPipeline\data\raw\Dataset_Daily_mileage.csv"):
    df = pd.read_csv(path,sep = ',', encoding="utf-8-sig")
    return df

def convert_date_col(df):
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    return df
def group_by_assert(df):
    return df.groupby('Asset_ID')

    
