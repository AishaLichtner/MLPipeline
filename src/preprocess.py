# Import necessary libraries
import pandas as pd
import numpy as np
import os
from prophet import Prophet

def run_preprocess(path_in=r"C:\Users\aisha\MLPipeline\data\raw\Dataset_Daily_mileage.csv", path_out= r"C:\Users\aisha\MLPipeline\data\processed"):
  
  df = load_data(path_in)
  #convert date col
  df['Date'] = pd.to_datetime(df['Date'])
  #group by provided assets
  asset_data = df.groupby('Asset_ID')

  #storage data
  data_dict={}
  for asset_id, as_group in asset_data:
    print(f"processing asset: {asset_id}")
    #run all preprocessing steps:
    data = handle_missing_rows(start_date=df["Date"].min(), end_date=df["Date"].max(), df=df, col="Daily_mileage", approach="zero")
    outliers = identifiy_outliers(data)
    data = normalize_outliers(outliers = outliers,df=data, approach="prophet")
    data= transform_to_prophet_format(data)
    write_processed_data_to_csv(data, os.path.join(path_out, f"{asset_id}.csv"))
    data_dict[asset_id]=data
  return data_dict

def write_processed_data_to_csv(data, path_out):
   data.to_csv(path_out)
   
   

def load_data(path=r"C:\Users\aisha\MLPipeline\data\raw\Dataset_Daily_mileage.csv"):
    df = pd.read_csv(path,sep = ',', encoding="utf-8-sig")
    return df
def transform_to_prophet_format(df, date_col = 'Date', dep_var = 'Daily_mileage'):
    # Prepare the training data for Prophet
    df = df.rename(columns={date_col: 'ds',  dep_var: 'y'})
    return df
def handle_missing_values(df, date_col = 'Date', col="Daily_mileage",approach="zero"):


  if len(df[df[col].isnull()]) == 0:
    print("no missing values found")
    return df

  filled_df = df.copy()


  if approach == "zero":
      #fill missing values with zero, assuming that missing values indicate no vehicle activity
      filled_df[col].fillna(0,inplace=True)
  elif approach == "mean":
      #fill missing values with mean values for the vehicle
      filled_df[col].fillna(df[col].mean(),inplace=True)
  elif approach == "median":
      #fill missing values with median values for the vehicle
      filled_df[col].fillna(df[col].median(),inplace=True)
  elif approach == "interpolate":
      #fill missing values by interpolation
      filled_df[col].interpolate(inplace=True)
  elif approach == "prophet":
      #leave missing values as NaN values for prophet model
      pass


  return filled_df

def handle_missing_rows(start_date, end_date, df, col="Daily_mileage", approach="zero"):


  #create a range of all the dates between start and end date
  full_date_range = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='D'), columns=['Date'])

  #right join to get missing dates
  merged_df = pd.merge(df,full_date_range, on='Date', how='right')
  merged_df["Asset_ID"] = df.Asset_ID.max()

  return handle_missing_values(df=merged_df,col=col,approach=approach)

def identifiy_outliers(df,col="Daily_mileage"):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
  Q1 = df[col].quantile(0.25)
  Q3 = df[col].quantile(0.75)
  IQR = Q3 - Q1

  # Define outlier bounds
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR

  # Identify outliers
  outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

  if outliers.empty:
    print("no outliers identified")
    return outliers
  return outliers

def normalize_outliers(df, outliers, col="Daily_mileage", approach="interpolate"):
  df.loc[df.index.isin(outliers.index),col] = np.nan
  return handle_missing_values(df=df, col=col, approach = approach)

#################################### 
def convert_date_col(df):
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    return df
def group_by_assert(df):
    return df.groupby('Asset_ID')

def split_train_test(df,date_col="Date", split_date= '2024-04-23'):
  train_data = df[df[date_col] < split_date]
  test_data = df[df[date_col] >= split_date]

###########################################################
    
if __name__ == "__main__":
   #change path to where your raw data lies
   run_preprocess(path_in=r"C:\Users\aisha\MLPipeline\MLPipeline\data\raw\Dataset_Daily_mileage.csv", 
                  path_out=r"C:\Users\aisha\MLPipeline\MLPipeline\data\processed")
                  