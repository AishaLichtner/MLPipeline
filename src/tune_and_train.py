# Import necessary libraries
import pandas as pd
import numpy as np
import os
import joblib
import itertools
from prophet import Prophet
from prophet.diagnostics import performance_metrics,cross_validation
from prophet.plot import plot_cross_validation_metric

def hyperparam_tuning(train_data, metric='rmse',initial= "120 days",period='2days',horizon = '21 days'):
  param_grid = {
      'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
      'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
  }

  # Generate all combinations of parameters
  all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
  rmses = []  # Store the RMSEs for each params here

  # Use cross validation to evaluate all parameters
  for params in all_params:
      model = Prophet(**params).fit(train_data)  # Fit model with given params with our chosen processed data
      df_cv = cross_validation(model,initial=initial, period=period , horizon = horizon)
      df_p = performance_metrics(df_cv, rolling_window=1)

      rmses.append(df_p[metric].values[0])

  # Find the best parameters
  tuning_results = pd.DataFrame(all_params)
  tuning_results[metric] = rmses
  best_params = all_params[tuning_results[metric].idxmin()]

  return best_params

def evaluate_model(model,model_name, path=None):
    #241 Tage Das ist 7 Monate und 28 Tage
    df_cv = cross_validation(model,initial="120 days", period='2days' , horizon = '21 days')
    print("this is the output of the cross validation ")
    print(df_cv.head())

    #compute performance metrics
    df_p = performance_metrics(df_cv)
    df_p.head()

    fig = plot_cross_validation_metric(df_cv, metric='rmse')
    fig.suptitle(model_name)

    return df_p, df_cv
def train_prothet_model(train_data, hyperparameter=None):

      # Initialize the Prophet model
      # Train final model with best parameters
      if hyperparameter:
        model = Prophet(**hyperparameter, weekly_seasonality=True, daily_seasonality=False)
      else:
        model = Prophet(weekly_seasonality=True, daily_seasonality=False)
      # Fit the model
      model.fit(train_data)
      return model

def save_model(model, model_name, path=None):
  # Save the model to a file
  if path:
    # Ensure the path directory exists
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, f'{model_name}.pkl')
  else:
    file_path = f'{model_name}.pkl'
  joblib.dump(value= model, filename=file_path)

def run_training(input_path=None, output_path=None):
  
    data = pd.read_csv(input_path,sep = ',', encoding="utf-8-sig")
    asset_id = data.Asset_ID[0]
    best_params =hyperparam_tuning(data)
    model = train_prothet_model(data, hyperparameter=best_params)
    df_cv , df_p = evaluate_model(model= model,model_name=f"{asset_id}_model", path = os.path.join(output_path, "documents"))
    save_model(model=model, model_name= f"{asset_id}_model", path = os.path.join(output_path, "models"))
    print(df_cv.head())
    print(df_p.head())

if __name__ == "__main__":    

    run_training(input_path=r"C:\Users\aisha\MLPipeline\MLPipeline\data\processed\1036708.csv", output_path= r"C:\Users\aisha\MLPipeline\MLPipeline\outputs")
    run_training(input_path=r"C:\Users\aisha\MLPipeline\MLPipeline\data\processed\1051822.csv", output_path= r"C:\Users\aisha\MLPipeline\MLPipeline\outputs")
    

