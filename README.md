# Quick Introduction to Project
## Notebook
For detailed information on model training process read the My_Mileage_forecast.ipynb notebook.
To build new models for different asset datasets follow the following instructions on how to install the packages and run the data preprocessing and model training.
## Install requirements

1. Create a local environment (using for example anaconda)
and install packages from requirements.txt.
(pip install -r requirements.txt)

## Run preprocessing 

1. Add your raw data in the CSV format to MLPipeline\data\raw folder.
2. Change the input path in the main function of preprocessing.py to your local path to the raw data.
3. Change the output path to <your_local_path>\MLPipeline\data\processed"
4. Run preprocess.py in your environment

Now you can find the preprocessed data in MLPipeline\data\processed

## Run training
1. Change the input path in the main function of tune_and_train.py to the local path of the data that was just preprocessed, something link: "\local_path\MLPipeline\data\processed\asset_id.csv"
2. Change the output path to <your_local_path>\MLPipeline\MLPipeline\outputs"
3. Run tune_and_train.py

Now you can find the trained model and performance plot in your output folder.









