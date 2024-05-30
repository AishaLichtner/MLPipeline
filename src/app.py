import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import requests

# Initialize Dash app
app = dash.Dash(__name__)

# Define asset IDs and their corresponding model endpoints
assets = {
    'asset1': 'http://model_endpoint1.com/predict',
    'asset2': 'http://model_endpoint2.com/predict',
    'asset3': 'http://model_endpoint3.com/predict'
}

# Define app layout
app.layout = html.Div([
    html.H1("Your Model Web App"),
    html.Label("Choose Asset ID:"),
    dcc.Dropdown(id="asset_id_dropdown", options=[{'label': asset, 'value': asset} for asset in assets.keys()], value=list(assets.keys())[0]),
    html.Button("Run Predictions", id="predict_button", n_clicks=0),
    html.Div(id="prediction_output"),
    dcc.Graph(id="performance_metrics_plot")
])

# Define callback to handle prediction
@app.callback(
    Output("prediction_output", "children"),
    [Input("predict_button", "n_clicks")],
    [Input("asset_id_dropdown", "value")]
)
def predict(n_clicks, asset_id):
    if n_clicks > 0:
        # Make HTTP request to the corresponding model endpoint based on the selected asset
        endpoint = assets[asset_id]
        response = requests.post(endpoint, json={'data': 'your_input_data'})
        prediction_result = response.json()["prediction"]  # Assuming the response contains a 'prediction' key
        return f"Prediction for {asset_id}: {prediction_result}"

if __name__ == "__main__":
    app.run_server(debug=True)