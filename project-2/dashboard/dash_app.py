# app.py
import dash
from dash import dcc, html, Input, Output, State
import requests

# URL of your FastAPI backend
API_URL = "http://127.0.0.1:8000/predict"

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div(
    style={"font-family": "Arial", "margin": "40px"},
    children=[
        html.H1("ML Model Prediction Dashboard"),
        html.P("Enter a number to get prediction from the FastAPI model:"),

        dcc.Input(
            id="input-x",
            type="number",
            placeholder="Enter a number",
            style={"marginRight": "10px"}
        ),
        html.Button("Predict", id="predict-button", n_clicks=0),
        html.Br(), html.Br(),
        html.Div(id="output-prediction", style={"fontSize": "20px", "color": "blue"})
    ]
)

# Callback to connect input → FastAPI → output
@app.callback(
    Output("output-prediction", "children"),
    Input("predict-button", "n_clicks"),
    State("input-x", "value")
)
def update_prediction(n_clicks, x_value):
    if n_clicks > 0 and x_value is not None:
        try:
            # Call FastAPI
            response = requests.get(API_URL, params={"x": x_value})
            if response.status_code == 200:
                result = response.json()
                return f"Prediction for x={x_value}: {result['prediction']}"
            else:
                return "Error: Could not get prediction from API."
        except Exception as e:
            return f"Error: {str(e)}"
    return ""  # no output before clicking

# Run the Dash app
if __name__ == "__main__":
    app.run(debug=True, port=8050)
