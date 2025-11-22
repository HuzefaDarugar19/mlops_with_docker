import dash
from dash import dcc, html, Input, Output, State
import requests
import os

app = dash.Dash(__name__)
server = app.server

API_URL = os.getenv("API_URL", "http://localhost:8001")

app.layout = html.Div([
    html.Div([
        html.H1("Car Price Predictor"),
        
        html.Div([
            html.Label("Age (years)"),
            dcc.Input(id="age", type="number", value=5, required=True),
        ], className="input-group"),
        
        html.Div([
            html.Label("Mileage (km)"),
            dcc.Input(id="mileage", type="number", value=50000, required=True),
        ], className="input-group"),
        
        html.Div([
            html.Label("Weight (lbs)"),
            dcc.Input(id="weight", type="number", value=3000, required=True),
        ], className="input-group"),
        
        html.Div([
            html.Label("Horsepower"),
            dcc.Input(id="horsepower", type="number", value=200, required=True),
        ], className="input-group"),
        
        html.Button("Predict Price", id="predict-btn", n_clicks=0),
        
        html.Div(id="prediction-output")
    ], className="container")
])

@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("age", "value"),
    State("mileage", "value"),
    State("weight", "value"),
    State("horsepower", "value"),
    prevent_initial_call=True
)
def predict_price(n_clicks, age, mileage, weight, horsepower):
    try:
        payload = {
            "age": int(age),
            "mileage": int(mileage),
            "weight": int(weight),
            "horsepower": int(horsepower)
        }
        response = requests.post(f"{API_URL}/predict", json=payload)
        if response.status_code == 200:
            price = response.json()["predicted_price"]
            return f"Estimated Price: ${price:,.2f}"
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error connecting to API: {e}"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8051)
