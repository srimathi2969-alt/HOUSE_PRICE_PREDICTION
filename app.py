import pandas as pd
import joblib

from fastapi import FastAPI
from pydantic import BaseModel

# Step 1: Create FastAPI Application

app = FastAPI(title="House Price Prediction API")

# Step 2: Load Trained Model

model = joblib.load("Models/house_price_model.pkl")

# Load Feature Columns

columns = joblib.load("Models/house_price_columns.pkl")

# Step 3: Define Input Data

class HouseData(BaseModel):
    Area: float
    Bedrooms: float
    Bathrooms: float
    Age: float

# Step 4: Home Route

@app.get("/")
def home():
    return {
        "message": "Welcome to House Price Prediction API"
    }

# Step 5: Prediction Route

@app.post("/predict")
def predict_price(house: HouseData):

    input_data = pd.DataFrame(
        [[
            house.Area,
            house.Bedrooms,
            house.Bathrooms,
            house.Age
        ]],
        columns=["Area", "Bedrooms", "Bathrooms", "Age"]
    )

    input_data = input_data[columns]

    prediction = model.predict(input_data)

    return {
        "Predicted House Price": round(float(prediction[0]), 2)
    }

print("House Price Prediction API Ready")