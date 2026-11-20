# ==========================================
# HOUSE PRICE PREDICTION
# ==========================================

import pandas as pd
import joblib

# Step 1: Load Trained Model

model = joblib.load("Models/house_price_model.pkl")

# Step 2: Load Feature Columns

columns = joblib.load("Models/house_price_columns.pkl")

# Step 3: Get House Details

area = float(input("Enter House Area: "))
bedrooms = float(input("Enter Number of Bedrooms: "))
bathrooms = float(input("Enter Number of Bathrooms: "))
age = float(input("Enter House Age: "))

# Step 4: Create Input Data

house_data = pd.DataFrame(
    [[area, bedrooms, bathrooms, age]],
    columns=["Area", "Bedrooms", "Bathrooms", "Age"]
)

# Step 5: Arrange Input Columns

house_data = house_data[columns]

# Step 6: Predict House Price

prediction = model.predict(house_data)

# Step 7: Display Result

print("--------------------------------")
print("HOUSE PRICE PREDICTION RESULT")
print("--------------------------------")

print("Area:", area)
print("Bedrooms:", bedrooms)
print("Bathrooms:", bathrooms)
print("Age:", age)

print("Predicted House Price: ₹", round(float(prediction[0]), 2))

print("--------------------------------")