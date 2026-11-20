# ==========================================
# HOUSE PRICE PREDICTION - API TESTING
# ==========================================

import requests

# Step 1: API URL

url = "http://127.0.0.1:8001"

# Step 2: Test Home API

response = requests.get(url)

print("HOME API RESPONSE:")
print(response.status_code)
print(response.json())

# Step 3: Test Prediction API

predict_url = url + "/predict"

house_data = {
    "Area": 1500,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Age": 5
}

response = requests.post(predict_url, json=house_data)

print("\nPREDICTION API RESPONSE:")
print(response.status_code)
print(response.json())