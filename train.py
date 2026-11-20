# ==========================================
# HOUSE PRICE PREDICTION
# ==========================================

import pandas as pd

# Step 1: Load Dataset

data = pd.read_csv("Dataset/Housing.csv")

print(data)
print(data.head())

# Step 2: Check Missing Values

print(data.isnull().sum())
print(data.isnull().sum().sum())

# Step 3: Handle Missing Values

# Remove rows where target column is missing

data = data.dropna(subset=["Price"])

numerical_columns = data.select_dtypes(include=["number"]).columns

categorical_columns = data.select_dtypes(include=["object"]).columns

for column in numerical_columns:
    data[column] = data[column].fillna(data[column].median())

for column in categorical_columns:
    if data[column].isnull().sum() > 0:
        data[column] = data[column].fillna(data[column].mode()[0])

print(data)
print(data.isnull().sum())

# Step 4: Check and Remove Duplicates

print("Duplicate values:", data.duplicated().sum())

data = data.drop_duplicates()

print(data)

# Step 5: Save Cleaned Dataset

data.to_csv("Dataset/Housing_clean.csv", index=False)

print("Cleaned data saved successfully")

# Step 6: Convert Categorical Columns into Numbers

data = pd.get_dummies(data, drop_first=True, dtype=int)

print(data.head())

# Step 7: Separate Input and Output

x = data.drop("Price", axis=1)

y = data["Price"]

print(x)
print(y)

# Step 8: Split Dataset

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Step 9: Import Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Step 10: Train Model

model.fit(x_train, y_train)

print("Model training completed successfully")

# Step 11: Predict House Prices

y_pred = model.predict(x_test)

print("Actual House Prices:")
print(y_test.values)

print("Predicted House Prices:")
print(y_pred)

# Step 12: Evaluate Model

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)

print("Mean Squared Error:", mse)

print("R2 Score:", r2)

print("R2 Score Percentage:", r2 * 100, "%")

# Step 13: Save Trained Model

import joblib
import os

os.makedirs("Models", exist_ok=True)

joblib.dump(model, "Models/house_price_model.pkl")

# Save feature columns for future prediction

joblib.dump(x.columns.tolist(), "Models/house_price_columns.pkl")

print("Model saved successfully")
print("Feature columns saved successfully")