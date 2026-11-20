# ==========================================
# HOUSE PRICE PREDICTION - MLFLOW TRACKING
# ==========================================

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Step 1: Load Dataset

data = pd.read_csv("Dataset/Housing.csv")

# Step 2: Handle Missing Values

data = data.dropna(subset=["Price"])

numerical_columns = data.select_dtypes(include=["number"]).columns

for column in numerical_columns:
    data[column] = data[column].fillna(data[column].median())

# Step 3: Remove Duplicates

data = data.drop_duplicates()

# Step 4: Separate Input and Output

x = data.drop("Price", axis=1)
y = data["Price"]

# Step 5: Split Dataset

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Step 6: Set MLflow Experiment

mlflow.set_experiment("House_Price_Prediction")

# Step 7: Start MLflow Run

with mlflow.start_run():

    # Step 8: Create Model

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # Step 9: Train Model

    model.fit(x_train, y_train)

    print("Model training completed")

    # Step 10: Predict

    y_pred = model.predict(x_test)

    # Step 11: Calculate Metrics

    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)

    print("MAE:", mae)
    print("MSE:", mse)
    print("R2 Score:", r2)

    # Step 12: Log Parameters

    mlflow.log_param("n_estimators", 100)

    mlflow.log_param("random_state", 42)

    # Step 13: Log Metrics

    mlflow.log_metric("MAE", mae)

    mlflow.log_metric("MSE", mse)

    mlflow.log_metric("R2_Score", r2)

    # Step 14: Log Model

    mlflow.sklearn.log_model(
    model,
    name="house_price_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)

    print("MLflow tracking completed successfully")