import joblib
import pandas as pd
import os

# 1. Load the model into memory ONLY ONCE when the server starts.
# This makes the API lightning fast because it doesn't reload the file for every request.
MODEL_PATH = os.path.join("models", "linreg.pkl")
model = joblib.load(MODEL_PATH)

def predict_clv_batch(features_df: pd.DataFrame) -> pd.DataFrame:
    """Applies the trained CLV model to the engineered features."""
    df = features_df.copy()
    
    # 2. Isolate the exact columns the model expects in the precise order
    X_predict = df[['TotalQty', 'AvgUnitPrice', 'Monetary_Value']]
    
    # 3. Generate the raw mathematical predictions
    raw_predictions = model.predict(X_predict)
    
    # 4. Apply the Business Logic (The Floor)
    # Customers cannot have a negative future value, so we force < 0 to $0.00
    df['Predicted_3M_CLV'] = [round(max(0, p), 2) for p in raw_predictions]
    
    # 5. Sort the DataFrame so the highest VIPs are at the top of the CSV
    df = df.sort_values(by='Predicted_3M_CLV', ascending=False).reset_index(drop=True)
    
    return df


def predict_single_clv(total_qty: float, avg_unit_price: float, monetary: float) -> float:
    """Predicts CLV for a single customer profile."""
    
    # 1. Package the single customer into the exact DataFrame structure
    customer_data = pd.DataFrame({
        'TotalQty': [total_qty],
        'AvgUnitPrice': [avg_unit_price],
        'Monetary_Value': [monetary]
    })
    
    # 2. Predict
    raw_prediction = model.predict(customer_data)[0]
    
    # 3. Apply the floor (no negative CLV)
    return round(max(0, raw_prediction), 2)