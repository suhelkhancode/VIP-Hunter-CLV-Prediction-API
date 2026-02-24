from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel 
import pandas as pd
import io

from cleaning import clean_raw_retail_data
from feature_engineering import engineer_customer_features
from ml_engine import predict_clv_batch, predict_single_clv 

app = FastAPI(title="CLV VIP Hunter API", description="End-to-End CLV Prediction Pipeline")

class CustomerProfile(BaseModel):
    TotalQty: float
    AvgUnitPrice: float
    Monetary_Value: float

@app.post("/predict/single")
async def predict_single(customer: CustomerProfile):
    """Instantly scores a single customer from a JSON payload."""
    try:
        # Pass the validated JSON data to your ML engine
        predicted_clv = predict_single_clv(
            total_qty=customer.TotalQty,
            avg_unit_price=customer.AvgUnitPrice,
            monetary=customer.Monetary_Value
        )
        
        # Return a clean JSON response
        return {
            "status": "success",
            "predicted_3M_clv_usd": predicted_clv
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/vip_batch")
async def process_vip_batch(file: UploadFile = File(...)):
    # 1. Validate the file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    
    try:
        # 2. Read the raw uploaded CSV into memory
        contents = await file.read()
        raw_df = pd.read_csv(io.BytesIO(contents))
        
        # 3. Execute the Pipeline Sequence
        # Step A: Clean and format
        cleaned_df = clean_raw_retail_data(raw_df)
        
        # Step B: Group and aggregate features
        features_df = engineer_customer_features(cleaned_df)
        
        # Step C: Predict and apply business logic
        final_vip_df = predict_clv_batch(features_df)
        
        # 4. Convert the finished VIP DataFrame back to a CSV format
        stream = io.StringIO()
        final_vip_df.to_csv(stream, index=False)
        
        # 5. Return the file as a direct download to the user
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=VIP_Predictions.csv"
        return response
        
    except Exception as e:
        # Catch and display any pipeline errors (like missing columns in the upload)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")