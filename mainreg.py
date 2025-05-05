from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import pickle
import io
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI()

# CORS mid
# dleware (ensure this is present)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Invoice Approval Predictor API. Visit /docs for API documentation."}
# Load model
try:
    with open('Amount_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    raise Exception("Amount_model.pkl not found")
except Exception as e:
    raise Exception(f"Error loading model: {str(e)}")

# Pydantic model for single prediction input
class InvoiceInput(BaseModel):
    Amount: float
    VATRate: float
    SupplierPrice: float
    PaymentStatus: str

# Pydantic model for prediction output
class PredictionOutput(BaseModel):
    prediction: str
    probabilities: List[float]

# Single prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
async def predict(invoice: InvoiceInput):
    try:
        # Validate inputs
        if invoice.Amount < 0 or invoice.VATRate < 0 or invoice.SupplierPrice < 0:
            raise HTTPException(status_code=400, detail="Inputs must be non-negative")

        # Preprocess input
        log_amount = np.log1p(invoice.Amount)
        payment_status_encoded = 0 if invoice.PaymentStatus.lower() == "paid" else 1

        input_data = pd.DataFrame({
            'Amount': [invoice.Amount],
            'VATRate': [invoice.VATRate],
            'SupplierPrice': [invoice.SupplierPrice],
            'PaymentStatus': [payment_status_encoded],
            'log_amount': [log_amount]
        })

        expected_columns = ['Amount', 'VATRate', 'SupplierPrice', 'PaymentStatus', 'log_amount']
        input_data = input_data[expected_columns]

        # Predict
        prediction = model.predict(input_data)[0]
        prediction_label = "Approved" if prediction == 1 else "Not Approved"
        probabilities = model.predict_proba(input_data)[0].tolist()

        return PredictionOutput(prediction=prediction_label, probabilities=probabilities)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")

        df_uploaded = pd.read_csv(file.file)
        required_columns = ['Amount', 'VATRate', 'SupplierPrice', 'PaymentStatus']
        if not all(col in df_uploaded.columns for col in required_columns):
            raise HTTPException(status_code=400, detail="CSV must have columns: Amount, VATRate, SupplierPrice, PaymentStatus")

        # Validate numeric columns
        for col in ['Amount', 'VATRate', 'SupplierPrice']:
            if not pd.to_numeric(df_uploaded[col], errors='coerce').notna().all():
                raise HTTPException(status_code=400, detail=f"Column {col} must contain numeric values")

        # Preprocess
        df_uploaded['log_amount'] = np.log1p(df_uploaded['Amount'])
        df_uploaded['PaymentStatus'] = df_uploaded['PaymentStatus'].str.lower().map({
            'paid': 0, 'unpaid': 1
        })
        if df_uploaded['PaymentStatus'].isna().any():
            raise HTTPException(status_code=400, detail="Invalid PaymentStatus values. Use 'Paid' or 'Unpaid'")

        batch_data = df_uploaded[expected_columns]
        predictions = model.predict(batch_data)
        df_uploaded['Prediction'] = ['Approved' if p == 1 else 'Not Approved' for p in predictions]

        # Return CSV
        csv_buffer = io.StringIO()
        df_uploaded.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        return StreamingResponse(
            io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=invoice_predictions.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)