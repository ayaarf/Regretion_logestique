import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import io



# Load model
try:
    with open('Amount_model.pkl', 'rb') as file:
        model = pickle.load(file)
    st.success("Model loaded successfully")
except FileNotFoundError:
    st.error("Error: Amount_model.pkl not found in the current directory")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# If using scaler or LabelEncoder, uncomment these
# try:
#     with open('scalerAmount.pkl', 'rb') as file:
#         scaler = pickle.load(file)
#     with open('label_encoder.pkl', 'rb') as file:
#         le = pickle.load(file)
# except FileNotFoundError:
#     st.error("Error: scalerAmount.pkl or label_encoder.pkl not found")
#     st.stop()

# Title
st.title("Invoice Approval Predictor")
st.write("Enter invoice details or upload a CSV to predict approval status")

# Single prediction
st.header("Single Prediction")
col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Amount", min_value=0.0, value=5000.0, step=0.01)
    vat_rate = st.number_input("VAT Rate (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.01)
with col2:
    supplier_price = st.number_input("Supplier Price", min_value=0.0, value=1000.0, step=0.01)
    payment_status = st.selectbox("Payment Status", ["Paid", "Unpaid"])

# Validate inputs
if amount < 0 or supplier_price < 0 or vat_rate < 0:
    st.error("Inputs must be non-negative")
    st.stop()

# Preprocess input
st.write("DEBUG: Preprocessing input")
log_amount = np.log1p(amount)
payment_status_encoded = 0 if payment_status == "Paid" else 1  # Manual encoding
# If using LabelEncoder: payment_status_encoded = le.transform([payment_status])[0]

input_data = pd.DataFrame({
    'Amount': [amount],
    'VATRate': [vat_rate],
    'SupplierPrice': [supplier_price],
    'PaymentStatus': [payment_status_encoded],
    'log_amount': [log_amount]
})

# Apply scaler if used
# num_cols = ['Amount', 'VATRate', 'SupplierPrice', 'log_amount']
# input_data[num_cols] = scaler.transform(input_data[num_cols])

# Ensure column order
expected_columns = ['Amount', 'VATRate', 'SupplierPrice', 'PaymentStatus', 'log_amount']
input_data = input_data[expected_columns]

# Predict
if st.button("Predict"):
    try:
        st.write("DEBUG: Making prediction")
        prediction = model.predict(input_data)[0]
        prediction_label = "Approved" if prediction == 1 else "Not Approved"
        st.markdown(f"### Prediction: **{prediction_label}**")

        # Probability chart
        probabilities = model.predict_proba(input_data)[0]
        prob_df = pd.DataFrame({
            'Class': ['Not Approved', 'Approved'],
            'Probability': probabilities
        })

        st.subheader("Prediction Confidence")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Probability', y='Class', data=prob_df, ax=ax, palette='Set2')
        ax.set_xlabel("Probability")
        ax.set_title("Likelihood of Approval")
        for i, v in enumerate(probabilities):
            ax.text(v + 0.01, i, f"{v:.2%}", va='center')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Batch prediction
st.header(" Prediction CSV")
st.write("Upload a CSV with columns: Amount, VATRate, SupplierPrice, PaymentStatus")
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    try:
        st.write("DEBUG: Processing CSV")
        df_uploaded = pd.read_csv(uploaded_file)
        required_columns = ['Amount', 'VATRate', 'SupplierPrice', 'PaymentStatus']
        if not all(col in df_uploaded.columns for col in required_columns):
            st.error("CSV must have columns: Amount, VATRate, SupplierPrice, PaymentStatus")
            st.stop()

        # Validate numeric columns
        for col in ['Amount', 'VATRate', 'SupplierPrice']:
            if not pd.to_numeric(df_uploaded[col], errors='coerce').notna().all():
                st.error(f"Column {col} must contain numeric values")
                st.stop()

        # Preprocess
        df_uploaded['log_amount'] = np.log1p(df_uploaded['Amount'])
        df_uploaded['PaymentStatus'] = df_uploaded['PaymentStatus'].str.lower().map({
            'paid': 0, 'unpaid': 1
        })
        if df_uploaded['PaymentStatus'].isna().any():
            st.error("Invalid PaymentStatus values in CSV. Use 'Paid' or 'Unpaid'")
            st.stop()

        # If using LabelEncoder:
        # df_uploaded['PaymentStatus'] = le.transform(df_uploaded['PaymentStatus'])

        batch_data = df_uploaded[expected_columns]
        # Apply scaler if used
        # batch_data[num_cols] = scaler.transform(batch_data[num_cols])

        predictions = model.predict(batch_data)
        df_uploaded['Prediction'] = ['Approved' if p == 1 else 'Not Approved' for p in predictions]

        st.subheader("CSV Results")
        st.dataframe(df_uploaded)

        # Download
        csv_buffer = io.StringIO()
        df_uploaded.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Results",
            data=csv_buffer.getvalue(),
            file_name="invoice_predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Batch processing error: {str(e)}")