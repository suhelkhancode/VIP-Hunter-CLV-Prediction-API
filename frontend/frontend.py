import streamlit as st
import requests
import pandas as pd
import io

# The URLs where your FastAPI backend is listening
API_URL_BATCH = "http://127.0.0.1:8000/predict/vip_batch"
API_URL_SINGLE = "http://127.0.0.1:8000/predict/single"

# --- Page Config ---
st.set_page_config(page_title="VIP Hunter", page_icon="🎯", layout="centered")

st.title(" VIP Hunter: CLV Prediction Engine")
st.markdown("Upload your raw retail data to identify your highest-value customers, or score a single customer in real-time.")

# --- Create Two Tabs ---
tab1, tab2 = st.tabs(["📁 Batch VIP Prediction", "👤 Single Customer Scoring"])

# ==========================================
# TAB 1: BATCH UPLOAD (The VIP List)
# ==========================================
with tab1:
    st.header("Process Raw Sales Data")
    st.write("Upload your raw system export. The ML pipeline will clean, engineer features, and predict the 3-Month CLV.")
    
    uploaded_file = st.file_uploader("Upload Raw Online Retail CSV", type=["csv"])
    
    if uploaded_file is not None:
        if st.button("Predict VIPs 🚀"):
            with st.spinner("Processing data through ML pipeline..."):
                # Send the file to the FastAPI backend
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                response = requests.post(API_URL_BATCH, files=files)
                
                if response.status_code == 200:
                    st.success("Pipeline Execution Complete!")
                    
                    # Read the CSV returned by the API
                    result_df = pd.read_csv(io.StringIO(response.text))
                    
                    # Show a preview of the top 5 VIPs
                    st.subheader("🏆 Top 5 Predicted VIPs")
                    st.dataframe(result_df.head(5), use_container_width=True)
                    
                    # Provide the download button for the full file
                    st.download_button(
                        label="⬇️ Download Full VIP List (.csv)",
                        data=response.text,
                        file_name="VIP_Predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"API Error: {response.text}")

# ==========================================
# TAB 2: SINGLE PREDICTION (CRM Engine)
# ==========================================
with tab2:
    st.header("Real-Time CRM Scoring")
    st.write("Manually enter a customer's current metrics to instantly predict their future value.")
    
    # Create 3 columns for neat input boxes
    col1, col2, col3 = st.columns(3)
    with col1:
        qty = st.number_input("Total Quantity Bought", min_value=0, value=5)
    with col2:
        price = st.number_input("Avg Unit Price ($)", min_value=0.0, value=450.0)
    with col3:
        monetary = st.number_input("Total Spent ($)", min_value=0.0, value=2250.0)
        
    if st.button("Calculate 3-Month CLV 🧠"):
        # Package the data into the exact JSON format your API expects
        payload = {
            "TotalQty": qty,
            "AvgUnitPrice": price,
            "Monetary_Value": monetary
        }
        
        # Send JSON to the FastAPI backend
        response = requests.post(API_URL_SINGLE, json=payload)
        
        if response.status_code == 200:
            clv = response.json().get("predicted_3M_clv_usd")
            
            # Display a massive, beautiful metric card
            st.metric(label="Predicted 3-Month Future Spend", value=f"${clv:,.2f}")
        else:
            st.error("Error connecting to the ML Backend")