# 🎯 VIP Hunter: End-to-End CLV Prediction Engine

**VIP Hunter** is a professional-grade Machine Learning SaaS application designed to identify high-value customers from raw retail data. It features a complete data pipeline—from raw CSV cleaning to real-time API predictions—packaged in a modern Streamlit dashboard.

---

## 🚀 The Product
Businesses often struggle to identify which customers will drive future revenue. This project solves that by transforming messy transaction logs into actionable business intelligence using a **Linear Regression** model with a 0.73 $R^2$ score.

### Key Features:
* **Automated Pipeline:** Upload a raw "Online Retail" CSV and get a cleaned, engineered VIP list instantly.
* **Real-Time Scoring:** A dedicated interface for sales teams to score individual customers manually.
* **Business-Logic Guardrails:** Integrated mathematical "flooring" to prevent negative revenue predictions.
* **Decoupled Architecture:** A FastAPI backend separated from a Streamlit frontend for scalability.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Machine Learning** | Scikit-Learn, Pandas, NumPy, Joblib |
| **Backend API** | FastAPI, Uvicorn, Pydantic |
| **Frontend UI** | Streamlit |
| **Notebooks/EDA** | Jupyter Notebook, Matplotlib, Seaborn |

---

## 🏗️ System Architecture



1.  **Cleaning Layer:** Handles missing values, removes cancellations (Invoice 'C'), and enforces strict data types.
2.  **Engineering Layer:** Aggregates transaction-level data into customer profiles (Total Quantity, Average Unit Price, Monetary Value).
3.  **Inference Layer:** Loads the trained `.pkl` model and calculates predicted 3-Month CLV.
4.  **API Layer:** Exposes the model via RESTful endpoints (`/predict/single` and `/predict/vip_batch`).
5.  **Frontend Layer:** A user-friendly dashboard built with Streamlit.

---

## 📁 Repository Structure

```text
VIP-Hunter-CLV-Prediction/
├── models/
│   └── champion_linreg.pkl                  # Trained Scikit-Learn Model
├── notebooks/
│   ├── 1-Data Cleaning and EDA.ipynb        # Initial data cleaning and exploration
│   ├── 2- feature_engineering.ipynb         # Grouping data and creating CLV features
│   ├── 3 - EDA and Data Preprocessing.ipynb # Final visual analysis and prep for ML
│   ├── 4 - BaseLine Models .ipynb           # Evaluating multiple algorithms
│   └── 5 -testing.ipynb                     # Pipeline and single-prediction testing
├── cleaning.py                              # Pipeline Step 1: Data Cleaning
├── feature_engineering.py                   # Pipeline Step 2: Feature Engineering
├── ml_engine.py                             # Pipeline Step 3: Inference Logic
├── main.py                                  # FastAPI Backend Routes
├── frontend.py                              # Streamlit Dashboard UI
├── requirements.txt                         # Project Dependencies
└── README.md                                # Project Documentation
```


## ⚙️ Installation & Setup
To run this project locally, follow these steps to set up your environment:

### 1. Clone the Repository
```bash
git clone [https://github.com/suhelkhancode/VIP-Hunter-CLV-Prediction.git]
cd VIP-Hunter-CLV-Prediction
```
### 2. Set Up Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```


## 🚀 How to Run
### 1. Launch the Backend API
The FastAPI server must be running for the frontend to work.

```bash
uvicorn main:app --reload
```
### Once running, you can access the interactive API documentation at http://127.0.0.1:8000/docs.

## 2. Launch the Dashboard
Open a second terminal window, ensure your virtual environment is activated, and run:

```bash
streamlit run frontend.py
```
The dashboard will automatically open in your default browser at http://localhost:8501.

---

## 📈 Model Insights & Logic

The VIP Hunter predictive engine uses a Linear Regression approach focused on three primary customer features. Through rigorous exploratory data analysis, it was discovered that AvgUnitPrice often acts as a negative predictor for future value in this specific retail context—meaning high-ticket "one-off" buyers are less likely to return compared to high-volume "wholesale" buyers. The engine accounts for this multicollinearity and retail behavior to ensure marketing teams focus on the most reliable revenue streams.

## 🤝 Contact & Author
Suhel Khan

Data Science Student

[GitHub](https://github.com/suhelkhancode) | [LinkedIn](https://www.linkedin.com/in/mohdsuhel-khan/)

