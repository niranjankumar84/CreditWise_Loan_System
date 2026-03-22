import streamlit as st
import numpy as np
import pickle

# Load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("💳 CreditWise Loan Approval System")

st.write("Fill details to predict loan approval")

# ===== INPUTS =====
Age = st.number_input("Age", value=30)
Dependents = st.number_input("Dependents", value=0)
Existing_Loans = st.number_input("Existing Loans", value=1)
Savings = st.number_input("Savings", value=50000)
Collateral_Value = st.number_input("Collateral Value", value=100000)
Loan_Amount = st.number_input("Loan Amount", value=200000)
Loan_Term = st.number_input("Loan Term", value=12)
Education_Level = st.number_input("Education Level", value=1)

# Employment Status (One-hot)
emp_status = st.selectbox("Employment Status", 
                         ["Salaried", "Self-employed", "Unemployed"])

Employment_Status_Salaried = 1 if emp_status == "Salaried" else 0
Employment_Status_Self_employed = 1 if emp_status == "Self-employed" else 0
Employment_Status_Unemployed = 1 if emp_status == "Unemployed" else 0

# Marital
married = st.selectbox("Marital Status", ["Single", "Married"])
Marital_Status_Single = 1 if married == "Single" else 0

# Loan Purpose
purpose = st.selectbox("Loan Purpose", 
                       ["Car", "Education", "Home", "Personal"])

Loan_Purpose_Car = 1 if purpose == "Car" else 0
Loan_Purpose_Education = 1 if purpose == "Education" else 0
Loan_Purpose_Home = 1 if purpose == "Home" else 0
Loan_Purpose_Personal = 1 if purpose == "Personal" else 0

# Property Area
area = st.selectbox("Property Area", ["Semiurban", "Urban", "Rural"])
Property_Area_Semiurban = 1 if area == "Semiurban" else 0
Property_Area_Urban = 1 if area == "Urban" else 0

# Gender
gender = st.selectbox("Gender", ["Male", "Female"])
Gender_Male = 1 if gender == "Male" else 0

# Employer Category
employer = st.selectbox("Employer Category", 
                       ["Government", "MNC", "Private", "Unemployed"])

Employer_Category_Government = 1 if employer == "Government" else 0
Employer_Category_MNC = 1 if employer == "MNC" else 0
Employer_Category_Private = 1 if employer == "Private" else 0
Employer_Category_Unemployed = 1 if employer == "Unemployed" else 0

# ===== EXTRA FEATURES =====
DTI_Ratio_sq = st.number_input("DTI Ratio Squared", value=0.1)
Credit_Score_sq = st.number_input("Credit Score Squared", value=500000)
Applicant_Income_log = st.number_input("Applicant Income (log)", value=10)
Coapplicant_Income_log = st.number_input("Coapplicant Income (log)", value=8)

# ===== FINAL DATA (ORDER MATCHED) =====
data = np.array([[
    Age, Dependents, Existing_Loans, Savings, Collateral_Value,
    Loan_Amount, Loan_Term, Education_Level,
    Employment_Status_Salaried, Employment_Status_Self_employed, Employment_Status_Unemployed,
    Marital_Status_Single,
    Loan_Purpose_Car, Loan_Purpose_Education, Loan_Purpose_Home, Loan_Purpose_Personal,
    Property_Area_Semiurban, Property_Area_Urban,
    Gender_Male,
    Employer_Category_Government, Employer_Category_MNC,
    Employer_Category_Private, Employer_Category_Unemployed,
    DTI_Ratio_sq, Credit_Score_sq,
    Applicant_Income_log, Coapplicant_Income_log
]])

# ===== PREDICTION =====
if st.button("Predict"):
    try:
        data_scaled = scaler.transform(data)
        result = model.predict(data_scaled)

        if result[0] == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

    except Exception as e:
        st.error(f"Error: {e}")