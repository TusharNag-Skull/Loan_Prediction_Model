


import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Loan Approval Prediction", layout="wide")

# Load model
model = joblib.load('model.pkl')

# Sidebar info
st.sidebar.title("📌 How to Use")
st.sidebar.markdown("""
Enter all required details on the left panel.

Click **Predict** to see whether the loan will be **approved or rejected**.
""")

# Main title
st.title("🏦 Loan Approval Prediction")
st.markdown("Please fill in the following fields:")

# Feature inputs
gender = st.selectbox("Gender", ("Male", "Female"))
married = st.selectbox("Married", ("Yes", "No"))
dependents = st.selectbox("Dependents", ("0", "1", "2", "3+"))
education = st.selectbox("Education", ("Graduate", "Not Graduate"))
self_employed = st.selectbox("Self Employed", ("Yes", "No"))

applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term", min_value=0)
credit_history = st.selectbox("Credit History", (1.0, 0.0))
property_area = st.selectbox("Property Area", ("Urban", "Semiurban", "Rural"))

# Preprocessing user input
dependents = 4 if dependents == '3+' else int(dependents)
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
property_area_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
property_area = property_area_map[property_area]

input_features = np.array([[gender, married, dependents, education, self_employed,
                            applicant_income, coapplicant_income, loan_amount,
                            loan_amount_term, credit_history, property_area]])

# Session state tracking for predictions
if 'approved_count' not in st.session_state:
    st.session_state.approved_count = 0
if 'rejected_count' not in st.session_state:
    st.session_state.rejected_count = 0

# Prediction
if st.button("🚀 Predict Loan Status"):
    prediction = model.predict(input_features)[0]
    
    if prediction == 1:
        st.success("✅ Loan is **Approved**.")
        st.session_state.approved_count += 1
    else:
        st.error("❌ Loan is **Rejected**.")
        st.session_state.rejected_count += 1

    # Pie chart of predictions
    st.markdown("### 📊 Prediction Summary")
    labels = ['Approved', 'Rejected']
    sizes = [st.session_state.approved_count, st.session_state.rejected_count]
    colors = ['#2ecc71', '#e74c3c']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')
    st.pyplot(fig)

