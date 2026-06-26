import joblib
import pandas as pd
import streamlit as st


preprocessor = CreditScorePreprocessor.load_preprocessor("preprocessor.pkl")
model = joblib.load("xgboost.pkl")

st.set_page_config(
    page_title="Credit Score Prediction",
    layout="wide"
)

st.title("Credit Score Prediction")


st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    customer_id = st.text_input("Customer ID")
    customer_name = st.text_input("Name")
    ssn = st.text_input("SSN")
    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=25,
    )
    occupation = st.selectbox(
        "Occupation",
        [
            "Journalist",
            "Teacher",
            "_______",
            "Developer",
            "Architect",
            "Doctor",
            "Media_Manager",
            "Accountant",
            "Entrepreneur",
            "Lawyer",
            "Writer",
            "Engineer",
            "Manager",
            "Musician",
            "Scientist",
            "Mechanic",
        ],
    )
    annual_income = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=0.0,
    )
    monthly_salary = st.number_input(
        "Monthly Inhand Salary",
        min_value=0.0,
        value=0.0,
    )
    month = st.selectbox(
        "Month",[
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",],
    )

with col2:
    num_bank_accounts = st.number_input(
        "Number of Bank Accounts",
        min_value=0,
        value=0,
    )
    num_credit_card = st.number_input(
        "Number of Credit Cards",
        min_value=0,
        value=0,
    )
    interest_rate = st.number_input(
        "Interest Rate",
        min_value=0.0,
        value=0.0,
    )
    num_of_loan = st.number_input(
        "Number of Loans",
        min_value=0,
        value=0,
    )
    st.subheader("Type of Loan")
    loan_options = [
        "Mortgage Loan",
        "Auto Loan",
        "Student Loan",
        "Personal Loan",
        "Payday Loan",
        "Credit-Builder Loan",
        "Debt Consolidation Loan",
        "Home Equity Loan",
    ]
    selected_loans = st.multiselect(
        "Select one or more loan types",
        options=loan_options,
    )
    delay_from_due_date = st.number_input(
        "Delay From Due Date",
        min_value=0,
        value=0,
    )
    num_delayed_payment = st.number_input(
        "Number of Delayed Payment",
        min_value=0,
        value=0,
    )

# Credit Information
st.header("Credit Information")
col3, col4 = st.columns(2)
with col3:
    changed_credit_limit = st.number_input(
        "Changed Credit Limit",
        value=0.0,
    )
    num_credit_inquiries = st.number_input(
        "Number of Credit Inquiries",
        min_value=0.0,
        value=0.0,
    )
    credit_mix = st.selectbox(
        "Credit Mix",
        [
            "Bad",
            "Standard",
            "Good",
        ],
    )
    outstanding_debt = st.number_input(
        "Outstanding Debt",
        value=0.0,
    )
    credit_utilization_ratio = st.number_input(
        "Credit Utilization Ratio",
        value=0.0,
    )
    history_year = st.number_input(
        "Credit History (Years)",
        min_value=0,
        value=0,
    )
    history_month = st.number_input(
        "Additional Months",
        min_value=0,
        max_value=11,
        value=0,
    )

with col4:
    payment_of_min_amount = st.selectbox(
        "Payment of Minimum Amount",
        [
            "No",
            "NM",
            "Yes",
        ],
    )
    total_emi = st.number_input(
        "Total EMI Per Month",
        value=0.0,
    )
    invested_monthly = st.number_input(
        "Amount Invested Monthly",
        value=0.0,
    )
    payment_behaviour = st.selectbox(
        "Payment Behaviour",[
            "Low_spent_Small_value_payments",
            "Low_spent_Medium_value_payments",
            "Low_spent_Large_value_payments",
            "High_spent_Small_value_payments",
            "High_spent_Medium_value_payments",
            "High_spent_Large_value_payments",],
    )
    monthly_balance = st.number_input(
        "Monthly Balance",
        value=0.0,
    )

# Prediction

if st.button("Predict Credit Score"):
    data = pd.DataFrame([{
        "Unnamed: 0": 0,
        "ID": "0x0000",
        "Customer_ID": customer_id,
        "Month": month,
        "Name": customer_name,
        "Age": age,
        "SSN": ssn,
        "Occupation": occupation,
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_salary,
        "Num_Bank_Accounts": num_bank_accounts,
        "Num_Credit_Card": num_credit_card,
        "Interest_Rate": interest_rate,
        "Num_of_Loan": num_of_loan,
        "Type_of_Loan": ", ".join(selected_loans),
        "Delay_from_due_date": delay_from_due_date,
        "Num_of_Delayed_Payment": num_delayed_payment,
        "Changed_Credit_Limit": changed_credit_limit,
        "Num_Credit_Inquiries": num_credit_inquiries,
        "Credit_Mix": credit_mix,
        "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": credit_utilization_ratio,
        "Credit_History_Age":
        f"{history_year} Years and {history_month} Months",
        "Payment_of_Min_Amount": payment_of_min_amount,
        "Total_EMI_per_month": total_emi,
        "Amount_invested_monthly": invested_monthly,
        "Payment_Behaviour": payment_behaviour,
        "Monthly_Balance": monthly_balance,
    }])

    X = preprocessor.transform(data)
    prediction = model.predict(X)
    label = preprocessor.label_encoder.inverse_transform(prediction)
    st.success(f"Predicted Credit Score : **{label[0]}**")
