import streamlit as st
import pandas as pd
import joblib

# ── Load trained artifacts ───────────────────────────────────────────
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Predictor")
st.write("Fill in the customer's details to predict their likelihood of churning.")

# ── Input form ────────────────────────────────────────────────────────
with st.form("churn_form"):
    st.subheader("Customer Profile")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

    with col2:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0, step=0.5)
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])

    st.subheader("Services")
    col3, col4 = st.columns(2)

    with col3:
        multiple_lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])

    with col4:
        device_protection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"])

    st.subheader("Account")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

    submitted = st.form_submit_button("Predict Churn")

# ── Build feature row on submit ──────────────────────────────────────
if submitted:
    # Start with every model column at 0
    row = {col: 0 for col in model_columns}

    # Numeric features (scaled below)
    row["SeniorCitizen"] = 1 if senior_citizen == "Yes" else 0
    row["tenure"] = tenure
    row["MonthlyCharges"] = monthly_charges
    row["TotalCharges"] = total_charges

    # One-hot flags — only set the ones that exist as columns (drop_first=True
    # means the "reference" category is implicitly 0 for all dummy columns)
    def set_flag(col_name):
        if col_name in row:
            row[col_name] = 1

    if gender == "Male":
        set_flag("gender_Male")
    if partner == "Yes":
        set_flag("Partner_Yes")
    if dependents == "Yes":
        set_flag("Dependents_Yes")
    if phone_service == "Yes":
        set_flag("PhoneService_Yes")

    if multiple_lines == "No phone service":
        set_flag("MultipleLines_No phone service")
    elif multiple_lines == "Yes":
        set_flag("MultipleLines_Yes")

    if internet_service == "Fiber optic":
        set_flag("InternetService_Fiber optic")
    elif internet_service == "No":
        set_flag("InternetService_No")

    for label, prefix in [
        (online_security, "OnlineSecurity"),
        (online_backup, "OnlineBackup"),
        (device_protection, "DeviceProtection"),
        (tech_support, "TechSupport"),
        (streaming_tv, "StreamingTV"),
        (streaming_movies, "StreamingMovies"),
    ]:
        if label == "No internet service":
            set_flag(f"{prefix}_No internet service")
        elif label == "Yes":
            set_flag(f"{prefix}_Yes")

    if contract == "One year":
        set_flag("Contract_One year")
    elif contract == "Two year":
        set_flag("Contract_Two year")

    if paperless_billing == "Yes":
        set_flag("PaperlessBilling_Yes")

    if payment_method == "Credit card (automatic)":
        set_flag("PaymentMethod_Credit card (automatic)")
    elif payment_method == "Electronic check":
        set_flag("PaymentMethod_Electronic check")
    elif payment_method == "Mailed check":
        set_flag("PaymentMethod_Mailed check")
    # "Bank transfer (automatic)" is the drop_first reference category → all zeros

    # Assemble dataframe in the exact column order the model expects
    X_new = pd.DataFrame([row])[model_columns]

    # Scale numeric columns with the SAME fitted scaler used in training
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    X_new[numeric_features] = scaler.transform(X_new[numeric_features])

    # Predict
    proba = model.predict_proba(X_new)[:, 1][0]
    prediction = model.predict(X_new)[0]

    st.divider()
    st.subheader("Prediction")

    if prediction == 1:
        st.error(f"⚠️ Likely to churn — probability: {proba:.1%}")
    else:
        st.success(f"✅ Likely to stay — churn probability: {proba:.1%}")

    st.progress(float(min(proba, 1.0)))
