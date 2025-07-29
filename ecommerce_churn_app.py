# import library
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Judul Utama
st.title('Customer Churn Predictor')
st.text('This web predicts the likelihood of a customer churning')

# Menambahkan sidebar
st.sidebar.header("Please input customer features")

def create_user_input():
    # Numerical Features
    Tenure = st.sidebar.slider('Tenure (in months)', min_value=0, max_value=61, value=12)
    WarehouseToHome = st.sidebar.slider('Warehouse To Home (in km)', min_value=5, max_value=126, value=20)
    HourSpendOnApp = st.sidebar.slider('Hour Spent on App (per day)', min_value=0.0, max_value=5.0, value=1.0)
    NumberOfDeviceRegistered = st.sidebar.slider('Number of Devices Registered', min_value=1, max_value=6, value=2)
    OrderAmountHikeFromlastYear = st.sidebar.slider('Order Amount Hike From Last Year (%)', min_value=11, max_value=26, value=15)
    CouponUsed = st.sidebar.slider('Coupons Used', min_value=0, max_value=16, value=3)
    OrderCount = st.sidebar.slider('Order Count', min_value=1, max_value=16, value=5)
    DaySinceLastOrder = st.sidebar.slider('Days Since Last Order', min_value=0, max_value=46, value=10)
    CashbackAmount = st.sidebar.slider('Cashback Amount', min_value=0.0, max_value=324.99, value=50.0)

    # Categorical Features
    # Categorical Features
    PreferredLoginDevice = st.sidebar.selectbox('Preferred Login Device', ['Mobile Phone', 'Computer'])
    CityTier = st.sidebar.selectbox('City Tier', ['1', '2', '3'])
    PreferredPaymentMode = st.sidebar.selectbox('Preferred Payment Mode', ['Debit Card', 'Credit Card', 'E wallet', 'Cash on Delivery', 'UPI'])
    Gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
    PreferedOrderCat = st.sidebar.selectbox('Preferred Order Category', ['Mobile Phone', 'Laptop & Accessory', 'Grocery', 'Fashion', 'Others'])
    SatisfactionScore = st.sidebar.selectbox('Satisfaction Score', ['1', '2', '3', '4', '5'])
    MaritalStatus = st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
    Complain = st.sidebar.radio('Customer Complaint?', [0, 1])  # 0 = no complain, 1 = complain
    CountOfAddress = st.sidebar.selectbox('Count of Address', ['1–2', '3', '4–6', '7+'])

    
    # Creating a dictionary with user input
    user_data = {
        'Tenure': Tenure,
        'WarehouseToHome': WarehouseToHome,
        'HourSpendOnApp': HourSpendOnApp,
        'NumberOfDeviceRegistered': NumberOfDeviceRegistered,
        'OrderAmountHikeFromlastYear': OrderAmountHikeFromlastYear,
        'CouponUsed': CouponUsed,
        'OrderCount': OrderCount,
        'DaySinceLastOrder': DaySinceLastOrder,
        'CashbackAmount': CashbackAmount,
        'PreferredLoginDevice': PreferredLoginDevice,
        'CityTier': CityTier,
        'PreferredPaymentMode': PreferredPaymentMode,
        'Gender': Gender,
        'PreferedOrderCat': PreferedOrderCat,
        'SatisfactionScore': SatisfactionScore,
        'MaritalStatus': MaritalStatus,
        'Complain': Complain,
        'CountOfAddress': CountOfAddress
    }

    # Convert the dictionary into a pandas DataFrame (for a single row)
    user_data_df = pd.DataFrame([user_data])
    return user_data_df

# Get customer data
data_customer = create_user_input()

# Membuat 2 kontainer
col1, col2 = st.columns(2)

# Kiri
with col1:
    st.subheader("Customer's Features")
    st.write(data_customer.transpose())

# Load model
with open('final_tuned_lightgbm_ros_selectkbest.pkl', 'rb') as f:
    model_loaded = pickle.load(f)

# Predict to data
kelas = model_loaded.predict(data_customer)
probability = model_loaded.predict_proba(data_customer)[0]  # Get the probabilities

# Bagian kanan (col2)
with col2:
    st.subheader('Prediction Result')
    if kelas[0] == 1:
        st.write('Churn: Yes – This customer is likely to churn.')
    else:
        st.write('Churn: No – This customer is likely to stay.')
    
    # Displaying the probability of churn
    st.write(f"Probability of Churn: {probability[1]:.2f}")
