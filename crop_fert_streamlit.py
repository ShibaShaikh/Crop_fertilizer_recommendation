import numpy as np
import streamlit as st
import joblib  # for save and load model
from sklearn.preprocessing import StandardScaler

# load model
crop_model = joblib.load('crop-rec.joblib')
crop_scaler = joblib.load('crop-scaler.joblib') 
fertilizer_model = joblib.load('fertilizer-rec.joblib')
fertilizer_scaler = joblib.load('fertilizer-scaler.joblib') 

# crop recommendation function
def reco(N, P, K, temp, hum, ph, rain):
    features = np.array([[N, P, K, temp, hum, ph, rain]])
    transformed_features = crop_scaler.transform(features)
    prediction = crop_model.predict(transformed_features).reshape(1, -1)
    crop_dict = {
    1 : 'rice', 2 : 'maize', 3 : 'chickpea', 4 : 'kidneybeans', 5 : 'pigeonpeas', 6 : 'mothbeans', 7 : 'mungbean', 8 : 'blackgram', 9 : 'lentil',
    10 : 'pomegranate', 11 : 'banana', 12 : 'mango', 13 : 'grapes', 14 : 'watermelon', 15 : 'muskmelon', 16 : 'apple', 17 : 'orange', 18 : 'papaya',
    19 : 'coconut', 20 : 'cotton', 21 : 'jute', 22 : 'coffee'
    }
    crop = crop_dict[prediction[0][0]]

    return f"{crop} is a best crop to grow" 

# fertilizer recommendation function
def recommend_fertilizer(Temp, Hum, Moisture, soil_type, crop_type, N, P, Phosphorous):
    soil_dict = {
        'Black': 0,
        'Clayey': 1,
        'Loamy': 2,
        'Red': 3,
        'Sandy': 4,
        'Loamy': 2
    }
    soil_type_value = soil_dict.get(soil_type)
    
    crop_dict = { 
        'Barley' : 0,         
        'Cotton' : 1, 
        'Ground Nuts' : 2,
        'Maize' : 3,
        'Millets' : 4,
        'Oil seeds' : 5,
        'Paddy' : 6, 
        'Pulses' : 7,
        'Sugarcane' : 8,
        'Tobacco' : 9, 
        'Wheat' : 10
    }
    crop_type_value = crop_dict.get(crop_type)
    
    features = np.array([[Temp, Hum, Moisture, soil_type_value, crop_type_value, N, P, Phosphorous]])
    transformed_features = fertilizer_scaler.transform(features)
    prediction = fertilizer_model.predict(transformed_features).reshape(1,-1)
    fert_dict = {1: 'Urea', 2: 'DAP', 3: '14-35-14', 4: '28-28', 5: '17-17-17', 6: '20-20', 7: '10-26-26'}
    fertilizer = fert_dict[prediction[0][0]]
    
    return f"{fertilizer} is a best fertilizer for the given conditions" 


# Streamlit app layout
st.title("Agriculture Recommendation System")

recommendation_type = st.radio("Which recommendation would you like?", ("Crop Recommendation", "Fertilizer Recommendation"))

if recommendation_type == "Crop Recommendation":
    st.header("Crop Recommendation")
    # Input fields for user input
    N = st.number_input("Enter Nitrogen (N) value", min_value=0)
    P = st.number_input("Enter Phosphorus (P) value", min_value=0)
    K = st.number_input("Enter Potassium (K) value", min_value=0)
    temp = st.number_input("Enter Temperature value")
    hum = st.number_input("Enter Humidity value", min_value=0.0, max_value=100.0)
    ph = st.number_input("Enter pH value", min_value=0.0, max_value=14.0)
    rain = st.number_input("Enter Rainfall value", min_value=0)
    # Button to trigger the recommendation
    if st.button("Get Crop Recommendation"):
        result = reco(N, P, K, temp, hum, ph, rain)
        st.write(result)
    
else:
    st.header("Fertilizer Recommendation")
    # Input fields for user input
    temp = st.number_input("Enter Temperature value")
    hum = st.number_input("Enter Humidity value", min_value=0.0, max_value=100.0)
    moisture = st.number_input("Enter Moisture value")
    soil_type = st.selectbox("Enter Soil Type", ['Select', 'Loamy', 'Clay', 'Sandy', 'Peaty', 'Saline'])
    crop_type = st.selectbox("Enter Crop Type", ['Select', 'Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts'])
    N = st.number_input("Enter Nitrogen (N) value", min_value=0)
    K = st.number_input("Enter Potassium (K) value", min_value=0)
    P = st.number_input("Enter Phosphorus (P) value", min_value=0)
    
    # Button to trigger the recommendation
    if st.button("Get Crop Recommendation"):
        result = recommend_fertilizer(temp, hum, moisture, soil_type, crop_type, N, K, P)
        st.write(result)
