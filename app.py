
import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('logi.sav')

# Define the feature names (ensure they match the training data order)
feature_names = ['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
                 'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
                 'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
                 'Warehouse_Processing_Time']

st.title('Delivery Delay Prediction App')
st.write('Enter the values for the delivery parameters to predict if there will be a delay.')

# Create input fields for each feature
input_data = {}
for feature in feature_names:
    if feature == 'Delivery_Distance':
        input_data[feature] = st.slider(f'Enter {feature}', min_value=0.0, max_value=100.0, value=20.0, step=0.1)
    elif feature == 'Package_Weight':
        input_data[feature] = st.slider(f'Enter {feature}', min_value=0.0, max_value=200.0, value=120.0, step=0.1)
    elif feature == 'Fuel_Efficiency':
        input_data[feature] = st.slider(f'Enter {feature}', min_value=0.0, max_value=30.0, value=12.0, step=0.1)
    elif feature == 'Warehouse_Processing_Time':
        input_data[feature] = st.slider(f'Enter {feature}', min_value=0, max_value=300, value=120)
    elif feature in ['Traffic_Congestion', 'Weather_Condition', 'Delivery_Slot', 'Road_Condition_Score']:
        input_data[feature] = st.slider(f'Enter {feature} (1-5 scale)', min_value=1, max_value=5, value=3)
    elif feature in ['Driver_Experience', 'Num_Stops', 'Vehicle_Age']:
        input_data[feature] = st.slider(f'Enter {feature}', min_value=0, max_value=20, value=5)
    else:
        input_data[feature] = st.number_input(f'Enter {feature}', value=0.0)

# Predict button
if st.button('Predict Delivery Delay'):
    # Convert input data to a numpy array in the correct order
    features_array = np.array([input_data[feature] for feature in feature_names]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features_array)
    prediction_proba = model.predict_proba(features_array)
    
    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error('Prediction: \ud83d\udea8 **Delay is predicted!**')
    else:
        st.success('Prediction: \ud83d\ude9a **No Delay is predicted.**')
        
    st.write(f"Probability of No Delay: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Delay: {prediction_proba[0][1]:.2f}")
