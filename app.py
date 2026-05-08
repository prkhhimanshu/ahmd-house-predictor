import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading saved files like model,location encoding,colums

model = joblib.load('house_price_model.pkl')
loc_mean = joblib.load('location_encoding.pkl')
model_columns = joblib.load('model_columns.pkl')

#App title

st.title("🏠 Ahmedabad House Price Predictor")

#Making inpute buttons

location = st.selectbox("Select location", list(loc_mean.index))
bhk = st.selectbox("Select BHK", [1, 2, 3, 4])
area = st.number_input("Select Total Area (sqft)", min_value=500, max_value=5000, value=1000)
floor = st.selectbox("Floor Level", ['Ground', '1st', '2nd', '3rd', '4th', '5th+'])
amenities = 5
age = st.number_input("Select Property Age (years)", min_value=0, max_value=50, value=5)
total_floors = st.number_input("Total Floors in Building", min_value=1, max_value=30, value=5)
parking = st.selectbox("It Has Parking?", ['Yes', 'No'])
balcony = st.selectbox("Select Balcony View You want", ['Park', 'Road', 'Lake'])
condition = st.selectbox("Select Condition of House", ['New', 'Resale', 'Under Construction'])
furnishing = st.selectbox("Select How Much Furnishing You want", ['Fully-Furnished', 'Semi-Furnished', 'Unfurnished'])
year = st.number_input("Year of Sale of The House", min_value=2020, max_value=2030, value=2024)

#making prediction button


if st.button("Predict Price"):
    floor_map = {'Ground':0,'1st':1,'2nd':2,'3rd':3,'4th':4,'5th+':5}

    input_data = {
        'Location': loc_mean[location],
        'BHK': bhk,
        'Total_Area_Sqft': area,
        'Floor_Level': floor_map[floor],
        'Nearby_Amenities_Score': 5,
        'Property_Age': age,
        'Total_Floors': total_floors,
        'Has_Parking': 1 if parking == 'Yes' else 0,
        'Year_Of_Sale': year,
        'Condition_Resale': 1 if condition == 'Resale' else 0,
        'Condition_Under Construction': 1 if condition == 'Under Construction' else 0,
        'Furnishing_Semi-Furnished': 1 if furnishing == 'Semi-Furnished' else 0,
        'Furnishing_Unfurnished': 1 if furnishing == 'Unfurnished' else 0,
        'Balcony_View_Park': 1 if balcony == 'Park' else 0,
        'Balcony_View_Road': 1 if balcony == 'Road' else 0,
    }
#converting input data and model columns into data frame
    
    input_df = pd.DataFrame([input_data])[model_columns]
    log_price = model.predict(input_df)[0]
    price = np.exp(log_price) #converting log(mean value prices) price value into real price
    st.success(f"🏠 Estimated Price: ₹{price:,.0f}")
    st.info(f"That is approximately ₹{price/100000:.1f} Lakhs")

#making chart

    st.subheader("📊 Where Your Price Falls")
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot([0, 16000000], [0, 0], color='#BDC3C7', linewidth=6, label='Price Scale')
    ax.plot(price, 0, 'o', color='#1A252F', markersize=18, label=f'Your House: ₹{price/100000:.1f}L')
    ax.set_xlim(0, 16000000)
    ax.set_xticks([0, 5000000, 10000000, 16000000])
    ax.set_xticklabels(['₹0', '₹50L', '₹1Cr', '₹1.6Cr'])
    ax.set_yticks([])
    ax.set_title(f"Your Price: ₹{price/100000:.1f}L")
    ax.legend(loc='upper right')
    st.pyplot(fig)