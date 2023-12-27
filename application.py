# Import necessary libraries
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
import pickle

# Load the CSV file with the car data
car=pd.read_csv('Cleaned_Car_data.csv')

# Write the title of the application on the UI
st.write("""
# Used Car Resale Price Prediction
""")


companies = [
    "Hyundai", "Mahindra", "Ford", "Maruti", "Skoda", "Audi", "Toyota",
    "Renault", "Honda", "Datsun", "Mitsubishi", "Tata", "Volkswagen",
    "Chevrolet", "Mini", "BMW", "Nissan", "Hindustan", "Fiat", "Force",
    "Mercedes", "Land", "Jaguar", "Jeep", "Volvo"
]
car_models = {
    "Hyundai": ["Santro Xing", "Grand i10", "Eon", "i20 Sportz", "i20 Magna", "Creta", "i10 Magna", "i20 Active", "i20 Asta", "Verna Transform", "Verna Fluidic", "i20", "Fluidic Verna", "Creta 1.6", "i10", "Accent GLX", "Verna", "i10 Sportz", "Accent", "Verna 1.4", "Verna 1.6", "Santro AE", "Getz Prime", "Santro", "Getz", "i20 Select"],
    "Mahindra": ["Jeep CL550", "Scorpio SLE", "Scorpio S10", "Bolero DI", "Scorpio S4", "Scorpio VLX", "Quanto C8", "Scorpio", "XUV500 W8", "XUV500", "Bolero SLE", "Scorpio SLX", "Xylo E4", "Jeep MM", "Bolero Power", "KUV100 K8", "Scorpio 2.6", "TUV300 T4", "Thar CRDe", "Xylo E8", "Xylo D2", "Scorpio Vlx", "Xylo", "Scorpio W", "TUV300 T8", "Scorpio LX", "Xylo E8", "XUV500 W10", "KUV100"],
    "Ford": ["EcoSport Titanium", "Figo", "EcoSport Ambiente", "Fiesta", "Ikon 1.3", "Figo Diesel", "Ikon 1.6", "Figo Duratorq", "Figo Petrol", "Endeavor 4x4", "Fusion 1.4"],
    "Maruti": ["Suzuki Alto", "Suzuki Stingray", "Suzuki Swift", "Suzuki Wagon", "Suzuki Baleno", "Suzuki Vitara", "Suzuki Dzire", "Suzuki SX4", "Suzuki Ciaz", "Suzuki Zen", "Suzuki A", "Suzuki Ertiga", "Suzuki Celerio", "Suzuki S", "Suzuki 800", "Suzuki Versa", "Suzuki Esteem", "Suzuki Ritz", "Suzuki Omni", "Suzuki Maruti", "Suzuki Versa", "Suzuki Swift", "Suzuki Wagon", "Suzuki Baleno", "Suzuki Ritz", "Suzuki Esteem", "Suzuki Zen", "Suzuki 800", "Suzuki Omni", "Suzuki Alto", "Suzuki Celerio", "Suzuki Eeco", "Suzuki Estilo", "Maruti Suzuki", "Suzuki 800", "Suzuki Omni", "Suzuki S"],
    "Skoda": ["Fabia Classic", "Yeti Ambition", "Fabia 1.2L", "Fabia", "Laura", "Octavia Classic", "Superb 1.8"],
    "Audi": ["A8", "Q7", "A4 1.8", "A4 2.0", "A6 2.0", "Q3 2.0", "A3 Cabriolet"],
    "Toyota": ["Innova 2.0", "Etios GD", "Innova 2.5", "Etios", "Fortuner", "Etios", "Qualis", "Etios G", "Corolla Altis", "Fortuner 3.0", "Corolla H2"],
    "Renault": ["Lodgy 85", "Duster 110", "Duster 85", "Duster", "Kwid", "Duster 110PS", "Scala RxL", "Kwid RXT", "Duster RxL"],
    "Honda": ["City 1.5", "Amaze", "Amaze 1.5", "City", "City ZX", "Brio", "City VX", "Mobilio", "Amaze 1.2", "Jazz VX", "Jazz S", "Brio V", "City SV", "Mobilio S", "City SV", "City VX"],
    "Datsun": ["Redi GO", "GO T"],
    "Mitsubishi": ["Pajero Sport", "Lancer 1.8"],
    "Tata": ["Indigo eCS", "Nano Cx", "Sumo Victa", "Indigo LX", "Nano GenX", "Nano", "Nano LX", "Nano Lx", "Manza Aura", "Venture EX", "Nano"],
    "Volkswagen": ["Polo Highline", "Polo Comfortline", "Polo", "Vento Highline", "Polo Trendline", "Vento Comfortline", "Vento Konekt", "Polo Highline1.2L", "Jetta Highline", "Vento"],
    "Chevrolet": ["Spark LS", "Beat LT", "Spark LT", "Enjoy 1.4", "Spark LS", "Spark 1.0", "Beat Diesel", "Sail UVA", "Beat PS", "Sail 1.2", "Beat Diesel", "Enjoy"],
    "Mini": ["Cooper S"],
    "BMW": ["3 Series", "7 Series", "X1 xDrive20d", "X1 sDrive20d", "5 Series"],
    "Nissan": ["Micra XV", "Sunny", "Terrano XL", "Micra XL", "X Trail", "Sunny XL", "Micra XV"],
    "Hindustan": ["Motors Ambassador"],
    "Fiat": ["Punto Emotion", "Linea Emotion", "Petra ELX"],
    "Force": ["Motors Force", "One"],
    "Mercedes": ["Benz GLA", "Benz B", "Benz C", "A"],
    "Land": ["Rover Freelander"],
    "Jaguar": ["XE XE", "XF 2.2"],
    "Jeep": ["Wrangler Unlimited"],
    "Volvo": ["S80 Summum"]
}
fuel_types = ["Diesel", "Petrol", "LPG"]

# Function to make the prediction using the pre-trained model
def model_pred(car_model,company,year,driven,fuel_type):
    carModel = company + " " + car_model

    ## Load the pre-trained model using pickle
    with open("LinearRegressionModel.pkl", "rb") as file:
        reg_model = pickle.load(file)

    # Prepare the input features
    prediction=reg_model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([carModel,company,year,driven,fuel_type]).reshape(1, 5)))
    print(prediction)

    return str(np.round(prediction[0],2))
    


# Create two columns in the UI
col1, col2 = st.columns(2)

# Create a drop-down menu to select the car company
company = col1.selectbox("Select Company",
                            companies)

# Create a drop-down menu to select the car model
car_model = col2.selectbox("Select Car Model",
                                    car_models[company])

# Create a drop-down menu to select purchase year
year = col1.selectbox("Purchase year",
                        list(range(2023, 1990, -1)))



# Create a drop-down menu to select the fuel type
fuel_type = col2.selectbox("Select fuel type",
                            fuel_types)

# Create a slider to set the km driven
driven = col1.slider("Set the approximate kilometers that the car has driven",
                        1000, 100000, step=500)





# Create a button to trigger the prediction
if(st.button("Predict Price")):
    # Make the prediction
    price = model_pred(car_model,company,year,driven,fuel_type)
    
     # Display the result on the UI
    st.text("Predicted price of the car: ₹"+ str(price))



# Display date on the UI
st.write ("Today's date is:" ,datetime.now().date())

# # Create a bar chart (commented out)
st.title('Car Price Bar Chart')
st.bar_chart(data=car, x= 'name', y= 'Price')