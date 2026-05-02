import streamlit as st
import joblib
import pandas as pd


st.set_page_config(
    page_title="power plant energy predictor"
)
st.title("power plant energy predictor")


# Load the model
#pip install ucimlrepo
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
# fetch dataset 
combined_cycle_power_plant = fetch_ucirepo(id=294) 
  
# data (as pandas dataframes) 
X = combined_cycle_power_plant.data.features 
y = combined_cycle_power_plant.data.targets 
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# build random forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5, min_samples_leaf=2, max_features='sqrt')
rf.fit(X_train, y_train.values.ravel())
# predict the model
pred = rf.predict(X_test)
# evaluate the model
mse = mean_squared_error(y_test, pred)
print("mean squared error", mse)
joblib.dump(rf, "power_plant_model.pkl")
model = joblib.load("power_plant_model.pkl")
col1, col2 = st.columns(2)
with col1:
    AT = st.number_input(
        "Ambient Temperature AT",
        min_value = -20.0,
        max_value = 50.0,
        value = 20.0,
        step = 0.1

    )
    V = st.number_input(
        "Exhaust Vacuum V",
        min_value = 25.0,
        max_value = 100.0,
        value = 50.0,
        step = 0.1
    )
    with col2:
        AP = st.number_input(
            "Ambient Pressure AP",
            min_value = 900.0,
            max_value = 1040.0,
            value= 1013.0,
            step = 0.1
        )
        RH = st.number_input(
            "Relative Humidity RH",
            min_value = 25.0,
            max_value = 100.0,
            value = 73.0,
            step =0.1
        )
    if st.button("predict Energy for combined cycle power plant"):
        input_data = pd.DataFrame({
            "AT" : [AT],
            "V" : [V],
            "AP" : [AP],
            "RH" : [RH]
        })
        pred = model.predict(input_data)[0]
        st.markdown(f"Predicted Energy {pred:.2f} MWh")
