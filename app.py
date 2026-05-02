import streamlit as st
import joblib
import pandas as pd


st.set_page_config(
    page_title="power plant energy predictor"
)
st.title("power plant energy predictor")


# Load the model
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
