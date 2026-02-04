import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="RainTomorrow Prediction", page_icon="🌧️", layout="wide")


# LOAD PIPELINE
@st.cache_resource
def load_pipeline():
    return joblib.load("models/best_rain_pipeline.joblib")


pipe = load_pipeline()

# kolumny, których pipeline oczekuje na wejściu (po fit)
expected_cols = pipe.named_steps["preprocessor"].feature_names_in_

# UI
st.title("🌧️ RainTomorrow Prediction — Melbourne")
st.write("Predykcja: czy jutro będzie padać.")

st.divider()
st.header("Dane wejściowe (podstawowe)")

col1, col2, col3 = st.columns(3)

with col1:
    min_temp = st.number_input(
        "MinTemp (°C)", min_value=-10.0, max_value=50.0, value=12.0, step=0.1
    )
    max_temp = st.number_input(
        "MaxTemp (°C)", min_value=-10.0, max_value=60.0, value=22.0, step=0.1
    )
    rainfall = st.number_input(
        "Rainfall (mm)", min_value=0.0, max_value=300.0, value=0.0, step=0.1
    )

with col2:
    humidity3pm = st.number_input(
        "Humidity3pm (%)", min_value=0.0, max_value=100.0, value=55.0, step=1.0
    )
    pressure3pm = st.number_input(
        "Pressure3pm (hPa)", min_value=950.0, max_value=1100.0, value=1015.0, step=0.1
    )
    cloud3pm = st.number_input(
        "Cloud3pm (oktas)", min_value=0.0, max_value=9.0, value=4.0, step=1.0
    )

with col3:
    wind_gust_speed = st.number_input(
        "WindGustSpeed (km/h)", min_value=0.0, max_value=200.0, value=35.0, step=1.0
    )
    wind_speed_3pm = st.number_input(
        "WindSpeed3pm (km/h)", min_value=0.0, max_value=150.0, value=20.0, step=1.0
    )
    rain_today = st.selectbox("RainToday", options=["No", "Yes"], index=0)

with st.expander("Dodatkowe pola (opcjonalnie)"):
    c1, c2, c3 = st.columns(3)

    wind_dirs = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]

    with c1:
        wind_dir_3pm = st.selectbox("WindDir3pm", options=wind_dirs, index=10)
        wind_dir_9am = st.selectbox("WindDir9am", options=wind_dirs, index=10)

    with c2:
        wind_gust_dir = st.selectbox("WindGustDir", options=wind_dirs, index=10)
        wind_speed_9am = st.number_input(
            "WindSpeed9am (km/h)", min_value=0.0, max_value=150.0, value=15.0, step=1.0
        )

    with c3:
        temp9am = st.number_input(
            "Temp9am (°C)", min_value=-10.0, max_value=60.0, value=16.0, step=0.1
        )
        pressure9am = st.number_input(
            "Pressure9am (hPa)",
            min_value=950.0,
            max_value=1050.0,
            value=1017.0,
            step=0.1,
        )

st.divider()


# BUILD INPUT DF
def build_input_df():
    row = {c: np.nan for c in expected_cols}

    if "Location" in row:
        row["Location"] = "Melbourne"

    # pola podstawowe
    if "MinTemp" in row:
        row["MinTemp"] = float(min_temp)
    if "MaxTemp" in row:
        row["MaxTemp"] = float(max_temp)
    if "Rainfall" in row:
        row["Rainfall"] = float(rainfall)
    if "Humidity3pm" in row:
        row["Humidity3pm"] = float(humidity3pm)
    if "Pressure3pm" in row:
        row["Pressure3pm"] = float(pressure3pm)
    if "Cloud3pm" in row:
        row["Cloud3pm"] = float(cloud3pm)
    if "WindGustSpeed" in row:
        row["WindGustSpeed"] = float(wind_gust_speed)
    if "WindSpeed3pm" in row:
        row["WindSpeed3pm"] = float(wind_speed_3pm)
    if "RainToday" in row:
        row["RainToday"] = str(rain_today)

    # dodatkowe (jeśli takie kolumny są w danych)
    if "WindDir3pm" in row:
        row["WindDir3pm"] = str(wind_dir_3pm)
    if "WindDir9am" in row:
        row["WindDir9am"] = str(wind_dir_9am)
    if "WindGustDir" in row:
        row["WindGustDir"] = str(wind_gust_dir)
    if "WindSpeed9am" in row:
        row["WindSpeed9am"] = float(wind_speed_9am)
    if "Temp9am" in row:
        row["Temp9am"] = float(temp9am)
    if "Pressure9am" in row:
        row["Pressure9am"] = float(pressure9am)

    return pd.DataFrame([row], columns=expected_cols)


# PREDICT
if st.button("🔍 Predict RainTomorrow", type="primary", width="stretch"):
    input_df = build_input_df()

    # pipeline robi preprocessing + model
    proba = pipe.predict_proba(input_df)[0]
    p_rain = float(proba[1])
    pred = int(p_rain >= 0.5)

    st.header("Wynik")
    c1, c2 = st.columns(2)

    with c1:
        if pred == 1:
            st.error("🌧️ JUTRO PRZEWIDUJĘ DESZCZ")
        else:
            st.success("☀️ JUTRO PRZEWIDUJĘ BRAK DESZCZU")

    with c2:
        st.metric("Prawdopodobieństwo deszczu", f"{p_rain:.1%}")
        st.metric("Prawdopodobieństwo braku deszczu", f"{float(proba[0]):.1%}")

    st.subheader("Dane wejściowe przekazane do pipeline")
    st.dataframe(input_df, use_container_width=True)

st.caption("Model: LightGBM + preprocessing w jednym pipeline (joblib).")
