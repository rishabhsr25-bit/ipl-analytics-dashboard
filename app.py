import streamlit as st
import pandas as pd
from src.data_loader import IPLDataLoader
from src.predictor import IPLPredictor

# Page Setup
st.set_page_config(page_title="IPL Analytics & Prediction Dashboard", layout="wide", page_icon="🏏")

st.title("🏏 IPL Analytics & Win Prediction Dashboard")
st.markdown("---")

# Initialize our backend components
data_loader = IPLDataLoader(matches_path="data/matches.csv", deliveries_path="data/deliveries.csv")
predictor = IPLPredictor()

# Sidebar Setup for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Historical Analysis", "Live Match Predictor", "Player Insights"])

# Sidebar Data Status Check
st.sidebar.markdown("---")
st.sidebar.subheader("Data Pipeline Status")
if data_loader.load_raw_data():
    st.sidebar.success("Database Connected Successfully!")
else:
    st.sidebar.warning("Awaiting CSV datasets in the /data folder.")

# Tab/Page Router Logic
if page == "Historical Analysis":
    st.header("📈 Historical Trends & Team Performance")
    st.info("Visualizations, toss impact metrics, and venue stats will render here.")

elif page == "Live Match Predictor":
    st.header("🔮 Real-Time Win Probability Predictor")
    st.info("Input match situations (runs needed, overs left, wickets down) to run model inferences.")

elif page == "Player Insights":
    st.header("📊 Batsman & Bowler Performance Matrix")
    st.info("Deep dive analytics for individual player metrics will render here.")