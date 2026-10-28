# Main file of the dashboard: defines the pages and the navigation.
# Run with:  streamlit run Start.py
import streamlit as st

st.set_page_config(page_title="Bikeshare Londen", page_icon="🚲", layout="wide")

# Define the pages
introduction = st.Page("app_pages/page_1_introduction.py", title="Introduction", icon="🚲", default=True)
data_preparation = st.Page("app_pages/page_2_data_preparation.py", title="Data Preparation", icon="🧹")
distance = st.Page("app_pages/page_3_distance_metro.py", title="Distance from Metro", icon="🚇")
weather = st.Page("app_pages/page_4_weather.py", title="Weather vs Rental", icon="🌧️")
conclusion = st.Page("app_pages/page_5_conclusion.py", title="Conclusion", icon="✅")

# Set up navigation
pg = st.navigation([introduction, data_preparation, distance, weather, conclusion])

# Run the selected page
pg.run()
