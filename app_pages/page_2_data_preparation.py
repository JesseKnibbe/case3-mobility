# Page 2: how the data was cleaned (cleaning log, weather log, rides per day)
import pandas as pd
import plotly.express as px
import streamlit as st

import weather_impact as wi

RAIN = "#2a6fdb"


@st.cache_data
def load():
    daily = pd.read_parquet(wi.PROC / "daily.parquet")
    cleaning = pd.read_csv(wi.PROC / "cleaning_log.csv")
    weather_log = pd.read_csv(wi.PROC / "weather_log.csv") if (wi.PROC / "weather_log.csv").exists() else None
    return daily, cleaning, weather_log


daily, cleaning, weather_log = load()

st.title("🧹 Data Preparation")

# Steps
st.markdown(
    """
1. **Inladen:** 53 wekelijkse TfL-bestanden van 2022. TfL wisselde rond september 2022 van formaat
   (andere kolomnamen, datumnotatie en station-ID's); beide formaten zijn omgezet naar één tabel.
2. **Opschonen:** elke stap hieronder is gelogd: hoeveel ritten eruit gingen en waarom.
3. **Koppelen:** ritten zijn gekoppeld aan stationslocaties (TfL API) en aan het weer van die dag.
4. **Aggregeren:** ±11 miljoen ritten samengevat per dag, per uur en per station, zodat dit dashboard snel laadt.
"""
)

# Cleaning log
st.header("Opschonen van de ritten")
c1, c2 = st.columns([3, 2])
log = cleaning.rename(columns={"step": "Stap", "before": "Voor", "removed": "Verwijderd",
                               "after": "Na", "pct_removed": "% verwijderd"})
c1.dataframe(log, hide_index=True)
fig = px.bar(cleaning.sort_values("removed"), x="removed", y="step", orientation="h", height=300,
             labels={"removed": "Verwijderde ritten", "step": ""})
fig.update_traces(marker_color=RAIN, hovertemplate="%{y}<br>%{x:,} ritten<extra></extra>")
fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
c2.plotly_chart(fig)

kept = cleaning["after"].iloc[-1] / cleaning["before"].iloc[0] * 100
st.markdown(f"Van de **{cleaning['before'].iloc[0]:,}** ruwe ritten blijft **{kept:.1f}%** over.".replace(",", "."))
st.caption("In één weekbestand (6–12 juli 2022) ontbrak het eindstation-ID. Die ritten zijn niet verwijderd, "
           "maar via de stationsnaam gekoppeld.")

# Weather log
if weather_log is not None:
    st.header("Opschonen van de weerdata")
    st.dataframe(weather_log.rename(columns={"step": "Stap", "count": "Aantal"}), hide_index=True)

# Rides per day
st.header("Resultaat: ritten per dag")
fig = px.line(daily.sort_values("date"), x="date", y="rides", height=320,
              labels={"date": "", "rides": "Ritten per dag"})
fig.update_traces(line=dict(color=RAIN, width=1.5))
fig.update_layout(hovermode="x unified", margin=dict(t=10))
st.plotly_chart(fig)
st.caption("Duidelijk seizoenspatroon: daarom vergelijken we regendagen steeds met droge dagen in dezelfde maand.")
