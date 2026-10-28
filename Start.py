# Start page of the team dashboard.
# Run the whole dashboard with:  streamlit run Start.py
# Every .py file in the pages/ folder automatically becomes a page in the sidebar.
import json

import pandas as pd
import plotly.express as px
import streamlit as st

import weather_impact as wi

st.set_page_config(page_title="Bikeshare Londen", page_icon="🚲", layout="wide")

RAIN = "#2a6fdb"


# ---------------------------------------------------------------- data loading (cached)
@st.cache_data
def load_overview():
    cfg = json.loads((wi.PROC / "config.json").read_text())
    daily = pd.read_parquet(wi.PROC / "daily.parquet")
    stations = pd.read_parquet(wi.PROC / "stations.parquet")
    cleaning = pd.read_csv(wi.PROC / "cleaning_log.csv")
    weather_log = pd.read_csv(wi.PROC / "weather_log.csv") if (wi.PROC / "weather_log.csv").exists() else None
    return cfg, daily, stations, cleaning, weather_log


cfg, daily, stations, cleaning, weather_log = load_overview()

# ---------------------------------------------------------------- research question
st.title("🚲 Bikeshare in Londen: zorgt regen voor minder ritten?")
st.markdown(
    """
**Hoofdvraag:** Zorgt regen ervoor dat de Bikeshare minder gebruikt wordt?

**Subvragen**
- **Afstand tot de metro** *(Jurre)*: wordt de Bikeshare minder gebruikt bij stations dicht bij een metrostation?
- **Weer** *(Jesse)*: waar en bij wie heeft regen de meeste impact, en verliezen stations dicht bij de metro meer ritten als het regent?

👈 Kies links een pagina.
"""
)

# ---------------------------------------------------------------- key figures
k1, k2, k3, k4 = st.columns(4)
start, end = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
full_year = start.year == end.year and (start.month, start.day, end.month, end.day) == (1, 1, 12, 31)
k1.metric("Periode", str(start.year) if full_year else f"{start:%d-%m-%y} – {end:%d-%m-%y}")
k2.metric("Ritten (na opschonen)", f"{daily['rides'].sum():,.0f}".replace(",", "."))
k3.metric("Dagen", f"{len(daily)}")
k4.metric("Stations", f"{len(stations)}")

# ---------------------------------------------------------------- data sources
st.header("Data")
st.table(pd.DataFrame([
    {"Dataset": "Bikeshare-ritten", "Bron": "TfL cycling.data.tfl.gov.uk (usage-stats)",
     "Inhoud": "Eén rij per rit: start/eind, stations, duur"},
    {"Dataset": "Stations", "Bron": "TfL BikePoint API",
     "Inhoud": "Locatie (lat/lon), aantal docks, ID's"},
    {"Dataset": "Weer", "Bron": "Meteostat (via Brightspace)",
     "Inhoud": "Per dag: temperatuur, neerslag, wind, luchtdruk"},
    {"Dataset": "Metro", "Bron": "Kaggle + doogal.co.uk",
     "Inhoud": "Metrostations met locatie"},
]))

# ---------------------------------------------------------------- cleaning
st.header("Opschonen")
st.caption("Elke stap is gelogd: hoeveel ritten eruit gingen en waarom.")
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

if weather_log is not None:
    st.subheader("Weerdata")
    st.dataframe(weather_log.rename(columns={"step": "Stap", "count": "Aantal"}), hide_index=True)

# ---------------------------------------------------------------- first look: rides through the year
st.header("Eerste blik: ritten per dag")
fig = px.line(daily.sort_values("date"), x="date", y="rides", height=320,
              labels={"date": "", "rides": "Ritten per dag"})
fig.update_traces(line=dict(color=RAIN, width=1.5))
fig.update_layout(hovermode="x unified", margin=dict(t=10))
st.plotly_chart(fig)
st.caption("Duidelijk seizoenspatroon: daarom vergelijken we regendagen steeds met droge dagen in dezelfde maand.")
