# Page: sub-question metro distance (Jurre).
# This is a STARTING POINT: it already works as soon as data/processed/metro_stations.parquet exists
# (columns: name, lat, lon). Replace/extend the sections marked "TODO Jurre".
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

import weather_impact as wi

st.set_page_config(page_title="Afstand tot metro", layout="wide")
RAIN = "#2a6fdb"


@st.cache_data
def load():
    stations = pd.read_parquet(wi.PROC / "stations.parquet")
    totals = pd.read_parquet(wi.PROC / "station_totals.parquet")
    metro_dist = wi.metro_distance_table(stations)
    metro = pd.read_parquet(wi.METRO_STATIONS_FILE) if wi.METRO_STATIONS_FILE.exists() else None
    return stations, totals, metro_dist, metro


stations, totals, metro_dist, metro = load()

st.title("🚇 Wordt de Bikeshare minder gebruikt dicht bij de metro?")

if metro_dist is None:
    st.info(
        "Nog geen metrodata. Sla de metrostations op als `data/processed/metro_stations.parquet` "
        "met de kolommen `name`, `lat` en `lon`. De afstand tot het dichtstbijzijnde metrostation "
        "wordt dan automatisch berekend (ook voor de weerpagina)."
    )
    st.code(
        '''# Example (in a notebook): doogal CSV -> metro_stations.parquet
import pandas as pd
m = pd.read_csv("data/raw/metro/London stations.csv")          # file from doogal.co.uk
m = m.rename(columns={"Station": "name", "Latitude": "lat", "Longitude": "lon"})
m[["name", "lat", "lon"]].to_parquet("data/processed/metro_stations.parquet", index=False)''',
        language="python",
    )
    st.stop()

df = totals.merge(metro_dist, on="bikepoint_id", how="inner").dropna(subset=["lat", "lon", "metro_distance_m"])

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Filters")
max_dist = st.sidebar.slider("Maximale afstand tot metro (m)", 100, int(df["metro_distance_m"].max()) + 100,
                             int(df["metro_distance_m"].max()) + 100, 50)
df = df[df["metro_distance_m"] <= max_dist]

# ---------------------------------------------------------------- KPIs
k1, k2, k3 = st.columns(3)
k1.metric("Fietsstations", f"{len(df)}")
k2.metric("Mediane afstand tot metro", f"{df['metro_distance_m'].median():.0f} m")
k3.metric("Gem. vertrekken per station per dag", f"{df['departures_per_day'].mean():.1f}")

# ---------------------------------------------------------------- usage per distance group
st.subheader("Gebruik per afstandsgroep")
g = (df.groupby("metro_group", observed=True)
       .agg(stations=("bikepoint_id", "size"), per_day=("departures_per_day", "mean")).reset_index())
g["text"] = g.apply(lambda r: f"{r['per_day']:.1f}  (n={r['stations']})", axis=1)
fig = px.bar(g, x="metro_group", y="per_day", text="text", height=380,
             labels={"metro_group": "Afstand tot dichtstbijzijnde metrostation", "per_day": "Gem. vertrekken per dag"})
fig.update_traces(marker_color=RAIN, textposition="outside", cliponaxis=False)
st.plotly_chart(fig)

rho, p = stats.spearmanr(df["metro_distance_m"], df["departures_per_day"])
st.markdown(f"**Spearman ρ = {rho:+.2f}** (p = {p:.3g}) tussen afstand tot de metro en gebruik.")

# ---------------------------------------------------------------- map
st.subheader("Kaart")
plain = st.checkbox("Kaart zonder achtergrond (werkt zonder internet)")
common = dict(color="metro_group", size="departures_per_day", size_max=14, hover_name="name",
              hover_data={"metro_distance_m": ":.0f", "departures_per_day": ":.1f"}, height=600,
              labels={"metro_group": "Afstand", "metro_distance_m": "Afstand (m)",
                      "departures_per_day": "Vertrekken/dag"},
              color_discrete_sequence=px.colors.sequential.Blues_r[1:])
if plain:
    fig = px.scatter(df, x="lon", y="lat", **common)
    fig.update_yaxes(scaleanchor="x", scaleratio=1.6, visible=False)
    fig.update_xaxes(visible=False)
    metro_trace = go.Scatter(x=metro["lon"], y=metro["lat"], mode="markers", name="Metrostation",
                             marker=dict(size=7, color="#d62728", symbol="square"), text=metro["name"],
                             hoverinfo="text") if metro is not None else None
else:
    try:
        fig = px.scatter_map(df, lat="lat", lon="lon", zoom=11, map_style="carto-positron", **common)
        cls = go.Scattermap
    except AttributeError:
        fig = px.scatter_mapbox(df, lat="lat", lon="lon", zoom=11, mapbox_style="carto-positron", **common)
        cls = go.Scattermapbox
    metro_trace = cls(lat=metro["lat"], lon=metro["lon"], mode="markers", name="Metrostation",
                      marker=dict(size=8, color="#d62728"), text=metro["name"],
                      hoverinfo="text") if metro is not None else None
if metro_trace is not None:
    fig.add_trace(metro_trace)
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.02))
st.plotly_chart(fig)

# ---------------------------------------------------------------- TODO Jurre
st.subheader("Verdere analyse")
st.caption("TODO Jurre: voeg hier je eigen analyses toe, bijvoorbeeld: drukste stations per groep, "
           "balans (aankomst − vertrek) per groep, of het verband met het aantal docks.")
