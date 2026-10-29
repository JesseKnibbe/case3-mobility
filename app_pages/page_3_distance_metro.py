# Page 3: sub-question distance to the metro (Jurre)
import streamlit as st

st.title("🚇 Distance from Metro")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.neighbors import BallTree

import weather_impact as wi

EARTH_RADIUS_M = 6_371_000
LABELS = ["Close (≤200 m)", "Medium (200-500 m)", "Far (>500 m)"]
COLORS = {"Metro station": "#1f1f1f", "Close (≤200 m)": "#2ca02c",
          "Medium (200-500 m)": "#ff7f0e", "Far (>500 m)": "#d62728"}


@st.cache_data
def load():
    # 1. Unique bike stations (with rentals per day) and metro stations
    stations = pd.read_parquet(wi.PROC / "stations.parquet")
    totals = pd.read_parquet(wi.PROC / "station_totals.parquet")[["bikepoint_id", "departures_per_day"]]
    bikes = (stations[["bikepoint_id", "name", "lat", "lon"]].dropna(subset=["lat", "lon"])
             .merge(totals, on="bikepoint_id", how="left").reset_index(drop=True))
    metro = (pd.read_csv(wi.METRO_RAW_CSV)[["station", "lat", "lon"]]
             .dropna(subset=["lat", "lon"]).rename(columns={"station": "name"}).reset_index(drop=True))

    # 2. Distance to the nearest metro station (haversine, in metres)
    tree = BallTree(np.radians(metro[["lat", "lon"]].values), metric="haversine")
    dist, idx = tree.query(np.radians(bikes[["lat", "lon"]].values), k=1)
    bikes["dist_to_metro_m"] = dist[:, 0] * EARTH_RADIUS_M
    bikes["nearest_metro"] = metro["name"].values[idx[:, 0]]

    # 3. Categories: <=200 close, 200-500 medium, >500 far
    bikes["metro_distance"] = pd.cut(bikes["dist_to_metro_m"], bins=[0, 200, 500, np.inf],
                                     labels=LABELS, include_lowest=True).astype(str)
    return bikes, metro


bikes, metro = load()

# -jurre- updating code for making a lineplot x=month, y=verhuringen, color=afstandscategorie van metro naar bike rental (close,medium,far)

@st.cache_data
def load_monthly(bikes: pd.DataFrame) -> pd.DataFrame:
    sd = pd.read_parquet(wi.PROC / "station_day.parquet")
    sd["date"] = pd.to_datetime(sd["date"])
    sd = sd[sd["date"].dt.year == 2022]

    # attach distance category per station
    sd = sd.merge(bikes[["bikepoint_id", "metro_distance"]],
                  on="bikepoint_id", how="inner")

    # total departures per month and category
    sd["month"] = sd["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (sd.groupby(["month", "metro_distance"], observed=True)["departures"]
                 .sum().reset_index())

    # normalise: per station, per day
    n_stations = bikes.groupby("metro_distance").size().rename("n_stations")
    monthly = monthly.merge(n_stations, on="metro_distance")
    monthly["days"] = monthly["month"].dt.days_in_month
    monthly["departures_per_station_day"] = (
        monthly["departures"] / (monthly["days"] * monthly["n_stations"])
    )
    return monthly

monthly = load_monthly(bikes)


st.title("🚇 Distance from Metro")
st.subheader("Wordt de Bikeshare minder gebruikt dicht bij de metro?")

# ---------------------------------------------------------------- key figures
counts = bikes["metro_distance"].value_counts()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Fietsstations", len(bikes))
k2.metric("Metrostations", len(metro))
k3.metric("Mediane afstand tot metro", f"{bikes['dist_to_metro_m'].median():.0f} m")
k4.metric("Binnen 200 m van de metro", f"{counts.get(LABELS[0], 0) / len(bikes):.0%}")

# ---------------------------------------------------------------- map (no internet needed)
st.markdown("**Fietsstations per afstand tot het dichtstbijzijnde metrostation**")
fig = go.Figure()
geo = wi.load_boroughs()   # borough outlines as background
for f in geo["features"]:
    polys = f["geometry"]["coordinates"]
    for poly in ([polys] if f["geometry"]["type"] == "Polygon" else polys):
        fig.add_trace(go.Scatter(x=[c[0] for c in poly[0]], y=[c[1] for c in poly[0]], mode="lines",
                                 line=dict(color="#d5d8dd", width=1), hoverinfo="skip", showlegend=False))
fig.add_trace(go.Scatter(x=metro["lon"], y=metro["lat"], mode="markers", name="Metro station",
                         marker=dict(color=COLORS["Metro station"], size=6, symbol="square"),
                         text=metro["name"], hoverinfo="text"))
for label in LABELS:
    b = bikes[bikes["metro_distance"] == label]
    fig.add_trace(go.Scatter(x=b["lon"], y=b["lat"], mode="markers", name=f"{label} (n={len(b)})",
                             marker=dict(color=COLORS[label], size=7, line=dict(width=0.5, color="white")),
                             text=[f"<b>{n}</b><br>{d:.0f} m tot {m}" for n, d, m in
                                   zip(b["name"], b["dist_to_metro_m"], b["nearest_metro"])], hoverinfo="text"))
fig.update_xaxes(visible=False, range=[bikes["lon"].min() - 0.01, bikes["lon"].max() + 0.01])
fig.update_yaxes(visible=False, range=[bikes["lat"].min() - 0.005, bikes["lat"].max() + 0.005],
                 scaleanchor="x", scaleratio=1.6)
fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor="white",
                  legend=dict(orientation="h", y=1.02))
st.plotly_chart(fig)

# ---------------------------------------------------------------- distance vs rentals - updated for time(month)
st.markdown("**Verhuringen per maand (2022), naar afstand tot de metro**")
fig = px.line(monthly, x="month", y="departures_per_station_day",
              color="metro_distance", markers=True,
              color_discrete_map=COLORS,
              category_orders={"metro_distance": LABELS},
              height=450,
              labels={"month": "Maand",
                      "departures_per_station_day": "Gem. verhuringen per station per dag",
                      "metro_distance": ""})
fig.update_xaxes(dtick="M1", tickformat="%b")
st.plotly_chart(fig)
# ---------------------------------------------------------------- distance vs rentals
st.markdown("**Afstand tot de metro vs. verhuringen per station**")
fig = px.scatter(bikes, x="dist_to_metro_m", y="departures_per_day", color="metro_distance",
                 color_discrete_map=COLORS, category_orders={"metro_distance": LABELS},
                 trendline="lowess", trendline_scope="overall", trendline_color_override="black",
                 log_y=True, hover_name="name", height=450,
                 labels={"dist_to_metro_m": "Afstand tot dichtstbijzijnde metro (m)",
                         "departures_per_day": "Verhuringen per dag (log-schaal)", "metro_distance": ""})
fig.update_yaxes(tickvals=[1, 2, 5, 10, 20, 50, 100, 200, 500])
st.plotly_chart(fig)

# ---------------------------------------------------------------- average per group
summary = (bikes.groupby("metro_distance")["departures_per_day"]
           .agg(stations="size", mean="mean", median="median").reindex(LABELS).round(1).reset_index())
c1, c2 = st.columns([3, 2])
fig = px.bar(summary, x="metro_distance", y="median", color="metro_distance", color_discrete_map=COLORS,
             text=summary["median"].map(lambda v: f"{v:.0f}"), height=350,
             labels={"metro_distance": "", "median": "Mediaan verhuringen per dag"})
fig.update_traces(textposition="outside", cliponaxis=False)
fig.update_layout(showlegend=False)
c1.plotly_chart(fig)
c2.dataframe(summary.rename(columns={"metro_distance": "Groep", "stations": "Stations",
                                     "mean": "Gemiddeld per dag", "median": "Mediaan per dag"}), hide_index=True)
