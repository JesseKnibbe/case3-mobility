# Page 3: sub-question distance to the metro (Jurre)
import streamlit as st
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
DAY_TYPES = ["Weekdag", "Weekend"]


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

    # weekday / weekend label and month
    sd["day_type"] = sd["date"].dt.dayofweek.ge(5).map({True: "Weekend", False: "Weekdag"})
    sd["month"] = sd["date"].dt.to_period("M").dt.to_timestamp()

    # total departures per month, category and day type
    monthly = (sd.groupby(["month", "metro_distance", "day_type"], observed=True)["departures"]
                 .sum().reset_index())

    # number of weekdays / weekend days in each month of 2022
    cal = pd.DataFrame({"date": pd.date_range("2022-01-01", "2022-12-31")})
    cal["day_type"] = cal["date"].dt.dayofweek.ge(5).map({True: "Weekend", False: "Weekdag"})
    cal["month"] = cal["date"].dt.to_period("M").dt.to_timestamp()
    n_days = cal.groupby(["month", "day_type"]).size().rename("n_days").reset_index()

    # number of stations per category
    n_stations = bikes.groupby("metro_distance").size().rename("n_stations").reset_index()

    monthly = monthly.merge(n_days, on=["month", "day_type"]).merge(n_stations, on="metro_distance")
    monthly["departures_per_station_day"] = (
        monthly["departures"] / (monthly["n_days"] * monthly["n_stations"])
    )
    return monthly

monthly = load_monthly(bikes)

@st.cache_data
def load_station_perday(bikes: pd.DataFrame) -> pd.DataFrame:
    sd = pd.read_parquet("station_day.parquet")
    sd["date"] = pd.to_datetime(sd["date"])
    sd = sd[sd["date"].dt.year == 2022]
    sd["day_type"] = sd["date"].dt.dayofweek.ge(5).map({True: "Weekend", False: "Weekdag"})

    # number of weekdays / weekend days in 2022
    cal = pd.Series(pd.date_range("2022-01-01", "2022-12-31"))
    n_days = cal.dt.dayofweek.ge(5).map({True: "Weekend", False: "Weekdag"}).value_counts()

    # every station x every day type, so stations without departures count as 0
    base = bikes[["bikepoint_id", "metro_distance"]].drop_duplicates("bikepoint_id")
    grid = base.merge(pd.DataFrame({"day_type": DAY_TYPES}), how="cross")
    tot = sd.groupby(["bikepoint_id", "day_type"], as_index=False)["departures"].sum()
    grid = grid.merge(tot, on=["bikepoint_id", "day_type"], how="left")
    grid["departures"] = grid["departures"].fillna(0)
    grid["departures_per_day"] = grid["departures"] / grid["day_type"].map(n_days)
    return grid

per_station = load_station_perday(bikes)

st.title("🚇 Distance from Metro")
st.subheader("Wordt de Bikeshare minder gebruikt dicht bij de metro?")

# ---------------------------------------------------------------- key figures
counts = bikes["metro_distance"].value_counts()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Fietsstations", len(bikes))
k2.metric("Metrostations", len(metro))
k3.metric("Mediane afstand tot metro", f"{bikes['dist_to_metro_m'].median():.0f} m")
k4.metric("Binnen 200 m van de metro", f"{counts.get(LABELS[0], 0) / len(bikes):.0%}")




# ------------------------------------------------------------------- edit into tabs from this point onward:

tab_map2, tab_line2, tab_summary = st.tabs(["🗺️ Kaart", "❓ Hoofdvraag", "🔮 Averages"])

with tab_map2:
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

with tab_line2:
          st.markdown("**Verhuringen per maand (2022), naar afstand tot de metro**")
          
          keuze = st.sidebar.selectbox(
              "Toon",
              ["Weekdagen en weekend", "Alleen weekdagen", "Alleen weekend"],
              key="weekdag_weekend",
          )

          if keuze == "Alleen weekdagen":
              day_types = ["Weekdag"]
          elif keuze == "Alleen weekend":
              day_types = ["Weekend"]
          else:
              day_types = DAY_TYPES

          monthly_view = monthly[monthly["day_type"].isin(day_types)]
                 
          DASH = {"Weekdag": "solid", "Weekend": "dash"}
          both = keuze == "Weekdagen en weekend"

          fig = go.Figure()

          # one legend entry per distance category (color)
          for cat in LABELS:
              first = True
              for dt in day_types:
                  d = (monthly_view[(monthly_view["metro_distance"] == cat) &
                          (monthly_view["day_type"] == dt)]
                       .sort_values("month"))
                  fig.add_trace(go.Scatter(
                      x=d["month"], y=d["departures_per_station_day"],
                      mode="lines+markers",
                      name=cat, legendgroup=cat, showlegend=first,
                      line=dict(color=COLORS[cat], dash=DASH[dt]),
                      opacity=0.5 if (both and dt == "Weekend") else 1,
                      hovertemplate=f"{cat} ({dt.lower()})<br>%{{x|%b}}: %{{y:.1f}}<extra></extra>",
                  ))
                  first = False

          # legend entries explaining the line style (only when both are shown)
          if both:
              for dt in DAY_TYPES:
                  fig.add_trace(go.Scatter(
                      x=[None], y=[None], mode="lines",
                      name=dt, legendgroup="stijl",
                      line=dict(color="grey", dash=DASH[dt]),
                      hoverinfo="skip",
                  ))

          fig.update_layout(
              height=450,
              showlegend=True,
              legend=dict(title_text=""),
              xaxis_title="Maand",
              yaxis_title="Gem. verhuringen per station per dag",
          )
          fig.update_xaxes(dtick="M1", tickformat="%b")
          fig.update_yaxes(rangemode="tozero")

          st.plotly_chart(fig)



with tab_summary:
          per_station_view = per_station[per_station["day_type"].isin(day_types)]

          summary = (per_station_view
                     .groupby(["metro_distance", "day_type"], observed=True)["departures_per_day"]
                     .agg(stations="size", mean="mean", median="median")
                     .round(1).reset_index())
          summary["metro_distance"] = pd.Categorical(summary["metro_distance"], LABELS, ordered=True)
          summary["day_type"] = pd.Categorical(summary["day_type"], DAY_TYPES, ordered=True)
          summary = summary.sort_values(["metro_distance", "day_type"])

          c1, c2 = st.columns([3, 2])

          fig = go.Figure()
          for dt in day_types:
              d = summary[summary["day_type"] == dt]
              fig.add_trace(go.Bar(
                  x=d["metro_distance"].astype(str), y=d["median"], name=dt,
                  marker_color=[COLORS[c] for c in d["metro_distance"]],
                  opacity=0.5 if (both and dt == "Weekend") else 1,
                  text=d["median"].map(lambda v: f"{v:.0f}"),
                  textposition="outside", cliponaxis=False, showlegend=False,
              ))

          # legend that explains weekday vs weekend, only when both are shown
          if both:
              fig.add_trace(go.Bar(x=[None], y=[None], name="Weekdag",
                         marker_color="grey", opacity=1))
              fig.add_trace(go.Bar(x=[None], y=[None], name="Weekend",
                         marker_color="grey", opacity=0.5))

          fig.update_layout(barmode="group", height=350, showlegend=both,
                  legend=dict(title_text=""),
                  xaxis=dict(categoryorder="array", categoryarray=LABELS, title=""),
                  yaxis_title="Mediaan verhuringen per dag")
          fig.update_yaxes(rangemode="tozero")
          c1.plotly_chart(fig)

          c2.dataframe(summary.rename(columns={"metro_distance": "Groep", "day_type": "Dagtype",
                                     "stations": "Stations", "mean": "Gemiddeld per dag",
                                     "median": "Mediaan per dag"}), hide_index=True)

          
