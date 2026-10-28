"""Shared functions for the weather sub-question (used by notebook 04 and the Streamlit app).

Sub-question: does the impact of rain differ per station, and do stations close to a
tube (metro) station lose more rides when it rains?
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROC = Path("data/processed")
DRY_MM = 0.2                                   # below this a day counts as dry
BANK = (51.5133, -0.0890)                      # Bank station, used as "city centre"
METRO_FILE = PROC / "station_metro_distance.parquet"   # delivered by Jurre
METRO_STATIONS_FILE = PROC / "metro_stations.parquet"  # optional: tube stations for the map layer
DEFAULT_GROUPS = [0, 200, 500, np.inf]                 # metres, same groups as Jurre uses
DEFAULT_LABELS = ["Close (≤200 m)", "Medium (200-500 m)", "Far (>500 m)"]


def wind_column(df):
    # Return the name of the wind-speed column if the weather data has one
    for c in ["wspd", "wind_speed", "windspeed", "wind"]:
        if c in df.columns:
            return c
    return None


def haversine_m(lat1, lon1, lat2, lon2):
    # Distance in metres between two points on earth (works on arrays)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(a))


def build_station_day(daily_station, daily, stations):
    # Station x day table INCLUDING days with 0 departures.
    # Only days between a station's first and last ride count (a station opened in June
    # should not get 0 rides for January).
    active = daily_station.groupby("start_bp")["date"].agg(first="min", last="max")
    all_days = pd.DataFrame({"date": pd.to_datetime(sorted(daily["date"].unique()))})
    grid = active.reset_index().merge(all_days, how="cross")
    grid = grid[(grid["date"] >= grid["first"]) & (grid["date"] <= grid["last"])].drop(columns=["first", "last"])

    sd = grid.merge(daily_station, on=["start_bp", "date"], how="left")
    sd["departures"] = sd["departures"].fillna(0).astype(int)

    weather_cols = ["date", "precip_mm", "temp_mean", "month", "day_type", "weekday", "is_holiday"]
    wc = wind_column(daily)
    if wc:
        weather_cols.append(wc)
    sd = sd.merge(daily[weather_cols], on="date", how="left")
    sd = sd.rename(columns={"start_bp": "bikepoint_id"})
    return sd


def rain_effect_per_station(sd, stations, threshold=1.0, day_types=None, min_per_day=5, min_rain_days=10):
    # Per station: % change in departures on rainy days compared to dry days
    # in the SAME month and of the SAME day type (corrects for season and weekends).
    # effect = sum(actual departures on rainy days) / sum(expected from dry days) - 1
    d = sd.dropna(subset=["precip_mm"])
    if day_types:
        d = d[d["day_type"].isin(day_types)]

    dry = d[d["precip_mm"] < DRY_MM]
    rainy = d[d["precip_mm"] >= threshold]

    ref = dry.groupby(["bikepoint_id", "month", "day_type"])["departures"].mean().rename("expected")
    rainy = rainy.join(ref, on=["bikepoint_id", "month", "day_type"]).dropna(subset=["expected"])

    per_station = rainy.groupby("bikepoint_id").agg(
        rain_days=("date", "nunique"), actual=("departures", "sum"), expected=("expected", "sum"))
    per_station["effect_pct"] = (per_station["actual"] / per_station["expected"] - 1) * 100

    volume = dry.groupby("bikepoint_id")["departures"].mean().rename("dry_departures_per_day")
    out = (per_station.join(volume, how="inner").reset_index()
           .merge(stations[["bikepoint_id", "name", "lat", "lon", "docks"]], on="bikepoint_id", how="left"))
    out = out[(out["dry_departures_per_day"] >= min_per_day) & (out["rain_days"] >= min_rain_days)]
    return out.dropna(subset=["lat", "lon"])


def citywide_effect(sd, threshold=1.0, day_types=None):
    # Same method as above, but for all stations together (one number for the KPI tile)
    city = sd.groupby(["date", "month", "day_type", "precip_mm"], dropna=False)["departures"].sum().reset_index()
    city = city.dropna(subset=["precip_mm"])
    if day_types:
        city = city[city["day_type"].isin(day_types)]
    ref = city[city["precip_mm"] < DRY_MM].groupby(["month", "day_type"])["departures"].mean().rename("expected")
    rainy = city[city["precip_mm"] >= threshold].join(ref, on=["month", "day_type"]).dropna(subset=["expected"])
    if rainy.empty:
        return np.nan, 0
    return (rainy["departures"].sum() / rainy["expected"].sum() - 1) * 100, len(rainy)


def metro_distance_table(stations):
    # Distance from every bike station to its nearest tube station.
    # 1) Jurre's own file (station_metro_distance.parquet) if it exists;
    # 2) otherwise computed here from metro_stations.parquet (columns: name, lat, lon);
    # 3) otherwise None.
    if METRO_FILE.exists():
        return pd.read_parquet(METRO_FILE)
    if METRO_STATIONS_FILE.exists():
        metro = pd.read_parquet(METRO_STATIONS_FILE).dropna(subset=["lat", "lon"])
        s = stations.dropna(subset=["lat", "lon"])
        # distance matrix: bike stations (rows) x tube stations (columns)
        dist = haversine_m(s["lat"].values[:, None], s["lon"].values[:, None],
                           metro["lat"].values[None, :], metro["lon"].values[None, :])
        nearest = dist.argmin(axis=1)
        out = pd.DataFrame({"bikepoint_id": s["bikepoint_id"].values,
                            "metro_distance_m": dist.min(axis=1).round(),
                            "nearest_metro": metro["name"].values[nearest]})
        out["metro_group"] = pd.cut(out["metro_distance_m"], DEFAULT_GROUPS, labels=DEFAULT_LABELS)
        return out
    return None


def add_metro_distance(effect, stations):
    # Use the metro distance groups if available; otherwise fall back to
    # distance to the city centre (Bank) so the charts already work.
    metro = metro_distance_table(stations)
    if metro is not None:
        out = effect.merge(metro, on="bikepoint_id", how="left")
        if "metro_group" not in out.columns and "metro_distance_m" in out.columns:
            out["metro_group"] = pd.cut(out["metro_distance_m"], DEFAULT_GROUPS, labels=DEFAULT_LABELS)
        return out, "metro"
    out = effect.copy()
    out["centre_distance_km"] = haversine_m(out["lat"], out["lon"], *BANK) / 1000
    out["metro_group"] = pd.cut(out["centre_distance_km"], [0, 2, 4, 6, np.inf],
                                labels=["< 2 km", "2–4 km", "4–6 km", "> 6 km"])
    return out, "centre"


def fit_daily_model(daily):
    # Linear regression on log(rides): each coefficient ≈ relative change in rides
    d = daily.dropna(subset=["precip_mm", "temp_mean"]).copy()
    wc = wind_column(d)
    formula = "np.log(rides) ~ precip_mm + temp_mean + C(weekday) + C(month) + is_holiday"
    if wc and d[wc].notna().mean() > 0.8:
        d[wc] = d[wc].fillna(d[wc].median())
        formula = formula.replace("temp_mean", f"temp_mean + {wc}")
    else:
        wc = None
    return smf.ols(formula, data=d).fit(), wc


def predict_rides(model, precip_mm, temp_mean, weekday, month, is_holiday=False, wind=None, wind_col=None):
    # Predict the number of rides for one hypothetical day (what-if tool)
    row = {"precip_mm": precip_mm, "temp_mean": temp_mean, "weekday": weekday,
           "month": month, "is_holiday": is_holiday}
    if wind_col:
        row[wind_col] = wind
    pred = model.get_prediction(pd.DataFrame([row])).summary_frame(alpha=0.05)
    # smearing correction: exp(log-prediction) underestimates the mean slightly
    smear = np.mean(np.exp(model.resid))
    return (np.exp(pred["mean"].iloc[0]) * smear,
            np.exp(pred["obs_ci_lower"].iloc[0]) * smear,
            np.exp(pred["obs_ci_upper"].iloc[0]) * smear)
