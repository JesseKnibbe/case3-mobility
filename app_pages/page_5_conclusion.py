# Page 5: key numbers + conclusion text.
# The text comes from teksten/conclusie.md, so it can be edited without touching Python.
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import weather_impact as wi

TEXT_FILE = Path("teksten/conclusie.md")


@st.cache_data
def key_numbers():
    daily = pd.read_parquet(wi.PROC / "daily.parquet")
    stations = pd.read_parquet(wi.PROC / "stations.parquet")
    sd_file = wi.PROC / "station_day.parquet"
    sd = pd.read_parquet(sd_file) if sd_file.exists() else \
        wi.build_station_day(pd.read_parquet(wi.PROC / "daily_station.parquet"), daily, stations)

    city_pct, n_rain = wi.citywide_effect(sd, threshold=1.0)
    model = wi.fit_rain_model(daily)
    per_mm = model.params["precip_mm"]

    effect, source = wi.add_metro_distance(wi.rain_effect_per_station(sd, stations, threshold=1.0), stations)
    by_group = None
    if source == "metro":
        by_group = (effect.dropna(subset=["metro_group"]).groupby("metro_group", observed=True)
                    .apply(lambda x: np.average(x["effect_pct"], weights=x["dry_departures_per_day"]),
                           include_groups=False))
    return city_pct, n_rain, per_mm, model.rsquared, by_group


city_pct, n_rain, per_mm, r2, by_group = key_numbers()

st.title("✅ Conclusion")

st.subheader("De belangrijkste cijfers")
k1, k2, k3 = st.columns(3)
k1.metric("Regendag (≥1 mm) vs. droge dag", f"{city_pct:+.0f}% ritten", help=f"{n_rain} regendagen in 2022")
k2.metric("Per mm regen", f"{per_mm:+,.0f} ritten".replace(",", "."), help="Lineaire regressie: ritten ~ neerslag")
k3.metric("Verklaarde variatie model (R²)", f"{r2:.2f}")

if by_group is not None and len(by_group):
    st.markdown("**Effect van regen per afstand tot de metro** (gewogen naar drukte):")
    cols = st.columns(len(by_group))
    for col, (group, value) in zip(cols, by_group.items()):
        col.metric(str(group), f"{value:+.0f}%")

st.divider()
if TEXT_FILE.exists():
    st.markdown(TEXT_FILE.read_text(encoding="utf-8"))
else:
    st.info("Schrijf de conclusie in `teksten/conclusie.md`.")
