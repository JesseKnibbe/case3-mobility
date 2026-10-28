# Page 5: key numbers + conclusion text.
# Edit the conclusion text at the bottom of this file (inside st.markdown(""" ... """)).
import numpy as np
import pandas as pd
import streamlit as st

import weather_impact as wi


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

# ---------------------------------------------------------------- conclusion text 
# Markdown: ### = heading, **text** = bold, - = bullet. Do not type three double quotes inside the text.
st.markdown("""
### Antwoord op de hoofdvraag
*Zorgt regen ervoor dat de Bikeshare minder gebruikt wordt?*

**Ja.** Op een regendag (≥ 1 mm) worden in 2022 gemiddeld **18% minder ritten** gemaakt dan op een vergelijkbare
droge dag (zelfde maand, zelfde type dag). Elke extra millimeter regen kost ongeveer **1.000 ritten per dag**.
Bij zware regen (> 5 mm) is het verlies het grootst.

### Subvraag: afstand tot de metro (Jurre)


### Subvraag: Weer
Regen raakt niet elk stadsdeel even hard: in de drukke, toeristische delen in het westen (Westminster,
Kensington and Chelsea) is het verlies groter (± 20%) dan in Islington en Tower Hamlets (± 16–17%).

### Kanttekeningen
- Het weer is per dag gemeten, niet per uur, en met één weerstation voor heel Londen.
- Neerslag alleen verklaart maar een klein deel van de verschillen tussen dagen (R² ≈ 0,11): seizoen en
  temperatuur spelen een grotere rol.
- We hebben geen metrogebruik per dag, dus overstappen op de metro konden we niet direct meten.
- Op 10 en 11 september 2022 ontbrak TfL-data; die dagen zijn uit de analyse gehaald.

