# Page 5: key numbers + conclusion text.
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import streamlit as st

import weather_impact as wi


def fit_model(d):
    # Same model as on the Weather vs Rental page: rides ~ precipitation + temperature + working day
    d = d.assign(werkdag=(d["day_type"] == "werkdag").astype(int))
    return smf.ols("rides ~ precip_mm + temp_mean + werkdag", data=d).fit()


@st.cache_data
def key_numbers():
    daily = pd.read_parquet(wi.PROC / "daily.parquet")
    daily = daily[daily["rides"] >= 0.1 * daily["rides"].median()].dropna(subset=["precip_mm", "temp_mean"])
    stations = pd.read_parquet(wi.PROC / "stations.parquet")
    sd_file = wi.PROC / "station_day.parquet"
    sd = pd.read_parquet(sd_file) if sd_file.exists() else \
        wi.build_station_day(pd.read_parquet(wi.PROC / "daily_station.parquet"), daily, stations)

    # 1. Fair comparison: rainy day vs. dry day in the same month and of the same day type
    city_pct, n_rain = wi.citywide_effect(sd, threshold=1.0)

    # 2. Prediction model (2022) and validation on 2021
    model = fit_model(daily)
    test = (pd.read_parquet(wi.PROC / "rides_2021_sample.parquet")
            .merge(pd.read_parquet(wi.PROC / "weather_2021.parquet"), on="date", how="inner")
            .dropna(subset=["rides", "temp_mean", "precip_mm"]))
    test = test.assign(werkdag=(test["day_type"] == "werkdag").astype(int))
    error = test["rides"] - model.predict(test[["precip_mm", "temp_mean", "werkdag"]])
    r2_test = 1 - (error ** 2).sum() / ((test["rides"] - test["rides"].mean()) ** 2).sum()

    # 3. Rain effect per distance group to the metro
    effect, source = wi.add_metro_distance(wi.rain_effect_per_station(sd, stations, threshold=1.0), stations)
    by_group = None
    if source == "metro":
        by_group = (effect.dropna(subset=["metro_group"]).groupby("metro_group", observed=True)
                    .apply(lambda x: np.average(x["effect_pct"], weights=x["dry_departures_per_day"]),
                           include_groups=False))
    return city_pct, n_rain, model.params["precip_mm"], model.rsquared, r2_test, len(test), by_group


city_pct, n_rain, per_mm, r2, r2_test, n_test, by_group = key_numbers()

st.title("✅ Conclusion")

st.subheader("De belangrijkste cijfers")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Regendag (≥1 mm) vs. droge dag", f"{city_pct:+.0f}% ritten", help=f"{n_rain} regendagen in 2022")
k2.metric("Per mm regen", f"{per_mm:+,.0f} ritten".replace(",", "."),
          help="Uit het model, gecorrigeerd voor temperatuur en werkdag/weekend")
k3.metric("R² model (2022)", f"{r2:.2f}", help="Model: ritten ~ neerslag + temperatuur + werkdag")
k4.metric("R² validatie (2021)", f"{r2_test:.2f}", help=f"Getest op {n_test} dagen uit 2021 die het model niet kende")

if by_group is not None and len(by_group):
    st.markdown("**Effect van regen per afstand tot de metro** (gewogen naar drukte):")
    cols = st.columns(len(by_group))
    for col, (group, value) in zip(cols, by_group.items()):
        col.metric(str(group), f"{value:+.0f}%")

st.divider()

# ---------------------------------------------------------------- conclusion text (edit here)
# Markdown: ### = heading, **text** = bold, - = bullet. Do not type three double quotes inside the text.
st.markdown("""
### Antwoord op de hoofdvraag
*Zorgt regen ervoor dat de Bikeshare minder gebruikt wordt?*

**Ja.** Op een regendag (≥ 1 mm) worden in 2022 gemiddeld **18% minder ritten** gemaakt dan op een vergelijkbare
droge dag (zelfde maand, zelfde type dag). Elke extra millimeter regen kost ongeveer **1.000 ritten per dag**,
ook als je rekening houdt met temperatuur en werkdag/weekend. Bij zware regen (> 5 mm) is het verlies het grootst.

Met neerslag, temperatuur en dagtype voorspellen we ongeveer de helft van de verschillen tussen dagen (R² ≈ 0,51).
Het model werkt ook op 2021, een jaar dat het niet kende (R² ≈ 0,53): het verband is dus niet toevallig voor 2022.

### Subvraag: afstand tot de metro
Stations dicht bij een metrostation (≤ 200 m) worden juist **méér** gebruikt: mediaan 44 verhuringen per dag,
tegen 32 bij stations verder dan 500 m. Fiets en metro lijken elkaar aan te vullen: fietsstations staan vaak
bij drukke metrostations.

### Subvraag: weer
Regen raakt niet elk stadsdeel even hard: in de drukke, toeristische delen in het westen (Westminster,
Kensington and Chelsea) is het verlies groter (± 20%) dan in Islington en Tower Hamlets (± 16–17%).
De afstand tot een metrostation maakt voor het effect van regen **geen** verschil (Close, Medium en Far alle drie
± 18–19%): dat mensen bij regen massaal de metro nemen, zien we in de data niet terug.

### Kanttekeningen
- Het weer is per dag gemeten, niet per uur, en met één weerstation voor heel Londen.
- Ook met neerslag, temperatuur en dagtype blijft ongeveer de helft van de verschillen tussen dagen onverklaard
  (bijvoorbeeld door vakanties, evenementen en stakingen).
- We hebben geen metrogebruik per dag, dus overstappen op de metro konden we niet direct meten.
- Op 10 en 11 september 2022 ontbrak TfL-data; die dagen zijn uit de analyse gehaald.
- Zowel metro- als fietsstations zijn geconcentreerd op drukke centrale punten. Afstand tot het centrum, drukke
  woonwijken of plaatsen met veel banen zijn misschien bepalender voor fietsverhuur dan nabijheid van de metro.
""")
