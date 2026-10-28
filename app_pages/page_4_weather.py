# Page 4: weather vs rental (Jesse)
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import weather_impact as wi

RAIN = "#2a6fdb"
INK = "#3d4450"
DAY_TYPES = ["werkdag", "weekend/feestdag"]
CATS = ["droog (<0,2 mm)", "licht (0,2–2)", "matig (2–5)", "zwaar (>5)"]
CAT_COLORS = dict(zip(CATS, ["#9aa4b1", "#9ecae1", "#4292c6", "#08519c"]))


# ---------------------------------------------------------------- data (cached)
@st.cache_data
def load_data():
    daily = pd.read_parquet(wi.PROC / "daily.parquet")
    daily = daily[daily["rides"] >= 0.1 * daily["rides"].median()]   # skip days with incomplete TfL data
    stations = pd.read_parquet(wi.PROC / "stations.parquet")
    sd_file = wi.PROC / "station_day.parquet"
    sd = pd.read_parquet(sd_file) if sd_file.exists() else \
        wi.build_station_day(pd.read_parquet(wi.PROC / "daily_station.parquet"), daily, stations)
    return daily, stations, sd


@st.cache_data
def station_effect(sd, stations, threshold, day_types):
    return wi.rain_effect_per_station(sd, stations, threshold, list(day_types), min_per_day=5, min_rain_days=5)


daily, stations, sd = load_data()

# ---------------------------------------------------------------- sidebar filters
st.sidebar.header("Filters")
threshold = st.sidebar.slider("Regendag vanaf (mm neerslag)", 0.5, 10.0, 1.0, 0.5,
                              help="Droge dag = minder dan 0,2 mm neerslag.")
day_types = st.sidebar.multiselect("Type dag", DAY_TYPES, default=DAY_TYPES)
if not day_types:
    st.warning("Kies minstens één type dag.")
    st.stop()
d = daily[daily["day_type"].isin(day_types)].dropna(subset=["precip_mm"]).copy()

# ---------------------------------------------------------------- header + KPIs
st.title("🌧️ Weather vs Rental")
st.caption("Elke regendag wordt vergeleken met droge dagen in **dezelfde maand** en van **hetzelfde type dag**, "
           "zodat seizoen en weekend het effect niet vertekenen.")

city_pct, n_rain = wi.citywide_effect(sd, threshold, day_types)
effect = station_effect(sd, stations, threshold, tuple(day_types))
k1, k2, k3 = st.columns(3)
k1.metric("Effect regen (heel Londen)", f"{city_pct:+.0f}%" if pd.notna(city_pct) else "–")
k2.metric("Regendagen / droge dagen", f"{n_rain} / {int((d['precip_mm'] < wi.DRY_MM).sum())}")
k3.metric("Stations geanalyseerd", f"{len(effect)}")

tab_main, tab_map, tab_model = st.tabs(["❓ Hoofdvraag", "🗺️ Kaart", "🔮 Voorspelling"])

# ---------------------------------------------------------------- tab 1: main question
with tab_main:
    st.subheader("Zorgt regen ervoor dat de Bikeshare minder gebruikt wordt?")
    d["rain_cat"] = pd.Categorical(d["rain_cat"].astype(str), CATS, ordered=True)

    # 1. Rides per day per rain category
    st.markdown("**1. Ritten per dag per regencategorie**")
    counts = d["rain_cat"].value_counts()
    d["label"] = d["rain_cat"].map(lambda c: f"{c} (n={counts.get(c, 0)})")
    fig = px.box(d.sort_values("rain_cat"), x="label", y="rides", color="rain_cat",
                 color_discrete_map=CAT_COLORS, height=380, labels={"label": "", "rides": "Ritten per dag"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)
    st.caption("Regendagen vallen vaker in herfst en winter, wanneer er sowieso minder gefietst wordt. "
               "Deze vergelijking overschat het effect van regen daarom; zie stap 2.")

    # 2. Fair comparison: vs. dry days in the same month and of the same day type
    st.markdown("**2. Eerlijke vergelijking: t.o.v. droge dagen in dezelfde maand en van hetzelfde type dag**")
    ref = d[d["rain_cat"] == CATS[0]].groupby(["month", "day_type"])["rides"].median().rename("ref_dry")
    d = d.join(ref, on=["month", "day_type"])
    d["pct_vs_dry"] = (d["rides"] / d["ref_dry"] - 1) * 100
    eff = d.dropna(subset=["pct_vs_dry"]).groupby("rain_cat", observed=True)["pct_vs_dry"].mean()
    fig = px.bar(x=eff.index.astype(str), y=eff.values, color=eff.index.astype(str), color_discrete_map=CAT_COLORS,
                 text=[f"{v:+.0f}%" for v in eff.values], height=360,
                 labels={"x": "", "y": "% ritten t.o.v. droge dag"})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False)
    fig.add_hline(y=0, line_color=INK, line_width=1)
    st.plotly_chart(fig)
    if CATS[3] in eff.index:
        st.markdown(f"Op een dag met **zware regen (>5 mm)** worden gemiddeld **{eff[CATS[3]]:+.0f}%** ritten "
                    f"gemaakt vergeleken met een droge dag in dezelfde maand.")

# ---------------------------------------------------------------- tab 2: map
with tab_map:
    st.subheader("Welke stations verliezen de meeste ritten bij regen?")
    st.caption("Kleur = verandering in vertrekken op regendagen (donkerder = groter verlies). "
               "Grootte = drukte op droge dagen.")
    if effect.empty:
        st.info("Te weinig data met deze filters.")
    else:
        m = effect.copy()
        m["loss_pct"] = (-m["effect_pct"]).clip(lower=0)
        m["Effect regen"] = m["effect_pct"].map(lambda v: f"{v:+.1f}%")
        common = dict(lat="lat", lon="lon", color="loss_pct", size="dry_departures_per_day", size_max=14,
                      color_continuous_scale="Blues", hover_name="name", zoom=11, height=600,
                      hover_data={"Effect regen": True, "lat": False, "lon": False, "loss_pct": False,
                                  "dry_departures_per_day": ":.0f"},
                      labels={"loss_pct": "Verlies (%)", "dry_departures_per_day": "Vertrekken/droge dag"})
        try:  # plotly >= 5.24
            fig = px.scatter_map(m, map_style="carto-positron", **common)
        except AttributeError:  # older plotly
            fig = px.scatter_mapbox(m, mapbox_style="carto-positron", **common)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig)

        cols = {"name": "Station", "effect_pct": "Effect (%)", "dry_departures_per_day": "Vertrekken/droge dag"}
        c1, c2 = st.columns(2)
        c1.markdown("**Meest gevoelig voor regen**")
        c1.dataframe(effect.nsmallest(10, "effect_pct")[list(cols)].rename(columns=cols).round(1), hide_index=True)
        c2.markdown("**Minst gevoelig voor regen**")
        c2.dataframe(effect.nlargest(10, "effect_pct")[list(cols)].rename(columns=cols).round(1), hide_index=True)

# ---------------------------------------------------------------- tab 3: prediction (precipitation only)
with tab_model:
    st.subheader("Voorspel het aantal ritten op basis van neerslag")
    model = wi.fit_rain_model(d)
    a, b = model.params["Intercept"], model.params["precip_mm"]

    mm = st.slider("Neerslag (mm)", 0.0, 25.0, 5.0, 0.5)
    pred = a + b * mm
    c1, c2, c3 = st.columns(3)
    c1.metric("Verwachte ritten", f"{pred:,.0f}".replace(",", "."), f"{pred - a:+,.0f} t.o.v. droge dag".replace(",", "."))
    c2.metric("Per mm regen", f"{b:+,.0f} ritten".replace(",", "."))
    c3.metric("R²", f"{model.rsquared:.2f}", help="Welk deel van de verschillen tussen dagen neerslag verklaart")

    fig = px.scatter(d, x="precip_mm", y="rides", opacity=.5, height=400,
                     labels={"precip_mm": "Neerslag (mm)", "rides": "Ritten per dag"})
    fig.update_traces(marker=dict(color="#9aa4b1", size=7))
    xs = np.array([0, max(25, d["precip_mm"].max())])
    fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", line=dict(color=RAIN, width=2), name="regressielijn"))
    fig.add_trace(go.Scatter(x=[mm], y=[pred], mode="markers", name="jouw keuze",
                             marker=dict(size=13, color=RAIN, line=dict(width=2, color="white"))))
    fig.update_layout(legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig)
    st.caption(f"Model: ritten = {a:,.0f} {b:+,.0f} × neerslag (mm). Alleen neerslag zit in het model; "
               "seizoen en temperatuur verklaren een groot deel van de rest.".replace(",", "."))
