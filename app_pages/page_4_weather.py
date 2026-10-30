# Page 4: weather vs rental (Jesse)
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import weather_impact as wi
import statsmodels.formula.api as smf


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
def borough_table(sd, effect, stations):
    b = wi.borough_summary(sd, effect, stations, wi.load_boroughs())
    # with fewer than 10 stations the percentage is unreliable: not coloured on the map
    b["loss_pct"] = (-b["effect_pct"]).clip(lower=0).where(b["stations"] >= 10)
    b["Effect regen"] = b["effect_pct"].map(lambda v: f"{v:+.1f}%")
    return b


@st.cache_data
def station_effect(sd, stations, threshold, day_types):
    return wi.rain_effect_per_station(sd, stations, threshold, list(day_types), min_per_day=5, min_rain_days=5)


def borough_map(geo, b, col, scale, title):
    # Draw the boroughs as filled shapes. Plain x/y plot (lon/lat): needs no map tiles or internet.
    from plotly.colors import sample_colorscale
    vals = b.set_index("borough")[col]
    lo, hi = vals.min(), vals.max()
    fig = go.Figure()
    for f in geo["features"]:
        name = f["properties"]["name"]
        polys = f["geometry"]["coordinates"]
        polys = [polys] if f["geometry"]["type"] == "Polygon" else polys
        has_data = name in vals.index and pd.notna(vals[name])
        if has_data:
            t = 0.5 if hi == lo else (vals[name] - lo) / (hi - lo)
            fill = sample_colorscale(scale, [0.15 + 0.85 * t])[0]
            row = b.set_index("borough").loc[name]
            hover = (f"<b>{name}</b><br>Stations: {row['stations']}<br>Ritten per dag: {row['rides_per_day']:,.0f}"
                     f"<br>Effect regen: {row['effect_pct']:+.1f}%").replace(",", ".")
        elif name in vals.index:
            fill, hover = "#dfe2e6", f"{name}<br>te weinig stations voor een betrouwbaar percentage"
        else:
            fill, hover = "#f1f2f4", f"{name}<br>geen Bikeshare-stations"
        for poly in polys:
            ring = poly[0]
            fig.add_trace(go.Scatter(x=[c[0] for c in ring], y=[c[1] for c in ring], fill="toself", mode="lines",
                                     line=dict(color="white", width=1), fillcolor=fill, hoverinfo="text",
                                     text=hover, hoveron="fills", showlegend=False))
    # invisible trace that only draws the colour bar
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(colorscale=[[0, sample_colorscale(scale, [0.15])[0]], [1, sample_colorscale(scale, [1.0])[0]]],
                                         cmin=lo, cmax=hi, color=[lo], showscale=True,
                                         colorbar=dict(title=title))))
    # zoom to the boroughs with stations
    with_data = [f for f in geo["features"] if f["properties"]["name"] in vals.dropna().index]
    xs = [c[0] for f in with_data for poly in ([f["geometry"]["coordinates"]] if f["geometry"]["type"] == "Polygon"
          else f["geometry"]["coordinates"]) for c in poly[0]]
    ys = [c[1] for f in with_data for poly in ([f["geometry"]["coordinates"]] if f["geometry"]["type"] == "Polygon"
          else f["geometry"]["coordinates"]) for c in poly[0]]
    pad = 0.01
    fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad])
    fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad], scaleanchor="x", scaleratio=1.6)
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor="white")
    return fig

# ---------------------------------------------------------------- aanpassing Jurre, voorspelling filtert temperatuur en weekdag/vrijedag uit
def fit_rain_model_v2(d):
    d = d.assign(werkdag=(d["day_type"] == "werkdag").astype(int))
    return smf.ols("rides ~ precip_mm + temp_mean + werkdag", data=d).fit()
# ---------------------------------------------------------------- einde aanpassing jurre


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

# ---------------------------------------------------------------- tab 2: map per borough
with tab_map:
    st.subheader("Ritten en effect van regen per stadsdeel (borough)")
    if effect.empty:
        st.info("Te weinig data met deze filters.")
    else:
        geo = wi.load_boroughs()
        b = borough_table(sd, effect, stations)
        choice = st.radio("Kleur de kaart op", ["Effect van regen", "Ritten per dag"], horizontal=True)
        if choice == "Effect van regen":
            col, scale, title = "loss_pct", "Blues", "Verlies bij regen (%)"
        else:
            col, scale, title = "rides_per_day", "Greens", "Ritten per dag"
        fig = borough_map(geo, b, col, scale, title)
        st.plotly_chart(fig)
        st.caption("Alleen stadsdelen met Bikeshare-stations zijn ingekleurd (bij effect van regen: minstens 10 stations). "
                   "Effect van regen = vertrekken op regendagen t.o.v. droge dagen in dezelfde maand.")

        tbl = (b.sort_values("effect_pct")[["borough", "stations", "rides_per_day", "effect_pct"]]
                 .rename(columns={"borough": "Stadsdeel", "stations": "Stations",
                                  "rides_per_day": "Ritten per dag", "effect_pct": "Effect regen (%)"}))
        st.dataframe(tbl.round({"Ritten per dag": 0, "Effect regen (%)": 1}), hide_index=True)

    st.dataframe(d)
# ---------------------------------------------------------------- tab 3: prediction (precipitation only)
with tab_model:
    st.subheader("Voorspel het aantal ritten op basis van neerslag, temperatuur en dagtype")
    model = fit_rain_model_v2(d)
    p = model.params

    c_in1, c_in2, c_in3 = st.columns(3)
    mm = c_in1.slider("Neerslag (mm)", 0.0, 25.0, 5.0, 0.5)
    temp = c_in2.slider("Gemiddelde temperatuur (°C)",
                    float(d["temp_mean"].min()), float(d["temp_mean"].max()),
                    float(d["temp_mean"].mean()), 0.5)
    dagtype = c_in3.radio("Dagtype", ["werkdag", "weekend/feestdag"], horizontal=True)
    wd = int(dagtype == "werkdag")

    def predict(precip):
        X = pd.DataFrame({"precip_mm": np.atleast_1d(precip), "temp_mean": temp, "werkdag": wd})
        return model.predict(X).to_numpy()

    pred = predict(mm)[0]
    pred_dry = predict(0.0)[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verwachte ritten", f"{pred:,.0f}".replace(",", "."),
          f"{pred - pred_dry:+,.0f} t.o.v. droge dag".replace(",", "."))
    c2.metric("Per mm regen", f"{p['precip_mm']:+,.0f} ritten".replace(",", "."))
    c3.metric("Per °C", f"{p['temp_mean']:+,.0f} ritten".replace(",", "."))
    c4.metric("Werkdag vs weekend", f"{p['werkdag']:+,.0f} ritten".replace(",", "."))
    st.caption(f"R² = {model.rsquared:.2f}")

    d_plot = d.assign(dagtype=d["day_type"])
    fig = px.scatter(d_plot, x="precip_mm", y="rides", color="dagtype", opacity=.5, height=400,
                 color_discrete_map={"werkdag": "#9aa4b1", "weekend/feestdag": "#e0b44c"},
                 labels={"precip_mm": "Neerslag (mm)", "rides": "Ritten per dag", "dagtype": "Dagtype"})
    xs = np.array([0, max(25, d["precip_mm"].max())])
    fig.add_trace(go.Scatter(x=xs, y=predict(xs), mode="lines",
                         line=dict(color=RAIN, width=2),
                         name=f"regressielijn ({dagtype}, {temp:.1f}°C)"))
    fig.add_trace(go.Scatter(x=[mm], y=[pred], mode="markers", name="jouw keuze",
                         marker=dict(size=13, color=RAIN, line=dict(width=2, color="white"))))
    fig.update_layout(legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig)





