# Page: weather sub-question (Jesse). Part of the team dashboard; start the whole app with: streamlit run Start.py
# Requires notebooks 01 + 02 (and preferably 04) to have been run, so data/processed/ exists.
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats

import weather_impact as wi

st.set_page_config(page_title="Regen & Bikeshare", layout="wide")

RAIN = "#2a6fdb"     # rainy days
DRY = "#9aa4b1"      # dry days
INK = "#3d4450"
DAY_TYPES = ["werkdag", "weekend/feestdag"]
WEEKDAYS = ["maandag", "dinsdag", "woensdag", "donderdag", "vrijdag", "zaterdag", "zondag"]
MONTHS = ["jan", "feb", "mrt", "apr", "mei", "jun", "jul", "aug", "sep", "okt", "nov", "dec"]


# ---------------------------------------------------------------- data loading (cached)
@st.cache_data
def load_data():
    daily = pd.read_parquet(wi.PROC / "daily.parquet")
    hourly = pd.read_parquet(wi.PROC / "hourly.parquet")
    stations = pd.read_parquet(wi.PROC / "stations.parquet")
    sd_file = wi.PROC / "station_day.parquet"
    if sd_file.exists():
        sd = pd.read_parquet(sd_file)
    else:  # notebook 04 not run yet: build it here
        sd = wi.build_station_day(pd.read_parquet(wi.PROC / "daily_station.parquet"), daily, stations)
    metro = pd.read_parquet(wi.METRO_STATIONS_FILE) if wi.METRO_STATIONS_FILE.exists() else None
    return daily, hourly, stations, sd, metro


@st.cache_data
def station_effect(sd, stations, threshold, day_types, min_per_day):
    eff = wi.rain_effect_per_station(sd, stations, threshold, list(day_types), min_per_day, min_rain_days=5)
    return wi.add_metro_distance(eff, stations)


@st.cache_resource
def model_for(daily):
    return wi.fit_daily_model(daily)


daily, hourly, stations, sd, metro = load_data()

# ---------------------------------------------------------------- sidebar filters
st.sidebar.header("Filters")
threshold = st.sidebar.slider("Regendag vanaf (mm neerslag)", 0.5, 10.0, 1.0, 0.5,
                              help="Droge dag = minder dan 0,2 mm. Dagen daartussen tellen niet mee.")
day_types = st.sidebar.multiselect("Type dag", DAY_TYPES, default=DAY_TYPES)
months = st.sidebar.select_slider("Maanden", options=list(range(1, 13)), value=(1, 12),
                                  format_func=lambda m: MONTHS[m - 1])
min_per_day = st.sidebar.slider("Alleen stations met minstens … vertrekken per droge dag", 1, 50, 5)
if not day_types:
    st.warning("Kies minstens één type dag.")
    st.stop()

in_months = lambda df: df[df["date"].dt.month.between(*months)]
sd_f, daily_f, hourly_f = in_months(sd), in_months(daily), in_months(hourly)

# ---------------------------------------------------------------- header + KPIs
st.title("Waar en bij wie heeft regen de meeste impact?")
st.caption("Elke regendag wordt vergeleken met droge dagen in **dezelfde maand** en van **hetzelfde type dag**, "
           "zodat seizoen en weekend het effect niet vertekenen.")

city_pct, n_rain = wi.citywide_effect(sd_f, threshold, day_types)
effect, dist_source = station_effect(sd_f, stations, threshold, tuple(day_types), min_per_day)
n_dry = int(((daily_f["precip_mm"] < wi.DRY_MM) & daily_f["day_type"].isin(day_types)).sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Effect regen (heel Londen)", f"{city_pct:+.0f}%" if pd.notna(city_pct) else "–",
          help="Verandering in vertrekken op regendagen t.o.v. vergelijkbare droge dagen")
k2.metric("Regendagen / droge dagen", f"{n_rain} / {n_dry}")
k3.metric("Stations geanalyseerd", f"{len(effect)}")
if len(effect):
    worst = effect.nsmallest(1, "effect_pct").iloc[0]
    k4.metric("Gevoeligste station", f"{worst['effect_pct']:+.0f}%", help=worst["name"])
    k4.caption(worst["name"])

if effect.empty:
    st.info("Te weinig data met deze filters. Verlaag de drempel of het minimum aantal vertrekken.")
    st.stop()

tab_map, tab_metro, tab_hour, tab_time, tab_whatif = st.tabs(
    ["🗺️ Kaart", "🚇 Afstand tot metro", "🕗 Per uur", "📈 Door het jaar", "🔮 Wat als?"])

# ---------------------------------------------------------------- tab 1: map
with tab_map:
    st.subheader("Welke stations verliezen de meeste ritten bij regen?")
    st.caption("Kleur = verandering in vertrekken op regendagen (donkerder = groter verlies). "
               "Grootte = drukte op droge dagen. Beweeg over een station voor details.")
    m = effect.copy()
    m["loss_pct"] = (-m["effect_pct"]).clip(lower=0)
    m["label"] = m["effect_pct"].map(lambda v: f"{v:+.1f}%")
    common = dict(lat="lat", lon="lon", color="loss_pct", size="dry_departures_per_day", size_max=14,
                  color_continuous_scale="Blues", hover_name="name", zoom=11, height=600,
                  hover_data={"label": True, "dry_departures_per_day": ":.0f", "rain_days": True,
                              "lat": False, "lon": False, "loss_pct": False})
    labels = {"label": "Effect regen", "dry_departures_per_day": "Vertrekken/droge dag",
              "rain_days": "Regendagen", "loss_pct": "Verlies (%)"}
    c_a, c_b = st.columns(2)
    show_metro = metro is not None and c_a.checkbox("Toon metrostations", value=True)
    # Fallback without map tiles: handy when the wifi fails during the presentation
    offline = c_b.checkbox("Kaart zonder achtergrond (werkt zonder internet)", value=False)
    if offline:
        plain = {k: v for k, v in common.items() if k not in ("zoom", "lat", "lon")}
        fig = px.scatter(m, x="lon", y="lat", labels=labels, **plain)
        fig.update_yaxes(scaleanchor="x", scaleratio=1.6, showgrid=False, visible=False)
        fig.update_xaxes(showgrid=False, visible=False)
        trace_cls = go.Scatter
        metro_xy = lambda: dict(x=metro["lon"], y=metro["lat"])
    else:
        try:  # plotly >= 5.24
            fig = px.scatter_map(m, map_style="carto-positron", labels=labels, **common)
            trace_cls = go.Scattermap
        except AttributeError:  # older plotly
            fig = px.scatter_mapbox(m, mapbox_style="carto-positron", labels=labels, **common)
            trace_cls = go.Scattermapbox
        metro_xy = lambda: dict(lat=metro["lat"], lon=metro["lon"])
    if show_metro:
        fig.add_trace(trace_cls(**metro_xy(), mode="markers", name="Metrostation",
                                marker=dict(size=8, color="#d62728"), text=metro["name"], hoverinfo="text"))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig)

    c1, c2 = st.columns(2)
    cols = {"name": "Station", "effect_pct": "Effect (%)", "dry_departures_per_day": "Vertrekken/droge dag"}
    c1.markdown("**Meest gevoelig**")
    c1.dataframe(effect.nsmallest(10, "effect_pct")[list(cols)].rename(columns=cols).round(1), hide_index=True)
    c2.markdown("**Minst gevoelig**")
    c2.dataframe(effect.nlargest(10, "effect_pct")[list(cols)].rename(columns=cols).round(1), hide_index=True)

# ---------------------------------------------------------------- tab 2: metro distance
with tab_metro:
    st.subheader("Verliezen stations dicht bij de metro meer ritten bij regen?")
    if dist_source == "centre":
        st.warning("Nog geen metrodata gevonden (`data/processed/metro_stations.parquet` of "
                   "`station_metro_distance.parquet`, zie de pagina van Jurre). "
                   "Tijdelijk gegroepeerd op afstand tot het centrum (Bank).")
    dist_col = "metro_distance_m" if "metro_distance_m" in effect.columns else "centre_distance_km"
    dist_label = "Afstand tot dichtstbijzijnde metrostation (m)" if dist_col == "metro_distance_m" \
        else "Afstand tot centrum (km)"

    g = effect.dropna(subset=["metro_group"]).groupby("metro_group", observed=True).apply(
        lambda x: pd.Series({"effect": np.average(x["effect_pct"], weights=x["dry_departures_per_day"]),
                             "stations": len(x)}), include_groups=False).reset_index()
    g["text"] = g.apply(lambda r: f"{r['effect']:+.0f}%  (n={int(r['stations'])})", axis=1)
    fig = px.bar(g, x="metro_group", y="effect", text="text", height=380,
                 labels={"metro_group": dist_label, "effect": "Effect regen (%, gewogen naar drukte)"})
    fig.update_traces(marker_color=RAIN, textposition="outside", cliponaxis=False,
                      hovertemplate="%{x}<br>Effect: %{y:.1f}%<extra></extra>")
    fig.add_hline(y=0, line_color=INK, line_width=1)
    st.plotly_chart(fig)

    rho, p = stats.spearmanr(effect[dist_col], effect["effect_pct"], nan_policy="omit")
    verdict = ("hoe verder van de metro, hoe kleiner het verlies" if rho > 0 else
               "hoe dichter bij de metro, hoe kleiner het verlies")
    signif = "statistisch significant" if p < 0.05 else "niet significant"
    st.markdown(f"**Spearman ρ = {rho:+.2f}** (p = {p:.3g}, {signif}): {verdict}.")

    fig = px.scatter(effect, x=dist_col, y="effect_pct", hover_name="name", opacity=.6, height=380,
                     labels={dist_col: dist_label, "effect_pct": "Effect regen (%)"})
    fig.update_traces(marker=dict(color=RAIN, size=8, line=dict(width=1, color="white")))
    fig.add_hline(y=0, line_color=INK, line_width=1)
    st.plotly_chart(fig)
    st.caption("Let op: we hebben geen dagelijkse metrodata, dus 'overstappen op de metro' is een mogelijke "
               "verklaring, geen bewijs.")

# ---------------------------------------------------------------- tab 3: per hour
with tab_hour:
    st.subheader("Valt vooral woon-werkverkeer of recreatie weg?")
    h = hourly_f[hourly_f["day_type"].isin(day_types)].dropna(subset=["precip_mm"]).copy()
    h["weer"] = np.select([h["precip_mm"] < wi.DRY_MM, h["precip_mm"] >= threshold], ["droog", "regen"], "")
    prof = h[h["weer"] != ""].groupby(["weer", "hour"])["rides"].mean().reset_index()
    fig = px.line(prof, x="hour", y="rides", color="weer", markers=True, height=380,
                  color_discrete_map={"droog": DRY, "regen": RAIN},
                  labels={"hour": "Uur van de dag", "rides": "Gem. ritten per uur", "weer": ""})
    fig.update_traces(line_width=2, marker_size=8)
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig)

    wide = prof.pivot(index="hour", columns="weer", values="rides")
    if {"droog", "regen"} <= set(wide.columns):
        wide["diff"] = (wide["regen"] / wide["droog"] - 1) * 100
        fig = px.bar(wide.reset_index(), x="hour", y="diff", height=300,
                     labels={"hour": "Uur van de dag", "diff": "% verschil regen vs. droog"})
        fig.update_traces(marker_color=RAIN, hovertemplate="%{x}:00<br>%{y:+.1f}%<extra></extra>")
        fig.add_hline(y=0, line_color=INK, line_width=1)
        st.plotly_chart(fig)
        st.caption("Zo lees je het: is de daling in de spits (8 en 17–18 uur) kleiner dan daarbuiten, "
                   "dan vallen vooral recreatieve ritten weg en blijven forenzen fietsen.")

# ---------------------------------------------------------------- tab 4: through the year
with tab_time:
    st.subheader("Ritten per dag en neerslag door het jaar")
    d = daily_f[daily_f["day_type"].isin(day_types)].sort_values("date")
    rainy = d["precip_mm"] >= threshold
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=d["date"], y=d["rides"], mode="lines", name="ritten per dag",
                             line=dict(color=DRY, width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.loc[rainy, "date"], y=d.loc[rainy, "rides"], mode="markers",
                             name=f"regendag (≥{threshold} mm)", marker=dict(color=RAIN, size=8,
                             line=dict(width=1, color="white"))), row=1, col=1)
    fig.add_trace(go.Bar(x=d["date"], y=d["precip_mm"], name="neerslag (mm)", marker_color=RAIN,
                         showlegend=False), row=2, col=1)
    fig.update_yaxes(title_text="ritten", row=1, col=1)
    fig.update_yaxes(title_text="mm", row=2, col=1)
    fig.update_layout(height=520, hovermode="x unified", legend=dict(orientation="h", y=1.05),
                      margin=dict(t=40))
    st.plotly_chart(fig)

# ---------------------------------------------------------------- tab 5: what-if
with tab_whatif:
    st.subheader("Wat als? Voorspel het aantal ritten voor een dag")
    model, wind_col = model_for(daily)
    c1, c2, c3 = st.columns(3)
    mm = c1.slider("Neerslag (mm)", 0.0, 25.0, 5.0, 0.5)
    temp = c1.slider("Gemiddelde temperatuur (°C)", -5.0, 30.0, 12.0, 0.5)
    wind = c2.slider("Windsnelheid (km/u)", 0.0, 50.0, 15.0, 1.0, disabled=wind_col is None)
    wd = c2.selectbox("Dag", range(7), format_func=lambda i: WEEKDAYS[i], index=1)
    mo = c3.selectbox("Maand", range(1, 13), format_func=lambda i: MONTHS[i - 1], index=9)
    hol = c3.checkbox("Feestdag")

    est, lo, hi = wi.predict_rides(model, mm, temp, wd, mo, hol, wind, wind_col)
    base, _, _ = wi.predict_rides(model, 0.0, temp, wd, mo, hol, wind, wind_col)
    r1, r2 = st.columns(2)
    r1.metric("Verwachte ritten", f"{est:,.0f}", f"{(est / base - 1) * 100:+.0f}% t.o.v. droge dag",
              delta_color="normal")
    r2.metric("95%-voorspelinterval", f"{lo:,.0f} – {hi:,.0f}")

    grid = np.arange(0, 25.5, 0.5)
    preds = [wi.predict_rides(model, x, temp, wd, mo, hol, wind, wind_col)[0] for x in grid]
    fig = go.Figure(go.Scatter(x=grid, y=preds, mode="lines", line=dict(color=RAIN, width=2), name="voorspelling",
                               hovertemplate="%{x} mm → %{y:,.0f} ritten<extra></extra>"))
    fig.add_trace(go.Scatter(x=[mm], y=[est], mode="markers", marker=dict(size=12, color=RAIN,
                             line=dict(width=2, color="white")), name="jouw keuze", hoverinfo="skip"))
    fig.update_layout(height=340, xaxis_title="Neerslag (mm)", yaxis_title="Verwachte ritten",
                      showlegend=False, margin=dict(t=10))
    st.plotly_chart(fig)

    with st.expander("Hoe werkt het model?"):
        rows = []
        for var, unit in [("precip_mm", "per mm regen"), ("temp_mean", "per °C warmer"),
                          (wind_col, "per km/u meer wind")]:
            if var:
                b = model.params[var]; l, u = model.conf_int().loc[var]
                rows.append({"Factor": unit, "Effect": f"{(np.exp(b) - 1) * 100:+.1f}%",
                             "95%-BI": f"{(np.exp(l) - 1) * 100:+.1f}% t/m {(np.exp(u) - 1) * 100:+.1f}%"})
        st.table(pd.DataFrame(rows))
        st.markdown(f"Lineaire regressie op log(ritten) met controle voor weekdag, maand en feestdag. "
                    f"R² = **{model.rsquared:.2f}**, getraind op {int(model.nobs)} dagen.")
