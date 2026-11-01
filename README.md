# Case 3 – Bikeshare in Londen

**Hoofdvraag:** Zorgt regen ervoor dat de Bikeshare minder gebruikt wordt?
Subvragen: afstand tot de metro en weer.

## Structuur

```
Start.py              hoofdbestand dashboard (navigatie)
app_pages/            de 5 pagina's van het dashboard
notebooks/            01 inladen · 02 opschonen
weather_impact.py     gedeelde rekenfuncties van het dashboard
data/processed/       kleine bestanden die het dashboard gebruikt
data/raw/             ruwe data: weather/, metro/ en geo/ staan op GitHub, tfl/ niet (te groot)
```

## Draaien

```bash
pip install -r requirements.txt
streamlit run Start.py
```

Volgorde notebooks: 01 → 02. Alle analyses staan in het dashboard.

De weer- en metrodata staan in `data/raw/` op GitHub. De TfL-ritdata en `rides_raw.parquet` 
zijn te groot voor GitHub: notebook 01 downloadt en maakt ze automatisch opnieuw aan 
