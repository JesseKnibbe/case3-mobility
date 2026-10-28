# Case 3 – Bikeshare in Londen

**Hoofdvraag:** Zorgt regen ervoor dat de Bikeshare minder gebruikt wordt?
Subvragen: afstand tot de metro en weer.

## Structuur

```
Start.py              hoofdbestand dashboard (navigatie)
app_pages/            de 5 pagina's van het dashboard
notebooks/            01 inladen · 02 opschonen · 04 subvraag weer + metro-afstand (optioneel)
weather_impact.py     gedeelde rekenfuncties van het dashboard
data/processed/       kleine bestanden die het dashboard gebruikt
data/raw/             ruwe data: weather/ en metro/ staan op GitHub, tfl/ niet (te groot)
```

Volgorde notebooks: 01 → 02 De analyse van de hoofdvraag staat in het dashboard
(pagina *Weather vs Rental*, tabblad *Hoofdvraag*).

De weer- en metrodata staan in `data/raw/` op GitHub. De TfL-ritdata (±1,5 GB) en `rides_raw.parquet` (±180 MB)
zijn te groot voor GitHub: notebook 01 downloadt en maakt ze automatisch opnieuw aan (eenmalig ±10–15 minuten).
