# Case 3 – Bikeshare in Londen

**Hoofdvraag:** Zorgt regen ervoor dat de Bikeshare minder gebruikt wordt?

- Subvraag metro-afstand: Jurre
- Subvraag weer: Jesse

## Structuur

```
Start.py                     startpagina (onderzoeksvraag, data, opschonen)
pages/1_Metro_afstand.py     subvraag Jurre
pages/2_Weer_en_regen.py     subvraag Jesse
weather_impact.py            gedeelde rekenfuncties
01–04_*.ipynb                notebooks: inladen, opschonen, analyse, subvraag weer
data/processed/              kleine bestanden die het dashboard gebruikt
data/raw/                    ruwe data (NIET op GitHub, zie .gitignore)
```

## Lokaal draaien

```bash
pip install -r requirements.txt
streamlit run Start.py
```

Draai eerst notebook 01 en 02 (en 04), zodat `data/processed/` gevuld is.

## Metrodata

Sla de metrostations op als `data/processed/metro_stations.parquet` (kolommen `name`, `lat`, `lon`).
De afstand van elk fietsstation tot het dichtstbijzijnde metrostation wordt dan automatisch berekend.
Wil je eigen afstandsgroepen, sla dan `data/processed/station_metro_distance.parquet` op met
`bikepoint_id`, `metro_distance_m` en `metro_group`; dat bestand gaat dan voor.
