# Kenya maize yield and climate anomalies: 2026 short-rains scenario

Machine learning pipeline linking seasonal climate (ERA5-Land) to county-level maize yield in Kenya (2012-2020), then applying it to the 2026 October-December (OND) seasonal forecast to map baseline yield, predicted yield and estimated impact for the 47 counties.

**Status:** exploratory. Yield is an annual total, the record is short, and model skill is modest (leave-one-year-out R2 about 0.39). Read the results as a scenario, not a yield forecast.

Author: [Your name]

## Pipeline (run in this order)

Each script has a CONFIG section at the top for file paths. Run them from the project folder.

| # | Script | Reads | Writes |
|---|---|---|---|
| 1 | `ond_county_features.py` | `kenya_ond_2012_2020.nc`, `counties.shp` | `kenya_county_climate_features.csv` |
| 2 | `merge_yield_climate.py` | the CSV above, `Maize-Production-2012-2020-Combined.xlsx` | `kenya_maize_climate_modeling_dataset.csv` |
| 3 | `mam_county_features.py` | `kenya_mam_2012_2020.nc`, the dataset above | `kenya_county_mam_features.csv`, `kenya_maize_climate_modeling_dataset_ond_mam.csv` |
| 4 | `train_evaluate_model.py` | the OND + MAM dataset | printed evaluation (leave-one-year-out, 2015/16 holdout) |
| 5 | `predict_2026_and_map.py` | the OND + MAM dataset, `kenya_ond_2026_forecast.nc`, `kenya_ond_hindcast_climate_mean.nc`, `kenya_mam_2026.nc`, `counties.shp` | `kenya_2026_maize_predictions_final.csv`, `kenya_2026_maize_maps_final.png` |

## Data (not stored in this repository)

The NetCDF climate files are excluded because they are large and can be re-downloaded. They need a free Copernicus Climate Data Store account and API key, saved in your home folder (never in this project).

- **ERA5-Land monthly averaged data** (`reanalysis-era5-land-monthly-means`): 2m temperature and total precipitation; OND (months 10-12) and MAM (months 3-5), years 2012-2020, plus MAM 2026. Area `[5.5, 33.5, -5, 42]` (N, W, S, E).
- **SEAS5 seasonal forecast** (`seasonal-monthly-single-levels`, ECMWF system 51): October 2026 start, lead months 1-3, area as above. Two downloads: `ensemble_mean` (the 2026 forecast) and `hindcast_climate_mean` (the model climatology used to compute anomalies).
- **Maize yield:** KNBS/Kilimodata, Maize Production by County 2012-2020 (county, year, area, production, yield).
- **County boundaries:** 47-county shapefile (`counties.*`).

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Main limitations

- Yield is annual (long plus short rains), while climate features cover MAM and OND only.
- Only 2012-2020 yield data (2019 missing), so 2015/16 is the only Super El Nino event; 1997/98 is not covered.
- The 2026 forecast is a single model at 1 degree resolution; many neighbouring counties share the same values.
- The model sees only temperature and rainfall, and partly relies on raw temperature as a county proxy.

See the full project report for details.
