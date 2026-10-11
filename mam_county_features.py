"""
Step 4b: Process the MAM (March-May) ERA5-Land file into county-level features
and merge them into the existing OND modeling dataset.

Inputs
    kenya_mam_2012_2020.nc                          (new MAM data)
    counties shapefile
    kenya_maize_climate_modeling_dataset.csv        (merge_yield_climate: OND + yield)

Outputs
    kenya_county_mam_features.csv                   (county, year, MAM features)
    kenya_maize_climate_modeling_dataset_ond_mam.csv   (OND + MAM + yield)

MAM feature columns (suffix _mam keeps them apart from the OND columns):
    t2m_c_mam, tp_mm_mam, t2m_c_mam_anom, tp_mm_mam_anom
Anomalies use each county's own 2012-2020 MAM mean as the baseline,
exactly as the OND anomalies do.
"""

import geopandas as gpd
import xarray as xr
import regionmask
import pandas as pd

# ----------------------- CONFIG: EDIT THESE PATHS -----------------------
MAM_NC_PATH = "kenya_mam_2012_2020.nc"
SHAPEFILE_PATH = "counties.shp"
OND_DATASET_CSV = "kenya_maize_climate_modeling_dataset.csv"
MAM_FEATURES_CSV = "kenya_county_mam_features.csv"
MERGED_CSV = "kenya_maize_climate_modeling_dataset_ond_mam.csv"
# --------------------------------------------------------------------------


def load_and_clean_shapefile(path):
    gdf = gpd.read_file(path)
    gdf = gdf[gdf["COUNTY_COD"] != 0].reset_index(drop=True)
    gdf["county_clean"] = gdf["COUNTY_NAM"].str.title().str.strip()
    gdf["county_clean"] = gdf["county_clean"].replace({
        "Elegeyo-Marakwet": "Elgeyo/Marakwet",
        "Tharaka - Nithi": "Tharaka-Nthi",
        "Homa Bay": "Homabay",
        "Taita Taveta": "Taita/Taveta",
    })
    return gdf


def seasonal_aggregates(ds):
    """Monthly means -> one value per year.
    t2m: mean of the 3 months, K -> C.
    tp : each month is a MEAN DAILY accumulation (m/day); x days-in-month x 1000
         gives mm for that month; SUM the 3 months for the season total."""
    days = xr.DataArray(
        pd.DatetimeIndex(ds["valid_time"].values).days_in_month.values,
        dims="valid_time", coords={"valid_time": ds["valid_time"]},
    )
    year = ds["valid_time"].dt.year
    t2m_season = (ds["t2m"] - 273.15).groupby(year).mean("valid_time")
    tp_season = (ds["tp"] * days * 1000).groupby(year).sum("valid_time")
    return xr.Dataset({"t2m_c_mam": t2m_season, "tp_mm_mam": tp_season})


def county_values(seasonal, gdf):
    regions = regionmask.from_geopandas(
        gdf, names="county_clean", abbrevs="_from_name", name="kenya_counties"
    )
    # drop=False keeps all 47 counties; look layers up by region NUMBER
    mask_3d = regions.mask_3D(seasonal["longitude"], seasonal["latitude"], drop=False)

    rows = []
    for i, county in enumerate(regions.names):
        cmask = mask_3d.sel(region=regions.numbers[i])
        if int(cmask.sum()) == 0:
            raise ValueError(f"{county}: no grid cell inside it (unexpected on a 0.1 degree grid)")
        for var in ["t2m_c_mam", "tp_mm_mam"]:
            vals = seasonal[var].where(cmask).mean(dim=["latitude", "longitude"])
            for yr, v in zip(vals["year"].values, vals.values):
                rows.append({"county": county, "year": int(yr), "variable": var, "value": float(v)})
    long_df = pd.DataFrame(rows)
    return long_df.pivot_table(index=["county", "year"], columns="variable",
                               values="value").reset_index()


def add_anomalies(df):
    for var in ["t2m_c_mam", "tp_mm_mam"]:
        df[f"{var}_anom"] = df[var] - df.groupby("county")[var].transform("mean")
    return df


def main():
    print("Loading shapefile and MAM NetCDF...")
    gdf = load_and_clean_shapefile(SHAPEFILE_PATH)
    ds = xr.open_dataset(MAM_NC_PATH)
    print(ds, "\n")

    months = sorted(set(pd.DatetimeIndex(ds["valid_time"].values).month))
    if months != [3, 4, 5]:
        print(f"WARNING: expected months [3, 4, 5], found {months}. Check your download.")

    seasonal = seasonal_aggregates(ds)
    mam = add_anomalies(county_values(seasonal, gdf))
    mam = mam.sort_values(["county", "year"]).reset_index(drop=True)
    mam.to_csv(MAM_FEATURES_CSV, index=False)
    print(f"Saved {len(mam)} MAM rows to {MAM_FEATURES_CSV} (expected 47 x 9 = 423)")

    ond = pd.read_csv(OND_DATASET_CSV)
    merged = ond.merge(mam, on=["county", "year"], how="left", validate="one_to_one")
    n_missing = merged["t2m_c_mam"].isna().sum()
    print(f"Merged with OND dataset: {len(merged)} rows, {n_missing} without MAM values")
    if n_missing:
        print(merged.loc[merged["t2m_c_mam"].isna(), ["county", "year"]].head(20))
    merged.to_csv(MERGED_CSV, index=False)
    print(f"Saved {MERGED_CSV}\n")

    cols = ["county", "year", "t2m_c", "tp_mm", "t2m_c_mam", "tp_mm_mam",
            "t2m_c_mam_anom", "tp_mm_mam_anom", "yield_mt_ha"]
    print(merged[cols].head(10).round(2).to_string(index=False))
    print("\nSanity check: MAM rainfall (long rains) is usually larger than OND in most "
          "of the country, and the highlands are cooler than the lowlands.")


if __name__ == "__main__":
    main()