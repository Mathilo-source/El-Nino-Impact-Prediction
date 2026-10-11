"""
step 4: Turns the Kenya ERA5-Land OND climate grid into county-level values,
then compute anomalies relative to each county's own 2012-2020 OND baseline.

Inputs:
    - kenya_ond_2012_2020.nc      (your ERA5-Land climate file)
    - counties.shp (+ .shx/.dbf/.prj)  (Kenya counties shapefile)
    - Maize-Production-2012-2020-Combined.xlsx (your yield file, for name-matching check)

Output:
    - kenya_county_climate_features.csv
      columns: county, year, t2m_c, tp_mm, swvl1, pev_mm,
               t2m_c_anom, tp_mm_anom, swvl1_anom, pev_mm_anom

"""

import geopandas as gpd
import xarray as xr
import regionmask
import pandas as pd
import numpy as np

# ----------------------- CONFIG: EDIT THESE PATHS -----------------------
CLIMATE_NC_PATH = "kenya_ond_2012_2020.nc"
SHAPEFILE_PATH = "counties.shp"
YIELD_XLSX_PATH = "Maize-Production-2012-2020-Combined.xlsx"
OUTPUT_CSV_PATH = "kenya_county_climate_features.csv"
# --------------------------------------------------------------------------


def load_and_clean_shapefile(path):
    """Load counties shapefile, drop the artifact row, standardize names."""
    gdf = gpd.read_file(path)

    # Drop the non-county artifact row (COUNTY_COD == 0, no name)
    gdf = gdf[gdf["COUNTY_COD"] != 0].reset_index(drop=True)

    # Standardize to Title Case, stripped, for matching against yield data
    gdf["county_clean"] = gdf["COUNTY_NAM"].str.title().str.strip()

    # Fix known spelling mismatches vs. the yield spreadsheet's naming
    # (edit this dict if you find more mismatches during the name-check step)
    rename_map = {
        "Elegeyo-Marakwet": "Elgeyo/Marakwet",
        "Tharaka - Nithi": "Tharaka-Nthi",
        "Homa Bay": "Homabay",
        "Taita Taveta": "Taita/Taveta",
    }
    gdf["county_clean"] = gdf["county_clean"].replace(rename_map)

    return gdf


def load_and_clean_yield(path):
    """Load yield spreadsheet, standardize county names the same way."""
    df = pd.read_excel(path, sheet_name="Sheet4")
    df["county_clean"] = df["County"].str.title().str.strip()
    return df


def check_name_match(climate_names, yield_names):
    """Print any counties that don't match between the two datasets."""
    climate_set = set(climate_names)
    yield_set = set(yield_names)

    only_in_climate = climate_set - yield_set
    only_in_yield = yield_set - climate_set

    print(f"Counties in shapefile: {len(climate_set)}")
    print(f"Counties in yield data: {len(yield_set)}")

    if only_in_climate:
        print("\n⚠️  In shapefile but NOT in yield data:")
        for name in sorted(only_in_climate):
            print(f"   - {name}")
    if only_in_yield:
        print("\n⚠️  In yield data but NOT in shapefile:")
        for name in sorted(only_in_yield):
            print(f"   - {name}")
    if not only_in_climate and not only_in_yield:
        print("\n✅ All county names match perfectly.")

    return len(only_in_climate) == 0 and len(only_in_yield) == 0


def compute_seasonal_aggregates(ds):
    """
    Convert monthly ERA5-Land values into one OND-season value per year,
    with correct units, for each variable.

    - t2m:   mean of Oct/Nov/Dec monthly means, K -> Celsius
    - tp:    each month's value is the MEAN DAILY accumulation (m/day);
             multiply by days-in-month to get that month's total, then
             SUM the three months for a season total, in mm
    - pev:   same treatment as tp (mean daily -> monthly total -> season sum), mm
    - swvl1: mean of Oct/Nov/Dec monthly means (already a sensible fraction, no unit conversion)
    """
    days_in_month = xr.DataArray(
        [d.days_in_month for d in pd.DatetimeIndex(ds["valid_time"].values)],
        dims="valid_time",
        coords={"valid_time": ds["valid_time"]},
    )

    # Monthly totals for accumulated variables (m -> mm via *1000)
    tp_monthly_total_mm = ds["tp"] * days_in_month * 1000
    pev_monthly_total_mm = ds["pev"] * days_in_month * 1000

    t2m_celsius = ds["t2m"] - 273.15

    ds = ds.assign(
        tp_monthly_mm=tp_monthly_total_mm,
        pev_monthly_mm=pev_monthly_total_mm,
        t2m_celsius=t2m_celsius,
    )

    # Group by year, summing OND months for precip/PET, averaging for temp/soil moisture
    year = ds["valid_time"].dt.year

    tp_season = ds["tp_monthly_mm"].groupby(year).sum("valid_time")
    pev_season = ds["pev_monthly_mm"].groupby(year).sum("valid_time")
    t2m_season = ds["t2m_celsius"].groupby(year).mean("valid_time")
    swvl1_season = ds["swvl1"].groupby(year).mean("valid_time")

    seasonal = xr.Dataset(
        {
            "t2m_c": t2m_season,
            "tp_mm": tp_season,
            "swvl1": swvl1_season,
            "pev_mm": pev_season,
        }
    )
    seasonal = seasonal.rename({"year": "year"})
    return seasonal


def extract_county_values(seasonal_ds, gdf):
    """Apply county mask to the seasonal grid and average within each county."""
    regions = regionmask.from_geopandas(
        gdf, names="county_clean", abbrevs="_from_name", name="kenya_counties"
    )
    # Use mask_3D: gives one boolean layer per county rather than forcing
    # every grid cell into a single 2D map. This avoids the overlap
    # ambiguity entirely and works across regionmask versions.
    mask_3d = regions.mask_3D(seasonal_ds["longitude"], seasonal_ds["latitude"])

    records = []
    for region_id, county_name in zip(mask_3d["region"].values, regions.names):
        county_mask = mask_3d.sel(region=region_id)
        for var in ["t2m_c", "tp_mm", "swvl1", "pev_mm"]:
            county_vals = seasonal_ds[var].where(county_mask).mean(
                dim=["latitude", "longitude"]
            )
            for yr, val in zip(county_vals["year"].values, county_vals.values):
                records.append(
                    {"county": county_name, "year": int(yr), "variable": var, "value": float(val)}
                )

    long_df = pd.DataFrame(records)
    wide_df = long_df.pivot_table(
        index=["county", "year"], columns="variable", values="value"
    ).reset_index()
    return wide_df


def add_anomalies(df):
    """For each county, subtract that county's own 2012-2020 mean from each year's value."""
    for var in ["t2m_c", "tp_mm", "swvl1", "pev_mm"]:
        baseline = df.groupby("county")[var].transform("mean")
        df[f"{var}_anom"] = df[var] - baseline
    return df


def main():
    print("Loading and cleaning shapefile...")
    gdf = load_and_clean_shapefile(SHAPEFILE_PATH)

    print("Loading and cleaning yield data...")
    yield_df = load_and_clean_yield(YIELD_XLSX_PATH)

    print("\nChecking county name matches...")
    ok = check_name_match(gdf["county_clean"], yield_df["county_clean"])
    if not ok:
        print(
            "\n>>> Fix the rename_map in load_and_clean_shapefile() before trusting "
            "the merge, then re-run. Continuing anyway for now...\n"
        )

    print("Loading climate NetCDF...")
    ds = xr.open_dataset(CLIMATE_NC_PATH)

    print("Computing OND seasonal aggregates (unit conversions + season totals/means)...")
    seasonal_ds = compute_seasonal_aggregates(ds)

    print("Extracting county-level values via zonal statistics...")
    county_df = extract_county_values(seasonal_ds, gdf)

    print("Computing anomalies relative to each county's 2012-2020 OND baseline...")
    county_df = add_anomalies(county_df)

    county_df = county_df.sort_values(["county", "year"]).reset_index(drop=True)
    county_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"\n✅ Done. Saved {len(county_df)} rows to {OUTPUT_CSV_PATH}")
    print(county_df.head(10))


if __name__ == "__main__":
    main()
