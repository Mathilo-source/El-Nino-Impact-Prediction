"""
step 5: Merge county-level climate features (from Step 4) with maize yield data
into one final modeling dataset.

Inputs:
    - kenya_county_climate_features.csv   (output of ond_county_features script)
    - Maize-Production-2012-2020-Combined.xlsx (raw yield data)

Output:
    - kenya_maize_climate_modeling_dataset.csv
      One row per county-year, with climate features/anomalies + yield/area/production.
"""

import pandas as pd

# ----------------------- CONFIG: EDIT THESE PATHS -----------------------
CLIMATE_FEATURES_CSV = "kenya_county_climate_features.csv"
YIELD_XLSX_PATH = "Maize-Production-2012-2020-Combined.xlsx"
OUTPUT_CSV_PATH = "kenya_maize_climate_modeling_dataset.csv"
# --------------------------------------------------------------------------


def load_climate_features(path):
    df = pd.read_csv(path)

    # Fix ERA5 sign convention: pev is negative (water leaving surface).
    # Flip so larger positive values = more evaporative demand, matching
    # the intuitive direction of t2m/tp.
    df["pev_mm"] = -df["pev_mm"]
    df["pev_mm_anom"] = -df["pev_mm_anom"]

    return df


def load_and_pivot_yield(path):
    """
    Raw yield data is in long format: one row per county-year-indicator
    (Area (HA), Production (MT), Yield(MT/HA)). Pivot so each indicator
    becomes its own column, one row per county-year.
    """
    df = pd.read_excel(path, sheet_name="Sheet4")
    df["county_clean"] = df["County"].str.title().str.strip()

    wide = df.pivot_table(
        index=["county_clean", "Year"],
        columns="Indicator",
        values="Value",
    ).reset_index()

    wide = wide.rename(
        columns={
            "county_clean": "county",
            "Year": "year",
            "Area (HA)": "area_ha",
            "Production (MT)": "production_mt",
            "Yield(MT/HA)": "yield_mt_ha",
        }
    )
    return wide


def merge_datasets(climate_df, yield_df):
    merged = pd.merge(
        climate_df,
        yield_df,
        on=["county", "year"],
        how="outer",
        indicator=True,
    )
    return merged


def report_merge_quality(merged):
    print("Merge match report:")
    print(merged["_merge"].value_counts())
    print()

    only_climate = merged[merged["_merge"] == "left_only"]
    only_yield = merged[merged["_merge"] == "right_only"]

    if len(only_climate) > 0:
        print(f"⚠️  {len(only_climate)} rows had climate data but no matching yield row:")
        print(only_climate[["county", "year"]].to_string(index=False))
        print()

    if len(only_yield) > 0:
        print(f"⚠️  {len(only_yield)} rows had yield data but no matching climate row:")
        print(only_yield[["county", "year"]].to_string(index=False))
        print()

    if len(only_climate) == 0 and len(only_yield) == 0:
        print("✅ Every county-year matched on both sides.")


def main():
    print("Loading climate features...")
    climate_df = load_climate_features(CLIMATE_FEATURES_CSV)

    print("Loading and pivoting yield data...")
    yield_df = load_and_pivot_yield(YIELD_XLSX_PATH)

    print("Merging on county + year...\n")
    merged = merge_datasets(climate_df, yield_df)

    report_merge_quality(merged)

    final = merged.drop(columns=["_merge"])
    final = final.sort_values(["county", "year"]).reset_index(drop=True)
    final.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"\n✅ Saved {len(final)} rows to {OUTPUT_CSV_PATH}")
    print(final.head(10))


if __name__ == "__main__":
    main()
