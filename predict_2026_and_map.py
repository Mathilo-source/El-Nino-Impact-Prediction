"""
Step 10: Final 2026 prediction using BOTH seasons, with corrected anomalies.

Inputs (all in the same folder, or edit the paths):
    kenya_maize_climate_modeling_dataset_ond_mam.csv   (training data, step4b output)
    kenya_ond_2026_forecast.nc                         (SEAS5 ensemble mean, Oct start, leads 1-3)
    kenya_ond_hindcast_climate_mean.nc                 (SEAS5 hindcast climate mean, same request)
    kenya_mam_2026.nc                                  (ERA5-Land observed Mar-May 2026)
    counties shapefile

How the 2026 inputs are built
  OND 2026 (forecast):
      anomaly  = SEAS5 2026 forecast - SEAS5 hindcast climate mean
      raw value = county's observed 2012-2020 ERA5-Land OND mean + anomaly
      (the forecast model's own climate cancels out its bias and grid effects)
  MAM 2026 (observed):
      ERA5-Land values, anomaly = 2026 value - county's 2012-2020 MAM mean

Predictions per county (annual maize yield, t/ha)
  yield_2026_pred       OND forecast + observed MAM 2026
  yield_baseline_pred   normal climate in both seasons (all anomalies = 0)
  yield_ond_only_pred   OND forecast anomalies only, MAM kept at normal
                        -> isolates the El Nino-season (OND) effect
  impact_total_pct      (2026 - baseline) / baseline * 100
  impact_ond_only_pct   (OND-only - baseline) / baseline * 100

Outputs
    kenya_2026_maize_predictions_final.csv
    kenya_2026_maize_maps_final.png
"""

import calendar
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import regionmask
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ----------------------- CONFIG: EDIT THESE PATHS -----------------------
TRAIN_CSV = "kenya_maize_climate_modeling_dataset_ond_mam.csv"
FORECAST_NC = "OND2026forecastdata.nc"
HINDCAST_NC = "kenya_ond_hindcast_climate_mean.nc"
MAM_2026_NC = "MAM2026data.nc"
SHAPEFILE_PATH = "counties.shp"
OUTPUT_CSV = "kenya_2026_maize_predictions_final.csv"
OUTPUT_PNG = "kenya_2026_maize_maps_final.png"
YEAR = 2026
# --------------------------------------------------------------------------

TARGET = "yield_mt_ha"
FEATURES = [
    "t2m_c", "tp_mm", "t2m_c_anom", "tp_mm_anom",                       # OND
    "t2m_c_mam", "tp_mm_mam", "t2m_c_mam_anom", "tp_mm_mam_anom",       # MAM
]
BASE_COLS = ["t2m_c", "tp_mm", "t2m_c_mam", "tp_mm_mam"]

# Map class breaks: adjust once you see the real prediction range.
#YIELD_BINS = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]    # t/ha
#IMPACT_BINS = [-100, -40, -20, -10, 10, 20, 40, 100]    # %
YIELD_BINS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.5]
IMPACT_BINS = [-40, -20, -10, -5, 5, 10, 20, 40, 70]

def new_model():
    return RandomForestRegressor(n_estimators=300, random_state=42)


# ------------------------------ shapefile ------------------------------
def load_shapefile(path):
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


# --------------------- seasonal aggregation from files ---------------------
def seasonal_from_rates(ds, year=YEAR):
    """SEAS5 files (forecast or hindcast climate mean): forecastMonth 1,2,3 = Oct,Nov,Dec.
    tprate is a RATE in m/s -> x seconds in month x 1000 = mm; sum over the season.
    t2m: K -> C, mean over the season."""
    need = {"t2m", "tprate", "forecastMonth"}
    have = set(ds.variables)
    if not need <= have:
        raise ValueError(f"Expected {sorted(need)} but the file has {sorted(have)}")
    fms = sorted(int(x) for x in ds["forecastMonth"].values)
    if fms != [1, 2, 3]:
        raise ValueError(f"Expected forecastMonth [1, 2, 3], found {fms}")
    month_of = {1: 10, 2: 11, 3: 12}
    secs = xr.DataArray(
        [calendar.monthrange(year, month_of[int(fm)])[1] * 86400 for fm in ds["forecastMonth"].values],
        dims="forecastMonth", coords={"forecastMonth": ds["forecastMonth"]},
    )
    tp = (ds["tprate"] * secs * 1000).sum("forecastMonth")
    t2m = (ds["t2m"] - 273.15).mean("forecastMonth")
    return xr.Dataset({"t2m_c": t2m, "tp_mm": tp}).squeeze(drop=True)


def mam_seasonal(ds):
    """ERA5-Land monthly means for Mar-May 2026 -> season values (same units logic as training)."""
    times = pd.DatetimeIndex(ds["valid_time"].values)
    if sorted(set(times.month)) != [3, 4, 5] or sorted(set(times.year)) != [YEAR]:
        raise ValueError(f"Expected Mar-May {YEAR} only; found months {sorted(set(times.month))}, "
                         f"years {sorted(set(times.year))}")
    days = xr.DataArray(times.days_in_month.values, dims="valid_time",
                        coords={"valid_time": ds["valid_time"]})
    t2m = (ds["t2m"] - 273.15).mean("valid_time")
    tp = (ds["tp"] * days * 1000).sum("valid_time")
    return xr.Dataset({"t2m_c_mam": t2m, "tp_mm_mam": tp}).squeeze(drop=True)


def county_values(seasonal, gdf, varnames):
    """Mean of grid cells inside each county; counties containing no cell centre
    (common on the 1-degree SEAS5 grid) use the cell nearest their centroid."""
    regions = regionmask.from_geopandas(
        gdf, names="county_clean", abbrevs="_from_name", name="kenya_counties")
    mask_3d = regions.mask_3D(seasonal["longitude"], seasonal["latitude"], drop=False)
    centroids = gdf.to_crs(epsg=32637).geometry.centroid.to_crs(epsg=4326)
    rows = []
    for i, county in enumerate(regions.names):
        cmask = mask_3d.sel(region=regions.numbers[i])
        row = {"county": county}
        for v in varnames:
            if int(cmask.sum()) > 0:
                val = seasonal[v].where(cmask).mean(dim=["latitude", "longitude"])
            else:
                val = seasonal[v].sel(latitude=centroids.iloc[i].y,
                                      longitude=centroids.iloc[i].x, method="nearest")
            row[v] = float(val.values)
        rows.append(row)
    return pd.DataFrame(rows).set_index("county")


# ------------------------------ features ------------------------------
def build_scenarios(train_all, ond_fc, ond_hc, mam):
    base = train_all.groupby("county")[BASE_COLS].mean()
    ond_anom = ond_fc[["t2m_c", "tp_mm"]] - ond_hc[["t2m_c", "tp_mm"]].loc[ond_fc.index]

    missing = set(ond_anom.index) ^ set(base.index)
    if missing:
        raise ValueError(f"County names differ between training data and shapefile: {sorted(missing)}")
    base, ond_anom, mam = base.sort_index(), ond_anom.sort_index(), mam.sort_index()

    print("2026 anomalies (should look like a plausible season, not +/-5 C):")
    print(f"  OND temp anomaly   : {ond_anom['t2m_c'].min():+.2f} to {ond_anom['t2m_c'].max():+.2f} C "
          f"(mean {ond_anom['t2m_c'].mean():+.2f})")
    print(f"  OND rain anomaly   : {ond_anom['tp_mm'].min():+.0f} to {ond_anom['tp_mm'].max():+.0f} mm "
          f"(mean {ond_anom['tp_mm'].mean():+.0f})")
    mam_t = mam["t2m_c_mam"] - base["t2m_c_mam"]
    mam_p = mam["tp_mm_mam"] - base["tp_mm_mam"]
    print(f"  MAM temp anomaly   : {mam_t.min():+.2f} to {mam_t.max():+.2f} C (mean {mam_t.mean():+.2f})")
    print(f"  MAM rain anomaly   : {mam_p.min():+.0f} to {mam_p.max():+.0f} mm (mean {mam_p.mean():+.0f})\n")

    def frame(ond_t, ond_p, mam_t_, mam_p_):
        f = pd.DataFrame(index=base.index)
        f["t2m_c"] = base["t2m_c"] + ond_t
        f["tp_mm"] = (base["tp_mm"] + ond_p).clip(lower=0)
        f["t2m_c_anom"] = f["t2m_c"] - base["t2m_c"]
        f["tp_mm_anom"] = f["tp_mm"] - base["tp_mm"]
        f["t2m_c_mam"] = base["t2m_c_mam"] + mam_t_
        f["tp_mm_mam"] = (base["tp_mm_mam"] + mam_p_).clip(lower=0)
        f["t2m_c_mam_anom"] = f["t2m_c_mam"] - base["t2m_c_mam"]
        f["tp_mm_mam_anom"] = f["tp_mm_mam"] - base["tp_mm_mam"]
        return f[FEATURES]

    zero = pd.Series(0.0, index=base.index)
    full = frame(ond_anom["t2m_c"], ond_anom["tp_mm"], mam_t, mam_p)
    ond_only = frame(ond_anom["t2m_c"], ond_anom["tp_mm"], zero, zero)
    normal = frame(zero, zero, zero, zero)
    return full, ond_only, normal


# ------------------------------ model ------------------------------
def quick_cv(train):
    r2s = []
    for yr in sorted(train["year"].unique()):
        tr, te = train[train["year"] != yr], train[train["year"] == yr]
        m = new_model().fit(tr[FEATURES], tr[TARGET])
        r2s.append(r2_score(te[TARGET], m.predict(te[FEATURES])))
    ni = [2015, 2016]
    tr, te = train[~train["year"].isin(ni)], train[train["year"].isin(ni)]
    m = new_model().fit(tr[FEATURES], tr[TARGET])
    print(f"Model check: leave-one-year-out R2 = {np.mean(r2s):.3f}, "
          f"2015/16 El Nino holdout R2 = {r2_score(te[TARGET], m.predict(te[FEATURES])):.3f}\n")


def predict(train, full, ond_only, normal):
    model = new_model().fit(train[FEATURES], train[TARGET])
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("Feature importances (final model):")
    for k, v in imp.items():
        print(f"  {k:15s} {v:.3f}")
    print()

    res = pd.DataFrame(index=full.index)
    res["yield_2026_pred"] = model.predict(full)
    res["yield_baseline_pred"] = model.predict(normal)
    res["yield_ond_only_pred"] = model.predict(ond_only)
    base = res["yield_baseline_pred"].where(res["yield_baseline_pred"] > 0.1)
    res["impact_total_pct"] = (res["yield_2026_pred"] - res["yield_baseline_pred"]) / base * 100
    res["impact_ond_only_pct"] = (res["yield_ond_only_pred"] - res["yield_baseline_pred"]) / base * 100
    res["ond_t2m_anom"] = full["t2m_c_anom"]
    res["ond_tp_anom_mm"] = full["tp_mm_anom"]
    res["mam_t2m_anom"] = full["t2m_c_mam_anom"]
    res["mam_tp_anom_mm"] = full["tp_mm_mam_anom"]
    return res.reset_index()


# ------------------------------ maps ------------------------------
def banded(bins):
    n = len(bins) - 1
    return ListedColormap(plt.cm.RdYlGn(np.linspace(0.05, 0.95, n))), BoundaryNorm(bins, n)


def draw(ax, g, col, bins, title, label):
    cmap, norm = banded(bins)
    g.plot(column=col, cmap=cmap, norm=norm, edgecolor="black", linewidth=0.4, ax=ax,
           missing_kwds={"color": "lightgrey"})
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    cb = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.04, pad=0.02, ticks=bins)
    cb.set_label(label)


def make_maps(gdf, res, out_png):
    g = gdf.merge(res, left_on="county_clean", right_on="county", how="left")
    fig, axes = plt.subplots(2, 2, figsize=(14, 15))
    draw(axes[0, 0], g, "yield_2026_pred", YIELD_BINS, "Predicted annual maize yield, 2026", "t/ha")
    draw(axes[0, 1], g, "yield_baseline_pred", YIELD_BINS, "Baseline yield (normal climate)", "t/ha")
    draw(axes[1, 0], g, "impact_total_pct", IMPACT_BINS,
         "Impact vs baseline: MAM 2026 (observed) + OND 2026 (forecast)", "% change")
    draw(axes[1, 1], g, "impact_ond_only_pct", IMPACT_BINS,
         "Impact vs baseline: OND 2026 forecast only (El Nino season)", "% change")
    fig.suptitle("Kenya: ML-based maize yield, 2026", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    print(f"Saved map figure to {out_png}")


# ------------------------------ main ------------------------------
def main():
    train_all = pd.read_csv(TRAIN_CSV)
    train = train_all.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"Training rows: {len(train)}\n")
    quick_cv(train)

    gdf = load_shapefile(SHAPEFILE_PATH)
    print("Processing SEAS5 forecast, SEAS5 hindcast climate mean and ERA5-Land MAM 2026...")
    ond_fc = county_values(seasonal_from_rates(xr.open_dataset(FORECAST_NC)), gdf, ["t2m_c", "tp_mm"])
    ond_hc = county_values(seasonal_from_rates(xr.open_dataset(HINDCAST_NC)), gdf, ["t2m_c", "tp_mm"])
    mam = county_values(mam_seasonal(xr.open_dataset(MAM_2026_NC)), gdf, ["t2m_c_mam", "tp_mm_mam"])

    full, ond_only, normal = build_scenarios(train_all, ond_fc, ond_hc, mam)
    res = predict(train, full, ond_only, normal)
    res.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved predictions for {len(res)} counties to {OUTPUT_CSV}")
    print(res.round(2).to_string(index=False))
    print()
    make_maps(gdf, res, OUTPUT_PNG)


if __name__ == "__main__":
    main()