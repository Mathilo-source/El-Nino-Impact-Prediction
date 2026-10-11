"""
Step 6-7 (OND + MAM temperature and rainfall features):
Train a model that predicts county-level annual maize yield from OND and MAM
climate features, then evaluate it two ways:
  (a) Leave-one-year-out cross-validation across all years (general performance)
  (b) A specific holdout on 2015 & 2016 -- our Super El Nino case -- trained
      on every other year, to see whether the model actually captures the
      El Nino signal, not just an average relationship.

Input:
    - kenya_maize_climate_modeling_dataset_ond_mam.csv   (output of mam_county_features script)

Output (printed, not saved):
    - Leave-one-year-out performance (R2, MAE, RMSE) per year
    - 2015/2016 holdout performance specifically
    - Feature importances from the final model (trained on all data)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ----------------------- CONFIG -----------------------
# Merged file from step4b_mam_features_and_merge.py (OND + MAM + yield)
DATA_PATH = "kenya_maize_climate_modeling_dataset_ond_mam.csv"
TARGET = "yield_mt_ha"

# Anomaly features are the core signal -- how unusual was this county's
# season compared to its own normal. Raw values included too, since
# baseline climate (e.g. a naturally drier county) also affects yield.
# Yield is an ANNUAL figure (long rains + short rains), so the model gets both
# seasons: OND (Oct-Dec, the El Nino season) and MAM (Mar-May, the long rains).
# Soil moisture and evaporation are NOT used because the 2026 forecast cannot
# supply comparable values. Each season contributes raw values and anomalies.
FEATURES = [
    # OND (short rains)
    "t2m_c", "tp_mm", "t2m_c_anom", "tp_mm_anom",
    # MAM (long rains)
    "t2m_c_mam", "tp_mm_mam", "t2m_c_mam_anom", "tp_mm_mam_anom",
]
# --------------------------------------------------------


def load_data(path):
    df = pd.read_csv(path)
    before = len(df)
    df = df.dropna(subset=[TARGET])
    print(f"Loaded {before} rows, dropped {before - len(df)} rows with missing yield "
          f"(expected: 47, from the 2019 gap). {len(df)} rows remain.\n")
    return df


def leave_one_year_out_eval(df):
    print("=" * 60)
    print("LEAVE-ONE-YEAR-OUT CROSS-VALIDATION")
    print("=" * 60)
    years = sorted(df["year"].unique())
    results = []

    for test_year in years:
        train = df[df["year"] != test_year]
        test = df[df["year"] == test_year]

        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(train[FEATURES], train[TARGET])
        preds = model.predict(test[FEATURES])

        r2 = r2_score(test[TARGET], preds)
        mae = mean_absolute_error(test[TARGET], preds)
        rmse = np.sqrt(mean_squared_error(test[TARGET], preds))

        results.append({"year": test_year, "r2": r2, "mae": mae, "rmse": rmse, "n": len(test)})
        print(f"  Year {test_year}: R2={r2:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}  (n={len(test)})")

    results_df = pd.DataFrame(results)
    print(f"\n  Average across years: R2={results_df['r2'].mean():.3f}  "
          f"MAE={results_df['mae'].mean():.3f}  RMSE={results_df['rmse'].mean():.3f}")
    print()
    return results_df


def el_nino_holdout_eval(df):
    print("=" * 60)
    print("SUPER EL NINO HOLDOUT: train on all years EXCEPT 2015 & 2016,")
    print("test specifically on 2015 & 2016")
    print("=" * 60)

    el_nino_years = [2015, 2016]
    train = df[~df["year"].isin(el_nino_years)]
    test = df[df["year"].isin(el_nino_years)]

    if len(test) == 0:
        print("  No 2015/2016 rows found in this dataset -- check your data.")
        return None

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(train[FEATURES], train[TARGET])
    preds = model.predict(test[FEATURES])

    r2 = r2_score(test[TARGET], preds)
    mae = mean_absolute_error(test[TARGET], preds)
    rmse = np.sqrt(mean_squared_error(test[TARGET], preds))

    print(f"  Trained on {len(train)} rows, tested on {len(test)} rows "
          f"(2015+2016 combined)")
    print(f"  R2={r2:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}\n")

    # Show per-county actual vs predicted for these two years, so you can
    # see WHERE the model does well or badly, not just the average error.
    comparison = test[["county", "year", TARGET]].copy()
    comparison["predicted"] = preds
    comparison["error"] = comparison["predicted"] - comparison[TARGET]
    print(comparison.sort_values("year").to_string(index=False))
    print()

    return comparison


def train_final_model_and_show_importance(df):
    print("=" * 60)
    print("FINAL MODEL: trained on ALL available data")
    print("(OND + MAM temperature and rainfall)")
    print("=" * 60)

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(df[FEATURES], df[TARGET])

    importances = pd.Series(model.feature_importances_, index=FEATURES)
    importances = importances.sort_values(ascending=False)

    print("\nFeature importances (which climate variables matter most):")
    for feat, imp in importances.items():
        print(f"  {feat:15s} {imp:.3f}")
    print()

    return model


def main():
    df = load_data(DATA_PATH)

    leave_one_year_out_eval(df)
    el_nino_holdout_eval(df)
    final_model = train_final_model_and_show_importance(df)

    print("Done. `final_model` is trained and ready")


if __name__ == "__main__":
    main()