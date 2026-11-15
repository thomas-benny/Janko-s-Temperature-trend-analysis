
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
import pymannkendall as mk

from scipy.stats import pearsonr, spearmanr


# =========================================================
# 1. CONFIGURATION
# =========================================================

DATA_DIR = Path(__file__).parent

OUTPUT_DIR = DATA_DIR / "output"
PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

TEMPERATURE = "T2MMEAN"

BASELINE_START = "2005-01-01"
BASELINE_END = "2020-12-31"

plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["axes.grid"] = True


# =========================================================
# 2. READ GIOVANNI CSV FILE
# =========================================================

def read_giovanni_csv(file_path):

    # Find the actual CSV header after Giovanni metadata
    with open(file_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    header_row = None

    for i, line in enumerate(lines):

        if line.strip().lower().startswith("time,"):
            header_row = i
            break

    if header_row is None:
        raise ValueError(
            f"CSV header not found: {file_path.name}"
        )

    df = pd.read_csv(
        file_path,
        skiprows=header_row
    )

    df.columns = df.columns.str.strip()

    if len(df.columns) != 2:
        raise ValueError(
            f"Expected 2 columns in {file_path.name}, "
            f"found {len(df.columns)}"
        )

    date_col = df.columns[0]
    value_col = df.columns[1]

    # Extract variable name from Giovanni column
    # Example: M2IMNXASM_5_12_4_QV2M -> QV2M

    variable = value_col.rsplit("_", 1)[-1]

    df = df.rename(columns={
        date_col: "Date",
        value_col: variable
    })

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df[variable] = pd.to_numeric(
        df[variable],
        errors="coerce"
    )

    df = df.dropna(subset=["Date"])

    # Convert dates to monthly timestamps
    df["Date"] = (
        df["Date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    # Remove Giovanni fill values
    df.loc[
        df[variable].abs() > 1e14,
        variable
    ] = np.nan

    if df["Date"].duplicated().any():
        raise ValueError(
            f"Duplicate dates in {file_path.name}"
        )

    return df, variable


# =========================================================
# 3. IMPORT AND JOIN ALL SEVEN DATASETS
# =========================================================

def import_data():

    datasets = {}

    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in the data folder."
        )

    for file_path in csv_files:

        df, variable = read_giovanni_csv(file_path)

        if variable in datasets:
            raise ValueError(
                f"Duplicate variable found: {variable}"
            )

        datasets[variable] = df

        print(
            f"Imported: {variable} "
            f"from {file_path.name}"
        )

    if TEMPERATURE not in datasets:
        raise ValueError(
            f"Temperature variable {TEMPERATURE} not found."
        )

    # Join using Date
    combined = None

    for df in datasets.values():

        if combined is None:
            combined = df.copy()

        else:
            combined = pd.merge(
                combined,
                df,
                on="Date",
                how="outer"
            )

    combined = (
        combined
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Convert Kelvin to Celsius
    combined["T2MMEAN_C"] = (
        combined[TEMPERATURE] - 273.15
    )

    # Calculate wind speed from U and V components
    if "U10M" in combined.columns and "V10M" in combined.columns:

        combined["WIND_SPEED"] = np.sqrt(
            combined["U10M"] ** 2 +
            combined["V10M"] ** 2
        )

    return combined


# =========================================================
# 4. DESCRIPTIVE STATISTICS
# =========================================================

def descriptive_statistics(df):

    numeric = df.select_dtypes(
        include="number"
    )

    result = numeric.describe().T

    result["median"] = numeric.median()
    result["variance"] = numeric.var()
    result["skewness"] = numeric.skew()
    result["missing"] = numeric.isna().sum()

    return result


# =========================================================
# 5. MONTHLY TEMPERATURE TREND
# =========================================================

def monthly_temperature_trend(df):

    data = df[
        ["Date", "T2MMEAN_C"]
    ].dropna().copy()

    if len(data) < 24:
        raise ValueError(
            "Insufficient monthly temperature observations."
        )

    # Convert dates into years
    data["Year_number"] = (
        data["Date"] - data["Date"].min()
    ).dt.total_seconds() / (365.2425 * 86400)

    # Calendar-month effects to account for seasonality
    month_dummies = pd.get_dummies(
        data["Date"].dt.month,
        prefix="month",
        drop_first=True,
        dtype=float
    )

    X = pd.concat(
        [
            data[["Year_number"]].reset_index(drop=True),
            month_dummies.reset_index(drop=True)
        ],
        axis=1
    )

    X = sm.add_constant(X)

    y = data["T2MMEAN_C"].reset_index(drop=True)

    # HAC robust standard errors
    model = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 12}
    )

    slope = model.params["Year_number"]
    p_value = model.pvalues["Year_number"]

    confidence_interval = model.conf_int().loc[
        "Year_number"
    ]

    results = pd.Series({

        "Trend_C_per_year": slope,

        "Trend_C_per_decade": slope * 10,

        "P_value_HAC": p_value,

        "CI_lower": confidence_interval.iloc[0],

        "CI_upper": confidence_interval.iloc[1],

        "R_squared": model.rsquared,

        "Observations": len(data)

    })

    return results


# =========================================================
# 6. ANNUAL TEMPERATURE TREND
# =========================================================

def annual_temperature_analysis(df):

    annual = (
        df.set_index("Date")["T2MMEAN_C"]
        .resample("YS")
        .agg(["mean", "count"])
    )

    # Use only complete years
    annual = annual[
        annual["count"] == 12
    ]["mean"]

    if len(annual) < 8:
        raise ValueError(
            "Not enough complete years for annual trend analysis."
        )

    years = annual.index.year.to_numpy()
    values = annual.values

    # Linear regression
    slope, intercept = np.polyfit(
        years,
        values,
        1
    )

    # Mann-Kendall test
    mk_result = mk.original_test(values)

    # Sen's slope
    sen_result = mk.sens_slope(values)

    results = pd.Series({

        "Linear_trend_C_per_year": slope,

        "Linear_trend_C_per_decade": slope * 10,

        "Mann_Kendall_p": mk_result.p,

        "Mann_Kendall_tau": mk_result.Tau,

        "Mann_Kendall_trend": mk_result.trend,

        "Sen_slope_C_per_year": sen_result.slope

    })

    annual_df = annual.rename(
        "Temperature_C"
    ).to_frame()

    annual_df.index.name = "Year"

    return annual_df, results


# =========================================================
# 7. SEASONAL MANN-KENDALL TEST
# =========================================================

def seasonal_mann_kendall(df):

    monthly = (
        df.set_index("Date")["T2MMEAN_C"]
        .asfreq("MS")
    )

    # Seasonal MK requires a complete monthly sequence
    if monthly.isna().any() or len(monthly) < 24:

        print(
            "Seasonal Mann-Kendall skipped: "
            "missing monthly observations."
        )

        return None

    result = mk.seasonal_test(
        monthly.values,
        period=12
    )

    return pd.Series({

        "Trend": result.trend,

        "P_value": result.p,

        "Kendall_tau": result.Tau,

        "Sen_slope": result.slope

    })


# =========================================================
# 8. TEMPERATURE ANOMALIES
# =========================================================

def calculate_anomalies(df):

    result = df[
        ["Date", "T2MMEAN_C"]
    ].copy()

    baseline = df[
        (df["Date"] >= BASELINE_START) &
        (df["Date"] <= BASELINE_END)
    ]

    # Average temperature for each calendar month
    climatology = baseline.groupby(
        baseline["Date"].dt.month
    )["T2MMEAN_C"].mean()

    result["Climatology_C"] = (
        result["Date"].dt.month.map(climatology)
    )

    result["Temperature_Anomaly_C"] = (
        result["T2MMEAN_C"] -
        result["Climatology_C"]
    )

    return result


# =========================================================
# 9. CORRELATION BETWEEN TEMPERATURE AND OTHER VARIABLES
# =========================================================

def temperature_correlations(df):

    variables = [
        "QV2M",
        "PRECTOT",
        "SWGDN",
        "PS",
        "WIND_SPEED"
    ]

    data = df.copy()

    # Remove monthly seasonal cycles from all variables
    for variable in variables:

        if variable not in data.columns:
            continue

        climatology = data.groupby(
            data["Date"].dt.month
        )[variable].transform("mean")

        data[variable + "_anomaly"] = (
            data[variable] - climatology
        )

    # Temperature anomaly
    data["Temperature_Anomaly"] = (
        data.groupby(
            data["Date"].dt.month
        )["T2MMEAN_C"]
        .transform(lambda x: x - x.mean())
    )

    results = []

    for variable in variables:

        anomaly_name = variable + "_anomaly"

        if anomaly_name not in data.columns:
            continue

        pair = data[
            ["Temperature_Anomaly", anomaly_name]
        ].dropna()

        if len(pair) < 3:
            continue

        if pair.nunique().min() < 2:
            continue

        pearson_r, pearson_p = pearsonr(
            pair["Temperature_Anomaly"],
            pair[anomaly_name]
        )

        spearman_r, spearman_p = spearmanr(
            pair["Temperature_Anomaly"],
            pair[anomaly_name]
        )

        results.append({

            "Variable": variable,

            "Pearson_r": pearson_r,

            "Pearson_p": pearson_p,

            "Spearman_rho": spearman_r,

            "Spearman_p": spearman_p,

            "Observations": len(pair)

        })

    return pd.DataFrame(results)


# =========================================================
# 10. PLOT MONTHLY TEMPERATURE
# =========================================================

def plot_monthly_temperature(df):

    data = df.dropna(
        subset=["T2MMEAN_C"]
    )

    plt.figure()

    plt.plot(
        data["Date"],
        data["T2MMEAN_C"],
        linewidth=1
    )

    plt.title(
        "Monthly Mean 2m Air Temperature"
    )

    plt.xlabel("Year")
    plt.ylabel("Temperature (°C)")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "temperature_monthly.png",
        dpi=300
    )

    plt.close()


# =========================================================
# 11. PLOT ANNUAL TEMPERATURE
# =========================================================

def plot_annual_temperature(annual):

    years = annual.index.year.to_numpy()
    values = annual["Temperature_C"].values

    slope, intercept = np.polyfit(
        years,
        values,
        1
    )

    plt.figure()

    plt.plot(
        years,
        values,
        marker="o",
        label="Annual mean"
    )

    plt.plot(
        years,
        slope * years + intercept,
        linestyle="--",
        label="Linear trend"
    )

    plt.title(
        "Annual Mean Temperature Trend"
    )

    plt.xlabel("Year")
    plt.ylabel("Temperature (°C)")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "temperature_annual.png",
        dpi=300
    )

    plt.close()


# =========================================================
# 12. PLOT TEMPERATURE ANOMALIES
# =========================================================

def plot_anomalies(anomalies):

    data = anomalies.dropna(
        subset=["Temperature_Anomaly_C"]
    )

    plt.figure()

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.plot(
        data["Date"],
        data["Temperature_Anomaly_C"]
    )

    plt.title(
        "Monthly Temperature Anomalies"
    )

    plt.xlabel("Year")
    plt.ylabel("Temperature Anomaly (°C)")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "temperature_anomalies.png",
        dpi=300
    )

    plt.close()


# =========================================================
# 13. PLOT SEASONAL TEMPERATURE
# =========================================================

def plot_seasonality(df):

    monthly = df.groupby(
        df["Date"].dt.month
    )["T2MMEAN_C"].mean()

    plt.figure()

    plt.plot(
        monthly.index,
        monthly.values,
        marker="o"
    )

    plt.xticks(range(1, 13))

    plt.title(
        "Average Monthly Temperature"
    )

    plt.xlabel("Month")
    plt.ylabel("Temperature (°C)")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "temperature_seasonality.png",
        dpi=300
    )

    plt.close()


# =========================================================
# 14. PLOT TEMPERATURE VS OTHER VARIABLES
# =========================================================

def plot_relationships(df):

    variables = [
        "QV2M",
        "PRECTOT",
        "SWGDN",
        "PS",
        "WIND_SPEED"
    ]

    for variable in variables:

        if variable not in df.columns:
            continue

        pair = df[
            ["T2MMEAN_C", variable]
        ].dropna()

        if len(pair) < 3:
            continue

        plt.figure()

        plt.scatter(
            pair[variable],
            pair["T2MMEAN_C"],
            alpha=0.6
        )

        plt.xlabel(variable)

        plt.ylabel("Temperature (°C)")

        plt.title(
            f"Temperature vs {variable}"
        )

        plt.tight_layout()

        plt.savefig(
            PLOT_DIR / f"temperature_vs_{variable}.png",
            dpi=300
        )

        plt.close()


# =========================================================
# 15. MAIN PROGRAM
# =========================================================

def main():

    print("\n========== IMPORTING DATA ==========")

    df = import_data()

    print("\n========== DATA INFORMATION ==========")

    print(df.info())

    print("\nMissing values:")

    print(df.isna().sum())

    # Save combined data
    df.to_csv(
        OUTPUT_DIR / "combined_data.csv",
        index=False
    )

    # Descriptive statistics
    print("\n========== DESCRIPTIVE STATISTICS ==========")

    stats = descriptive_statistics(df)

    print(stats)

    stats.to_csv(
        OUTPUT_DIR / "statistics.csv"
    )

    # Monthly trend
    print("\n========== MONTHLY TEMPERATURE TREND ==========")

    monthly_trend = monthly_temperature_trend(df)

    print(monthly_trend)

    monthly_trend.to_csv(
        OUTPUT_DIR / "monthly_temperature_trend.csv"
    )

    # Annual trend
    print("\n========== ANNUAL TEMPERATURE TREND ==========")

    annual, annual_results = annual_temperature_analysis(df)

    print(annual_results)

    annual.to_csv(
        OUTPUT_DIR / "annual_temperature.csv"
    )

    annual_results.to_csv(
        OUTPUT_DIR / "annual_trend_statistics.csv"
    )

    # Seasonal Mann-Kendall
    print("\n========== SEASONAL MANN-KENDALL ==========")

    seasonal = seasonal_mann_kendall(df)

    if seasonal is not None:

        print(seasonal)

        seasonal.to_csv(
            OUTPUT_DIR / "seasonal_mann_kendall.csv"
        )

    # Temperature anomalies
    print("\n========== TEMPERATURE ANOMALIES ==========")

    anomalies = calculate_anomalies(df)

    anomalies.to_csv(
        OUTPUT_DIR / "temperature_anomalies.csv",
        index=False
    )

    # Correlations
    print("\n========== TEMPERATURE CORRELATIONS ==========")

    correlations = temperature_correlations(df)

    print(correlations)

    correlations.to_csv(
        OUTPUT_DIR / "temperature_correlations.csv",
        index=False
    )

    # Generate plots
    print("\n========== GENERATING PLOTS ==========")

    plot_monthly_temperature(df)

    plot_annual_temperature(annual)

    plot_anomalies(anomalies)

    plot_seasonality(df)

    plot_relationships(df)

    print("\n========== ANALYSIS COMPLETED ==========")

    print("Output folder:", OUTPUT_DIR)

    print("Plot folder:", PLOT_DIR)


# =========================================================
# 16. EXECUTE PROGRAM
# =========================================================

if __name__ == "__main__":

    main()
