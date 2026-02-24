"""Assignment 4: Scipy-focused analysis on NBA player season data.

This script reads the provided CSV, filters NBA regular season data,
finds the player with the most regular seasons, computes 3-point
accuracy per season, fits a linear regression to the accuracy, integrates
the fit to get an average accuracy across seasons, interpolates missing
seasons, computes statistical summaries for FGM and FGA using SciPy, and
runs paired and independent t-tests.

Only SciPy, pandas and numpy are used.
"""

import sys
from typing import List

import numpy as np
import pandas as pd
from scipy import integrate, interpolate, stats


def load_and_filter(csv_path: str) -> pd.DataFrame:
    """Load CSV and filter to NBA Regular Season rows.

    Returns a cleaned DataFrame with numeric columns converted.
    """
    df = pd.read_csv(csv_path)
    # Filter to NBA and Regular_Season stage
    df = df[(df['League'] == 'NBA') & (df['Stage'] == 'Regular_Season')].copy()

    # Convert season start year to integer (e.g., '1999 - 2000' -> 1999)
    df['SeasonStart'] = df['Season'].str.split(' - ').str[0].astype(int)

    # Ensure numeric columns are coerced
    numeric_cols = ['GP', 'MIN', 'FGM', 'FGA', '3PM', '3PA', 'FTM', 'FTA']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    return df


def player_with_most_seasons(df: pd.DataFrame) -> str:
    """Return the player name who played the most (distinct) regular seasons."""
    counts = df.groupby('Player')['SeasonStart'].nunique()
    top_player = counts.idxmax()
    return top_player


def compute_3pt_accuracy_for_player(df: pd.DataFrame, player: str) -> pd.DataFrame:
    """Return a DataFrame with SeasonStart, 3PM, 3PA, and 3P accuracy for the player."""
    p = df[df['Player'] == player].copy()
    p = p.sort_values('SeasonStart')
    # avoid division by zero
    p['3P_Accuracy'] = p.apply(lambda r: (r['3PM'] / r['3PA']) if (r['3PA'] and not pd.isna(r['3PA']) and r['3PA'] > 0) else np.nan, axis=1)
    return p[['SeasonStart', '3PM', '3PA', '3P_Accuracy']]


def fit_linear_regression(x: np.ndarray, y: np.ndarray):
    """Perform linear regression using SciPy and return slope, intercept, rvalue, pvalue, stderr."""
    res = stats.linregress(x, y)
    return res


def integrate_fit_average(slope: float, intercept: float, x_min: float, x_max: float) -> float:
    """Integrate the linear fit (slope*x + intercept) over [x_min, x_max] and return the average value."""
    f = lambda x: slope * x + intercept
    integral, _ = integrate.quad(f, x_min, x_max)
    average = integral / (x_max - x_min)
    return average


def interpolate_missing_seasons(season_years: List[int], values: List[float], missing_years: List[int]) -> dict:
    """Interpolate values for missing_years given season_years and corresponding values.

    Returns dict mapping missing_year -> interpolated_value.
    """
    # Build interpolator; drop NaN pairs
    arr = np.array(values)
    mask = ~np.isnan(arr)
    interp_fun = interpolate.interp1d(np.array(season_years)[mask], arr[mask], kind='linear', fill_value='extrapolate')
    estimates = {y: float(interp_fun(y)) for y in missing_years}
    return estimates


def compute_stats_scipy(series: pd.Series) -> dict:
    """Compute mean, variance, skew, kurtosis using SciPy's describe/skew/kurtosis."""
    clean = series.dropna().values
    if clean.size == 0:
        return {'mean': np.nan, 'variance': np.nan, 'skew': np.nan, 'kurtosis': np.nan}
    desc = stats.describe(clean)
    # stats.describe returns variance (population) by default
    return {
        'mean': float(desc.mean),
        'variance': float(desc.variance),
        'skew': float(stats.skew(clean, bias=False)),
        'kurtosis': float(stats.kurtosis(clean, fisher=True, bias=False)),
    }


def run_tests_and_print(df: pd.DataFrame):
    """Run all steps and print human-readable outputs."""
    # 1. Identify player with most regular seasons
    top_player = player_with_most_seasons(df)
    print(f"Player with most regular seasons: {top_player}")

    # 2. Compute player's 3P accuracy by season
    p_df = compute_3pt_accuracy_for_player(df, top_player)
    print('\nSeasons and 3P accuracy for', top_player)
    print(p_df.to_string(index=False))

    # Prepare x (season years) and y (accuracy) for regression; drop NaNs
    reg_df = p_df.dropna(subset=['3P_Accuracy'])
    x = reg_df['SeasonStart'].values.astype(float)
    y = reg_df['3P_Accuracy'].values.astype(float)

    if x.size < 2:
        print('\nNot enough 3P accuracy data for regression.')
    else:
        # 3. Linear regression
        res = fit_linear_regression(x, y)
        print('\nLinear regression of 3P accuracy vs season start year:')
        print(f"slope={res.slope:.6f}, intercept={res.intercept:.6f}, r={res.rvalue:.4f}, p={res.pvalue:.4g}")

        # 4. Integrate fit line to compute average accuracy across played seasons
        x_min = x.min()
        x_max = x.max()
        avg_accuracy_from_fit = integrate_fit_average(res.slope, res.intercept, x_min, x_max)
        print(f"\nAverage 3P accuracy (integrated fit) between {int(x_min)} and {int(x_max)}: {avg_accuracy_from_fit:.4f}")

        # Compare to actual average number of 3-pointers made per season
        avg_3pm_actual = p_df['3PM'].dropna().mean()
        print(f"Actual average 3-pointers made per season (mean 3PM): {avg_3pm_actual:.3f}")
        print('\nNote: integrated accuracy is a fraction (accuracy), while avg 3PM is count per season.')

        # 5. Interpolate missing seasons: specifically 2002-2003 and 2015-2016
        missing_years = [2002, 2015]  # SeasonStart values for those seasons
        # Interpolate both 3PM and 3PA separately
        interp_3pm = interpolate_missing_seasons(p_df['SeasonStart'].tolist(), p_df['3PM'].tolist(), missing_years)
        interp_3pa = interpolate_missing_seasons(p_df['SeasonStart'].tolist(), p_df['3PA'].tolist(), missing_years)
        print('\nInterpolated estimates for missing seasons (SeasonStart -> estimated values):')
        for y_ in missing_years:
            est_3pm = interp_3pm.get(y_, np.nan)
            est_3pa = interp_3pa.get(y_, np.nan)
            est_acc = (est_3pm / est_3pa) if (est_3pa and not np.isnan(est_3pa) and est_3pa > 0) else np.nan
            print(f"{y_}: 3PM≈{est_3pm:.3f}, 3PA≈{est_3pa:.3f}, implied 3P_acc≈{est_acc:.4f}")

    # 6. Statistics for FGM and FGA columns (using only SciPy functions)
    print('\nStatistics for FGM (Field Goals Made):')
    fgm_stats = compute_stats_scipy(df['FGM'])
    for k, v in fgm_stats.items():
        print(f"{k}: {v}")

    print('\nStatistics for FGA (Field Goals Attempted):')
    fga_stats = compute_stats_scipy(df['FGA'])
    for k, v in fga_stats.items():
        print(f"{k}: {v}")

    print('\nComparison: FGA variance vs FGM variance shows how attempts vary relative to makes.')

    # 7. Paired (relational) t-test between FGM and FGA (player-season aligned rows)
    # We align by rows; use dropna on pairs
    paired = df[['FGM', 'FGA']].dropna()
    if paired.shape[0] == 0:
        print('\nNo paired data available for t-tests.')
        return
    t_rel_res = stats.ttest_rel(paired['FGM'], paired['FGA'])
    print('\nPaired t-test (FGM vs FGA):')
    print(f"statistic={t_rel_res.statistic:.6f}, pvalue={t_rel_res.pvalue:.6g}")

    # 8. Independent t-tests on FGM and FGA individually against their own distributions
    # For a "regular t-test on the FGM and FGA columns individually" we can test
    # whether each column's mean differs significantly from zero (one-sample t-test).
    t_fgm_one = stats.ttest_1samp(paired['FGM'], 0.0)
    t_fga_one = stats.ttest_1samp(paired['FGA'], 0.0)
    print('\nOne-sample t-test for FGM (mean vs 0):')
    print(f"statistic={t_fgm_one.statistic:.6f}, pvalue={t_fgm_one.pvalue:.6g}")
    print('\nOne-sample t-test for FGA (mean vs 0):')
    print(f"statistic={t_fga_one.statistic:.6f}, pvalue={t_fga_one.pvalue:.6g}")

    print("\nInterpretation: The paired t-test compares FGM and FGA per season directly (dependent samples),\nwhile the one-sample t-tests evaluate each column against 0 independently.\nThe paired test's p-value indicates whether makes differ from attempts on average when paired by season.")
def main():
    csv_path = 'players_stats_by_season_full_details.csv'
    try:
        df = load_and_filter(csv_path)
    except FileNotFoundError:
        print(f"CSV file not found at {csv_path}. Please run from the folder containing the CSV.")
        sys.exit(1)

    run_tests_and_print(df)


if __name__ == '__main__':
    main()