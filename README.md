# assignment4
# NBA Statistics Analysis with SciPy

## Project Purpose

This project analyzes NBA player statistics using **SciPy** to uncover trends in shooting performance.  
It demonstrates data filtering, regression analysis, interpolation, descriptive statistics, and hypothesis testing.

---

## Features

- Filters dataset to NBA regular season only
- Identifies player with most seasons played
- Calculates three-point accuracy per season
- Performs linear regression and integration
- Interpolates missing seasons (2002–2003 and 2015–2016)
- Computes mean, variance, skew, kurtosis (FGM & FGA)
- Performs paired and independent t-tests

---

## Class Design

### `NBAStatsAnalyzer`

Encapsulates all dataset handling and statistical analysis.

### Attributes

- `file_path` – Path to CSV file  
- `data` – Full dataset  
- `regular_season_data` – Filtered NBA regular season data  
- `player_data` – Data for player with most seasons  

### Methods

- `load_data()` – Loads CSV  
- `filter_regular_season()` – Filters NBA regular season  
- `get_most_seasons_player()` – Finds player with most seasons  
- `calculate_three_point_accuracy()` – Computes 3P accuracy  
- `perform_linear_regression()` – Generates best-fit line  
- `calculate_integrated_average()` – Integrates regression line  
- `interpolate_missing_seasons()` – Estimates missing seasons  
- `compute_descriptive_statistics()` – Mean, variance, skew, kurtosis  
- `perform_t_tests()` – Paired and independent t-tests  

---

## Limitations

- Assumes linear trend for regression
- Interpolation estimates missing values
- Statistical tests assume approximate normality
- Results depend on dataset accuracy
