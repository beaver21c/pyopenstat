"""Synthetic sample data for panel unit-root and panel cointegration tests (Stata xtunitroot / xtcointtest).

All values are simulated (seed fixed); no real survey or administrative records are used.

Panel_UnitRoot_Regions.csv   strongly balanced panel, 25 regions x 38 years (1987-2024), ASCII variable names

    id, region   panel identifier (1..25) and region code (R01..R25)
    year         time variable
    y_stat       stationary AR(1) around region-specific means (rho_i 0.45-0.75, AR(2) part in some regions)
    y_ur         unit-root series: random walk with small region-specific drift and AR(1) differences
    y_trend      trend-stationary: region intercept + region trend + AR(1) deviation
    x1, x2       independent I(1) regressors (random walks with drift)
    y_coint      cointegrated with x1, x2: a_i + b1_i*x1 + b2_i*x2 + stationary AR(1) error
    x3           I(1) regressor
    y_nocoint    I(1) series not cointegrated with x3 (independent random walk; own random stream)
    share        bounded stationary share (0-1) with a common time shock (cross-sectional dependence)

Unbalanced copy used in tests and do-files (rule, no internal gaps): drop the first (id % 4) * 2 years of every
region whose id is divisible by 3 (ids 3, 6, ..., 24 lose 0-6 initial years).

Run:  python sample_data/make_panel_unitroot_sample_data.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _ar1(rng: np.random.Generator, rho: float, sd: float, n: int, burn: int = 60) -> np.ndarray:
    e = rng.normal(0, sd, n + burn)
    u = np.zeros(n + burn)
    for t in range(1, n + burn):
        u[t] = rho * u[t - 1] + e[t]
    return u[burn:]


def make_panel(seed: int = 20261108, n_regions: int = 25, first_year: int = 1987, n_years: int = 38) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rng_nc = np.random.default_rng(seed + 1013)      # separate stream for the non-cointegrated pair
    years = np.arange(first_year, first_year + n_years)
    T = n_years
    tt = np.arange(1, T + 1)
    common = _ar1(rng, 0.5, 0.04, T)                # common shock for `share`
    rows = []
    for i in range(1, n_regions + 1):
        mu = rng.normal(10, 2)
        rho = rng.uniform(0.45, 0.75)
        if i % 4 == 0:                               # AR(2) stationary part in some regions
            e = rng.normal(0, 1, T + 60)
            u = np.zeros(T + 60)
            for t in range(2, T + 60):
                u[t] = rho * u[t - 1] + 0.15 * u[t - 2] + e[t]
            u = u[60:]
        else:
            u = _ar1(rng, rho, 1.0, T)
        y_stat = mu + u
        # unit root with AR(1) differences
        dy = rng.uniform(0.0, 0.05) + _ar1(rng, 0.3, 0.8, T)
        y_ur = rng.normal(50, 5) + np.cumsum(dy)
        # trend stationary
        y_trend = rng.normal(20, 3) + rng.uniform(0.2, 0.6) * tt + _ar1(rng, rng.uniform(0.3, 0.6), 1.2, T)
        # I(1) regressors
        x1 = rng.normal(5, 1) + np.cumsum(rng.uniform(0.02, 0.10) + rng.normal(0, 0.5, T))
        x2 = rng.normal(3, 1) + np.cumsum(rng.uniform(-0.05, 0.05) + rng.normal(0, 0.4, T))
        b1, b2 = 1.0 + rng.normal(0, 0.15), 0.6 + rng.normal(0, 0.10)
        y_coint = rng.normal(2, 1) + b1 * x1 + b2 * x2 + _ar1(rng, 0.5, 0.6, T)
        x3 = rng_nc.normal(8, 1) + np.cumsum(0.05 + rng_nc.normal(0, 0.5, T))
        y_nocoint = rng_nc.normal(12, 1) + np.cumsum(0.03 + rng_nc.normal(0, 0.6, T))
        share = 1 / (1 + np.exp(-(rng.normal(-0.5, 0.4) + _ar1(rng, 0.6, 0.25, T) + 3 * common)))
        for t in range(T):
            rows.append({
                "id": i, "region": f"R{i:02d}", "year": int(years[t]),
                "y_stat": round(float(y_stat[t]), 5), "y_ur": round(float(y_ur[t]), 5),
                "y_trend": round(float(y_trend[t]), 5), "x1": round(float(x1[t]), 5), "x2": round(float(x2[t]), 5),
                "y_coint": round(float(y_coint[t]), 5), "x3": round(float(x3[t]), 5),
                "y_nocoint": round(float(y_nocoint[t]), 5), "share": round(float(share[t]), 5),
            })
    return pd.DataFrame(rows)


def unbalanced(df: pd.DataFrame) -> pd.DataFrame:
    """Rule used by the tests: ids divisible by 3 lose their first (id % 4) * 2 years (no internal gaps)."""
    first = df.groupby("id")["year"].transform("min")
    drop = (df["id"] % 3 == 0) & (df["year"] < first + (df["id"] % 4) * 2)
    return df.loc[~drop].reset_index(drop=True)


def main() -> None:
    make_panel().to_csv(os.path.join(HERE, "Panel_UnitRoot_Regions.csv"), index=False)
    print("written: Panel_UnitRoot_Regions.csv")


if __name__ == "__main__":
    main()
