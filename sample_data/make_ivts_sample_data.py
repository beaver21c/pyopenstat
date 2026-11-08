"""Synthetic sample data for instrumental-variable models and time-series (ARIMA) analysis.

All values are simulated (seed fixed); no real survey or administrative records are used.

IV_Schooling.csv       cross-section, returns to schooling with unobserved ability (ivregress)
IV_Panel_CareStaff.csv balanced panel of 80 regions x 10 years, care staff instrumented by grants (xtivreg)
TS_Monthly_Welfare.csv monthly series 2008-01 .. 2023-12 with trend, seasonality, ARMA errors (arima, dfuller)

Run:  python sample_data/make_ivts_sample_data.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def make_schooling(seed: int = 20261107, n: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    region = rng.integers(1, 31, n)                      # 30 regions (cluster variable)
    reg_eff = rng.normal(0, 0.08, 31)[region]
    ability = rng.normal(0, 1, n)
    meduc = np.clip(np.round(rng.normal(11, 3, n)), 0, 20)
    nearc4 = (rng.uniform(size=n) < 0.55 + 0.1 * (region % 3 == 0)).astype(int)
    black = (rng.uniform(size=n) < 0.22).astype(int)
    south = (region <= 12).astype(int)
    urban = (rng.uniform(size=n) < 0.65).astype(int)
    age = rng.integers(25, 56, n)
    educ = np.clip(np.round(7.0 + 0.30 * meduc + 0.75 * nearc4 + 0.9 * ability - 0.4 * black
                            + 0.3 * urban + rng.normal(0, 1.6, n)), 6, 20)
    gap = rng.integers(0, 7, n) * (rng.uniform(size=n) < 0.45)  # career interruptions
    exper = np.clip(age - educ - 6 - gap, 0, None)
    expersq = exper ** 2 / 100.0
    # heteroskedastic error correlated with ability through the omitted term
    e = rng.normal(0, 0.30 + 0.04 * np.abs(ability), n)
    lwage = (4.9 + 0.075 * educ + 0.035 * exper - 0.050 * expersq - 0.15 * black - 0.10 * south
             + 0.12 * urban + 0.12 * ability + reg_eff + e)
    return pd.DataFrame({
        "id": np.arange(1, n + 1), "region": region, "lwage": np.round(lwage, 5), "educ": educ.astype(int),
        "exper": exper.astype(int), "expersq": np.round(expersq, 4), "age": age, "black": black, "south": south,
        "urban": urban, "nearc4": nearc4, "meduc": meduc.astype(int),
    })


def make_panel(seed: int = 20261108, n: int = 80, T: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(1, n + 1), T)
    years = np.tile(np.arange(2014, 2014 + T), n)
    a_i = rng.normal(0, 0.25, n)[ids - 1]                 # regional effect
    g_i = rng.normal(0, 0.20, n)[ids - 1]
    tt = years - 2014
    lincome = 3.0 + 0.4 * g_i + 0.02 * tt + rng.normal(0, 0.10, n * T)
    aged = 15 + 4 * g_i + 0.4 * tt + rng.normal(0, 1.0, n * T)
    lgrant = 1.0 + 0.3 * a_i + 0.03 * tt + rng.normal(0, 0.20, n * T)
    lrule = 0.5 + 0.02 * aged + rng.normal(0, 0.15, n * T)
    shock = rng.normal(0, 0.12, n * T)
    lstaff = (0.2 + 0.55 * lgrant + 0.40 * lrule + 0.3 * lincome + 0.5 * a_i + 0.6 * shock
              + rng.normal(0, 0.08, n * T))
    lcare = (1.0 + 0.45 * lstaff + 0.25 * lincome + 0.015 * aged + a_i - 0.5 * shock
             + rng.normal(0, 0.06, n * T))
    return pd.DataFrame({
        "region": ids, "year": years, "lcare": np.round(lcare, 5), "lstaff": np.round(lstaff, 5),
        "lincome": np.round(lincome, 5), "aged": np.round(aged, 3), "lgrant": np.round(lgrant, 5),
        "lrule": np.round(lrule, 5),
    })


def make_monthly(seed: int = 20261109) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.period_range("2008-01", "2023-12", freq="M")
    T = len(dates)
    t = np.arange(1, T + 1)
    month = dates.month.to_numpy()
    temp = 12.5 - 11.0 * np.cos(2 * np.pi * (month - 1) / 12) + rng.normal(0, 1.5, T)
    # ARMA(1,1) disturbance
    eps = rng.normal(0, 0.035, T + 50)
    u = np.zeros(T + 50)
    for i in range(1, T + 50):
        u[i] = 0.6 * u[i - 1] + eps[i] + 0.3 * eps[i - 1]
    u = u[50:]
    season = 0.06 * np.cos(2 * np.pi * (month - 1) / 12) + 0.03 * np.sin(4 * np.pi * (month - 1) / 12)
    lvisits = 7.0 + 0.0035 * t + season - 0.004 * temp + u
    # random walk with drift (unit-root example)
    cpi = 100 + np.cumsum(0.18 + rng.normal(0, 0.35, T))
    # unemployment-type series: seasonal ARIMA(1,0,0)(1,0,0)12 around a mean
    e2 = rng.normal(0, 0.25, T + 60)
    z = np.zeros(T + 60)
    for i in range(13, T + 60):
        z[i] = 0.7 * z[i - 1] + 0.5 * z[i - 12] - 0.35 * z[i - 13] + e2[i]
    unemp = 3.6 + z[60:]
    return pd.DataFrame({
        "date": [str(d) for d in dates], "t": t, "year": dates.year.to_numpy(), "month": month,
        "lvisits": np.round(lvisits, 5), "temp": np.round(temp, 2), "cpi": np.round(cpi, 3),
        "unemp": np.round(unemp, 3),
    })


def main() -> None:
    make_schooling().to_csv(os.path.join(HERE, "IV_Schooling.csv"), index=False)
    make_panel().to_csv(os.path.join(HERE, "IV_Panel_CareStaff.csv"), index=False)
    make_monthly().to_csv(os.path.join(HERE, "TS_Monthly_Welfare.csv"), index=False)
    print("written: IV_Schooling.csv, IV_Panel_CareStaff.csv, TS_Monthly_Welfare.csv")


if __name__ == "__main__":
    main()
