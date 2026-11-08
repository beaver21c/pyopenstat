"""Synthetic sample data for the Hausman–Taylor / Amemiya–MaCurdy estimators (Stata xthtaylor).

All values are simulated (seed fixed); no real survey or administrative records are used.

Panel_HT_Wage.csv  balanced panel of 300 workers x 7 years (2015-2021), log wage with an unobserved individual
                   effect u_i that is correlated with schooling (educ), union membership and experience.

Columns (ASCII names so that the same file runs in Stata):
    id, year          panel and time identifiers
    lwage             log hourly wage
    wks               weeks worked (time-varying, exogenous; its panel mean is correlated with educ)
    south             lives in the south (time-varying, exogenous; some workers move)
    occ               blue-collar occupation (time-varying, exogenous)
    exper             years of experience (time-varying, correlated with u_i)
    union             wage set by a union contract (time-varying, correlated with u_i)
    female            woman (time-invariant, exogenous)
    rural             grew up in a rural area (time-invariant, exogenous)
    educ              years of schooling (time-invariant, correlated with u_i)

Run:  python sample_data/make_panel_ht_sample_data.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def make_ht_wage(seed: int = 20261108, n: int = 300, T: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(1, n + 1), T)
    years = np.tile(np.arange(2015, 2015 + T), n)
    tt = years - 2015
    u = rng.normal(0, 0.35, n)                          # individual effect
    w = rng.normal(0, 1, n)                             # work-attachment propensity (exogenous)
    female = (rng.uniform(size=n) < 0.45).astype(int)
    rural = (rng.uniform(size=n) < 0.30).astype(int)
    s_exog = 1.2 * w - 0.8 * rural + rng.normal(0, 1.4, n)      # schooling part unrelated to u_i
    educ = np.clip(np.round(12.5 + 1.65 * u / 0.35 + s_exog), 8, 20)
    exper0 = np.clip(np.round(rng.uniform(1, 25, n) + 2.0 * u / 0.35), 0, 35)
    south0 = (rng.uniform(size=n) < 0.35).astype(int)
    move = rng.uniform(size=n) < 0.12
    move_at = rng.integers(1, T, n)

    U, W, F, R, E, S = u[ids - 1], w[ids - 1], female[ids - 1], rural[ids - 1], educ[ids - 1], s_exog[ids - 1]
    exper = exper0[ids - 1] + tt
    south = np.where(move[ids - 1] & (tt >= move_at[ids - 1]), 1 - south0[ids - 1], south0[ids - 1])
    wks = np.clip(np.round(45.0 + 2.2 * W + rng.normal(0, 3.0, n * T)), 20, 52)
    occ = (rng.uniform(size=n * T) < 1 / (1 + np.exp(-(-0.2 - 0.45 * S + 0.3 * R)))).astype(int)
    union = (rng.uniform(size=n * T) < 1 / (1 + np.exp(-(-0.6 + 1.8 * U + 0.2 * occ)))).astype(int)
    e = rng.normal(0, 0.15, n * T)
    lwage = (4.6 + 0.035 * exper - 0.0004 * exper ** 2 + 0.006 * wks - 0.05 * south - 0.04 * occ + 0.07 * union
             + 0.085 * E - 0.22 * F - 0.06 * R + U + e)
    return pd.DataFrame({
        "id": ids, "year": years, "lwage": np.round(lwage, 6), "wks": wks.astype(int), "south": south.astype(int),
        "occ": occ, "exper": exper.astype(int), "union": union, "female": F, "rural": R, "educ": E.astype(int),
    })


def main() -> None:
    out = os.path.join(HERE, "Panel_HT_Wage.csv")
    make_ht_wage().to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
