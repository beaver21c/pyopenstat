"""Synthetic regional policy panel for difference-in-differences and event studies (seeded, synthetic only).

Panel_DiD_Policy.csv — 150 regions x 2015-2024 (balanced), regions nested in 15 provinces
    region      region id 1..150 (panel variable)
    province    province id 1..15 (10 regions each; level of the single-timing policy)
    year        2015 ... 2024
    x1          time-varying covariate (log local income), exogenous
    x2          time-invariant covariate (baseline elderly share, standardized); earlier adoption when x2 is high
                in the staggered design, and the untreated outcome trend depends on x2 (conditional parallel trends)
    treat1      single-timing policy: provinces 1-6 (60 regions) treated from 2020, the rest never treated
    y1          outcome of the single-timing design:
                y1 = a_i + g_t + 0.5 x1 + (1.5 + 0.3 (year - 2020)) treat1 + e
    cohort2     first treatment year of the staggered design (2018, 2020, 2022; 0 = never treated)
    treat2      staggered adoption, switches on at cohort2 and stays on
    y2          outcome of the staggered design with cohort-specific and dynamic effects:
                y2 = a_i + g_t + 0.4 x1 + 0.12 x2 (year - 2015)
                     + treat2 (b_g + 0.25 (year - cohort2) + 0.3 x2),  b_2018 = 1.0, b_2020 = 2.0, b_2022 = 3.0

Run: python sample_data/make_panel_did_sample_data.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def make(n: int = 150, years=range(2015, 2025), seed: int = 20261108) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    years = np.array(list(years))
    T = len(years)
    province = np.repeat(np.arange(1, 16), n // 15)
    a_i = rng.normal(0, 1.0, n)
    g_t = np.cumsum(rng.normal(0.15, 0.10, T))
    x2 = rng.normal(0, 1, n)
    inc0 = 3.0 + 0.3 * a_i + rng.normal(0, 0.3, n)
    # staggered adoption: propensity to adopt early rises with x2
    score = 0.9 * x2 + rng.logistic(0, 1, n)
    cut = np.quantile(score, [1 - 0.20, 1 - 0.43, 1 - 0.66])        # 20% 2018, 23% 2020, 23% 2022, 34% never
    cohort2 = np.where(score >= cut[0], 2018, np.where(score >= cut[1], 2020, np.where(score >= cut[2], 2022, 0)))
    base = {2018: 1.0, 2020: 2.0, 2022: 3.0}
    rows = []
    for i in range(n):
        for k, yr in enumerate(years):
            x1 = inc0[i] + 0.05 * k + rng.normal(0, 0.15)
            t1 = int(province[i] <= 6 and yr >= 2020)
            y1 = a_i[i] + g_t[k] + 0.5 * x1 + (1.5 + 0.3 * (yr - 2020)) * t1 + rng.normal(0, 0.6)
            c = int(cohort2[i])
            t2 = int(c > 0 and yr >= c)
            eff = (base[c] + 0.25 * (yr - c) + 0.3 * x2[i]) * t2 if c else 0.0
            y2 = a_i[i] + g_t[k] + 0.4 * x1 + 0.12 * x2[i] * (yr - 2015) + eff + rng.normal(0, 0.6)
            rows.append({"region": i + 1, "province": int(province[i]), "year": int(yr),
                         "x1": round(x1, 4), "x2": round(float(x2[i]), 4),
                         "treat1": t1, "y1": round(y1, 4), "cohort2": c, "treat2": t2, "y2": round(y2, 4)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = make()
    out = os.path.join(HERE, "Panel_DiD_Policy.csv")
    df.to_csv(out, index=False, encoding="utf-8")
    print(out, df.shape)
    print(df.groupby("cohort2")["region"].nunique())
