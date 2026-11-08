"""Synthetic regional welfare-spending panel for dynamic panel models (seeded, synthetic only).

Dynamic_Panel_Welfare.csv — 200 regions x 2014-2023 (late entry / early exit for some regions, no internal gaps)
    지역코드        region id (R001 ...)
    연도            year
    복지지출        log per-capita welfare spending, AR(1) with region fixed effects:
                    y_it = 0.55 y_i,t-1 + 0.04 고령인구비율 + 0.015 재정자립도 - 0.03 실업률 + eta_i + d_t + e_it
    고령인구비율    elderly population share (%), strictly exogenous
    재정자립도      fiscal self-reliance (%), predetermined (responds to last year's spending shock)
    실업률          unemployment rate (%), endogenous (persistent, correlated with the current shock)
    시도            province code (cluster variable containing regions)

Run: python sample_data/make_panel_sample_data.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def make(n: int = 200, years=range(2014, 2024), seed: int = 20261107) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    years = list(years)
    T = len(years)
    burn = 20
    eta = rng.normal(0, 0.12, n)
    a_old = rng.normal(16, 3, n)
    a_fis = rng.normal(30, 8, n)
    a_une = rng.normal(3.5, 0.8, n)
    d_t = np.cumsum(rng.normal(0.02, 0.01, burn + T))
    y = np.zeros((n, burn + T))
    old = np.zeros_like(y)
    fis = np.zeros_like(y)
    une = np.zeros_like(y)
    eps = rng.normal(0, 0.08, (n, burn + T))
    y_prev = (eta + 0.04 * a_old + 0.015 * a_fis - 0.03 * a_une + 3.0) / (1 - 0.55)
    fis_prev = a_fis.copy()
    une_prev = a_une.copy()
    for t in range(burn + T):
        trend = 0.25 * (t - burn)
        old[:, t] = a_old + trend + rng.normal(0, 0.4, n)
        e_prev = eps[:, t - 1] if t > 0 else np.zeros(n)
        fis[:, t] = 0.6 * fis_prev + 0.4 * a_fis + 12.0 * e_prev + rng.normal(0, 1.0, n)
        une[:, t] = 0.7 * une_prev + 0.3 * a_une + 1.5 * eps[:, t] + rng.normal(0, 0.3, n)
        y[:, t] = (3.0 + 0.55 * y_prev + 0.04 * old[:, t] + 0.015 * fis[:, t] - 0.03 * une[:, t]
                   + eta + d_t[t] + eps[:, t])
        y_prev, fis_prev, une_prev = y[:, t], fis[:, t], une[:, t]
    rows = []
    start = np.zeros(n, int)
    end = np.full(n, T)
    late = rng.choice(n, size=20, replace=False)
    start[late] = rng.integers(1, 3, size=20)
    early = rng.choice(np.setdiff1d(np.arange(n), late), size=10, replace=False)
    end[early] = T - rng.integers(1, 3, size=10)
    prov = rng.integers(1, 18, n)
    for i in range(n):
        for k in range(start[i], end[i]):
            t = burn + k
            rows.append({"지역코드": f"R{i + 1:03d}", "연도": years[k], "복지지출": round(y[i, t], 6),
                         "고령인구비율": round(old[i, t], 4), "재정자립도": round(fis[i, t], 4),
                         "실업률": round(une[i, t], 4), "시도": f"P{prov[i]:02d}"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = make()
    out = os.path.join(HERE, "Dynamic_Panel_Welfare.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(out, df.shape)
