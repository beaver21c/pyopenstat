"""Synthetic long-term-care panel for nonlinear panel models (seeded, synthetic only).

Panel_Nonlinear_Care.csv — 420 older adults x waves 1-7, unbalanced: late entry, early exit and internal gaps
(about 8% of person-waves missing at random), some persons observed once.

    pid       person id (1 ...)
    wave      survey wave 1-7
    region    region code 1-12 (cluster variable containing persons)
    female    1 = woman (time-invariant)
    adl       ADL limitation score (time-varying; correlated with the person effect a_i)
    alone     1 = living alone (time-varying)
    months    months covered by the wave's visit record (6-12; exposure for count models)
    unmet     1 = unmet care need. logit: -1.2 + 0.7 adl + 0.5 alone - 0.4 female + u_i,
              u_i = 0.9 a_i + N(0, 0.8^2)  (person effect correlated with adl, so FE and RE differ)
    visits    outpatient visits: Poisson(months/12 * exp(0.6 + 0.25 adl + 0.2 alone + 0.15 female + v_i) * g_it),
              v_i = 0.3 a_i + N(0, 0.4^2), g_it ~ Gamma(shape 2, scale 0.5) (overdispersion within person)
    carehrs   weekly informal care hours, left-censored at 0:
              max(0, 1 + 3 adl + 4 alone - 1.5 female + c_i + e_it), c_i ~ N(0, 3^2), e_it ~ N(0, 4^2)
    oop_lo    out-of-pocket spending band (10,000 KRW): lower bound; missing = left-censored (< 20)
    oop_hi    upper bound; missing = right-censored (>= 100); oop_lo == oop_hi for exactly reported values (~10%)
              latent: 50 + 15 adl + 10 alone + 5 female + s_i + e_it, s_i ~ N(0, 15^2), e_it ~ N(0, 20^2)
    sat       service satisfaction 1-4 (ordinal): latent -0.6 adl + 0.5 alone + 0.3 female + q_i + logistic,
              q_i ~ N(0, 1), cut points -1.5, 0, 1.5; about 1% missing

Run: python sample_data/make_panel_nonlinear_sample_data.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def make(n: int = 420, waves: int = 7, seed: int = 20261108) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)                                   # latent frailty
    female = (rng.uniform(size=n) < 0.58).astype(int)
    region = rng.integers(1, 13, n)
    u = 0.9 * a + rng.normal(0, 0.8, n)
    v = 0.3 * a + rng.normal(0, 0.4, n)
    c = rng.normal(0, 3, n)
    s = rng.normal(0, 15, n)
    q = rng.normal(0, 1, n)
    start = rng.choice([1, 1, 1, 1, 2, 3], size=n)
    stop = np.minimum(waves, start + rng.integers(0, waves + 2, size=n))
    rows = []
    alone_prev = (rng.uniform(size=n) < 0.3).astype(int)
    for i in range(n):
        for w in range(1, waves + 1):
            # living alone evolves for everyone (Markov), observed only in kept waves
            p = 0.9 if alone_prev[i] else 0.07
            alone_prev[i] = int(rng.uniform() < p)
            if w < start[i] or w > stop[i]:
                continue
            if w > start[i] and rng.uniform() < 0.08:          # internal gap
                continue
            adl = 0.6 * a[i] + 0.15 * (w - 1) + rng.normal(0, 1)
            alone = alone_prev[i]
            months = int(rng.integers(6, 13))
            eta_b = -1.2 + 0.7 * adl + 0.5 * alone - 0.4 * female[i] + u[i]
            unmet = int(rng.uniform() < 1 / (1 + np.exp(-eta_b)))
            mu = months / 12 * np.exp(0.6 + 0.25 * adl + 0.2 * alone + 0.15 * female[i] + v[i])
            visits = int(rng.poisson(mu * rng.gamma(2.0, 0.5)))
            hrs = max(0.0, 1 + 3 * adl + 4 * alone - 1.5 * female[i] + c[i] + rng.normal(0, 4))
            oop = 50 + 15 * adl + 10 * alone + 5 * female[i] + s[i] + rng.normal(0, 20)
            if rng.uniform() < 0.10:
                lo = hi = round(oop, 1)
            elif oop < 20:
                lo, hi = np.nan, 20.0
            elif oop >= 100:
                lo, hi = 100.0, np.nan
            else:
                lo = 20.0 * np.floor(oop / 20.0)
                hi = lo + 20.0
            lat = -0.6 * adl + 0.5 * alone + 0.3 * female[i] + q[i] + rng.logistic()
            sat = 1 + int(lat > -1.5) + int(lat > 0) + int(lat > 1.5)
            if rng.uniform() < 0.01:
                sat = np.nan
            rows.append((i + 1, w, region[i], female[i], round(adl, 3), alone, months, unmet, visits,
                         round(hrs, 2), lo, hi, sat))
    df = pd.DataFrame(rows, columns=["pid", "wave", "region", "female", "adl", "alone", "months", "unmet", "visits",
                                     "carehrs", "oop_lo", "oop_hi", "sat"])
    df["sat"] = df["sat"].astype("Int64")
    return df


if __name__ == "__main__":
    out = os.path.join(HERE, "Panel_Nonlinear_Care.csv")
    d = make()
    d.to_csv(out, index=False)
    ti = d.groupby("pid").size()
    print(f"wrote {out}: {len(d)} rows, {d.pid.nunique()} persons, T_i min {ti.min()} max {ti.max()} mean {ti.mean():.2f}")
    g = d.groupby("pid")["unmet"].agg(["min", "max"])
    print("unmet all-0/all-1 persons:", int(((g["min"] == g["max"])).sum()), " visits all-zero persons:",
          int((d.groupby("pid")["visits"].max() == 0).sum()), " carehrs censored:", float((d.carehrs == 0).mean()))
