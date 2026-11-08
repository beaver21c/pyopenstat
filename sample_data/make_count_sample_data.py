"""
Synthetic sample for count-data regression (seed 2026): Count_HealthVisits.csv

1,500 adults with a 3–12 month observation window (관찰개월, exposure), 40 districts (시군구, cluster variable):

- 외래방문   : outpatient visits — zero-inflated NB2 (structural non-users) with exposure; excess zeros
- 응급실방문 : emergency visits — plain Poisson with exposure (Poisson fits; estat gof should not reject)
- 입원일수   : length of stay of the 약 25% hospitalized (missing otherwise) — zero-truncated NB2 (y >= 1)
- covariates : 연령, 여성, 만성질환수, 소득분위 (1–5, factor), 지역유형 (대도시·중소도시·농어촌, text factor), 의료급여

Run: python sample_data/make_count_sample_data.py   (writes the CSV next to this file)
The data are synthetic; make_count() also returns the true parameters used to generate them.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 2026
REGIONS = ["대도시", "중소도시", "농어촌"]


def _rnb2(rng, mu, alpha):
    """NB2 draws: Poisson with gamma(1/alpha, alpha) heterogeneity."""
    lam = mu * rng.gamma(1.0 / alpha, alpha, size=len(mu))
    return rng.poisson(lam)


def make_count(n: int = 1500, seed: int = SEED):
    rng = np.random.default_rng(seed)
    sgg = rng.integers(1, 41, n)
    district_effect = rng.normal(0, 0.15, 41)[sgg]
    age = rng.uniform(20, 85, n).round()
    female = (rng.uniform(size=n) < 0.52).astype(int)
    chronic = np.minimum(rng.poisson(0.4 + 0.025 * (age - 20)), 6)
    income = rng.integers(1, 6, n)
    region = np.array(REGIONS)[rng.choice(3, n, p=[0.45, 0.35, 0.20])]
    aid = (rng.uniform(size=n) < np.where(income == 1, 0.30, 0.03)).astype(int)
    months = rng.integers(3, 13, n)
    za = (age - 50) / 10.0
    inc_eff = np.array([0.0, 0.0, -0.05, -0.10, -0.12, -0.20])[income]
    reg_eff = pd.Series(region).map({"대도시": 0.0, "중소도시": -0.08, "농어촌": -0.25}).to_numpy()

    # outpatient visits: ZINB (logit inflation: young, healthy, men more often structural non-users)
    beta = {"_cons": 0.55, "연령z": 0.10, "여성": 0.20, "만성질환수": 0.30, "의료급여": 0.25}
    eta = (beta["_cons"] + beta["연령z"] * za + beta["여성"] * female + beta["만성질환수"] * chronic
           + beta["의료급여"] * aid + inc_eff + reg_eff + district_effect + np.log(months / 12.0))
    gamma = {"_cons": -0.6, "연령z": -0.35, "만성질환수": -0.55, "여성": -0.30}
    zeta = gamma["_cons"] + gamma["연령z"] * za + gamma["만성질환수"] * chronic + gamma["여성"] * female
    structural = rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-zeta))
    alpha = 0.6
    visits = np.where(structural, 0, _rnb2(rng, np.exp(eta), alpha))

    # emergency visits: Poisson with exposure
    eta_er = -1.6 + 0.12 * za + 0.25 * chronic + 0.30 * aid - 0.15 * female + np.log(months / 12.0)
    er = rng.poisson(np.exp(eta_er))

    # length of stay among the hospitalized: zero-truncated NB2 (redraw zeros)
    hosp = rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-1.6 + 0.35 * za + 0.30 * chronic)))
    mu_los = np.exp(1.2 + 0.12 * za + 0.15 * chronic + 0.20 * (region == "농어촌") - 0.10 * female)
    los = np.full(n, np.nan)
    idx = np.where(hosp)[0]
    draw = _rnb2(rng, mu_los[idx], 0.5)
    for _ in range(200):
        zero = draw == 0
        if not zero.any():
            break
        draw[zero] = _rnb2(rng, mu_los[idx][zero], 0.5)
    los[idx] = draw

    df = pd.DataFrame({
        "ID": np.arange(1, n + 1),
        "시군구": [f"S{c:02d}" for c in sgg],
        "연령": age.astype(int),
        "여성": female,
        "만성질환수": chronic.astype(int),
        "소득분위": income.astype(int),
        "지역유형": region,
        "의료급여": aid,
        "관찰개월": months.astype(int),
        "외래방문": visits.astype(int),
        "응급실방문": er.astype(int),
        "입원일수": los,
    })
    # a few missing values in covariates (listwise deletion is reported by the analysis)
    miss = rng.uniform(size=n) < 0.01
    df.loc[miss, "만성질환수"] = np.nan
    df["만성질환수"] = df["만성질환수"].astype("Int64")
    df["입원일수"] = df["입원일수"].astype("Int64")
    truth = {"beta": beta, "gamma": gamma, "alpha": alpha}
    return df, truth


def main():
    df, _ = make_count()
    df.to_csv(os.path.join(HERE, "Count_HealthVisits.csv"), index=False, encoding="utf-8")
    print(f"Count_HealthVisits.csv: {len(df)} rows, 외래방문 zeros = {(df['외래방문'] == 0).mean():.1%}, "
          f"hospitalized = {df['입원일수'].notna().sum()}")


if __name__ == "__main__":
    main()
