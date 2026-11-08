"""
Synthetic sample for SEM with categorical (binary / ordinal) variables — WLSMV (Mplus CATEGORICAL) (seed 2026).

- SEM_Categorical_Care.csv : 800 older adults using community care services
    covariates      연령 (continuous), 여성, 독거 (0/1)
    서비스만족      만족_친절, 만족_시간, 만족_전문성, 만족_소통      Likert 1–5 (ordinal)
    기관신뢰        신뢰_정보, 신뢰_약속, 신뢰_안전                    Likert 1–4 (ordinal)
    돌봄필요        이용_방문요양, 이용_주간보호, 이용_방문간호, 이용_식사배달, 이용_이동지원   0/1 (binary)
    건강상태        주관적건강 (ordinal 1–5), 우울점수, 일상활동점수   (mixed ordinal + continuous)
    outcomes        재이용의향 (0/1), 추천의향 (1–3 ordinal), 삶의만족도 (continuous 0–10)
  Latent responses follow probit (normal) models; about 2–3 % missing in three variables.

Run: python sample_data/make_semcat_sample_data.py   (writes the CSV next to this file)
"""

import os

import numpy as np
import pandas as pd


def _cut(ystar, cuts):
    return np.searchsorted(np.asarray(cuts, float), ystar) + 0


def make_care(n: int = 800, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.round(rng.uniform(65, 90, n), 0)
    female = (rng.random(n) < 0.58).astype(int)
    alone = (rng.random(n) < 0.35).astype(int)
    agec = (age - 77) / 7
    # latent variables
    need = 0.45 * agec + 0.35 * alone + rng.normal(0, 0.9, n)                 # 돌봄필요
    health = -0.40 * agec - 0.30 * need + rng.normal(0, 0.85, n)              # 건강상태
    sat = 0.20 * agec + 0.15 * female + 0.25 * health + rng.normal(0, 0.95, n)  # 서비스만족
    trust = 0.55 * sat + rng.normal(0, 0.80, n)                               # 기관신뢰
    df = pd.DataFrame({"ID": np.arange(1, n + 1), "연령": age, "여성": female, "독거": alone})
    lik5 = [-1.6, -0.7, 0.3, 1.3]
    for name, lam, shift in (("만족_친절", 0.85, -0.3), ("만족_시간", 0.70, 0.1), ("만족_전문성", 0.80, -0.1),
                             ("만족_소통", 0.65, 0.2)):
        ys = lam * sat + rng.normal(0, np.sqrt(1 - 0.5 * lam ** 2), n) + shift
        df[name] = _cut(ys, lik5) + 1
    lik4 = [-1.2, -0.1, 1.0]
    for name, lam in (("신뢰_정보", 0.80), ("신뢰_약속", 0.75), ("신뢰_안전", 0.60)):
        ys = lam * trust + rng.normal(0, 0.7, n)
        df[name] = _cut(ys, lik4) + 1
    for name, lam, th in (("이용_방문요양", 0.9, 0.2), ("이용_주간보호", 0.7, 0.6), ("이용_방문간호", 0.8, 0.9),
                          ("이용_식사배달", 0.6, 0.3), ("이용_이동지원", 0.75, 0.7)):
        ys = lam * need + rng.normal(0, 0.75, n)
        df[name] = (ys > th).astype(int)
    df["주관적건강"] = _cut(0.8 * health + rng.normal(0, 0.6, n), [-1.3, -0.4, 0.5, 1.4]) + 1
    df["우울점수"] = np.round(12 - 3.2 * health + rng.normal(0, 2.6, n), 1)
    df["일상활동점수"] = np.round(70 + 9.0 * health - 4.0 * need + rng.normal(0, 7.0, n), 1)
    reuse_star = 0.75 * sat + 0.35 * trust + 0.2 * female + rng.normal(0, 1, n)
    df["재이용의향"] = (reuse_star > -0.4).astype(int)
    rec_star = 0.6 * reuse_star + 0.3 * sat + rng.normal(0, 0.8, n)
    df["추천의향"] = _cut(rec_star, [-0.4, 0.9]) + 1
    df["삶의만족도"] = np.round(np.clip(5.5 + 0.9 * sat + 0.8 * health - 0.25 * need + rng.normal(0, 1.4, n), 0, 10), 1)
    for col, rate in (("만족_소통", 0.02), ("신뢰_안전", 0.025), ("우울점수", 0.03)):
        m = rng.random(n) < rate
        df.loc[m, col] = np.nan
    for c in df.columns:
        if c not in ("우울점수", "일상활동점수", "삶의만족도", "연령") and df[c].notna().all():
            df[c] = df[c].astype(int)
    return df


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    d = make_care()
    d.to_csv(os.path.join(here, "SEM_Categorical_Care.csv"), index=False, encoding="utf-8-sig")
    print("SEM_Categorical_Care.csv", d.shape)
