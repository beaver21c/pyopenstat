"""
Synthetic samples for latent class / profile, latent transition and latent class growth analysis (seed 2026).

- Mixture_LCA_Care.csv     : 1,200 older adults, 3 care-need classes (6 binary + 1 ordinal care items,
                             3 continuous indicators, covariates 연령·독거, distal 월의료비·시설입소의향)
- Mixture_LTA_Long.csv     : 800 persons x 3 survey waves (long), 3 mental-health classes, 5 binary items,
                             measurement invariant across waves, time-specific transitions, wave-3 attrition
- Mixture_LCGA_Long.csv    : 600 persons x 5 years (long), 3 life-satisfaction trajectory classes, covariate
                             소득분위, MAR dropout depending on the previous observed outcome

Run: python sample_data/make_mixture_sample_data.py   (writes the CSVs next to this file)
The make_* functions also return the true structure used by tests/test_mixture.py.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 2026

LCA_BINARY = ["외출지원필요", "식사지원필요", "가사지원필요", "목욕지원필요", "투약관리필요", "정서지원필요"]
LCA_ORDINAL = "인지기능저하"
LCA_CONT = ["우울점수", "주간사회활동시간", "주관적건강"]


def make_lca(n: int = 1200, seed: int = SEED):
    rng = np.random.default_rng(seed)
    age = rng.uniform(65, 95, n).round()
    alone = (rng.uniform(size=n) < 0.35).astype(int)
    za = (age - 80) / 8.0
    # multinomial logit, reference = class 3 (집중돌봄형)
    gamma = np.array([[1.2, -0.9, -0.4],    # class 1 자립형: intercept, z(연령), 독거
                      [0.5, -0.3, 0.5]])    # class 2 부분지원형
    eta = np.column_stack([gamma[0, 0] + gamma[0, 1] * za + gamma[0, 2] * alone,
                           gamma[1, 0] + gamma[1, 1] * za + gamma[1, 2] * alone, np.zeros(n)])
    pc = np.exp(eta - eta.max(1, keepdims=True))
    pc /= pc.sum(1, keepdims=True)
    cls = (rng.uniform(size=(n, 1)) > np.cumsum(pc, 1)[:, :-1]).sum(1)
    pbin = np.array([[0.10, 0.08, 0.12, 0.05, 0.15, 0.20],
                     [0.65, 0.50, 0.70, 0.30, 0.55, 0.45],
                     [0.92, 0.88, 0.93, 0.85, 0.88, 0.65]])
    pord = np.array([[0.80, 0.17, 0.03], [0.40, 0.45, 0.15], [0.10, 0.35, 0.55]])
    mu = np.array([[8.0, 13.0, 4.0], [17.0, 7.0, 3.0], [26.0, 2.0, 2.0]])
    sd = np.array([5.0, 3.0, 0.6])
    dist_mu = np.array([15.0, 25.0, 40.0])
    dist_cat = np.array([[0.85, 0.12, 0.03], [0.60, 0.30, 0.10], [0.30, 0.40, 0.30]])
    df = pd.DataFrame({"ID": np.arange(1, n + 1)})
    for j, name in enumerate(LCA_BINARY):
        df[name] = (rng.uniform(size=n) < pbin[cls, j]).astype(float)
    df[LCA_ORDINAL] = (rng.uniform(size=(n, 1)) > np.cumsum(pord[cls], 1)[:, :-1]).sum(1).astype(float)
    for j, name in enumerate(LCA_CONT):
        df[name] = (mu[cls, j] + sd[j] * rng.standard_normal(n)).round(2)
    items = LCA_BINARY + [LCA_ORDINAL] + LCA_CONT
    miss = rng.uniform(size=(n, len(items))) < 0.03
    vals = df[items].to_numpy(dtype=float)
    vals[miss] = np.nan
    df[items] = vals
    df["연령"] = age
    df["독거"] = alone
    df["월의료비"] = (dist_mu[cls] + 8.0 * rng.standard_normal(n)).round(1)
    df["시설입소의향"] = (rng.uniform(size=(n, 1)) > np.cumsum(dist_cat[cls], 1)[:, :-1]).sum(1)
    truth = {"cls": cls, "pbin": pbin, "pord": pord, "mu": mu, "sd": sd, "gamma": gamma,
             "dist_mu": dist_mu, "dist_cat": dist_cat}
    return df, truth


LTA_ITEMS = ["우울감", "외로움", "수면문제", "무기력", "사회적고립"]


def make_lta(n: int = 800, seed: int = SEED + 1):
    rng = np.random.default_rng(seed)
    P = np.array([[0.08, 0.10, 0.15, 0.07, 0.05],
                  [0.65, 0.70, 0.45, 0.40, 0.25],
                  [0.85, 0.90, 0.80, 0.80, 0.75]])
    pi = np.array([0.55, 0.28, 0.17])
    tau = np.array([[[0.85, 0.12, 0.03], [0.20, 0.70, 0.10], [0.05, 0.15, 0.80]],
                    [[0.80, 0.15, 0.05], [0.15, 0.75, 0.10], [0.05, 0.10, 0.85]]])
    c = np.empty((n, 3), dtype=int)
    c[:, 0] = rng.choice(3, size=n, p=pi)
    for t in range(1, 3):
        cum = np.cumsum(tau[t - 1][c[:, t - 1]], 1)
        c[:, t] = (rng.uniform(size=(n, 1)) > cum[:, :-1]).sum(1)
    sex = (rng.uniform(size=n) < 0.5).astype(int)
    rows = []
    drop3 = rng.uniform(size=n) < 0.06
    for i in range(n):
        for t in range(3):
            if t == 2 and drop3[i]:
                continue
            row = {"ID": i + 1, "조사차수": t + 1, "성별": sex[i]}
            for j, name in enumerate(LTA_ITEMS):
                row[name] = float(rng.uniform() < P[c[i, t], j])
            rows.append(row)
    df = pd.DataFrame(rows)
    miss = rng.uniform(size=(len(df), len(LTA_ITEMS))) < 0.02
    vals = df[LTA_ITEMS].to_numpy(dtype=float)
    vals[miss] = np.nan
    df[LTA_ITEMS] = vals
    return df, {"cls": c, "P": P, "pi": pi, "tau": tau}


def make_lcga(n: int = 600, seed: int = SEED + 2):
    rng = np.random.default_rng(seed)
    years = np.arange(2019, 2024)
    t = years - years[0]
    income = rng.integers(1, 6, size=n)
    zi = income - 3.0
    eta = np.column_stack([0.4 + 0.6 * zi, 0.3 + 0.1 * zi, np.zeros(n)])  # ref = class 3
    pc = np.exp(eta - eta.max(1, keepdims=True))
    pc /= pc.sum(1, keepdims=True)
    cls = (rng.uniform(size=(n, 1)) > np.cumsum(pc, 1)[:, :-1]).sum(1)
    beta = np.array([[7.2, 0.05], [6.0, -0.45], [3.8, 0.40]])
    rsd = np.array([0.90, 0.80, 0.80, 0.85, 0.95])
    Y = beta[cls, :1] + beta[cls, 1:] * t[None] + rsd[None] * rng.standard_normal((n, len(t)))
    obs = np.ones_like(Y, dtype=bool)
    for k in range(1, len(t)):
        prev = np.where(obs[:, k - 1], Y[:, k - 1], np.nan)
        pdrop = 1.0 / (1.0 + np.exp(3.2 + 0.5 * (np.nan_to_num(prev, nan=5.0) - 5.0)))
        stay = obs[:, k - 1] & (rng.uniform(size=n) > pdrop)
        obs[:, k] = stay
    inter = rng.uniform(size=Y.shape) < 0.03
    rows = []
    for i in range(n):
        for k, yr in enumerate(years):
            if not obs[i, k]:
                continue
            rows.append({"ID": i + 1, "연도": int(yr), "삶의만족도": np.nan if inter[i, k] else round(float(Y[i, k]), 2),
                         "소득분위": int(income[i])})
    df = pd.DataFrame(rows)
    return df, {"cls": cls, "beta": beta, "rsd": rsd, "eta_slope": np.array([0.6, 0.1])}


def main():
    lca, _ = make_lca()
    lca.to_csv(os.path.join(HERE, "Mixture_LCA_Care.csv"), index=False, encoding="utf-8-sig")
    lta, _ = make_lta()
    lta.to_csv(os.path.join(HERE, "Mixture_LTA_Long.csv"), index=False, encoding="utf-8-sig")
    lcga, _ = make_lcga()
    lcga.to_csv(os.path.join(HERE, "Mixture_LCGA_Long.csv"), index=False, encoding="utf-8-sig")
    print("written:", len(lca), len(lta), len(lcga))


if __name__ == "__main__":
    main()
