"""
Synthetic samples for limited-dependent-variable models (seed 2026)

1) LimDep_Employment_Wage.csv  — 1,600 women aged 25–59: employment (selection) and hourly wage
   - 취업여부   : employed (0/1) — probit selection on 연령, 교육연수, 미취학자녀수, 배우자소득, 지역유형
   - 시간당임금 : hourly wage (만원) observed only when employed (missing otherwise); log-free linear outcome on
                 교육연수, 경력, 지역유형; selection and wage errors correlated (rho = 0.65, sigma = 0.50)
   - 고임금여부 : 1 if 시간당임금 ≥ 1.4 among the employed (missing otherwise) — heckprobit outcome
   - 시군구     : 60 districts (cluster variable)
   Models: heckman (ML / twostep), heckprobit, probit / logit (취업여부)

2) LimDep_Satisfaction_Service.csv — 1,200 older adults using long-term care information
   - 서비스만족도 : 1–5 ordered satisfaction (ordered logit with cutpoints -1.6, -0.4, 0.9, 2.2)
   - 이용서비스   : 재가방문 / 주간보호 / 시설입소 / 미이용 (multinomial logit, 미이용 most frequent)
   - 정책인지     : aware of the integrated care policy (0/1, logit)
   - covariates   : 연령, 여성, 독거, 일상생활장애(ADL, 0–6), 월소득(백만원), 지역유형 (대도시·중소도시·농어촌)
   Models: ologit / oprobit, mlogit, logit / probit / cloglog

3) LimDep_MedicalCost.csv — 1,000 adults, annual out-of-pocket medical expenditure (만원)
   - 본인부담의료비 : latent normal y* censored at 0 (about 30% zeros) and top-coded at 300
   - covariates     : 연령, 여성, 만성질환수, 민간보험, 월소득(백만원), 지역유형
   Models: tobit ll(0) [ul(300)], truncreg ll(0) on the positive part

Run: python sample_data/make_limdep_sample_data.py   (writes the CSVs next to this file)
The data are synthetic; each make_* function also returns the true parameters used to generate them.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 2026
REGIONS = ["대도시", "중소도시", "농어촌"]


def make_employment(n: int = 1600, seed: int = SEED):
    rng = np.random.default_rng(seed)
    district = rng.integers(1, 61, n)
    age = rng.integers(25, 60, n)
    educ = np.clip(np.round(rng.normal(13.2, 2.4, n)), 6, 20).astype(int)
    kids = rng.choice([0, 1, 2, 3], n, p=[0.55, 0.25, 0.15, 0.05])
    spouse = np.round(np.clip(rng.lognormal(1.2, 0.6, n), 0, 30), 2)          # 배우자소득 (백만원/월)
    region = np.array(REGIONS)[rng.choice(3, n, p=[0.45, 0.35, 0.20])]
    exper = np.clip(np.round(age - educ - 6 - rng.uniform(0, 8, n)), 0, None).astype(int)
    reg_s = pd.Series(region).map({"대도시": 0.0, "중소도시": 0.10, "농어촌": 0.20}).to_numpy()
    reg_w = pd.Series(region).map({"대도시": 0.0, "중소도시": -0.12, "농어촌": -0.25}).to_numpy()
    gamma = {"_cons": -1.10, "연령": 0.020, "교육연수": 0.12, "미취학자녀수": -0.45, "배우자소득": -0.08}
    beta = {"_cons": 0.10, "교육연수": 0.085, "경력": 0.018}
    rho, sigma = 0.65, 0.50
    u2 = rng.normal(size=n)
    u1 = sigma * (rho * u2 + np.sqrt(1 - rho ** 2) * rng.normal(size=n))
    zg = (gamma["_cons"] + gamma["연령"] * age + gamma["교육연수"] * educ + gamma["미취학자녀수"] * kids
          + gamma["배우자소득"] * spouse + reg_s)
    employed = (zg + u2 > 0).astype(int)
    wage = beta["_cons"] + beta["교육연수"] * educ + beta["경력"] * exper + reg_w + u1
    wage = np.round(np.clip(wage, 0.05, None), 3)
    df = pd.DataFrame({
        "개인ID": np.arange(1, n + 1), "시군구": district, "연령": age, "교육연수": educ, "경력": exper,
        "미취학자녀수": kids, "배우자소득": spouse, "지역유형": region, "취업여부": employed,
        "시간당임금": np.where(employed == 1, wage, np.nan),
    })
    df["고임금여부"] = np.where(employed == 1, (df["시간당임금"] >= 1.4).astype(float), np.nan)
    return df, {"gamma": gamma, "beta": beta, "rho": rho, "sigma": sigma}


def make_satisfaction(n: int = 1200, seed: int = SEED + 1):
    rng = np.random.default_rng(seed)
    age = rng.integers(65, 96, n)
    female = (rng.uniform(size=n) < 0.58).astype(int)
    alone = (rng.uniform(size=n) < 0.35).astype(int)
    adl = np.minimum(rng.poisson(0.6 + 0.06 * (age - 65)), 6)
    income = np.round(np.clip(rng.lognormal(0.3, 0.55, n), 0.2, 8), 2)
    region = np.array(REGIONS)[rng.choice(3, n, p=[0.40, 0.35, 0.25])]
    za = (age - 75) / 10.0
    # ordered satisfaction (logit)
    reg_o = pd.Series(region).map({"대도시": 0.0, "중소도시": 0.15, "농어촌": 0.35}).to_numpy()
    xb = 0.30 * za + 0.25 * female - 0.40 * alone - 0.18 * adl + 0.20 * income + reg_o
    cuts = np.array([-1.6, -0.4, 0.9, 2.2])
    ystar = xb + rng.logistic(size=n)
    sat = 1 + (ystar[:, None] > cuts[None, :]).sum(axis=1)
    # multinomial service use (base: 미이용)
    reg_m = pd.Series(region).map({"대도시": 0.0, "중소도시": -0.10, "농어촌": -0.40}).to_numpy()
    U = np.zeros((n, 4))
    U[:, 1] = -1.4 + 0.45 * adl + 0.30 * alone + 0.20 * za + reg_m              # 재가방문
    U[:, 2] = -1.9 + 0.35 * adl + 0.10 * female + 0.15 * income + reg_m         # 주간보호
    U[:, 3] = -3.4 + 0.65 * adl + 0.50 * alone + 0.45 * za                      # 시설입소
    U += rng.gumbel(size=(n, 4))
    labels = np.array(["미이용", "재가방문", "주간보호", "시설입소"])
    service = labels[U.argmax(axis=1)]
    # binary awareness (logit)
    eta = -0.5 - 0.35 * za + 0.15 * female - 0.30 * alone + 0.30 * income - 0.25 * (region == "농어촌")
    aware = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(int)
    df = pd.DataFrame({"응답자ID": np.arange(1, n + 1), "연령": age, "여성": female, "독거": alone,
                       "일상생활장애": adl, "월소득": income, "지역유형": region, "서비스만족도": sat,
                       "이용서비스": service, "정책인지": aware})
    return df, {"cuts": cuts.tolist()}


def make_medical_cost(n: int = 1000, seed: int = SEED + 2):
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 86, n)
    female = (rng.uniform(size=n) < 0.52).astype(int)
    chronic = np.minimum(rng.poisson(0.3 + 0.03 * (age - 20)), 6)
    private = (rng.uniform(size=n) < np.clip(0.75 - 0.006 * (age - 20), 0.2, 0.9)).astype(int)
    income = np.round(np.clip(rng.lognormal(1.0, 0.5, n), 0.3, 15), 2)
    region = np.array(REGIONS)[rng.choice(3, n, p=[0.45, 0.35, 0.20])]
    reg = pd.Series(region).map({"대도시": 0.0, "중소도시": -5.0, "농어촌": -12.0}).to_numpy()
    beta = {"_cons": -60.0, "연령": 1.2, "여성": 12.0, "만성질환수": 28.0, "민간보험": 15.0, "월소득": 4.0}
    sigma = 70.0
    ystar = (beta["_cons"] + beta["연령"] * age + beta["여성"] * female + beta["만성질환수"] * chronic
             + beta["민간보험"] * private + beta["월소득"] * income + reg + rng.normal(0, sigma, n))
    y = np.clip(np.round(ystar, 1), 0, 300)
    df = pd.DataFrame({"ID": np.arange(1, n + 1), "연령": age, "여성": female, "만성질환수": chronic,
                       "민간보험": private, "월소득": income, "지역유형": region, "본인부담의료비": y})
    return df, {"beta": beta, "sigma": sigma, "ll": 0.0, "ul": 300.0}


def main():
    for name, fn in (("LimDep_Employment_Wage.csv", make_employment),
                     ("LimDep_Satisfaction_Service.csv", make_satisfaction),
                     ("LimDep_MedicalCost.csv", make_medical_cost)):
        df, _ = fn()
        path = os.path.join(HERE, name)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{name}: {len(df)} rows -> {path}")


if __name__ == "__main__":
    main()
