"""
Synthetic three-level sample for multilevel (mixed-effects) models (seed 2026).

- Multilevel_School3L.csv : students (level 1) in classes (level 2) in schools (level 3)
    50 schools, 3-6 classes per school, 15-30 students per class (about 5,000 students).
    학급ID is numbered 1, 2, ... inside each school (unique only within 학교ID, as in many school surveys),
    so the class level must be read as nested in the school level (Stata `|| 학교ID: || 학급ID:`).

  level 1 (학생): 수학성취도 (outcome), 가정SES (standardized family SES), 성별 (0 = 남, 1 = 여), 학습시간 (hours/week)
  level 2 (학급): 학급규모 (students in class), 교사경력 (years)
  level 3 (학교): 학교유형 (공립·사립), 지역 (대도시·중소도시·읍면)

  True model (values used to generate the data):
    수학성취도 = 50 + 3.0·SES_within + 5.0·SES_school_mean + 1.5·여 + 0.6·학습시간 − 0.15·(학급규모 − 24)
                + 0.10·교사경력 + 2.0·사립 − 1.0·중소도시 − 2.5·읍면 + 1.0·가정SES·사립
                + u0_school + u1_school·가정SES + v0_class + v1_class·(학습시간 − 10) + e
    school: sd(u0) = 4.0, sd(u1) = 1.2, corr(u0, u1) = 0.3; class: sd(v0) = 2.5, sd(v1) = 0.3 (independent); sd(e) = 8.0
  Missing values: 수학성취도 1%, 학습시간 1.5% (listwise deletion in the analysis).

  Outcomes for generalized multilevel models (generated afterwards from a separate random stream, seed 2027,
  so the columns above do not depend on them):
    기초학력미달 (0/1): logit P = −1.6 − 0.9·가정SES − 0.08·(학습시간 − 10) + 0.4·읍면 + a_school + a_class,
                       sd(a_school) = 0.7, sd(a_class) = 0.45                                  (melogit)
    결석일수 (count):   log E = 0.9 − 0.25·가정SES − 0.3·여 + c_school + c_class, sd 0.35 and 0.25   (mepoisson)

Run: python sample_data/make_multilevel_sample_data.py   (writes the CSV next to this file)
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 2026

TRUE = {
    "sd_school_cons": 4.0, "sd_school_ses": 1.2, "corr_school": 0.3,
    "sd_class_cons": 2.5, "sd_class_hours": 0.3, "sd_resid": 8.0,
}


def make_school3l(n_school: int = 50, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    sid = 0
    cov_s = np.array([[TRUE["sd_school_cons"] ** 2, TRUE["corr_school"] * TRUE["sd_school_cons"] * TRUE["sd_school_ses"]],
                      [TRUE["corr_school"] * TRUE["sd_school_cons"] * TRUE["sd_school_ses"], TRUE["sd_school_ses"] ** 2]])
    for s in range(1, n_school + 1):
        private = int(rng.uniform() < 0.35)
        region = rng.choice(["대도시", "중소도시", "읍면"], p=[0.4, 0.4, 0.2])
        u0, u1 = rng.multivariate_normal([0.0, 0.0], cov_s)
        school_ses = rng.normal(0.3 * private - 0.3 * (region == "읍면"), 0.45)
        for c in range(1, int(rng.integers(3, 7)) + 1):
            size = int(rng.integers(15, 31))
            exper = int(rng.integers(1, 31))
            v0 = rng.normal(0.0, TRUE["sd_class_cons"])
            v1 = rng.normal(0.0, TRUE["sd_class_hours"])
            class_ses = school_ses + rng.normal(0.0, 0.2)
            ses = class_ses + rng.normal(0.0, 0.8, size)
            female = (rng.uniform(size=size) < 0.5).astype(int)
            hours = np.clip(rng.normal(10.0 + 1.5 * ses, 4.0), 0.0, 30.0).round(1)
            y = (50.0 + 3.0 * (ses - class_ses) + 5.0 * class_ses + 1.5 * female + 0.6 * hours
                 - 0.15 * (size - 24) + 0.10 * exper + 2.0 * private
                 - 1.0 * (region == "중소도시") - 2.5 * (region == "읍면") + 1.0 * ses * private
                 + u0 + u1 * ses + v0 + v1 * (hours - 10.0) + rng.normal(0.0, TRUE["sd_resid"], size))
            for i in range(size):
                sid += 1
                rows.append({"학생ID": sid, "학교ID": s, "학급ID": c, "수학성취도": round(float(y[i]), 2),
                             "가정SES": round(float(ses[i]), 3), "성별": int(female[i]), "학습시간": float(hours[i]),
                             "학급규모": size, "교사경력": exper, "학교유형": "사립" if private else "공립",
                             "지역": str(region)})
    df = pd.DataFrame(rows)
    n = len(df)
    y = df["수학성취도"].to_numpy(dtype=float)
    y[rng.uniform(size=n) < 0.01] = np.nan
    df["수학성취도"] = y
    h = df["학습시간"].to_numpy(dtype=float)
    h[rng.uniform(size=n) < 0.015] = np.nan
    df["학습시간"] = h
    rng2 = np.random.default_rng(seed + 1)
    sch = df["학교ID"].to_numpy()
    cls = df.groupby(["학교ID", "학급ID"]).ngroup().to_numpy()
    a_s = rng2.normal(0.0, 0.7, sch.max() + 1)[sch]
    a_c = rng2.normal(0.0, 0.45, cls.max() + 1)[cls]
    ses = df["가정SES"].to_numpy()
    hrs = np.nan_to_num(h, nan=10.0)
    eta = -1.6 - 0.9 * ses - 0.08 * (hrs - 10.0) + 0.4 * (df["지역"] == "읍면").to_numpy() + a_s + a_c
    df["기초학력미달"] = (rng2.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(int)
    c_s = rng2.normal(0.0, 0.35, sch.max() + 1)[sch]
    c_c = rng2.normal(0.0, 0.25, cls.max() + 1)[cls]
    mu = np.exp(0.9 - 0.25 * ses - 0.3 * df["성별"].to_numpy() + c_s + c_c)
    df["결석일수"] = rng2.poisson(mu)
    return df


def main() -> None:
    df = make_school3l()
    path = os.path.join(HERE, "Multilevel_School3L.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"{path}: {len(df)} rows, {df['학교ID'].nunique()} schools, "
          f"{df.groupby(['학교ID', '학급ID']).ngroups} classes")


if __name__ == "__main__":
    main()
