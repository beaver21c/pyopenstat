"""
Synthetic longitudinal samples for the SEM model templates (seed 2026, generated from known population models).

- SEM_Longitudinal_Panel.csv : 800 persons x 5 waves (wide), observed repeated measures
    우울_1..5 · 외로움_1..5      RI-CLPM (random intercepts depend on 연령·여성, stationary within-person lagged paths)
    건강_1..5 · 의료이용_1..5    CLPM (no stable between-person component)
    삶의만족도_1..5             conditional linear growth (time-invariant 연령·소득, time-varying 건강_t)
    사회참여_1..5 · 여가활동_1..5 parallel-process linear growth (slopes regressed on both intercepts)
    인지기능_1..5               quadratic growth
    적응_1..5                   latent-basis growth (time scores 0, .45, .72, .90, 1)
    자아존중감_1..5             univariate dual change score model
- SEM_Longitudinal_Items.csv : 800 persons x 4 waves (wide), 3 items per construct and wave
    우울A..C_1..4 · 사회적지지A..C_1..4   multiple-indicator CLPM (scalar invariance, stable item-specific residuals)
    삶의질A..C_1..4                       second-order (curve-of-factors) linear growth

Both files have monotone MCAR dropout (3% of the remaining persons at each wave after the first); covariates are
complete. Run: python sample_data/make_semtpl_sample_data.py (writes the CSVs next to this file). The make_*
functions also return the population values used by tests/test_sem_templates.py.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 2026
N = 800

TRUE = {
    "riclpm": {"ar_x": 0.40, "ar_y": 0.35, "cl_x_to_y": 0.15, "cl_y_to_x": 0.20,
               "ri_var": (4.0, 2.0), "ri_cov": 1.2, "w1_var": (3.0, 2.0), "w1_cov": 0.8,
               "innov_var": (2.0, 1.4), "innov_cov": 0.4, "ri_on_age": (0.05, 0.03), "ri_on_female": (0.8, 0.5)},
    "clpm": {"ar_x": 0.60, "ar_y": 0.50, "cl_x_to_y": -0.40, "cl_y_to_x": -0.05,
             "res_var": (0.30, 1.20), "res_cov": -0.10},
    "conditional": {"i_int": 2.4, "i_on_age": 0.010, "i_on_income": 0.20, "s_int": 0.30, "s_on_age": -0.004,
                    "s_on_income": 0.03, "tvc": 0.25, "res_i": 0.60, "res_s": 0.04, "res_is": -0.03,
                    "e_var": (0.30, 0.28, 0.26, 0.28, 0.32)},
    "parallel": {"i_mean": (3.0, 2.5), "i_var": (0.50, 0.40), "i_cov": 0.15,
                 "sx_int": 0.30, "sx_on_ix": -0.06, "sx_on_iy": 0.04, "sy_int": 0.10, "sy_on_iy": -0.05,
                 "sy_on_ix": 0.08, "s_res_var": (0.03, 0.02), "s_res_cov": 0.006, "e_var": (0.25, 0.20)},
    "quadratic": {"mean": (27.0, -0.20, -0.10),
                  "cov": ((4.0, 0.30, -0.05), (0.30, 0.50, -0.05), (-0.05, -0.05, 0.02)), "e_var": 0.8},
    "basis": {"basis": (0.0, 0.45, 0.72, 0.90, 1.0), "mean": (2.0, 1.5), "var": (0.50, 0.30), "cov": 0.05,
              "e_var": (0.15, 0.14, 0.13, 0.13, 0.15)},
    "lcs": {"l1_mean": 30.0, "l1_var": 16.0, "g_mean": 3.0, "g_var": 1.0, "cov": 1.0, "beta": -0.08, "e_var": 4.0},
    "latent_clpm": {"load_x": (1.0, 0.85, 0.90), "load_y": (1.0, 1.10, 0.80), "nu_x": (2.0, 1.9, 2.1),
                    "nu_y": (3.0, 2.8, 3.1), "f1_cov": ((0.40, -0.12), (-0.12, 0.50)),
                    "ar_x": 0.50, "ar_y": 0.55, "cl_y_to_x": -0.20, "cl_x_to_y": -0.10, "alpha": (0.05, -0.03),
                    "zeta_var": (0.20, 0.25), "zeta_cov": -0.04, "u_var": (0.05, 0.06), "v_var": (0.15, 0.18)},
    "second_order": {"load": (1.0, 0.90, 1.10), "nu": (0.0, 0.30, -0.20), "i_mean": 3.0, "s_mean": 0.15,
                     "i_var": 0.50, "s_var": 0.04, "is_cov": 0.01, "d_var": 0.15, "u_var": 0.04, "v_var": 0.16},
}


def _mvn(rng, mean, cov, n):
    return rng.multivariate_normal(np.asarray(mean, float), np.asarray(cov, float), size=n)


def _dropout(rng, n, T, p=0.03):
    """Monotone MCAR dropout: wave index (0-based) from which a person is missing (T = never)."""
    start = np.full(n, T)
    for t in range(1, T):
        drop = (start == T) & (rng.uniform(size=n) < p)
        start[drop] = t
    return start


def make_panel(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    T = 5
    age = np.clip(rng.normal(60, 10, n), 30, 90).round()
    female = (rng.uniform(size=n) < 0.52).astype(int)
    income = np.clip(rng.normal(4.5, 1.5, n), 0.5, None).round(2)
    df = pd.DataFrame({"ID": np.arange(1, n + 1), "연령": age, "여성": female, "소득": income})
    series = {}

    # --- RI-CLPM: 우울 (x), 외로움 (y)
    p = TRUE["riclpm"]
    zeta = _mvn(rng, [0, 0], [[p["ri_var"][0], p["ri_cov"]], [p["ri_cov"], p["ri_var"][1]]], n)
    ri_x = p["ri_on_age"][0] * (age - 60) + p["ri_on_female"][0] * female + zeta[:, 0]
    ri_y = p["ri_on_age"][1] * (age - 60) + p["ri_on_female"][1] * female + zeta[:, 1]
    w = _mvn(rng, [0, 0], [[p["w1_var"][0], p["w1_cov"]], [p["w1_cov"], p["w1_var"][1]]], n)
    wx, wy = [w[:, 0]], [w[:, 1]]
    icov = [[p["innov_var"][0], p["innov_cov"]], [p["innov_cov"], p["innov_var"][1]]]
    for t in range(1, T):
        u = _mvn(rng, [0, 0], icov, n)
        wx.append(p["ar_x"] * wx[-1] + p["cl_y_to_x"] * wy[-1] + u[:, 0])
        wy.append(p["ar_y"] * wy[-1] + p["cl_x_to_y"] * wx[-2] + u[:, 1])
    mu_x, mu_y = [12.0, 12.3, 12.5, 12.8, 13.0], [7.0, 7.1, 7.3, 7.4, 7.6]
    series["우울"] = [mu_x[t] + ri_x + wx[t] for t in range(T)]
    series["외로움"] = [mu_y[t] + ri_y + wy[t] for t in range(T)]

    # --- CLPM: 건강 (x), 의료이용 (y)
    p = TRUE["clpm"]
    xy = _mvn(rng, [3.2, 5.0], [[0.64, -0.30], [-0.30, 2.25]], n)
    hx, hy = [xy[:, 0]], [xy[:, 1]]
    a_t, b_t = [1.45, 1.43, 1.40, 1.38], [3.75, 3.80, 3.85, 3.90]
    rc = [[p["res_var"][0], p["res_cov"]], [p["res_cov"], p["res_var"][1]]]
    for t in range(1, T):
        u = _mvn(rng, [0, 0], rc, n)
        hx.append(a_t[t - 1] + p["ar_x"] * hx[-1] + p["cl_y_to_x"] * hy[-1] + u[:, 0])
        hy.append(b_t[t - 1] + p["ar_y"] * hy[-1] + p["cl_x_to_y"] * hx[-2] + u[:, 1])
    series["건강"], series["의료이용"] = hx, hy

    # --- conditional LGM: 삶의만족도 with TIC 연령·소득 and TVC 건강_t
    p = TRUE["conditional"]
    z = _mvn(rng, [0, 0], [[p["res_i"], p["res_is"]], [p["res_is"], p["res_s"]]], n)
    I = p["i_int"] + p["i_on_age"] * age + p["i_on_income"] * income + z[:, 0]
    S = p["s_int"] + p["s_on_age"] * age + p["s_on_income"] * income + z[:, 1]
    series["삶의만족도"] = [I + S * t + p["tvc"] * hx[t] + rng.normal(0, np.sqrt(p["e_var"][t]), n) for t in range(T)]

    # --- parallel process: 사회참여 (x), 여가활동 (y)
    p = TRUE["parallel"]
    ii = _mvn(rng, p["i_mean"], [[p["i_var"][0], p["i_cov"]], [p["i_cov"], p["i_var"][1]]], n)
    sr = _mvn(rng, [0, 0], [[p["s_res_var"][0], p["s_res_cov"]], [p["s_res_cov"], p["s_res_var"][1]]], n)
    Sx = p["sx_int"] + p["sx_on_ix"] * ii[:, 0] + p["sx_on_iy"] * ii[:, 1] + sr[:, 0]
    Sy = p["sy_int"] + p["sy_on_iy"] * ii[:, 1] + p["sy_on_ix"] * ii[:, 0] + sr[:, 1]
    series["사회참여"] = [ii[:, 0] + Sx * t + rng.normal(0, np.sqrt(p["e_var"][0]), n) for t in range(T)]
    series["여가활동"] = [ii[:, 1] + Sy * t + rng.normal(0, np.sqrt(p["e_var"][1]), n) for t in range(T)]

    # --- quadratic LGM: 인지기능
    p = TRUE["quadratic"]
    g = _mvn(rng, p["mean"], p["cov"], n)
    series["인지기능"] = [g[:, 0] + g[:, 1] * t + g[:, 2] * t * t + rng.normal(0, np.sqrt(p["e_var"]), n)
                      for t in range(T)]

    # --- latent basis LGM: 적응
    p = TRUE["basis"]
    g = _mvn(rng, p["mean"], [[p["var"][0], p["cov"]], [p["cov"], p["var"][1]]], n)
    series["적응"] = [g[:, 0] + g[:, 1] * p["basis"][t] + rng.normal(0, np.sqrt(p["e_var"][t]), n) for t in range(T)]

    # --- dual change score: 자아존중감
    p = TRUE["lcs"]
    g = _mvn(rng, [p["l1_mean"], p["g_mean"]], [[p["l1_var"], p["cov"]], [p["cov"], p["g_var"]]], n)
    L = [g[:, 0]]
    for t in range(1, T):
        L.append(L[-1] + g[:, 1] + p["beta"] * L[-1])
    series["자아존중감"] = [L[t] + rng.normal(0, np.sqrt(p["e_var"]), n) for t in range(T)]

    start = _dropout(rng, n, T)
    for stem, vals in series.items():
        for t in range(T):
            v = np.round(vals[t], 2)
            v[start <= t] = np.nan
            df[f"{stem}_{t + 1}"] = v
    return df


def make_items(n: int = N, seed: int = SEED + 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    T = 4
    df = pd.DataFrame({"ID": np.arange(1, n + 1)})
    p = TRUE["latent_clpm"]
    f1 = _mvn(rng, [0, 0], p["f1_cov"], n)
    Fx, Fy = [f1[:, 0]], [f1[:, 1]]
    zc = [[p["zeta_var"][0], p["zeta_cov"]], [p["zeta_cov"], p["zeta_var"][1]]]
    for t in range(1, T):
        zt = _mvn(rng, [0, 0], zc, n)
        Fx.append(p["alpha"][0] + p["ar_x"] * Fx[-1] + p["cl_y_to_x"] * Fy[-1] + zt[:, 0])
        Fy.append(p["alpha"][1] + p["ar_y"] * Fy[-1] + p["cl_x_to_y"] * Fx[-2] + zt[:, 1])
    items = {}
    for stem, F, lam, nu, k in (("우울", Fx, p["load_x"], p["nu_x"], 0), ("사회적지지", Fy, p["load_y"], p["nu_y"], 1)):
        for j, code in enumerate("ABC"):
            u = rng.normal(0, np.sqrt(p["u_var"][k]), n)          # stable item-specific part
            items[f"{stem}{code}"] = [nu[j] + lam[j] * F[t] + u + rng.normal(0, np.sqrt(p["v_var"][k]), n)
                                      for t in range(T)]
    p = TRUE["second_order"]
    g = _mvn(rng, [p["i_mean"], p["s_mean"]], [[p["i_var"], p["is_cov"]], [p["is_cov"], p["s_var"]]], n)
    F = [g[:, 0] + g[:, 1] * t + rng.normal(0, np.sqrt(p["d_var"]), n) for t in range(T)]
    for j, code in enumerate("ABC"):
        u = rng.normal(0, np.sqrt(p["u_var"]), n)
        items[f"삶의질{code}"] = [p["nu"][j] + p["load"][j] * F[t] + u + rng.normal(0, np.sqrt(p["v_var"]), n)
                               for t in range(T)]
    start = _dropout(rng, n, T)
    for stem, vals in items.items():
        for t in range(T):
            v = np.round(vals[t], 2)
            v[start <= t] = np.nan
            df[f"{stem}_{t + 1}"] = v
    return df


def main():
    panel = make_panel()
    panel.to_csv(os.path.join(HERE, "SEM_Longitudinal_Panel.csv"), index=False, encoding="utf-8-sig")
    items = make_items()
    items.to_csv(os.path.join(HERE, "SEM_Longitudinal_Items.csv"), index=False, encoding="utf-8-sig")
    print(f"SEM_Longitudinal_Panel.csv: {panel.shape}, SEM_Longitudinal_Items.csv: {items.shape}")


if __name__ == "__main__":
    main()
