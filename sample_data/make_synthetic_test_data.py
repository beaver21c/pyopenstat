"""
Generates arbitrary (synthetic) operational-test datasets for OpenStat Studio (seed fixed => reproducible).
1. synthetic_clinical_wide.csv : N=120 subjects, 3 arms, repeated BP / QoL at waves 1-3 (Wide format)
2. synthetic_site_lookup.csv   : subject-level site info for Merge testing (m:1 / 1:1 on subj_id)
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20261104)

N = 120
subj = np.arange(1001, 1001 + N)
arm = np.repeat(["Control", "LowDose", "HighDose"], N // 3)
rng.shuffle(arm)
sex = rng.choice(["F", "M"], size=N)
age = rng.integers(22, 79, size=N)
height = np.round(rng.normal(166, 8.5, size=N), 1)
weight = np.round(height - 100 + rng.normal(0, 9, size=N), 1)

effect = pd.Series(arm).map({"Control": 0.0, "LowDose": -5.0, "HighDose": -11.0}).to_numpy()
bp_w1 = np.round(rng.normal(148, 10, size=N) + 0.15 * (age - 50), 1)
bp_w2 = np.round(bp_w1 + effect * 0.5 + rng.normal(0, 4, size=N), 1)
bp_w3 = np.round(bp_w1 + effect + rng.normal(0, 4, size=N), 1)
qol_w1 = np.round(rng.normal(55, 8, size=N), 1)
qol_w2 = np.round(qol_w1 - effect * 0.3 + rng.normal(0, 3, size=N), 1)
qol_w3 = np.round(qol_w1 - effect * 0.6 + rng.normal(0, 3, size=N), 1)

# A few missing values to exercise the Variable Inspector "Missing" column
qol_w3[rng.choice(N, 4, replace=False)] = np.nan

wide = pd.DataFrame({
    "subj_id": subj, "arm": arm, "sex": sex, "age": age,
    "height_cm": height, "weight_kg": weight,
    "bp_w1": bp_w1, "bp_w2": bp_w2, "bp_w3": bp_w3,
    "qol_w1": qol_w1, "qol_w2": qol_w2, "qol_w3": qol_w3,
})
wide.to_csv(os.path.join(HERE, "synthetic_clinical_wide.csv"), index=False)

lookup = pd.DataFrame({
    "subj_id": subj,
    "region": rng.choice(["Seoul", "Busan", "Daegu", "Gwangju"], size=N),
    "hospital_tier": rng.choice(["Tertiary", "General", "Clinic"], size=N, p=[0.3, 0.45, 0.25]),
    "baseline_risk": np.round(rng.uniform(0.05, 0.6, size=N), 3),
})
lookup.to_csv(os.path.join(HERE, "synthetic_site_lookup.csv"), index=False)
print(f"wide={wide.shape}, lookup={lookup.shape}")
