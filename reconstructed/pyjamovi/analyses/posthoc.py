"""
pyjamovi Comprehensive Post-Hoc Multiple Comparisons Module
Supports Tukey HSD, Games-Howell, Scheffe, Bonferroni, and Holm.
"""

from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg

from .base import ResultsTable


def compute_posthoc(df: pd.DataFrame, dep: str, group: str, method: str = "tukey") -> ResultsTable:
    clean = df[[dep, group]].dropna().copy()
    clean[dep] = pd.to_numeric(clean[dep], errors="coerce")
    clean = clean.dropna()
    clean[group] = clean[group].astype(str)

    groups = sorted(clean[group].unique())
    k = len(groups)
    n_total = len(clean)
    method_low = (method or "tukey").lower()

    title_map = {
        "tukey": "Tukey HSD",
        "gameshowell": "Games-Howell (Unequal Variances)",
        "games-howell": "Games-Howell (Unequal Variances)",
        "scheffe": "Scheffé",
        "bonferroni": "Bonferroni",
        "holm": "Holm"
    }
    nice_name = title_map.get(method_low, method)

    table = ResultsTable(name=f"posthoc_{method_low}", title=f"Post-Hoc Comparisons — {group} ({nice_name})")
    table.add_column("g1", f"{group} (I)", col_type="text")
    table.add_column("g2", f"{group} (J)", col_type="text")
    table.add_column("md", "Mean Difference", col_type="number")
    table.add_column("se", "SE", col_type="number")
    table.add_column("df", "df", col_type="number")
    table.add_column("t", "t", col_type="number")
    table.add_column("p", f"p ({nice_name})", col_type="number", fmt="pvalue")
    table.add_column("d", "Cohen's d", col_type="number")

    if k < 2:
        return table

    if method_low in ("gameshowell", "games-howell"):
        gh = pg.pairwise_gameshowell(data=clean, dv=dep, between=group)
        for _, r in gh.iterrows():
            table.add_row({
                "g1": str(r["A"]),
                "g2": str(r["B"]),
                "md": float(r["diff"]),
                "se": float(r["se"]),
                "df": float(r["df"]),
                "t": float(r["T"]),
                "p": float(r["pval"]),
                "d": float(r["hedges"]) if "hedges" in r else np.nan
            })
        return table

    # Pooled MSE for classical post-hoc tests (Tukey, Scheffe, Bonferroni, Holm)
    df_within = n_total - k
    ss_within = sum(((clean[clean[group] == g][dep] - clean[clean[group] == g][dep].mean()) ** 2).sum() for g in groups)
    mse = ss_within / max(1, df_within)
    sd_pooled = np.sqrt(mse)

    raw_pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            g1, g2 = groups[i], groups[j]
            s1 = clean[clean[group] == g1][dep].values
            s2 = clean[clean[group] == g2][dep].values
            n1, n2 = len(s1), len(s2)
            md = float(np.mean(s1) - np.mean(s2))
            se = float(np.sqrt(mse * (1.0 / n1 + 1.0 / n2)))
            t_val = md / se if se > 0 else 0.0
            p_unadj = float(2.0 * (1.0 - stats.t.cdf(abs(t_val), df_within)))
            cohen_d = md / sd_pooled if sd_pooled > 0 else np.nan
            raw_pairs.append({
                "g1": g1, "g2": g2, "md": md, "se": se,
                "df": float(df_within), "t": t_val, "p_raw": p_unadj, "d": cohen_d
            })

    m_pairs = len(raw_pairs)

    if method_low == "tukey":
        for item in raw_pairs:
            q_stat = abs(item["t"]) * np.sqrt(2.0)
            p_tukey = float(1.0 - stats.studentized_range.cdf(q_stat, k, df_within))
            item["p"] = max(0.0, min(1.0, p_tukey))
    elif method_low == "scheffe":
        for item in raw_pairs:
            f_scheffe = (item["t"] ** 2) / max(1, k - 1)
            p_sch = float(1.0 - stats.f.cdf(f_scheffe, k - 1, df_within))
            item["p"] = max(0.0, min(1.0, p_sch))
    elif method_low == "bonferroni":
        for item in raw_pairs:
            item["p"] = min(1.0, item["p_raw"] * m_pairs)
    elif method_low == "holm":
        order = np.argsort([x["p_raw"] for x in raw_pairs])
        running_max = 0.0
        for rank, idx in enumerate(order):
            adj = min(1.0, raw_pairs[idx]["p_raw"] * (m_pairs - rank))
            running_max = max(running_max, adj)
            raw_pairs[idx]["p"] = running_max
    else:
        for item in raw_pairs:
            item["p"] = item["p_raw"]

    for item in raw_pairs:
        table.add_row({
            "g1": item["g1"],
            "g2": item["g2"],
            "md": item["md"],
            "se": item["se"],
            "df": item["df"],
            "t": item["t"],
            "p": item["p"],
            "d": item["d"]
        })

    table.add_note(f"Comparisons based on pooled error term (df = {df_within}), adjusted via {nice_name}.")
    return table


def compute_games_howell(df: pd.DataFrame, dep: str, group: str) -> ResultsTable:
    return compute_posthoc(df, dep, group, method="gamesHowell")


def compute_bonferroni(df: pd.DataFrame, dep: str, group: str) -> ResultsTable:
    return compute_posthoc(df, dep, group, method="bonferroni")
