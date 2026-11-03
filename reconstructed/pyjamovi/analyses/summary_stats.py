"""
pyjamovi Summary Statistics Analysis Module (JASP Summary Stats Compatible)
Calculates:
- Two-Sample Independent T-Test from Group Means, SDs, and Sample Sizes (m1, s1, n1, m2, s2, n2)
- Welch's T-Test (Unequal Variances)
- Effect Size (Cohen's d, Hedges' g) and 95% Confidence Interval
- JASP / JZS Bayes Factor (BF10, BF01) and Evidence Interpretation
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseAnalysis, ResultsTable
from .bayesian import jzs_bayes_factor_ttest, _interpret_bf10


class SummaryTTest(BaseAnalysis):
    """
    Computes Independent Samples T-Test & Bayes Factor from summary statistics alone
    (without requiring a raw microdata dataset).
    """
    def __init__(
        self,
        m1: float,
        s1: float,
        n1: int,
        m2: float,
        s2: float,
        n2: int,
        group1_name: str = "Group 1",
        group2_name: str = "Group 2",
        hypothesis: str = "different",
        prior_width: float = 0.707
    ):
        dummy_df = pd.DataFrame({"dummy": [0]})
        super().__init__(
            name="summaryTTest",
            title="Summary Statistics Independent Samples T-Test (JASP)",
            dataset=dummy_df,
            options={
                "m1": float(m1), "s1": float(s1), "n1": int(n1),
                "m2": float(m2), "s2": float(s2), "n2": int(n2),
                "group1_name": group1_name, "group2_name": group2_name,
                "hypothesis": hypothesis, "prior_width": float(prior_width)
            }
        )
        self.m1 = float(m1)
        self.s1 = float(s1)
        self.n1 = int(n1)
        self.m2 = float(m2)
        self.s2 = float(s2)
        self.n2 = int(n2)
        self.g1_name = group1_name
        self.g2_name = group2_name
        self.prior_width = float(prior_width)
        self.syntax_python = (
            f"jmv.summaryTTest(m1={self.m1}, s1={self.s1}, n1={self.n1}, "
            f"m2={self.m2}, s2={self.s2}, n2={self.n2}, prior_width={self.prior_width})"
        )
        self.syntax_r = (
            f"jaspSummaryStats::summaryTTest(m1={self.m1}, s1={self.s1}, n1={self.n1}, "
            f"m2={self.m2}, s2={self.s2}, n2={self.n2})"
        )
        self.run()

    def run(self):
        m1, s1, n1 = self.m1, self.s1, self.n1
        m2, s2, n2 = self.m2, self.s2, self.n2

        if n1 < 2 or n2 < 2:
            raise ValueError(f"Sample sizes must be >= 2 (got n1={n1}, n2={n2})")
        if s1 <= 0 or s2 <= 0:
            raise ValueError(f"Standard deviations must be > 0 (got s1={s1}, s2={s2})")

        # 1. Student's T-Test (Pooled variance)
        df_stud = n1 + n2 - 2
        sp2 = (((n1 - 1) * (s1 ** 2)) + ((n2 - 1) * (s2 ** 2))) / df_stud
        sp = np.sqrt(sp2)
        se_stud = sp * np.sqrt((1.0 / n1) + (1.0 / n2))
        diff = m1 - m2
        t_stud = diff / se_stud
        p_stud = 2.0 * (1.0 - stats.t.cdf(abs(t_stud), df_stud))
        cohen_d = diff / sp

        # 2. Welch's T-Test (Unequal variance)
        v1 = (s1 ** 2) / n1
        v2 = (s2 ** 2) / n2
        se_welch = np.sqrt(v1 + v2)
        t_welch = diff / se_welch
        df_welch = ((v1 + v2) ** 2) / (((v1 ** 2) / (n1 - 1)) + ((v2 ** 2) / (n2 - 1)))
        p_welch = 2.0 * (1.0 - stats.t.cdf(abs(t_welch), df_welch))

        # 3. Bayes Factor
        bf10 = jzs_bayes_factor_ttest(t_stud, n1, n2, r=self.prior_width)
        bf01 = 1.0 / max(bf10, 1e-12)
        ev_text = _interpret_bf10(bf10)

        # 4. CI of difference
        crit = stats.t.ppf(0.975, df_stud)
        ci_lower = diff - crit * se_stud
        ci_upper = diff + crit * se_stud

        # Result Table 1: T-Test results
        tbl = ResultsTable(name="summary_ttest", title="Independent Samples T-Test (from Summary Statistics)")
        tbl.add_column("test", "Test", col_type="text")
        tbl.add_column("diff", "Mean Difference", col_type="number")
        tbl.add_column("t", "t", col_type="number")
        tbl.add_column("df", "df", col_type="number")
        tbl.add_column("p", "p", col_type="number", fmt="pval")
        tbl.add_column("d", "Cohen's d", col_type="number")
        tbl.add_column("bf10", "BF₁₀", col_type="number")
        tbl.add_column("evidence", "Evidence (Jeffreys)", col_type="text")

        tbl.add_row({
            "test": "Student's t",
            "diff": diff, "t": t_stud, "df": float(df_stud),
            "p": p_stud, "d": cohen_d, "bf10": bf10, "evidence": ev_text
        })
        tbl.add_row({
            "test": "Welch's t",
            "diff": diff, "t": t_welch, "df": float(df_welch),
            "p": p_welch, "d": diff / np.sqrt((s1**2 + s2**2)/2.0),
            "bf10": jzs_bayes_factor_ttest(t_welch, n1, n2, r=self.prior_width),
            "evidence": ev_text
        })
        tbl.add_note(f"95% CI of Difference: [{ci_lower:.3f}, {ci_upper:.3f}]. Cauchy prior width r={self.prior_width}.")
        self.add_table(tbl)

        # Result Table 2: Input group parameters
        desc_tbl = ResultsTable(name="group_summary", title="Group Summary Descriptives")
        desc_tbl.add_column("group", "Group", col_type="text")
        desc_tbl.add_column("n", "N", col_type="integer")
        desc_tbl.add_column("mean", "Mean", col_type="number")
        desc_tbl.add_column("sd", "SD", col_type="number")
        desc_tbl.add_column("se", "SE", col_type="number")

        desc_tbl.add_row({"group": self.g1_name, "n": n1, "mean": m1, "sd": s1, "se": s1 / np.sqrt(n1)})
        desc_tbl.add_row({"group": self.g2_name, "n": n2, "mean": m2, "sd": s2, "se": s2 / np.sqrt(n2)})
        self.add_table(desc_tbl)

        self.apa_narrative = (
            f"An independent-samples t-test from summary statistics revealed a difference between {self.g1_name} "
            f"(M = {m1:.2f}, SD = {s1:.2f}, N = {n1}) and {self.g2_name} (M = {m2:.2f}, SD = {s2:.2f}, N = {n2}), "
            f"t({df_stud}) = {t_stud:.3f}, p = {p_stud:.4f}, Cohen's d = {cohen_d:.3f}. "
            f"Bayesian evaluation yielded BF₁₀ = {bf10:.3f}, indicating {ev_text}."
        )
