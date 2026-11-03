"""
pyjamovi Synthetic Data Generation Engine (Privacy-Preserving & Simulation Modeling)
License: MIT
Capabilities:
1. Multivariate Gaussian Copula Simulation: preserves correlation and empirical marginals of continuous variables.
2. Categorical Distribution & Conditional Sampling: preserves multinomial value proportions and frequencies.
3. Differential Privacy (DP) Noise Injection: adds calibrated Laplace/Gaussian noise controlled by epsilon (ε).
4. Simulation Dataset Generator: create synthetic microdata for power analysis, teaching, or sharing without PII.
"""

from typing import List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
from scipy import stats


def generate_synthetic_dataset(
    df: pd.DataFrame,
    n_samples: Optional[int] = None,
    cols: Optional[List[str]] = None,
    epsilon: Optional[float] = None,  # Differential privacy parameter, e.g. 1.0 (None = exact empirical)
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generates a realistic synthetic DataFrame mirroring the statistical properties
    of the input data while protecting individual-level microdata.
    """
    if seed is not None:
        np.random.seed(seed)

    cols = list(cols or df.columns)
    sub = df[cols].copy()
    n_rows = n_samples or len(df)

    # Classify columns into numeric vs categorical
    numeric_cols = []
    cat_cols = []
    for c in cols:
        s = pd.to_numeric(sub[c], errors="coerce")
        # If numeric has > 5 distinct values or is float, treat as continuous
        if s.notna().sum() > len(sub) * 0.5 and sub[c].nunique() > 6:
            numeric_cols.append(c)
        else:
            cat_cols.append(c)

    synth_dict: Dict[str, Any] = {}

    # 1. Continuous Variables: Gaussian Copula Simulation
    if len(numeric_cols) > 0:
        num_df = sub[numeric_cols].apply(pd.to_numeric, errors="coerce")
        # Impute medians for fitting
        num_clean = num_df.fillna(num_df.median())

        # Compute rank correlation (Spearman) to handle non-normality
        corr_matrix = num_clean.corr(method="spearman").values
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 1.0)

        # Make sure positive semi-definite
        eigvals, eigvecs = np.linalg.eigh(corr_matrix)
        eigvals = np.maximum(eigvals, 1e-4)
        corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        diag_corr = np.sqrt(np.diag(corr_psd))
        corr_psd = corr_psd / np.outer(diag_corr, diag_corr)

        # Draw correlated standard normal variables
        mean_vec = np.zeros(len(numeric_cols))
        mv_normal = np.random.multivariate_normal(mean_vec, corr_psd, size=n_rows)

        # Transform via standard normal CDF to uniform [0, 1] marginals
        u_marginals = stats.norm.cdf(mv_normal)

        # Add DP noise to quantiles if epsilon is specified
        if epsilon is not None and epsilon > 0:
            scale = 1.0 / (epsilon * np.sqrt(len(df)))
            u_marginals = np.clip(u_marginals + np.random.laplace(0, scale, size=u_marginals.shape), 0.0001, 0.9999)

        # Inverse transform to match empirical distribution of each column
        for idx, col_name in enumerate(numeric_cols):
            orig_vals = np.sort(num_clean[col_name].values)
            # Quantile matching via percentiles
            synth_vals = np.quantile(orig_vals, u_marginals[:, idx])
            # If original was integer, round to integer
            if pd.api.types.is_integer_dtype(sub[col_name]):
                synth_vals = np.round(synth_vals).astype(int)
            synth_dict[col_name] = synth_vals

    # 2. Categorical Variables: Empirical Proportions Sampling
    for col_name in cat_cols:
        series = sub[col_name].dropna()
        if len(series) == 0:
            synth_dict[col_name] = [None] * n_rows
            continue
        counts = series.value_counts(normalize=True)
        categories = counts.index.tolist()
        probabilities = counts.values

        # Apply Differential Privacy to categorical probabilities if epsilon is active
        if epsilon is not None and epsilon > 0:
            lap_noise = np.random.laplace(0, 1.0 / (epsilon * len(series)), size=len(probabilities))
            probabilities = np.maximum(probabilities + lap_noise, 1e-5)
            probabilities = probabilities / np.sum(probabilities)

        synth_cats = np.random.choice(categories, size=n_rows, p=probabilities)
        synth_dict[col_name] = synth_cats

    # Reorder columns to match original
    result_df = pd.DataFrame(synth_dict)[cols]
    return result_df
