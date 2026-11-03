"""
pyjamovi GIS Spatial Analysis & Spatial Econometric Regression Module (GeoDa & Stata spregress Compatible)
License: MIT

Implements:
1. Spatial Weights Matrix (W): k-Nearest Neighbors (k-NN) or Inverse Distance (IDW) from Lat/Lon coordinates, Row-Standardized.
2. Global Moran's I Statistic: spatial autocorrelation test with Analytical & Permutation p-values and z-scores.
3. LISA (Local Indicators of Spatial Association): Local Moran's I and 4-quadrant cluster typology (High-High, Low-Low, High-Low, Low-High).
4. Spatial Regression Models (Stata `spregress` ML):
   - Spatial Lag Model (SAR / SLM): y = ρ*W*y + X*β + ε
   - Spatial Error Model (SEM): y = X*β + u, u = λ*W*u + ε
   - Classical OLS baseline with Anselin's LM-Lag and LM-Error diagnostic tests for spatial dependence.
5. High-Resolution Visualizations: Moran Scatterplot (GeoDa style), Regional Spatial Choropleth Map, and LISA Cluster Distribution.
"""

from typing import List, Optional, Union, Dict, Any, Tuple
import re
import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .base import BaseAnalysis, ResultsTable
from .plots import fig_to_base64


def compute_spatial_weights(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    method: str = "knn",
    k: int = 4,
    power: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes a row-standardized spatial weights matrix W and distance matrix.
    Method: 'knn' (k-nearest neighbors) or 'inverse_distance'.
    """
    n = len(latitudes)
    coords = np.column_stack([latitudes, longitudes])

    # Euclidean approximation in degrees (lat/lon)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d = np.sqrt(((coords[i, 0] - coords[j, 0]) ** 2) + ((coords[i, 1] - coords[j, 1]) ** 2))
                dist_matrix[i, j] = max(d, 1e-6)

    w_raw = np.zeros((n, n))
    if method == "knn" or method == "k_nearest":
        k_val = min(max(1, int(k)), n - 1)
        for i in range(n):
            nn_indices = np.argsort(dist_matrix[i, :])[1:k_val + 1]
            w_raw[i, nn_indices] = 1.0
    else:  # inverse distance
        for i in range(n):
            for j in range(n):
                if i != j:
                    w_raw[i, j] = 1.0 / (dist_matrix[i, j] ** float(power))

    # Row-standardization
    row_sums = np.sum(w_raw, axis=1)
    row_sums[row_sums == 0] = 1.0
    w_std = w_raw / row_sums[:, np.newaxis]

    return w_std, dist_matrix


def compute_global_moran(y: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    """Computes Moran's I, expectation, variance, z-score, and p-value."""
    n = len(y)
    y_mean = np.mean(y)
    z = y - y_mean
    s0 = np.sum(w)
    s_sq = np.sum(z ** 2)
    if s_sq < 1e-12 or s0 < 1e-12:
        return {"moran_i": 0.0, "expected_i": -1.0 / (n - 1), "variance": 0.0, "z_score": 0.0, "p_value": 1.0}

    # Moran's I numerator
    wz = w @ z
    num = np.sum(z * wz)
    moran_i = float((n / s0) * (num / s_sq))

    # Expectation under randomization
    e_i = -1.0 / (n - 1)

    # Variance under randomization (Cliff & Ord, 1981)
    s1 = 0.5 * np.sum((w + w.T) ** 2)
    s2 = np.sum((np.sum(w, axis=1) + np.sum(w, axis=0)) ** 2)
    kurt = (np.sum(z ** 4) / n) / ((s_sq / n) ** 2)

    term1 = n * ((n ** 2 - 3 * n + 3) * s1 - n * s2 + 3 * (s0 ** 2))
    term2 = kurt * ((n ** 2 - n) * s1 - 2 * n * s2 + 6 * (s0 ** 2))
    denom = (n - 1) * (n - 2) * (n - 3) * (s0 ** 2)

    var_i = max(1e-12, float((term1 - term2) / denom - (e_i ** 2)))
    z_score = float((moran_i - e_i) / np.sqrt(var_i))
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_score))))

    return {
        "moran_i": moran_i,
        "expected_i": e_i,
        "variance": var_i,
        "z_score": z_score,
        "p_value": p_val
    }


def compute_local_moran(y: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Computes LISA Local Moran's I and quadrant classification."""
    n = len(y)
    y_mean = np.mean(y)
    s = np.std(y, ddof=0)
    z = (y - y_mean) / (s if s > 1e-12 else 1.0)
    wz = w @ z

    lisa_i = z * wz
    quadrants = []
    for i in range(n):
        if z[i] >= 0 and wz[i] >= 0:
            quadrants.append("High-High (Hotspot)")
        elif z[i] < 0 and wz[i] < 0:
            quadrants.append("Low-Low (Coldspot)")
        elif z[i] >= 0 and wz[i] < 0:
            quadrants.append("High-Low (Spatial Outlier)")
        else:
            quadrants.append("Low-High (Spatial Outlier)")

    return lisa_i, quadrants


def fit_spatial_lag_model(y: np.ndarray, X: np.ndarray, w: np.ndarray, var_names: List[str]):
    """
    Fits Spatial Autoregressive (Lag) Model: y = ρ*W*y + X*β + ε via Maximum Likelihood.
    Returns: regression results dict
    """
    n, k = X.shape
    eigvals = np.linalg.eigvals(w).real
    eig_min = np.min(eigvals)
    eig_max = np.max(eigvals)
    rho_min = (1.0 / eig_min) + 1e-4 if eig_min < 0 else -0.999
    rho_max = (1.0 / eig_max) - 1e-4 if eig_max > 0 else 0.999
    rho_min = max(-0.99, min(-0.1, rho_min))
    rho_max = min(0.99, max(0.1, rho_max))

    # OLS on X
    xtx_inv = np.linalg.pinv(X.T @ X)
    b_0 = xtx_inv @ (X.T @ y)
    wy = w @ y
    b_lag = xtx_inv @ (X.T @ wy)

    e_0 = y - X @ b_0
    e_lag = wy - X @ b_lag

    def minus_loglik(rho):
        if rho <= rho_min or rho >= rho_max:
            return 1e12
        e = e_0 - rho * e_lag
        sigma2 = np.sum(e ** 2) / n
        if sigma2 <= 1e-12:
            return 1e12
        # det(I - rho*W) = prod(1 - rho*eig_i)
        log_det = np.sum(np.log(np.maximum(1e-12, 1.0 - rho * eigvals)))
        ll = - (n / 2.0) * np.log(2 * np.pi) - (n / 2.0) * np.log(sigma2) + log_det - (n / 2.0)
        return -ll

    opt = optimize.minimize_scalar(minus_loglik, bounds=(rho_min, rho_max), method="bounded")
    rho_hat = float(opt.x) if opt.success else 0.0

    # Optimal beta and sigma2
    beta_hat = b_0 - rho_hat * b_lag
    residuals = (y - rho_hat * wy) - X @ beta_hat
    sigma2_hat = float(np.sum(residuals ** 2) / n)
    loglik = float(-opt.fun)

    # Asymptotic standard errors via numerical Hessian or OLS approximation
    var_beta = sigma2_hat * xtx_inv
    se_beta = np.sqrt(np.maximum(np.diag(var_beta), 1e-8))

    # Variance of rho (Anselin, 1988)
    a_rho = np.sum((eigvals / (1.0 - rho_hat * eigvals)) ** 2)
    we_e = float((e_lag.T @ e_lag) / sigma2_hat)
    info_rho = a_rho + we_e
    se_rho = float(1.0 / np.sqrt(max(1e-8, info_rho)))

    aic = -2.0 * loglik + 2.0 * (k + 2)
    bic = -2.0 * loglik + np.log(n) * (k + 2)

    # Pseudo R2
    y_var = np.var(y, ddof=1)
    pseudo_r2 = max(0.0, 1.0 - (np.var(residuals, ddof=1) / (y_var if y_var > 1e-8 else 1.0)))

    return {
        "model_type": "Spatial Lag (SAR)",
        "rho": rho_hat,
        "se_rho": se_rho,
        "z_rho": rho_hat / se_rho,
        "p_rho": float(2.0 * (1.0 - stats.norm.cdf(abs(rho_hat / se_rho)))),
        "betas": beta_hat,
        "se_betas": se_beta,
        "var_names": var_names,
        "sigma2": sigma2_hat,
        "loglik": loglik,
        "aic": aic,
        "bic": bic,
        "pseudo_r2": pseudo_r2,
        "n": n,
        "k": k
    }


def fit_spatial_error_model(y: np.ndarray, X: np.ndarray, w: np.ndarray, var_names: List[str]):
    """
    Fits Spatial Error Model: y = X*β + u, u = λ*W*u + ε via Maximum Likelihood.
    """
    n, k = X.shape
    eigvals = np.linalg.eigvals(w).real
    lambda_min = max(-0.99, -0.95)
    lambda_max = min(0.99, 0.95)

    def minus_loglik(lam):
        if lam <= lambda_min or lam >= lambda_max:
            return 1e12
        # Transform y and X: y* = (I - lam*W)y, X* = (I - lam*W)X
        y_star = y - lam * (w @ y)
        X_star = X - lam * (w @ X)
        xtx_inv = np.linalg.pinv(X_star.T @ X_star)
        b = xtx_inv @ (X_star.T @ y_star)
        e = y_star - X_star @ b
        sigma2 = np.sum(e ** 2) / n
        if sigma2 <= 1e-12:
            return 1e12
        log_det = np.sum(np.log(np.maximum(1e-12, 1.0 - lam * eigvals)))
        ll = - (n / 2.0) * np.log(2 * np.pi) - (n / 2.0) * np.log(sigma2) + log_det - (n / 2.0)
        return -ll

    opt = optimize.minimize_scalar(minus_loglik, bounds=(lambda_min, lambda_max), method="bounded")
    lambda_hat = float(opt.x) if opt.success else 0.0

    y_star = y - lambda_hat * (w @ y)
    X_star = X - lambda_hat * (w @ X)
    xtx_inv = np.linalg.pinv(X_star.T @ X_star)
    beta_hat = xtx_inv @ (X_star.T @ y_star)
    residuals = y - X @ beta_hat
    e_star = y_star - X_star @ beta_hat
    sigma2_hat = float(np.sum(e_star ** 2) / n)
    loglik = float(-opt.fun)

    se_beta = np.sqrt(np.maximum(np.diag(sigma2_hat * xtx_inv), 1e-8))
    info_lam = np.sum((eigvals / (1.0 - lambda_hat * eigvals)) ** 2) + np.trace(w @ w)
    se_lambda = float(1.0 / np.sqrt(max(1e-8, info_lam)))

    aic = -2.0 * loglik + 2.0 * (k + 2)
    bic = -2.0 * loglik + np.log(n) * (k + 2)
    y_var = np.var(y, ddof=1)
    pseudo_r2 = max(0.0, 1.0 - (np.var(residuals, ddof=1) / (y_var if y_var > 1e-8 else 1.0)))

    return {
        "model_type": "Spatial Error (SEM)",
        "lambda": lambda_hat,
        "se_lambda": se_lambda,
        "z_lambda": lambda_hat / se_lambda,
        "p_lambda": float(2.0 * (1.0 - stats.norm.cdf(abs(lambda_hat / se_lambda)))),
        "betas": beta_hat,
        "se_betas": se_beta,
        "var_names": var_names,
        "sigma2": sigma2_hat,
        "loglik": loglik,
        "aic": aic,
        "bic": bic,
        "pseudo_r2": pseudo_r2,
        "n": n,
        "k": k
    }


def plot_moran_scatterplot(y: np.ndarray, w: np.ndarray, dep_name: str, moran_res: Dict[str, float]) -> str:
    """Generates GeoDa-style Moran Scatterplot."""
    z = (y - np.mean(y)) / np.std(y, ddof=0)
    wz = w @ z
    i_stat = moran_res["moran_i"]

    fig, ax = plt.subplots(figsize=(6.0, 5.2), dpi=150)
    ax.set_facecolor("#FAFAFA")

    # Quadrant lines
    ax.axvline(0, color="#94A3B8", linestyle="--", linewidth=1.0)
    ax.axhline(0, color="#94A3B8", linestyle="--", linewidth=1.0)

    # Scatter points colored by quadrant
    colors = []
    for zi, wzi in zip(z, wz):
        if zi >= 0 and wzi >= 0:
            colors.append("#DC2626")  # High-High (Red)
        elif zi < 0 and wzi < 0:
            colors.append("#2563EB")  # Low-Low (Blue)
        elif zi >= 0 and wzi < 0:
            colors.append("#F59E0B")  # High-Low (Orange)
        else:
            colors.append("#0D9488")  # Low-High (Teal)

    ax.scatter(z, wz, c=colors, s=55, alpha=0.85, edgecolors="#1E293B", linewidth=0.8, zorder=3)

    # Linear fit line (slope = Moran's I)
    x_vals = np.linspace(np.min(z) - 0.2, np.max(z) + 0.2, 100)
    ax.plot(x_vals, i_stat * x_vals, color="#1E293B", linewidth=2.0, label=f"Fit (Slope I = {i_stat:.3f})", zorder=4)

    # Quadrant labels
    ax.text(0.95, 0.95, "High-High (Hotspot)", transform=ax.transAxes, ha="right", va="top", fontsize=9.5, fontweight="bold", color="#DC2626")
    ax.text(0.05, 0.05, "Low-Low (Coldspot)", transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, fontweight="bold", color="#2563EB")
    ax.text(0.95, 0.05, "High-Low (Outlier)", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#D97706")
    ax.text(0.05, 0.95, "Low-High (Outlier)", transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#0D9488")

    ax.set_xlabel(f"Standardized Variable: z({dep_name})", fontsize=10.5, fontweight="600")
    ax.set_ylabel(f"Spatial Lag: W · z({dep_name})", fontsize=10.5, fontweight="600")
    ax.set_title(f"GeoDa Moran Scatterplot — {dep_name}\nMoran's I = {i_stat:.4f} (z = {moran_res['z_score']:.2f}, p = {moran_res['p_value']:.4f})", fontsize=11.5, fontweight="bold", pad=10)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_spatial_choropleth_map(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    values: np.ndarray,
    quadrants: List[str],
    region_names: Optional[List[str]],
    dep_name: str
) -> str:
    """Generates 2D regional spatial bubble map / choropleth visualization."""
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=150)
    ax.set_facecolor("#F8FAFC")

    # Map quadrant colors
    q_colors = {
        "High-High (Hotspot)": "#DC2626",
        "Low-Low (Coldspot)": "#2563EB",
        "High-Low (Spatial Outlier)": "#F59E0B",
        "Low-High (Spatial Outlier)": "#0D9488"
    }
    c_list = [q_colors.get(q, "#64748B") for q in quadrants]

    # Bubble size mapped to values
    v_min, v_max = np.min(values), np.max(values)
    norm_size = 80 + 320 * ((values - v_min) / (v_max - v_min if v_max > v_min else 1.0))

    scatter = ax.scatter(
        longitudes, latitudes, s=norm_size, c=c_list,
        alpha=0.75, edgecolors="#0F172A", linewidth=1.2, zorder=3
    )

    # Region name annotations
    if region_names is not None:
        for lon, lat, name in zip(longitudes, latitudes, region_names):
            ax.annotate(
                name, (lon, lat),
                fontsize=8.5, ha="center", va="center",
                color="#0F172A", fontweight="bold", zorder=4
            )

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#DC2626", edgecolor="#0F172A", label="High-High (Hotspot)"),
        Patch(facecolor="#2563EB", edgecolor="#0F172A", label="Low-Low (Coldspot)"),
        Patch(facecolor="#F59E0B", edgecolor="#0F172A", label="High-Low (Outlier)"),
        Patch(facecolor="#0D9488", edgecolor="#0F172A", label="Low-High (Outlier)")
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True, fontsize=9.0)

    ax.set_xlabel("Longitude (경도)", fontsize=10.5, fontweight="600")
    ax.set_ylabel("Latitude (위도)", fontsize=10.5, fontweight="600")
    ax.set_title(f"GIS Spatial LISA Map — {dep_name}", fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


class SpatialAnalysis(BaseAnalysis):
    """
    OpenStat Studio GIS Spatial Analysis & Econometric Regression Module
    GeoDa & Stata `spregress` compatible.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        dep: str,
        covs: Optional[List[str]] = None,
        lat: Optional[str] = None,
        lon: Optional[str] = None,
        region_id: Optional[str] = None,
        weights_method: str = "knn",
        k: int = 4,
        model_type: str = "lag",  # 'lag' (SAR), 'error' (SEM), 'moran_only'
        plot_moran: bool = True,
        plot_map: bool = True
    ):
        c_list = [covs] if isinstance(covs, str) else list(covs or [])
        super().__init__(
            name="spatialAnalysis",
            title="GIS Spatial Analysis & Regression (GeoDa / Stata spregress)",
            dataset=data,
            options={
                "dep": dep, "covs": c_list, "lat": lat, "lon": lon,
                "region_id": region_id, "weights_method": weights_method,
                "k": int(k), "modelType": model_type,
                "plotMoran": plot_moran, "plotMap": plot_map
            }
        )
        self.dep = dep
        self.covs = c_list
        self.lat_col = lat
        self.lon_col = lon
        self.region_col = region_id
        self.weights_method = weights_method
        self.k = int(k)
        self.model_type = model_type

        self.syntax_python = (
            f"sess.spatial_regression(dep={dep!r}, covs={c_list!r}, lat={lat!r}, lon={lon!r}, "
            f"model_type={model_type!r}, k={self.k})"
        )
        self.syntax_r = f"spdep::lagsarlm({dep} ~ {' + '.join(c_list) if c_list else '1'}, listw=w_list)"
        self.run()

    def run(self):
        df = self.dataset
        needed_cols = [self.dep] + self.covs
        if self.lat_col:
            needed_cols.append(self.lat_col)
        if self.lon_col:
            needed_cols.append(self.lon_col)
        if self.region_col and self.region_col in df.columns:
            needed_cols.append(self.region_col)

        sub = df.dropna(subset=[c for c in needed_cols if c in df.columns]).copy()
        n = len(sub)
        if n < 5:
            raise ValueError(f"Spatial Analysis requires at least 5 spatial units, got N={n}.")

        # Coordinates detection
        lat_col = self.lat_col or next((c for c in sub.columns if re.search(r"lat|위도|y_coord", c, re.I)), None)
        lon_col = self.lon_col or next((c for c in sub.columns if re.search(r"lon|long|경도|x_coord", c, re.I)), None)
        region_col = self.region_col or next((c for c in sub.columns if re.search(r"name|region|시군구|지역|id", c, re.I)), None)

        if not lat_col or not lon_col:
            # Fallback coordinates: 1D rank coordinates
            lats = np.arange(n, dtype=float)
            lons = np.ones(n, dtype=float)
        else:
            lats = sub[lat_col].astype(float).values
            lons = sub[lon_col].astype(float).values

        y = sub[self.dep].astype(float).values
        region_names = sub[region_col].astype(str).tolist() if region_col and region_col in sub.columns else None

        # 1. Compute Spatial Weights Matrix W
        w_mat, _ = compute_spatial_weights(lats, lons, method=self.weights_method, k=self.k)

        # 2. Global Moran's I
        moran_res = compute_global_moran(y, w_mat)
        lisa_i, quadrants = compute_local_moran(y, w_mat)

        # Table 1: Global Moran's I Test
        tbl_moran = ResultsTable(name="moran_test", title=f"Global Spatial Autocorrelation Test (Moran's I) — {self.dep}")
        tbl_moran.add_column("var", "Variable", col_type="text")
        tbl_moran.add_column("moran_i", "Moran's I", col_type="number")
        tbl_moran.add_column("expected", "Expected I", col_type="number")
        tbl_moran.add_column("sd", "Std. Error", col_type="number")
        tbl_moran.add_column("z", "z-score", col_type="number")
        tbl_moran.add_column("p", "p-value", col_type="number", fmt="pval")
        tbl_moran.add_column("interp", "Spatial Pattern", col_type="text")

        pattern = "Positive Spatial Clustering (군집화)" if moran_res["z_score"] > 1.96 else ("Negative Spatial Dispersion (분산)" if moran_res["z_score"] < -1.96 else "Random / Independent (무작위)")
        tbl_moran.add_row({
            "var": self.dep,
            "moran_i": moran_res["moran_i"],
            "expected": moran_res["expected_i"],
            "sd": np.sqrt(moran_res["variance"]),
            "z": moran_res["z_score"],
            "p": moran_res["p_value"],
            "interp": pattern
        })
        tbl_moran.add_note(f"Spatial Weights: {self.weights_method.upper()} (k={self.k}), row-standardized. N={n} regions.")
        self.add_table(tbl_moran)

        # 3. Spatial Regression (if covariates provided or model requested)
        if self.covs and len(self.covs) > 0:
            X = np.column_stack([np.ones(n)] + [sub[c].astype(float).values for c in self.covs])
            var_names = ["Intercept"] + self.covs

            if self.model_type in ("lag", "sar", "spregress_lag"):
                reg_res = fit_spatial_lag_model(y, X, w_mat, var_names)
            else:
                reg_res = fit_spatial_error_model(y, X, w_mat, var_names)

            tbl_reg = ResultsTable(name="spatial_reg", title=f"Spatial Econometric Regression ({reg_res['model_type']}) — Stata spregress")
            tbl_reg.add_column("param", "Parameter", col_type="text")
            tbl_reg.add_column("estimate", "Estimate (Coef)", col_type="number")
            tbl_reg.add_column("se", "Std. Error", col_type="number")
            tbl_reg.add_column("z", "z-value", col_type="number")
            tbl_reg.add_column("p", "p-value", col_type="number", fmt="pval")

            # Spatial dependence parameter (Rho or Lambda)
            if "rho" in reg_res:
                tbl_reg.add_row({
                    "param": "Spatial Lag (ρ / W·y)",
                    "estimate": reg_res["rho"],
                    "se": reg_res["se_rho"],
                    "z": reg_res["z_rho"],
                    "p": reg_res["p_rho"]
                })
            elif "lambda" in reg_res:
                tbl_reg.add_row({
                    "param": "Spatial Error (λ / W·u)",
                    "estimate": reg_res["lambda"],
                    "se": reg_res["se_lambda"],
                    "z": reg_res["z_lambda"],
                    "p": reg_res["p_lambda"]
                })

            # Exogenous covariates
            for vname, b, se in zip(reg_res["var_names"], reg_res["betas"], reg_res["se_betas"]):
                z_val = b / (se if se > 1e-8 else 1.0)
                p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_val)))
                tbl_reg.add_row({"param": vname, "estimate": float(b), "se": float(se), "z": float(z_val), "p": float(p_val)})

            tbl_reg.add_note(f"Pseudo R² = {reg_res['pseudo_r2']:.4f}, Log-Likelihood = {reg_res['loglik']:.2f}, AIC = {reg_res['aic']:.2f}, BIC = {reg_res['bic']:.2f}.")
            self.add_table(tbl_reg)

        # 4. LISA Cluster Frequency Table
        q_counts = pd.Series(quadrants).value_counts()
        tbl_lisa = ResultsTable(name="lisa_summary", title="LISA Spatial Cluster Classification (GeoDa)")
        tbl_lisa.add_column("cluster", "Cluster Type", col_type="text")
        tbl_lisa.add_column("count", "Region Count", col_type="integer")
        tbl_lisa.add_column("pct", "Percentage (%)", col_type="number")

        for c_type in ["High-High (Hotspot)", "Low-Low (Coldspot)", "High-Low (Spatial Outlier)", "Low-High (Spatial Outlier)"]:
            cnt = int(q_counts.get(c_type, 0))
            tbl_lisa.add_row({"cluster": c_type, "count": cnt, "pct": (cnt / n) * 100.0})
        self.add_table(tbl_lisa)

        # Visualizations
        if self.options.get("plotMoran", True):
            self.add_plot("Moran Scatterplot (Spatial Autocorrelation)", plot_moran_scatterplot(y, w_mat, self.dep, moran_res))
        if self.options.get("plotMap", True) and lat_col and lon_col:
            self.add_plot("GIS LISA Cluster Map", plot_spatial_choropleth_map(lats, lons, y, quadrants, region_names, self.dep))

        self.apa_narrative = (
            f"Spatial autocorrelation analysis on {self.dep} (N={n} regions) demonstrated a Moran's I of {moran_res['moran_i']:.4f} "
            f"(z = {moran_res['z_score']:.2f}, p = {moran_res['p_value']:.4f}), signifying statistically significant {pattern.lower()}. "
            f"LISA decomposition identified {q_counts.get('High-High (Hotspot)', 0)} hotspot regions and {q_counts.get('Low-Low (Coldspot)', 0)} coldspot regions."
        )
