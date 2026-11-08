"""
OpenStat SEM — categorical (binary / ordinal) variables and WLS-family estimators (Mplus CATEGORICAL ARE).

Sample statistics (Muthén, 1984; three-stage limited information)
    stage 1  univariate: thresholds of each ordinal variable (probit regression on the covariates x when x is
             conditioned on), intercept / slopes / residual variance of each continuous variable (linear regression)
    stage 2  bivariate (two-step, univariate parameters fixed): polychoric (ordinal–ordinal), polyserial
             (ordinal–continuous) and Pearson (continuous–continuous) correlations of the (residual) latent responses
    stage 3  asymptotic covariance matrix Γ of all sample statistics from case-wise influence functions
             (score vectors of the univariate and bivariate likelihoods, as lavaan `muthen1984`)
    statistics vector s = (thresholds / means, vec(slopes), variances of continuous variables, off-diagonal
    covariances: polychoric r, polyserial r·sd, Pearson cov)

Estimators (Mplus names)
    WLSMV  diagonal weight W = diag(Γ)⁻¹, robust (sandwich) SE, mean- and variance-adjusted χ² (scaled-shifted,
           Asparouhov & Muthén 2010)                                           — Mplus default for CATEGORICAL
    WLSM   diagonal weight, robust SE, mean-adjusted (Satorra–Bentler) χ²
    WLS    full weight W = Γ⁻¹, SE (Δ'WΔ)⁻¹, χ² = N·F
    ULSMV  W = I, robust SE, scaled-shifted χ²;   ULS  W = I, robust SE, no χ² test
Parameterizations: delta (default; Var(y*) = 1, residual variances of y* not parameters) and theta (residual
variances of y* fixed at 1, scale factors derived).

Library components (checked numerically against lavaan 0.7.2)
    stage-1 probit regressions on x: statsmodels Probit (binary) / OrderedModel(distr="probit") (K ≥ 3)
    thresholds without x: scipy.stats.norm.ppf of cumulative proportions; minimization: scipy.optimize.least_squares
    implemented from the published methods (no library equivalent found; Genz 2004; Olsson 1979; Olsson, Drasgow &
    Dorans 1982; Muthén, du Toit & Spisic 1997; Asparouhov & Muthén 2010, 2018): bivariate normal CDF (Genz BVNU, identical to scipy's
    multivariate_normal.cdf to 2e-16 but vectorized for per-case use), polychoric / polyserial two-step ML with fixed
    thresholds (semopy.polycorr fails on SciPy ≥ 1.18 and its polyserial standardizes by the variance), case-wise
    scores and Γ, categorical moment structure, sandwich SE, scaled-shifted / Satorra–Bentler χ², baseline, SRMR,
    WRMR and modification indices (semopy's WLS/DWLS treat the data as continuous).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import ndtr

SQRT2PI = math.sqrt(2.0 * math.pi)
_GLX, _GLW = np.polynomial.legendre.leggauss(20)
_GLX, _GLW = _GLX[:10], _GLW[:10]          # negative half (Genz's tabulation)


class WLSError(ValueError):
    pass


# ======================================================================================= bivariate normal
def _phi(z):
    z = np.asarray(z, float)
    out = np.zeros_like(z)
    ok = np.isfinite(z)
    out[ok] = np.exp(-0.5 * z[ok] ** 2) / SQRT2PI
    return out


def _bvnu(h: np.ndarray, k: np.ndarray, r: float) -> np.ndarray:
    """P(X > h, Y > k), corr r, finite h and k (Genz 2004, BVNU; 20-point Gauss–Legendre throughout)."""
    h = np.asarray(h, float)
    k = np.asarray(k, float)
    tp = 2.0 * math.pi
    hk = h * k
    if r == 0.0:
        return ndtr(-h) * ndtr(-k)
    if abs(r) < 0.925:
        hs = 0.5 * (h * h + k * k)
        asr = math.asin(r)
        bvn = np.zeros_like(h)
        for xi, wi in zip(_GLX, _GLW):
            for sn in (math.sin(asr * (xi + 1) / 2), math.sin(asr * (-xi + 1) / 2)):
                bvn = bvn + wi * np.exp((sn * hk - hs) / (1 - sn * sn))
        return bvn * asr / (2 * tp) + ndtr(-h) * ndtr(-k)
    if r < 0:
        k = -k
        hk = -hk
    bvn = np.zeros_like(h)
    if abs(r) < 1:
        as_ = (1 - r) * (1 + r)
        a = math.sqrt(as_)
        bs = (h - k) ** 2
        c = (4 - hk) / 8
        d = (12 - hk) / 16
        bvn = a * np.exp(-(bs / as_ + hk) / 2) * (1 - c * (bs - as_) * (1 - d * bs / 5) / 3 + c * d * as_ * as_ / 5)
        b = np.sqrt(bs)
        with np.errstate(over="ignore", invalid="ignore"):
            corr = np.exp(-hk / 2) * math.sqrt(tp) * ndtr(-b / a) * b * (1 - c * bs * (1 - d * bs / 5) / 3)
        bvn = bvn - np.where(hk > -160, corr, 0.0)
        a2 = a / 2
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for xi, wi in zip(_GLX, _GLW):
                xs = (a2 * (xi + 1)) ** 2
                rs = math.sqrt(1 - xs)
                bvn = bvn + a2 * wi * (np.exp(-bs / (2 * xs) - hk / (1 + rs)) / rs
                                       - np.exp(-(bs / xs + hk) / 2) * (1 + c * xs * (1 + d * xs)))
                xs = as_ * (-xi + 1) ** 2 / 4
                rs = math.sqrt(1 - xs)
                bvn = bvn + a2 * wi * np.exp(-(bs / xs + hk) / 2) * (np.exp(-hk * (1 - rs) / (2 * (1 + rs))) / rs
                                                                    - (1 + c * xs * (1 + d * xs)))
        bvn = -bvn / tp
    if r > 0:
        return bvn + ndtr(-np.maximum(h, k))
    bvn = -bvn
    add = np.where(h < 0, ndtr(k) - ndtr(h), ndtr(-h) - ndtr(-k))
    return bvn + np.where(k > h, add, 0.0)


def pbvn(h, k, r: float) -> np.ndarray:
    """P(X < h, Y < k) for standard bivariate normal with correlation r; h, k may be ±inf."""
    h = np.asarray(h, float)
    k = np.asarray(k, float)
    h, k = np.broadcast_arrays(h, k)
    out = np.zeros(h.shape)
    fin = np.isfinite(h) & np.isfinite(k)
    if fin.any():
        out[fin] = _bvnu(-h[fin], -k[fin], float(r))
    hp = np.isposinf(h) & ~np.isneginf(k)
    out[hp] = ndtr(k[hp])
    kp = np.isposinf(k) & np.isfinite(h)
    out[kp] = ndtr(h[kp])
    return np.clip(out, 0.0, 1.0)


def dbvn(h, k, r: float) -> np.ndarray:
    h = np.asarray(h, float)
    k = np.asarray(k, float)
    h, k = np.broadcast_arrays(h, k)
    out = np.zeros(h.shape)
    fin = np.isfinite(h) & np.isfinite(k)
    if fin.any():
        hh, kk = h[fin], k[fin]
        rr = 1 - r * r
        out[fin] = np.exp(-(hh * hh - 2 * r * hh * kk + kk * kk) / (2 * rr)) / (2 * math.pi * math.sqrt(rr))
    return out


# ================================================================================ stage 1: univariate fits
@dataclass
class UniFit:
    kind: str                          # "ord" | "num"
    n: int
    nexo: int
    th: np.ndarray                     # ordinal: thresholds; numeric: [intercept]
    beta: np.ndarray                   # slopes on the covariates
    evar: float = 1.0                  # numeric: residual variance (ML, /N)
    y: Optional[np.ndarray] = None     # ordinal codes 0..K-1 (int) or numeric values
    z1: Optional[np.ndarray] = None    # ordinal: tau_{y} - x'b  (upper limit), +inf for the top category
    z2: Optional[np.ndarray] = None    # ordinal: tau_{y-1} - x'b (lower limit), -inf for the bottom category
    resid: Optional[np.ndarray] = None
    U1: Optional[np.ndarray] = None    # n x nth indicator: tau_k is the upper limit of the case's category
    U2: Optional[np.ndarray] = None    # n x nth indicator: tau_k is the lower limit
    scores: Optional[np.ndarray] = None  # n x (nth [+ nexo] [+ 1 var])  univariate log-likelihood scores
    converged: bool = True


def _ord_pieces(y, th, beta, X):
    n = len(y)
    K = len(th) + 1
    eta = X @ beta if X is not None and X.shape[1] else np.zeros(n)
    tt = np.concatenate([[-np.inf], th, [np.inf]])
    z1 = tt[y + 1] - eta
    z2 = tt[y] - eta
    return z1, z2


def _probit_regression(y: np.ndarray, X: np.ndarray, nth: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Univariate probit regression of an ordinal y (codes 0..K-1) on covariates X (lavaan conditional.x stage 1):
    P(y <= k | x) = Φ(τ_k − x'β). Binary y: statsmodels Probit (τ = −intercept); K >= 3: statsmodels OrderedModel."""
    import statsmodels.api as sm
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if nth == 1:
            res = sm.Probit(y, np.column_stack([np.ones(len(y)), X])).fit(method="newton", maxiter=100, tol=1e-12,
                                                                           disp=0)
            th, beta = np.array([-res.params[0]]), np.asarray(res.params[1:], float)
        else:
            om = OrderedModel(y, X, distr="probit")
            res = om.fit(method="bfgs", maxiter=5000, gtol=1e-10, disp=0)
            th = np.asarray(om.transform_threshold_params(res.params)[1:-1], float)
            beta = np.asarray(res.params[:X.shape[1]], float)
    conv = bool(res.mle_retvals.get("converged", True)) and np.all(np.isfinite(th)) and np.all(np.diff(th) > 0)
    return th, beta, conv


def fit_ordinal(y: np.ndarray, X: Optional[np.ndarray]) -> UniFit:
    y = np.asarray(y, int)
    n = len(y)
    K = int(y.max()) + 1
    nth = K - 1
    nexo = 0 if X is None else X.shape[1]
    if nexo == 0:
        freq = np.bincount(y, minlength=K).astype(float)
        th, beta, conv = stats.norm.ppf(np.cumsum(freq[:-1]) / n), np.zeros(0), True     # closed-form ML thresholds
    else:
        th, beta, conv = _probit_regression(y, X, nth)
    z1, z2 = _ord_pieces(y, th, beta, X)
    pi = ndtr(z1) - ndtr(z2)
    a, b = _phi(z1), _phi(z2)
    U1 = (y[:, None] == np.arange(nth)[None, :]).astype(float)
    U2 = (y[:, None] - 1 == np.arange(nth)[None, :]).astype(float)
    sc = [(U1 * a[:, None] - U2 * b[:, None]) / pi[:, None]]
    if nexo:
        sc.append(-X * ((a - b) / pi)[:, None])
    sc = np.hstack(sc)
    if nexo and not conv:
        # the optimizer's own flag can stay False at gtol 1e-10; accept when the analytic mean score vanishes
        conv = bool(np.all(np.isfinite(sc)) and np.max(np.abs(sc.mean(axis=0))) < 1e-6)
    return UniFit("ord", n, nexo, th, beta, 1.0, y, z1, z2, None, U1, U2, sc, conv)


def fit_numeric(y: np.ndarray, X: Optional[np.ndarray]) -> UniFit:
    y = np.asarray(y, float)
    n = len(y)
    nexo = 0 if X is None else X.shape[1]
    X1 = np.column_stack([np.ones(n)] + ([X] if nexo else []))
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    res = y - X1 @ coef
    evar = float(res @ res / n)
    if evar <= 0:
        raise WLSError("연속형 변수의 (잔차)분산이 0")
    sc_b = X1 * (res / evar)[:, None]
    sc_v = -1.0 / (2 * evar) + res * res / (2 * evar * evar)
    return UniFit("num", n, nexo, coef[:1], coef[1:], evar, y, None, None, res, None, None,
                  np.column_stack([sc_b, sc_v]))


# ============================================================================== stage 2: bivariate fits
def _polychoric_noexo(f1: UniFit, f2: UniFit) -> float:
    K1, K2 = len(f1.th) + 1, len(f2.th) + 1
    freq = np.zeros((K1, K2))
    np.add.at(freq, (f1.y, f2.y), 1.0)
    # lavaan zero-cell handling: 2x2 table — two empty cells -> ±1, one empty cell -> add 0.5 keeping margins
    if K1 == 2 and K2 == 2:
        zero = np.argwhere(freq.ravel() == 0).ravel()
        if len(zero) == 2:
            return 1.0 if freq[0, 0] > 0 else -1.0
        if len(zero) == 1:
            idx = zero[0]
            sgn = 1.0 if idx in (0, 3) else -1.0
            freq = freq + sgn * 0.5 * np.array([[1.0, -1.0], [-1.0, 1.0]])
        if freq[0, 0] * freq[1, 1] == freq[0, 1] * freq[1, 0]:
            return 0.0
    t1 = np.concatenate([[-np.inf], f1.th, [np.inf]])
    t2 = np.concatenate([[-np.inf], f2.th, [np.inf]])
    H, Kk = np.meshgrid(t1, t2, indexing="ij")

    def negll(rho):
        P = pbvn(H, Kk, rho)
        cell = P[1:, 1:] - P[:-1, 1:] - P[1:, :-1] + P[:-1, :-1]
        cell = np.maximum(cell, 1e-300)
        return -float(np.sum(freq * np.log(cell)))

    res = optimize.minimize_scalar(negll, bounds=(-0.999, 0.999), method="bounded",
                                   options={"xatol": 1e-12, "maxiter": 500})
    return float(res.x)


def _pair_cells(f1: UniFit, f2: UniFit, rho):
    return (pbvn(f1.z1, f2.z1, rho) - pbvn(f1.z2, f2.z1, rho) - pbvn(f1.z1, f2.z2, rho) + pbvn(f1.z2, f2.z2, rho))


def _polychoric_exo(f1: UniFit, f2: UniFit) -> float:
    def negll(rho):
        lik = np.maximum(_pair_cells(f1, f2, rho), 1e-300)
        return -float(np.sum(np.log(lik)))

    res = optimize.minimize_scalar(negll, bounds=(-0.999, 0.999), method="bounded",
                                   options={"xatol": 1e-12, "maxiter": 500})
    return float(res.x)


def _polyserial(fn: UniFit, fo: UniFit) -> float:
    z = fn.resid / math.sqrt(fn.evar)

    def negll(rho):
        r = math.sqrt(1 - rho * rho)
        with np.errstate(invalid="ignore"):
            a = np.where(np.isposinf(fo.z1), 1.0, ndtr((fo.z1 - rho * z) / r))
            b = np.where(np.isneginf(fo.z2), 0.0, ndtr((fo.z2 - rho * z) / r))
        p = np.maximum(a - b, 1e-300)
        return -float(np.sum(np.log(p)))

    res = optimize.minimize_scalar(negll, bounds=(-0.995, 0.995), method="bounded",
                                   options={"xatol": 1e-12, "maxiter": 500})
    return float(res.x)


def _pearson(f1: UniFit, f2: UniFit) -> float:
    e1, e2 = f1.resid, f2.resid
    return float((e1 @ e2) / math.sqrt((e1 @ e1) * (e2 @ e2)))


# ---------------------------------------------------------------------------------- bivariate scores
def _safe_mul(dens, other):
    """dens * other with dens == 0 where the limit is infinite (0 * inf = 0)."""
    with np.errstate(invalid="ignore", over="ignore"):
        out = dens * other
    return np.where(dens == 0, 0.0, out)


def _sc_ordord(f1: UniFit, f2: UniFit, rho: float, X):
    r = math.sqrt(1 - rho * rho)
    lik = _pair_cells(f1, f2, rho)
    lik = np.where(lik < np.finfo(float).eps ** (2 / 3), np.nan, lik)

    def part(za, zb1, zb2):
        d = _phi(za)
        with np.errstate(invalid="ignore"):
            u = np.where(np.isposinf(zb1), 1.0, ndtr((zb1 - rho * za) / r))
            v = np.where(np.isneginf(zb2), 0.0, ndtr((zb2 - rho * za) / r))
        return _safe_mul(d, u - v)

    y1_z1 = part(f1.z1, f2.z1, f2.z2)
    y1_z2 = part(f1.z2, f2.z1, f2.z2)
    y2_z1 = part(f2.z1, f1.z1, f1.z2)
    y2_z2 = part(f2.z2, f1.z1, f1.z2)
    dth1 = (f1.U1 * y1_z1[:, None] - f1.U2 * y1_z2[:, None]) / lik[:, None]
    dth2 = (f2.U1 * y2_z1[:, None] - f2.U2 * y2_z2[:, None]) / lik[:, None]
    dsl1 = dsl2 = None
    if X is not None:
        dsl1 = ((y1_z2 - y1_z1) / lik)[:, None] * X
        dsl2 = ((y2_z2 - y2_z1) / lik)[:, None] * X
    dx = (dbvn(f1.z1, f2.z1, rho) - dbvn(f1.z2, f2.z1, rho) - dbvn(f1.z1, f2.z2, rho) + dbvn(f1.z2, f2.z2, rho))
    return {"rho": dx / lik, "th1": dth1, "th2": dth2, "sl1": dsl1, "sl2": dsl2, "var1": None, "var2": None}


def _sc_numord(fn: UniFit, fo: UniFit, rho: float, X):
    """y1 numeric, y2 ordinal (lavaan lav_bvmix_cor_sc)."""
    r = math.sqrt(1 - rho * rho)
    sd = math.sqrt(fn.evar)
    z = fn.resid / sd
    with np.errstate(invalid="ignore"):
        ts1 = np.where(np.isposinf(fo.z1), np.inf, (fo.z1 - rho * z) / r)
        ts2 = np.where(np.isneginf(fo.z2), -np.inf, (fo.z2 - rho * z) / r)
    yz1, yz2 = _phi(ts1), _phi(ts2)
    py = ndtr(ts1) - ndtr(ts2)
    py = np.maximum(py, np.finfo(float).eps)
    inv = 1.0 / py
    dd = yz1 - yz2
    dmu = (1 / sd) * (z + inv * (rho / r) * dd)
    dvar = 1 / (2 * fn.evar) * ((z * z - 1) + (inv * rho * z / r) * dd)
    dth2 = (fo.U1 * yz1[:, None] - fo.U2 * yz2[:, None]) * (inv / r)[:, None]
    dsl1 = dsl2 = None
    if X is not None:
        dsl1 = dmu[:, None] * X
        dsl2 = ((yz2 - yz1) * inv / r)[:, None] * X
    t1 = _safe_mul(yz1, fo.z1 * rho - z)
    t2 = _safe_mul(yz2, fo.z2 * rho - z)
    drho = inv / (r ** 3) * (t1 - t2)
    return {"rho": drho, "th1": dmu[:, None], "th2": dth2, "sl1": dsl1, "sl2": dsl2, "var1": dvar, "var2": None}


def _sc_numnum(f1: UniFit, f2: UniFit, rho: float, X):
    r = 1 - rho * rho
    y1c, y2c = f1.resid, f2.resid
    v1, v2 = f1.evar, f2.evar
    s1, s2 = math.sqrt(v1), math.sqrt(v2)
    dmu1 = (2 * y1c / v1 - 2 * rho * y2c / (s1 * s2)) / (2 * r)
    dmu2 = -(2 * rho * y1c / (s1 * s2) - 2 * y2c / v2) / (2 * r)
    dv1 = -(0.5 / v1 - ((y1c * y1c) / (v1 * v1) - rho * y1c * y2c / (v1 * s1 * s2)) / (2 * r))
    dv2 = -(0.5 / v2 + (rho * y1c * y2c / (v2 * s1 * s2) - (y2c * y2c) / (v2 * v2)) / (2 * r))
    dsl1 = dsl2 = None
    if X is not None:
        dsl1 = dmu1[:, None] * X
        dsl2 = dmu2[:, None] * X
    zz = (y1c * y1c) / v1 - 2 * rho * y1c * y2c / (s1 * s2) + (y2c * y2c) / v2
    drho = rho / r + (y1c * y2c / (s1 * s2 * r) - zz * rho / (r * r))
    return {"rho": drho, "th1": dmu1[:, None], "th2": dmu2[:, None], "sl1": dsl1, "sl2": dsl2, "var1": dv1, "var2": dv2}


# ===================================================================================== sample statistics
@dataclass
class CatSampleStats:
    names: List[str]
    types: List[str]                  # "ord" | "num" per variable
    nlev: List[int]
    xnames: List[str]
    n: int
    th: List[np.ndarray]              # per variable: thresholds (ord) or [intercept/mean] (num)
    slopes: np.ndarray                # p x nexo
    var: np.ndarray                   # p (1 for ordinal)
    cor: np.ndarray                   # p x p (residual) correlations
    cov: np.ndarray                   # p x p: cor scaled by sd of continuous variables
    s: np.ndarray                     # statistics vector
    Gamma: Optional[np.ndarray]       # N x asymptotic covariance of s (lavaan NACOV, sign of means +)
    labels: List[Tuple[str, Any]]     # per element of s: ("th", (var, k)) | ("mean", var) | ("slope", (var, x)) |
                                      #                    ("var", var) | ("cov", (vi, vj))
    th_pos: Dict[Tuple[int, int], int] = field(default_factory=dict)
    slope_pos: Dict[Tuple[int, int], int] = field(default_factory=dict)
    var_pos: Dict[int, int] = field(default_factory=dict)
    cov_pos: Dict[Tuple[int, int], int] = field(default_factory=dict)
    mean_x: Optional[np.ndarray] = None
    cov_x: Optional[np.ndarray] = None
    warnings: List[str] = field(default_factory=list)


def sample_statistics(Y: np.ndarray, types: Sequence[str], names: Sequence[str], X: Optional[np.ndarray] = None,
                      xnames: Sequence[str] = (), gamma: bool = True) -> CatSampleStats:
    """Y: n x p (ordinal coded 0..K-1, continuous as values), complete data. X: n x nexo covariates or None."""
    Y = np.asarray(Y, float)
    n, p = Y.shape
    types = list(types)
    nexo = 0 if X is None else X.shape[1]
    Xc = None if nexo == 0 else np.asarray(X, float)
    warnings: List[str] = []
    fits: List[UniFit] = []
    for j in range(p):
        if types[j] == "ord":
            f = fit_ordinal(Y[:, j].astype(int), Xc)
            if not f.converged:
                warnings.append(f"프로빗 회귀 수렴 기준 미충족: {names[j]}")
        else:
            f = fit_numeric(Y[:, j], Xc)
        fits.append(f)
    cor = np.eye(p)
    for j in range(p - 1):
        for i in range(j + 1, p):
            fi, fj = fits[i], fits[j]
            if fi.kind == "num" and fj.kind == "num":
                rho = _pearson(fi, fj)
            elif fi.kind == "num":
                rho = _polyserial(fi, fj)
            elif fj.kind == "num":
                rho = _polyserial(fj, fi)
            else:
                rho = _polychoric_noexo(fj, fi) if nexo == 0 else _polychoric_exo(fj, fi)
            cor[i, j] = cor[j, i] = rho
            if abs(rho) > 0.99:
                warnings.append(f"상관이 ±1에 근접: {names[j]} – {names[i]} ({rho:.3f})")
    var = np.array([f.evar if f.kind == "num" else 1.0 for f in fits])
    sd = np.sqrt(var)
    cov = cor * np.outer(sd, sd)
    # ---------------------------------------------------------------- layout of the statistics vector
    nth_j = [len(f.th) for f in fits]
    th_start = np.concatenate([[0], np.cumsum(nth_j)[:-1]]).astype(int)
    nth = int(sum(nth_j))
    num_idx = [j for j in range(p) if types[j] == "num"]
    nsl = p * nexo
    labels: List[Tuple[str, Any]] = []
    th_pos, slope_pos, var_pos, cov_pos = {}, {}, {}, {}
    for j in range(p):
        for k in range(nth_j[j]):
            th_pos[(j, k)] = len(labels)
            labels.append(("th", (names[j], k)) if types[j] == "ord" else ("mean", names[j]))
    for xk in range(nexo):
        for j in range(p):
            slope_pos[(j, xk)] = len(labels)
            labels.append(("slope", (names[j], xnames[xk])))
    for j in num_idx:
        var_pos[j] = len(labels)
        labels.append(("var", names[j]))
    for j in range(p - 1):
        for i in range(j + 1, p):
            cov_pos[(i, j)] = len(labels)
            labels.append(("cov", (names[i], names[j])))
    s = np.zeros(len(labels))
    for j in range(p):
        s[th_start[j]:th_start[j] + nth_j[j]] = fits[j].th
    for (j, xk), pos in slope_pos.items():
        s[pos] = fits[j].beta[xk]
    for j, pos in var_pos.items():
        s[pos] = var[j]
    for (i, j), pos in cov_pos.items():
        s[pos] = cov[i, j]
    out = CatSampleStats(list(names), types, [len(f.th) + 1 if f.kind == "ord" else 0 for f in fits], list(xnames), n,
                         [f.th.copy() for f in fits], np.array([f.beta for f in fits]).reshape(p, nexo), var, cor, cov,
                         s, None, labels, th_pos, slope_pos, var_pos, cov_pos, warnings=warnings)
    if nexo:
        out.mean_x = Xc.mean(0)
        out.cov_x = np.atleast_2d(np.cov(Xc.T, bias=False))     # lavaan (categorical): unbiased cov.x
    if not gamma:
        return out
    # ---------------------------------------------------------------- stage 3: Gamma via influence functions
    a11_size = nth + nsl + len(num_idx)
    pstar = p * (p - 1) // 2
    own_idx: List[List[int]] = []
    for j in range(p):
        idx = list(range(th_start[j], th_start[j] + nth_j[j]))
        idx += [nth + j + p * xk for xk in range(nexo)]
        if types[j] == "num":
            idx.append(nth + nsl + num_idx.index(j))
        own_idx.append(idx)
    sc_uni = np.zeros((n, a11_size))
    for j, f in enumerate(fits):
        sc = f.scores
        sc_uni[:, th_start[j]:th_start[j] + nth_j[j]] = sc[:, :nth_j[j]]
        if nexo:
            sc_uni[:, [nth + j + p * xk for xk in range(nexo)]] = sc[:, nth_j[j]:nth_j[j] + nexo]
        if f.kind == "num":
            sc_uni[:, nth + nsl + num_idx.index(j)] = sc[:, -1]
    sc_cor = np.zeros((n, pstar))
    A21 = np.zeros((pstar, a11_size))
    H21 = np.zeros((pstar, a11_size))
    H22 = np.eye(pstar)
    for j in range(p - 1):
        for i in range(j + 1, p):
            pos = cov_pos[(i, j)] - a11_size
            fi, fj = fits[i], fits[j]
            rho = cor[i, j]
            if fi.kind == "num" and fj.kind == "num":
                d = _sc_numnum(fi, fj, rho, Xc)
                di = {"th": d["th1"], "sl": d["sl1"], "var": d["var1"]}
                dj = {"th": d["th2"], "sl": d["sl2"], "var": d["var2"]}
            elif fi.kind == "num":
                d = _sc_numord(fi, fj, rho, Xc)
                di = {"th": d["th1"], "sl": d["sl1"], "var": d["var1"]}
                dj = {"th": d["th2"], "sl": d["sl2"], "var": None}
            elif fj.kind == "num":
                d = _sc_numord(fj, fi, rho, Xc)
                di = {"th": d["th2"], "sl": d["sl2"], "var": None}
                dj = {"th": d["th1"], "sl": d["sl1"], "var": d["var1"]}
            else:
                d = _sc_ordord(fi, fj, rho, Xc)
                di = {"th": d["th1"], "sl": d["sl1"], "var": None}
                dj = {"th": d["th2"], "sl": d["sl2"], "var": None}
            scr = d["rho"]
            sc_cor[:, pos] = scr
            for v, dv in ((i, di), (j, dj)):
                cols = own_idx[v]
                blocks = [dv["th"]]
                if nexo:
                    blocks.append(dv["sl"])
                if dv["var"] is not None:
                    blocks.append(dv["var"][:, None])
                D = np.hstack(blocks)
                A21[pos, cols] = np.nansum(scr[:, None] * D, axis=0)
            num_i, num_j = fi.kind == "num", fj.kind == "num"
            vi, vj = var[i], var[j]
            if num_i and num_j:
                H21[pos, nth + nsl + num_idx.index(i)] = math.sqrt(vj) * rho / (2 * math.sqrt(vi))
                H21[pos, nth + nsl + num_idx.index(j)] = math.sqrt(vi) * rho / (2 * math.sqrt(vj))
                H22[pos, pos] = math.sqrt(vi) * math.sqrt(vj)
            elif num_i:
                H21[pos, nth + nsl + num_idx.index(i)] = rho / (2 * math.sqrt(vi))
                H22[pos, pos] = math.sqrt(vi)
            elif num_j:
                H21[pos, nth + nsl + num_idx.index(j)] = rho / (2 * math.sqrt(vj))
                H22[pos, pos] = math.sqrt(vj)
    if np.isnan(sc_cor).any():
        warnings.append("일부 사례의 이변량 확률이 0에 근접: 해당 사례의 점수(score)를 0으로 처리")
    sc_cor = np.nan_to_num(sc_cor)
    SC = np.hstack([sc_uni, sc_cor])
    inner = SC.T @ SC
    A11 = np.zeros((a11_size, a11_size))
    for idx in own_idx:
        A11[np.ix_(idx, idx)] = inner[np.ix_(idx, idx)]
    A22d = np.sum(sc_cor * sc_cor, axis=0)
    # as R's solve(): computationally singular (rcond < eps) -> generalized inverse (MASS::ginv tolerance)
    try:
        rc = 1.0 / np.linalg.cond(A11, 1) if a11_size else 1.0
    except np.linalg.LinAlgError:
        rc = 0.0
    if rc < np.finfo(float).eps:
        A11i = np.linalg.pinv(A11, rcond=math.sqrt(np.finfo(float).eps))
        warnings.append("가중행렬 구성 중 A11(단변량 통계량 정보행렬)이 특이 — 일반화 역행렬 사용 "
                        "(예: 0/1 변수를 연속형 결합 분포에 포함한 경우)")
    else:
        A11i = np.linalg.inv(A11)
    A22i = np.diag(np.where(A22d > 0, 1.0 / np.where(A22d > 0, A22d, 1.0), 0.0))
    Binv = np.zeros((a11_size + pstar, a11_size + pstar))
    Binv[:a11_size, :a11_size] = A11i
    Binv[a11_size:, :a11_size] = -A22i @ A21 @ A11i
    Binv[a11_size:, a11_size:] = A22i
    W = Binv @ inner @ Binv.T
    if num_idx:
        Hm = np.zeros_like(W)
        Hm[:a11_size, :a11_size] = np.eye(a11_size)
        Hm[a11_size:, :a11_size] = H21
        Hm[a11_size:, a11_size:] = H22
        W = Hm @ W @ Hm.T
    out.Gamma = W * n
    return out


# ============================================================================================ the model
from . import sem_engine as se  # noqa: E402  (sem_engine does not import this module at load time)

WLS_ESTIMATORS = ("wlsmv", "wlsm", "wls", "ulsmv", "uls")
ESTIMATOR_LABELS = {
    "wlsmv": "WLSMV (대각 가중 최소제곱, 평균·분산 보정 χ²)",
    "wlsm": "WLSM (대각 가중 최소제곱, 평균 보정 χ²)",
    "wls": "WLS (전체 가중행렬 가중 최소제곱)",
    "ulsmv": "ULSMV (비가중 최소제곱, 평균·분산 보정 χ²)",
    "uls": "ULS (비가중 최소제곱)",
}
MAX_CATEGORIES = 10


class CatRAM(se.RAMModel):
    """RAM matrices with y* residual variances implied by the delta parameterization: Var(y* | x) = 1/Δ², where Δ
    is the scale factor (Mplus {y}; fixed at 1 unless freed with 'y ~*~ y', as for later time points of a growth
    model)."""

    def __init__(self, spec: se.ModelSpec, xmom):
        super().__init__(spec, xmom)
        self.cat_idx = np.array([spec.idx[v] for v in spec.ov if v in spec.categorical], dtype=int)
        self.delta = spec.parameterization == "delta"
        self.xs = np.arange(spec.py, spec.p, dtype=int)
        self.eye = np.eye(spec.m)
        pos = {int(k): n for n, k in enumerate(self.cat_idx)}
        self.scale = [[(pos[r.i], r.pid if r.free else None, r.value) for r in spec.rows
                       if r.group == g and r.op == "~*~" and r.i in pos] for g in range(spec.G)]

    def scale_factors(self, theta, g):
        d = np.ones(len(self.cat_idx))
        for k, pid, val in self.scale[g]:
            d[k] = theta[pid] if pid is not None else val
        return d

    def matrices(self, theta, g):
        A, S, M = super().matrices(theta, g)
        if self.delta and len(self.cat_idx):
            O = self.cat_idx
            B = np.linalg.inv(self.eye - A)
            S0 = S.copy()
            if len(self.xs):
                S0[np.ix_(self.xs, self.xs)] = 0.0
            S0[O, O] = 0.0
            om0 = np.einsum("ij,jk,ik->i", B[O], S0, B[O])
            Mc = B[np.ix_(O, O)] ** 2
            S[O, O] = np.linalg.solve(Mc, 1.0 / self.scale_factors(theta, g) ** 2 - om0)
        return A, S, M

    def conditional(self, theta, g):
        """Moments of (y*, continuous y) given x: intercepts, residual covariance, slopes on x."""
        A, S, M = self.matrices(theta, g)
        B = np.linalg.inv(self.eye - A)
        if len(self.xs):
            S = S.copy()
            M = M.copy()
            S[np.ix_(self.xs, self.xs)] = 0.0
            M[self.xs] = 0.0
        py = self.spec.py
        Om = B @ S @ B.T
        mu = B @ M
        return mu[:py], Om[:py, :py], B[:py][:, self.xs], S


@dataclass
class _Layout:
    th_ord: np.ndarray      # positions in s of ordinal thresholds
    th_var: np.ndarray      # ov index of each threshold
    th_par: List[Any]       # (pid or None, fixed value) per threshold position
    mean_pos: np.ndarray
    mean_var: np.ndarray
    sl_pos: np.ndarray
    sl_var: np.ndarray
    sl_x: np.ndarray
    var_pos: np.ndarray
    var_var: np.ndarray
    cov_pos: np.ndarray
    cov_i: np.ndarray
    cov_j: np.ndarray


def _layout(spec: se.ModelSpec, ss: CatSampleStats, g: int = 0) -> _Layout:
    thr = {(r.i, r.j): r for r in spec.rows if r.group == g and r.op == "|"}
    th_ord, th_var, th_par, mean_pos, mean_var = [], [], [], [], []
    for (j, k), pos in sorted(ss.th_pos.items(), key=lambda kv: kv[1]):
        if ss.types[j] == "ord":
            r = thr[(spec.idx[ss.names[j]], k)]
            th_ord.append(pos)
            th_var.append(j)
            th_par.append((r.pid if r.free else None, r.value))
        else:
            mean_pos.append(pos)
            mean_var.append(j)
    sl = sorted(ss.slope_pos.items(), key=lambda kv: kv[1])
    vv = sorted(ss.var_pos.items(), key=lambda kv: kv[1])
    cv = sorted(ss.cov_pos.items(), key=lambda kv: kv[1])
    ia = lambda x: np.array(x, dtype=int)
    return _Layout(ia(th_ord), ia(th_var), th_par, ia(mean_pos), ia(mean_var),
                   ia([p for _, p in sl]), ia([k[0] for k, _ in sl]), ia([k[1] for k, _ in sl]),
                   ia([p for _, p in vv]), ia([k for k, _ in vv]),
                   ia([p for _, p in cv]), ia([k[0] for k, _ in cv]), ia([k[1] for k, _ in cv]))


class WLSModel:
    """Moment structure sigma(theta) for the WLS sample statistics and its least-squares fit."""

    def __init__(self, spec: se.ModelSpec, ram: CatRAM, ss: CatSampleStats, weight: np.ndarray):
        self.spec, self.ram, self.ss = spec, ram, ss
        self.lay = _layout(spec, ss)
        self.s = ss.s
        self.W = weight                       # vector (diagonal) or matrix
        self.diag = weight.ndim == 1
        self.ord_mask = np.array([t == "ord" for t in ss.types])
        self.q = spec.q

    def sigma(self, theta: np.ndarray) -> np.ndarray:
        lay = self.lay
        mu, Om, Pi, _ = self.ram.conditional(theta, 0)
        dg = np.diag(Om).copy()
        if np.any(dg[self.ord_mask] <= 0):
            raise np.linalg.LinAlgError("non-positive y* variance")
        d = np.ones(len(dg))
        d[self.ord_mask] = 1.0 / np.sqrt(dg[self.ord_mask])
        out = np.empty(len(self.s))
        tau = np.array([theta[pid] if pid is not None else val for pid, val in lay.th_par]) if lay.th_par else np.zeros(0)
        out[lay.th_ord] = (tau - mu[lay.th_var]) * d[lay.th_var]
        out[lay.mean_pos] = mu[lay.mean_var]
        if len(lay.sl_pos):
            out[lay.sl_pos] = Pi[lay.sl_var, lay.sl_x] * d[lay.sl_var]
        out[lay.var_pos] = dg[lay.var_var]
        out[lay.cov_pos] = Om[lay.cov_i, lay.cov_j] * d[lay.cov_i] * d[lay.cov_j]
        return out

    def jacobian(self, theta: np.ndarray, fun=None) -> np.ndarray:
        fun = fun or self.sigma
        f0 = fun(theta)
        J = np.zeros((len(f0), len(theta)))
        for k in range(len(theta)):
            h = 1e-6 * max(1.0, abs(theta[k]))
            tp, tm = theta.copy(), theta.copy()
            tp[k] += h
            tm[k] -= h
            J[:, k] = (fun(tp) - fun(tm)) / (2 * h)
        return J

    def objective(self, theta: np.ndarray) -> float:
        try:
            r = self.s - self.sigma(theta)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            return float("inf")
        if not np.all(np.isfinite(r)):
            return float("inf")
        return float(r @ (self.W * r) if self.diag else r @ self.W @ r)

    def _wmul(self, M):
        return self.W[:, None] * M if self.diag else self.W @ M

    def residuals(self, theta: np.ndarray) -> np.ndarray:
        """Weighted residuals L'(s − σ(θ)) with W = LL' (diagonal: √w·(s − σ)), so that F = ‖·‖²."""
        try:
            r = self.s - self.sigma(theta)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            return np.full(len(self.s), 1e10)
        if not np.all(np.isfinite(r)):
            return np.full(len(self.s), 1e10)
        return self._sqrtW * r if self.diag else self._L.T @ r

    def fit(self, theta0: np.ndarray, max_iter: int = 500) -> Tuple[np.ndarray, bool, int]:
        """Minimize F(θ) = (s − σ(θ))'W(s − σ(θ)) with scipy.optimize.least_squares (trust-region reflective)."""
        if self.diag:
            self._sqrtW = np.sqrt(self.W)
        else:
            self._L = np.linalg.cholesky(self.W)
        if not np.isfinite(self.objective(theta0)):
            raise WLSError("시작값에서 모형 내재 상관행렬 계산 불가 (모형 식별·시작값 확인)")
        jac = lambda th: -(self._sqrtW[:, None] * self.jacobian(th) if self.diag else self._L.T @ self.jacobian(th))
        res = optimize.least_squares(self.residuals, theta0, jac=jac, method="trf", xtol=1e-15, ftol=1e-15,
                                     gtol=1e-13, max_nfev=max_iter * 4, x_scale="jac")
        theta = res.x
        F = self.objective(theta)
        g = self._wmul(self.jacobian(theta)).T @ (self.s - self.sigma(theta))
        converged = bool(res.status > 0 and np.max(np.abs(g)) < 1e-7 * (1.0 + F))
        return theta, converged, int(res.nfev)


def _start_values(spec: se.ModelSpec, ss: CatSampleStats, statements) -> np.ndarray:
    theta = np.zeros(spec.q)
    seen = set()
    cols = {n: k for k, n in enumerate(ss.names)}
    cov = ss.cov
    var = np.diag(cov)
    first_ind: Dict[str, str] = {}
    for s in statements:
        if s.op == "=~" and s.lhs not in first_ind:
            first_ind[s.lhs] = s.terms[0].name

    def ref_var(f, depth=0):
        r = first_ind.get(f)
        if r is None or depth > 5:
            return 1.0
        if r in cols:
            return 0.5 * var[cols[r]] if var[cols[r]] > 0 else 1.0
        return ref_var(r, depth + 1) * 0.8

    for r in spec.rows:
        if not r.free or r.pid in seen:
            continue
        seen.add(r.pid)
        if r.start is not None:
            v = r.start
        elif r.op == "=~":
            ref = first_ind.get(r.lhs)
            v = 1.0
            if ref in cols and r.rhs in cols:
                phi = ref_var(r.lhs)
                v = float(np.clip(cov[cols[r.rhs], cols[ref]] / phi, -10, 10)) if phi > 0 else 1.0
                if abs(v) < 0.05:
                    v = 0.5
        elif r.op == "~":
            v = 0.0
        elif r.op == "~*~":
            v = 1.0
        elif r.op == "~~":
            if r.lhs == r.rhs:
                if r.lhs in spec.categorical:
                    v = 1.0                       # theta: residual variance of y* (Mplus fixes it at 1 by default)
                elif r.lhs in cols:
                    vv = var[cols[r.lhs]]
                    v = vv if r.lhs in spec.exo_obs else 0.5 * vv
                else:
                    v = ref_var(r.lhs) * (0.5 if r.lhs in spec.dependents else 1.0)
            else:
                v = 0.0
        elif r.op == "~1":
            v = float(ss.th[cols[r.lhs]][0]) if r.lhs in cols and ss.types[cols[r.lhs]] == "num" else 0.0
        elif r.op == "|":
            j = cols[r.lhs]
            v = float(ss.th[j][r.j]) if r.j < len(ss.th[j]) else 0.0
            if spec.parameterization == "theta":
                v *= math.sqrt(2.0)
        else:
            v = 0.0
        theta[r.pid] = v
    return theta


# ===================================================================================== data handling
def recode_categorical(series: pd.Series, name: str) -> Tuple[np.ndarray, List[Any], str]:
    """Codes 0..K-1 (NaN kept) and the original category values in order."""
    s = series
    if isinstance(s.dtype, pd.CategoricalDtype):
        if s.cat.ordered:
            cats = [c for c in s.cat.categories if (s == c).any()]
            how = "순서형 범주(지정 순서)"
        else:
            cats = sorted([c for c in s.cat.categories if (s == c).any()], key=lambda v: str(v))
            how = "범주 이름 정렬 순서"
    elif pd.api.types.is_numeric_dtype(s):
        vals = s.dropna().unique()
        bad = [v for v in vals if not float(v).is_integer()]
        if bad:
            raise WLSError(f"범주형 변수 {name}에 정수가 아닌 값 포함 ({bad[0]}): 범주 부호(0/1, 1–5 등) 확인")
        cats = sorted(vals.tolist())
        how = "값 오름차순"
        cats = [int(c) if float(c).is_integer() else c for c in cats]
    else:
        cats = sorted(s.dropna().astype(str).unique().tolist())
        s = s.astype("object").where(s.isna(), s.astype(str))
        how = "문자 정렬 순서(순서가 다르면 숫자 부호화 또는 순서형 범주로 지정)"
    if len(cats) < 2:
        raise WLSError(f"범주형 변수 {name}의 범주가 2개 미만 (분석 표본 기준)")
    if len(cats) > MAX_CATEGORIES:
        raise WLSError(f"범주형 변수 {name}의 범주 수 {len(cats)}개 — 최대 {MAX_CATEGORIES}개 (Mplus 규칙). "
                       "연속형으로 분석하거나 범주 통합")
    if pd.api.types.is_numeric_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype):
        arr = s.to_numpy(dtype=float)
        out = np.full(len(arr), np.nan)
        for k, c in enumerate(cats):
            out[arr == float(c)] = k
        return out, cats, how
    code = {c: k for k, c in enumerate(cats)}
    out = np.array([code.get(v, np.nan) if not (isinstance(v, float) and math.isnan(v)) and v is not None
                    else np.nan for v in s.tolist()], dtype=float)
    return out, cats, how


class WLSSEM:
    """Counterpart of sem_engine.SEMModel for WLS-family estimation with categorical variables."""

    def __init__(self, df: pd.DataFrame, model: str, estimator: str = "wlsmv", categorical: Sequence[str] = (),
                 parameterization: str = "delta", conditional_x: bool = True, missing: str = "listwise",
                 group: Optional[str] = None, mimic: str = "mplus"):
        self.model_text = model
        self.statements = se.parse_model(model)
        self.estimator = estimator
        self.missing = missing
        self.information = "expected"
        self.parameterization = parameterization
        self.conditional_x = bool(conditional_x)
        self.mimic = mimic
        self.group = None
        self.group_levels = [None]
        self.invariance = "configural"
        self.notes: List[str] = []
        if group:
            raise WLSError("범주형 변수 모형의 다집단 분석은 아직 지원하지 않음 (집단별 분석 또는 집단 더미 공변량 사용)")
        if missing not in ("listwise",):
            raise WLSError(f"WLS 계열 추정의 결측 처리 '{missing}' 미지원 — 목록별 삭제(listwise)만 지원 "
                           "(FIML은 ML 계열 추정에서 사용)")
        cat = [str(c) for c in categorical]
        missing_cols = [c for c in cat if c not in df.columns]
        if missing_cols:
            raise WLSError(f"범주형 변수가 데이터에 없음: {', '.join(missing_cols)}")
        # pass 1: variables of the model (number of categories not yet known)
        spec0 = se.ModelSpec(self.statements, df.columns, 1, "configural", {c: MAX_CATEGORIES + 1 for c in cat},
                             parameterization, conditional_x)
        used = spec0.obs
        bad = [c for c in used if c not in cat and not pd.api.types.is_numeric_dtype(df[c])]
        if bad:
            raise WLSError(f"숫자형이 아닌 변수: {', '.join(bad)} — 이분·순서형이면 범주형 변수(categorical)로 지정")
        self.n_input = len(df)
        keep = df[used].notna().all(axis=1)
        if (~keep).sum():
            self.notes.append(f"목록별 삭제: 모형 변수에 결측이 있는 {int((~keep).sum())}개 사례 제외 "
                              "(Mplus WLSMV 기본은 쌍별 이용, lavaan 기본은 목록별 삭제)")
        d = df.loc[keep, used].copy()
        self.df_used = df.loc[keep].copy()
        self.cat_info: Dict[str, Dict[str, Any]] = {}
        ncat: Dict[str, int] = {}
        for c in cat:
            codes, cats, how = recode_categorical(d[c], c)
            d[c] = codes
            ncat[c] = len(cats)
            counts = np.bincount(codes.astype(int), minlength=len(cats))
            self.cat_info[c] = {"categories": cats, "order": how, "counts": counts.tolist()}
        self.spec = se.ModelSpec(self.statements, df.columns, 1, "configural", ncat, parameterization, conditional_x)
        spec = self.spec
        if spec.q == 0:
            raise WLSError("추정할 자유 모수 없음")
        self.N = int(len(d))
        self.group_n = [self.N]
        if self.N < 10:
            raise WLSError("사례 수 부족")
        Y = d[spec.ov].to_numpy(float)
        X = d[spec.ox].to_numpy(float) if spec.qx else None
        types = ["ord" if v in ncat else "num" for v in spec.ov]
        if X is not None:
            Xc = np.column_stack([np.ones(len(X)), X])
            if np.linalg.matrix_rank(Xc) < Xc.shape[1]:
                raise WLSError("공변량(x) 행렬의 계수 부족 (상수·완전 공선 공변량 확인)")
        self.ss = sample_statistics(Y, types, spec.ov, X, spec.ox)
        for w in self.ss.warnings:
            self.notes.append(w)
        if spec.qx:
            self.xmom = [(self.ss.mean_x, self.ss.cov_x)]
        else:
            self.xmom = [(np.zeros(0), np.zeros((0, 0)))]
        self.ram = CatRAM(spec, self.xmom)
        self.data = []
        G = self.ss.Gamma
        if estimator in ("wlsmv", "wlsm"):
            dg = np.diag(G)
            if np.any(dg <= 0) or not np.all(np.isfinite(dg)):
                raise WLSError("가중행렬(Γ 대각) 계산 불가")
            self.W = 1.0 / dg
        elif estimator == "wls":
            try:
                self.W = np.linalg.inv(G)
            except np.linalg.LinAlgError as e:
                raise WLSError("WLS 가중행렬 Γ의 역행렬 계산 불가 (표본 크기 대비 통계량 과다 — WLSMV 사용)") from e
            if np.min(np.linalg.eigvalsh(0.5 * (G + G.T))) <= 0:
                raise WLSError("WLS 가중행렬 Γ가 양정치가 아님 (WLSMV 사용)")
        else:
            self.W = np.ones(len(self.ss.s))
        self.model = WLSModel(spec, self.ram, self.ss, self.W)


def _traces(U: np.ndarray, Gm: np.ndarray) -> Tuple[float, float]:
    UG = U @ Gm
    return float(np.trace(UG)), float(np.sum(UG * UG.T))


def _u_matrix(W, D):
    Wm = np.diag(W) if W.ndim == 1 else W
    WD = Wm @ D
    E = D.T @ WD
    return Wm - WD @ np.linalg.solve(E, WD.T)


def estimate_wls(df: pd.DataFrame, model: str, estimator: str = "wlsmv", categorical: Sequence[str] = (),
                 parameterization: str = "delta", conditional_x: bool = True, missing: str = "listwise",
                 group: Optional[str] = None, mimic: str = "mplus") -> se.SEMResult:
    try:
        sem = WLSSEM(df, model, estimator, categorical, parameterization, conditional_x, missing, group, mimic)
    except WLSError as e:
        raise se.SEMError(str(e)) from e
    spec, mdl, ss = sem.spec, sem.model, sem.ss
    N = sem.N
    notes = list(sem.notes)
    th0 = _start_values(spec, ss, sem.statements)
    try:
        theta, converged, iters = mdl.fit(th0)
    except (WLSError, np.linalg.LinAlgError) as e:
        raise se.SEMError(f"추정 실패: {e}") from e
    if not converged:
        alt = th0.copy()
        for r in spec.rows:
            if r.free and r.op == "=~":
                alt[r.pid] = 0.7
        try:
            th2, c2, it2 = mdl.fit(alt)
            if c2 and mdl.objective(th2) <= mdl.objective(theta) + 1e-12:
                theta, converged, iters = th2, c2, it2
        except (WLSError, np.linalg.LinAlgError):
            pass
    if not converged:
        notes.append("수렴 기준 미충족: 추정치 해석 주의 (시작값·모형 식별 확인)")
    Fmin = mdl.objective(theta)
    D = mdl.jacobian(theta)
    Wm = np.diag(sem.W) if sem.W.ndim == 1 else sem.W
    WD = Wm @ D
    E = D.T @ WD
    q = spec.q
    identified, worst = se.identification_check(E)
    Gam = ss.Gamma
    mp = mimic == "mplus"
    # Mplus 8.8 (checked on the User's Guide examples 3.12–3.14, 5.2, 5.3, 5.10, 6.4, 6.5, 6.15): V = E⁻¹(Δ'WΓWΔ)E⁻¹/N,
    # χ² from 2N·F_Mplus = N·(s − σ)'W(s − σ), RMSEA with N. lavaan (mimic = "lavaan"): /(N − 1) throughout.
    # (lavaan's own mimic = "Mplus" gives SE larger than Mplus by the factor N/(N − 1).)
    nn = N if mp else N - 1
    se_fac = 1.0 / nn
    if not identified:
        V = np.full((q, q), np.nan)
        bad = [r for r in spec.rows if r.pid == worst]
        nm = f"{bad[0].lhs} {bad[0].op} {bad[0].rhs}".strip() if bad else str(worst)
        notes.append(f"모형 식별 문제: Δ'WΔ가 비정칙 — 표준오차 계산 불가 (관련 모수: {nm})")
    else:
        Ei = np.linalg.inv(E)
        if estimator == "wls":
            V = Ei / nn                           # W = Γ⁻¹: the sandwich reduces to E⁻¹ (lavaan se = "standard")
        else:
            V = Ei @ (WD.T @ Gam @ WD) @ Ei * se_fac
    sev = np.sqrt(np.clip(np.diag(V), 0, None))
    # ------------------------------------------------------------------ test statistics
    nstat = len(ss.s)
    df_ = int(nstat - q)
    T0 = nn * Fmin                            # Mplus: 2N·F with F = ½(s − σ)'W(s − σ); lavaan: (N − 1)·F
    Gt = Gam
    fs: Dict[str, Any] = {"npar": q, "N": N, "G": 1, "df": df_, "chi2": T0, "fmin": Fmin, "nstat": nstat,
                          "estimator": estimator, "ll0": float("nan"), "ll1": float("nan"), "aic": float("nan"),
                          "bic": float("nan"), "sabic": float("nan")}

    def adjust(T, dfm, U):
        if estimator in ("wls", "uls") or dfm <= 0:
            return T, float("nan"), float("nan")
        trUG, trUG2 = _traces(U, Gt)
        if estimator == "wlsm":
            c = trUG / dfm
            return T / c, c, float("nan")
        a = math.sqrt(dfm / trUG2) if trUG2 > 0 else float("nan")
        b = dfm - a * trUG
        return a * T + b, 1.0 / a, b

    U = _u_matrix(sem.W, D) if (identified and df_ > 0) else None
    if U is not None:
        T, sc, sh = adjust(T0, df_, U)
    else:
        T, sc, sh = (T0, float("nan"), float("nan"))
    if df_ <= 0:
        T = 0.0
    fs.update({"chi2_test": T, "scaling": sc, "shift": sh})
    test_ok = estimator != "uls"
    fs["pvalue"] = float(stats.chi2.sf(T, df_)) if (df_ > 0 and np.isfinite(T) and test_ok) else float("nan")
    # ------------------------------------------------------------------ baseline (independence) model
    lay = mdl.lay
    # baseline: thresholds / means / variances free, slopes 0 (Mplus), correlations 0 except among observed
    # exogenous covariates in the joint structure (conditional_x = False; lavaan baseline.fixed.x.free.cov)
    xj = {ss.names.index(v) for v in spec.x_names if v in ss.names}
    free_cov = [pos for (i, j), pos in ss.cov_pos.items() if i in xj and j in xj]
    free_b = np.concatenate([lay.th_ord, lay.mean_pos, lay.var_pos, np.array(free_cov, dtype=int)]).astype(int)
    Db = np.zeros((nstat, len(free_b)))
    Db[free_b, np.arange(len(free_b))] = 1.0
    if sem.W.ndim == 1:
        sb = np.zeros(nstat)
        sb[free_b] = ss.s[free_b]           # diagonal weight: free baseline statistics are reproduced exactly
    else:
        tb = np.linalg.solve(Db.T @ Wm @ Db, Db.T @ Wm @ ss.s)
        sb = Db @ tb
    rb = ss.s - sb
    Tb0 = nn * float(rb @ Wm @ rb)
    dfb = int(nstat - len(free_b))
    Ub = _u_matrix(sem.W, Db)
    TB, scb, shb = adjust(Tb0, dfb, Ub)
    fs.update({"chi2_base_std": Tb0, "chi2_base": TB, "df_base": dfb, "scaling_base": scb, "shift_base": shb})
    fs["p_base"] = float(stats.chi2.sf(TB, dfb)) if dfb > 0 and np.isfinite(TB) and test_ok else float("nan")
    if test_ok and df_ > 0:
        est, lo, up, pc = se.rmsea_ci(T, df_, nn, 1)         # Mplus: N; lavaan: N − G for least squares
    else:
        est, lo, up, pc = (0.0, 0.0, 0.0, float("nan")) if df_ <= 0 else (float("nan"),) * 4
    fs["rmsea"], fs["rmsea_lo"], fs["rmsea_hi"], fs["rmsea_p"] = est, lo, up, pc
    if test_ok:
        num = max(T - df_, 0.0)
        den = max(T - df_, TB - dfb, 0.0)
        fs["cfi"] = 1.0 - num / den if den > 0 else 1.0
        fs["tli"] = ((TB / dfb) - (T / df_)) / ((TB / dfb) - 1.0) if (dfb > 0 and df_ > 0 and TB / dfb != 1) else 1.0
        if mp:
            fs["tli"] = min(fs["tli"], 1.0)    # Mplus prints TLI truncated at 1
    else:
        fs["cfi"] = fs["tli"] = float("nan")
    if mp:
        fs["srmr"] = srmr_mplus(ss, ss.s, mdl.sigma(theta))
    else:
        # lavaan categorical SRMR: off-diagonal correlation residuals over p(p+1)/2
        mu, Om, Pi, _ = sem.ram.conditional(theta, 0)
        dsd = np.sqrt(np.diag(Om))
        R_imp = Om / np.outer(dsd, dsd)
        R_obs = ss.cor
        p = len(ss.names)
        il = np.tril_indices(p, -1)
        fs["srmr"] = float(math.sqrt(np.sum((R_obs[il] - R_imp[il]) ** 2) / (p * (p + 1) / 2))) if p > 1 else 0.0
    fs["wrmr"] = float(math.sqrt(T0 / nstat)) if estimator in ("wlsmv", "wlsm", "wls") else float("nan")
    fs["h1"] = None
    fit = se.FitResult(spec, sem.ram, [], theta, float("nan"), 0.0, bool(converged), int(iters), "")
    for r in spec.rows:
        if r.op == "~~" and r.lhs == r.rhs and r.free and theta[r.pid] < 0:
            notes.append(f"음의 (잔차)분산 추정: {r.lhs} = {theta[r.pid]:.3f} (부적절 해, 모형 수정 검토)")
    A, S, M = sem.ram.matrices(theta, 0)
    for v in spec.ov:
        if v in spec.categorical:
            k = spec.idx[v]
            if S[k, k] < 0:
                notes.append(f"범주형 변수 {v}의 y* 잔차분산이 음수({S[k, k]:.3f}) — 부적절 해")
    res = se.SEMResult(sem, fit, V, sev, fs, notes, None, identified)
    return res


# ============================================================================== SRMR (Mplus 8.1 definition)
def _stat_parts(vec: np.ndarray, ss: CatSampleStats):
    """Thresholds / intercepts, slopes and residual covariance matrix of (y*, y) | x from a statistics vector."""
    p, q = len(ss.names), len(ss.xnames)
    tau = [np.array([vec[ss.th_pos[(j, k)]] for k in range(len(ss.th[j]))]) for j in range(p)]
    beta = np.zeros((p, q))
    for (j, xk), pos in ss.slope_pos.items():
        beta[j, xk] = vec[pos]
    th = np.eye(p)
    for j, pos in ss.var_pos.items():
        th[j, j] = vec[pos]
    for (i, j), pos in ss.cov_pos.items():
        th[i, j] = th[j, i] = vec[pos]
    return tau, beta, th


def srmr_mplus(ss: CatSampleStats, s_h1: np.ndarray, s_h0: np.ndarray) -> float:
    """SRMR for the WLS family as defined in Asparouhov & Muthén (2018), "SRMR in Mplus", section 2.7 (Mplus 8.1+):
    correlation residuals of the joint (y*, y, x) matrix V = [θ + βSxβ', βSx; Sxβ', Sx], standardized mean residuals
    of continuous dependent variables, variance residuals and category-probability residuals of categorical
    variables, divided by the number of residuals that are not zero by default (equations 12–18)."""
    p, q = len(ss.names), len(ss.xnames)
    ordm = [t == "ord" for t in ss.types]
    if q:
        Sx = np.atleast_2d(ss.cov_x) * (ss.n - 1) / ss.n          # sample covariance of x (ML, divisor N)
        mx = np.asarray(ss.mean_x, float)
    else:
        Sx, mx = np.zeros((0, 0)), np.zeros(0)

    def joint(vec):
        tau, beta, th = _stat_parts(vec, ss)
        V = np.zeros((p + q, p + q))
        V[:p, :p] = th + beta @ Sx @ beta.T
        if q:
            V[:p, p:] = beta @ Sx
            V[p:, :p] = V[:p, p:].T
            V[p:, p:] = Sx
        return tau, beta, V

    t1, b1, V1 = joint(s_h1)
    t0, b0, V0 = joint(s_h0)
    sd1, sd0 = np.sqrt(np.diag(V1)), np.sqrt(np.diag(V0))
    R1, R0 = V1 / np.outer(sd1, sd1), V0 / np.outer(sd0, sd0)
    il = np.tril_indices(p + q, -1)
    S = float(np.sum((R1[il] - R0[il]) ** 2))
    for j in range(p):
        if ordm[j]:
            e1 = float(b1[j] @ mx) if q else 0.0
            e0 = float(b0[j] @ mx) if q else 0.0
            c1 = np.diff(np.concatenate([[0.0], stats.norm.cdf((t1[j] - e1) / sd1[j]), [1.0]]))
            c0 = np.diff(np.concatenate([[0.0], stats.norm.cdf((t0[j] - e0) / sd0[j]), [1.0]]))
            S += float(np.sum((c1 - c0) ** 2))
        else:
            m1 = t1[j][0] + (float(b1[j] @ mx) if q else 0.0)
            m0 = t0[j][0] + (float(b0[j] @ mx) if q else 0.0)
            S += (m1 / sd1[j] - m0 / sd0[j]) ** 2
        S += ((V1[j, j] - V0[j, j]) / V1[j, j]) ** 2
    ptot = p + q
    p1 = sum(ordm)
    lsum = sum(ss.nlev[j] for j in range(p) if ordm[j])
    d = ptot * (ptot + 3) / 2 + lsum - (p1 + q * (q + 3) / 2 if q else 2 * p1)
    return float(math.sqrt(S / d)) if d > 0 else 0.0


# ============================================================================== modification indices
def modification_indices_wls(res: se.SEMResult, min_mi: float = 10.0) -> List[Dict[str, Any]]:
    """Score tests for fixed parameters under WLS-family estimation (lavaan modindices definition):
    MI = N·(k·Δc'Wr)²/(Δc'WΔc − Δc'WΔ(Δ'WΔ)⁻¹Δ'WΔc) with k = (N − 1)/N (gradient of lavaan's fmin), EPC = MI/(N·k·Δc'Wr),
    r = s − σ(θ̂)."""
    sem, spec = res.sem, res.spec
    mdl: WLSModel = sem.model
    theta = res.theta
    N = sem.N
    W = sem.W
    Wm = np.diag(W) if W.ndim == 1 else W
    r = mdl.s - mdl.sigma(theta)
    D = mdl.jacobian(theta)
    WD = Wm @ D
    E = D.T @ WD
    try:
        Ei = np.linalg.inv(E)
    except np.linalg.LinAlgError:
        return []
    lat = spec.latent
    existing = {(rr.group, rr.mat, min(rr.i, rr.j) if rr.mat == "S" else rr.i, max(rr.i, rr.j) if rr.mat == "S" else rr.j)
                for rr in spec.rows}
    reg_pairs = {(rr.lhs, rr.rhs) for rr in spec.rows if rr.op == "~"}
    cands = []
    obs_ind = [v for v in spec.ov if v in spec.indicators]
    for f in lat:
        for y in obs_ind:
            k = (0, "A", spec.idx[y], spec.idx[f])
            if k not in existing:
                cands.append(("A", spec.idx[y], spec.idx[f], f, "=~", y, "BY"))
    struct_nodes = lat + [v for v in spec.ov if v in spec.dep_reg and v not in spec.indicators]
    for a in struct_nodes:
        for b in lat:
            if a == b or (a, b) in reg_pairs or (b, a) in reg_pairs:
                continue
            if (0, "A", spec.idx[a], spec.idx[b]) not in existing:
                cands.append(("A", spec.idx[a], spec.idx[b], a, "~", b, "ON"))
    ys = [v for v in spec.ov if v in spec.dependents]
    for groupvars in (ys, lat):
        for ai, a in enumerate(groupvars):
            for b in groupvars[ai + 1:]:
                i, j = sorted((spec.idx[a], spec.idx[b]))
                if (0, "S", i, j) not in existing:
                    cands.append(("S", i, j, a, "~~", b, "WITH"))
    ram = sem.ram
    der = se.Derived(res)
    A0, S0, M0, var, BM = der._group_stats(theta, 0)
    sdv = lambda k: math.sqrt(var[k]) if var[k] > 0 else float("nan")
    out = []
    for mat, i, j, lhs, op, rhs, kind in cands:
        base = ram.A0[0] if mat == "A" else ram.S0[0]

        def sig_at(v):
            old = (base[i, j], base[j, i])
            base[i, j] = v
            if mat == "S":
                base[j, i] = v
            try:
                return mdl.sigma(theta)
            finally:
                base[i, j], base[j, i] = old

        h = 1e-6
        v0 = base[i, j]
        try:
            dc = (sig_at(v0 + h) - sig_at(v0 - h)) / (2 * h)
        except np.linalg.LinAlgError:
            continue
        g = float(dc @ Wm @ r)
        Wdc = Wm @ dc
        vcc = float(dc @ Wdc - (WD.T @ dc) @ Ei @ (WD.T @ dc))
        if vcc < np.finfo(float).eps ** (1 / 3):
            continue
        k = (N - 1) / N
        mi = N * (k * g) ** 2 / vcc
        epc = k * g / vcc
        if mat == "A":
            std_epc = epc * sdv(j) / sdv(i)
        else:
            Sm = ram.matrices(theta, 0)[1]
            std_epc = epc / math.sqrt(Sm[i, i] * Sm[j, j]) if Sm[i, i] > 0 and Sm[j, j] > 0 else float("nan")
        if mi >= min_mi:
            out.append({"group": 0, "lhs": lhs, "op": op, "rhs": rhs, "kind": kind, "mi": float(mi), "epc": float(epc),
                        "std_epc": float(std_epc)})
    out.sort(key=lambda d_: -d_["mi"])
    return out
