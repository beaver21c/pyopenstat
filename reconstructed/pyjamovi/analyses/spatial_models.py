"""
Maximum-likelihood spatial econometric models for cross-section and panel data
(equivalent in scope to Stata spregress, spxtregress and the user-written xsmle).

Per period t (N units, W is N x N, stacked as T blocks):
    y_t = rho W y_t + X_t beta + W X_t theta + u_t,      u_t = lambda W u_t + e_t,   e ~ N(0, sigma2 I)

Models: OLS, SLX, SAR (rho), SEM (lambda), SDM (rho + WX), SDEM (lambda + WX), SAC/SARAR (rho + lambda).
Panel effects:
- fixed effects (individual / time / two-way) by demeaning; Lee & Yu (2010) transformation approach
  (log-likelihood scaled by N(T-1), (N-1)T or (N-1)(T-1) observations; time effects need row-standardised W)
- random effects (SAR / SDM, Elhorst 2014): quasi-demeaning with phi = sigma2 / (T sigma2_mu + sigma2)
Inference: observed information (numerical Hessian of the full log-likelihood, Stata vce(oim) concept).
Effects (LeSage & Pace 2009): direct / indirect / total with delta-method or simulated standard errors.
Diagnostics: Moran's I of OLS residuals, LM-lag / LM-error and robust versions (Anselin et al. 1996; Elhorst 2014 for panels),
Jarque-Bera, Koenker-Bassett, multicollinearity condition number; LR tests between nested models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats

from .spatial_weights import SpatialWeights

MODEL_LABELS = {
    "ols": "OLS(공간효과 없음)",
    "slx": "SLX(설명변수 공간시차)",
    "sar": "SAR(공간시차모형)",
    "sem": "SEM(공간오차모형)",
    "sdm": "SDM(공간더빈모형)",
    "sdem": "SDEM(공간더빈오차모형)",
    "sac": "SAC(공간시차·오차 혼합모형)",
}
MODEL_PARTS = {  # (spatial lag of y, spatial error, spatial lags of X)
    "ols": (False, False, False), "slx": (False, False, True), "sar": (True, False, False),
    "sem": (False, True, False), "sdm": (True, False, True), "sdem": (False, True, True), "sac": (True, True, False),
}
FE_LABELS = {"none": "효과 없음(합동)", "ind": "개체(지역) 고정효과", "time": "시점 고정효과", "both": "양방향(지역·시점) 고정효과"}
LR_PAIRS = [  # (restricted, unrestricted, hypothesis)
    ("ols", "sar", "ρ = 0"), ("ols", "sem", "λ = 0"), ("ols", "slx", "θ = 0"),
    ("sar", "sdm", "θ = 0"), ("slx", "sdm", "ρ = 0"), ("sem", "sdm", "θ = −ρβ(공통요인)"),
    ("sem", "sdem", "θ = 0"), ("slx", "sdem", "λ = 0"), ("sar", "sac", "λ = 0"), ("sem", "sac", "ρ = 0"),
]


@dataclass
class PanelLayout:
    N: int
    T: int
    fe: str = "none"         # none | ind | time | both
    leeyu: bool = True
    re: bool = False

    def demean(self, v: np.ndarray) -> np.ndarray:
        if self.fe == "none" or self.re:
            return v
        if v.ndim == 1:
            return self._dm(v.reshape(self.T, self.N)).reshape(-1)
        return np.column_stack([self._dm(v[:, j].reshape(self.T, self.N)).reshape(-1) for j in range(v.shape[1])])

    def _dm(self, M: np.ndarray) -> np.ndarray:
        if self.fe == "ind":
            return M - M.mean(axis=0, keepdims=True)
        if self.fe == "time":
            return M - M.mean(axis=1, keepdims=True)
        return M - M.mean(axis=0, keepdims=True) - M.mean(axis=1, keepdims=True) + M.mean()

    def scaling(self, row_std: bool) -> Tuple[float, float, bool]:
        """(effective observations, multiplier of log-determinants, subtract ln(1-rho) term)."""
        N, T = self.N, self.T
        if self.re or self.fe == "none" or not self.leeyu:
            return float(N * T), float(T), False
        if self.fe == "ind":
            return float(N * (T - 1)), float(T - 1), False
        if not row_std:
            return float(N * T), float(T), False
        if self.fe == "time":
            return float((N - 1) * T), float(T), True
        return float((N - 1) * (T - 1)), float(T - 1), True


@dataclass
class FitResult:
    model: str
    names: List[str]
    coef: np.ndarray
    vcov: np.ndarray
    rho: Optional[float]
    lam: Optional[float]
    sigma2: float
    loglik: float
    n_obs: int
    n_params: int
    pseudo_r2: float
    resid: np.ndarray
    fitted: np.ndarray
    extra: Dict[str, Any] = field(default_factory=dict)
    durbin: List[str] = field(default_factory=list)
    base_names: List[str] = field(default_factory=list)
    hessian_ok: bool = True

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.vcov), 0.0))

    @property
    def aic(self) -> float:
        return -2.0 * self.loglik + 2.0 * self.n_params

    @property
    def bic(self) -> float:
        return -2.0 * self.loglik + np.log(self.n_obs) * self.n_params

    def param_index(self, name: str) -> int:
        return self.names.index(name)


def _lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    if X.shape[1] == 0:
        return np.zeros(0)
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _num_hessian(f, x: np.ndarray, steps: np.ndarray) -> np.ndarray:
    p = len(x)
    H = np.zeros((p, p))
    f0 = f(x)
    for i in range(p):
        ei = np.zeros(p); ei[i] = steps[i]
        H[i, i] = (f(x + ei) - 2.0 * f0 + f(x - ei)) / steps[i] ** 2
        for j in range(i):
            ej = np.zeros(p); ej[j] = steps[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4.0 * steps[i] * steps[j])
    return H


def _safe_inv(M: np.ndarray) -> Tuple[np.ndarray, bool]:
    try:
        V = np.linalg.inv(M)
        ok = bool(np.all(np.isfinite(V)) and np.all(np.diag(V) > 0))
        if ok:
            return V, True
    except np.linalg.LinAlgError:
        pass
    return np.linalg.pinv(M), False


def _row_std(W: SpatialWeights) -> bool:
    rs = np.asarray(W.W.sum(axis=1)).ravel()
    return bool(np.allclose(rs[rs > 0], 1.0))


# ---------------------------------------------------------------- design
@dataclass
class Design:
    y: np.ndarray            # transformed y (NT)
    Wy: np.ndarray           # transformed W y
    X: np.ndarray            # transformed regressors (incl. intercept / WX)
    names: List[str]
    base_names: List[str]
    durbin: List[str]
    dropped: List[str]
    y_raw: np.ndarray
    X_raw: np.ndarray        # untransformed regressors matching names (for prediction)


def build_design(y: np.ndarray, X: np.ndarray, names: Sequence[str], W: SpatialWeights, layout: PanelLayout,
                 durbin: Sequence[str], intercept: bool) -> Design:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float).reshape(len(y), -1)
    names = list(names)
    cols, labels = [X], list(names)
    dvars = [d for d in durbin if d in names]
    if dvars:
        idx = [names.index(d) for d in dvars]
        cols.append(W.lag(X[:, idx]))
        labels += [f"W·{d}" for d in dvars]
    Xall = np.column_stack(cols) if cols else np.zeros((len(y), 0))
    if intercept:
        Xall = np.column_stack([np.ones(len(y)), Xall])
        labels = ["(절편)"] + labels
    Wy_raw = W.lag(y)
    ys, Wys, Xs = layout.demean(y), layout.demean(Wy_raw), layout.demean(Xall)
    dropped: List[str] = []
    if not layout.re and layout.fe != "none":
        keep = [j for j in range(Xs.shape[1]) if np.std(Xs[:, j]) > 1e-10 * max(1.0, np.std(Xall[:, j]))]
        dropped = [labels[j] for j in range(Xs.shape[1]) if j not in keep]
        Xs, Xall, labels = Xs[:, keep], Xall[:, keep], [labels[j] for j in keep]
    # drop exactly collinear columns (e.g. W·x identical to x)
    if Xs.shape[1]:
        q, r = np.linalg.qr(Xs)
        diag = np.abs(np.diag(r))
        tol = diag.max() * 1e-9 if diag.size else 0
        keep = [j for j in range(Xs.shape[1]) if diag[j] > tol]
        if len(keep) < Xs.shape[1]:
            dropped += [labels[j] for j in range(Xs.shape[1]) if j not in keep]
            Xs, Xall, labels = Xs[:, keep], Xall[:, keep], [labels[j] for j in keep]
    return Design(ys, Wys, Xs, labels, names, [d for d in dvars if f"W·{d}" in labels], dropped, y, Xall)


# ---------------------------------------------------------------- estimation
class SpatialML:
    def __init__(self, W: SpatialWeights, layout: PanelLayout):
        self.W = W
        self.layout = layout
        self.row_std = _row_std(W)
        self.n_eff, self.t_eff, self.minus_one = layout.scaling(self.row_std)
        self.bounds = W.spectrum()["bounds"]

    def _logdet(self, a: float) -> float:
        if a == 0.0:
            return 0.0
        ld = self.W.logdet(a)
        if self.minus_one:
            ld -= np.log(1.0 - a)
        return ld

    def _transform(self, d: Design, rho: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
        yr = d.y - rho * d.Wy if rho else d.y
        if lam:
            return yr - lam * self.W.lag(yr), d.X - lam * self.W.lag(d.X)
        return yr, d.X

    def conc_ll(self, d: Design, rho: float, lam: float) -> float:
        yt, Xt = self._transform(d, rho, lam)
        b = _lstsq(Xt, yt)
        e = yt - Xt @ b
        s2 = float(e @ e) / self.n_eff
        if s2 <= 0:
            return -1e300
        return -0.5 * self.n_eff * (np.log(2 * np.pi) + np.log(s2) + 1.0) + self.t_eff * (self._logdet(rho) + self._logdet(lam))

    def full_ll(self, d: Design, beta: np.ndarray, rho: float, lam: float, s2: float) -> float:
        if s2 <= 0:
            return -1e300
        yt, Xt = self._transform(d, rho, lam)
        e = yt - Xt @ beta
        return (-0.5 * self.n_eff * (np.log(2 * np.pi) + np.log(s2)) + self.t_eff * (self._logdet(rho) + self._logdet(lam))
                - 0.5 * float(e @ e) / s2)

    def fit(self, d: Design, model: str, start: Optional[Dict[str, float]] = None) -> FitResult:
        has_lag, has_err, _ = MODEL_PARTS[model]
        lo, hi = self.bounds
        rho = lam = 0.0
        if has_lag and has_err:
            cands = [(0.0, 0.0)]
            if start:
                cands.append((start.get("rho", 0.0), start.get("lam", 0.0)))
            cands += [(0.3, 0.0), (0.0, 0.3), (0.3, 0.3), (-0.3, 0.3), (0.3, -0.3)]
            best = None
            for r0, l0 in cands:
                r0 = float(np.clip(r0, lo + 1e-3, hi - 1e-3)); l0 = float(np.clip(l0, lo + 1e-3, hi - 1e-3))
                res = optimize.minimize(lambda v: -self.conc_ll(d, v[0], v[1]), x0=[r0, l0], method="L-BFGS-B",
                                        bounds=[(lo, hi), (lo, hi)], options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 500})
                if best is None or res.fun < best.fun:
                    best = res
            rho, lam = float(best.x[0]), float(best.x[1])
        elif has_lag:
            res = optimize.minimize_scalar(lambda r: -self.conc_ll(d, r, 0.0), bounds=(lo, hi), method="bounded",
                                           options={"xatol": 1e-10})
            rho = float(res.x)
        elif has_err:
            res = optimize.minimize_scalar(lambda l: -self.conc_ll(d, 0.0, l), bounds=(lo, hi), method="bounded",
                                           options={"xatol": 1e-10})
            lam = float(res.x)
        yt, Xt = self._transform(d, rho, lam)
        beta = _lstsq(Xt, yt)
        e = yt - Xt @ beta
        s2 = float(e @ e) / self.n_eff
        ll = self.full_ll(d, beta, rho, lam, s2)

        # observed information by numerical Hessian (steps on the standard-error scale)
        k = len(beta)
        theta = np.concatenate([beta, [rho] if has_lag else [], [lam] if has_err else [], [s2]])
        xtx = np.einsum("ij,ij->j", Xt, Xt) if k else np.zeros(0)
        se_b = np.sqrt(s2 / np.maximum(xtx, 1e-300))
        steps = np.concatenate([np.maximum(se_b, 1e-12) * 0.01,
                                [0.002] * (int(has_lag) + int(has_err)),
                                [s2 * np.sqrt(2.0 / self.n_eff) * 0.01]])

        def unpack(t):
            b = t[:k]; pos = k
            r = t[pos] if has_lag else 0.0; pos += int(has_lag)
            l_ = t[pos] if has_err else 0.0; pos += int(has_err)
            return b, r, l_, t[pos]

        H = _num_hessian(lambda t: self.full_ll(d, *unpack(t)), theta, steps)
        V, ok = _safe_inv(-H)
        names = list(d.names) + (["ρ (W·y)"] if has_lag else []) + (["λ (W·u)"] if has_err else [])
        p = len(names)
        V = V[:p, :p]

        # fitted values on the estimation scale: reduced form A^-1 X beta (per period)
        Xb = Xt @ beta if not has_err else d.X @ beta
        fitted = self._reduced_form(Xb, rho) if has_lag else Xb
        resid_struct = (d.y - rho * d.Wy - d.X @ beta)
        innov = resid_struct - lam * self.W.lag(resid_struct) if has_err else resid_struct
        r = np.corrcoef(d.y, fitted)[0, 1] if np.std(fitted) > 0 and np.std(d.y) > 0 else 0.0
        return FitResult(
            model=model, names=names, coef=np.concatenate([beta, [rho] if has_lag else [], [lam] if has_err else []]),
            vcov=V, rho=rho if has_lag else None, lam=lam if has_err else None, sigma2=s2, loglik=ll,
            n_obs=len(d.y), n_params=p + 1, pseudo_r2=float(r ** 2), resid=innov, fitted=fitted,
            durbin=list(d.durbin), base_names=list(d.base_names), hessian_ok=ok,
            extra={"n_eff": self.n_eff, "dropped": d.dropped},
        )

    def _reduced_form(self, v: np.ndarray, rho: float) -> np.ndarray:
        """(I - rho W)^-1 v period by period, via the eigen structure when available, else a dense solve."""
        N = self.W.n
        A = np.eye(N) - rho * self.W.dense()
        M = v.reshape(-1, N).T
        return np.linalg.solve(A, M).T.reshape(-1)


def fit_ols(d: Design, layout: PanelLayout, n_eff: Optional[float] = None) -> FitResult:
    """Non-spatial OLS on the (transformed) design with classical standard errors.
    The log-likelihood uses the same effective sample size as the spatial models, so LR tests are comparable."""
    X, y = d.X, d.y
    n, k = X.shape
    n_ll = float(n_eff) if n_eff else float(n)
    beta = _lstsq(X, y)
    e = y - X @ beta
    absorbed = {"none": 0, "ind": layout.N, "time": layout.T, "both": layout.N + layout.T - 1}[layout.fe if not layout.re else "none"]
    dof = max(n - k - absorbed, 1)
    s2 = float(e @ e) / dof
    XtX_inv = np.linalg.pinv(X.T @ X)
    V = s2 * XtX_inv
    s2_ml = float(e @ e) / n_ll
    ll = -0.5 * n_ll * (np.log(2 * np.pi) + np.log(s2_ml) + 1.0)
    ss_tot = float(((y - y.mean()) ** 2).sum()) if "(절편)" in d.names else float((y ** 2).sum())
    r2 = 1.0 - float(e @ e) / ss_tot if ss_tot > 0 else 0.0
    res = FitResult("ols", list(d.names), beta, V, None, None, s2_ml, ll, n, k + 1, r2, e, X @ beta,
                    durbin=list(d.durbin), base_names=list(d.base_names),
                    extra={"dof": dof, "s2_unbiased": s2, "r2": r2, "XtX_inv": XtX_inv, "dropped": d.dropped})
    return res


# ---------------------------------------------------------------- random effects (SAR / SDM)
def fit_random_effects(y: np.ndarray, X: np.ndarray, names: Sequence[str], W: SpatialWeights, N: int, T: int,
                       durbin: Sequence[str], lag: bool) -> FitResult:
    layout = PanelLayout(N, T, fe="none", re=True)
    d = build_design(y, X, names, W, layout, durbin, intercept=True)
    lo, hi = W.spectrum()["bounds"]

    def qd(v: np.ndarray, phi: float) -> np.ndarray:
        c = 1.0 - np.sqrt(phi)
        if v.ndim == 1:
            M = v.reshape(T, N)
            return (M - c * M.mean(axis=0, keepdims=True)).reshape(-1)
        return np.column_stack([qd(v[:, j], phi) for j in range(v.shape[1])])

    def parts(rho, phi):
        yt = qd(d.y - rho * d.Wy, phi)
        Xt = qd(d.X, phi)
        return yt, Xt

    def conc(v):
        rho, phi = (v[0], v[1]) if lag else (0.0, v[0])
        yt, Xt = parts(rho, phi)
        b = _lstsq(Xt, yt)
        e = yt - Xt @ b
        s2 = float(e @ e) / (N * T)
        return -(-0.5 * N * T * (np.log(2 * np.pi) + np.log(s2) + 1) + T * W.logdet(rho) + 0.5 * N * np.log(phi))

    best = None
    starts = [(0.0, 0.5), (0.3, 0.2), (0.3, 0.8), (0.1, 0.05)] if lag else [(0.5,), (0.1,), (0.9,)]
    bnds = [(lo, hi), (1e-6, 1.0)] if lag else [(1e-6, 1.0)]
    for s in starts:
        res = optimize.minimize(conc, x0=list(s), method="L-BFGS-B", bounds=bnds, options={"ftol": 1e-13, "gtol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    rho, phi = (float(best.x[0]), float(best.x[1])) if lag else (0.0, float(best.x[0]))
    yt, Xt = parts(rho, phi)
    beta = _lstsq(Xt, yt)
    e = yt - Xt @ beta
    s2 = float(e @ e) / (N * T)
    k = len(beta)

    def full(t):
        b = t[:k]
        r = t[k] if lag else 0.0
        ph = t[k + int(lag)]
        sg = t[k + int(lag) + 1]
        if sg <= 0 or ph <= 0 or ph > 1.5:
            return -1e300
        yy, XX = parts(r, ph)
        ee = yy - XX @ b
        return -0.5 * N * T * (np.log(2 * np.pi) + np.log(sg)) + T * W.logdet(r) + 0.5 * N * np.log(ph) - 0.5 * float(ee @ ee) / sg

    theta = np.concatenate([beta, [rho] if lag else [], [phi, s2]])
    xtx = np.einsum("ij,ij->j", Xt, Xt)
    steps = np.concatenate([np.sqrt(s2 / np.maximum(xtx, 1e-300)) * 0.01, [0.002] if lag else [],
                            [max(phi, 1e-3) * 0.01], [s2 * np.sqrt(2.0 / (N * T)) * 0.01]])
    H = _num_hessian(full, theta, steps)
    V, ok = _safe_inv(-H)
    names_out = list(d.names) + (["ρ (W·y)"] if lag else [])
    p = len(names_out)
    ll = full(theta)
    Xb = Xt @ beta
    fitted = SpatialML(W, layout)._reduced_form(Xb, rho) if lag else Xb
    r = np.corrcoef(yt, fitted)[0, 1] if np.std(fitted) > 0 else 0.0
    sigma_mu2 = s2 * (1.0 - phi) / (T * phi) if phi > 0 else np.nan
    return FitResult(
        model="sdm" if d.durbin else ("sar" if lag else "ols"), names=names_out,
        coef=np.concatenate([beta, [rho] if lag else []]), vcov=V[:p, :p], rho=rho if lag else None, lam=None,
        sigma2=s2, loglik=ll, n_obs=N * T, n_params=p + 2, pseudo_r2=float(r ** 2),
        resid=yt - Xt @ beta, fitted=fitted, durbin=list(d.durbin),
        base_names=list(d.base_names), hessian_ok=ok,
        extra={"phi": phi, "sigma_mu2": sigma_mu2, "se_phi": float(np.sqrt(max(V[p, p], 0))) if V.shape[0] > p else np.nan,
               "dropped": d.dropped},
    )


# ---------------------------------------------------------------- effects
def effects(fit: FitResult, W: SpatialWeights, method: str = "delta", nsim: int = 1000, seed: int = 42,
            level: float = 0.95) -> List[Dict[str, Any]]:
    """Direct / indirect / total effects of each explanatory variable (LeSage & Pace 2009)."""
    spec = W.spectrum()
    om, a = spec["omega"], spec["a"]
    n = W.n
    has_rho = fit.rho is not None
    base = [v for v in fit.base_names if v in fit.names]
    idx_rho = fit.names.index("ρ (W·y)") if has_rho else None

    def mult(r):
        f0 = 1.0 / (1.0 - r * om)
        m0 = float(np.sum(f0).real) / n
        m1 = float(np.sum(om * f0).real) / n
        t0 = float(np.sum(a * f0).real) / n
        t1 = float(np.sum(a * om * f0).real) / n
        f2 = f0 ** 2
        d_m0 = float(np.sum(om * f2).real) / n
        d_m1 = float(np.sum(om ** 2 * f2).real) / n
        d_t0 = float(np.sum(a * om * f2).real) / n
        d_t1 = float(np.sum(a * om ** 2 * f2).real) / n
        return m0, m1, t0, t1, d_m0, d_m1, d_t0, d_t1

    rho_hat = fit.rho if has_rho else 0.0
    m0, m1, t0, t1, dm0, dm1, dt0, dt1 = mult(rho_hat)
    zcrit = stats.norm.ppf(0.5 + level / 2.0)
    out = []
    rng = np.random.default_rng(seed)
    for v in base:
        ib = fit.names.index(v)
        it = fit.names.index(f"W·{v}") if f"W·{v}" in fit.names else None
        b = fit.coef[ib]
        th = fit.coef[it] if it is not None else 0.0
        est = {"직접효과": b * m0 + th * m1, "총효과": b * t0 + th * t1}
        est["간접효과"] = est["총효과"] - est["직접효과"]
        idx = [ib] + ([it] if it is not None else []) + ([idx_rho] if has_rho else [])
        Vs = fit.vcov[np.ix_(idx, idx)]
        ses: Dict[str, float] = {}
        if method == "sim":
            draws = rng.multivariate_normal(fit.coef[idx], Vs, size=int(nsim), check_valid="ignore")
            bd = draws[:, 0]
            td = draws[:, 1] if it is not None else np.zeros(len(draws))
            rd = draws[:, -1] if has_rho else np.zeros(len(draws))
            rd = np.clip(rd, *spec["bounds"])
            sims = np.array([mult(r)[:4] for r in rd])
            dir_s = bd * sims[:, 0] + td * sims[:, 1]
            tot_s = bd * sims[:, 2] + td * sims[:, 3]
            for key, arr in (("직접효과", dir_s), ("간접효과", tot_s - dir_s), ("총효과", tot_s)):
                ses[key] = float(np.std(arr, ddof=1))
        else:
            g_dir = [m0] + ([m1] if it is not None else []) + ([b * dm0 + th * dm1] if has_rho else [])
            g_tot = [t0] + ([t1] if it is not None else []) + ([b * dt0 + th * dt1] if has_rho else [])
            g_ind = list(np.array(g_tot) - np.array(g_dir))
            for key, g in (("직접효과", g_dir), ("간접효과", g_ind), ("총효과", g_tot)):
                g = np.asarray(g)
                ses[key] = float(np.sqrt(max(g @ Vs @ g, 0.0)))
        for key in ("직접효과", "간접효과", "총효과"):
            e_, s_ = float(est[key]), ses[key]
            z = e_ / s_ if s_ > 0 else np.nan
            out.append({"effect": key, "var": v, "est": e_, "se": s_, "z": z,
                        "p": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
                        "lo": e_ - zcrit * s_, "hi": e_ + zcrit * s_})
    order = {"직접효과": 0, "간접효과": 1, "총효과": 2}
    out.sort(key=lambda r: (order[r["effect"]], base.index(r["var"])))
    return out


# ---------------------------------------------------------------- diagnostics
def lm_tests(d: Design, W: SpatialWeights, T: int) -> Dict[str, Tuple[float, float]]:
    """LM tests on OLS residuals of the (transformed) design (Anselin et al. 1996; Elhorst 2014 for panels)."""
    X, y = d.X, d.y
    beta = _lstsq(X, y)
    e = y - X @ beta
    n = len(y)
    s2 = float(e @ e) / n
    Wd = W.W
    tW = float((Wd.T @ Wd).diagonal().sum() + (Wd @ Wd).diagonal().sum())
    TT = T * tW
    We = W.lag(e)
    WXb = W.lag(X @ beta)
    if X.shape[1]:
        M_WXb = WXb - X @ _lstsq(X, WXb)
    else:
        M_WXb = WXb
    J = (float(M_WXb @ M_WXb) + TT * s2) / s2
    r_err = float(e @ We) / s2
    r_lag = float(e @ d.Wy) / s2
    lm_err = r_err ** 2 / TT
    lm_lag = r_lag ** 2 / J
    rlm_lag = (r_lag - r_err) ** 2 / (J - TT) if J - TT > 0 else np.nan
    rlm_err = (r_err - TT / J * r_lag) ** 2 / (TT * (1 - TT / J)) if TT * (1 - TT / J) > 0 else np.nan
    out = {"LM 공간오차": lm_err, "강건 LM 공간오차": rlm_err, "LM 공간시차": lm_lag, "강건 LM 공간시차": rlm_lag,
           "LM 혼합(SARMA)": lm_err + rlm_lag}
    return {k: (v, float(stats.chi2.sf(v, 2 if "SARMA" in k else 1)) if np.isfinite(v) else np.nan) for k, v in out.items()}


def moran_residuals(e: np.ndarray, X: np.ndarray, W: SpatialWeights) -> Dict[str, float]:
    """Moran's I of OLS residuals with the exact normal-approximation moments (Cliff & Ord 1981)."""
    Wd = W.W
    n, k = X.shape
    S0 = float(Wd.sum())
    I = (n / S0) * float(e @ (Wd @ e)) / float(e @ e)
    A = np.linalg.pinv(X.T @ X)
    WX = Wd @ X
    WtX = Wd.T @ X
    trW = float(Wd.diagonal().sum())
    tr_MW = trW - np.trace(A @ (X.T @ WX))
    tr_WWt = float((Wd.multiply(Wd)).sum())
    tr_PWWt = np.trace(A @ (WtX.T @ WtX))
    tr_PWtW = np.trace(A @ (WX.T @ WX))
    XWX = X.T @ WX
    XWtX = X.T @ WtX
    tr_PWPWt = np.trace(A @ XWX @ A @ XWtX)
    tr_MWMWt = tr_WWt - tr_PWWt - tr_PWtW + tr_PWPWt
    tr_W2 = float((Wd @ Wd).diagonal().sum())
    W2X = Wd @ WX
    tr_MWMW = tr_W2 - 2 * np.trace(A @ (X.T @ W2X)) + np.trace(A @ XWX @ A @ XWX)
    EI = (n / S0) * tr_MW / (n - k)
    VI = (n / S0) ** 2 * (tr_MWMWt + tr_MWMW + tr_MW ** 2) / ((n - k) * (n - k + 2)) - EI ** 2
    z = (I - EI) / np.sqrt(VI) if VI > 0 else np.nan
    return {"I": I, "E": EI, "V": VI, "z": z, "p": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan}


def ols_diagnostics(fit: FitResult, d: Design) -> Dict[str, Tuple[float, float, float]]:
    """Jarque-Bera, Koenker-Bassett (studentised Breusch-Pagan) and multicollinearity condition number."""
    e = fit.resid
    n = len(e)
    m2 = np.mean(e ** 2)
    skew = np.mean(e ** 3) / m2 ** 1.5
    kurt = np.mean(e ** 4) / m2 ** 2
    jb = n / 6.0 * (skew ** 2 + (kurt - 3) ** 2 / 4.0)
    X = d.X
    Z = X[:, [j for j, nm in enumerate(d.names) if nm != "(절편)"]]
    out = {"Jarque–Bera 정규성": (jb, 2, float(stats.chi2.sf(jb, 2)))}
    if Z.shape[1]:
        u = e ** 2
        Zc = np.column_stack([np.ones(n), Z])
        g = _lstsq(Zc, u)
        fitted = Zc @ g
        kb = n * (1 - float(((u - fitted) ** 2).sum()) / float(((u - u.mean()) ** 2).sum()))
        out["Koenker–Bassett 이분산"] = (kb, Z.shape[1], float(stats.chi2.sf(kb, Z.shape[1])))
        Xn = X / np.sqrt((X ** 2).sum(axis=0))
        sv = np.linalg.svd(Xn, compute_uv=False)
        out["다중공선성 조건수"] = (float(sv.max() / sv.min()) if sv.min() > 0 else np.inf, np.nan, np.nan)
    return out


def lr_test(restricted: FitResult, unrestricted: FitResult) -> Tuple[float, int, float]:
    lr = max(0.0, 2.0 * (unrestricted.loglik - restricted.loglik))
    df = max(1, unrestricted.n_params - restricted.n_params)
    return lr, df, float(stats.chi2.sf(lr, df))


def hausman(fe: FitResult, re: FitResult) -> Dict[str, float]:
    common = [nm for nm in fe.names if nm in re.names and nm != "(절편)"]
    i1 = [fe.names.index(c) for c in common]
    i2 = [re.names.index(c) for c in common]
    q = fe.coef[i1] - re.coef[i2]
    Vd = fe.vcov[np.ix_(i1, i1)] - re.vcov[np.ix_(i2, i2)]
    stat = float(q @ np.linalg.pinv(Vd) @ q)
    pd_ok = bool(np.all(np.linalg.eigvalsh((Vd + Vd.T) / 2) > -1e-12))
    return {"stat": abs(stat), "df": len(common), "p": float(stats.chi2.sf(abs(stat), len(common))), "pd": pd_ok,
            "vars": common}


def fit_spec(model: str, y, X, names, W: SpatialWeights, layout: PanelLayout, durbin: Sequence[str],
             intercept: bool) -> Tuple[FitResult, Design]:
    _, _, wx = MODEL_PARTS[model]
    d = build_design(y, X, names, W, layout, list(durbin) if wx else [], intercept)
    if model in ("ols", "slx"):
        r = fit_ols(d, layout, n_eff=layout.scaling(_row_std(W))[0])
        r.model = model
        return r, d
    return SpatialML(W, layout).fit(d, model), d
