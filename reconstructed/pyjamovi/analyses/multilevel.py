"""
OpenStat Studio — multilevel (mixed-effects) models with up to four nested levels (Stata mixed · melogit · mepoisson).
License: MIT

Computation
- Linear mixed models: statsmodels MixedLM (library core). The top grouping level is `groups` with its random
  intercept / slopes in `exog_re` (unstructured covariance, or independent through the `free` mask); every lower
  nested level enters as variance components (`exog_vc`, one independent variance per term), which is Stata's
  default covariance(independent) for nested levels. ML or REML.
- Glue on top of statsmodels (Stata reporting conventions): (X′V̂⁻¹X)⁻¹ standard errors from the statsmodels
  Hessian, variance-component SEs from the numerical Hessian of statsmodels' (restricted) log likelihood in the
  log-SD / atanh-correlation / log-σ_e metric with β profiled out (Stata mixed, Methods and formulas), confidence
  intervals on that scale, LR test against linear regression, estat icc, BLUP tables, Stata/R code.
- Own implementation (no library found; searched statsmodels MixedLM / BinomialBayesMixedGLM / PoissonBayesMixedGLM,
  pymer4 (needs R), bambi (Bayesian)): (a) linear models with an identity or exchangeable structure at the top level
  or a correlated (unstructured / exchangeable / identity) structure at a nested level — profiled ML/REML deviance in
  the lme4 formulation (Bates, Mächler, Bolker & Walker 2015) with a tree-structured Cholesky elimination;
  (b) generalized models (Bernoulli-logit, Poisson-log) by the Laplace approximation (Stata intmethod(laplace),
  lme4 glmer nAGQ = 1). Both are labelled "자체 구현 추정" in the output and were checked against Stata 19.5 and lme4.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .base import BaseAnalysis, ResultsTable, resolve_choice

COV_STRUCTS = ("independent", "unstructured", "identity", "exchangeable")
_COV_ALIASES = {"ind": "independent", "indep": "independent", "diag": "independent", "diagonal": "independent",
                "un": "unstructured", "uns": "unstructured", "unstr": "unstructured", "full": "unstructured",
                "id": "identity", "ide": "identity", "iden": "identity", "ex": "exchangeable", "exc": "exchangeable",
                "exch": "exchangeable", "cs": "exchangeable", "compound": "exchangeable"}
COV_LABEL = {"independent": "독립 (Independent)", "unstructured": "비구조 (Unstructured)",
             "identity": "동일 (Identity)", "exchangeable": "교환 (Exchangeable)"}
LOG2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------------------------------------------
class _Level:
    """One grouping level: random-effect design Z (n × q), covariance structure, nested group index."""

    def __init__(self, name: str, terms: List[str], Z: np.ndarray, cov: str, codes: np.ndarray, G: int,
                 parent: Optional[np.ndarray]):
        self.name = name
        self.terms = list(terms)            # Stata order: slopes ..., "_cons"
        self.Z = np.asarray(Z, dtype=float)
        self.q = self.Z.shape[1]
        self.cov = cov if self.q > 1 else "identity"
        self.codes = codes                  # (n,) group index, lexicographic in (ancestors, own id)
        self.G = int(G)
        self.parent = parent                # (G,) index of the parent group (None for the top level)

    # --- θ (relative Cholesky factor, lme4 metric) ---
    @property
    def n_theta(self) -> int:
        q = self.q
        return {"unstructured": q * (q + 1) // 2, "independent": q, "identity": 1, "exchangeable": 2}[self.cov]

    def theta_start(self) -> List[float]:
        """lme4 start: identity relative factor."""
        q = self.q
        if self.cov == "unstructured":
            return [1.0 if i == j else 0.0 for j in range(q) for i in range(j, q)]
        if self.cov == "independent":
            return [1.0] * q
        if self.cov == "identity":
            return [1.0]
        return [1.0, 0.0]

    # --- η: unconstrained optimisation metric (log of the diagonal / scale, bounded correlation map) ---
    def eta_start(self, zsd: np.ndarray) -> List[float]:
        """Start with every random effect contributing about the residual variance: relative SD 1/sd(z)."""
        s = np.where(zsd > 1e-12, 1.0 / np.maximum(zsd, 1e-12), 1.0)
        q = self.q
        if self.cov == "unstructured":
            return [math.log(s[i]) if i == j else 0.0 for j in range(q) for i in range(j, q)]
        if self.cov == "independent":
            return list(np.log(s))
        if self.cov == "identity":
            return [float(np.mean(np.log(s)))]
        return [float(np.mean(np.log(s))), 0.0]

    def is_log(self) -> List[bool]:
        q = self.q
        if self.cov == "unstructured":
            return [i == j for j in range(q) for i in range(j, q)]
        if self.cov in ("independent", "identity"):
            return [True] * self.n_theta
        return [True, False]

    def theta_from_eta(self, eta: np.ndarray) -> np.ndarray:
        th = np.array(eta, dtype=float)
        for i, lg in enumerate(self.is_log()):
            if lg:
                th[i] = math.exp(eta[i])
        if self.cov == "exchangeable":
            lo, hi = self.theta_bounds()[1]
            th[1] = lo + (hi - lo) / (1.0 + math.exp(-eta[1]))
        return th

    def theta_bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        q = self.q
        if self.cov == "unstructured":
            return [(0.0, None) if i == j else (None, None) for j in range(q) for i in range(j, q)]
        if self.cov in ("independent", "identity"):
            return [(0.0, None)] * self.n_theta
        return [(0.0, None), (-1.0 / (q - 1) + 1e-6, 1.0 - 1e-6)]

    def lam(self, th: np.ndarray) -> np.ndarray:
        q = self.q
        if self.cov == "unstructured":
            L = np.zeros((q, q))
            k = 0
            for j in range(q):
                for i in range(j, q):
                    L[i, j] = th[k]
                    k += 1
            return L
        if self.cov == "independent":
            return np.diag(np.asarray(th, dtype=float))
        if self.cov == "identity":
            return float(th[0]) * np.eye(q)
        r = float(th[1])
        R = (1.0 - r) * np.eye(q) + r * np.ones((q, q))
        return float(th[0]) * np.linalg.cholesky(R)

    # --- φ (Stata estimation metric: log SD, atanh correlation) ---
    @property
    def n_phi(self) -> int:
        return self.n_theta

    def phi_from_cov(self, S: np.ndarray) -> List[float]:
        q = self.q
        sd = np.sqrt(np.maximum(np.diag(S), 1e-300))
        if self.cov == "unstructured":
            R = S / np.outer(sd, sd)
            return list(np.log(sd)) + [float(np.arctanh(np.clip(R[i, j], -1 + 1e-15, 1 - 1e-15)))
                                       for i in range(q) for j in range(i + 1, q)]
        if self.cov == "independent":
            return list(np.log(sd))
        if self.cov == "identity":
            return [float(np.log(sd[0]))]
        v = float(np.mean(np.diag(S)))
        c = float(np.mean(S[~np.eye(q, dtype=bool)]))
        return [0.5 * math.log(max(v, 1e-300)), float(np.arctanh(np.clip(c / v, -1 + 1e-15, 1 - 1e-15)))]

    def cov_from_phi(self, ph: np.ndarray) -> np.ndarray:
        q = self.q
        if self.cov == "unstructured":
            sd = np.exp(ph[:q])
            R = np.eye(q)
            k = q
            for i in range(q):
                for j in range(i + 1, q):
                    R[i, j] = R[j, i] = math.tanh(ph[k])
                    k += 1
            return R * np.outer(sd, sd)
        if self.cov == "independent":
            return np.diag(np.exp(2.0 * np.asarray(ph)))
        if self.cov == "identity":
            return math.exp(2.0 * ph[0]) * np.eye(q)
        v = math.exp(2.0 * ph[0])
        r = math.tanh(ph[1])
        return v * ((1.0 - r) * np.eye(q) + r * np.ones((q, q)))


def _chol_inv(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Batched Cholesky factor L (lower) and L⁻¹ of SPD stacks (G, q, q); closed form for q ≤ 2."""
    q = A.shape[1]
    if q == 1:
        Lc = np.sqrt(A)
        return Lc, 1.0 / Lc
    if q == 2:
        a = np.sqrt(A[:, 0, 0])
        b = A[:, 1, 0] / a
        c = np.sqrt(A[:, 1, 1] - b * b)
        G = A.shape[0]
        Lc = np.zeros((G, 2, 2))
        Li = np.zeros((G, 2, 2))
        Lc[:, 0, 0], Lc[:, 1, 0], Lc[:, 1, 1] = a, b, c
        Li[:, 0, 0], Li[:, 1, 0], Li[:, 1, 1] = 1.0 / a, -b / (a * c), 1.0 / c
        return Lc, Li
    Lc = np.linalg.cholesky(A)
    return Lc, np.linalg.inv(Lc)


class _LMM:
    """Profiled-deviance linear mixed model for nested grouping levels (top level first)."""

    def __init__(self, y: np.ndarray, X: np.ndarray, levels: List[_Level]):
        self.y = np.asarray(y, dtype=float)
        self.X = np.asarray(X, dtype=float)
        self.n, self.p = self.X.shape
        self.levels = levels
        self.K = len(levels)
        leaf = levels[-1]
        order = np.argsort(leaf.codes, kind="stable")
        codes = leaf.codes[order]
        starts = np.flatnonzero(np.r_[True, np.diff(codes) != 0])
        # W_z columns: leaf level first, then its parent, ..., top level (own block first at every level)
        Wz = np.hstack([lev.Z for lev in reversed(levels)])[order]
        Wxy = np.hstack([self.X, self.y[:, None]])[order]
        Q = Wz.shape[1]
        self.Q = Q
        self.ZZraw = np.empty((leaf.G, Q, Q))
        self.ZXraw = np.empty((leaf.G, Q, self.p + 1))
        for a in range(Q):
            self.ZZraw[:, a, :] = np.add.reduceat(Wz[:, a:a + 1] * Wz, starts, axis=0)
            self.ZXraw[:, a, :] = np.add.reduceat(Wz[:, a:a + 1] * Wxy, starts, axis=0)
        self.XXraw = Wxy.T @ Wxy
        # reduceat start indices of the children of each parent group (levels are lexicographically ordered)
        self.child_starts: List[Optional[np.ndarray]] = [None]
        for k in range(1, self.K):
            par = levels[k].parent
            if np.any(np.diff(par) < 0):
                raise RuntimeError("internal: groups not ordered by parent")
            self.child_starts.append(np.flatnonzero(np.r_[True, np.diff(par) != 0]))
        self.theta_slices = []
        pos = 0
        for lev in levels:
            self.theta_slices.append(slice(pos, pos + lev.n_theta))
            pos += lev.n_theta
        self.n_theta = pos
        self.nfev = 0

    # ---------------- penalized least squares for given relative factors ----------------
    def lams(self, theta: np.ndarray) -> List[np.ndarray]:
        return [lev.lam(theta[s]) for lev, s in zip(self.levels, self.theta_slices)]

    def pls(self, lams: List[np.ndarray], store: bool = False) -> Dict[str, Any]:
        p = self.p
        T = np.zeros((self.Q, self.Q))
        pos = 0
        for lev, L in zip(reversed(self.levels), reversed(lams)):
            T[pos:pos + lev.q, pos:pos + lev.q] = L
            pos += lev.q
        ZZ = np.matmul(np.matmul(T.T, self.ZZraw), T)
        ZX = np.matmul(T.T, self.ZXraw)
        F = self.XXraw.copy()
        logdet = 0.0
        saved = [None] * self.K
        for k in range(self.K - 1, -1, -1):
            q = self.levels[k].q
            A = ZZ[:, :q, :q] + np.eye(q)
            Lc, Linv = _chol_inv(A)
            Ct = np.matmul(Linv, ZX[:, :q, :])
            F -= np.einsum("gij,gik->jk", Ct, Ct)
            logdet += 2.0 * float(np.sum(np.log(np.diagonal(Lc, axis1=1, axis2=2))))
            Bt = None
            if k > 0:
                Bt = np.matmul(Linv, ZZ[:, :q, q:])
                D = ZZ[:, q:, q:] - np.matmul(np.transpose(Bt, (0, 2, 1)), Bt)
                E = ZX[:, q:, :] - np.matmul(np.transpose(Bt, (0, 2, 1)), Ct)
                cs = self.child_starts[k]
                ZZ = np.add.reduceat(D, cs, axis=0)
                ZX = np.add.reduceat(E, cs, axis=0)
            if store:
                saved[k] = {"Linv": Linv, "Ct": Ct, "Bt": Bt}
        Fxx, Fxy, Fyy = F[:p, :p], F[:p, p], float(F[p, p])
        try:
            Rx = np.linalg.cholesky(Fxx)
        except np.linalg.LinAlgError:
            return {"ok": False}
        beta = np.linalg.solve(Rx.T, np.linalg.solve(Rx, Fxy))
        r2 = max(Fyy - float(Fxy @ beta), 1e-300)
        out = {"ok": True, "logdet": logdet, "logdetX": 2.0 * float(np.sum(np.log(np.diag(Rx)))),
               "beta": beta, "r2": r2, "Fxx": Fxx}
        if store:
            out["saved"] = saved
        return out

    def deviance(self, theta: np.ndarray, reml: bool) -> float:
        self.nfev += 1
        r = self.pls(self.lams(np.asarray(theta, dtype=float)))
        if not r["ok"]:
            return np.inf
        n, p = self.n, self.p
        if reml:
            return r["logdet"] + r["logdetX"] + (n - p) * (1.0 + math.log(2.0 * math.pi * r["r2"] / (n - p)))
        return r["logdet"] + n * (1.0 + math.log(2.0 * math.pi * r["r2"] / n))

    # ---------------- optimisation ----------------
    def _grad(self, f, x: np.ndarray, lb: np.ndarray, h: float = 1e-6) -> np.ndarray:
        g = np.zeros_like(x)
        for i in range(len(x)):
            hi = h * max(1.0, abs(x[i]))
            if x[i] - hi < lb[i]:
                e = np.zeros_like(x)
                e[i] = hi
                g[i] = (-3.0 * f(x) + 4.0 * f(x + e) - f(x + 2 * e)) / (2.0 * hi)
            else:
                e = np.zeros_like(x)
                e[i] = hi
                g[i] = (f(x + e) - f(x - e)) / (2.0 * hi)
        return g

    def _theta_from_eta(self, eta: np.ndarray) -> np.ndarray:
        return np.concatenate([lev.theta_from_eta(eta[s]) for lev, s in zip(self.levels, self.theta_slices)])

    def _bounded(self, f, x0, lb, ub, bounds, max_iter):
        res = optimize.minimize(f, x0, jac=lambda th: self._grad(f, th, lb), method="L-BFGS-B", bounds=bounds,
                                options={"maxiter": max_iter, "ftol": 1e-15, "gtol": 1e-10, "maxls": 50})
        th = np.clip(res.x, lb, ub)
        fval = f(th)
        # Newton polish on the parameters away from their bounds (numerical Hessian, step halving)
        for _ in range(10):
            free = np.flatnonzero((th - lb > 1e-7) & (ub - th > 1e-7))
            if not len(free):
                break
            g = self._grad(f, th, lb)[free]
            H = self._hess(f, th, free)
            try:
                step = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(step)) or g @ step >= 0:
                break
            t, improved = 1.0, False
            while t > 1e-4:
                cand = th.copy()
                cand[free] = np.clip(th[free] + t * step, lb[free], ub[free])
                fc = f(cand)
                if fc <= fval:
                    improved = fc < fval - 1e-13
                    th, fval = cand, fc
                    break
                t *= 0.5
            if not improved or np.max(np.abs(t * step)) < 1e-10:
                break
        return th, float(fval), res

    def fit(self, reml: bool, max_iter: int = 2000) -> Dict[str, Any]:
        """Two phases: unconstrained log-Cholesky search from a scale-aware start, then the bounded lme4 metric
        (diagonal ≥ 0, so a variance can reach exactly zero) with a Newton polish; the lme4 identity start is
        also tried and the lower deviance kept."""
        bounds = [b for lev in self.levels for b in lev.theta_bounds()]
        lb = np.array([-np.inf if b[0] is None else b[0] for b in bounds])
        ub = np.array([np.inf if b[1] is None else b[1] for b in bounds])
        f = lambda th: self.deviance(th, reml)
        self.nfev = 0
        zsd = [np.std(lev.Z, axis=0) for lev in self.levels]
        for lev, s in zip(self.levels, zsd):       # intercept column (sd 0) keeps relative SD 1
            s[np.all(lev.Z == 1.0, axis=0)] = 1.0
        eta0 = np.concatenate([lev.eta_start(s) for lev, s in zip(self.levels, zsd)])
        is_log = np.array([v for lev in self.levels for v in lev.is_log()])
        e_lb = np.where(is_log, -25.0, -np.inf)
        e_bounds = [(-25.0, 15.0) if lg else (None, None) for lg in is_log]
        fe = lambda e: f(self._theta_from_eta(e))

        def fwd_grad(e):
            f0 = fe(e)
            g = np.zeros_like(e)
            for i in range(len(e)):
                hi = 1e-7 * max(1.0, abs(e[i]))
                x = e.copy()
                x[i] += hi
                g[i] = (fe(x) - f0) / hi
            return g

        r1 = optimize.minimize(fe, eta0, jac=fwd_grad, method="L-BFGS-B", bounds=e_bounds,
                               options={"maxiter": max_iter, "ftol": 1e-11, "gtol": 1e-6, "maxls": 50})
        th1 = self._theta_from_eta(r1.x)
        th1 = np.where(is_log & (th1 < 1e-6), 0.0, th1)
        th1 = np.clip(th1, lb, ub)
        best = self._bounded(f, th1, lb, ub, bounds, max_iter)
        at_lb = np.any(is_log & (best[0] - lb <= 1e-7))
        g_best = self._grad(f, best[0], lb)
        free_b = best[0] - lb > 1e-7
        gn_best = float(np.max(np.abs(np.where(free_b, g_best, np.minimum(g_best, 0.0))))) if len(g_best) else 0.0
        if at_lb or gn_best > 1e-3:
            # safeguard for boundary solutions: also start from the lme4 identity factor and keep the lower deviance
            th0 = np.array([v for lev in self.levels for v in lev.theta_start()], dtype=float)
            alt = self._bounded(f, th0, lb, ub, bounds, max_iter)
            if alt[1] < best[1] - 1e-9:
                best = alt
        th, fval, res = best
        grad = self._grad(f, th, lb)
        free = (th - lb > 1e-7)
        gnorm = float(np.max(np.abs(np.where(free, grad, np.minimum(grad, 0.0))))) if len(th) else 0.0
        return {"theta": th, "deviance": float(fval), "converged": bool(gnorm < 1e-2),
                "message": str(res.message), "nit": int(res.nit + r1.nit), "nfev": int(self.nfev), "gnorm": gnorm,
                "at_bound": [bool(th[i] - lb[i] <= 1e-7) for i in range(len(th))]}

    def _hess(self, f, x: np.ndarray, idx: np.ndarray, h: float = 1e-4) -> np.ndarray:
        m = len(idx)
        H = np.zeros((m, m))
        f0 = f(x)
        hs = [h * max(1.0, abs(x[i])) for i in idx]
        for a in range(m):
            ea = np.zeros_like(x)
            ea[idx[a]] = hs[a]
            H[a, a] = (f(x + ea) - 2.0 * f0 + f(x - ea)) / hs[a] ** 2
            for b in range(a + 1, m):
                eb = np.zeros_like(x)
                eb[idx[b]] = hs[b]
                H[a, b] = H[b, a] = (f(x + ea + eb) - f(x + ea - eb) - f(x - ea + eb) + f(x - ea - eb)) / (4.0 * hs[a] * hs[b])
        return H

    # ---------------- log likelihood with explicit σ (Stata metric) ----------------
    def loglik_phi(self, phi: np.ndarray, reml: bool) -> float:
        """(Restricted) log likelihood at φ = (per level: log SDs, atanh correlations; then log σ_e), β profiled."""
        sig = math.exp(phi[-1])
        lams, pos = [], 0
        for lev in self.levels:
            S = lev.cov_from_phi(phi[pos:pos + lev.n_phi])
            pos += lev.n_phi
            if lev.cov == "independent":
                lams.append(np.diag(np.sqrt(np.diag(S))) / sig)
            elif lev.cov == "identity":
                lams.append(math.sqrt(S[0, 0]) / sig * np.eye(lev.q))
            else:
                try:
                    lams.append(np.linalg.cholesky(S) / sig)
                except np.linalg.LinAlgError:
                    return -np.inf
        r = self.pls(lams)
        if not r["ok"]:
            return -np.inf
        n, p = self.n, self.p
        if reml:
            return -0.5 * (r["logdet"] + r["logdetX"] + (n - p) * (LOG2PI + 2.0 * phi[-1]) + r["r2"] / sig ** 2)
        return -0.5 * (r["logdet"] + n * (LOG2PI + 2.0 * phi[-1]) + r["r2"] / sig ** 2)

    # ---------------- random-effect predictions (conditional modes) and conditional variances ----------------
    def blups(self, lams: List[np.ndarray], beta: np.ndarray, sigma2: float, cond_var: bool = True):
        r = self.pls(lams, store=True)
        saved = r["saved"]
        p = self.p
        rhs_coef = np.r_[-beta, 1.0]
        out_b, out_v = [], []
        path_u = None       # (G_k, Q_k) [u_own, u_parent, ...]
        path_inv = None     # (G_k, Q_k, Q_k)
        for k, lev in enumerate(self.levels):
            s = saved[k]
            Linv, Ct, Bt = s["Linv"], s["Ct"], s["Bt"]
            q = lev.q
            c = Ct @ rhs_coef                                   # (G, q)
            LinvT = np.transpose(Linv, (0, 2, 1))
            if k == 0:
                u = np.einsum("gij,gj->gi", LinvT, c)
                new_path_u = u
                if cond_var:
                    Minv = np.matmul(LinvT, Linv)
                    new_path_inv = Minv
            else:
                par = lev.parent
                pu = path_u[par]                                # (G, Q_{k-1})
                u = np.einsum("gij,gj->gi", LinvT, c - np.einsum("gij,gj->gi", Bt, pu))
                new_path_u = np.hstack([u, pu])
                if cond_var:
                    Pi = path_inv[par]                          # (G, Qa, Qa)
                    BtPi = np.matmul(Bt, Pi)                    # (G, q, Qa)
                    inner = np.eye(q) + np.matmul(BtPi, np.transpose(Bt, (0, 2, 1)))
                    Minv = np.matmul(np.matmul(LinvT, inner), Linv)
                    X_ = -np.matmul(LinvT, BtPi)                # (G, q, Qa)
                    top = np.concatenate([Minv, X_], axis=2)
                    bot = np.concatenate([np.transpose(X_, (0, 2, 1)), Pi], axis=2)
                    new_path_inv = np.concatenate([top, bot], axis=1)
            Lam = lams[k]
            out_b.append(u @ Lam.T)                              # b = Λ u
            if cond_var:
                out_v.append(sigma2 * np.matmul(np.matmul(Lam, Minv), Lam.T))
            path_u = new_path_u
            if cond_var:
                path_inv = new_path_inv
        return out_b, (out_v if cond_var else None)

    # ---------------- variance-component inference in the Stata metric ----------------
    def phi_hat(self, theta: np.ndarray, reml: bool) -> Tuple[np.ndarray, float, List[np.ndarray]]:
        lams = self.lams(theta)
        r = self.pls(lams)
        s2 = r["r2"] / ((self.n - self.p) if reml else self.n)
        covs = [s2 * (L @ L.T) for L in lams]
        phi = []
        for lev, S in zip(self.levels, covs):
            phi += lev.phi_from_cov(S)
        phi.append(0.5 * math.log(s2))
        return np.array(phi), s2, covs



# ---------------------------------------------------------------------------------------------------------------
# statsmodels MixedLM path (library core for the structures it supports)
# ---------------------------------------------------------------------------------------------------------------
def _statsmodels_supports(levs: List[_Level]) -> bool:
    """statsmodels MixedLM: top level = groups with re_formula (unstructured, or independent via `free`);
    nested levels = variance components (one independent variance per term), i.e. Stata cov(independent)."""
    top, nested = levs[0], levs[1:]
    ok_top = top.q == 1 or top.cov in ("unstructured", "independent")
    ok_nested = all(lev.q == 1 or lev.cov == "independent" for lev in nested)
    return ok_top and ok_nested


def _ols_loglik(y: np.ndarray, X: np.ndarray, reml: bool) -> float:
    """Log likelihood of the linear regression without random effects (ML or REML, same constants as mixed)."""
    import statsmodels.api as sm
    ols = sm.OLS(y, X).fit()
    n, p = X.shape
    if not reml:
        return float(ols.llf)
    rss = float(ols.ssr)
    ld = float(np.linalg.slogdet(X.T @ X)[1])
    return -0.5 * ((n - p) * (LOG2PI + 1.0 + math.log(rss / (n - p))) + ld)


class _SMFit:
    """Fit through statsmodels MixedLM and expose what the Stata-style tables need."""

    def __init__(self, y: np.ndarray, X: np.ndarray, levs: List[_Level], reml: bool, max_iter: int = 2000):
        import warnings
        from statsmodels.regression.mixed_linear_model import MixedLM, MixedLMParams, VCSpec
        self.levs, self.reml = levs, reml
        self.n, self.p = X.shape
        top = levs[0]
        g0 = top.codes
        self.rows = [np.flatnonzero(g0 == gi) for gi in range(top.G)]
        names, colnames, mats, self.vc_index = [], [], [], []
        for k in range(1, len(levs)):
            lev = levs[k]
            for j, term in enumerate(lev.terms):
                cn, mt = [], []
                for rows in self.rows:
                    sub = lev.codes[rows]
                    uniq = np.unique(sub)
                    M = (sub[:, None] == uniq[None, :]).astype(float) * lev.Z[rows, j][:, None]
                    mt.append(M)
                    cn.append([str(u) for u in uniq])
                names.append(f"L{k}T{j}")
                colnames.append(cn)
                mats.append(mt)
                self.vc_index.append((k, j))
        vcs = VCSpec(names, colnames, mats) if names else None
        self.model = MixedLM(y, X, groups=g0, exog_re=top.Z, exog_vc=vcs)
        free = None
        if top.q > 1 and top.cov == "independent":
            free = MixedLMParams.from_components(fe_params=np.ones(self.p), cov_re=np.eye(top.q),
                                                 vcomp=np.ones(self.model.k_vc))
        self.free = free
        cands = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for plan in (["bfgs"], ["lbfgs", "bfgs"]):
                try:
                    r = self.model.fit(reml=reml, free=free, method=[plan[0]], gtol=1e-8, maxiter=max_iter)
                    if len(plan) > 1:
                        start = MixedLMParams.from_components(fe_params=np.asarray(r.fe_params),
                                                              cov_re=np.asarray(r.cov_re) / r.scale,
                                                              vcomp=np.asarray(r.vcomp) / r.scale)
                        r = self.model.fit(reml=reml, free=free, method=[plan[1]], gtol=1e-8, maxiter=max_iter,
                                           start_params=start)
                    if np.isfinite(r.llf):
                        cands.append((bool(r.converged), float(r.llf), "+".join(plan), r))
                        if r.converged and not self._at_boundary(r):
                            break           # interior solution from bfgs: the second plan is only a safeguard
                except Exception:          # one optimiser failing is not fatal; the other plan is still tried
                    continue
        if not cands:
            raise RuntimeError("statsmodels MixedLM 추정 실패")
        conv = [c for c in cands if c[0]]
        best = max(conv or cands, key=lambda c: c[1])
        self.converged, self.ll, self.method, self.res = best[0], best[1], best[2], best[3]
        self.model.reml = reml
        r = self.res
        self.beta = np.asarray(r.fe_params, dtype=float)
        self.s2 = float(r.scale)
        H, _ = self.model.hessian(r.params_object)
        self.vcov_b = np.linalg.inv(-H[:self.p, :self.p])           # (X'V⁻¹X)⁻¹ σ̂², as Stata reports it
        cov_top = np.atleast_2d(np.asarray(r.cov_re, dtype=float))
        vcomp = np.asarray(r.vcomp, dtype=float)
        self.covs_abs = [cov_top]
        for k in range(1, len(levs)):
            self.covs_abs.append(np.diag([vcomp[i] for i, (kk, j) in enumerate(self.vc_index) if kk == k]))
        self.n_vc = int(top.n_theta + len(self.vc_index))

    @staticmethod
    def _at_boundary(r) -> bool:
        cr = np.atleast_2d(np.asarray(r.cov_re, dtype=float))
        sd = np.sqrt(np.maximum(np.diag(cr), 0.0)) if cr.size else np.array([])
        vc = np.asarray(r.vcomp, dtype=float)
        lim = 1e-3 * math.sqrt(max(float(r.scale), 1e-300))
        return bool((sd.size and sd.min() < lim) or (vc.size and np.sqrt(np.maximum(vc, 0.0)).min() < lim))

    def phi_hat(self) -> np.ndarray:
        phi = []
        for lev, S in zip(self.levs, self.covs_abs):
            phi += lev.phi_from_cov(S)
        phi.append(0.5 * math.log(self.s2))
        return np.array(phi)

    def loglik_phi(self, phi: np.ndarray) -> float:
        """(Restricted) log likelihood at φ with β profiled out and σ_e explicit, from statsmodels' profile likelihood:
        ℓ(γ, σ²) = ℓ_p(γ) + m/2 (log ŝ²(γ) + 1) − m/2 (log σ² + ŝ²(γ)/σ²), m = n (ML) or n − p (REML)."""
        from statsmodels.regression.mixed_linear_model import MixedLMParams
        s2 = math.exp(2.0 * phi[-1])
        pos = 0
        covs = []
        for lev in self.levs:
            covs.append(lev.cov_from_phi(phi[pos:pos + lev.n_phi]))
            pos += lev.n_phi
        cov_re = covs[0] / s2
        vc = []
        for k, j in self.vc_index:
            vc.append(covs[k][j, j] / s2)
        vc = np.array(vc)
        try:
            prm = MixedLMParams.from_components(fe_params=np.zeros(self.p), cov_re=cov_re, vcomp=vc)
            lp = float(self.model.loglike(prm, profile_fe=True))
            fe, _ = self.model.get_fe_params(cov_re, vc)
            sh = float(self.model.get_scale(fe, cov_re, vc))
        except Exception:
            return -np.inf
        m = (self.n - self.p) if self.reml else self.n
        return lp + 0.5 * m * (math.log(sh) + 1.0) - 0.5 * m * (math.log(s2) + sh / s2)

    def _theta_of_phi(self, phi: np.ndarray) -> np.ndarray:
        """φ (Stata metric) -> statsmodels relative parameters (cov_re lower triangle, row-major; vcomp) and σ²."""
        s2 = math.exp(2.0 * phi[-1])
        pos, covs = 0, []
        for lev in self.levs:
            covs.append(lev.cov_from_phi(phi[pos:pos + lev.n_phi]))
            pos += lev.n_phi
        q = self.levs[0].q
        ix = np.tril_indices(q)
        gam = list((covs[0] / s2)[ix]) + [covs[k][j, j] / s2 for k, j in self.vc_index]
        return np.array(gam + [s2])

    def phi_cov(self, phi: np.ndarray, free_phi: np.ndarray) -> np.ndarray:
        """Covariance of φ̂ from the statsmodels analytic Hessian (relative parameters, σ² profiled):
        fixed effects profiled out by a Schur complement, σ² restored explicitly
        (H_γγ = H_p − m/2 g gᵀ/ŝ⁴·ŝ², H_γs = m/2 g/ŝ², H_ss = −m/2/ŝ², g = ∂ŝ²/∂γ), then mapped to φ by the
        Jacobian of φ → (γ, σ²) (the gradient is zero at the MLE)."""
        p, q = self.p, self.levs[0].q
        k_re2 = q * (q + 1) // 2
        H, _ = self.model.hessian(self.res.params_object)
        H = np.asarray(H, dtype=float)
        top_free = [True] * k_re2
        if self.free is not None:                                   # independent top level: diagonal only
            r_, c_ = np.tril_indices(q)
            top_free = list(r_ == c_)
        gfree = np.array(top_free + [True] * len(self.vc_index))
        gi = np.flatnonzero(gfree)
        Hb = H[:p, :p]
        Hbg = H[:p, p + gi]
        Hp = H[np.ix_(p + gi, p + gi)] - Hbg.T @ np.linalg.solve(Hb, Hbg)
        cov_re = np.asarray(self.res.cov_re, dtype=float) / self.s2
        vc = np.asarray(self.res.vcomp, dtype=float) / self.s2
        gam0 = np.r_[cov_re[np.tril_indices(q)], vc]

        def shat(gm):
            cr = np.zeros((q, q))
            cr[np.tril_indices(q)] = gm[:k_re2]
            cr = cr + np.tril(cr, -1).T
            v_ = gm[k_re2:]
            fe, _ = self.model.get_fe_params(cr, v_)
            return float(self.model.get_scale(fe, cr, v_))

        sh = shat(gam0)
        g = np.zeros(len(gi))
        for a, i in enumerate(gi):
            h = 1e-5 * max(abs(gam0[i]), 1e-3)
            e = np.zeros_like(gam0)
            e[i] = h
            g[a] = (shat(gam0 + e) - shat(gam0 - e)) / (2.0 * h)
        m = (self.n - self.p) if self.reml else self.n
        k = len(gi)
        Hf = np.zeros((k + 1, k + 1))
        Hf[:k, :k] = Hp - 0.5 * m * np.outer(g, g) / sh ** 2
        Hf[:k, k] = Hf[k, :k] = 0.5 * m * g / sh ** 2
        Hf[k, k] = -0.5 * m / sh ** 2
        fi = np.flatnonzero(free_phi)
        J = np.zeros((k + 1, len(fi)))
        sel = np.r_[gi, len(gam0)]
        for a, i in enumerate(fi):
            h = 1e-6
            e = np.zeros_like(phi)
            e[i] = h
            J[:, a] = (self._theta_of_phi(phi + e)[sel] - self._theta_of_phi(phi - e)[sel]) / (2.0 * h)
        Hphi = J.T @ Hf @ J
        C = np.full((len(phi), len(phi)), np.nan)
        try:
            Ci = np.linalg.inv(-Hphi)
        except np.linalg.LinAlgError:
            return C
        if np.any(np.diag(Ci) <= 0):
            return C
        C[np.ix_(fi, fi)] = Ci
        return C

    def blups(self):
        """Conditional means / variances of the random effects (statsmodels random_effects, random_effects_cov)."""
        levs = self.levs
        b = [np.zeros((lev.G, lev.q)) for lev in levs]
        v = [np.zeros((lev.G, lev.q, lev.q)) for lev in levs]
        re = self.res.random_effects
        rc = self.res.random_effects_cov
        q0 = levs[0].q
        for gi in range(levs[0].G):
            key = gi if gi in re else list(re.keys())[gi]
            s = re[key]
            c = rc[key]
            vals = np.asarray(s.values, dtype=float)
            cm = np.asarray(c.values, dtype=float)
            b[0][gi] = vals[:q0]
            v[0][gi] = cm[:q0, :q0]
            for pos, lab in enumerate(s.index[q0:], start=q0):
                m = re_vc_label.match(str(lab))
                if not m:
                    continue
                k, j = int(m.group(1)), int(m.group(2))
                g = int(m.group(3))
                b[k][g, j] = vals[pos]
                v[k][g, j, j] = cm[pos, pos]
        return b, v

    def fitted(self) -> np.ndarray:
        return np.asarray(self.res.fittedvalues, dtype=float)


re_vc_label = re.compile(r"L(\d+)T(\d+)\[(\d+)\]")


def _phi_cov(fneg, phi: np.ndarray, free: np.ndarray, h: float = 1e-3) -> np.ndarray:
    """Inverse of the numerical Hessian of −log likelihood over the free φ entries (NaN elsewhere)."""
    idx = np.flatnonzero(free)
    C = np.full((len(phi), len(phi)), np.nan)
    if not len(idx):
        return C
    H = _num_hess(fneg, np.asarray(phi, dtype=float), idx, h)
    try:
        Ci = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return C
    if np.any(np.diag(Ci) <= 0):
        return C
    C[np.ix_(idx, idx)] = Ci
    return C


# ---------------------------------------------------------------------------------------------------------------
# Generalized linear mixed model (Laplace approximation, lme4 glmer nAGQ = 1)
# ---------------------------------------------------------------------------------------------------------------
class _GLMM:
    """Bernoulli-logit / Poisson-log mixed model with nested levels; Laplace deviance optimised over (θ, β).

    For given (θ, β) the conditional mode û of the spherical random effects is found by penalized IRLS (Newton on
    log p(y | u) − ||u||²/2, step halving); each Newton system (ΛᵀZᵀWZΛ + I) is solved by the same tree elimination
    as the linear model. Laplace: −2 log L ≈ −2 log p(y | û) + ||û||² + log|L|².
    """

    def __init__(self, y: np.ndarray, X: np.ndarray, levels: List[_Level], family: str,
                 offset: Optional[np.ndarray] = None):
        self.family = family
        self.levels = levels
        self.K = len(levels)
        leaf = levels[-1]
        order = np.argsort(leaf.codes, kind="stable")
        self.order = order
        self.inv = np.empty_like(order)
        self.inv[order] = np.arange(len(order))
        codes = leaf.codes[order]
        self.starts = np.flatnonzero(np.r_[True, np.diff(codes) != 0])
        self.y = np.asarray(y, dtype=float)[order]
        self.X = np.asarray(X, dtype=float)[order]
        self.off = (np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float))[order]
        self.n, self.p = self.X.shape
        self.Wz = np.hstack([lev.Z for lev in reversed(levels)])[order]
        self.Q = self.Wz.shape[1]
        self.codes = [lev.codes[order] for lev in levels]
        self.child_starts: List[Optional[np.ndarray]] = [None]
        for k in range(1, self.K):
            self.child_starts.append(np.flatnonzero(np.r_[True, np.diff(levels[k].parent) != 0]))
        self.theta_slices, pos = [], 0
        for lev in levels:
            self.theta_slices.append(slice(pos, pos + lev.n_theta))
            pos += lev.n_theta
        self.n_theta = pos
        self.Zs = [lev.Z[order] for lev in levels]
        self.const = -float(np.sum(_lgamma1(self.y))) if family == "poisson" else 0.0
        self._u = None
        self.nfev = 0

    def lams(self, theta: np.ndarray) -> List[np.ndarray]:
        return [lev.lam(theta[s]) for lev, s in zip(self.levels, self.theta_slices)]

    def _mu_w(self, eta: np.ndarray):
        if self.family == "binomial":
            mu = 1.0 / (1.0 + np.exp(-eta))
            return mu, np.maximum(mu * (1.0 - mu), 1e-12)
        mu = np.exp(np.minimum(eta, 700.0))
        return mu, np.maximum(mu, 1e-12)

    def loglik_cond(self, eta: np.ndarray) -> float:
        if self.family == "binomial":
            return float(np.sum(self.y * eta - np.logaddexp(0.0, eta)))
        return float(np.sum(self.y * eta - np.exp(np.minimum(eta, 700.0)))) + self.const

    def _zb(self, lams, u) -> np.ndarray:
        out = np.zeros(self.n)
        for k in range(self.K):
            b = u[k] @ lams[k].T
            out += np.einsum("ij,ij->i", self.Zs[k], b[self.codes[k]])
        return out

    def _solve(self, lams, w: np.ndarray, r: np.ndarray):
        """u solving (ΛᵀZᵀWZΛ + I) u = ΛᵀZᵀ r, and log|ΛᵀZᵀWZΛ + I|."""
        T = np.zeros((self.Q, self.Q))
        pos = 0
        for lev, L in zip(reversed(self.levels), reversed(lams)):
            T[pos:pos + lev.q, pos:pos + lev.q] = L
            pos += lev.q
        Wz = self.Wz
        ZZ = np.empty((len(self.starts), self.Q, self.Q))
        for a in range(self.Q):
            ZZ[:, a, :] = np.add.reduceat((w * Wz[:, a])[:, None] * Wz, self.starts, axis=0)
        Zr = np.add.reduceat(Wz * r[:, None], self.starts, axis=0)
        ZZ = np.matmul(np.matmul(T.T, ZZ), T)
        Zr = Zr @ T
        logdet = 0.0
        saved = [None] * self.K
        for k in range(self.K - 1, -1, -1):
            q = self.levels[k].q
            Lc, Linv = _chol_inv(ZZ[:, :q, :q] + np.eye(q))
            ct = np.einsum("gij,gj->gi", Linv, Zr[:, :q])
            logdet += 2.0 * float(np.sum(np.log(np.diagonal(Lc, axis1=1, axis2=2))))
            Bt = None
            if k > 0:
                Bt = np.matmul(Linv, ZZ[:, :q, q:])
                D = ZZ[:, q:, q:] - np.matmul(np.transpose(Bt, (0, 2, 1)), Bt)
                e = Zr[:, q:] - np.einsum("gji,gj->gi", Bt, ct)
                cs = self.child_starts[k]
                ZZ = np.add.reduceat(D, cs, axis=0)
                Zr = np.add.reduceat(e, cs, axis=0)
            saved[k] = (Linv, ct, Bt)
        u, path = [], None
        for k, lev in enumerate(self.levels):
            Linv, ct, Bt = saved[k]
            LinvT = np.transpose(Linv, (0, 2, 1))
            if k == 0:
                uk = np.einsum("gij,gj->gi", LinvT, ct)
                path = uk
            else:
                pu = path[lev.parent]
                uk = np.einsum("gij,gj->gi", LinvT, ct - np.einsum("gij,gj->gi", Bt, pu))
                path = np.hstack([uk, pu])
            u.append(uk)
        return u, logdet

    def pirls(self, theta: np.ndarray, beta: np.ndarray, u0=None, tol: float = 1e-13, max_it: int = 60):
        lams = self.lams(theta)
        fx = self.X @ beta + self.off
        u = [np.zeros((lev.G, lev.q)) for lev in self.levels] if u0 is None else [x.copy() for x in u0]

        def pdev(uu):
            eta = fx + self._zb(lams, uu)
            return -2.0 * self.loglik_cond(eta) + float(sum(np.sum(x * x) for x in uu)), eta

        cur, eta = pdev(u)
        for _ in range(max_it):
            mu, w = self._mu_w(eta)
            zb = eta - fx
            u_new, _ = self._solve(lams, w, w * zb + (self.y - mu))
            t = 1.0
            while True:
                cand = [a + t * (b - a) for a, b in zip(u, u_new)]
                val, eta_c = pdev(cand)
                if val <= cur + 1e-10 * abs(cur) or t < 1e-6:
                    break
                t *= 0.5
            done = abs(cur - val) < tol * (abs(val) + 0.1)
            u, cur, eta = cand, val, eta_c
            if done:
                break
        mu, w = self._mu_w(eta)
        _, logdet = self._solve(lams, w, np.zeros(self.n))
        return u, cur, logdet, eta

    def deviance(self, theta: np.ndarray, beta: np.ndarray) -> float:
        """Laplace −2 log L (with all constants)."""
        self.nfev += 1
        u, pd_, logdet, eta = self.pirls(theta, beta, self._u)
        if not np.isfinite(pd_):
            return np.inf
        self._u = u
        return pd_ + logdet - 2.0 * 0.0

    def fit(self, beta0: np.ndarray, max_iter: int = 2000) -> Dict[str, Any]:
        nt, p = self.n_theta, self.p
        bounds_t = [b for lev in self.levels for b in lev.theta_bounds()]
        lb = np.r_[[-np.inf if b[0] is None else b[0] for b in bounds_t], np.full(p, -np.inf)]
        ub = np.r_[[np.inf if b[1] is None else b[1] for b in bounds_t], np.full(p, np.inf)]
        f = lambda x: self.deviance(x[:nt], x[nt:])
        is_log = np.array([v for lev in self.levels for v in lev.is_log()])
        th_eta = lambda e: np.concatenate([lev.theta_from_eta(e[s]) for lev, s in zip(self.levels, self.theta_slices)])
        eta0 = np.concatenate([lev.eta_start(np.where(np.all(lev.Z == 1.0, axis=0), 1.0, np.std(lev.Z, axis=0)))
                               for lev in self.levels])
        fe = lambda z: f(np.r_[th_eta(z[:nt]), z[nt:]])
        e_bounds = [(-12.0, 8.0) if lg else (None, None) for lg in is_log] + [(None, None)] * p

        def cgrad(fun, x, lo, h=1e-5):
            g = np.zeros_like(x)
            for i in range(len(x)):
                hi = h * max(1.0, abs(x[i]))
                e = np.zeros_like(x)
                e[i] = hi
                if x[i] - hi < lo[i]:
                    g[i] = (-3.0 * fun(x) + 4.0 * fun(x + e) - fun(x + 2 * e)) / (2.0 * hi)
                else:
                    g[i] = (fun(x + e) - fun(x - e)) / (2.0 * hi)
            return g

        self.nfev = 0
        e_lb = np.r_[np.where(is_log, -12.0, -np.inf), np.full(p, -np.inf)]
        z0 = np.r_[eta0 + math.log(0.5), beta0]          # start: random-effect SD about 0.5 on the linear-predictor scale
        r1 = optimize.minimize(fe, z0, jac=lambda z: cgrad(fe, z, e_lb), method="L-BFGS-B", bounds=e_bounds,
                               options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-6, "maxls": 50})
        x = np.r_[th_eta(r1.x[:nt]), r1.x[nt:]]
        x[:nt] = np.where(is_log & (x[:nt] < 1e-5), 0.0, x[:nt])
        r2 = optimize.minimize(f, x, jac=lambda z: cgrad(f, z, lb), method="L-BFGS-B",
                               bounds=list(bounds_t) + [(None, None)] * p,
                               options={"maxiter": max_iter, "ftol": 1e-15, "gtol": 1e-9, "maxls": 50})
        x = np.clip(r2.x, lb, ub)
        fval = f(x)
        for _ in range(6):                                     # Newton polish (interior parameters)
            free = np.flatnonzero((x - lb > 1e-6) & (ub - x > 1e-6))
            g = cgrad(f, x, lb)[free]
            H = _num_hess(f, x, free, 1e-4)
            try:
                step = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(step)) or g @ step >= 0:
                break
            t, ok = 1.0, False
            while t > 1e-4:
                cand = x.copy()
                cand[free] = np.clip(x[free] + t * step, lb[free], ub[free])
                fc = f(cand)
                if fc <= fval:
                    ok = fc < fval - 1e-12
                    x, fval = cand, fc
                    break
                t *= 0.5
            if not ok or np.max(np.abs(t * step)) < 1e-9:
                break
        g = cgrad(f, x, lb)
        free = x - lb > 1e-6
        gnorm = float(np.max(np.abs(np.where(free, g, np.minimum(g, 0.0)))))
        return {"theta": x[:nt], "beta": x[nt:], "deviance": float(fval), "converged": bool(gnorm < 2e-2),
                "gnorm": gnorm, "nfev": int(self.nfev), "nit": int(r1.nit + r2.nit)}

    def loglik_phi(self, beta: np.ndarray, phi: np.ndarray) -> float:
        """Laplace log likelihood at (β, φ): φ per level = log SDs (and atanh correlations), Stata metric."""
        lams, pos = [], 0
        for lev in self.levels:
            S = lev.cov_from_phi(phi[pos:pos + lev.n_phi])
            pos += lev.n_phi
            if lev.cov == "independent":
                lams.append(np.diag(np.sqrt(np.diag(S))))
            elif lev.cov == "identity":
                lams.append(math.sqrt(S[0, 0]) * np.eye(lev.q))
            else:
                try:
                    lams.append(np.linalg.cholesky(S))
                except np.linalg.LinAlgError:
                    return -np.inf
        theta = np.concatenate([_theta_of_lam(lev, L) for lev, L in zip(self.levels, lams)])
        return -0.5 * self.deviance(theta, beta)


def _lgamma1(y: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln
    return gammaln(np.asarray(y, dtype=float) + 1.0)


def _theta_of_lam(lev: _Level, L: np.ndarray) -> np.ndarray:
    """θ vector of a level whose relative factor is the lower-triangular L (inverse of _Level.lam)."""
    q = lev.q
    if lev.cov == "unstructured":
        return np.array([L[i, j] for j in range(q) for i in range(j, q)])
    if lev.cov == "independent":
        return np.diag(L).copy()
    if lev.cov == "identity":
        return np.array([L[0, 0]])
    s = math.sqrt(max((L @ L.T)[0, 0], 0.0))
    r = (L @ L.T)[0, 1] / (s * s) if s > 0 else 0.0
    return np.array([s, r])


def _num_hess(f, x: np.ndarray, idx: np.ndarray, h: float) -> np.ndarray:
    m = len(idx)
    H = np.zeros((m, m))
    f0 = f(x)
    hs = [h * max(1.0, abs(x[i])) for i in idx]
    for a in range(m):
        ea = np.zeros_like(x)
        ea[idx[a]] = hs[a]
        H[a, a] = (f(x + ea) - 2.0 * f0 + f(x - ea)) / hs[a] ** 2
        for b in range(a + 1, m):
            eb = np.zeros_like(x)
            eb[idx[b]] = hs[b]
            H[a, b] = H[b, a] = (f(x + ea + eb) - f(x + ea - eb) - f(x - ea + eb) + f(x - ea - eb)) / (4.0 * hs[a] * hs[b])
    return H


# ---------------------------------------------------------------------------------------------------------------
# Option helpers (copied from registry.py to avoid an import cycle)
# ---------------------------------------------------------------------------------------------------------------
def _as_list(val: Any) -> List[str]:
    if val is None or val == "":
        return []
    if isinstance(val, str):
        return [v.strip() for v in val.split(",") if v.strip()]
    if isinstance(val, dict):
        return [str(k) for k in val]
    return [str(v) for v in val if v is not None and str(v) != ""]


def _as_single(val: Any) -> Any:
    if isinstance(val, (list, tuple)):
        return val[0] if len(val) > 0 else None
    if val == "":
        return None
    return val


def _opt(p: Dict[str, Any], camel: str, snake: str, default: Any) -> Any:
    for k in (camel, snake):
        if k in p and p[k] is not None and p[k] != "":
            return p[k]
    return default


def _flag(p: Dict[str, Any], camel: str, snake: str, default: bool) -> bool:
    v = _opt(p, camel, snake, default)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


def _fmt_level(v: Any) -> str:
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _sorted_levels(s: pd.Series) -> List[Any]:
    """Factor levels in natural order (numeric order for numeric codes, as Stata i. and R factor())."""
    vals = pd.unique(s.dropna())
    try:
        return sorted(vals, key=lambda x: float(x))
    except (TypeError, ValueError):
        return sorted(vals, key=lambda x: str(x))


def _split_terms(term: str) -> Tuple[List[str], bool]:
    """'a*b' -> (['a', 'b'], True: main effects + interactions); 'a:b' / 'a#b' -> (['a', 'b'], False)."""
    t = str(term).strip()
    if "*" in t or "##" in t:
        parts = [x.strip() for x in re.split(r"\*|##", t) if x.strip()]
        return parts, True
    parts = [x.strip() for x in re.split(r":|#", t) if x.strip()]
    return parts, False


_METHODS = ("reml", "ml")
FAMILIES = ("gaussian", "binomial", "poisson")
_FAMILY_ALIASES = {"normal": "gaussian", "linear": "gaussian", "logit": "binomial", "logistic": "binomial",
                   "bernoulli": "binomial", "melogit": "binomial", "count": "poisson", "mepoisson": "poisson"}
_METHOD_ALIASES = {"mle": "ml", "maximum likelihood": "ml", "restricted": "reml", "restricted ml": "reml"}
_RESTATS = ("var", "sd", "both")
_RESTAT_ALIASES = {"variance": "var", "variances": "var", "stddev": "sd", "stddeviations": "sd", "sds": "sd",
                   "std": "sd", "all": "both"}


# ---------------------------------------------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------------------------------------------
class MultilevelLinearModel(BaseAnalysis):
    """Linear mixed model with up to four nested levels (Stata `mixed y x || l3: || l2:`)."""

    def __init__(self, data: pd.DataFrame, dep: str, levels: Sequence[str], covs: Optional[List[str]] = None,
                 factors: Optional[List[str]] = None, interactions: Optional[List[str]] = None,
                 slopes: Optional[Dict[str, List[str]]] = None, cov_struct: Optional[Dict[str, str]] = None,
                 noconstant: Optional[Dict[str, bool]] = None, method: str = "reml", re_stat: str = "var",
                 icc: bool = True, lr_test: bool = True, group_info: bool = True, blups: bool = False,
                 caterpillar: bool = False, resid_plot: bool = False, center_grand: Optional[List[str]] = None,
                 center_group: Optional[List[str]] = None, group_means: Optional[List[str]] = None,
                 center_level: Optional[str] = None, ci_width: float = 95.0, max_iter: int = 2000,
                 family: str = "gaussian", exp_b: bool = False, offset: Optional[str] = None,
                 exposure: Optional[str] = None):
        levels = [str(v) for v in levels]
        if not dep:
            raise ValueError("다층모형: 종속변수(dep) 지정 필요")
        if not levels:
            raise ValueError("다층모형: 집단 변수(수준) 최소 1개 필요 (cluster 또는 levels)")
        if len(levels) > 3:
            raise ValueError(f"다층모형: 집단 수준은 최대 3개(4수준 모형)까지 지원 — 지정: {levels}")
        if len(set(levels)) != len(levels):
            raise ValueError(f"다층모형: 같은 집단 변수가 여러 수준에 지정됨 — {levels}")
        if dep in levels:
            raise ValueError(f"다층모형: 종속변수 '{dep}'가 집단 변수로도 지정됨")
        self.dep = dep
        self.levels = levels
        self.covs = list(covs or [])
        self.factors = list(factors or [])
        self.interactions = list(interactions or [])
        slopes = {str(k): list(v or []) for k, v in (slopes or {}).items()}
        cov_struct = {str(k): v for k, v in (cov_struct or {}).items()}
        noconstant = {str(k): bool(v) for k, v in (noconstant or {}).items()}
        for nm, dct in (("확률기울기", slopes), ("공분산 구조", cov_struct), ("절편 제외", noconstant)):
            bad = [k for k in dct if k not in levels]
            if bad:
                raise ValueError(f"다층모형: {nm} 지정 대상 '{', '.join(bad)}'는 집단 변수가 아님 (집단 변수: {', '.join(levels)})")
        self.slopes = {g: slopes.get(g, []) for g in levels}
        self.cov_struct = {}
        for g in levels:
            cs = cov_struct.get(g)
            self.cov_struct[g] = resolve_choice(f"공분산 구조({g})", cs, COV_STRUCTS, _COV_ALIASES,
                                                default="independent")
        self.noconstant = {g: bool(noconstant.get(g, False)) for g in levels}
        for g in levels:
            if self.noconstant[g] and not self.slopes[g]:
                raise ValueError(f"다층모형: '{g}' 수준에서 절편을 제외하면 확률기울기 변수가 1개 이상 필요")
        self.method = resolve_choice("추정 방법(method)", method, _METHODS, _METHOD_ALIASES, default="reml")
        self.re_stat = resolve_choice("확률효과 표시(reStat)", re_stat, _RESTATS, _RESTAT_ALIASES, default="var")
        self.center_grand = list(center_grand or [])
        self.center_group = list(center_group or [])
        self.group_means = list(group_means or [])
        both = sorted(set(self.center_grand) & set(self.center_group))
        if both:
            raise ValueError(f"다층모형: 같은 변수에 전체평균 중심화와 집단평균 중심화를 함께 지정함 — {', '.join(both)}")
        if center_level in (None, ""):
            center_level = levels[-1]
        if str(center_level) not in levels:
            raise ValueError(f"다층모형: 중심화 기준 수준 '{center_level}'는 집단 변수가 아님 (가능: {', '.join(levels)})")
        self.center_level = str(center_level)
        ci_width = float(ci_width)
        if not 50.0 <= ci_width < 100.0:
            raise ValueError(f"다층모형: 신뢰구간 폭(ciWidth)은 50 이상 100 미만 — 지정: {ci_width}")
        self.ci_width = ci_width
        self.want = {"icc": bool(icc), "lrTest": bool(lr_test), "groupInfo": bool(group_info), "blups": bool(blups),
                     "caterpillar": bool(caterpillar), "residPlot": bool(resid_plot)}
        self.max_iter = int(max_iter)
        self.family = resolve_choice("분포(family)", family, FAMILIES, _FAMILY_ALIASES, default="gaussian")
        self.exp_b = bool(exp_b)
        self.offset_var = str(offset) if offset not in (None, "") else None
        self.exposure_var = str(exposure) if exposure not in (None, "") else None
        if self.family == "gaussian" and (self.offset_var or self.exposure_var or self.exp_b):
            raise ValueError("다층모형: offset·exposure·expB는 로지스틱·포아송 다층모형(family='binomial'·'poisson')에서만 사용")
        if self.exposure_var and self.family != "poisson":
            raise ValueError("다층모형: exposure는 포아송 다층모형(family='poisson')에서만 사용")
        if self.offset_var and self.exposure_var:
            raise ValueError("다층모형: offset과 exposure는 함께 지정할 수 없음")
        if self.family != "gaussian":
            self.method = "laplace"
        options = {
            "dep": dep, "levels": levels, "cluster": levels[-1], "covs": self.covs, "factors": self.factors,
            "interactions": self.interactions, "randomSlopes": {g: v for g, v in self.slopes.items() if v},
            "covariance": dict(self.cov_struct), "noconstant": [g for g in levels if self.noconstant[g]],
            "method": self.method, "reml": self.method == "reml", "reStat": self.re_stat,
            "icc": bool(icc), "lrTest": bool(lr_test), "groupInfo": bool(group_info), "blups": bool(blups),
            "caterpillar": bool(caterpillar), "residPlot": bool(resid_plot), "centerGrand": self.center_grand,
            "centerGroup": self.center_group, "groupMeans": self.group_means, "centerLevel": self.center_level,
            "ciWidth": ci_width, "maxIter": self.max_iter, "family": self.family, "expB": self.exp_b,
            "offset": self.offset_var or "", "exposure": self.exposure_var or "",
        }
        if self.family == "gaussian":
            options["reml"] = self.method == "reml"
        else:
            options.pop("reml", None)
        title = {"gaussian": "다층모형 · 선형 혼합모형 (Stata mixed)",
                 "binomial": "다층 로지스틱 회귀 (Stata melogit · 라플라스 근사)",
                 "poisson": "다층 포아송 회귀 (Stata mepoisson · 라플라스 근사)"}[self.family]
        super().__init__(name="mixedModel", title=title, dataset=data, options=options)
        self.syntax_python = self._python_call()
        self.syntax_r = "\n".join(r_code("mixedModel", options) or [])
        self.run()

    # ------------------------------------------------------------------ helpers
    def _python_call(self) -> str:
        o = self.options
        parts = [f"dep={self.dep!r}", f"levels={self.levels!r}"]
        for key in ("covs", "factors", "interactions"):
            if o[key]:
                parts.append(f"{key}={o[key]!r}")
        if o["randomSlopes"]:
            parts.append(f"randomSlopes={o['randomSlopes']!r}")
        cov = {g: v for g, v in o["covariance"].items() if v != "independent"}
        if cov:
            parts.append(f"covariance={cov!r}")
        if o["noconstant"]:
            parts.append(f"noconstant={o['noconstant']!r}")
        if self.family == "gaussian":
            parts.append(f"method={self.method!r}")
        else:
            parts.append(f"family={self.family!r}")
            if self.exp_b:
                parts.append("expB=True")
            if self.offset_var:
                parts.append(f"offset={self.offset_var!r}")
            if self.exposure_var:
                parts.append(f"exposure={self.exposure_var!r}")
        for key in ("centerGrand", "centerGroup", "groupMeans"):
            if o[key]:
                parts.append(f"{key}={o[key]!r}")
        if (o["centerGroup"] or o["groupMeans"]) and self.center_level != self.levels[-1]:
            parts.append(f"centerLevel={self.center_level!r}")
        if self.re_stat != "var":
            parts.append(f"reStat={self.re_stat!r}")
        for key, default in (("icc", True), ("lrTest", True), ("groupInfo", True), ("blups", False),
                             ("caterpillar", False), ("residPlot", False)):
            if bool(o[key]) != default:
                parts.append(f"{key}={bool(o[key])}")
        if self.ci_width != 95.0:
            parts.append(f"ciWidth={self.ci_width:g}")
        return f"res = sess.run('mixedModel', {', '.join(parts)})"

    def _check_columns(self) -> None:
        df = self.dataset
        need = {"종속변수": [self.dep], "집단 변수": self.levels, "공변량": self.covs, "요인": self.factors,
                "전체평균 중심화": self.center_grand, "집단평균 중심화": self.center_group, "집단평균 투입": self.group_means}
        for g in self.levels:
            need[f"확률기울기({g})"] = self.slopes[g]
        missing = []
        for role, cols in need.items():
            for c in cols:
                if c not in df.columns:
                    missing.append(f"{role} '{c}'")
        if missing:
            raise ValueError(f"다층모형: 자료에 없는 변수 — {', '.join(missing)}. 사용 가능한 변수: {list(df.columns)}")

    # ------------------------------------------------------------------ run
    def run(self):
        self._check_columns()
        df = self.dataset
        notes_info: List[str] = []

        # ---- interaction components (resolved against original variable names)
        inter_specs = []
        for term in self.interactions:
            parts, full = _split_terms(term)
            if len(parts) < 2:
                raise ValueError(f"다층모형: 상호작용 항 '{term}' 형식 오류 (예: 'x*z', 'x:z')")
            inter_specs.append((term, parts, full))
        base_vars: List[str] = []
        for v in self.covs + self.factors:
            if v not in base_vars:
                base_vars.append(v)
        for term, parts, full in inter_specs:
            for v in parts:
                if v not in df.columns:
                    raise ValueError(f"다층모형: 상호작용 항 '{term}'의 변수 '{v}'가 자료에 없음")
                if full and v not in base_vars:
                    base_vars.append(v)         # 'a*b' adds the main effects (R / Stata ##)
        slope_vars = [v for g in self.levels for v in self.slopes[g]]
        extra = [v for v in self.center_grand + self.center_group + self.group_means if v not in base_vars]
        inter_vars = [v for _, parts, _ in inter_specs for v in parts]
        used = []
        off_vars = [v for v in (self.offset_var, self.exposure_var) if v]
        for v in off_vars:
            if v not in df.columns:
                raise ValueError(f"다층모형: 자료에 없는 변수 — offset/exposure '{v}'. 사용 가능한 변수: {list(df.columns)}")
        for v in [self.dep] + self.levels + base_vars + slope_vars + extra + inter_vars + off_vars:
            if v not in used:
                used.append(v)
        clean = df[used].copy()
        n_raw = len(clean)
        clean = clean.dropna()
        dep_num = pd.to_numeric(clean[self.dep], errors="coerce")
        if dep_num.isna().any():
            raise ValueError(f"다층모형: 종속변수 '{self.dep}'에 숫자가 아닌 값 포함")
        clean[self.dep] = dep_num.astype(float)
        if len(clean) < n_raw:
            notes_info.append(f"결측 제외(목록별 삭제): {n_raw - len(clean)}개 관측치 제외")

        is_factor = {}
        for v in set(base_vars + inter_vars + slope_vars + extra):
            is_factor[v] = (v in self.factors) or not pd.api.types.is_numeric_dtype(clean[v])
        for v in slope_vars:
            if is_factor[v]:
                raise ValueError(f"다층모형: 확률기울기 변수 '{v}'는 숫자형이어야 함 (범주형은 더미변수를 만든 뒤 지정)")
        for v in self.center_grand + self.center_group + self.group_means:
            if is_factor[v]:
                raise ValueError(f"다층모형: 중심화·집단평균 변수 '{v}'는 숫자형이어야 함")

        # ---- nested group codes (level k = combination of the ids of all higher levels and its own id)
        codes, parents, labels = [], [], []
        prev = None
        for k, g in enumerate(self.levels):
            grp = clean.groupby(self.levels[:k + 1], sort=True)
            cd = grp.ngroup().to_numpy()
            G = int(cd.max()) + 1 if len(cd) else 0
            keys = list(grp.groups.keys())
            labels.append(["/".join(_fmt_level(x) for x in (key if isinstance(key, tuple) else (key,))) for key in keys])
            par = None
            if k > 0:
                par = np.zeros(G, dtype=int)
                par[cd] = prev
                n_raw_ids = clean[g].nunique()
                if n_raw_ids < G:
                    up = self.levels[k - 1]
                    notes_info.append(f"'{g}' 값이 상위 수준 '{up}' 안에서만 고유 → '{up}'와 결합한 조합으로 집단 구분 "
                                      f"({n_raw_ids}개 값 → {G}개 집단, Stata `|| {up}: || {g}:`, lme4 `(1 | {up}:{g})`와 동일)")
            codes.append(cd)
            parents.append(par)
            prev = cd
        n = len(clean)
        for k, g in enumerate(self.levels):
            G = int(codes[k].max()) + 1
            if G < 2:
                raise ValueError(f"다층모형: '{g}' 수준의 집단 수가 {G}개 — 2개 이상 필요")
            if G >= n:
                raise ValueError(f"다층모형: '{g}' 수준의 집단 수({G})가 관측치 수({n}) 이상 — 집단 변수 확인 필요")
            if k > 0 and G == int(codes[k - 1].max()) + 1:
                notes_info.append(f"'{g}' 수준의 집단 수가 상위 수준 '{self.levels[k - 1]}'와 같음 — 두 수준의 분산이 식별되지 않을 수 있음")

        # ---- centering and group means (estimation sample)
        cl_path = self.levels[:self.levels.index(self.center_level) + 1]
        rename: Dict[str, str] = {}
        derived = []
        for v in self.center_grand:
            nm = f"{v}_cgm"
            clean[nm] = clean[v].astype(float) - float(clean[v].astype(float).mean())
            rename[v] = nm
            derived.append(f"{nm} = {v} − 전체평균")
        gmean = {}
        for v in sorted(set(self.center_group + self.group_means), key=(self.center_group + self.group_means).index):
            gmean[v] = clean.groupby(cl_path, sort=False)[v].transform("mean").astype(float)
        for v in self.center_group:
            nm = f"{v}_cwc"
            clean[nm] = clean[v].astype(float) - gmean[v]
            rename[v] = nm
            derived.append(f"{nm} = {v} − {self.center_level} 평균")
        mean_names = {}
        for v in self.group_means:
            nm = f"{v}_{self.center_level}평균"
            clean[nm] = gmean[v]
            mean_names[v] = nm
            derived.append(f"{nm} = {self.center_level}별 {v} 평균")
        self._derived = derived

        # ---- fixed-effects design
        cols: List[np.ndarray] = [np.ones(n)]
        names: List[str] = ["Intercept"]
        dummies: Dict[str, List[Tuple[str, np.ndarray]]] = {}

        def factor_cols(v: str) -> List[Tuple[str, np.ndarray]]:
            if v not in dummies:
                lv = _sorted_levels(clean[v])
                if len(lv) < 2:
                    raise ValueError(f"다층모형: 범주형 변수 '{v}'의 범주가 1개뿐")
                s = clean[v]
                dummies[v] = [(f"{v}_{_fmt_level(l)}", (s == l).to_numpy(dtype=float)) for l in lv[1:]]
            return dummies[v]

        def num_col(v: str) -> Tuple[str, np.ndarray]:
            nm = rename.get(v, v)
            return nm, clean[nm].to_numpy(dtype=float)

        def add(nm: str, col: np.ndarray):
            if nm not in names:
                names.append(nm)
                cols.append(col)

        for v in base_vars:
            if is_factor[v]:
                for nm, c in factor_cols(v):
                    add(nm, c)
            else:
                add(*num_col(v))
                if v in mean_names:
                    add(mean_names[v], clean[mean_names[v]].to_numpy(dtype=float))
        for v in self.group_means:
            add(mean_names[v], clean[mean_names[v]].to_numpy(dtype=float))
        for v in self.center_grand + self.center_group:
            if v not in base_vars and v not in self.group_means:
                notes_info.append(f"중심화 변수 '{v}'가 고정효과에 없음 — 확률기울기·상호작용에만 사용")

        def term_cols(v: str) -> List[Tuple[str, np.ndarray]]:
            if is_factor[v]:
                return factor_cols(v)
            return [num_col(v)]

        for term, parts, full in inter_specs:
            combos: List[List[str]] = []
            if full:
                from itertools import combinations
                for r_ in range(2, len(parts) + 1):
                    combos += [list(c) for c in combinations(parts, r_)]
            else:
                combos = [parts]
            for combo in combos:
                prods: List[Tuple[str, np.ndarray]] = [("", np.ones(n))]
                for v in combo:
                    prods = [((a + ":" if a else "") + nm, ca * cb) for a, ca in prods for nm, cb in term_cols(v)]
                for nm, c in prods:
                    add(nm, c)

        X = np.column_stack(cols)
        # collinear columns are dropped (Stata: "omitted")
        keep = [0]
        for j in range(1, X.shape[1]):
            trial = X[:, keep + [j]]
            if np.linalg.matrix_rank(trial, tol=1e-9 * max(1.0, np.abs(trial).max()) * n) == len(keep) + 1:
                keep.append(j)
            else:
                notes_info.append(f"고정효과 '{names[j]}' 제외: 다른 변수와 완전 공선 (Stata omitted)")
        X = X[:, keep]
        names = [names[j] for j in keep]
        fixed_set = set(names)

        # ---- random-effects design per level
        levs: List[_Level] = []
        for k, g in enumerate(self.levels):
            terms, zcols = [], []
            for v in self.slopes[g]:
                nm, c = num_col(v)
                terms.append(nm)
                zcols.append(c)
                if nm not in fixed_set:
                    notes_info.append(f"확률기울기 '{nm}'({g})가 고정효과에 없음 — 평균 0인 기울기 편차로 추정 (Stata와 동일)")
            if not self.noconstant[g]:
                terms.append("_cons")
                zcols.append(np.ones(n))
            G = int(codes[k].max()) + 1
            levs.append(_Level(g, terms, np.column_stack(zcols), self.cov_struct[g], codes[k], G, parents[k]))

        y = clean[self.dep].to_numpy(dtype=float)
        if self.family != "gaussian":
            return self._run_glmm(clean, X, names, levs, codes, labels, notes_info, derived, y)
        reml = self.method == "reml"
        p = X.shape[1]
        if n - p <= 0:
            raise ValueError("다층모형: 관측치 수가 고정효과 수보다 적음")
        use_sm = _statsmodels_supports(levs)
        engine_note = None
        sm_checked = None
        if use_sm:
            smf_ = _SMFit(y, X, levs, reml, max_iter=self.max_iter)
            if not smf_.converged:
                # no convergence signal (typically a variance at the boundary): check against the own optimiser
                own = _LMM(y, X, levs)
                of = own.fit(reml, max_iter=self.max_iter)
                ll_own = -0.5 * of["deviance"]
                if ll_own > smf_.ll + 1e-6:
                    use_sm = False
                    engine_note = ("statsmodels MixedLM 최적화가 수렴하지 않아 자체 구현 추정(프로파일 우도, lme4 방식)으로 대체 "
                                   f"(로그우도 {smf_.ll:.4f} → {ll_own:.4f})")
                elif of["converged"]:
                    sm_checked = ll_own - smf_.ll
        if use_sm:
            engine = "statsmodels MixedLM"
            beta, vcov_b, s2, covs_abs, ll = smf_.beta, smf_.vcov_b, smf_.s2, smf_.covs_abs, smf_.ll
            n_vc = smf_.n_vc
            phi = smf_.phi_hat()
            fneg = lambda x: -smf_.loglik_phi(x)
            conv_ok = smf_.converged or sm_checked is not None
            conv_txt = f"statsmodels MixedLM ({smf_.method}, gtol 1e-8)"
            if sm_checked is not None:
                engine_note = ("statsmodels가 수렴 신호를 내지 않음(경계 근처 해) — 자체 구현 추정의 최적값과 로그우도 일치 확인 "
                               f"(차이 {sm_checked:+.1e})")
            blup_fn = smf_.blups
            fitted_fn = smf_.fitted
        else:
            engine = "자체 구현 추정"
            if engine_note is None:
                engine_note = ("이 확률효과 구조(상위 수준 동일·교환 구조 또는 하위 수준 비구조·동일·교환 구조)는 "
                               "statsmodels MixedLM이 지원하지 않아 자체 구현 추정(프로파일 우도, lme4 방식; Stata·lme4와 대조 검증)으로 계산")
                own = _LMM(y, X, levs)
                of = own.fit(reml, max_iter=self.max_iter)
            theta = of["theta"]
            lams = own.lams(theta)
            pls = own.pls(lams)
            beta = pls["beta"]
            s2 = pls["r2"] / ((n - p) if reml else n)
            vcov_b = s2 * np.linalg.inv(pls["Fxx"])
            ll = -0.5 * of["deviance"]
            n_vc = own.n_theta
            phi, _, covs_abs = own.phi_hat(theta, reml)
            fneg = lambda x: -own.loglik_phi(x, reml)
            conv_ok = of["converged"]
            conv_txt = f"자체 구현 (최대 경사 {of['gnorm']:.1e}, 함수 평가 {of['nfev']}회)"
            blup_fn = lambda: own.blups(lams, beta, s2, cond_var=True)

            def fitted_fn():
                bb, _ = own.blups(lams, beta, s2, cond_var=False)
                ft = X @ beta
                for k_, lev_ in enumerate(levs):
                    ft = ft + np.einsum("ij,ij->i", lev_.Z, bb[k_][lev_.codes])
                return ft
        se_b = np.sqrt(np.maximum(np.diag(vcov_b), 0.0))
        n_par = p + n_vc + 1
        aic = -2.0 * ll + 2.0 * n_par
        bic = -2.0 * ll + math.log(n) * n_par
        zc = float(stats.norm.ppf(0.5 + self.ci_width / 200.0))
        ci_lbl = f"{self.ci_width:g}%"

        # ---- variance components in the Stata metric (log SD, atanh correlation, log σ_e; β profiled out)
        free = np.ones(len(phi), dtype=bool)
        boundary_lv = []
        pos = 0
        for lev, S in zip(levs, covs_abs):
            sd = np.sqrt(np.maximum(np.diag(S), 0.0))
            sl = slice(pos, pos + lev.n_phi)
            if lev.cov == "independent" and lev.q > 1:          # a zero variance does not affect the others
                small = sd < 1e-4 * math.sqrt(s2)
                if np.any(small):
                    free[pos + np.flatnonzero(small)] = False
                    boundary_lv.append(lev.name)
                pos += lev.n_phi
                continue
            bad = bool(np.any(sd < 1e-4 * math.sqrt(s2)))
            if lev.cov == "unstructured" and not bad and lev.q > 1:
                R = S / np.outer(sd, sd)
                bad = bool(np.max(np.abs(R[~np.eye(lev.q, dtype=bool)])) > 1 - 1e-4) or np.linalg.eigvalsh(R).min() < 1e-6
            if lev.cov == "exchangeable" and not bad:
                bad = abs(math.tanh(phi[pos + 1])) > 1 - 1e-6
            if bad:
                free[sl] = False
                boundary_lv.append(lev.name)
            pos += lev.n_phi
        if use_sm and not np.all(free):
            # a variance at the boundary: the analytic Hessian of the remaining parameters is not separable
            C = _phi_cov(fneg, phi, free)
        elif use_sm:
            C = smf_.phi_cov(phi, free)
            if np.any(~np.isfinite(np.diag(C))[free]):
                C = _phi_cov(fneg, phi, free)          # analytic route failed (near-singular): numerical Hessian
        else:
            C = _phi_cov(fneg, phi, free)
        self.engine = engine
        self._fit = {"engine": engine, "phi": phi, "phi_cov": C, "beta": beta, "vcov": vcov_b, "names": names,
                     "s2": s2, "covs": covs_abs, "levels": levs, "labels": labels, "clean": clean, "X": X, "y": y}

        # ================= tables =================
        meth_lbl = "REML (제한 최대우도)" if reml else "ML (최대우도, Stata mixed 기본)"
        ll_lbl = "제한 로그우도 (Log restricted-likelihood)" if reml else "로그우도 (Log likelihood)"
        info = ResultsTable(name="info", title="모형 정보 (Model information)")
        info.add_column("info", "항목", col_type="text")
        info.add_column("val", "값", col_type="text")
        info.add_row({"info": "추정 방법", "val": meth_lbl})
        info.add_row({"info": "계산", "val": engine})
        info.add_row({"info": "종속변수", "val": self.dep})
        info.add_row({"info": "관측치 수 (Number of obs)", "val": f"{n:,}"})
        info.add_row({"info": "수준 구조 (상위 → 하위)", "val": " > ".join(self.levels) + " > 관측치"})
        info.add_row({"info": ll_lbl, "val": f"{ll:.4f}"})
        info.add_row({"info": "AIC", "val": f"{aic:.3f}"})
        info.add_row({"info": "BIC", "val": f"{bic:.3f}"})
        wald = None
        idx_nc = [j for j, nm in enumerate(names) if nm != "Intercept"]
        if idx_nc:
            bsub = beta[idx_nc]
            Vsub = vcov_b[np.ix_(idx_nc, idx_nc)]
            try:
                w = float(bsub @ np.linalg.solve(Vsub, bsub))
                wald = (w, len(idx_nc), float(stats.chi2.sf(w, len(idx_nc))))
                info.add_row({"info": f"Wald χ²({len(idx_nc)})", "val": f"{w:.2f}"})
                info.add_row({"info": "Prob > χ²", "val": f"{wald[2]:.4f}" if wald[2] >= 1e-4 else "< .0001"})
            except np.linalg.LinAlgError:
                pass
        info.add_row({"info": "수렴", "val": ("수렴" if conv_ok else "수렴 실패 — 결과 해석 주의") + f" ({conv_txt})"})
        if engine_note:
            info.add_note(engine_note)
        for t in notes_info:
            info.add_note(t)
        for t in derived:
            info.add_note(f"생성 변수: {t}")
        if not conv_ok:
            info.add_note("최적화가 수렴 기준을 충족하지 못함 — 모형 단순화(확률기울기·공분산 구조) 또는 변수 척도 조정 검토")
        self.add_table(info)

        if self.want["groupInfo"]:
            gt = ResultsTable(name="groups", title="집단 구조 (Group structure)")
            gt.add_column("level", "집단 변수", col_type="text")
            gt.add_column("groups", "집단 수", col_type="integer")
            gt.add_column("min", "최소", col_type="integer", super_title="집단당 관측치")
            gt.add_column("avg", "평균", col_type="number", super_title="집단당 관측치")
            gt.add_column("max", "최대", col_type="integer", super_title="집단당 관측치")
            for k, g in enumerate(self.levels):
                cnt = np.bincount(codes[k])
                gt.add_row({"level": g, "groups": int(len(cnt)), "min": int(cnt.min()), "avg": float(cnt.mean()),
                            "max": int(cnt.max())})
            self.add_table(gt)

        fe = ResultsTable(name="fixed", title=f"고정효과 (Fixed effects) — {self.dep}")
        fe.add_column("name", "항", col_type="text")
        fe.add_column("est", "계수", col_type="number")
        fe.add_column("se", "표준오차", col_type="number")
        fe.add_column("z", "z", col_type="number")
        fe.add_column("p", "p", col_type="number", fmt="pvalue")
        fe.add_column("cil", "하한", col_type="number", super_title=f"{ci_lbl} 신뢰구간")
        fe.add_column("ciu", "상한", col_type="number", super_title=f"{ci_lbl} 신뢰구간")
        for j, nm in enumerate(names):
            zv = beta[j] / se_b[j] if se_b[j] > 0 else np.nan
            fe.add_row({"name": nm, "est": float(beta[j]), "se": float(se_b[j]), "z": float(zv),
                        "p": float(2.0 * stats.norm.sf(abs(zv))), "cil": float(beta[j] - zc * se_b[j]),
                        "ciu": float(beta[j] + zc * se_b[j])})
        fe.mark_coefficients(estimate="est", se="se", p="p", stat="z", ci=("cil", "ciu"), label="name")
        fe.add_note("z 검정·정규 신뢰구간 (Stata mixed 기본); 표준오차 = (X′V̂⁻¹X)⁻¹" + (" (REML)" if reml else " (ML)"))
        self.add_table(fe)

        self._random_table(levs, covs_abs, phi, C, s2, zc, ci_lbl, boundary_lv)

        # ---- LR test vs linear regression
        if self.want["lrTest"]:
            lr = max(0.0, 2.0 * (ll - _ols_loglik(y, X, reml)))
            lt = ResultsTable(name="lrtest", title="선형회귀 대비 LR 검정 (LR test vs. linear model)")
            lt.add_column("stat", "검정", col_type="text")
            lt.add_column("chi2", "χ²", col_type="number")
            lt.add_column("df", "df", col_type="integer")
            lt.add_column("p", "p", col_type="number", fmt="pvalue")
            if n_vc == 1:
                pv = 0.5 * float(stats.chi2.sf(lr, 1)) if lr > 0 else 1.0
                lt.add_row({"stat": "chibar2(01)", "chi2": lr, "df": 1, "p": pv})
                lt.add_note("분산 성분 1개: 경계 검정 χ̄²(01), p = ½·P(χ²₁ > LR)")
            else:
                pv = float(stats.chi2.sf(lr, n_vc))
                lt.add_row({"stat": f"chi2({n_vc})", "chi2": lr, "df": n_vc, "p": pv})
                lt.add_note("분산 성분이 여러 개인 경계 검정: 보수적 검정이며 참고용 (Stata 주석과 동일)")
            if reml:
                lt.add_note("REML: 고정효과가 같은 두 모형의 제한 로그우도 비교")
            self.add_table(lt)
            self._lr = (lr, n_vc, pv)
        else:
            self._lr = None

        # ---- ICC (estat icc)
        icc_vals = self._icc_table(levs, covs_abs, phi, C, s2, zc, ci_lbl, boundary_lv)

        # ---- BLUPs
        need_blup = self.want["blups"] or self.want["caterpillar"]
        blup_b = blup_v = None
        if need_blup:
            blup_b, blup_v = blup_fn()
            self._fit["blups"] = (blup_b, blup_v)
        if self.want["blups"]:
            bt = ResultsTable(name="blups", title="확률효과 예측값 (BLUP · 조건부 최빈값)")
            bt.add_column("level", "수준", col_type="text")
            bt.add_column("group", "집단", col_type="text")
            bt.add_column("term", "항", col_type="text")
            bt.add_column("blup", "예측값", col_type="number")
            bt.add_column("se", "조건부 표준편차", col_type="number")
            for k, lev in enumerate(levs):
                for gi in range(lev.G):
                    for j, term in enumerate(lev.terms):
                        bt.add_row({"level": lev.name, "group": labels[k][gi], "term": term,
                                    "blup": float(blup_b[k][gi, j]), "se": float(math.sqrt(max(blup_v[k][gi, j, j], 0.0)))})
            bt.add_note("예측값: 확률효과의 조건부 평균 (Stata predict, reffects; lme4 ranef); 조건부 표준편차는 분산 성분·β̂를 "
                        "고정한 조건부 분산의 제곱근 (Stata reses, lme4 condVar"
                        + ("; statsmodels random_effects·random_effects_cov로 계산)" if use_sm else ")"))
            bt.add_note("집단 표기: 상위 수준 ID/하위 수준 ID")
            self.add_table(bt)

        # ---- plots
        if self.want["caterpillar"]:
            from .plot_style import finalize_and_serialize_figure
            for k, lev in enumerate(levs):
                src = self._caterpillar(lev, blup_b[k], blup_v[k], zc, labels[k])
                if src:
                    self.add_plot(f"확률효과 예측값 — {lev.name} (caterpillar)", src)
        if self.want["residPlot"]:
            fitted = fitted_fn()
            src = self._resid_plot(fitted, y - fitted)
            if src:
                self.add_plot("잔차 진단 (조건부 잔차)", src)

        # ---- model statistics and narrative
        G_low = int(levs[-1].G)
        self.model_stats = {"N": int(n), "Groups": G_low, "LL": float(ll), "AIC": float(aic), "BIC": float(bic),
                            "Var_within": float(s2)}
        if wald:
            self.model_stats["Chi2"] = (wald[0], wald[2])
            self.model_stats["df"] = int(wald[1])
        if icc_vals:
            self.model_stats["ICC"] = float(icc_vals[-1])
        if len(levs) == 1 and levs[0].terms == ["_cons"]:
            self.model_stats["Var_between"] = float(covs_abs[0][0, 0])
        grp_txt = ", ".join(f"{lev.name} {lev.G}개" for lev in levs)
        icc_txt = f", 최하위 집단 ICC = {icc_vals[-1]:.3f}" if icc_vals else ""
        self.apa_narrative = (f"{len(levs) + 1}수준 선형 혼합모형({'REML' if reml else 'ML'})으로 '{self.dep}'를 분석함"
                              f"(관측치 {n:,}개, {grp_txt}; 로그우도 = {ll:.2f}{icc_txt}).")

    # ------------------------------------------------------------------ generalized models (melogit / mepoisson)
    def _run_glmm(self, clean, X, names, levs, codes, labels, notes_info, derived, y):
        import statsmodels.api as sm
        fam = self.family
        n, p = X.shape
        if fam == "binomial":
            vals = set(np.unique(y))
            if not vals <= {0.0, 1.0}:
                notes_info.append(f"종속변수 '{self.dep}': 0이 아닌 값을 1(사건)로 처리 (Stata melogit과 동일)")
            y = (y != 0).astype(float)
            if len(np.unique(y)) < 2:
                raise ValueError(f"다층 로지스틱: 종속변수 '{self.dep}'가 한 값뿐")
        else:
            if np.any(y < 0):
                raise ValueError(f"다층 포아송: 종속변수 '{self.dep}'에 음수 포함")
            if np.any(np.abs(y - np.round(y)) > 1e-9):
                notes_info.append(f"종속변수 '{self.dep}'에 정수가 아닌 값 포함 — 포아송 우도를 그대로 적용")
        off = None
        if self.exposure_var:
            ex = clean[self.exposure_var].to_numpy(dtype=float)
            if np.any(ex <= 0):
                raise ValueError(f"다층 포아송: exposure '{self.exposure_var}'는 양수여야 함")
            off = np.log(ex)
        elif self.offset_var:
            off = clean[self.offset_var].to_numpy(dtype=float)
        glm_family = sm.families.Binomial() if fam == "binomial" else sm.families.Poisson()
        glm = sm.GLM(y, X, family=glm_family, offset=off).fit()
        model = _GLMM(y, X, levs, fam, off)
        fit = model.fit(np.asarray(glm.params, dtype=float), max_iter=self.max_iter)
        theta, beta = fit["theta"], fit["beta"]
        lams = model.lams(theta)
        covs_abs = [L @ L.T for L in lams]
        ll = -0.5 * fit["deviance"]
        n_vc = model.n_theta
        n_par = p + n_vc
        aic = -2.0 * ll + 2.0 * n_par
        bic = -2.0 * ll + math.log(n) * n_par
        zc = float(stats.norm.ppf(0.5 + self.ci_width / 200.0))
        ci_lbl = f"{self.ci_width:g}%"

        # Stata metric: (β, log SDs / atanh correlations); OIM from the Laplace log likelihood
        phi = []
        for lev, S in zip(levs, covs_abs):
            phi += lev.phi_from_cov(S)
        phi = np.array(phi)
        free = np.ones(len(phi), dtype=bool)
        boundary_lv, pos = [], 0
        for lev, S in zip(levs, covs_abs):
            sd = np.sqrt(np.maximum(np.diag(S), 0.0))
            sl = slice(pos, pos + lev.n_phi)
            bad = bool(np.any(sd < 1e-4))
            if lev.cov == "unstructured" and not bad and lev.q > 1:
                R = S / np.outer(sd, sd)
                bad = bool(np.max(np.abs(R[~np.eye(lev.q, dtype=bool)])) > 1 - 1e-6)
            if lev.cov == "exchangeable" and not bad:
                bad = abs(math.tanh(phi[pos + 1])) > 1 - 1e-6
            if bad:
                free[sl] = False
                boundary_lv.append(lev.name)
            pos += lev.n_phi
        xhat = np.r_[beta, phi]
        fneg = lambda x: -model.loglik_phi(x[:p], x[p:])
        idx = np.r_[np.arange(p), p + np.flatnonzero(free)]
        H = _num_hess(fneg, xhat, idx, 1e-4)
        Cfull = np.full((len(xhat), len(xhat)), np.nan)
        try:
            Ci = np.linalg.inv(H)
            if np.all(np.diag(Ci) > 0):
                Cfull[np.ix_(idx, idx)] = Ci
        except np.linalg.LinAlgError:
            pass
        vcov_b = Cfull[:p, :p]
        se_b = np.sqrt(np.where(np.diag(vcov_b) > 0, np.diag(vcov_b), np.nan))
        C = np.full((len(phi) + 1, len(phi) + 1), np.nan)   # + dummy residual slot for the shared table code
        C[:len(phi), :len(phi)] = Cfull[p:, p:]
        phi_ext = np.r_[phi, 0.0]
        self._fit = {"model": model, "fit": fit, "phi": phi, "phi_cov": Cfull[p:, p:], "beta": beta, "vcov": vcov_b,
                     "names": names, "covs": covs_abs, "levels": levs, "labels": labels, "clean": clean, "X": X, "y": y,
                     "lams": lams, "glm": glm}

        fam_lbl = {"binomial": "베르누이 · 로짓 연결 (melogit)", "poisson": "포아송 · 로그 연결 (mepoisson)"}[fam]
        info = ResultsTable(name="info", title="모형 정보 (Model information)")
        info.add_column("info", "항목", col_type="text")
        info.add_column("val", "값", col_type="text")
        info.add_row({"info": "분포 · 연결함수", "val": fam_lbl})
        info.add_row({"info": "적분 근사", "val": "라플라스 근사 (Stata intmethod(laplace), lme4 glmer nAGQ = 1)"})
        info.add_row({"info": "계산", "val": "자체 구현 추정"})
        info.add_row({"info": "종속변수", "val": self.dep})
        info.add_row({"info": "관측치 수 (Number of obs)", "val": f"{n:,}"})
        info.add_row({"info": "수준 구조 (상위 → 하위)", "val": " > ".join(self.levels) + " > 관측치"})
        if self.exposure_var:
            info.add_row({"info": "노출 변수 (exposure)", "val": f"ln({self.exposure_var})를 오프셋으로 사용"})
        elif self.offset_var:
            info.add_row({"info": "오프셋 (offset)", "val": self.offset_var})
        info.add_row({"info": "로그우도 (Log likelihood)", "val": f"{ll:.4f}"})
        info.add_row({"info": "AIC", "val": f"{aic:.3f}"})
        info.add_row({"info": "BIC", "val": f"{bic:.3f}"})
        wald = None
        idx_nc = [j for j, nm in enumerate(names) if nm != "Intercept"]
        if idx_nc and np.all(np.isfinite(vcov_b[np.ix_(idx_nc, idx_nc)])):
            bsub = beta[idx_nc]
            try:
                w = float(bsub @ np.linalg.solve(vcov_b[np.ix_(idx_nc, idx_nc)], bsub))
                wald = (w, len(idx_nc), float(stats.chi2.sf(w, len(idx_nc))))
                info.add_row({"info": f"Wald χ²({len(idx_nc)})", "val": f"{w:.2f}"})
                info.add_row({"info": "Prob > χ²", "val": f"{wald[2]:.4f}" if wald[2] >= 1e-4 else "< .0001"})
            except np.linalg.LinAlgError:
                pass
        info.add_row({"info": "수렴", "val": ("수렴" if fit["converged"] else "수렴 실패 — 결과 해석 주의")
                      + f" (최대 경사 {fit['gnorm']:.1e}, 함수 평가 {fit['nfev']}회)"})
        for t in notes_info:
            info.add_note(t)
        for t in derived:
            info.add_note(f"생성 변수: {t}")
        info.add_note("자체 구현 추정: 최대우도 다층 GLM을 제공하는 파이썬 라이브러리가 없어(statsmodels의 혼합 GLM은 베이즈 "
                      "변분·사후최빈 추정) 라플라스 근사 우도를 직접 계산 — Stata melogit, intmethod(laplace)와 로그우도·추정치 대조 검증")
        info.add_note("Stata melogit·mepoisson의 기본 적분법은 평균·분산 적응형 가우스-에르미트 7점(mvaghermite) — "
                      "라플라스 근사와 소수 셋째 자리 안팎의 차이 가능, 같은 조건 비교는 Stata intmethod(laplace)")
        if not fit["converged"]:
            info.add_note("최적화가 수렴 기준을 충족하지 못함 — 모형 단순화 또는 변수 척도 조정 검토")
        self.add_table(info)

        if self.want["groupInfo"]:
            gt = ResultsTable(name="groups", title="집단 구조 (Group structure)")
            gt.add_column("level", "집단 변수", col_type="text")
            gt.add_column("groups", "집단 수", col_type="integer")
            gt.add_column("min", "최소", col_type="integer", super_title="집단당 관측치")
            gt.add_column("avg", "평균", col_type="number", super_title="집단당 관측치")
            gt.add_column("max", "최대", col_type="integer", super_title="집단당 관측치")
            for k, g in enumerate(self.levels):
                cnt = np.bincount(codes[k])
                gt.add_row({"level": g, "groups": int(len(cnt)), "min": int(cnt.min()), "avg": float(cnt.mean()),
                            "max": int(cnt.max())})
            self.add_table(gt)

        scale = "로그오즈" if fam == "binomial" else "로그 기대빈도"
        exp_lbl = "OR" if fam == "binomial" else "IRR"
        fe = ResultsTable(name="fixed", title=f"고정효과 (Fixed effects, {scale}) — {self.dep}")
        fe.add_column("name", "항", col_type="text")
        fe.add_column("est", "계수", col_type="number")
        fe.add_column("se", "표준오차", col_type="number")
        fe.add_column("z", "z", col_type="number")
        fe.add_column("p", "p", col_type="number", fmt="pvalue")
        fe.add_column("cil", "하한", col_type="number", super_title=f"{ci_lbl} 신뢰구간")
        fe.add_column("ciu", "상한", col_type="number", super_title=f"{ci_lbl} 신뢰구간")
        if self.exp_b:
            fe.add_column("exp", exp_lbl, col_type="number")
            fe.add_column("exp_se", f"{exp_lbl} 표준오차", col_type="number")
            fe.add_column("exp_cil", "하한", col_type="number", super_title=f"{exp_lbl} {ci_lbl} 신뢰구간")
            fe.add_column("exp_ciu", "상한", col_type="number", super_title=f"{exp_lbl} {ci_lbl} 신뢰구간")
        for j, nm in enumerate(names):
            zv = beta[j] / se_b[j] if np.isfinite(se_b[j]) and se_b[j] > 0 else np.nan
            row = {"name": nm, "est": float(beta[j]), "se": float(se_b[j]), "z": float(zv),
                   "p": float(2.0 * stats.norm.sf(abs(zv))) if np.isfinite(zv) else np.nan,
                   "cil": float(beta[j] - zc * se_b[j]), "ciu": float(beta[j] + zc * se_b[j])}
            if self.exp_b:
                e = math.exp(beta[j])
                row.update({"exp": e, "exp_se": e * float(se_b[j]), "exp_cil": math.exp(row["cil"]),
                            "exp_ciu": math.exp(row["ciu"])})
            fe.add_row(row)
        fe.mark_coefficients(estimate="est", se="se", p="p", stat="z", ci=("cil", "ciu"), label="name",
                             exp_estimate="exp" if self.exp_b else None)
        fe.add_note("z 검정·정규 신뢰구간; 표준오차는 (β, 로그 표준편차) 전체의 관측 정보행렬(라플라스 로그우도 수치 Hessian) 역행렬")
        if self.exp_b:
            fe.add_note(f"{exp_lbl} = exp(계수), 표준오차는 델타법, 신뢰구간은 계수 구간의 지수 변환 (Stata {'or' if fam == 'binomial' else 'irr'} 옵션)")
        self.add_table(fe)

        self._random_table(levs, covs_abs, phi_ext, C, 1.0, zc, ci_lbl, boundary_lv, residual=False)

        if self.want["lrTest"]:
            ll0 = float(glm.llf)
            lr = max(0.0, 2.0 * (ll - ll0))
            lt = ResultsTable(name="lrtest", title=f"단일수준 {'로지스틱' if fam == 'binomial' else '포아송'} 회귀 대비 LR 검정")
            lt.add_column("stat", "검정", col_type="text")
            lt.add_column("chi2", "χ²", col_type="number")
            lt.add_column("df", "df", col_type="integer")
            lt.add_column("p", "p", col_type="number", fmt="pvalue")
            if n_vc == 1:
                pv = 0.5 * float(stats.chi2.sf(lr, 1)) if lr > 0 else 1.0
                lt.add_row({"stat": "chibar2(01)", "chi2": lr, "df": 1, "p": pv})
                lt.add_note("분산 성분 1개: 경계 검정 χ̄²(01), p = ½·P(χ²₁ > LR)")
            else:
                pv = float(stats.chi2.sf(lr, n_vc))
                lt.add_row({"stat": f"chi2({n_vc})", "chi2": lr, "df": n_vc, "p": pv})
                lt.add_note("분산 성분이 여러 개인 경계 검정: 보수적 검정이며 참고용 (Stata 주석과 동일)")
            self.add_table(lt)
            self._lr = (lr, n_vc, pv)

        icc_vals = None
        if fam == "binomial":
            icc_vals = self._icc_table(levs, covs_abs, phi_ext, C, 1.0, zc, ci_lbl, boundary_lv,
                                       gamma=math.pi ** 2 / 3.0, title_extra=" — 잠재 반응 척도")

        need_blup = self.want["blups"] or self.want["caterpillar"]
        if need_blup:
            u, _, _, eta = model.pirls(theta, beta, model._u)
            mu, w = model._mu_w(eta)
            w_obs = w[model.inv]                       # back to the analysis row order
            sw = np.sqrt(w_obs)
            wlevs = [_Level(lev.name, lev.terms, lev.Z * sw[:, None], lev.cov, lev.codes, lev.G, lev.parent) for lev in levs]
            aux = _LMM(np.zeros(n), X * sw[:, None], wlevs)
            _, blup_v = aux.blups(lams, np.zeros(p), 1.0, cond_var=True)
            blup_b = [uk @ L.T for uk, L in zip(u, lams)]
            self._fit["blups"] = (blup_b, blup_v)
            if self.want["blups"]:
                bt = ResultsTable(name="blups", title="확률효과 예측값 (조건부 최빈값)")
                bt.add_column("level", "수준", col_type="text")
                bt.add_column("group", "집단", col_type="text")
                bt.add_column("term", "항", col_type="text")
                bt.add_column("blup", "예측값", col_type="number")
                bt.add_column("se", "조건부 표준편차", col_type="number")
                for k, lev in enumerate(levs):
                    for gi in range(lev.G):
                        for j, term in enumerate(lev.terms):
                            bt.add_row({"level": lev.name, "group": labels[k][gi], "term": term,
                                        "blup": float(blup_b[k][gi, j]),
                                        "se": float(math.sqrt(max(blup_v[k][gi, j, j], 0.0)))})
                bt.add_note("조건부 최빈값 (lme4 ranef, Stata predict, reffects ebmodes); 조건부 표준편차는 "
                            "(ΛᵀZᵀŴZΛ + I)⁻¹ 기반 (lme4 condVar)")
                bt.add_note("집단 표기: 상위 수준 ID/하위 수준 ID")
                self.add_table(bt)
            if self.want["caterpillar"]:
                for k, lev in enumerate(levs):
                    src = self._caterpillar(lev, blup_b[k], blup_v[k], zc, labels[k])
                    if src:
                        self.add_plot(f"확률효과 예측값 — {lev.name} (caterpillar)", src)
        if self.want["residPlot"]:
            self.get_table("info").add_note("잔차 진단 그림은 선형 혼합모형에서만 제공")

        self.engine = "자체 구현 추정"
        self._fit["engine"] = self.engine
        self.model_stats = {"N": int(n), "Groups": int(levs[-1].G), "LL": float(ll), "AIC": float(aic), "BIC": float(bic)}
        if wald:
            self.model_stats["Chi2"] = (wald[0], wald[2])
            self.model_stats["df"] = int(wald[1])
        if icc_vals:
            self.model_stats["ICC"] = float(icc_vals[-1])
        grp_txt = ", ".join(f"{lev.name} {lev.G}개" for lev in levs)
        kind = "로지스틱" if fam == "binomial" else "포아송"
        self.apa_narrative = (f"{len(levs) + 1}수준 {kind} 혼합모형(라플라스 근사)으로 '{self.dep}'를 분석함"
                              f"(관측치 {n:,}개, {grp_txt}; 로그우도 = {ll:.2f}).")

    # ------------------------------------------------------------------ random-effects table
    def _vc_rows(self, lev: _Level, S: np.ndarray, ph: np.ndarray, Cl: np.ndarray, zc: float):
        """Rows (label_var, label_sd, var-metric (est, se, lo, hi), sd-metric (est, se, lo, hi)) in Stata order."""
        rows = []
        q = lev.q
        terms = lev.terms
        joined = " ".join(terms)

        def var_row(lnsd, se_lnsd):
            v = math.exp(2 * lnsd)
            sd = math.exp(lnsd)
            if np.isfinite(se_lnsd):
                return ((v, 2 * v * se_lnsd, math.exp(2 * (lnsd - zc * se_lnsd)), math.exp(2 * (lnsd + zc * se_lnsd))),
                        (sd, sd * se_lnsd, math.exp(lnsd - zc * se_lnsd), math.exp(lnsd + zc * se_lnsd)))
            return ((v, np.nan, np.nan, np.nan), (sd, np.nan, np.nan, np.nan))

        def se_of(i):
            return math.sqrt(Cl[i, i]) if np.isfinite(Cl[i, i]) and Cl[i, i] >= 0 else np.nan

        if lev.cov in ("independent", "unstructured"):
            for j in range(q):
                vm, sm = var_row(ph[j], se_of(j))
                rows.append((f"var({terms[j]})", f"sd({terms[j]})", vm, sm))
            if lev.cov == "unstructured":
                k = q
                for i in range(q):
                    for j in range(i + 1, q):
                        a = ph[k]
                        r = math.tanh(a)
                        sdi, sdj = math.exp(ph[i]), math.exp(ph[j])
                        cov = r * sdi * sdj
                        # delta method for the covariance (gradient in φ: ln sd_i, ln sd_j, atanh r)
                        g = np.zeros(len(ph))
                        g[i] = cov
                        g[j] = cov
                        g[k] = (1 - r * r) * sdi * sdj
                        Csub = Cl
                        var_cov = float(g @ np.nan_to_num(Csub, nan=0.0) @ g) if np.all(np.isfinite(Csub[np.ix_([i, j, k], [i, j, k])])) else np.nan
                        se_cov = math.sqrt(var_cov) if np.isfinite(var_cov) and var_cov >= 0 else np.nan
                        se_a = se_of(k)
                        cm = (cov, se_cov, cov - zc * se_cov, cov + zc * se_cov)
                        rm = (r, (1 - r * r) * se_a, math.tanh(a - zc * se_a), math.tanh(a + zc * se_a)) if np.isfinite(se_a) \
                            else (r, np.nan, np.nan, np.nan)
                        rows.append((f"cov({terms[i]},{terms[j]})", f"corr({terms[i]},{terms[j]})", cm, rm))
                        k += 1
        elif lev.cov == "identity":
            vm, sm = var_row(ph[0], se_of(0))
            rows.append((f"var({joined})", f"sd({joined})", vm, sm))
        else:   # exchangeable
            vm, sm = var_row(ph[0], se_of(0))
            rows.append((f"var({joined})", f"sd({joined})", vm, sm))
            a = ph[1]
            r = math.tanh(a)
            v = math.exp(2 * ph[0])
            cov = r * v
            ok = np.all(np.isfinite(Cl[:2, :2]))
            if ok:
                g = np.array([2 * cov, (1 - r * r) * v])
                se_cov = math.sqrt(max(float(g @ Cl[:2, :2] @ g), 0.0))
                se_a = math.sqrt(Cl[1, 1])
                cm = (cov, se_cov, cov - zc * se_cov, cov + zc * se_cov)
                rm = (r, (1 - r * r) * se_a, math.tanh(a - zc * se_a), math.tanh(a + zc * se_a))
            else:
                cm = (cov, np.nan, np.nan, np.nan)
                rm = (r, np.nan, np.nan, np.nan)
            rows.append((f"cov({joined})", f"corr({joined})", cm, rm))
        return rows

    def _random_table(self, levs, covs_abs, phi, C, s2, zc, ci_lbl, boundary_lv, residual: bool = True):
        mode = self.re_stat
        rt = ResultsTable(name="random", title="확률효과 모수 (Random-effects parameters)")
        rt.add_column("grp", "수준 · 공분산 구조", col_type="text")
        rt.add_column("name", "모수", col_type="text")
        groups = [("var", "분산·공분산"), ("sd", "표준편차·상관")] if mode == "both" else [(mode, "")]
        for key, st in groups:
            pre = "" if mode != "both" else f"{key}_"
            rt.add_column(f"{pre}est", "추정치", col_type="number", super_title=st)
            rt.add_column(f"{pre}se", "표준오차", col_type="number", super_title=st)
            rt.add_column(f"{pre}cil", "하한", col_type="number", super_title=(st + " " if st else "") + f"{ci_lbl} 신뢰구간")
            rt.add_column(f"{pre}ciu", "상한", col_type="number", super_title=(st + " " if st else "") + f"{ci_lbl} 신뢰구간")
        pos = 0
        self._vc_records = []
        for lev, S in zip(levs, covs_abs):
            ph = phi[pos:pos + lev.n_phi]
            Cl = C[pos:pos + lev.n_phi, pos:pos + lev.n_phi]
            grp_lbl = f"{lev.name}: {COV_LABEL[lev.cov]}"
            for i, (lv, ls, vm, sm) in enumerate(self._vc_rows(lev, S, ph, Cl, zc)):
                row = {"grp": grp_lbl if i == 0 else "", "name": lv if mode == "var" else (ls if mode == "sd" else lv.split("(", 1)[1][:-1])}
                for key, _ in groups:
                    pre = "" if mode != "both" else f"{key}_"
                    vals = vm if key == "var" else sm
                    row.update({f"{pre}est": vals[0], f"{pre}se": vals[1], f"{pre}cil": vals[2], f"{pre}ciu": vals[3]})
                rt.add_row(row)
                self._vc_records.append({"level": lev.name, "var_label": lv, "sd_label": ls, "var": vm, "sd": sm})
            pos += lev.n_phi
        if not residual:
            rt.add_note("표준오차: (β, 로그 표준편차·atanh 상관) 척도의 관측 정보행렬 역행렬 → 델타법; 신뢰구간은 그 척도에서 "
                        "구한 뒤 역변환 (Stata 방식)")
            if boundary_lv:
                rt.add_note(f"경계 추정치(분산≈0 또는 상관≈±1): {', '.join(boundary_lv)} 수준 — 표준오차·신뢰구간 산출 생략")
            self.add_table(rt)
            return
        lnse = math.sqrt(C[-1, -1]) if np.isfinite(C[-1, -1]) else np.nan
        lnsd = phi[-1]
        v = math.exp(2 * lnsd)
        sd = math.exp(lnsd)
        vm = (v, 2 * v * lnse, math.exp(2 * (lnsd - zc * lnse)), math.exp(2 * (lnsd + zc * lnse)))
        sm = (sd, sd * lnse, math.exp(lnsd - zc * lnse), math.exp(lnsd + zc * lnse))
        row = {"grp": "잔차", "name": "var(Residual)" if mode == "var" else ("sd(Residual)" if mode == "sd" else "Residual")}
        for key, _ in groups:
            pre = "" if mode != "both" else f"{key}_"
            vals = vm if key == "var" else sm
            row.update({f"{pre}est": vals[0], f"{pre}se": vals[1], f"{pre}cil": vals[2], f"{pre}ciu": vals[3]})
        rt.add_row(row)
        self._vc_records.append({"level": "Residual", "var_label": "var(Residual)", "sd_label": "sd(Residual)", "var": vm, "sd": sm})
        rt.add_note("표준오차: 로그 표준편차·atanh 상관 척도에서 (제한) 로그우도의 수치 Hessian(β 프로파일) 역행렬 → 델타법; "
                    "분산·표준편차·상관의 신뢰구간은 그 척도에서 구한 뒤 역변환, 공분산은 대칭 Wald 구간 (Stata mixed 방식)")
        if boundary_lv:
            rt.add_note(f"경계 추정치(분산≈0 또는 상관≈±1): {', '.join(boundary_lv)} 수준 — 해당 수준 모수의 표준오차·신뢰구간 산출 생략, "
                        "확률효과 구조 단순화 검토")
        self.add_table(rt)

    # ------------------------------------------------------------------ ICC
    def _icc_table(self, levs, covs_abs, phi, C, s2, zc, ci_lbl, boundary_lv, gamma: Optional[float] = None,
                   title_extra: str = "") -> Optional[List[float]]:
        """Stata estat icc: level-g ICC = Σ_{l ≥ g} σ²_l(_cons) / (γ + Σ_l σ²_l(_cons)); γ = σ²_e (linear) or π²/3 (logit).

        Random-coefficient models give conditional ICCs (random-effect covariates at 0), as Stata reports them.
        """
        K = len(levs)
        idx: List[Optional[int]] = []
        pos = 0
        for lev in levs:
            if "_cons" not in lev.terms:
                idx.append(None)
            elif lev.cov in ("identity", "exchangeable"):
                idx.append(pos)
            else:
                idx.append(pos + lev.terms.index("_cons"))
            pos += lev.n_phi
        if any(i is None for i in idx):
            if self.want["icc"]:
                self.get_table("random").add_note("확률절편이 없는 수준이 있어 ICC 생략 (Stata estat icc와 동일)")
            return None
        par = [i for i in idx if i is not None] + ([len(phi) - 1] if gamma is None else [])
        Csub = C[np.ix_(par, par)]
        v_lev = [math.exp(2 * phi[i]) if i is not None else 0.0 for i in idx]
        resid = math.exp(2 * phi[-1]) if gamma is None else float(gamma)
        tot = sum(v_lev) + resid
        conditional = any(lev.terms != ["_cons"] for lev in levs)
        iccs, ses = [], []
        for k in range(K):
            num = sum(v_lev[:k + 1])
            g = []
            for j in range(K):
                if idx[j] is None:
                    continue
                dnum = 2 * v_lev[j] if j <= k else 0.0
                g.append((dnum * tot - num * 2 * v_lev[j]) / tot ** 2)
            if gamma is None:
                g.append(-num * 2 * resid / tot ** 2)
            g = np.array(g)
            var_icc = float(g @ Csub @ g) if np.all(np.isfinite(Csub)) else np.nan
            iccs.append(num / tot)
            ses.append(math.sqrt(var_icc) if np.isfinite(var_icc) and var_icc >= 0 else np.nan)
        if self.want["icc"]:
            kind = "조건부 급내상관" if conditional else "급내상관"
            it = ResultsTable(name="icc", title=f"{kind} (Intraclass correlation · estat icc){title_extra}")
            it.add_column("level", "수준", col_type="text")
            it.add_column("icc", "ICC", col_type="number", fmt="zto")
            it.add_column("se", "표준오차", col_type="number")
            it.add_column("cil", "하한", col_type="number", super_title=f"{ci_lbl} 신뢰구간")
            it.add_column("ciu", "상한", col_type="number", super_title=f"{ci_lbl} 신뢰구간")
            for k in range(K):
                lbl = "|".join(levs[j].name for j in range(k, -1, -1))
                icc, se = iccs[k], ses[k]
                if np.isfinite(se) and 0 < icc < 1:
                    lg = math.log(icc / (1 - icc))
                    slg = se / (icc * (1 - icc))
                    lo, hi = 1 / (1 + math.exp(-(lg - zc * slg))), 1 / (1 + math.exp(-(lg + zc * slg)))
                else:
                    lo = hi = np.nan
                it.add_row({"level": lbl, "icc": icc, "se": se, "cil": lo, "ciu": hi})
            names = [lev.name for lev in levs]
            res_txt = "잔차 분산" if gamma is None else "π²/3 (로지스틱 잠재 반응의 잔차 분산)"
            it.add_note(f"수준 k의 ICC = (그 수준과 상위 수준의 절편 분산 합) / (모든 절편 분산 + {res_txt}) — "
                        "같은 집단에 속한 관측치 간 상관 (Stata estat icc)")
            if K > 1:
                it.add_note(f"예: '{names[0]}' ICC는 같은 {names[0]}·다른 {names[1]} 관측치 간 상관, "
                            f"'{names[1]}|{names[0]}' ICC는 같은 {names[1]} 관측치 간 상관")
            if conditional:
                it.add_note("확률기울기 포함 모형: 확률효과 공변량 값 0 기준의 조건부 ICC (Stata conditional intraclass correlation)")
            it.add_note("표준오차: 델타법, 신뢰구간: 로짓 척도에서 계산 후 역변환 (Stata 방식)")
            if boundary_lv:
                it.add_note("경계 추정치가 있어 일부 표준오차 생략")
            self.add_table(it)
        return iccs

    # ------------------------------------------------------------------ plots
    def _caterpillar(self, lev: _Level, b: np.ndarray, v: np.ndarray, zc: float, labels: List[str]) -> str:
        import matplotlib.pyplot as plt
        from .plot_style import finalize_and_serialize_figure, apply_style_to_axes, get_active_style
        q = lev.q
        G = lev.G
        col = get_active_style()["colors"]
        fig, axes = plt.subplots(1, q, figsize=(min(4.2 * q, 12.6), max(3.6, min(9.0, 0.045 * G + 2.6))), squeeze=False)
        for j in range(q):
            ax = axes[0, j]
            est = b[:, j]
            se = np.sqrt(np.maximum(v[:, j, j], 0.0))
            order = np.argsort(est)
            ypos = np.arange(G)
            ax.errorbar(est[order], ypos, xerr=zc * se[order], fmt="o", ms=3.2 if G <= 120 else 2.0,
                        color=col[0], ecolor=col[1 % len(col)], elinewidth=0.8, capsize=0)
            ax.axvline(0.0, color="#64748B", lw=0.9, ls="--")
            ax.set_title(lev.terms[j], fontsize=10.5)
            ax.set_xlabel("예측값 ± 신뢰구간")
            if G <= 40:
                ax.set_yticks(ypos)
                ax.set_yticklabels([labels[i] for i in order], fontsize=7.5)
            else:
                ax.set_yticks([])
                ax.set_ylabel(f"{lev.name} ({G}개, 예측값 순)")
            apply_style_to_axes(ax)
        fig.suptitle(f"{lev.name} 수준 확률효과 예측값", fontsize=11.5, fontweight="bold")
        fig.tight_layout()
        return finalize_and_serialize_figure(fig)

    def _resid_plot(self, fitted: np.ndarray, resid: np.ndarray) -> str:
        import matplotlib.pyplot as plt
        from .plot_style import finalize_and_serialize_figure, apply_style_to_axes, get_active_style
        col = get_active_style()["colors"]
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
        ax = axes[0]
        ax.scatter(fitted, resid, s=8 if len(resid) > 2000 else 14, alpha=0.5, color=col[0], edgecolors="none")
        ax.axhline(0.0, color="#334155", lw=1.0, ls="--")
        ax.set_xlabel("적합값 (고정효과 + 확률효과)")
        ax.set_ylabel("조건부 잔차")
        ax.set_title("잔차 대 적합값")
        apply_style_to_axes(ax)
        ax = axes[1]
        sd = np.std(resid, ddof=1) if len(resid) > 1 else 1.0
        stats.probplot(resid / (sd if sd > 0 else 1.0), dist="norm", plot=ax)
        ax.get_lines()[0].set_markerfacecolor(col[0])
        ax.get_lines()[0].set_markeredgecolor(col[0])
        ax.get_lines()[0].set_markersize(3)
        ax.get_lines()[1].set_color(col[1 % len(col)])
        ax.set_title("정규 Q-Q (표준화 잔차)")
        ax.set_xlabel("이론 분위수")
        ax.set_ylabel("표본 분위수")
        apply_style_to_axes(ax)
        fig.tight_layout()
        return finalize_and_serialize_figure(fig)


# ---------------------------------------------------------------------------------------------------------------
# Parameters (UI camelCase / SDK snake_case / stored options) -> normalized specification
# ---------------------------------------------------------------------------------------------------------------
def _normalize(p: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(p or {})
    dep = _as_single(p.get("dep"))
    notes: List[str] = []
    lv = p.get("levels")
    cl_raw = p.get("cluster") if p.get("cluster") not in (None, "", []) else (p.get("random") or p.get("group"))
    cl = _as_list(cl_raw) if isinstance(cl_raw, (list, tuple)) else ([str(_as_single(cl_raw))] if _as_single(cl_raw) else [])
    c3 = _as_single(_opt(p, "cluster3", "cluster_3", None))
    c4 = _as_single(_opt(p, "cluster4", "cluster_4", None))
    if lv not in (None, "", []):
        levels = _as_list(lv)
        if len(cl) == 1 and cl[0] != levels[-1]:
            raise ValueError(f"다층모형: levels의 최하위 집단 '{levels[-1]}'와 cluster '{cl[0]}' 불일치")
    elif len(cl) > 1:
        levels = cl
    else:
        if c4 and not c3:
            raise ValueError("다층모형: 4수준 집단 변수(cluster4)는 3수준 집단 변수(cluster3)와 함께 지정")
        if (c3 or c4) and not cl:
            raise ValueError("다층모형: 3·4수준 집단 변수를 쓰려면 2수준 집단 변수(cluster) 지정 필요")
        levels = [str(x) for x in (c4, c3) if x] + cl
    K = len(levels)

    def level_of(num: int) -> Optional[str]:     # level number 2, 3, 4 -> grouping variable
        i = K - (num - 1)
        return levels[i] if 0 <= i < K else None

    slopes: Dict[str, List[str]] = {}

    def add_slopes(g: str, vs: List[str]):
        cur = slopes.setdefault(g, [])
        for v in vs:
            if v not in cur:
                cur.append(v)

    rs = _opt(p, "randomSlopes", "random_slopes", None)
    if isinstance(rs, dict):
        for g, vs in rs.items():
            add_slopes(str(g), _as_list(vs))
    sl = p.get("slopes")
    if isinstance(sl, dict):
        for g, vs in sl.items():
            add_slopes(str(g), _as_list(vs))
    elif sl not in (None, "", []) and K:
        add_slopes(levels[-1], _as_list(sl))
    for num in (2, 3, 4):
        vs = _as_list(_opt(p, f"slopes{num}", f"slopes_{num}", None))
        if vs:
            g = level_of(num)
            if g is None:
                notes.append(f"{num}수준 집단 변수가 없어 slopes{num} 설정({', '.join(vs)}) 미적용")
            else:
                add_slopes(g, vs)

    cov: Dict[str, str] = {}
    cv = _opt(p, "covariance", "cov_struct", None)
    if isinstance(cv, dict):
        cov.update({str(k): v for k, v in cv.items()})
    elif cv not in (None, ""):
        cov.update({g: cv for g in levels})
    for num in (2, 3, 4):
        v = _opt(p, f"covStruct{num}", f"cov_struct{num}", None)
        if v not in (None, ""):
            g = level_of(num)
            if g is not None:
                cov[g] = v

    nocons: Dict[str, bool] = {}
    nc = _opt(p, "noconstant", "no_constant", None)
    if isinstance(nc, dict):
        nocons.update({str(k): bool(v) for k, v in nc.items()})
    elif isinstance(nc, (list, tuple)):
        nocons.update({str(g): True for g in nc})
    elif isinstance(nc, str) and nc.strip():
        if nc.strip().lower() in ("true", "1", "yes"):
            nocons.update({g: True for g in levels})
        elif nc.strip().lower() not in ("false", "0", "no"):
            nocons.update({g.strip(): True for g in nc.split(",") if g.strip()})
    elif isinstance(nc, bool) and nc:
        nocons.update({g: True for g in levels})
    for num in (2, 3, 4):
        if _flag(p, f"noConstant{num}", f"no_constant{num}", False):
            g = level_of(num)
            if g is None:
                notes.append(f"{num}수준 집단 변수가 없어 noConstant{num} 설정 미적용")
            else:
                nocons[g] = True

    method = _opt(p, "method", "estimation", None)
    if method is None:
        method = "reml" if _flag(p, "reml", "reml", True) else "ml"
    if _flag(p, "stddeviations", "std_deviations", False):
        re_stat = "sd"
    else:
        re_stat = _opt(p, "reStat", "re_stat", "var")
    spec = {
        "dep": dep, "levels": levels, "covs": _as_list(p.get("covs")), "factors": _as_list(p.get("factors")),
        "interactions": _as_list(_opt(p, "interactions", "interaction_terms", None)),
        "slopes": slopes, "cov_struct": cov, "noconstant": nocons, "method": method, "re_stat": re_stat,
        "icc": _flag(p, "icc", "icc", True), "lr_test": _flag(p, "lrTest", "lr_test", True),
        "group_info": _flag(p, "groupInfo", "group_info", True), "blups": _flag(p, "blups", "blup", False),
        "caterpillar": _flag(p, "caterpillar", "caterpillar_plot", False),
        "resid_plot": _flag(p, "residPlot", "resid_plot", False),
        "center_grand": _as_list(_opt(p, "centerGrand", "center_grand", None)),
        "center_group": _as_list(_opt(p, "centerGroup", "center_group", None)),
        "group_means": _as_list(_opt(p, "groupMeans", "group_means", None)),
        "center_level": _as_single(_opt(p, "centerLevel", "center_level", None)),
        "ci_width": float(_opt(p, "ciWidth", "ci_width", 95.0)),
        "max_iter": int(_opt(p, "maxIter", "max_iter", 2000)),
        "family": _opt(p, "family", "family", "gaussian"),
        "exp_b": _flag(p, "expB", "exp_b", False) or _flag(p, "oddsRatio", "or", False) or _flag(p, "irr", "irr", False),
        "offset": _as_single(_opt(p, "offset", "offset", None)),
        "exposure": _as_single(_opt(p, "exposure", "exposure", None)),
    }
    cl = spec["center_level"]
    if isinstance(cl, str) and re.fullmatch(r"(level)?[234]", cl.strip().lower()):
        num = int(cl.strip()[-1])
        spec["center_level"] = level_of(num)
        if spec["center_level"] is None and levels:
            spec["center_level"] = levels[-1]
            if spec["center_group"] or spec["group_means"]:
                notes.append(f"{num}수준 집단 변수가 없어 집단평균 기준을 2수준 집단 '{levels[-1]}'로 적용")
    return {"spec": spec, "notes": notes}


def _run_from_params(df: pd.DataFrame, params: Dict[str, Any]) -> BaseAnalysis:
    norm = _normalize(params)
    spec = norm["spec"]
    if not spec["dep"]:
        raise ValueError(f"다층모형: 종속변수(dep) 지정 필요. 사용 가능한 변수: {list(df.columns)}")
    if not spec["levels"]:
        raise ValueError(f"다층모형: 집단 변수(cluster 또는 levels) 지정 필요. 사용 가능한 변수: {list(df.columns)}")
    res = MultilevelLinearModel(df, **spec)
    for t in norm["notes"]:
        res.get_table("info").add_note(t)
    return res


# ---------------------------------------------------------------------------------------------------------------
# Code export (Stata .do / R)
# ---------------------------------------------------------------------------------------------------------------
_STATA_COV = {"independent": "independent", "unstructured": "unstructured", "identity": "identity",
              "exchangeable": "exchangeable"}


def _code_spec(params: Dict[str, Any]) -> Dict[str, Any]:
    spec = _normalize(params)["spec"]
    levels = spec["levels"]
    spec["slopes"] = {g: spec["slopes"].get(g, []) for g in levels}
    spec["cov_struct"] = {g: resolve_choice(f"공분산 구조({g})", spec["cov_struct"].get(g), COV_STRUCTS, _COV_ALIASES,
                                            default="independent") for g in levels}
    spec["noconstant"] = {g: bool(spec["noconstant"].get(g, False)) for g in levels}
    spec["family"] = resolve_choice("분포(family)", spec["family"], FAMILIES, _FAMILY_ALIASES, default="gaussian")
    if spec["family"] == "gaussian":
        spec["method"] = resolve_choice("추정 방법(method)", spec["method"], _METHODS, _METHOD_ALIASES, default="reml")
    else:
        spec["method"] = "laplace"
    spec["re_stat"] = resolve_choice("확률효과 표시(reStat)", spec["re_stat"], _RESTATS, _RESTAT_ALIASES, default="var")
    if not spec["center_level"] and levels:
        spec["center_level"] = levels[-1]
    return spec


def _derived_names(spec: Dict[str, Any]) -> Dict[str, str]:
    ren = {v: f"{v}_cgm" for v in spec["center_grand"]}
    ren.update({v: f"{v}_cwc" for v in spec["center_group"]})
    return ren


def _fixed_terms(spec: Dict[str, Any]):
    """[(kind, payload)] in model order: ('var', v) | ('mean', v) | ('inter', [vars])."""
    out, seen = [], []
    inter = []
    base = list(dict.fromkeys(spec["covs"] + spec["factors"]))
    for term in spec["interactions"]:
        parts, full = _split_terms(term)
        if full:
            for v in parts:
                if v not in base:
                    base.append(v)
            from itertools import combinations
            for r_ in range(2, len(parts) + 1):
                inter += [list(c) for c in combinations(parts, r_)]
        else:
            inter.append(parts)
    for v in base:
        out.append(("var", v))
        if v in spec["group_means"]:
            out.append(("mean", v))
    for v in spec["group_means"]:
        if v not in base:
            out.append(("mean", v))
    for parts in inter:
        out.append(("inter", parts))
    return out


def _all_vars(spec: Dict[str, Any]) -> List[str]:
    vs = [spec["dep"]] + spec["levels"]
    for kind, pl in _fixed_terms(spec):
        vs += (pl if kind == "inter" else [pl])
    for g in spec["levels"]:
        vs += spec["slopes"][g]
    vs += spec["center_grand"] + spec["center_group"] + spec["group_means"]
    return list(dict.fromkeys(v for v in vs if v))


def stata_code(t: str, params: Dict[str, Any]) -> Optional[List[str]]:
    if t != "mixedModel":
        return None
    sp = _code_spec(params)
    if not sp["dep"] or not sp["levels"]:
        return None
    levels = sp["levels"]
    ren = _derived_names(sp)
    fac = set(sp["factors"])
    out = []
    centering = sp["center_grand"] or sp["center_group"] or sp["group_means"]
    touse = ""
    if centering:
        out.append(f"generate byte __touse = !missing({', '.join(_all_vars(sp))})")
        touse = " if __touse"
        path = " ".join(levels[:levels.index(sp["center_level"]) + 1])
        for v in sp["center_grand"]:
            out += [f"summarize {v} if __touse, meanonly", f"generate double {v}_cgm = {v} - r(mean) if __touse"]
        for v in dict.fromkeys(sp["center_group"] + sp["group_means"]):
            out.append(f"egen double __m_{v} = mean({v}) if __touse, by({path})")
        for v in sp["center_group"]:
            out.append(f"generate double {v}_cwc = {v} - __m_{v} if __touse")
        for v in sp["group_means"]:
            out.append(f"generate double {v}_{sp['center_level']}평균 = __m_{v} if __touse")

    def nm(v: str, prefix: bool) -> str:
        if v in fac:
            return f"i.{v}" if prefix else f"i.{v}"
        return f"c.{ren.get(v, v)}" if prefix else ren.get(v, v)

    fx = []
    for kind, pl in _fixed_terms(sp):
        if kind == "var":
            fx.append(nm(pl, False))
        elif kind == "mean":
            fx.append(f"{pl}_{sp['center_level']}평균")
        else:
            fx.append("#".join(nm(v, True) for v in pl))
    if fac:
        out.insert(0, "* i. 요인변수는 숫자 코드여야 함 (문자형이면 encode 후 지정)")
    stata_cmd = {"gaussian": "mixed", "binomial": "melogit", "poisson": "mepoisson"}[sp["family"]]
    cmd = f"{stata_cmd} {sp['dep']}" + (" " + " ".join(fx) if fx else "") + touse
    fe_opts = []
    if sp["family"] != "gaussian" and sp["offset"]:
        fe_opts.append(f"offset({sp['offset']})")
    if sp["family"] == "poisson" and sp["exposure"]:
        fe_opts.append(f"exposure({sp['exposure']})")
    if fe_opts:
        cmd += ", " + " ".join(fe_opts)                      # offset()/exposure() belong to the fixed-effects equation
    for g in levels:
        sl = [ren.get(v, v) for v in sp["slopes"][g]]
        opts = []
        q = len(sl) + (0 if sp["noconstant"][g] else 1)
        if q > 1 and sp["cov_struct"][g] != "independent":
            opts.append(f"covariance({_STATA_COV[sp['cov_struct'][g]]})")
        if sp["noconstant"][g]:
            opts.append("noconstant")
        cmd += f" || {g}:" + (" " + " ".join(sl) if sl else "") + (", " + " ".join(opts) if opts else "")
    if sp["family"] == "gaussian":
        gopts = ["reml" if sp["method"] == "reml" else "mle"]
    else:
        gopts = ["intmethod(laplace)"]
        if sp["exp_b"]:
            gopts.append("or" if sp["family"] == "binomial" else "irr")
    if sp["re_stat"] == "sd":
        gopts.append("stddeviations")
    if sp["ci_width"] != 95.0:
        gopts.append(f"level({sp['ci_width']:g})")
    cmd += ", " + " ".join(gopts)
    out.append(cmd)
    if sp["family"] != "poisson" and not any(sp["noconstant"][g] for g in levels):
        out.append("estat icc")
    out.append("estat ic")
    if sp["blups"] or sp["caterpillar"]:
        names = " ".join(f"__re{i + 1}" for i in range(sum(len(sp["slopes"][g]) + (0 if sp["noconstant"][g] else 1) for g in levels)))
        eb = " ebmodes" if sp["family"] != "gaussian" else ""
        out.append(f"predict {names}, reffects{eb} reses({' '.join(n_ + 'se' for n_ in names.split())})")
    return out


def _rq(name: str) -> str:
    return name if re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", str(name)) else f"`{name}`"


def _rs(name: str) -> str:
    return '"' + str(name).replace('"', '\\"') + '"'


def r_code(t: str, params: Dict[str, Any]) -> Optional[List[str]]:
    if t != "mixedModel":
        return None
    sp = _code_spec(params)
    if not sp["dep"] or not sp["levels"]:
        return None
    levels = sp["levels"]
    ren = _derived_names(sp)
    fac = set(sp["factors"])
    allv = _all_vars(sp) + [v for v in (sp["offset"], sp["exposure"]) if v]
    out = [f"data <- data[stats::complete.cases(data[, c({', '.join(_rs(v) for v in allv)})]), ]"]
    path = levels[:levels.index(sp["center_level"]) + 1] if sp["center_level"] in levels else levels
    grp_args = ", ".join(f"data[[{_rs(g)}]]" for g in path)
    for v in sp["center_grand"]:
        out.append(f"data[[{_rs(v + '_cgm')}]] <- data[[{_rs(v)}]] - mean(data[[{_rs(v)}]])")
    for v in sp["center_group"]:
        out.append(f"data[[{_rs(v + '_cwc')}]] <- data[[{_rs(v)}]] - ave(data[[{_rs(v)}]], {grp_args})")
    for v in sp["group_means"]:
        out.append(f"data[[{_rs(v + '_' + sp['center_level'] + '평균')}]] <- ave(data[[{_rs(v)}]], {grp_args})")

    def rn(v: str) -> str:
        return f"factor({_rq(v)})" if v in fac else _rq(ren.get(v, v))

    fx = []
    for kind, pl in _fixed_terms(sp):
        if kind == "var":
            fx.append(rn(pl))
        elif kind == "mean":
            fx.append(_rq(f"{pl}_{sp['center_level']}평균"))
        else:
            fx.append(":".join(rn(v) for v in pl))
    if sp["offset"]:
        fx.append(f"offset({_rq(sp['offset'])})")
    if sp["exposure"]:
        fx.append(f"offset(log({_rq(sp['exposure'])}))")
    rhs = " + ".join(fx) if fx else "1"
    reml = "TRUE" if sp["method"] == "reml" else "FALSE"
    if sp["family"] != "gaussian":
        re_terms = []
        for k, g in enumerate(levels):
            grp = ":".join(_rq(x) for x in levels[:k + 1])
            sl = [_rq(ren.get(v, v)) for v in sp["slopes"][g]]
            lhs = " + ".join((["0"] if sp["noconstant"][g] else ["1"]) + sl)
            bar = "||" if (sl and sp["cov_struct"][g] == "independent") else "|"
            re_terms.append(f"({lhs} {bar} {grp})")
        out.append(f"fit <- lme4::glmer({_rq(sp['dep'])} ~ {rhs} + {' + '.join(re_terms)}, data = data, "
                   f"family = {sp['family']}, nAGQ = 1)")
        out.append("summary(fit)")
        if sp["exp_b"]:
            out.append("exp(lme4::fixef(fit))")
        if any(sp["cov_struct"][g] in ("identity", "exchangeable") and
               len(sp["slopes"][g]) + (0 if sp["noconstant"][g] else 1) > 1 for g in levels):
            out.append("# lme4는 동일·교환 공분산 구조를 직접 지원하지 않음 — 위 식은 비구조/독립 구조로 근사")
        return out
    use_nlme = any(sp["cov_struct"][g] in ("identity", "exchangeable") and
                   len(sp["slopes"][g]) + (0 if sp["noconstant"][g] else 1) > 1 for g in levels)
    if not use_nlme:
        re_terms = []
        for k, g in enumerate(levels):
            grp = ":".join(_rq(x) for x in levels[:k + 1])
            sl = [_rq(ren.get(v, v)) for v in sp["slopes"][g]]
            lhs = " + ".join((["0"] if sp["noconstant"][g] else ["1"]) + sl)
            bar = "||" if (sl and sp["cov_struct"][g] == "independent") else "|"
            re_terms.append(f"({lhs} {bar} {grp})")
        out.append(f"fit <- lme4::lmer({_rq(sp['dep'])} ~ {rhs} + {' + '.join(re_terms)}, data = data, REML = {reml})")
        out.append("summary(fit)")
        out.append("lme4::VarCorr(fit)")
        if sp["blups"] or sp["caterpillar"]:
            out.append("lme4::ranef(fit, condVar = TRUE)")
    else:
        pd_map = {"unstructured": "pdSymm", "independent": "pdDiag", "identity": "pdIdent", "exchangeable": "pdCompSymm"}
        parts = []
        for g in levels:
            sl = [_rq(ren.get(v, v)) for v in sp["slopes"][g]]
            f = "~ " + (" + ".join(sl) + " - 1" if sp["noconstant"][g] else (" + ".join(sl) if sl else "1"))
            parts.append(f"{_rq(g)} = nlme::{pd_map[sp['cov_struct'][g]]}({f})")
        out.append(f"fit <- nlme::lme({_rq(sp['dep'])} ~ {rhs}, random = list({', '.join(parts)}), data = data, "
                   f"method = {_rs('REML' if reml == 'TRUE' else 'ML')})")
        out.append("summary(fit)")
        out.append("nlme::intervals(fit, which = \"var-cov\")")
    return out


# ---------------------------------------------------------------------------------------------------------------
# Stata command -> Python (mixed / xtmixed, estat icc)
# ---------------------------------------------------------------------------------------------------------------
_NAME = r"[A-Za-z_\u0080-￿][\w\u0080-￿]*"


def _ni(msg: str, line: str) -> str:
    return f"raise NotImplementedError({(msg + ' — ' + line)!r})"


def _split_opts(seg: str) -> Tuple[str, List[Tuple[str, str]]]:
    """'school: x1, cov(un) nocons' -> ('school: x1', [('cov', 'un'), ('nocons', '')])."""
    depth, cut = 0, None
    for i, ch in enumerate(seg):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            cut = i
            break
    if cut is None:
        return seg.strip(), []
    main, rest = seg[:cut].strip(), seg[cut + 1:]
    opts = []
    for m in re.finditer(r"([A-Za-z_]\w*)\s*(?:\(([^()]*(?:\([^()]*\)[^()]*)*)\))?", rest):
        opts.append((m.group(1).lower(), (m.group(2) or "").strip()))
    leftover = re.sub(r"([A-Za-z_]\w*)\s*(?:\(([^()]*(?:\([^()]*\)[^()]*)*)\))?", "", rest).strip(" ,\t")
    if leftover:
        opts.append(("__bad__", leftover))
    return main, opts


def _abbr(word: str, full: str, minlen: int) -> bool:
    return len(word) >= minlen and full.startswith(word)


def transpile(line: str) -> Optional[str]:
    s = line.strip()
    low = s.lower()
    if re.match(r"^estat\s+icc\b", low):
        return ("# [Stata] estat icc → 혼합모형 결과의 ICC 표(icc=True, 기본 포함; 확률기울기 모형은 조건부 ICC) — "
                "level(#) 대신 분석 옵션 ciWidth 사용")
    if re.match(r"^estat\s+(group|recov|recovariance)\b", low):
        return "# [Stata] " + s + " → 혼합모형 결과의 집단 구조·확률효과 모수 표에 포함"
    if re.match(r"^estat\s+sd\b", low):
        return "# [Stata] estat sd → 분석 옵션 reStat='sd' (확률효과 표준편차·상관 표시) 또는 reStat='both'"
    if re.match(r"^estat\s+(wcor|wcorrelation)\b", low):
        return _ni("estat wcorrelation 미지원 (관측치 수준 공분산 행렬)", s)
    m = re.match(r"^(mixed|xtmixed|melogit|mepoisson)\s+(.*)$", s, re.IGNORECASE)
    if not m:
        return None
    cmd = m.group(1).lower()
    family = {"melogit": "binomial", "mepoisson": "poisson"}.get(cmd, "gaussian")
    body = m.group(2)
    if re.search(r"\[\s*(pw|pweight|fw|fweight|aw|iw)\b|\[", body, re.IGNORECASE):
        return _ni(f"{cmd} 가중치 미지원", s)
    segs = [x.strip() for x in body.split("||")]
    if len(segs) < 2:
        return _ni("mixed: 확률효과 수준(|| 집단변수:) 지정 필요 — 확률효과 없는 모형은 regress 사용", s)
    fixed_main, fixed_opts = _split_opts(segs[0])
    toks = fixed_main.split()
    if not toks:
        return _ni("mixed: 종속변수 필요", s)
    if re.search(r"\s(if|in)\s", " " + fixed_main + " "):
        return _ni("mixed: if/in 조건 미지원 — keep if … 로 표본을 먼저 제한", s)
    dep = toks[0]
    covs, factors, inters = [], [], []
    for tok in toks[1:]:
        if re.search(r"[*?~\-]", tok) and "#" not in tok:
            return _ni(f"mixed: 변수 목록 약어·와일드카드 '{tok}' 미지원 — 변수 이름을 모두 나열", s)
        if "#" in tok:
            full = "##" in tok
            comps = [c for c in re.split(r"##|#", tok) if c]
            names = []
            for c in comps:
                mm = re.fullmatch(rf"(i|c)\.({_NAME})", c, re.IGNORECASE) or re.fullmatch(rf"({_NAME})", c)
                if not mm:
                    return _ni(f"mixed: 상호작용 표기 '{tok}' 미지원 (i.x#c.z, c.x##c.z 형식만)", s)
                if mm.lastindex == 2:
                    pre, nm_ = mm.group(1).lower(), mm.group(2)
                else:
                    pre, nm_ = "i", mm.group(1)      # Stata: a bare variable in # is a factor
                if pre == "i" and nm_ not in factors:
                    factors.append(nm_)
                names.append(nm_)
            inters.append(("*" if full else ":").join(names))
            continue
        mm = re.fullmatch(rf"(i|c)\.({_NAME})", tok, re.IGNORECASE)
        if mm:
            (factors if mm.group(1).lower() == "i" else covs).append(mm.group(2))
            continue
        if re.fullmatch(_NAME, tok):
            covs.append(tok)
            continue
        return _ni(f"mixed: 변수 표기 '{tok}' 미지원 (ib#.·L.·D. 등 연산자는 변수를 만든 뒤 지정)", s)
    method = "ml"
    re_stat = "var"
    extra: Dict[str, Any] = {}
    unsupported: List[str] = []
    laplace_note = family != "gaussian"

    def global_opt(name: str, val: str) -> bool:
        nonlocal method, re_stat, laplace_note
        if family == "gaussian" and name == "reml":
            method = "reml"
        elif family == "gaussian" and name in ("mle", "ml"):
            method = "ml"
        elif family != "gaussian" and _abbr(name, "intmethod", 7):
            if val.strip().lower() not in ("laplace", "lap"):
                return False
            laplace_note = False
        elif family == "binomial" and name == "or":
            extra["expB"] = True
        elif family == "poisson" and name == "irr":
            extra["expB"] = True
        elif family == "poisson" and _abbr(name, "exposure", 3) and val:
            extra["exposure"] = val.strip()
        elif family != "gaussian" and _abbr(name, "offset", 3) and val:
            extra["offset"] = val.strip()
        elif _abbr(name, "stddeviations", 6):
            re_stat = "sd"
        elif _abbr(name, "variance", 3):
            re_stat = "var"
        elif _abbr(name, "level", 3) and val:
            try:
                extra["ciWidth"] = float(val)
            except ValueError:
                unsupported.append(f"level({val})")
        elif name == "nolrtest":
            extra["lrTest"] = False
        elif _abbr(name, "nogroup", 5):
            extra["groupInfo"] = False
        elif _abbr(name, "iterate", 4) and val:
            extra["maxIter"] = int(float(val))
        elif name in ("matlog", "matsqrt"):
            pass                                # optimisation metric only; estimates are unchanged
        else:
            return False
        return True

    for name, val in fixed_opts:
        if name == "__bad__" or not global_opt(name, val):
            unsupported.append(name if name != "__bad__" else val)
    levels: List[str] = []
    slopes: Dict[str, List[str]] = {}
    cov: Dict[str, str] = {}
    nocons: List[str] = []
    for seg in segs[1:]:
        main, opts = _split_opts(seg)
        mm = re.fullmatch(rf"({_NAME})\s*:\s*(.*)", main, re.DOTALL)
        if not mm:
            return _ni(f"mixed: 확률효과 식 '{seg}' 형식 오류 (집단변수: [기울기 변수] [, 옵션])", s)
        g, vl = mm.group(1), mm.group(2).split()
        if g.lower() == "_all":
            return _ni("mixed: 교차분류 확률효과(_all: R.x) 미지원 — 위계(내포) 수준만 지원", s)
        if g in levels:
            return _ni(f"mixed: 집단 변수 '{g}' 중복", s)
        levels.append(g)
        sl = []
        for tok in vl:
            m2 = re.fullmatch(rf"(?:c\.)?({_NAME})", tok, re.IGNORECASE)
            if not m2 or tok.lower().startswith("r."):
                return _ni(f"mixed: 확률효과 변수 표기 '{tok}' 미지원 (숫자형 변수만, R.·i. 표기 제외)", s)
            sl.append(m2.group(1))
        if sl:
            slopes[g] = sl
        for name, val in opts:
            if name == "__bad__":
                unsupported.append(val)
            elif _abbr(name, "covariance", 3):
                try:
                    cov[g] = resolve_choice("covariance", val, COV_STRUCTS, _COV_ALIASES)
                except ValueError:
                    unsupported.append(f"covariance({val})")
            elif _abbr(name, "noconstant", 5):
                nocons.append(g)
            elif not global_opt(name, val):
                unsupported.append(name + (f"({val})" if val else ""))
    if len(levels) > 3:
        return _ni("mixed: 집단 수준은 최대 3개(4수준 모형)까지 지원", s)
    if unsupported:
        return _ni(f"{cmd} 옵션 미지원: " + ", ".join(unsupported), s)
    args = [f"dep={dep!r}", f"levels={levels!r}"]
    if covs:
        args.append(f"covs={covs!r}")
    if factors:
        args.append(f"factors={factors!r}")
    if inters:
        args.append(f"interactions={inters!r}")
    if slopes:
        args.append(f"randomSlopes={slopes!r}")
    # a level with a single random effect has no covariance structure to choose
    cov = {g: v for g, v in cov.items()
           if v != "independent" and len(slopes.get(g, [])) + (0 if g in nocons else 1) > 1}
    if cov:
        args.append(f"covariance={cov!r}")
    if nocons:
        args.append(f"noconstant={nocons!r}")
    if family == "gaussian":
        args.append(f"method={method!r}")
    else:
        args.append(f"family={family!r}")
    if re_stat != "var":
        args.append(f"reStat={re_stat!r}")
    for k, v in extra.items():
        args.append(f"{k}={v!r}")
    out = f"res = sess.run('mixedModel', {', '.join(args)})"
    if laplace_note:
        out += "  # 라플라스 근사(intmethod(laplace))로 추정 — Stata 기본 mvaghermite 7점과 소수 셋째 자리 안팎 차이 가능"
    return out


# ---------------------------------------------------------------------------------------------------------------
# Extension registration (analyses/extensions.py)
# ---------------------------------------------------------------------------------------------------------------
def dispatch(t: str, df: pd.DataFrame, params: Dict[str, Any]):
    if t != "mixedModel":
        return None
    return _run_from_params(df, params or {})


EXTENSION = {
    "catalog": [{"id": "mixedModel", "group": "Linear Models",
                 "title": "Multilevel Linear Mixed Model (up to 4 levels, Stata mixed)", "module": "openstat",
                 "icon": "analysis-linearmodels.svg"}],
    "aliases": {"mixed": "mixedModel", "xtmixed": "mixedModel", "multilevel": "mixedModel", "hlm": "mixedModel",
                "lmer": "mixedModel", "mixed_model": "mixedModel", "multilevel_model": "mixedModel"},
    "dispatch": dispatch,
    "stata": stata_code,
    "r": r_code,
    "transpile": transpile,
}
