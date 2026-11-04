"""
Finite mixture engines: latent class / latent profile analysis (LCA, LPA), latent transition analysis (LTA)
and latent class growth analysis (LCGA), estimated by EM with Mplus TYPE = MIXTURE defaults.

Conventions
- Random starts (Asparouhov & Muthén, 2019, "Random starting values and multistage optimization"):
  STARTS = 20 4, STITERATIONS = 10, STSEED = 0. A global seed draws one integer seed per start; a start
  perturbs the unperturbed values as v = w + s·r·b with s = 5, r ~ U(-.5, .5), b = 2 for thresholds and
  class logits, b = 2·max(1, sd) for means, b = .4 for latent class regressions (LTA transitions) and no
  perturbation of variances. Even seeds draw thresholds and class logits from ordered uniforms instead.
- ESTIMATOR = MLR: observed-information Hessian with the sandwich (score outer-product) covariance;
  ESTIMATOR = ML: inverse observed information.
- Missing indicators: full-information ML under MAR (each case contributes the indicators it has).
- Categorical indicators: class-specific category probabilities. Mplus thresholds are logit cumulative
  probabilities, P(U <= c | class) = 1 / (1 + exp(-tau_c)). Logits that reach +-15 are held fixed when the
  information matrix is formed, as in Mplus.
- LPA: class-specific means, variances equal across classes, covariances zero (local independence).
- LTA: first-order Markov chain with time-specific transition probabilities (Mplus c2 ON c1).
- LCGA: class-specific growth factor means, growth factor variances fixed at zero, residual variances
  time-specific and equal across classes.
- Likelihood-ratio tests for k-1 against k classes: VLMR / adjusted LMR (TECH11; Lo, Mendell & Rubin, 2001;
  Vuong, 1989; Mplus replaces minus the inverse Hessian by the sandwich covariance, Vermunt, 2024) and
  the parametric bootstrap LRT (TECH14; LRTSTARTS = 0 0 40 8, Asparouhov & Muthén Web Note 14).
- Three-step auxiliary models (Vermunt, 2010; Bakk, Tekle & Vermunt, 2013): R3STEP (ML-corrected
  multinomial logit), BCH (distal means with BCH weights), ML three-step for categorical distal outcomes.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import warnings

from scipy import integrate, optimize
from scipy.special import expit, logit, logsumexp

LOGIT_LIMIT = 15.0
LOG2PI = math.log(2.0 * math.pi)
TINY = 1e-12
MPLUS_S = 5.0  # STSCALE


# ============================================================ numerical helpers
def numeric_jacobian(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, rel: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(f(x), dtype=float).ravel()
    J = np.empty((f0.size, x.size))
    for j in range(x.size):
        h = rel * max(1.0, abs(x[j]))
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        J[:, j] = (np.asarray(f(xp), dtype=float).ravel() - np.asarray(f(xm), dtype=float).ravel()) / (2 * h)
    return J


def numeric_case_scores(case_ll: Callable[[np.ndarray], np.ndarray], theta: np.ndarray, rel: float = 1e-5) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    n = case_ll(theta).size
    S = np.empty((n, theta.size))
    for j in range(theta.size):
        h = rel * max(1.0, abs(theta[j]))
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        S[:, j] = (case_ll(tp) - case_ll(tm)) / (2 * h)
    return S


def _mn_pack(P: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Baseline logits of each row of P (rows sum to 1) against the reference category of that row."""
    out = []
    lp = np.log(np.clip(P, 1e-300, None))
    for r in range(P.shape[0]):
        keep = [c for c in range(P.shape[1]) if c != ref[r]]
        out.append(lp[r, keep] - lp[r, ref[r]])
    return np.concatenate(out) if out else np.zeros(0)


def _mn_unpack(eta: np.ndarray, ref: np.ndarray, C: int) -> np.ndarray:
    R = len(ref)
    P = np.empty((R, C))
    pos = 0
    for r in range(R):
        z = np.zeros(C)
        keep = [c for c in range(C) if c != ref[r]]
        z[keep] = eta[pos:pos + C - 1]
        pos += C - 1
        z -= z.max()
        e = np.exp(z)
        P[r] = e / e.sum()
    return P


def _mn_fixed(P: np.ndarray, ref: np.ndarray) -> np.ndarray:
    eta = _mn_pack(P, ref)
    return eta < -LOGIT_LIMIT


def imhof_upper(x: float, lam: np.ndarray) -> float:
    """P(sum_j lam_j * chi2_1 > x) by Imhof (1961); weights may be negative."""
    lam = np.asarray(lam, dtype=float)
    if lam.size == 0:
        return float(x <= 0)
    scale = np.max(np.abs(lam))
    if scale <= 0:
        return float(x < 0)
    lam = lam[np.abs(lam) > 1e-9 * scale]

    def integrand(u):
        if u == 0.0:
            return 0.5 * (lam.sum() - x)
        th = 0.5 * np.sum(np.arctan(lam * u)) - 0.5 * x * u
        rho = np.prod((1.0 + (lam * u) ** 2) ** 0.25)
        return math.sin(th) / (u * rho)

    total, upper = 0.0, 0.0
    step = 10.0 / scale
    for _ in range(400):  # integrate in blocks until the tail is negligible
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val, _err = integrate.quad(integrand, upper, upper + step, limit=200)
        total += val
        upper += step
        rho_u = np.prod((1.0 + (lam * upper) ** 2) ** 0.25)
        if 1.0 / (upper * rho_u) < 1e-10:
            break
        step *= 1.6
    return float(np.clip(0.5 + total / math.pi, 0.0, 1.0))


# ============================================================ indicator block (one latent class variable)
class Measurement:
    """Categorical (multinomial) and continuous (normal) indicators measured on the same rows."""

    def __init__(self, U: Optional[np.ndarray], ncat: Sequence[int], Y: Optional[np.ndarray],
                 variances: str = "equal", covariances: str = "zero"):
        n = U.shape[0] if U is not None and U.size else (Y.shape[0] if Y is not None else 0)
        self.n = int(n)
        self.U = np.zeros((self.n, 0), dtype=int) if U is None else np.asarray(U, dtype=int).reshape(self.n, -1)
        self.ncat = [int(c) for c in ncat]
        self.Y = np.zeros((self.n, 0)) if Y is None else np.asarray(Y, dtype=float).reshape(self.n, -1)
        self.Jc, self.Jy = self.U.shape[1], self.Y.shape[1]
        self.obsU = self.U >= 0
        self.Uc = np.where(self.obsU, self.U, 0)
        self.onehot = [np.eye(c)[self.Uc[:, j]] * self.obsU[:, j, None] for j, c in enumerate(self.ncat)]
        self.obsY = np.isfinite(self.Y)
        self.Y0 = np.where(self.obsY, self.Y, 0.0)
        if variances not in ("equal", "free") or covariances not in ("zero", "equal", "free"):
            raise ValueError("variances: equal|free, covariances: zero|equal|free")
        if covariances == "equal" and variances != "equal":
            raise ValueError("계층 간 동일 공분산은 계층 간 동일 분산과 함께 지정")
        if covariances == "free" and variances != "free":
            raise ValueError("계층별 공분산은 계층별 분산과 함께 지정")
        self.variances, self.covariances = variances, covariances
        self.full = covariances != "zero" and self.Jy > 1
        self.tied = variances == "equal"
        if self.Jy:
            cnt = self.obsY.sum(0)
            self.ymean = np.where(cnt > 0, self.Y0.sum(0) / np.maximum(cnt, 1), 0.0)
            dev = np.where(self.obsY, self.Y - self.ymean, 0.0)
            self.yvar = np.where(cnt > 1, (dev ** 2).sum(0) / np.maximum(cnt, 1), 1.0)
            self.yvar = np.where(self.yvar > 0, self.yvar, 1.0)
        else:
            self.ymean = self.yvar = np.zeros(0)
        self.floor = 1e-6 * self.yvar
        if self.full:
            keys = {}
            for i, row in enumerate(self.obsY):
                keys.setdefault(tuple(row.tolist()), []).append(i)
            self.patterns = [(np.array(k, dtype=bool), np.array(v)) for k, v in keys.items()]
        self.ref: Optional[List[np.ndarray]] = None

    # ---------------------------------------------------------- bookkeeping
    def n_free(self, K: int) -> int:
        p = sum(K * (c - 1) for c in self.ncat) + K * self.Jy
        if not self.Jy:
            return p
        if self.full:
            m = self.Jy * (self.Jy + 1) // 2
            return p + (m if self.tied else K * m)
        return p + (self.Jy if self.tied else K * self.Jy)

    def subset(self, rows: np.ndarray) -> "Measurement":
        return Measurement(self.U[rows], self.ncat, self.Y[rows], self.variances, self.covariances)

    def set_reference(self, par: Dict[str, Any]) -> None:
        self.ref = [np.argmax(P, axis=1) for P in par["P"]]

    # ---------------------------------------------------------- likelihood
    def log_dens(self, par: Dict[str, Any]) -> np.ndarray:
        K = par["mu"].shape[0] if self.Jy else par["P"][0].shape[0]
        L = np.zeros((self.n, K))
        for j in range(self.Jc):
            lp = np.log(np.clip(par["P"][j], 1e-300, None))
            L += lp[:, self.Uc[:, j]].T * self.obsU[:, j, None]
        if self.Jy:
            if not self.full:
                var = par["var"]
                d = self.Y0[:, None, :] - par["mu"][None]
                q = -0.5 * (LOG2PI + np.log(var)[None] + d * d / var[None]) * self.obsY[:, None, :]
                L += q.sum(2)
            else:
                for mask, rows in self.patterns:
                    o = np.where(mask)[0]
                    if o.size == 0:
                        continue
                    yo = self.Y[np.ix_(rows, o)]
                    for k in range(K):
                        S = par["S"][k][np.ix_(o, o)]
                        c = np.linalg.cholesky(S)
                        z = np.linalg.solve(c, (yo - par["mu"][k, o]).T)
                        L[rows, k] += -0.5 * (o.size * LOG2PI + 2 * np.log(np.diag(c)).sum() + (z * z).sum(0))
        return L

    # ---------------------------------------------------------- EM M-step
    def m_step(self, W: np.ndarray, old: Dict[str, Any]) -> Dict[str, Any]:
        K = W.shape[1]
        new: Dict[str, Any] = {"P": []}
        for j, C in enumerate(self.ncat):
            counts = W.T @ self.onehot[j]
            den = counts.sum(1, keepdims=True)
            Pj = np.where(den > 0, (counts + TINY) / (den + C * TINY), 1.0 / C)
            new["P"].append(Pj)
        if not self.Jy:
            new["mu"] = np.zeros((K, 0))
            return new
        if not self.full:
            Wo = W.T @ self.obsY.astype(float)
            Sw = W.T @ self.Y0
            mu = np.where(Wo > TINY, Sw / np.maximum(Wo, TINY), old["mu"])
            R = W.T @ (self.Y0 ** 2) - 2 * mu * Sw + mu * mu * Wo
            if self.tied:
                v = R.sum(0) / np.maximum(Wo.sum(0), TINY)
                var = np.tile(np.maximum(v, self.floor), (K, 1))
            else:
                var = np.where(Wo > TINY, R / np.maximum(Wo, TINY), old["var"])
                var = np.maximum(var, self.floor[None])
            new["mu"], new["var"] = mu, var
            return new
        mu0, S0 = old["mu"], old["S"]
        J = self.Jy
        T1 = np.zeros((K, J))
        T2 = np.zeros((K, J, J))
        Wk = W.sum(0)
        for mask, rows in self.patterns:
            o, m = np.where(mask)[0], np.where(~mask)[0]
            Wr = W[rows]
            for k in range(K):
                Yh = np.empty((rows.size, J))
                Cf = np.zeros((J, J))
                if m.size == 0:
                    Yh[:] = self.Y[rows]
                elif o.size == 0:
                    Yh[:] = mu0[k]
                    Cf = S0[k].copy()
                else:
                    Soo = S0[k][np.ix_(o, o)]
                    B = np.linalg.solve(Soo, S0[k][np.ix_(o, m)]).T
                    Yh[:, o] = self.Y[np.ix_(rows, o)]
                    Yh[:, m] = mu0[k, m] + (self.Y[np.ix_(rows, o)] - mu0[k, o]) @ B.T
                    Cf[np.ix_(m, m)] = S0[k][np.ix_(m, m)] - B @ S0[k][np.ix_(o, m)]
                w = Wr[:, k]
                T1[k] += w @ Yh
                T2[k] += (Yh * w[:, None]).T @ Yh + w.sum() * Cf
        mu = np.where(Wk[:, None] > TINY, T1 / np.maximum(Wk[:, None], TINY), mu0)
        Sk = np.array([T2[k] / max(Wk[k], TINY) - np.outer(mu[k], mu[k]) for k in range(K)])
        if self.tied:
            pooled = sum(T2[k] - Wk[k] * np.outer(mu[k], mu[k]) for k in range(K)) / max(Wk.sum(), TINY)
            Sk = np.array([pooled] * K)
        Sk = np.array([self._fix_cov(S) for S in Sk])
        new["mu"], new["S"] = mu, Sk
        new["var"] = np.array([np.diag(S) for S in Sk])
        return new

    def _fix_cov(self, S: np.ndarray) -> np.ndarray:
        S = (S + S.T) / 2
        w, V = np.linalg.eigh(S)
        w = np.maximum(w, self.floor.min() if self.floor.size else 1e-8)
        return (V * w) @ V.T

    # ---------------------------------------------------------- starting values
    def default_start(self, K: int) -> Dict[str, Any]:
        par: Dict[str, Any] = {"P": []}
        for j, C in enumerate(self.ncat):
            cnt = self.onehot[j].sum(0) + 0.5
            par["P"].append(np.tile(cnt / cnt.sum(), (K, 1)))
        par["mu"] = np.tile(self.ymean, (K, 1)) if self.Jy else np.zeros((K, 0))
        if self.Jy:
            par["var"] = np.tile(self.yvar, (K, 1))
            if self.full:
                par["S"] = np.array([np.diag(self.yvar)] * K)
        return par

    def random_start(self, rng: np.random.Generator, K: int, alt: bool) -> Dict[str, Any]:
        par = self.default_start(K)
        for j, C in enumerate(self.ncat):
            cum = np.cumsum(par["P"][j][0])[:-1]
            tau0 = logit(np.clip(cum, 1e-4, 1 - 1e-4))
            Pj = np.empty((K, C))
            for k in range(K):
                if alt:
                    tau = logit(np.clip(np.sort(rng.uniform(size=C - 1)), 1e-4, 1 - 1e-4))
                else:
                    tau = np.sort(tau0 + MPLUS_S * rng.uniform(-0.5, 0.5, size=C - 1) * 2.0)
                cp = np.concatenate([[0.0], expit(tau), [1.0]])
                pk = np.clip(np.diff(cp), 1e-6, None)
                Pj[k] = pk / pk.sum()
            par["P"][j] = Pj
        if self.Jy:
            b = 2.0 * np.maximum(1.0, np.sqrt(self.yvar))
            par["mu"] = self.ymean[None] + MPLUS_S * rng.uniform(-0.5, 0.5, size=(K, self.Jy)) * b[None]
        return par

    # ---------------------------------------------------------- unconstrained parameter vector
    def pack(self, par: Dict[str, Any]) -> np.ndarray:
        if self.ref is None:
            self.set_reference(par)
        parts = [_mn_pack(P, self.ref[j]) for j, P in enumerate(par["P"])]
        if self.Jy:
            parts.append(par["mu"].ravel())
            if self.full:
                Ss = par["S"][:1] if self.tied else par["S"]
                for S in Ss:
                    L = np.linalg.cholesky(S)
                    parts.append(np.log(np.diag(L)))
                    parts.append(L[np.tril_indices(self.Jy, -1)])
            else:
                parts.append(np.log(par["var"][0] if self.tied else par["var"].ravel()))
        return np.concatenate(parts) if parts else np.zeros(0)

    def unpack(self, theta: np.ndarray, K: int) -> Dict[str, Any]:
        par: Dict[str, Any] = {"P": []}
        pos = 0
        for j, C in enumerate(self.ncat):
            m = K * (C - 1)
            par["P"].append(_mn_unpack(theta[pos:pos + m], self.ref[j], C))
            pos += m
        if not self.Jy:
            par["mu"] = np.zeros((K, 0))
            return par
        J = self.Jy
        par["mu"] = theta[pos:pos + K * J].reshape(K, J)
        pos += K * J
        if self.full:
            nS = 1 if self.tied else K
            Ss = []
            for _ in range(nS):
                L = np.zeros((J, J))
                L[np.diag_indices(J)] = np.exp(theta[pos:pos + J])
                pos += J
                m = J * (J - 1) // 2
                L[np.tril_indices(J, -1)] = theta[pos:pos + m]
                pos += m
                Ss.append(L @ L.T)
            par["S"] = np.array(Ss * K if self.tied else Ss)
            par["var"] = np.array([np.diag(S) for S in par["S"]])
        else:
            if self.tied:
                par["var"] = np.tile(np.exp(theta[pos:pos + J]), (K, 1))
                pos += J
            else:
                par["var"] = np.exp(theta[pos:pos + K * J]).reshape(K, J)
                pos += K * J
        return par

    def fixed_mask(self, par: Dict[str, Any]) -> np.ndarray:
        if self.ref is None:
            self.set_reference(par)
        parts = [_mn_fixed(P, self.ref[j]) for j, P in enumerate(par["P"])]
        if self.Jy:
            K = par["mu"].shape[0]
            parts.append(np.zeros(K * self.Jy, dtype=bool))
            if self.full:
                nS = 1 if self.tied else K
                parts.append(np.zeros(nS * self.Jy * (self.Jy + 1) // 2, dtype=bool))
            else:
                v = par["var"][0] if self.tied else par["var"].ravel()
                fl = self.floor if self.tied else np.tile(self.floor, K)
                parts.append(v <= fl * 1.0001)
        return np.concatenate(parts) if parts else np.zeros(0, dtype=bool)

    def case_grad(self, par: Dict[str, Any], W: np.ndarray) -> Optional[np.ndarray]:
        """Per-row gradient of sum_k W_ik log f_k(y_i) in the packed parameterization (None: use numeric)."""
        if self.full:
            return None
        K = W.shape[1]
        cols = []
        for j, P in enumerate(par["P"]):
            oh = self.onehot[j]
            obs = self.obsU[:, j]
            for k in range(K):
                for c in range(P.shape[1]):
                    if c == self.ref[j][k]:
                        continue
                    cols.append(W[:, k] * (oh[:, c] - P[k, c] * obs))
        G = [np.column_stack(cols)] if cols else []
        if self.Jy:
            var = par["var"]
            d = (self.Y0[:, None, :] - par["mu"][None]) * self.obsY[:, None, :]
            G.append((W[:, :, None] * d / var[None]).reshape(self.n, -1))
            u = W[:, :, None] * 0.5 * (d * d / var[None] - self.obsY[:, None, :])
            G.append(u.sum(1) if self.tied else u.reshape(self.n, -1))
        return np.column_stack(G) if G else np.zeros((self.n, 0))

    def permute(self, par: Dict[str, Any], order: np.ndarray) -> Dict[str, Any]:
        out = {"P": [P[order] for P in par["P"]], "mu": par["mu"][order]}
        for key in ("var", "S"):
            if key in par:
                out[key] = par[key][order]
        return out

    def simulate(self, par: Dict[str, Any], z: np.ndarray, rng: np.random.Generator) -> "Measurement":
        n = z.size
        U = np.full((n, self.Jc), -1, dtype=int)
        for j, P in enumerate(par["P"]):
            cum = np.cumsum(P[z], axis=1)
            U[:, j] = (rng.uniform(size=(n, 1)) > cum[:, :-1]).sum(1)
        Y = np.full((n, self.Jy), np.nan)
        if self.Jy:
            if self.full:
                for i in range(n):
                    Y[i] = rng.multivariate_normal(par["mu"][z[i]], par["S"][z[i]])
            else:
                Y = par["mu"][z] + np.sqrt(par["var"][z]) * rng.standard_normal((n, self.Jy))
        U[~self.obsU[:n]] = -1
        Y[~self.obsY[:n]] = np.nan
        return Measurement(U, self.ncat, Y, self.variances, self.covariances)


# ============================================================ growth measurement (LCGA)
GROWTH_ORDERS = {"linear": 2, "quadratic": 3, "cubic": 4}
GROWTH_NAMES = ["I", "S", "Q", "C"]


class GrowthMeasurement:
    """Repeated continuous outcome with class-specific polynomial growth means and zero growth variances."""

    def __init__(self, Y: np.ndarray, times: Sequence[float], q: int, resvar: str = "time_class_invariant"):
        self.Y = np.asarray(Y, dtype=float)
        self.n, self.T = self.Y.shape
        self.times = np.asarray(times, dtype=float)
        self.q = int(q)
        self.X = self.times[:, None] ** np.arange(self.q)[None]
        if resvar not in ("time_class_invariant", "class_specific", "equal"):
            raise ValueError("residualVariances")
        self.resvar = resvar
        self.obs = np.isfinite(self.Y)
        self.Y0 = np.where(self.obs, self.Y, 0.0)
        cnt = self.obs.sum(0)
        m = self.Y0.sum(0) / np.maximum(cnt, 1)
        v = (np.where(self.obs, self.Y - m, 0.0) ** 2).sum(0) / np.maximum(cnt, 1)
        self.yvar_t = np.where(v > 0, v, 1.0)
        self.floor = 1e-6 * self.yvar_t
        self.ref = None

    def n_free(self, K: int) -> int:
        nv = {"time_class_invariant": self.T, "class_specific": K * self.T, "equal": 1}[self.resvar]
        return K * self.q + nv

    def subset(self, rows):
        return GrowthMeasurement(self.Y[rows], self.times, self.q, self.resvar)

    def set_reference(self, par):
        return None

    def means(self, par) -> np.ndarray:
        return par["beta"] @ self.X.T

    def log_dens(self, par) -> np.ndarray:
        m = self.means(par)
        var = par["var"]
        d = self.Y0[:, None, :] - m[None]
        q = -0.5 * (LOG2PI + np.log(var)[None] + d * d / var[None]) * self.obs[:, None, :]
        return q.sum(2)

    def m_step(self, W: np.ndarray, old) -> Dict[str, Any]:
        K = W.shape[1]
        var0 = old["var"]
        Wo = W.T @ self.obs.astype(float)
        Sw = W.T @ self.Y0
        beta = np.empty((K, self.q))
        for k in range(K):
            a = Wo[k] / var0[k]
            b = Sw[k] / var0[k]
            A = self.X.T @ (a[:, None] * self.X)
            if np.linalg.matrix_rank(A) < self.q:
                beta[k] = old["beta"][k]
            else:
                beta[k] = np.linalg.solve(A, self.X.T @ b)
        m = beta @ self.X.T
        R = W.T @ (self.Y0 ** 2) - 2 * m * Sw + m * m * Wo
        if self.resvar == "time_class_invariant":
            var = np.tile(np.maximum(R.sum(0) / np.maximum(Wo.sum(0), TINY), self.floor), (K, 1))
        elif self.resvar == "class_specific":
            var = np.maximum(np.where(Wo > TINY, R / np.maximum(Wo, TINY), var0), self.floor[None])
        else:
            var = np.full((K, self.T), max(R.sum() / max(Wo.sum(), TINY), self.floor.min()))
        return {"beta": beta, "var": var}

    def _pooled_ols(self):
        t = np.repeat(self.times[None], self.n, 0)[self.obs]
        y = self.Y[self.obs]
        X = t[:, None] ** np.arange(self.q)[None]
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = np.where(self.obs, self.Y - (b @ self.X.T)[None], 0.0)
        cnt = self.obs.sum(0)
        rv = (resid ** 2).sum(0) / np.maximum(cnt, 1)
        return b, np.where(rv > 0, rv, self.yvar_t)

    def default_start(self, K: int) -> Dict[str, Any]:
        b, rv = self._pooled_ols()
        var = np.tile(rv, (K, 1))
        if self.resvar == "equal":
            var = np.full((K, self.T), rv.mean())
        return {"beta": np.tile(b, (K, 1)), "var": var}

    def random_start(self, rng, K: int, alt: bool) -> Dict[str, Any]:
        par = self.default_start(K)
        sd = float(np.sqrt(np.mean(self.yvar_t)))
        span = max(float(self.times.max() - self.times.min()), 1.0)
        bscale = 2.0 * max(1.0, sd) / span ** np.arange(self.q)
        par["beta"] = par["beta"] + MPLUS_S * rng.uniform(-0.5, 0.5, size=(K, self.q)) * bscale[None]
        return par

    def pack(self, par) -> np.ndarray:
        v = par["var"]
        lv = {"time_class_invariant": v[0], "class_specific": v.ravel(), "equal": v[:1, :1].ravel()}[self.resvar]
        return np.concatenate([par["beta"].ravel(), np.log(lv)])

    def unpack(self, theta, K: int) -> Dict[str, Any]:
        beta = theta[:K * self.q].reshape(K, self.q)
        lv = np.exp(theta[K * self.q:])
        if self.resvar == "time_class_invariant":
            var = np.tile(lv, (K, 1))
        elif self.resvar == "class_specific":
            var = lv.reshape(K, self.T)
        else:
            var = np.full((K, self.T), lv[0])
        return {"beta": beta, "var": var}

    def fixed_mask(self, par) -> np.ndarray:
        K = par["beta"].shape[0]
        v = par["var"]
        if self.resvar == "time_class_invariant":
            fv = v[0] <= self.floor * 1.0001
        elif self.resvar == "class_specific":
            fv = (v <= self.floor[None] * 1.0001).ravel()
        else:
            fv = np.array([v[0, 0] <= self.floor.min() * 1.0001])
        return np.concatenate([np.zeros(K * self.q, dtype=bool), fv])

    def case_grad(self, par, W) -> np.ndarray:
        K = W.shape[1]
        m = self.means(par)
        var = par["var"]
        d = (self.Y0[:, None, :] - m[None]) * self.obs[:, None, :]
        z = W[:, :, None] * d / var[None]
        gb = np.einsum("nkt,tq->nkq", z, self.X).reshape(self.n, K * self.q)
        u = W[:, :, None] * 0.5 * (d * d / var[None] - self.obs[:, None, :])
        if self.resvar == "time_class_invariant":
            gv = u.sum(1)
        elif self.resvar == "class_specific":
            gv = u.reshape(self.n, -1)
        else:
            gv = u.sum((1, 2))[:, None]
        return np.column_stack([gb, gv])

    def permute(self, par, order):
        return {"beta": par["beta"][order], "var": par["var"][order]}

    def simulate(self, par, z, rng) -> "GrowthMeasurement":
        m = self.means(par)
        Y = m[z] + np.sqrt(par["var"][z]) * rng.standard_normal((z.size, self.T))
        Y[~self.obs[:z.size]] = np.nan
        return GrowthMeasurement(Y, self.times, self.q, self.resvar)


# ============================================================ single latent class variable (LCA, LPA, LCGA)
class LatentClassModel:
    """K-class mixture over independent units; the measurement object supplies class-conditional densities."""

    def __init__(self, meas, K: int):
        self.meas = copy.copy(meas)  # own packing references (they depend on K); data arrays are shared
        self.meas.ref = None
        self.K = int(K)
        self.n = meas.n
        self.cref: Optional[int] = None

    @property
    def n_free(self) -> int:
        return (self.K - 1) + self.meas.n_free(self.K)

    def log_joint(self, par) -> np.ndarray:
        return np.log(np.clip(par["p"], 1e-300, None))[None] + self.meas.log_dens(par["meas"])

    def e_step(self, par):
        lj = self.log_joint(par)
        ll_i = logsumexp(lj, axis=1)
        return np.exp(lj - ll_i[:, None]), ll_i

    def m_step(self, post, par):
        p = post.mean(0)
        p = np.clip(p, 1e-300, None)
        return {"p": p / p.sum(), "meas": self.meas.m_step(post, par["meas"])}

    def default_start(self):
        return {"p": np.full(self.K, 1.0 / self.K), "meas": self.meas.default_start(self.K)}

    def random_start(self, rng, alt: bool):
        K = self.K
        if alt:
            r = np.sort(rng.uniform(size=K - 1))
            p = np.diff(np.concatenate([[0.0], r, [1.0]]))
        else:
            a = np.concatenate([MPLUS_S * rng.uniform(-0.5, 0.5, size=K - 1) * 2.0, [0.0]])
            p = np.exp(a - a.max())
        p = np.clip(p, 1e-3, None)
        return {"p": p / p.sum(), "meas": self.meas.random_start(rng, K, alt)}

    def set_reference(self, par):
        self.cref = int(np.argmax(par["p"]))
        self.meas.set_reference(par["meas"])

    def pack(self, par) -> np.ndarray:
        if self.cref is None:
            self.set_reference(par)
        return np.concatenate([_mn_pack(par["p"][None], np.array([self.cref])), self.meas.pack(par["meas"])])

    def unpack(self, theta):
        k1 = self.K - 1
        p = _mn_unpack(theta[:k1], np.array([self.cref]), self.K)[0]
        return {"p": p, "meas": self.meas.unpack(theta[k1:], self.K)}

    def fixed_mask(self, par) -> np.ndarray:
        return np.concatenate([_mn_fixed(par["p"][None], np.array([self.cref])), self.meas.fixed_mask(par["meas"])])

    def case_ll(self, theta) -> np.ndarray:
        return self.e_step(self.unpack(theta))[1]

    def case_scores(self, theta) -> np.ndarray:
        par = self.unpack(theta)
        post, _ = self.e_step(par)
        g = self.meas.case_grad(par["meas"], post)
        if g is None:
            return numeric_case_scores(self.case_ll, theta)
        keep = [k for k in range(self.K) if k != self.cref]
        return np.column_stack([post[:, keep] - par["p"][keep][None], g])

    def permute(self, par, order):
        return {"p": par["p"][order], "meas": self.meas.permute(par["meas"], order)}

    def simulate_model(self, par, rng, K_new: Optional[int] = None) -> "LatentClassModel":
        z = rng.choice(self.K, size=self.n, p=par["p"])
        return LatentClassModel(self.meas.simulate(par["meas"], z, rng), K_new or self.K)

    def sort_order(self, par) -> np.ndarray:
        return np.argsort(-par["p"], kind="stable")


# ============================================================ latent transition model
class LatentTransitionModel:
    """K classes at each of T occasions, first-order Markov transitions; invariant or time-specific measurement."""

    def __init__(self, meas, K: int, n: int, T: int, invariant: bool):
        # Measurement over stacked rows (t-major) if invariant, else list of T Measurements; own packing references
        self.meas = copy.copy(meas) if invariant else [copy.copy(m) for m in meas]
        for m in ([self.meas] if invariant else self.meas):
            m.ref = None
        self.K, self.n, self.T = int(K), int(n), int(T)
        self.invariant = bool(invariant)
        self.cref = None
        self.tref = None

    @property
    def n_free(self) -> int:
        m = self.meas.n_free(self.K) if self.invariant else sum(mt.n_free(self.K) for mt in self.meas)
        return (self.K - 1) + (self.T - 1) * self.K * (self.K - 1) + m

    def log_emis(self, par) -> np.ndarray:
        if self.invariant:
            L = self.meas.log_dens(par["meas"])
            return L.reshape(self.T, self.n, self.K).transpose(1, 0, 2)
        return np.stack([self.meas[t].log_dens(par["meas"][t]) for t in range(self.T)], axis=1)

    def forward_backward(self, par):
        E = self.log_emis(par)
        n, T, K = E.shape
        lt = np.log(np.clip(par["tau"], 1e-300, None))  # (T-1, K, K)
        la = np.empty((n, T, K))
        la[:, 0] = np.log(np.clip(par["pi"], 1e-300, None))[None] + E[:, 0]
        for t in range(1, T):
            la[:, t] = logsumexp(la[:, t - 1, :, None] + lt[t - 1][None], axis=1) + E[:, t]
        ll_i = logsumexp(la[:, T - 1], axis=1)
        lb = np.zeros((n, T, K))
        for t in range(T - 2, -1, -1):
            lb[:, t] = logsumexp(lt[t][None] + (E[:, t + 1] + lb[:, t + 1])[:, None, :], axis=2)
        gamma = np.exp(la + lb - ll_i[:, None, None])
        xi = np.empty((n, max(T - 1, 0), K, K))
        for t in range(T - 1):
            xi[:, t] = np.exp(la[:, t, :, None] + lt[t][None] + (E[:, t + 1] + lb[:, t + 1])[:, None, :] - ll_i[:, None, None])
        return ll_i, gamma, xi

    def e_step(self, par):
        ll_i, gamma, xi = self.forward_backward(par)
        return (gamma, xi), ll_i

    def m_step(self, stats, par):
        gamma, xi = stats
        pi = np.clip(gamma[:, 0].mean(0), 1e-300, None)
        num = xi.sum(0)
        den = gamma[:, :-1].sum(0)[..., None]
        tau = np.where(den > TINY, num / np.maximum(den, TINY), par["tau"])
        tau = np.clip(tau, 1e-300, None)
        tau /= tau.sum(2, keepdims=True)
        if self.invariant:
            Wst = gamma.transpose(1, 0, 2).reshape(self.T * self.n, self.K)
            meas = self.meas.m_step(Wst, par["meas"])
        else:
            meas = [self.meas[t].m_step(gamma[:, t], par["meas"][t]) for t in range(self.T)]
        return {"pi": pi / pi.sum(), "tau": tau, "meas": meas}

    def default_start(self):
        K = self.K
        meas = self.meas.default_start(K) if self.invariant else [m.default_start(K) for m in self.meas]
        return {"pi": np.full(K, 1.0 / K), "tau": np.full((self.T - 1, K, K), 1.0 / K), "meas": meas}

    def random_start(self, rng, alt: bool):
        K = self.K
        if alt:
            r = np.sort(rng.uniform(size=K - 1))
            pi = np.diff(np.concatenate([[0.0], r, [1.0]]))
        else:
            a = np.concatenate([MPLUS_S * rng.uniform(-0.5, 0.5, size=K - 1) * 2.0, [0.0]])
            pi = np.exp(a - a.max())
        pi = np.clip(pi, 1e-3, None)
        eta = MPLUS_S * rng.uniform(-0.5, 0.5, size=(self.T - 1, K, K)) * 0.4
        eta[..., -1] = 0.0
        tau = np.exp(eta - eta.max(2, keepdims=True))
        tau /= tau.sum(2, keepdims=True)
        if self.invariant:
            meas = self.meas.random_start(rng, K, alt)
        else:
            m0 = self.meas[0].random_start(rng, K, alt)
            meas = [m0] + [{k: (v.copy() if isinstance(v, np.ndarray) else [x.copy() for x in v]) for k, v in m0.items()}
                           for _ in range(self.T - 1)]
        return {"pi": pi / pi.sum(), "tau": tau, "meas": meas}

    def set_reference(self, par):
        self.cref = int(np.argmax(par["pi"]))
        self.tref = np.argmax(par["tau"], axis=2)  # (T-1, K)
        if self.invariant:
            self.meas.set_reference(par["meas"])
        else:
            for t in range(self.T):
                self.meas[t].set_reference(par["meas"][t])

    def pack(self, par) -> np.ndarray:
        if self.cref is None:
            self.set_reference(par)
        parts = [_mn_pack(par["pi"][None], np.array([self.cref]))]
        for t in range(self.T - 1):
            parts.append(_mn_pack(par["tau"][t], self.tref[t]))
        if self.invariant:
            parts.append(self.meas.pack(par["meas"]))
        else:
            parts.extend(self.meas[t].pack(par["meas"][t]) for t in range(self.T))
        return np.concatenate(parts)

    def unpack(self, theta):
        K = self.K
        pos = K - 1
        pi = _mn_unpack(theta[:pos], np.array([self.cref]), K)[0]
        tau = np.empty((self.T - 1, K, K))
        for t in range(self.T - 1):
            m = K * (K - 1)
            tau[t] = _mn_unpack(theta[pos:pos + m], self.tref[t], K)
            pos += m
        if self.invariant:
            meas = self.meas.unpack(theta[pos:], K)
        else:
            meas = []
            for t in range(self.T):
                m = self.meas[t].n_free(K)
                meas.append(self.meas[t].unpack(theta[pos:pos + m], K))
                pos += m
        return {"pi": pi, "tau": tau, "meas": meas}

    def fixed_mask(self, par) -> np.ndarray:
        parts = [_mn_fixed(par["pi"][None], np.array([self.cref]))]
        parts += [_mn_fixed(par["tau"][t], self.tref[t]) for t in range(self.T - 1)]
        if self.invariant:
            parts.append(self.meas.fixed_mask(par["meas"]))
        else:
            parts += [self.meas[t].fixed_mask(par["meas"][t]) for t in range(self.T)]
        return np.concatenate(parts)

    def case_ll(self, theta) -> np.ndarray:
        return self.forward_backward(self.unpack(theta))[0]

    def case_scores(self, theta) -> np.ndarray:
        par = self.unpack(theta)
        _, gamma, xi = self.forward_backward(par)
        K = self.K
        cols = [gamma[:, 0, [k for k in range(K) if k != self.cref]] - par["pi"][[k for k in range(K) if k != self.cref]][None]]
        for t in range(self.T - 1):
            for k in range(K):
                keep = [l for l in range(K) if l != self.tref[t][k]]
                cols.append(xi[:, t, k, keep] - gamma[:, t, k, None] * par["tau"][t, k, keep][None])
        if self.invariant:
            Wst = gamma.transpose(1, 0, 2).reshape(self.T * self.n, K)
            g = self.meas.case_grad(par["meas"], Wst)
            if g is None:
                return numeric_case_scores(self.case_ll, theta)
            cols.append(g.reshape(self.T, self.n, -1).sum(0))
        else:
            for t in range(self.T):
                g = self.meas[t].case_grad(par["meas"][t], gamma[:, t])
                if g is None:
                    return numeric_case_scores(self.case_ll, theta)
                cols.append(g)
        return np.column_stack(cols)

    def permute(self, par, order):
        tau = par["tau"][:, order][:, :, order]
        if self.invariant:
            meas = self.meas.permute(par["meas"], order)
        else:
            meas = [self.meas[t].permute(par["meas"][t], order) for t in range(self.T)]
        return {"pi": par["pi"][order], "tau": tau, "meas": meas}

    def marginals(self, par) -> np.ndarray:
        out = [par["pi"]]
        for t in range(self.T - 1):
            out.append(out[-1] @ par["tau"][t])
        return np.array(out)

    def sort_order(self, par) -> np.ndarray:
        prev = self.marginals(par).mean(0) if self.invariant else par["pi"]
        return np.argsort(-prev, kind="stable")


# ============================================================ estimation with Mplus random starts
def run_em(model, par, max_iter: int, tol: float = 1e-6):
    ll_old = -np.inf
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        stats, ll_i = model.e_step(par)
        ll = float(ll_i.sum())
        if not np.isfinite(ll):
            return par, -np.inf, it, False
        if it > 1 and abs(ll - ll_old) < tol:
            converged = True
            return par, ll, it, converged
        par = model.m_step(stats, par)
        ll_old = ll
    ll = float(model.e_step(par)[1].sum())
    return par, ll, it, converged


@dataclass
class StartRecord:
    ll: float
    seed: int
    start: int
    iterations: int
    converged: bool


@dataclass
class EstimationResult:
    par: Dict[str, Any]
    ll: float
    finals: List[StartRecord]
    replicated: Optional[bool]
    converged: bool


def estimate(model, starts: int = 20, final: int = 4, stiter: int = 10, seed: int = 0, max_iter: int = 5000,
             tol: float = 1e-6, init: Optional[Dict[str, Any]] = None) -> EstimationResult:
    if init is not None or model.K == 1 or starts <= 0:
        par0 = init if init is not None else model.default_start()
        par, ll, it, conv = run_em(model, par0, max_iter, tol)
        rec = StartRecord(ll, 0, 0, it, conv)
        return EstimationResult(par, ll, [rec], None, conv)
    rng = np.random.default_rng(int(seed))
    seeds = rng.integers(1, 1_000_001, size=int(starts))
    stage1 = []
    for j, cj in enumerate(seeds, 1):
        r = np.random.default_rng(int(cj))
        par0 = model.random_start(r, alt=(int(cj) % 2 == 0))
        par, ll, it, _ = run_em(model, par0, int(stiter), tol)
        if np.isfinite(ll):
            stage1.append((ll, j, int(cj), par))
    if not stage1:
        raise ValueError("모든 시작값에서 추정 실패")
    stage1.sort(key=lambda s: -s[0])
    finals = []
    for ll1, j, cj, par0 in stage1[:max(1, int(final))]:
        par, ll, it, conv = run_em(model, par0, max_iter, tol)
        finals.append((StartRecord(ll, cj, j, it, conv), par))
    finals.sort(key=lambda f: -f[0].ll)
    best_rec, best_par = finals[0]
    tol_rep = max(1e-3, 1e-7 * abs(best_rec.ll))
    rep = sum(1 for f in finals if abs(f[0].ll - best_rec.ll) <= tol_rep) >= 2 if len(finals) >= 2 else None
    return EstimationResult(best_par, best_rec.ll, [f[0] for f in finals], rep, best_rec.converged)


@dataclass
class MixtureFit:
    model: Any
    par: Dict[str, Any]
    ll: float
    est: EstimationResult
    post: np.ndarray = field(default=None)
    ll_i: np.ndarray = field(default=None)

    @property
    def npar(self) -> int:
        return self.model.n_free

    @property
    def n(self) -> int:
        return self.model.n

    @property
    def aic(self) -> float:
        return -2 * self.ll + 2 * self.npar

    @property
    def bic(self) -> float:
        return -2 * self.ll + self.npar * math.log(self.n)

    @property
    def sabic(self) -> float:
        return -2 * self.ll + self.npar * math.log((self.n + 2) / 24.0)


def fit_model(model, starts=20, final=4, stiter=10, seed=0, max_iter=5000, tol=1e-6, init=None) -> MixtureFit:
    est = estimate(model, starts, final, stiter, seed, max_iter, tol, init)
    par = est.par
    order = model.sort_order(par)
    par = model.permute(par, order)
    stats, ll_i = model.e_step(par)
    post = stats[0] if isinstance(stats, tuple) else stats
    return MixtureFit(model, par, float(ll_i.sum()), est, post, ll_i)


def relative_entropy(post: np.ndarray) -> float:
    n, K = post.shape
    if K < 2:
        return float("nan")
    p = np.clip(post, 1e-300, 1.0)
    return float(1.0 - (-(post * np.log(p)).sum()) / (n * math.log(K)))


def lta_entropy(gamma: np.ndarray, xi: np.ndarray) -> float:
    """Relative entropy of the joint class-pattern posterior, via the chain rule over occasions."""
    n, T, K = gamma.shape
    if K < 2:
        return float("nan")
    g1 = np.clip(gamma[:, 0], 1e-300, 1)
    H = -(gamma[:, 0] * np.log(g1)).sum()
    for t in range(T - 1):
        cond = xi[:, t] / np.clip(gamma[:, t, :, None], 1e-300, None)
        H += -(xi[:, t] * np.log(np.clip(cond, 1e-300, None))).sum()
    return float(1.0 - H / (n * T * math.log(K)))


# ============================================================ standard errors
@dataclass
class Inference:
    theta: np.ndarray
    V: np.ndarray
    free: np.ndarray
    scores: np.ndarray
    hessian: np.ndarray
    singular: bool
    n_fixed: int


def inference(model, par, estimator: str = "mlr") -> Inference:
    model.set_reference(par)
    theta = model.pack(par)
    fixed = model.fixed_mask(par)
    free = ~fixed
    S = model.case_scores(theta)

    def total_grad(th_free):
        th = theta.copy()
        th[free] = th_free
        return model.case_scores(th)[:, free].sum(0)

    H = numeric_jacobian(total_grad, theta[free])
    H = (H + H.T) / 2
    singular = False
    try:
        np.linalg.cholesky(-H)
        Ainv = np.linalg.inv(-H)
    except np.linalg.LinAlgError:
        singular = True
        Ainv = np.linalg.pinv(-H)
    Sf = S[:, free]
    Vf = Ainv if estimator == "ml" else Ainv @ (Sf.T @ Sf) @ Ainv
    V = np.zeros((theta.size, theta.size))
    V[np.ix_(free, free)] = (Vf + Vf.T) / 2
    return Inference(theta, V, free, S, H, singular, int(fixed.sum()))


def delta(fn: Callable[[np.ndarray], np.ndarray], inf: Inference) -> Tuple[np.ndarray, np.ndarray]:
    est = np.asarray(fn(inf.theta), dtype=float).ravel()
    J = numeric_jacobian(fn, inf.theta, rel=1e-6)
    cov = J @ inf.V @ J.T
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    se = np.where(se < 1e-10, np.nan, se)
    return est, se


def delta_cov(fn, inf: Inference) -> Tuple[np.ndarray, np.ndarray]:
    est = np.asarray(fn(inf.theta), dtype=float).ravel()
    J = numeric_jacobian(fn, inf.theta, rel=1e-6)
    return est, J @ inf.V @ J.T


# ============================================================ categorical frequency-table fit
def pattern_chi2(meas: Measurement, par: Dict[str, Any], npar: int, max_cells: int = 200_000):
    if meas.Jy or not meas.Jc or (~meas.obsU).any():
        return None
    cells = int(np.prod(meas.ncat))
    if cells > max_cells:
        return None
    strides = np.ones(meas.Jc, dtype=np.int64)
    for j in range(meas.Jc - 2, -1, -1):
        strides[j] = strides[j + 1] * meas.ncat[j + 1]
    codes = meas.U @ strides
    counts = np.bincount(codes, minlength=cells).astype(float)
    grid = np.array(np.unravel_index(np.arange(cells), meas.ncat)).T
    lp = np.log(np.clip(par["p"], 1e-300, None))[None].repeat(cells, 0)
    for j in range(meas.Jc):
        lp = lp + np.log(np.clip(par["meas"]["P"][j], 1e-300, None))[:, grid[:, j]].T
    E = meas.n * np.exp(logsumexp(lp, axis=1))
    pearson = float(((counts - E) ** 2 / np.maximum(E, 1e-300)).sum())
    pos = counts > 0
    lr = float(2 * (counts[pos] * np.log(counts[pos] / E[pos])).sum())
    df = cells - 1 - npar
    if df <= 0:
        return None
    from scipy.stats import chi2
    return {"cells": cells, "pearson": pearson, "lr": lr, "df": df,
            "p_pearson": float(chi2.sf(pearson, df)), "p_lr": float(chi2.sf(lr, df))}


# ============================================================ likelihood-ratio tests for k-1 vs k classes
def vlmr_test(inf1: Inference, inf0: Inference, ll1: float, ll0: float, n: int, method: str = "mplus") -> Dict[str, Any]:
    """Vuong-Lo-Mendell-Rubin LRT and the Lo-Mendell-Rubin adjusted LRT (Mplus TECH11)."""
    S1, S0 = inf1.scores[:, inf1.free], inf0.scores[:, inf0.free]
    A1, A0 = inf1.hessian, inf0.hessian
    B1, B0, B10 = S1.T @ S1, S0.T @ S0, S1.T @ S0
    A1i, A0i = np.linalg.pinv(A1), np.linalg.pinv(A0)
    if method == "vuong":
        W = np.block([[-B1 @ A1i, -B10 @ A0i], [B10.T @ A1i, B0 @ A0i]])
    else:
        V1, V0 = A1i @ B1 @ A1i, A0i @ B0 @ A0i
        W = np.block([[B1 @ V1, B10 @ V0], [-B10.T @ V1, -B0 @ V0]])
    lam = np.real(np.linalg.eigvals(W))
    lr = 2.0 * (ll1 - ll0)
    dp = int(inf1.free.size - inf0.free.size)
    adj = lr / (1.0 + 1.0 / (max(dp, 1) * math.log(n)))
    return {"lr": lr, "dp": dp, "mean": float(lam.sum()), "sd": float(math.sqrt(2 * (lam ** 2).sum())),
            "p": imhof_upper(lr, lam), "adj": adj, "p_adj": imhof_upper(adj, lam), "weights": lam}


def blrt(fit0: MixtureFit, K1: int, lr_obs: float, draws: int = 100, lrtstarts=(0, 0, 40, 8), stiter: int = 10,
         seed: int = 0, max_iter: int = 5000, tol: float = 1e-6, make_model=None) -> Dict[str, Any]:
    """Parametric bootstrap LRT (Mplus TECH14): data generated from the k-1 class estimates."""
    rng = np.random.default_rng(int(seed) + 7919)
    lrs = []
    make_model = make_model or (lambda meas, K: LatentClassModel(meas, K))
    for b in range(int(draws)):
        r = np.random.default_rng(int(rng.integers(1, 1_000_001)))
        z = r.choice(fit0.model.K, size=fit0.n, p=fit0.par["p"])
        sim = fit0.model.meas.simulate(fit0.par["meas"], z, r)
        m0 = make_model(sim, fit0.model.K)
        if lrtstarts[0] > 0:
            f0 = estimate(m0, lrtstarts[0], lrtstarts[1], stiter, int(r.integers(1, 1_000_001)), max_iter, tol)
        else:
            f0 = estimate(m0, init=fit0.par, max_iter=max_iter, tol=tol)
        m1 = make_model(sim, K1)
        f1 = estimate(m1, lrtstarts[2], lrtstarts[3], stiter, int(r.integers(1, 1_000_001)), max_iter, tol)
        lrs.append(2.0 * (f1.ll - f0.ll))
    lrs = np.array(lrs)
    return {"lr": lr_obs, "draws": int(draws), "p": float(np.mean(lrs >= lr_obs - 1e-9)), "boot": lrs}


# ============================================================ three-step auxiliary models
def most_likely(post: np.ndarray) -> np.ndarray:
    return np.argmax(post, axis=1)


def classification_error(post: np.ndarray, mlc: np.ndarray) -> np.ndarray:
    """D[k, s] = P(most likely class s | latent class k) (Mplus classification probabilities, rows = latent class)."""
    K = post.shape[1]
    M = np.eye(K)[mlc]
    D = post.T @ M
    return D / np.maximum(D.sum(1, keepdims=True), TINY)


def average_posterior(post: np.ndarray, mlc: np.ndarray) -> np.ndarray:
    """Average latent class probabilities for most likely class membership (row) by latent class (column)."""
    K = post.shape[1]
    out = np.full((K, K), np.nan)
    for s in range(K):
        m = mlc == s
        if m.any():
            out[s] = post[m].mean(0)
    return out


def _sandwich(H: np.ndarray, S: np.ndarray, estimator: str) -> Tuple[np.ndarray, bool]:
    H = (H + H.T) / 2
    try:
        np.linalg.cholesky(-H)
        Ai = np.linalg.inv(-H)
        sing = False
    except np.linalg.LinAlgError:
        Ai = np.linalg.pinv(-H)
        sing = True
    V = Ai if estimator == "ml" else Ai @ (S.T @ S) @ Ai
    return (V + V.T) / 2, sing


def r3step(post: np.ndarray, Z: np.ndarray, estimator: str = "mlr") -> Dict[str, Any]:
    """ML-corrected three-step multinomial logit of the latent class on covariates; reference = last class.

    The classification-error matrix comes from all step-1 cases; cases with a missing covariate are
    excluded from step 3 (listwise, as in Mplus R3STEP).
    """
    K = post.shape[1]
    mlc_all = most_likely(post)
    D = np.clip(classification_error(post, mlc_all), 1e-15, 1)
    Z = np.asarray(Z, dtype=float).reshape(post.shape[0], -1)
    ok = np.isfinite(Z).all(1)
    Z, post, mlc = Z[ok], post[ok], mlc_all[ok]
    n = post.shape[0]
    Dcol = D[:, mlc].T  # (n, K): P(s_i | class k)
    Xt = np.column_stack([np.ones(n), Z])
    q = Xt.shape[1]
    p0 = np.clip(post.mean(0), 1e-6, None)
    g0 = np.zeros((K - 1, q))
    g0[:, 0] = np.log(p0[:-1] / p0[-1])

    def probs(g):
        eta = np.column_stack([Xt @ g.reshape(K - 1, q).T, np.zeros(n)])
        eta -= eta.max(1, keepdims=True)
        e = np.exp(eta)
        return e / e.sum(1, keepdims=True)

    def negll(g):
        return -np.log(np.clip((probs(g) * Dcol).sum(1), 1e-300, None)).sum()

    def case_scores(g):
        pz = probs(g)
        r = pz * Dcol
        r /= np.clip(r.sum(1, keepdims=True), 1e-300, None)
        d = (r - pz)[:, :K - 1]
        return (d[:, :, None] * Xt[:, None, :]).reshape(n, -1)

    res = optimize.minimize(negll, g0.ravel(), jac=lambda g: -case_scores(g).sum(0), method="BFGS",
                            options={"gtol": 1e-8, "maxiter": 2000})
    g = res.x
    for _ in range(20):  # Newton polish from the BFGS solution
        grad = case_scores(g).sum(0)
        if np.max(np.abs(grad)) < 1e-6:
            break
        Hn = numeric_jacobian(lambda x: case_scores(x).sum(0), g)
        try:
            g = g - np.linalg.solve(Hn, grad)
        except np.linalg.LinAlgError:
            break
    grad_max = float(np.max(np.abs(case_scores(g).sum(0))))
    H = numeric_jacobian(lambda x: case_scores(x).sum(0), g)
    V, sing = _sandwich(H, case_scores(g), estimator)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    return {"coef": g.reshape(K - 1, q), "se": se.reshape(K - 1, q), "V": V, "ll": -float(negll(g)),
            "converged": grad_max < 1e-3, "singular": sing, "n": n, "n_dropped": int((~ok).sum()), "D": D}


def bch_weights(post: np.ndarray) -> np.ndarray:
    mlc = most_likely(post)
    D = classification_error(post, mlc)
    return np.linalg.pinv(D)[mlc]  # (n, K)


def bch_continuous(post: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Class-specific means of a continuous distal outcome with BCH weights and Wald equality tests."""
    w = bch_weights(post)
    ok = np.isfinite(y)
    w, yy = w[ok], y[ok]
    K = post.shape[1]
    Wk = w.sum(0)
    mu = (w * yy[:, None]).sum(0) / Wk
    var = (w * (yy[:, None] - mu[None]) ** 2).sum(0) / Wk
    e = w * (yy[:, None] - mu[None]) / Wk[None]
    V = e.T @ e
    return {"mean": mu, "var": var, "V": V, "se": np.sqrt(np.clip(np.diag(V), 0, None)), "n": int(ok.sum()),
            "tests": wald_equality(mu, V)}


def wald_equality(est: np.ndarray, V: np.ndarray, groups: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
    """Overall and pairwise Wald chi-square tests of equal class values (est indexed by class, or class x category)."""
    from scipy.stats import chi2
    K = est.shape[0]
    est2 = est.reshape(K, -1)
    m = est2.shape[1]
    idx = np.arange(K * m).reshape(K, m)

    def test(rows):
        C = []
        for a, b in rows:
            for c in range(m):
                v = np.zeros(K * m)
                v[idx[a, c]] = 1.0
                v[idx[b, c]] = -1.0
                C.append(v)
        C = np.array(C)
        d = C @ est2.ravel()
        S = C @ V @ C.T
        stat = float(d @ np.linalg.pinv(S) @ d)
        df = C.shape[0]
        return stat, df, float(chi2.sf(stat, df))

    overall = test([(k, K - 1) for k in range(K - 1)])
    pairs = []
    for a in range(K):
        for b in range(a + 1, K):
            pairs.append((a, b) + test([(a, b)]))
    return {"overall": overall, "pairs": pairs}


def ml3step_categorical(post: np.ndarray, u: np.ndarray, C: int, estimator: str = "mlr", max_iter: int = 2000) -> Dict[str, Any]:
    """ML three-step model for a categorical distal outcome (class-specific category probabilities)."""
    n, K = post.shape
    mlc = most_likely(post)
    D = np.clip(classification_error(post, mlc), 1e-15, 1)
    ok = u >= 0
    Dcol = D[:, mlc[ok]].T
    uu = u[ok]
    oh = np.eye(C)[uu]
    pi = np.clip(post.mean(0), 1e-6, None)
    P = np.tile(np.clip(oh.mean(0), 1e-6, None), (K, 1))
    ll_old = -np.inf
    for _ in range(max_iter):
        r = pi[None] * Dcol * P[:, uu].T
        s = r.sum(1, keepdims=True)
        ll = float(np.log(s).sum())
        r /= s
        pi = r.mean(0)
        P = (r.T @ oh + TINY) / (r.sum(0)[:, None] + C * TINY)
        if abs(ll - ll_old) < 1e-9:
            break
        ll_old = ll
    cref = int(np.argmax(pi))
    pref = np.argmax(P, axis=1)

    def unpack(th):
        return _mn_unpack(th[:K - 1], np.array([cref]), K)[0], _mn_unpack(th[K - 1:], pref, C)

    def case_scores(th):
        pi_, P_ = unpack(th)
        rr = pi_[None] * Dcol * P_[:, uu].T
        rr /= rr.sum(1, keepdims=True)
        cols = [rr[:, [k for k in range(K) if k != cref]] - pi_[[k for k in range(K) if k != cref]][None]]
        for k in range(K):
            keep = [c for c in range(C) if c != pref[k]]
            cols.append(rr[:, k, None] * (oh[:, keep] - P_[k, keep][None]))
        return np.column_stack(cols)

    th = np.concatenate([_mn_pack(pi[None], np.array([cref])), _mn_pack(P, pref)])
    fixed = np.concatenate([_mn_fixed(pi[None], np.array([cref])), _mn_fixed(P, pref)])
    free = ~fixed

    def tg(x):
        t = th.copy()
        t[free] = x
        return case_scores(t)[:, free].sum(0)

    H = numeric_jacobian(tg, th[free])
    Vf, sing = _sandwich(H, case_scores(th)[:, free], estimator)
    Vth = np.zeros((th.size, th.size))
    Vth[np.ix_(free, free)] = Vf
    J = numeric_jacobian(lambda x: unpack(x)[1].ravel(), th, rel=1e-6)
    VP = J @ Vth @ J.T
    se = np.sqrt(np.clip(np.diag(VP), 0, None)).reshape(K, C)
    se = np.where(se < 1e-10, np.nan, se)
    sub = np.array([[k * C + c for c in range(C - 1)] for k in range(K)]).ravel()
    tests = wald_equality(P[:, :C - 1], VP[np.ix_(sub, sub)])
    return {"prob": P, "se": se, "class_prop": pi, "n": int(ok.sum()), "tests": tests, "singular": sing}
