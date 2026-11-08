"""
OpenStat Studio — count-data regression (Stata poisson · nbreg · zip · zinb · tpoisson · tnbreg; logit hurdle as R pscl::hurdle).
License: MIT

One analysis id ``countReg`` with a ``family`` option:

    poisson          Poisson regression                                   (Stata poisson)
    nb2              negative binomial, mean dispersion  Var = mu(1 + alpha mu)  (nbreg)
    nb1              negative binomial, constant dispersion Var = mu(1 + delta)  (nbreg, dispersion(constant))
    zip / zinb       zero-inflated Poisson / NB2, inflation link logit | probit  (zip, zinb [, probit])
    ztp              truncated Poisson, y > ll (default ll = 0)            (tpoisson, ll(#))
    ztnb / ztnb1     truncated NB2 / NB1, y > ll                           (tnbreg, ll(#) [dispersion(constant)])
    hurdle_poisson   logit hurdle P(y > 0) + zero-truncated Poisson         (R pscl::hurdle; Stata has no built-in command)
    hurdle_nb        logit hurdle P(y > 0) + zero-truncated NB2             (R pscl::hurdle(dist = "negbin"))

Computation: statsmodels discrete count models (likelihood, analytic score, fitting, predictions; see CountModel).  The
observed information matrix (OIM, Stata's default ``vce(oim)``) is the statsmodels.tools.numdiff Jacobian of the
statsmodels analytic score.  ``vce(robust)`` uses the sandwich with the
N/(N-1) factor and ``vce(cluster)`` the G/(G-1) factor (Stata ``ml`` conventions).  Parameterizations follow Stata:
``/lnalpha`` (NB2) or ``/lndelta`` (NB1) is estimated, alpha = exp(lnalpha) is shown with delta-method SE and the
CI transformed from the lnalpha CI; R's theta = 1/alpha.
"""
from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP, ZeroInflatedPoisson
from statsmodels.discrete.discrete_model import NegativeBinomialP
from statsmodels.discrete.truncated_model import TruncatedLFNegativeBinomialP, TruncatedLFPoisson
from statsmodels.tools.numdiff import approx_fprime, approx_hess3

from .base import BaseAnalysis, ResultsTable, resolve_choice

# --------------------------------------------------------------------------------------------- families
FAMILIES: Dict[str, Dict[str, str]] = {
    "poisson": {"dist": "poisson", "kind": "plain", "ko": "포아송 회귀", "stata": "poisson"},
    "nb2": {"dist": "nb2", "kind": "plain", "ko": "음이항 회귀 NB2", "stata": "nbreg"},
    "nb1": {"dist": "nb1", "kind": "plain", "ko": "음이항 회귀 NB1", "stata": "nbreg, dispersion(constant)"},
    "zip": {"dist": "poisson", "kind": "zi", "ko": "영과잉 포아송 회귀", "stata": "zip"},
    "zinb": {"dist": "nb2", "kind": "zi", "ko": "영과잉 음이항 회귀", "stata": "zinb"},
    "ztp": {"dist": "poisson", "kind": "trunc", "ko": "절단 포아송 회귀", "stata": "tpoisson"},
    "ztnb": {"dist": "nb2", "kind": "trunc", "ko": "절단 음이항 회귀 NB2", "stata": "tnbreg"},
    "ztnb1": {"dist": "nb1", "kind": "trunc", "ko": "절단 음이항 회귀 NB1", "stata": "tnbreg, dispersion(constant)"},
    "hurdle_poisson": {"dist": "poisson", "kind": "hurdle", "ko": "허들 포아송 회귀", "stata": "logit + tpoisson"},
    "hurdle_nb": {"dist": "nb2", "kind": "hurdle", "ko": "허들 음이항 회귀", "stata": "logit + tnbreg"},
}
FAMILY_ALIASES = {
    "nbreg": "nb2", "negbin": "nb2", "nb": "nb2", "negative_binomial": "nb2", "nbreg_constant": "nb1",
    "tpoisson": "ztp", "zero_truncated_poisson": "ztp", "tnbreg": "ztnb", "tnbreg_constant": "ztnb1",
    "hurdle": "hurdle_poisson", "hp": "hurdle_poisson", "hnb": "hurdle_nb", "hurdle_negbin": "hurdle_nb",
    "hurdlepoisson": "hurdle_poisson", "hurdlenb": "hurdle_nb",
}
# analysis names accepted by sess.run / CLI / MCP and the family each one presets
ALIAS_PRESETS = {"poisson": "poisson", "nbreg": "nb2", "zip": "zip", "zinb": "zinb", "tpoisson": "ztp",
                 "tnbreg": "ztnb", "ztp": "ztp", "ztnb": "ztnb", "hurdle": "hurdle_poisson", "countreg": None}
VCE_TYPES = ("oim", "robust", "cluster", "opg")
_VCE_ALIASES = {"conventional": "oim", "classical": "oim", "default": "oim", "hc0": "robust", "hc": "robust",
                "sandwich": "robust", "clustered": "cluster", "cl": "cluster"}
LINKS = ("logit", "probit")


# --------------------------------------------------------------------------------------------- option helpers
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


def _as_list(val: Any) -> List[str]:
    if val is None or val == "":
        return []
    if isinstance(val, str):
        return [v.strip() for v in val.split(",") if v.strip()]
    return [str(v) for v in val if v is not None and str(v) != ""]


def _as_single(val: Any) -> Any:
    if isinstance(val, (list, tuple)):
        return val[0] if len(val) > 0 else None
    if val == "":
        return None
    return val


def _parse_ref_levels(val: Any) -> Dict[str, str]:
    """{'region': '서울'} or "region=서울; edu=2" -> {var: level text}."""
    if val is None or val == "" or val == {}:
        return {}
    if isinstance(val, dict):
        return {str(k).strip(): str(v).strip() for k, v in val.items() if str(k).strip()}
    txt = str(val)
    parts = re.split(r"[;\n]+", txt) if re.search(r"[;\n]", txt) else txt.split(",")
    out = {}
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"기준 범주(refLevels) 형식 오류: '{part}' (예: 지역=대도시; 소득분위=1)")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip().strip("'\"")
    return out


def _level_label(v: Any) -> str:
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


# --------------------------------------------------------------------------------------------- design matrices
class _Design:
    """Numeric covariates and dummy-coded factors with levels fixed on the estimation sample."""

    def __init__(self, frame: pd.DataFrame, factors: Sequence[str], ref_levels: Dict[str, str]):
        self.factors = list(factors)
        self.levels: Dict[str, List[Any]] = {}
        self.base: Dict[str, Any] = {}
        for f in self.factors:
            col = frame[f]
            if isinstance(col.dtype, pd.CategoricalDtype):
                lv = [c for c in col.cat.categories if (col == c).any()]
            elif pd.api.types.is_numeric_dtype(col):
                lv = sorted(pd.unique(col.astype(float)))
            else:
                lv = sorted(pd.unique(col.astype(str)))
            if len(lv) < 2:
                raise ValueError(f"요인 '{f}': 분석 표본에서 수준이 1개뿐 ({_level_label(lv[0]) if lv else '없음'})")
            self.levels[f] = lv
            ref = ref_levels.get(f)
            if ref is None:
                self.base[f] = lv[0]                     # Stata default base: lowest level
            else:
                match = [v for v in lv if _level_label(v) == ref or str(v) == ref]
                if not match:
                    try:
                        match = [v for v in lv if isinstance(v, (int, float, np.number)) and float(v) == float(ref)]
                    except ValueError:
                        match = []
                if not match:
                    raise ValueError(f"요인 '{f}': 기준 범주 '{ref}' 없음 (가능: {', '.join(_level_label(v) for v in lv)})")
                self.base[f] = match[0]
        for f in ref_levels:
            if f not in self.levels:
                raise ValueError(f"기준 범주(refLevels): '{f}'는 요인(factors)으로 지정되지 않은 변수")

    def _codes(self, col: pd.Series, f: str) -> pd.Series:
        lv = self.levels[f]
        if isinstance(lv[0], (float, np.floating)) and pd.api.types.is_numeric_dtype(col):
            return col.astype(float)
        if isinstance(col.dtype, pd.CategoricalDtype):
            return col.astype(object)
        return col.astype(str) if not isinstance(lv[0], (float, np.floating)) else col.astype(float)

    def build(self, frame: pd.DataFrame, variables: Sequence[str], constant: bool):
        """Matrix, column names and term map [(variable, kind, level, column index)]."""
        cols, names, terms = [], [], []
        for v in variables:
            if v in self.levels:
                codes = self._codes(frame[v], v)
                for lev in self.levels[v]:
                    if lev == self.base[v]:
                        continue
                    terms.append((v, "factor", lev, len(cols)))
                    cols.append((codes == lev).to_numpy(dtype=float))
                    names.append(f"{v}={_level_label(lev)}")
            else:
                terms.append((v, "cont", None, len(cols)))
                cols.append(frame[v].to_numpy(dtype=float))
                names.append(v)
        if constant:
            terms.append(("_cons", "const", None, len(cols)))
            cols.append(np.ones(len(frame)))
            names.append("_cons")
        M = np.column_stack(cols) if cols else np.zeros((len(frame), 0))
        return M, names, terms


def _drop_collinear(M: np.ndarray, names: List[str], terms: List[tuple], keep_last: bool) -> Tuple[np.ndarray, List[str], List[tuple], List[str]]:
    """Drop columns that are linear combinations of earlier ones (Stata: 'omitted because of collinearity')."""
    if M.shape[1] == 0:
        return M, names, terms, []
    order = list(range(M.shape[1]))
    if keep_last and names and names[-1] == "_cons":   # the constant is kept first, as Stata does
        order = [M.shape[1] - 1] + order[:-1]
    keep, dropped = [], []
    scale = np.sqrt((M ** 2).mean(axis=0)) + 1e-300
    for j in order:
        trial = keep + [j]
        sub = M[:, trial] / scale[trial]
        if np.linalg.matrix_rank(sub, tol=1e-9 * max(sub.shape)) == len(trial):
            keep.append(j)
        else:
            dropped.append(names[j])
    keep = sorted(keep)
    remap = {old: new for new, old in enumerate(keep)}
    new_terms = [(v, k, lev, remap[c]) for (v, k, lev, c) in terms if c in remap]
    return M[:, keep], [names[j] for j in keep], new_terms, dropped


# --------------------------------------------------------------------------------------------- statsmodels engine
_P_OF = {"nb2": 2, "nb1": 1}


def _numeric_score(cls):
    """statsmodels class with the score replaced by centered numerical derivatives of its log likelihood.

    Used where the statsmodels score is unusable (statsmodels 0.15):
    - ZeroInflatedPoisson / ZeroInflatedNegativeBinomialP with inflation='probit': score_obs is a forward-difference
      approx_fprime (SEs from its Jacobian 7% (ZIP) to 63% (ZINB) off Stata on Count_HealthVisits);
    - TruncatedLFNegativeBinomialP(p=1): analytic score wrong (2013 at Stata's maximum, numerical gradient ~1e-6),
      so the fit stops 5.2 log-likelihood units below Stata; its log likelihood equals Stata's to 1e-12.
    """
    class _Patched(cls):
        def score_obs(self, params):
            return approx_fprime(np.asarray(params, dtype=float), self.loglikeobs, centered=True)

        def score(self, params):
            return approx_fprime(np.asarray(params, dtype=float), self.loglike, centered=True)

        def hessian(self, params):
            return approx_hess3(np.asarray(params, dtype=float), self.loglike)
    _Patched.__name__ = cls.__name__ + "NumericScore"
    return _Patched


def _quiet():
    ctx = warnings.catch_warnings()
    ctx.__enter__()
    warnings.simplefilter("ignore")
    return ctx


class CountModel:
    """One statsmodels count model; parameters are exposed in Stata order: count eq., inflate / zero eq., /lnalpha.

    Likelihood, analytic score, optimisation and predictions come from statsmodels
    (Poisson, NegativeBinomialP, ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP, TruncatedLFPoisson,
    TruncatedLFNegativeBinomialP; hurdle = Logit/Probit for P(y>0) + zero-truncated model on y>0).  Glue: Stata parameter
    order, alpha -> ln(alpha), and the observed information as the numerical Jacobian of the statsmodels score
    (statsmodels' analytic ZIP Hessian omits the inflate x count block, see notes).
    """

    def __init__(self, y, X, offset=None, dist="poisson", kind="plain", Z=None, link="logit", trunc=0):
        self.y = np.asarray(y, dtype=float)
        self.n = len(self.y)
        X = np.asarray(X, dtype=float)
        self.kx_true = X.shape[1]
        self.fixed_zero = X.shape[1] == 0           # offset-only comparison model (noconstant): one coefficient fixed at 0
        self.X = np.zeros((self.n, 1)) if self.fixed_zero else X
        self.off = np.zeros(self.n) if offset is None else np.asarray(offset, dtype=float)
        self.dist, self.kind, self.link, self.trunc = dist, kind, link, int(trunc)
        self.Z = np.asarray(Z, dtype=float) if kind in ("zi", "hurdle") else np.zeros((self.n, 0))
        self.kx, self.kz = self.X.shape[1], self.Z.shape[1]
        self.has_a = dist != "poisson"
        self.k = self.kx + self.kz + int(self.has_a)
        y, X, off = self.y, self.X, self.off
        tnb = _numeric_score(TruncatedLFNegativeBinomialP) if dist == "nb1" else TruncatedLFNegativeBinomialP
        # numerical Hessian of the log likelihood where the statsmodels score is numerical or wrong (see _numeric_score)
        self.numeric = (kind == "zi" and link == "probit") or (kind in ("trunc", "hurdle") and dist == "nb1")
        if kind == "plain":
            self.m = sm.Poisson(y, X, offset=off) if dist == "poisson" else NegativeBinomialP(y, X, offset=off, p=_P_OF[dist])
        elif kind == "trunc":
            self.m = (TruncatedLFPoisson(y, X, offset=off, truncation=self.trunc) if dist == "poisson"
                      else tnb(y, X, offset=off, truncation=self.trunc, p=_P_OF[dist]))
        elif kind == "zi":
            zp = ZeroInflatedPoisson if dist == "poisson" else ZeroInflatedNegativeBinomialP
            zp = _numeric_score(zp) if link == "probit" else zp
            self.m = (zp(y, X, exog_infl=self.Z, offset=off, inflation=link) if dist == "poisson"
                      else zp(y, X, exog_infl=self.Z, offset=off, inflation=link, p=2))
        else:
            self.pos = y > 0
            self.mb = (sm.Logit if link == "logit" else sm.Probit)(self.pos.astype(float), self.Z)
            p = self.pos
            self.mc = (TruncatedLFPoisson(y[p], X[p], offset=off[p], truncation=0) if dist == "poisson"
                       else tnb(y[p], X[p], offset=off[p], truncation=0, p=_P_OF[dist]))

    # ---- parameter maps (Stata order <-> statsmodels order)
    def split(self, t):
        t = np.asarray(t, dtype=float)
        return t[: self.kx], t[self.kx: self.kx + self.kz], (float(t[-1]) if self.has_a else None)

    @property
    def lib_name(self) -> str:
        def nm(m):
            base = type(m).__mro__[1].__name__ if type(m).__name__.endswith("NumericScore") else type(m).__name__
            p_ = getattr(m, "parameterization", None) or getattr(getattr(m, "model_main", None), "parameterization", None)
            return f"{base}(p={p_})" if p_ is not None and "NegativeBinomialP" in base else base
        return f"{nm(self.mb)} + {nm(self.mc)}" if self.kind == "hurdle" else nm(self.m)

    def _lib(self, t):
        b, g, a = self.split(t)
        tail = [np.exp(a)] if self.has_a else []
        if self.kind == "zi":
            return np.concatenate([g, b, tail])
        if self.kind == "hurdle":
            return g, np.concatenate([b, tail])
        return np.concatenate([b, tail])

    def _from_lib(self, p, pc=None):
        if self.kind == "hurdle":
            b = pc[: self.kx]
            return np.concatenate([b, p, [np.log(pc[-1])] if self.has_a else []])
        if self.kind == "zi":
            g, b = p[: self.kz], p[self.kz: self.kz + self.kx]
            return np.concatenate([b, g, [np.log(p[-1])] if self.has_a else []])
        return np.concatenate([p[: self.kx], [np.log(p[-1])] if self.has_a else []])

    # ---- likelihood and score (statsmodels)
    def loglike_obs(self, t):
        if self.kind == "hurdle":
            g, pc = self._lib(t)
            ll = self.mb.loglikeobs(g)
            ll[self.pos] += self.mc.loglikeobs(pc)
            return ll
        return self.m.loglikeobs(self._lib(t))

    def loglike(self, t):
        with np.errstate(all="ignore"):
            v = float(np.sum(self.loglike_obs(t)))
        return v if np.isfinite(v) else -np.inf

    def score_obs(self, t):
        """n x k scores in Stata order (d/d ln alpha = alpha * d/d alpha)."""
        with np.errstate(all="ignore"):
            if self.kind == "hurdle":
                g, pc = self._lib(t)
                Sc = np.zeros((self.n, len(pc)))
                Sc[self.pos] = self.mc.score_obs(pc)
                S = np.hstack([Sc[:, : self.kx], self.mb.score_obs(g), Sc[:, self.kx:]])
            else:
                p = self._lib(t)
                Sl = self.m.score_obs(p)
                if self.kind == "zi":
                    S = np.hstack([Sl[:, self.kz: self.kz + self.kx], Sl[:, : self.kz], Sl[:, self.kz + self.kx:]])
                else:
                    S = Sl
            if self.has_a:
                S = S.copy()
                S[:, -1] *= np.exp(t[-1])
        if self.fixed_zero:
            S = S.copy()
            S[:, 0] = 0.0
        return S

    def score(self, t):
        return self.score_obs(t).sum(axis=0)

    def hessian(self, t):
        """Observed information (negative): statsmodels.tools.numdiff Jacobian of the statsmodels analytic score
        (approx_hess3 of the statsmodels log likelihood for the classes in _numeric_score)."""
        t = np.asarray(t, dtype=float)
        with np.errstate(all="ignore"):
            H = approx_hess3(t, self.loglike) if self.numeric else approx_fprime(t, self.score, centered=True)
        H = 0.5 * (H + H.T)
        if self.fixed_zero:
            H[0, :] = H[:, 0] = 0.0
            H[0, 0] = -1.0
        return H

    # ---- predictions (statsmodels predict)
    def mean(self, t, X=None, Z=None, off=None):
        """E[y | x]: Stata predict n (truncated models: exp(xb) before truncation, Stata default)."""
        X = self.X if X is None else X
        off = self.off if off is None else off
        if self.kind == "hurdle":
            g, pc = self._lib(t)
            Z = self.Z if Z is None else Z
            return self.mb.predict(g, exog=Z) * self.mc.predict(pc, exog=X, offset=off, which="mean")
        p = self._lib(t)
        if self.kind == "zi":
            return self.m.predict(p, exog=X, exog_infl=self.Z if Z is None else Z, offset=off, which="mean")
        if self.kind == "trunc":
            return self.m.predict(p, exog=X, offset=off, which="mean-main")
        return self.m.predict(p, exog=X, offset=off)

    def prob_matrix(self, t, counts: Sequence[int]) -> np.ndarray:
        """P(y = j | x_i) (rows j, columns i); truncated models: P(y = j | y > ll)."""
        counts = np.asarray(list(counts), dtype=float)
        if self.kind == "hurdle":
            g, pc = self._lib(t)
            p1 = self.mb.predict(g)
            Pc = np.atleast_2d(self.mc.predict(pc, exog=self.X, offset=self.off, which="prob", y_values=counts))
            out = p1[:, None] * Pc
            out[:, counts == 0] = (1.0 - p1)[:, None]
            return out.T
        P = np.atleast_2d(self.m.predict(self._lib(t), which="prob", y_values=counts))
        if self.kind == "trunc":
            P = np.where(counts[None, :] > self.trunc, P, 0.0)
        return P.T

    # ---- start values (statsmodels GLM / Logit / Probit)
    def default_start(self, beta=None):
        y, X, off = self.y, self.X, self.off
        sel = (y > 0) if self.kind == "hurdle" else np.ones(self.n, dtype=bool)
        q = _quiet()
        try:
            if beta is None:
                beta = np.zeros(self.kx)
                if not self.fixed_zero:
                    try:
                        beta = sm.GLM(y[sel], X[sel], family=sm.families.Poisson(), offset=off[sel]).fit().params
                    except Exception:
                        pass
            parts = [np.asarray(beta, dtype=float)]
            if self.kz:
                target = (y > 0) if self.kind == "hurdle" else (y == 0)
                try:
                    parts.append((sm.Logit if self.link == "logit" else sm.Probit)(target.astype(float), self.Z).fit(disp=0).params)
                except Exception:
                    parts.append(np.zeros(self.kz))
            if self.has_a:
                mu = np.exp(np.clip(X[sel] @ parts[0] + off[sel], -700, 700))
                ys = y[sel]
                v = np.sum((ys - mu) ** 2 - ys) / max(np.sum(mu ** 2 if self.dist == "nb2" else mu), 1e-12)
                parts.append(np.array([np.log(min(max(v, 0.05), 20.0))]))
        finally:
            q.__exit__(None, None, None)
        return np.concatenate(parts)

    # ---- fitting (statsmodels fit; best likelihood over the start values)
    def _fit_one(self, start):
        q = _quiet()
        try:
            if self.fixed_zero:
                free = np.arange(1, self.k)
                if free.size == 0:
                    return np.zeros(self.k)
                f = lambda u: -self.loglike(np.concatenate([[0.0], u]))
                res = optimize.minimize(f, start[1:], method="BFGS", jac=lambda u: -self.score(np.concatenate([[0.0], u]))[1:])
                return np.concatenate([[0.0], res.x])
            if self.kind == "hurdle":
                g0, pc0 = self._lib(start)
                g = self.mb.fit(start_params=g0, method="newton", maxiter=100, disp=0).params
                pc = self._fit_lib(self.mc, pc0, ("newton", "bfgs"))
                return self._from_lib(g, pc)
            methods = ("bfgs",) if self.kind == "zi" else ("newton", "bfgs")
            return self._from_lib(self._fit_lib(self.m, self._lib(start), methods))
        except Exception:
            return None
        finally:
            q.__exit__(None, None, None)

    @staticmethod
    def _fit_lib(model, p0, methods):
        best, best_ll = None, -np.inf
        for meth in methods:
            try:
                kw = {"gtol": 1e-10} if meth == "bfgs" else {}
                r = model.fit(start_params=p0, method=meth, maxiter=5000 if meth == "bfgs" else 200, disp=0, **kw)
            except Exception:
                continue
            ll = float(model.loglike(r.params))
            if np.isfinite(ll) and ll > best_ll + 1e-10:
                best, best_ll = r.params, ll
        if best is None:
            raise RuntimeError("statsmodels fit failed")
        return best

    def fit(self, starts=()) -> Dict[str, Any]:
        cands = [self.default_start()] + [np.asarray(s, dtype=float) for s in starts
                                          if s is not None and len(s) == self.k and np.all(np.isfinite(s))]
        best_t, best_ll = None, -np.inf
        for st in cands:
            t = self._fit_one(st)
            if t is None:
                continue
            ll = self.loglike(t)
            if ll > best_ll + 1e-9:
                best_t, best_ll = t, ll
        if best_t is None:
            best_t, best_ll = cands[0], self.loglike(cands[0])
        H = self.hessian(best_t)
        s = self.score(best_t)
        try:
            ev = np.linalg.eigvalsh(-H)
            pd_ok = bool(ev.min() > 1e-10 * max(1.0, ev.max()))
            crit = float(s @ np.linalg.solve(-H, s)) if pd_ok else np.inf
        except np.linalg.LinAlgError:
            pd_ok, crit = False, np.inf
        conv = bool(pd_ok and np.isfinite(best_ll) and abs(crit) < 1e-6)
        return {"theta": best_t, "ll": float(best_ll), "H": H, "converged": conv, "iterations": len(cands), "pd": pd_ok,
                "grad_max": float(np.max(np.abs(s))) if s.size else 0.0}


def vcov(lik: CountModel, fit: Dict[str, Any], vce: str = "oim", clusters: Optional[np.ndarray] = None) -> np.ndarray:
    """Stata ml variance: oim = (-H)^-1, opg = (S'S)^-1, robust = sandwich x N/(N-1), cluster = sandwich x G/(G-1).

    The sandwich equals statsmodels cov_type='HC0' (robust) and cov_type='cluster' with use_correction=False times the
    Stata factor; it is formed here because the bread must be the corrected observed information (ZIP)."""
    H = fit["H"]
    try:
        Vo = np.linalg.inv(-H)
    except np.linalg.LinAlgError:
        Vo = np.linalg.pinv(-H)
    if vce == "oim":
        return Vo
    S = lik.score_obs(fit["theta"])
    if vce == "opg":
        return np.linalg.pinv(S.T @ S)
    if vce == "robust":
        n = S.shape[0]
        return Vo @ (S.T @ S) @ Vo * (n / (n - 1.0))
    codes, uniq = pd.factorize(pd.Series(clusters))
    G = len(uniq)
    Sg = np.zeros((G, S.shape[1]))
    np.add.at(Sg, codes, S)
    return Vo @ (Sg.T @ Sg) @ Vo * (G / (G - 1.0))


def _unidentified(H: np.ndarray) -> List[int]:
    """Parameters loading on (numerically) zero-curvature directions of the observed information."""
    if H.size == 0 or not np.all(np.isfinite(H)):
        return list(range(H.shape[0])) if H.size else []
    ev, vec_ = np.linalg.eigh(-H)
    small = ev < 1e-9 * max(1.0, float(np.abs(ev).max()))
    if not small.any():
        return []
    load = np.abs(vec_[:, small]).max(axis=1)
    return [int(j) for j in np.where(load > 0.05)[0]]


def vuong_test(ll1: np.ndarray, ll2: np.ndarray, k1: int, k2: int) -> List[Dict[str, Any]]:
    """Vuong non-nested test: raw z (Stata zip/zinb forcevuong), AIC- and BIC-corrected z (pscl::vuong); p = Pr(Z > z)."""
    m = np.asarray(ll1) - np.asarray(ll2)
    ok = np.isfinite(m)
    m = m[ok]
    n = len(m)
    s = float(np.std(m, ddof=1))
    out = []
    for lab, adj in (("raw", 0.0), ("aic", float(k1 - k2)), ("bic", (k1 - k2) * np.log(n) / 2.0)):
        z = (float(np.sum(m)) - adj) / (s * np.sqrt(n)) if s > 0 else np.nan
        p = float(stats.norm.sf(z))
        out.append({"type": lab, "z": z, "p": p, "favors": 1 if z > 0 else 2})
    return out


# --------------------------------------------------------------------------------------------- options
_DEFAULTS = {
    "family": "poisson", "inflate": [], "inflateLink": "logit", "truncPoint": 0, "exposure": None, "offset": None,
    "covType": "oim", "clusterVar": None, "irr": False, "constant": True, "refLevels": {}, "margins": True,
    "predProbs": True, "maxCount": 9, "predPlot": True, "vuong": False, "gof": True, "ciLevel": 95.0,
}


def _family_from_alias(params: Dict[str, Any]) -> Optional[str]:
    """Family preset of the analysis name the user typed (sess.run('nbreg', ...) -> 'nb2').

    The registry passes the typed name as params['_alias'] for extensions that declare "uses_alias".
    """
    name = params.get("_alias") or params.get("_analysis")
    if not name:
        return None
    return ALIAS_PRESETS.get(str(name).strip().lower())


def normalize_options(params: Dict[str, Any], preset: Optional[str] = None) -> Dict[str, Any]:
    """UI camelCase / snake_case parameters -> validated canonical options (unknown select values raise)."""
    p = params or {}
    fam_raw = _opt(p, "family", "family", None)
    if fam_raw is None:
        fam_raw = _opt(p, "model", "dist", None)
    if fam_raw is None:
        fam = preset or "poisson"
    else:
        fam = resolve_choice("모형(family)", fam_raw, tuple(FAMILIES), FAMILY_ALIASES)
        if preset and fam != preset:
            raise ValueError(f"분석 이름과 모형 불일치: 분석 이름은 '{preset}' 모형, family='{fam_raw}' — 둘 중 하나만 지정")
    disp = _opt(p, "dispersion", "dispersion", None)
    if disp is not None:
        disp = resolve_choice("산포 형태(dispersion)", disp, ("mean", "constant"))
        if disp == "constant":
            if fam in ("nb2", "nb1"):
                fam = "nb1"
            elif fam in ("ztnb", "ztnb1"):
                fam = "ztnb1"
            else:
                raise ValueError(f"dispersion(constant)는 nb·ztnb 모형에만 해당 (현재 모형: {fam})")
        elif fam == "nb1" or fam == "ztnb1":
            raise ValueError(f"family='{fam}'와 dispersion='mean' 불일치")
    vce = resolve_choice("표준오차(covType)", _opt(p, "covType", "cov_type", _opt(p, "vce", "vce", "oim")), VCE_TYPES, _VCE_ALIASES)
    cluster = _as_single(_opt(p, "clusterVar", "cluster_var", _opt(p, "cluster", "cluster", None)))
    if vce == "cluster" and not cluster:
        raise ValueError("군집 표준오차(cluster): 군집 변수(clusterVar) 지정 필요")
    level = float(_opt(p, "ciLevel", "ci_level", _opt(p, "level", "level", 95.0)))
    if level < 1:
        level *= 100.0
    if not 10.0 <= level <= 99.99:
        raise ValueError(f"신뢰수준(ciLevel): 10~99.99 사이 값 필요 (입력 {level})")
    trunc = _opt(p, "truncPoint", "trunc_point", _opt(p, "ll", "ll", 0))
    try:
        trunc_f = float(trunc)
    except (TypeError, ValueError):
        raise ValueError(f"절단점(truncPoint): 0 이상의 정수 필요 (입력 '{trunc}'; 변수 절단점은 미지원)")
    if trunc_f < 0 or not float(trunc_f).is_integer():
        raise ValueError(f"절단점(truncPoint): 0 이상의 정수 필요 (입력 {trunc})")
    max_count = int(_opt(p, "maxCount", "max_count", 9))
    if not 1 <= max_count <= 100:
        raise ValueError(f"예측 빈도 최대 횟수(maxCount): 1~100 (입력 {max_count})")
    exposure = _as_single(_opt(p, "exposure", "exposure", None))
    offset = _as_single(_opt(p, "offset", "offset", None))
    if exposure and offset:
        raise ValueError("exposure와 offset은 함께 지정할 수 없음 (Stata 규칙과 동일)")
    return {
        "dep": _as_single(p.get("dep")),
        "covs": _as_list(p.get("covs")),
        "factors": _as_list(p.get("factors")),
        "family": fam,
        "inflate": [v for v in _as_list(_opt(p, "inflate", "inflate_vars", [])) if v != "_cons"],
        "inflateLink": resolve_choice("팽창 방정식 연결함수(inflateLink)", _opt(p, "inflateLink", "inflate_link",
                                      "probit" if _flag(p, "probit", "probit", False) else "logit"), LINKS),
        "truncPoint": int(trunc_f),
        "exposure": exposure,
        "offset": offset,
        "covType": vce,
        "clusterVar": cluster if vce == "cluster" else None,
        "irr": _flag(p, "irr", "irr", False),
        "constant": _flag(p, "constant", "constant", not _flag(p, "noconstant", "no_constant", False)),
        "refLevels": _parse_ref_levels(_opt(p, "refLevels", "ref_levels", {})),
        "margins": _flag(p, "margins", "margins", True),
        "predProbs": _flag(p, "predProbs", "pred_probs", True),
        "maxCount": max_count,
        "predPlot": _flag(p, "predPlot", "pred_plot", True),
        "vuong": _flag(p, "vuong", "vuong", False),
        "gof": _flag(p, "gof", "gof", True),
        "ciLevel": level,
    }


# --------------------------------------------------------------------------------------------- analysis
def _zp(est, se):
    est, se = np.asarray(est, dtype=float), np.asarray(se, dtype=float)
    with np.errstate(all="ignore"):
        z = np.where(se > 0, est / se, np.nan)
    return z, 2.0 * stats.norm.sf(np.abs(z))


def _pstr(p: float) -> str:
    return "< .001" if p < 0.001 else f"= {p:.3f}".replace("0.", ".")


class CountRegression(BaseAnalysis):
    """Count-data regression (Stata poisson / nbreg / zip / zinb / tpoisson / tnbreg; pscl::hurdle)."""

    def __init__(self, data: pd.DataFrame, dep: str, covs: Optional[Sequence[str]] = None,
                 factors: Optional[Sequence[str]] = None, family: str = "poisson", **kwargs):
        params = dict(kwargs, dep=dep, covs=list(covs or []), factors=list(factors or []), family=family)
        opts = normalize_options(params)
        fam = FAMILIES[opts["family"]]
        super().__init__(name="countReg", title=f"{fam['ko']} ({fam['stata']})", dataset=data, options=opts)
        self.spec = fam
        self.run()

    # ---- helpers
    def _fit(self, y, X, off, dist, kind, Z=None, x0=None, extra_starts=()):
        """ML fit from the default start (or x0) and any extra starts; the highest likelihood is kept."""
        lik = CountModel(y, X, off, dist=dist, kind=kind, Z=Z, link=self.options["inflateLink"], trunc=self.options["truncPoint"])
        return lik, lik.fit(([x0] if x0 is not None else []) + list(extra_starts))

    def run(self):
        o = self.options
        df = self.dataset
        fam_key, dist, kind = o["family"], self.spec["dist"], self.spec["kind"]
        dep = o["dep"]
        if not dep:
            raise ValueError(f"가산자료 회귀: 종속변수(dep) 지정 필요. 사용 가능한 변수: {list(df.columns)}")
        notes_model: List[str] = []
        factors = list(dict.fromkeys(o["factors"]))
        covs = [c for c in dict.fromkeys(o["covs"]) if c not in factors]
        infl, infl_fac = [], []
        for v in o["inflate"]:                      # 'i.var' marks a factor used in the inflate equation only
            nm = v[2:] if v.startswith("i.") else v
            if v.startswith("i."):
                infl_fac.append(nm)
            if nm not in infl:
                infl.append(nm)
        if kind not in ("zi", "hurdle") and infl:
            notes_model.append("팽창(영) 방정식 변수는 zip·zinb·허들 모형에서만 사용 — 이 모형에서는 제외")
            infl, infl_fac = [], []
        clash = [v for v in infl_fac if v in covs]
        if clash:
            raise ValueError(f"같은 변수를 계수 방정식(연속)과 팽창 방정식(요인)에서 다르게 코딩할 수 없음: {', '.join(clash)}")
        roles = {"dep": [dep], "covs": covs, "factors": factors, "inflate": infl,
                 "exposure": [o["exposure"]] if o["exposure"] else [], "offset": [o["offset"]] if o["offset"] else [],
                 "clusterVar": [o["clusterVar"]] if o["clusterVar"] else []}
        missing = [f"{r}='{c}'" for r, cs in roles.items() for c in cs if c not in df.columns]
        if missing:
            raise ValueError(f"가산자료 회귀: 변수 없음 — {', '.join(missing)}. 사용 가능한 변수: {list(df.columns)}")
        if dep in covs + factors + infl:
            raise ValueError(f"가산자료 회귀: 종속변수 '{dep}'를 설명변수로 함께 지정할 수 없음")
        count_vars = list(dict.fromkeys(list(o["covs"]) + factors))
        fac_set = list(dict.fromkeys(factors + infl_fac))
        # non-numeric covariates are dummy coded (as linReg / logRegBin do), with a note
        for c in count_vars + infl:
            if c not in fac_set and not pd.api.types.is_numeric_dtype(df[c]):
                fac_set.append(c)
                notes_model.append(f"'{c}': 숫자형이 아니어서 요인(더미 코딩)으로 처리")
        factors = fac_set
        used = list(dict.fromkeys([dep] + count_vars + infl + roles["exposure"] + roles["offset"] + roles["clusterVar"]))
        frame = df[used].copy()
        for c in [dep] + roles["exposure"] + roles["offset"]:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
        n_all = len(frame)
        frame = frame.dropna()
        n_drop = n_all - len(frame)
        if n_drop:
            notes_model.append(f"결측 관측 {n_drop}개 제외 (목록별 삭제)")
        y = frame[dep].to_numpy(dtype=float)
        if len(y) == 0:
            raise ValueError("가산자료 회귀: 결측 제외 후 분석할 관측 없음")
        if (y < 0).any():
            raise ValueError(f"가산자료 회귀: 종속변수 '{dep}'에 음수 {int((y < 0).sum())}개 — 0 이상의 횟수 필요")
        nonint = ~np.isclose(y, np.round(y))
        if nonint.any():
            if fam_key != "poisson":
                raise ValueError(f"{self.spec['ko']}: 종속변수 '{dep}'에 정수가 아닌 값 {int(nonint.sum())}개 — 횟수(0, 1, 2, ...) 필요")
            notes_model.append("종속변수에 정수가 아닌 값 포함 — 포아송 준최우추정(PPML)으로 해석 (강건 표준오차 권장)")
        else:
            y = np.round(y)
        k_tr = o["truncPoint"]
        if kind == "trunc":
            bad = int((y <= k_tr).sum())
            if bad:
                raise ValueError(f"{self.spec['ko']}: 종속변수가 절단점 {k_tr} 이하인 관측 {bad}개 — "
                                 f"{dep} > {k_tr}인 관측만 남긴 뒤 실행 (예: sess.filter_rows('{dep} > {k_tr}'))")
        if kind in ("zi", "hurdle") and not (y == 0).any():
            raise ValueError(f"{self.spec['ko']}: 종속변수에 0이 없음 — 영과잉·허들 모형 추정 불가")
        if kind == "hurdle" and not (y > 0).any():
            raise ValueError(f"{self.spec['ko']}: 양(+)의 횟수가 없음")
        off = np.zeros(len(y))
        if o["exposure"]:
            e = frame[o["exposure"]].to_numpy(dtype=float)
            if (e <= 0).any():
                raise ValueError(f"exposure 변수 '{o['exposure']}'에 0 이하 값 {int((e <= 0).sum())}개 — 양수 필요")
            off = np.log(e)
        elif o["offset"]:
            off = frame[o["offset"]].to_numpy(dtype=float)
        design = _Design(frame, factors, o["refLevels"])
        X, xn, xt = design.build(frame, count_vars, o["constant"])
        X, xn, xt, dropped_x = _drop_collinear(X, xn, xt, keep_last=True)
        if X.shape[1] == 0:
            raise ValueError("가산자료 회귀: 추정할 계수가 없음 (공변량을 지정하거나 상수항 포함)")
        Z, zn, zt, dropped_z = None, [], [], []
        if kind in ("zi", "hurdle"):
            Z, zn, zt = design.build(frame, infl, True)
            Z, zn, zt, dropped_z = _drop_collinear(Z, zn, zt, keep_last=True)
        for nm in dropped_x:
            notes_model.append(f"계수 방정식 '{nm}' 공선성으로 제외 (Stata: omitted because of collinearity)")
        for nm in dropped_z:
            notes_model.append(f"{'팽창' if kind == 'zi' else '영 허들'} 방정식 '{nm}' 공선성으로 제외")
        n = len(y)
        clusters = frame[o["clusterVar"]].to_numpy() if o["clusterVar"] else None
        if clusters is not None and pd.Series(clusters).nunique() < 2:
            raise ValueError("군집 표준오차: 군집이 2개 이상 필요")
        self._frame, self._design, self._xn, self._zn = frame, design, xn, zn
        self._count_vars, self._infl_vars, self._off = count_vars, infl, off

        # ---- Poisson counterpart (LR test of alpha = 0, start values, Vuong)
        pois_fit = None
        if dist != "poisson":
            lik_p, pois_fit = self._fit(y, X, off, "poisson", kind, Z)
        x0 = None
        if pois_fit is not None:
            x0 = np.concatenate([pois_fit["theta"], CountModel(y, X, off, dist=dist, kind=kind, Z=Z, link=o["inflateLink"],
                                 trunc=k_tr).default_start(pois_fit["theta"][: X.shape[1]])[-1:]])
        # default start values, plus the Poisson-counterpart solution for NB families (guards against a local optimum)
        lik, fit = self._fit(y, X, off, dist, kind, Z, extra_starts=[x0] if x0 is not None else ())
        theta = fit["theta"]
        V = vcov(lik, fit, o["covType"], clusters)
        se = np.sqrt(np.clip(np.diag(V), 0, None))
        all_names = xn + [f"inflate:{z_}" for z_ in zn] + (["/lnalpha" if dist == "nb2" else "/lndelta"] if dist != "poisson" else [])
        bad_ix = _unidentified(fit["H"])
        if bad_ix:
            se[bad_ix] = np.nan
            V[bad_ix, :] = np.nan
            V[:, bad_ix] = np.nan
            notes_model.append(f"관측 정보행렬 비정칙: {', '.join(all_names[j] for j in bad_ix)} — 추정치가 발산하는 방향 "
                               "(분리·식별 불가 가능성), 표준오차 계산 불가 (Stata는 매우 큰 표준오차로 표시)")
        self._lik, self._fit_res, self._V = lik, fit, V

        # ---- constant-only comparison model (Stata: count equation constant only; zip/zinb keep the inflate equation)
        const_ix = [j for j, nm in enumerate(xn) if nm == "_cons"]
        X0 = X[:, const_ix]
        Z0 = Z
        if kind == "hurdle":
            Z0 = Z[:, [j for j, nm in enumerate(zn) if nm == "_cons"]]
        # start the comparison model from the full model's inflate / dispersion estimates as well (ZINB likelihoods can
        # have a second, degenerate maximum with the inflation probability near 0 for everybody)
        b_full, g_full, a_full = lik.split(theta)
        sel = (y > k_tr) if kind in ("trunc", "hurdle") else np.ones(len(y), dtype=bool)
        b00 = CountModel(y[sel], X0[sel], off[sel]).default_start() if X0.shape[1] else np.zeros(1)
        g00 = g_full if kind == "zi" else (g_full[[j for j, nm in enumerate(zn) if nm == "_cons"]] if kind == "hurdle" else np.zeros(0))
        st0 = np.concatenate([b00, g00, [a_full] if lik.has_a else []])
        lik0, fit0 = self._fit(y, X0, off, dist, kind, Z0, extra_starts=[st0])
        df_lr = (X.shape[1] - X0.shape[1]) + ((Z.shape[1] - Z0.shape[1]) if kind == "hurdle" else 0)
        ll, ll0 = fit["ll"], fit0["ll"]
        if fam_key == "ztp" and X0.shape[1] == 1:
            # Stata tpoisson stores e(ll_0) evaluated at the untruncated constant-only Poisson estimate (mean count)
            b_naive = np.log(np.sum(y) / np.sum(np.exp(off)))
            ll0_st = lik0.loglike(np.array([b_naive]))
            notes_model.append(f"Stata tpoisson의 e(ll_0)은 절단을 고려하지 않은 상수항 추정치(평균 횟수)에서 계산한 값 "
                               f"({ll0_st:.4f}; 이 값 기준 LR χ² = {2 * (ll - ll0_st):.3f}, Pseudo R2 = {1 - ll / ll0_st:.4f}) — "
                               "이 표는 절단 포아송 상수항 모형의 최우추정값 사용")
        k_par = lik.k
        zc = stats.norm.ppf(0.5 + o["ciLevel"] / 200.0)
        test_ix = [j for j, nm in enumerate(xn) if nm != "_cons"]
        if kind == "hurdle":
            test_ix += [X.shape[1] + j for j, nm in enumerate(zn) if nm != "_cons"]
        if o["covType"] in ("oim", "opg"):
            chi_lbl, chi = "LR χ²", 2.0 * (ll - ll0)
            chi_df = df_lr
        else:
            chi_lbl = "Wald χ²"
            b_t = theta[test_ix]
            Vt = V[np.ix_(test_ix, test_ix)]
            chi = float(b_t @ np.linalg.pinv(Vt) @ b_t) if test_ix else np.nan
            chi_df = int(np.linalg.matrix_rank(Vt)) if test_ix else 0
            if chi_df < len(test_ix):
                notes_model.append(f"Wald 검정: 분산행렬 계수 {chi_df} < 검정 계수 {len(test_ix)} (군집 수 부족 가능)")
        chi_p = float(stats.chi2.sf(chi, chi_df)) if chi_df > 0 and np.isfinite(chi) else np.nan
        r2 = 1.0 - ll / ll0 if ll0 != 0 else np.nan
        aic = -2 * ll + 2 * k_par
        bic = -2 * ll + k_par * np.log(n)

        # ---- tables
        self._summary_table(n, y, ll, ll0, chi_lbl, chi, chi_df, chi_p, r2, aic, bic, k_par, fit, fit0, notes_model, kind, df_lr)
        kx, kz = X.shape[1], (Z.shape[1] if Z is not None else 0)
        ct = self._coef_table("coef", f"계수 방정식 (Count equation) — {dep}", xn, theta[:kx], se[:kx], zc, irr=o["irr"])
        ct.add_note(self._coef_note(kind))
        self.add_table(ct)
        if kz:
            lab = ("팽창 방정식 (Inflate equation, " + o["inflateLink"] + ": P(구조적 0))") if kind == "zi" \
                else ("영 허들 방정식 (Zero hurdle, " + o["inflateLink"] + ": P(y > 0))")
            it = self._coef_table("inflate" if kind == "zi" else "hurdle", lab, zn, theta[kx:kx + kz], se[kx:kx + kz], zc)
            if not infl:
                it.add_note("팽창 방정식: 상수항만 사용 (Stata inflate(_cons))" if kind == "zi" else "영 허들 방정식: 상수항만 사용")
            if kind == "zi":
                it.add_note("양(+)의 계수: 구조적 0(항상 0인 집단)에 속할 확률 증가. Stata zip·zinb의 inflate 방정식과 같은 부호")
            else:
                it.add_note("R pscl::hurdle 규약: 이항 부분은 y > 0일 확률을 모형화 (양의 계수 = 0을 넘을 확률 증가)")
            self.add_table(it)
        alpha_lr = None
        if lik.has_a:
            a_ix = kx + kz
            alpha_lr = self._aux_table(theta[a_ix], se[a_ix], zc, dist, pois_fit, ll, o["covType"])
        self._tests_table(lik, fit, X, Z, off, y, dist, kind, alpha_lr, n, k_par, theta)
        if o["margins"]:
            self._margins_table(theta, V, zc)
        if o["predProbs"]:
            self._predprob_table(lik, theta, y)

        self.model_stats = {"N": int(n), "LL": float(ll), "AIC": float(aic), "BIC": float(bic),
                            "R2_McF": float(r2), "Chi2": (float(chi), chi_p), "df": int(chi_df)}
        self._narrative(chi_lbl, chi, chi_df, chi_p, ll, r2, alpha_lr, theta, se, kx, kz, dist)
        self.syntax_python = python_call(o)
        self.syntax_r = "\n".join(r_code("countReg", o) or [])

    # ---- table builders
    def _summary_table(self, n, y, ll, ll0, chi_lbl, chi, chi_df, chi_p, r2, aic, bic, k_par, fit, fit0, notes, kind, df_lr):
        o = self.options
        t = ResultsTable(name="summary", title=f"모형 정보 (Model summary) — {self.spec['ko']}")
        t.add_column("stat", "항목", col_type="text")
        t.add_column("value", "값", col_type="number")
        t.add_row({"stat": "관측수 (N)", "value": int(n), "_rowfmt": "integer"})
        if kind != "trunc":
            nz = int((y == 0).sum())
            t.add_row({"stat": "0 관측수 (Zero obs)", "value": nz, "_rowfmt": "integer"})
            t.add_row({"stat": "0 비율", "value": 100.0 * nz / n, "_rowfmt": "pc"})
        else:
            t.add_row({"stat": f"절단점 (ll, y > {o['truncPoint']})", "value": int(o["truncPoint"]), "_rowfmt": "integer"})
        if o["covType"] == "cluster":
            t.add_row({"stat": f"군집 수 ({o['clusterVar']})", "value": int(pd.Series(self._frame[o['clusterVar']]).nunique()), "_rowfmt": "integer"})
        t.add_row({"stat": "로그우도 (Log likelihood)", "value": ll})
        t.add_row({"stat": "상수항 모형 로그우도", "value": ll0})
        t.add_row({"stat": f"{chi_lbl}", "value": chi})
        t.add_row({"stat": "자유도", "value": int(chi_df), "_rowfmt": "integer"})
        t.add_row({"stat": f"p (Prob > χ²)", "value": chi_p, "_rowfmt": "pvalue"})
        t.add_row({"stat": "McFadden 유사 R² (Pseudo R2)", "value": r2, "_rowfmt": "zto"})
        t.add_row({"stat": "AIC", "value": aic})
        t.add_row({"stat": "BIC", "value": bic})
        t.add_row({"stat": "추정 모수 수", "value": int(k_par), "_rowfmt": "integer"})
        vce_ko = {"oim": "관측 정보행렬 (OIM, Stata 기본)", "opg": "OPG", "robust": "강건 (robust, N/(N−1) 보정)",
                  "cluster": f"군집 강건 (cluster {o['clusterVar']}, G/(G−1) 보정)"}[o["covType"]]
        t.add_note(f"표준오차: {vce_ko}. 신뢰구간 {o['ciLevel']:g}%, z 검정")
        t.add_note(f"계산: statsmodels {self._lik.lib_name} (최우추정)")
        if o["exposure"]:
            t.add_note(f"노출 변수 exposure({o['exposure']}): offset = ln({o['exposure']}), 계수 1 고정")
        elif o["offset"]:
            t.add_note(f"offset({o['offset']}): 계수 1로 고정")
        if kind == "zi":
            t.add_note(f"{chi_lbl}: 계수 방정식 상수항 모형 대비 (팽창 방정식은 그대로 유지, Stata zip·zinb 방식)")
        elif kind == "hurdle":
            t.add_note(f"{chi_lbl}: 두 방정식 모두 상수항만 둔 모형 대비 (자유도 {df_lr})")
        elif not o["constant"]:
            t.add_note(f"{chi_lbl}: 상수항 제외 모형 — 비교 모형은 모든 계수 0 (offset만 포함)")
        t.add_note("AIC = −2LL + 2k, BIC = −2LL + k·ln(N) (k = 산포 모수 포함 전체 모수 수, Stata estat ic)")
        if kind in ("zi", "hurdle"):
            t.add_note("유사 R²: 위 상수항 모형 대비 McFadden 값 (Stata zip·zinb는 출력하지 않음)")
        if not fit["converged"]:
            t.add_note(f"수렴 기준 미충족 (시작값 {fit['iterations']}개 시도, 최대 기울기 {fit.get('grad_max', np.nan):.2e}"
                       + ("" if fit.get("pd", True) else ", 헤시안 비정칙") + ") — 추정치·표준오차 해석 주의")
        if not fit0["converged"]:
            t.add_note("상수항 모형 수렴 기준 미충족 — LR 검정·유사 R² 해석 주의")
        for nt in notes:
            t.add_note(nt)
        self.add_table(t)

    def _coef_table(self, name, title, names, est, se, zc, irr=False):
        o = self.options
        lvl = f"{o['ciLevel']:g}%"
        t = ResultsTable(name=name, title=title)
        t.add_column("term", "변수", col_type="text")
        t.add_column("b", "계수", col_type="number")
        t.add_column("se", "표준오차", col_type="number")
        t.add_column("z", "z", col_type="number")
        t.add_column("p", "p", col_type="number", fmt="pvalue")
        t.add_column("ci_l", "하한", col_type="number", super_title=f"{lvl} 신뢰구간")
        t.add_column("ci_u", "상한", col_type="number", super_title=f"{lvl} 신뢰구간")
        if irr:
            t.add_column("irr", "IRR", col_type="number", super_title="발생률비 (IRR)")
            t.add_column("irr_se", "표준오차", col_type="number", super_title="발생률비 (IRR)")
            t.add_column("irr_l", "하한", col_type="number", super_title="발생률비 (IRR)")
            t.add_column("irr_u", "상한", col_type="number", super_title="발생률비 (IRR)")
        z, p = _zp(est, se)
        for j, nm in enumerate(names):
            lo, hi = est[j] - zc * se[j], est[j] + zc * se[j]
            row = {"term": nm, "b": float(est[j]), "se": float(se[j]), "z": float(z[j]), "p": float(p[j]),
                   "ci_l": float(lo), "ci_u": float(hi)}
            if irr:
                e = float(np.exp(est[j]))
                row.update({"irr": e, "irr_se": e * float(se[j]), "irr_l": float(np.exp(lo)), "irr_u": float(np.exp(hi))})
            t.add_row(row)
        if irr:
            t.add_note("IRR = exp(계수), 표준오차 = IRR × SE(계수) (델타 방법), 신뢰구간 = exp(계수 신뢰구간), Stata irr 표시 방식. "
                       "_cons의 IRR은 기준 발생률")
        t.mark_coefficients(estimate="b", se="se", p="p", stat="z", ci=("ci_l", "ci_u"),
                            exp_estimate="irr" if irr else None, label="term")
        return t

    def _coef_note(self, kind) -> str:
        bases = [f"{f}={_level_label(self._design.base[f])}" for f in self._design.factors]
        txt = "로그 연결함수: 계수 = ln(기대 횟수)의 변화"
        if kind == "trunc":
            txt += f". 절단 전(모집단) 평균 exp(xb)의 모형, 표본은 y > {self.options['truncPoint']}"
        if kind == "hurdle":
            txt += ". 양(+)의 횟수에 대한 0-절단 분포의 계수"
        if bases:
            txt += f". 요인 기준 범주: {', '.join(bases)}"
        return txt

    def _aux_table(self, a, se_a, zc, dist, pois_fit, ll, vce):
        o = self.options
        nm, ko = ("alpha", "α") if dist == "nb2" else ("delta", "δ")
        lvl = f"{o['ciLevel']:g}%"
        t = ResultsTable(name="aux", title="산포 모수 (Dispersion)")
        t.add_column("term", "모수", col_type="text")
        t.add_column("est", "추정치", col_type="number")
        t.add_column("se", "표준오차", col_type="number")
        t.add_column("ci_l", "하한", col_type="number", super_title=f"{lvl} 신뢰구간")
        t.add_column("ci_u", "상한", col_type="number", super_title=f"{lvl} 신뢰구간")
        t.add_row({"term": f"/ln{nm}", "est": a, "se": se_a, "ci_l": a - zc * se_a, "ci_u": a + zc * se_a})
        e = float(np.exp(a))
        t.add_row({"term": nm, "est": e, "se": e * se_a, "ci_l": float(np.exp(a - zc * se_a)), "ci_u": float(np.exp(a + zc * se_a))})
        if dist == "nb2":
            t.add_note("NB2 (평균 산포): Var(y|x) = μ(1 + αμ). R MASS::glm.nb·pscl의 theta = 1/α")
        else:
            t.add_note("NB1 (상수 산포): Var(y|x) = μ(1 + δ), Stata nbreg, dispersion(constant)")
        t.add_note(f"{nm} = exp(/ln{nm}); 표준오차는 델타 방법, 신뢰구간은 /ln{nm} 신뢰구간의 지수 변환")
        self.add_table(t)
        if pois_fit is None or vce not in ("oim", "opg"):
            return None
        stat = max(0.0, 2.0 * (ll - pois_fit["ll"]))
        p = 0.5 * float(stats.chi2.sf(stat, 1))          # Stata: 0.500 when chibar2 = 0
        return {"stat": stat, "p": p, "name": nm}

    def _tests_table(self, lik, fit, X, Z, off, y, dist, kind, alpha_lr, n, k_par, theta):
        o = self.options
        rows, notes = [], []
        fam_names = {"poisson": "poisson", "nb2": "nbreg", "nb1": "nbreg(constant)"}
        if alpha_lr is not None and kind == "zi" and o["inflateLink"] == "probit":
            notes.append("LR 검정 alpha = 0: probit 팽창 zip 대비 (Stata 19 zinb의 zip 옵션은 probit 지정 시에도 logit 팽창 zip과 비교)")
        if alpha_lr is not None:
            nm = alpha_lr["name"]
            base = {"plain": "포아송", "trunc": "절단 포아송", "zi": "zip", "hurdle": "허들 포아송"}[kind]
            rows.append({"test": f"LR 검정 {nm} = 0 ({base} 대비, chibar2(01))", "stat": alpha_lr["stat"], "df": 1,
                         "p": alpha_lr["p"], "note": "경계 검정: p = 0.5 × P(χ²₁ ≥ 통계량)"})
        elif lik.has_a and o["covType"] not in ("oim", "opg"):
            notes.append("강건·군집 표준오차에서는 산포 모수 LR 검정을 표시하지 않음 (Stata와 동일)")
        if o["vuong"]:
            if kind in ("zi", "hurdle"):
                lik_b, fit_b = self._fit(y, X, off, dist, "plain")
                ll1 = lik.loglike_obs(fit["theta"])
                ll2 = lik_b.loglike_obs(fit_b["theta"])
                other = fam_names[dist]
                this = {"zi": ("zip" if dist == "poisson" else "zinb"), "hurdle": ("허들 포아송" if dist == "poisson" else "허들 음이항")}[kind]
                for r in vuong_test(ll1, ll2, lik.k, lik_b.k):
                    lab = {"raw": "보정 없음", "aic": "AIC 보정", "bic": "BIC 보정"}[r["type"]]
                    rows.append({"test": f"Vuong 검정 {this} 대 {other} ({lab})", "stat": r["z"], "df": None, "p": r["p"],
                                 "note": f"z {'>' if r['favors'] == 1 else '<'} 0: {this if r['favors'] == 1 else other} 쪽"})
                notes.append("Vuong 검정: 보정 없음 z는 Stata zip·zinb forcevuong 옵션(Stata 19에서 vuong 옵션은 오류 처리)과 같은 정의, "
                             "AIC·BIC 보정은 R pscl::vuong 방식. p = P(Z > z) (Stata 방식 단측), z < 0이면 기본 모형 쪽. "
                             "영과잉 여부 검정으로 쓰는 데 대한 비판(Wilson, 2015) 참고")
            else:
                notes.append("Vuong 검정은 zip·zinb·허들 모형에서만 계산")
        if o["gof"] and dist == "poisson" and kind == "plain":
            glm = sm.GLM(y, X, family=sm.families.Poisson(), offset=off).fit()      # statsmodels GLM: deviance, Pearson
            dev, pear, dfg = float(glm.deviance), float(glm.pearson_chi2), int(round(glm.df_resid))
            rows.append({"test": "적합도: 이탈도 (Deviance)", "stat": dev, "df": dfg, "p": float(stats.chi2.sf(dev, dfg)),
                         "note": "Stata estat gof"})
            rows.append({"test": "적합도: Pearson χ²", "stat": pear, "df": dfg, "p": float(stats.chi2.sf(pear, dfg)),
                         "note": f"Pearson χ²/df = {pear / dfg:.3f}"})
            notes.append("적합도 검정의 p < .05: 포아송 가정(평균 = 분산) 기각 — 음이항·영과잉 모형 검토")
        if not rows:
            return
        t = ResultsTable(name="tests", title="모형 검정 (Tests)")
        t.add_column("test", "검정", col_type="text")
        t.add_column("stat", "통계량", col_type="number")
        t.add_column("df", "자유도", col_type="integer")
        t.add_column("p", "p", col_type="number", fmt="pvalue")
        t.add_column("note", "비고", col_type="text")
        for r in rows:
            t.add_row(r)
        for nt in notes:
            t.add_note(nt)
        self.add_table(t)
        self._tests_rows = rows

    # ---- margins (Stata margins, dydx(*): default prediction = number of events)
    def _xz(self, frame):
        X, xn, _ = self._design.build(frame, self._count_vars, self.options["constant"])
        X = X[:, [xn.index(c) for c in self._xn]]
        Z = None
        if self._zn:
            Z, zn, _ = self._design.build(frame, self._infl_vars, True)
            Z = Z[:, [zn.index(c) for c in self._zn]]
        return X, Z

    def _margins_table(self, theta, V, zc):
        lik, frame = self._lik, self._frame
        items = []           # (label, kind, payload)
        seen = []
        for v in self._count_vars + [w for w in self._infl_vars if w not in self._count_vars]:
            if v in seen:
                continue
            seen.append(v)
            if v in self._design.levels:
                base = self._design.base[v]
                f0 = frame.copy()
                f0[v] = base
                X0, Z0 = self._xz(f0)
                for lev in self._design.levels[v]:
                    if lev == base:
                        continue
                    f1 = frame.copy()
                    f1[v] = lev
                    X1, Z1 = self._xz(f1)
                    items.append((f"{v}={_level_label(lev)}", "factor", (X0, Z0, X1, Z1)))
            else:
                bx = self._xn.index(v) if v in self._xn else None
                gz = self._zn.index(v) if v in self._zn else None
                if bx is None and gz is None:
                    continue
                h = 1e-5 * max(1.0, float(np.std(frame[v].to_numpy(dtype=float))))
                items.append((v, "cont", (bx, gz, h)))
        if not items:
            return

        Xs, Zs = lik.X, (lik.Z if lik.kz else None)

        def ame(t):
            """Average of dE[y]/dx (central difference of the statsmodels prediction) or of the discrete change."""
            out = []
            for _, kind, pl in items:
                if kind == "cont":
                    bx, gz, h = pl
                    Xp, Xm, Zp, Zm = Xs, Xs, Zs, Zs
                    if bx is not None:
                        Xp, Xm = Xs.copy(), Xs.copy()
                        Xp[:, bx] += h
                        Xm[:, bx] -= h
                    if gz is not None:
                        Zp, Zm = Zs.copy(), Zs.copy()
                        Zp[:, gz] += h
                        Zm[:, gz] -= h
                    out.append(float(np.mean((lik.mean(t, Xp, Zp) - lik.mean(t, Xm, Zm)) / (2.0 * h))))
                else:
                    X0, Z0, X1, Z1 = pl
                    out.append(float(np.mean(lik.mean(t, X1, Z1)) - np.mean(lik.mean(t, X0, Z0))))
            return np.array(out)

        est = ame(theta)
        J = np.zeros((len(est), len(theta)))
        for j in range(len(theta)):
            h = 1e-6 * max(1.0, abs(theta[j]))
            tp, tm = theta.copy(), theta.copy()
            tp[j] += h
            tm[j] -= h
            J[:, j] = (ame(tp) - ame(tm)) / (2 * h)
        Vm = J @ V @ J.T
        se = np.sqrt(np.clip(np.diag(Vm), 0, None))
        z, p = _zp(est, se)
        lvl = f"{self.options['ciLevel']:g}%"
        t = ResultsTable(name="margins", title="평균 한계효과 (AME, margins, dydx(*))")
        t.add_column("term", "변수", col_type="text")
        t.add_column("dydx", "dy/dx", col_type="number")
        t.add_column("se", "표준오차", col_type="number")
        t.add_column("z", "z", col_type="number")
        t.add_column("p", "p", col_type="number", fmt="pvalue")
        t.add_column("ci_l", "하한", col_type="number", super_title=f"{lvl} 신뢰구간")
        t.add_column("ci_u", "상한", col_type="number", super_title=f"{lvl} 신뢰구간")
        for (lab, _, _), e, s, zz, pp in zip(items, est, se, z, p):
            t.add_row({"term": lab, "dydx": float(e), "se": float(s), "z": float(zz), "p": float(pp),
                       "ci_l": float(e - zc * s), "ci_u": float(e + zc * s)})
        kind = self.spec["kind"]
        pred = {"plain": "기대 횟수 exp(xb)", "trunc": "절단 전 기대 횟수 exp(xb) (Stata predict n)",
                "zi": "기대 횟수 (1 − π)·exp(xb)", "hurdle": "기대 횟수 P(y > 0)·E(y | y > 0)"}[kind]
        t.add_note(f"예측값: {pred}" + (", exposure·offset 포함" if (self.options['exposure'] or self.options['offset']) else ""))
        t.add_note("연속변수: 관측별 도함수의 평균(수치 미분), 요인: 기준 범주 대비 이산 변화의 평균 (Stata margins, dydx(*)). "
                   "표준오차: 델타 방법")
        t.mark_coefficients(estimate="dydx", se="se", p="p", stat="z", ci=("ci_l", "ci_u"), label="term")
        self.add_table(t)

    # ---- observed vs predicted count distribution (countfit / prcounts)
    def _predprob_table(self, lik, theta, y):
        o = self.options
        lo = o["truncPoint"] + 1 if self.spec["kind"] == "trunc" else 0
        hi = max(o["maxCount"], lo + 5)
        counts = list(range(lo, hi + 1))
        P = lik.prob_matrix(theta, counts)
        pred = P.mean(axis=1)
        n = len(y)
        t = ResultsTable(name="predprob", title="관측·예측 빈도 비교 (Observed vs predicted counts)")
        t.add_column("count", "횟수", col_type="integer")
        t.add_column("obs_n", "관측 빈도", col_type="integer")
        t.add_column("obs", "관측 비율", col_type="number", fmt="zto")
        t.add_column("pred", "예측 비율", col_type="number", fmt="zto")
        t.add_column("diff", "차이 (관측 − 예측)", col_type="number", fmt="zto")
        t.add_column("pearson", "Pearson 기여분", col_type="number")
        obs_all, pear_sum = [], 0.0
        for j, pj in zip(counts, pred):
            nj = int((y == j).sum())
            ob = nj / n
            pe = n * (ob - pj) ** 2 / pj if pj > 0 else np.nan
            pear_sum += pe if np.isfinite(pe) else 0.0
            obs_all.append(ob)
            t.add_row({"count": j, "obs_n": nj, "obs": ob, "pred": float(pj), "diff": ob - float(pj), "pearson": pe})
        t.add_note(f"예측 비율: 관측별 예측확률 P(y = j | x)의 평균 (Stata countfit·prcounts 방식). "
                   f"표시 범위 예측 비율 합 = {pred.sum():.3f}, Pearson 기여분 합 = {pear_sum:.2f}")
        if self.spec["kind"] == "trunc":
            t.add_note(f"절단 모형: 예측확률은 P(y = j | y > {o['truncPoint']})")
        self.add_table(t)
        if o["predPlot"]:
            self.add_plot("관측·예측 빈도 비교", _plot_counts(counts, obs_all, list(pred), self.spec["ko"]))

    def _narrative(self, chi_lbl, chi, chi_df, chi_p, ll, r2, alpha_lr, theta, se, kx, kz, dist):
        fam = self.spec["ko"]
        sig = "유의함" if (np.isfinite(chi_p) and chi_p < 0.05) else "유의하지 않음"
        txt = (f"{fam} 결과 모형 전체 검정 {sig} ({chi_lbl}({chi_df}) = {chi:.2f}, p {_pstr(chi_p) if np.isfinite(chi_p) else '= —'}; "
               f"로그우도 = {ll:.2f}, McFadden 유사 R² = {r2:.3f})")
        if self._lik.has_a:
            a = float(np.exp(theta[kx + kz]))
            nm = "α" if dist == "nb2" else "δ"
            txt += f"; 산포 모수 {nm} = {a:.3f}"
            if alpha_lr is not None:
                txt += f" (LR 검정 χ̄²(01) = {alpha_lr['stat']:.2f}, p {_pstr(alpha_lr['p'])})"
        self.apa_narrative = txt + "."


def _plot_counts(counts, obs, pred, fam_label) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plots import _col, _clean_ax, _hdr_col, fig_to_base64
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(counts))
    ax.bar(x, obs, width=0.62, color=_col(0), alpha=0.55, label="관측 비율")
    ax.plot(x, pred, color=_col(2), marker="o", linewidth=2.0, label=f"예측 비율 ({fam_label})")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in counts])
    ax.set_xlabel("횟수")
    ax.set_ylabel("비율")
    ax.set_title("관측·예측 빈도 비교", color=_hdr_col())
    ax.legend(frameon=False)
    _clean_ax(ax)
    return fig_to_base64(fig)


# --------------------------------------------------------------------------------------------- code generation
def _non_default(o: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, dv in _DEFAULTS.items():
        v = o.get(k, dv)
        if k == "family" or v in (None, "", [], {}) and dv in (None, "", [], {}):
            continue
        if v != dv:
            out[k] = v
    return out


def python_call(o: Dict[str, Any]) -> str:
    kw = [f"dep={o['dep']!r}"]
    if o.get("covs"):
        kw.append(f"covs={list(o['covs'])!r}")
    if o.get("factors"):
        kw.append(f"factors={list(o['factors'])!r}")
    kw.append(f"family={o.get('family', 'poisson')!r}")
    for k, v in _non_default(o).items():
        kw.append(f"{k}={v!r}")
    return f"res = sess.run('countReg', {', '.join(kw)})"


def _resolve_t(t: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Canonical options for the code export, or None when t is not a count-model name."""
    low = str(t).strip().lower()
    if low not in ALIAS_PRESETS and low != "countreg" and t != "countReg":
        return None
    try:
        return normalize_options(params or {}, preset=ALIAS_PRESETS.get(low))
    except ValueError:
        return None


def _stata_term(v: str, o: Dict[str, Any], factors: Sequence[str]) -> str:
    if v in factors:
        ref = o.get("refLevels", {}).get(v)
        if ref is not None and re.fullmatch(r"-?\d+", str(ref)):
            return f"ib{ref}.{v}"
        return f"i.{v}"
    return v


def stata_code(t: str, params: Dict[str, Any]) -> Optional[List[str]]:
    o = _resolve_t(t, params)
    if o is None:
        return None
    fam = o["family"]
    spec = FAMILIES[fam]
    factors = o["factors"]
    infl_f = [v[2:] for v in o["inflate"] if v.startswith("i.")]
    xs = " ".join(_stata_term(v, o, factors) for v in dict.fromkeys(o["covs"] + factors))
    zs = " ".join(_stata_term(v[2:] if v.startswith("i.") else v, o, factors + infl_f) for v in o["inflate"]) or "_cons"
    out: List[str] = []
    for f in factors + infl_f:
        ref = o.get("refLevels", {}).get(f)
        if ref is not None and not re.fullmatch(r"-?\d+", str(ref)):
            out.append(f"* 요인 {f}의 기준 범주 '{ref}': 문자형이면 encode {f}, generate({f}_n) 후 fvset base <코드> {f}_n")
    opts = []
    if o["exposure"]:
        opts.append(f"exposure({o['exposure']})")
    if o["offset"]:
        opts.append(f"offset({o['offset']})")
    if not o["constant"]:
        opts.append("noconstant")
    if o["irr"]:
        opts.append("irr")
    vce = {"oim": "", "opg": "vce(opg)", "robust": "vce(robust)", "cluster": f"vce(cluster {o['clusterVar']})"}[o["covType"]]
    if vce:
        opts.append(vce)
    if abs(o["ciLevel"] - 95.0) > 1e-9:
        opts.append(f"level({o['ciLevel']:g})")
    dep = o["dep"]
    kind = spec["kind"]
    if kind == "plain":
        cmd = "poisson" if fam == "poisson" else "nbreg"
        if fam == "nb1":
            opts.append("dispersion(constant)")
        out.append(f"{cmd} {dep} {xs}".rstrip() + (", " + " ".join(opts) if opts else ""))
        if fam == "poisson" and o["gof"]:
            out.append("estat gof")
    elif kind == "zi":
        zo = [f"inflate({zs})"] + (["probit"] if o["inflateLink"] == "probit" else [])
        robust_like = o["covType"] in ("robust", "cluster")
        if fam == "zinb" and not robust_like and o["inflateLink"] == "logit":
            zo.append("zip")                                       # LR test of alpha = 0 against zip
        vu = ["forcevuong"] if o["vuong"] and not robust_like else []
        out.append(f"{fam} {dep} {xs}".rstrip() + ", " + " ".join(zo + vu + opts))
        if o["vuong"] and robust_like:
            out += ["* Vuong 검정: forcevuong은 vce(robust)·vce(cluster)와 함께 쓸 수 없어 기본 vce로 따로 추정",
                    f"quietly {fam} {dep} {xs}".rstrip() + ", " + " ".join(zo + ["forcevuong"] + [x for x in opts if not x.startswith("vce(")]),
                    "display \"Vuong z = \" e(vuong) \"  Pr>z = \" normal(-e(vuong))"]
        if fam == "zinb" and o["inflateLink"] == "probit":
            out.append("* LR 검정 alpha = 0: zinb의 zip 옵션은 probit 지정 시에도 logit zip과 비교 — probit zip을 따로 추정해 lrtest 사용")
    elif kind == "trunc":
        cmd = "tpoisson" if fam == "ztp" else "tnbreg"
        to = [f"ll({o['truncPoint']})"] + (["dispersion(constant)"] if fam == "ztnb1" else [])
        out.append(f"{cmd} {dep} {xs}".rstrip() + ", " + " ".join(to + opts))
    else:
        cmd = "tpoisson" if fam == "hurdle_poisson" else "tnbreg"
        bin_cmd = "logit" if o["inflateLink"] == "logit" else "probit"
        bo = [x for x in opts if x.startswith("vce(") or x.startswith("level(")]
        out += ["* 허들 모형: Stata 내장 명령 없음 — 이항 부분과 0-절단 부분의 우도가 분리되므로 두 모형을 따로 추정 (추정치 동일)",
                f"generate byte _pos_{dep} = ({dep} > 0) if !missing({dep})",
                f"{bin_cmd} _pos_{dep} {'' if zs == '_cons' else zs}".rstrip() + (", " + " ".join(bo) if bo else ""),
                f"{cmd} {dep} {xs} if {dep} > 0".replace("  ", " ") + ", " + " ".join(["ll(0)"] + opts),
                "* 전체 로그우도 = 두 모형 로그우도의 합"]
    if o["margins"] and kind == "hurdle":
        out.append("* 평균 한계효과(E(y) = P(y > 0)·E(y | y > 0) 기준): 두 부분을 따로 추정하면 margins로 직접 계산 불가 — OpenStat 결과표 참고")
    elif o["margins"]:
        out.append("margins, dydx(*)")
    if o["predProbs"]:
        out.append(f"* 관측·예측 빈도 비교: countfit 또는 prcounts (SSC spost13_ado), 최대 {o['maxCount']}회")
    return out


def _r_formula(o: Dict[str, Any], with_offset: bool = True) -> str:
    factors = o["factors"]
    infl_f = [v[2:] for v in o["inflate"] if v.startswith("i.")]

    def term(v, facs):
        if v in facs:
            ref = o.get("refLevels", {}).get(v)
            return f"relevel(factor({v}), ref = \"{ref}\")" if ref is not None else f"factor({v})"
        return v
    rhs = [term(v, factors) for v in dict.fromkeys(o["covs"] + factors)]
    if with_offset and o["exposure"]:
        rhs.append(f"offset(log({o['exposure']}))")
    elif with_offset and o["offset"]:
        rhs.append(f"offset({o['offset']})")
    if not o["constant"]:
        rhs.append("0")
    f = f"{o['dep']} ~ {' + '.join(rhs) if rhs else '1'}"
    if FAMILIES[o["family"]]["kind"] in ("zi", "hurdle"):
        zr = [term(v[2:] if v.startswith("i.") else v, factors + infl_f) for v in o["inflate"]]
        f += f" | {' + '.join(zr) if zr else '1'}"
    return f


def r_code(t: str, params: Dict[str, Any]) -> Optional[List[str]]:
    o = _resolve_t(t, params)
    if o is None:
        return None
    fam = o["family"]
    kind = FAMILIES[fam]["kind"]
    f = _r_formula(o)
    link = o["inflateLink"]
    out: List[str] = []
    if fam == "poisson":
        out.append(f"m <- glm({f}, family = poisson, data = data)")
    elif fam == "nb2":
        out += [f"m <- MASS::glm.nb({f}, data = data)",
                "# Stata alpha = 1 / m$theta; glm.nb의 계수 표준오차는 theta를 고정한 기대 정보행렬 기준(Stata OIM과 소수 셋째 자리 이하 차이 가능)"]
    elif fam == "nb1":
        out += ["# NB1 (Var = mu(1 + delta)): R 기본 패키지에 대응 함수 없음 — gamlss의 NBII 분포가 같은 모형(sigma = delta)",
                f"m <- gamlss::gamlss({f}, family = gamlss.dist::NBII, data = data)"]
    elif kind == "zi":
        out.append(f"m <- pscl::zeroinfl({f}, dist = \"{'poisson' if fam == 'zip' else 'negbin'}\", link = \"{link}\", data = data)")
        if fam == "zinb":
            out.append("# Stata alpha = 1 / m$theta")
    elif kind == "trunc":
        if o["truncPoint"] == 0 and fam != "ztnb1":
            vfam = "VGAM::pospoisson()" if fam == "ztp" else "VGAM::posnegbinomial()"
            out += [f"m <- VGAM::vglm({f}, family = {vfam}, data = data)",
                    "# VGAM 표준오차는 기대 정보행렬 기준; posnegbinomial의 size = 1/alpha"]
        else:
            out.append(f"# 절단점 {o['truncPoint']}·{fam}: R 표준 패키지에 대응 함수 없음 — countreg::zerotrunc(0-절단) 또는 우도 직접 작성")
    else:
        out.append(f"m <- pscl::hurdle({f}, dist = \"{'poisson' if fam == 'hurdle_poisson' else 'negbin'}\", "
                   f"zero.dist = \"binomial\", link = \"{link}\", data = data)")
    if not out[-1].startswith("# 절단점"):
        if o["covType"] == "oim":
            out.append("summary(m)")
        elif o["covType"] == "robust":
            out += ["n <- nobs(m)", "lmtest::coeftest(m, vcov = sandwich::sandwich(m) * n / (n - 1))  # Stata vce(robust)"]
        elif o["covType"] == "cluster":
            out.append(f"lmtest::coeftest(m, vcov = sandwich::vcovCL(m, cluster = ~ {o['clusterVar']}, type = \"HC0\", cadjust = TRUE))  # Stata vce(cluster)")
        else:
            out.append("lmtest::coeftest(m, vcov = solve(crossprod(sandwich::estfun(m))))  # OPG")
        if o["irr"]:
            out.append("exp(coef(m))  # IRR")
        if fam == "poisson" and o["gof"]:
            out.append("c(deviance = deviance(m), pearson = sum(residuals(m, type = \"pearson\")^2), df = df.residual(m))")
        if o["vuong"] and kind in ("zi", "hurdle"):
            base = "glm(" + _r_formula(dict(o, family="poisson")) + ", family = poisson, data = data)" if FAMILIES[fam]["dist"] == "poisson" \
                else "MASS::glm.nb(" + _r_formula(dict(o, family="nb2")) + ", data = data)"
            out.append(f"pscl::vuong(m, {base})")
        if o["margins"]:
            out.append("marginaleffects::avg_slopes(m)  # 평균 한계효과 (Stata margins, dydx(*))")
    return out


# --------------------------------------------------------------------------------------------- Stata command input
_CMD_RE = re.compile(r"^(poisson|nbreg|gnbreg|zip|zinb|tpoisson|tnbreg|ztp|ztnb|zipcv|zinbcv)\b(.*)$", re.IGNORECASE)
_HARMLESS = ("nolog", "log", "difficult", "iterate", "tolerance", "ltolerance", "nrtolerance", "nonrtolerance", "technique",
             "trace", "gradient", "showstep", "hessian", "showtolerance", "noheader", "nocnsreport", "nodisplay", "coeflegend",
             "noomitted", "vsquish", "noemptycells", "baselevels", "allbaselevels", "nofvlabel", "fvwrap", "fvwrapon",
             "cformat", "pformat", "sformat", "nolstretch")
# (full name, minimum abbreviation)
_OPTS = [("exposure", "e"), ("offset", "off"), ("irr", "irr"), ("vce", "vce"), ("robust", "r"), ("cluster", "cl"),
         ("noconstant", "nocon"), ("level", "l"), ("dispersion", "d"), ("inflate", "inf"), ("probit", "probit"),
         ("vuong", "vuong"), ("forcevuong", "forcevuong"), ("zip", "zip"), ("ll", "ll"), ("ul", "ul"), ("constraints", "c"), ("collinear", "coll"),
         ("from", "from"), ("nolrtest", "nolrt")] + [(h, h) for h in _HARMLESS]


def _canon(name: str) -> Optional[str]:
    low = name.lower()
    for full, _ in _OPTS:
        if low == full:
            return full
    for full, ab in _OPTS:
        if len(low) >= len(ab) and full.startswith(low) and low.startswith(ab):
            return full
    return None


def _parse_opts(txt: str) -> List[Tuple[str, str, str]]:
    """'inflate(a b) irr vce(cluster id)' -> [(raw name, canonical or None, value)]."""
    out, i, n = [], 0, len(txt)
    while i < n:
        while i < n and txt[i] in " \t":
            i += 1
        j = i
        while j < n and txt[j] not in " \t(":
            j += 1
        name = txt[i:j]
        if not name:
            break
        val = ""
        if j < n and txt[j] == "(":
            depth, k = 0, j
            while k < n:
                depth += txt[k] == "("
                depth -= txt[k] == ")"
                if depth == 0:
                    break
                k += 1
            val = txt[j + 1:k].strip()
            j = k + 1
        out.append((name, _canon(name), val))
        i = j
    return out


def _split_comma(s: str) -> Tuple[str, str]:
    depth = 0
    for i, ch in enumerate(s):
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            return s[:i], s[i + 1:]
    return s, ""


def _parse_varlist(tokens: Sequence[str], raw: str):
    """Stata varlist with i./ib#./c. prefixes -> (variables, factor variables, reference levels) or an error text."""
    vars_, facs, refs = [], [], {}
    for tok in tokens:
        if re.search(r"#|[*?~]|-", tok):
            return None, None, None, f"상호작용(#)·변수 범위(-)·와일드카드 미지원 — 변수를 직접 나열: {raw}"
        m = re.fullmatch(r"(?:(i|c)\.|ib(\d+)\.|ib\((\d+|first)\)\.)?([A-Za-z_À-￿][\wÀ-￿]*)", tok)
        if not m:
            return None, None, None, f"변수 표기 미지원 '{tok}': {raw}"
        pre, b1, b2, v = m.group(1), m.group(2), m.group(3), m.group(4)
        vars_.append(v)
        if pre == "i" or b1 or b2:
            facs.append(v)
            ref = b1 or (b2 if b2 and b2 != "first" else None)
            if ref:
                refs[v] = ref
    return vars_, facs, refs, None


def transpile(line: str) -> Optional[str]:
    s = line.strip()
    m = _CMD_RE.match(s)
    if not m:
        return None
    cmd = m.group(1).lower()

    def stop(msg: str) -> str:
        return f"raise NotImplementedError({(cmd + ': ' + msg)!r})"

    if cmd in ("gnbreg", "zipcv", "zinbcv"):
        return stop(f"미지원 명령 (가능: poisson, nbreg, zip, zinb, tpoisson, tnbreg) — {s}")
    body, opt_txt = _split_comma(m.group(2))
    if re.search(r"\[|\]", body):
        return stop(f"가중치([fweight=…] 등) 미지원 — {s}")
    if re.search(r"\b(if|in)\b", body):
        return stop(f"if·in 조건 미지원 — keep if 등으로 자료를 먼저 선택한 뒤 실행: {s}")
    toks = body.split()
    if not toks:
        return stop(f"종속변수 필요 — {s}")
    vars_, facs, refs, err = _parse_varlist(toks[1:], s)
    if err:
        return stop(err)
    dep = toks[0]
    if not re.fullmatch(r"[A-Za-z_À-￿][\wÀ-￿]*", dep):
        return stop(f"종속변수 표기 미지원 '{dep}'")
    family = {"poisson": "poisson", "nbreg": "nb2", "zip": "zip", "zinb": "zinb", "tpoisson": "ztp", "tnbreg": "ztnb",
              "ztp": "ztp", "ztnb": "ztnb"}[cmd]
    kw: Dict[str, Any] = {}
    comments: List[str] = []
    bad: List[str] = []
    seen_infl = False
    notes_zip = False
    for raw, name, val in _parse_opts(opt_txt):
        if name is None:
            bad.append(raw)
        elif name in _HARMLESS:
            comments.append(raw + (f"({val})" if val else ""))
        elif name == "exposure":
            kw["exposure"] = val
        elif name == "offset":
            kw["offset"] = val
        elif name == "irr":
            kw["irr"] = True
        elif name == "noconstant":
            kw["constant"] = False
        elif name == "robust":
            kw["covType"] = "robust"
        elif name == "cluster":
            kw["covType"], kw["clusterVar"] = "cluster", val
        elif name == "vce":
            vt = val.split()
            v0 = vt[0].lower() if vt else ""
            if v0 in ("oim",):
                kw["covType"] = "oim"
            elif v0 == "opg":
                kw["covType"] = "opg"
            elif "robust".startswith(v0) and v0:
                kw["covType"] = "robust"
            elif "cluster".startswith(v0) and len(v0) >= 2 and len(vt) == 2:
                kw["covType"], kw["clusterVar"] = "cluster", vt[1]
            else:
                bad.append(f"vce({val})")
        elif name == "level":
            try:
                kw["ciLevel"] = float(val)
            except ValueError:
                bad.append(f"level({val})")
        elif name == "dispersion":
            dv = val.lower()
            if dv and "constant".startswith(dv):
                if family == "nb2":
                    family = "nb1"
                elif family == "ztnb":
                    family = "ztnb1"
                else:
                    bad.append(f"dispersion({val})")
            elif not (dv and "mean".startswith(dv)) or family not in ("nb2", "ztnb"):
                bad.append(f"dispersion({val})")
        elif name == "inflate" and family in ("zip", "zinb"):
            ib, isub = _split_comma(val)
            if isub.strip():
                bad.append(f"inflate(…, {isub.strip()})")
                continue
            seen_infl = True
            itoks = [t for t in ib.split() if t != "_cons"]
            iv, ifac, irefs, err = _parse_varlist(itoks, s)
            if err:
                return stop(err)
            kw["inflate"] = [("i." + v if v in ifac and v not in facs else v) for v in iv]
            for v, r in irefs.items():
                refs.setdefault(v, r)
        elif name == "probit" and family in ("zip", "zinb"):
            kw["inflateLink"] = "probit"
        elif name in ("vuong", "forcevuong") and family in ("zip", "zinb"):
            kw["vuong"] = True
        elif name == "zip" and family == "zinb":
            notes_zip = True
        elif name == "ll" and family in ("ztp", "ztnb", "ztnb1"):
            if not re.fullmatch(r"\d+", val):
                bad.append(f"ll({val}) — 0 이상의 정수만 가능(변수 절단점 미지원)")
            else:
                kw["truncPoint"] = int(val)
        else:
            bad.append(raw + (f"({val})" if val else ""))
    if family in ("zip", "zinb") and not seen_infl:
        return stop(f"inflate() 옵션 필요 (상수항만이면 inflate(_cons)) — {s}")
    if bad:
        return stop(f"옵션 미지원: {', '.join(bad)} — {s}")
    covs = [v for v in vars_ if v not in facs]
    parts = [f"dep={dep!r}"]
    if covs:
        parts.append(f"covs={covs!r}")
    if facs:
        parts.append(f"factors={facs!r}")
    parts.append(f"family={family!r}")
    if refs:
        parts.append(f"refLevels={refs!r}")
    for k, v in kw.items():
        parts.append(f"{k}={v!r}")
    code = f"res = sess.run('countReg', {', '.join(parts)})"
    if comments:
        code += f"\n# [Stata] {', '.join(comments)}: 반복·출력 제어 옵션 — 추정 결과와 무관하여 생략"
    if notes_zip:
        code += "\n# [Stata] zinb의 zip 옵션: zinb 대 zip LR 검정(alpha = 0)은 기본 출력에 포함"
    return code


# --------------------------------------------------------------------------------------------- plug-in
def dispatch(t: str, df: pd.DataFrame, params: Dict[str, Any]):
    if t != "countReg":
        return None
    p = dict(params or {})
    preset = _family_from_alias(p)
    for k in ("_alias", "_analysis"):
        p.pop(k, None)
    o = normalize_options(p, preset=preset)
    return CountRegression(df, **o)


def _stata_entry(t: str, params: Dict[str, Any]):
    return stata_code(t, params)


def _r_entry(t: str, params: Dict[str, Any]):
    return r_code(t, params)


EXTENSION = {
    "catalog": [{"id": "countReg", "group": "Regression",
                 "title": "Count Regression (Poisson, NB, ZIP, ZINB, Truncated, Hurdle)",
                 "module": "openstat", "icon": "analysis-regression.svg"}],
    "aliases": {"poisson": "countReg", "nbreg": "countReg", "zip": "countReg", "zinb": "countReg",
                "tpoisson": "countReg", "tnbreg": "countReg", "ztp": "countReg", "ztnb": "countReg",
                "hurdle": "countReg", "count_regression": "countReg", "countreg": "countReg"},
    "dispatch": dispatch,
    "uses_alias": True,          # registry passes the typed name as params["_alias"] (alias -> family preset)
    "stata": _stata_entry,
    "r": _r_entry,
    "transpile": transpile,
}
