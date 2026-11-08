"""
OpenStat Studio — 제한종속변수 모형 (limited dependent variable models; features and options follow Stata)
License: MIT

Analysis ids, the Stata command they follow, and what computes them:
- binaryReg   : logit / probit / cloglog — statsmodels Logit, Probit, GLM(Binomial, CLogLog) by Newton with the
                observed Hessian; margins dydx(*) / atmeans via statsmodels get_margeff (continuous regressors);
                estat classification, estat gof group(), lroc (sklearn AUC) as glue
- ordinalReg  : ologit / oprobit — statsmodels OrderedModel, thresholds converted to Stata cut1..cutK-1 (delta
                method); predicted probabilities and AMEs per outcome as glue; Brant test of parallel lines:
                K-1 statsmodels Logit fits + Brant (1990) covariance (own glue, no library found)
- multinomReg : mlogit — statsmodels MNLogit (base outcome by recoding), get_margeff for AMEs
- tobitReg    : tobit ll()/ul() — own log likelihood as a statsmodels GenericLikelihoodModel (no maintained
                library: py4etrics fails with statsmodels 0.15); t statistics with Stata's e(df_r)
- truncReg    : truncreg ll()/ul() — own log likelihood, statsmodels GenericLikelihoodModel
- heckman     : heckman ML — own log likelihood, statsmodels GenericLikelihoodModel; twostep — statsmodels
                Probit + OLS with the Heckman/Greene corrected covariance (glue), rho truncation as Stata rhosigma
- heckprobit  : heckprobit — own log likelihood with a vectorised Genz bivariate normal CDF, GenericLikelihoodModel

Standard errors: observed information (Stata vce(oim)); vce(robust) = statsmodels HC0 sandwich x N/(N-1);
vce(cluster) = statsmodels cluster sandwich (no correction) x G/(G-1) — Stata's factors for ML estimators.
Marginal effects follow Stata margins: sample averages (or atmeans), discrete change from the base level for
factor dummies with the other levels of that factor at 0, delta-method SEs.
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import special, stats

from .base import BaseAnalysis, ResultsTable, resolve_choice

LOG_2PI = math.log(2.0 * math.pi)

# =============================================================================================== option helpers
def _as_list(val: Any) -> List[str]:
    if val is None or val == "":
        return []
    if isinstance(val, str):
        return [v.strip() for v in val.split(",") if v.strip()]
    if isinstance(val, (list, tuple)):
        return [str(v) for v in val if v is not None and str(v) != ""]
    return [str(val)]


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


def _check_columns(df: pd.DataFrame, label: str, required: Dict[str, Any], optional: Dict[str, Any]) -> None:
    absent = [role for role, col in required.items() if not col]
    if absent:
        raise ValueError(f"{label}: 필수 변수 지정 필요 — {', '.join(absent)}")
    missing = []
    for role, col in list(required.items()) + list(optional.items()):
        for c in (col if isinstance(col, (list, tuple)) else [col]):
            if c and c not in df.columns:
                missing.append(f"{role}='{c}'")
    if missing:
        raise ValueError(f"{label}: 자료에 없는 변수 — {', '.join(missing)} "
                         f"(사용 가능한 변수: {', '.join(map(str, df.columns))})")


COV_TYPES = ("oim", "robust", "cluster")
_COV_ALIASES = {"conventional": "oim", "classical": "oim", "standard": "oim", "nonrobust": "oim", "default": "oim",
                "observed": "oim", "hc0": "robust", "hc1": "robust", "huber": "robust", "sandwich": "robust",
                "white": "robust", "clustered": "cluster", "cl": "cluster"}


def _cov_type(p: Dict[str, Any]) -> str:
    return resolve_choice("표준오차 유형(covType)", _opt(p, "covType", "cov_type", None) or _opt(p, "vce", "vce", None),
                          COV_TYPES, _COV_ALIASES, default="oim")


def _cluster_var(p: Dict[str, Any]) -> Optional[str]:
    v = _as_single(_opt(p, "clusterVar", "cluster_var", None) or _opt(p, "cluster", "cluster", None))
    return str(v) if v else None


def _level(p: Dict[str, Any]) -> float:
    v = float(_opt(p, "ciWidth", "ci_width", 95))
    if not (10.0 <= v < 100.0):
        raise ValueError(f"신뢰수준(ciWidth): 10 이상 100 미만 필요 (입력: {v})")
    return v


def _lvl_str(v: Any) -> str:
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (float, np.floating)) and np.isfinite(v) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _sorted_levels(s: pd.Series) -> list:
    s = s.dropna()
    if isinstance(s.dtype, pd.CategoricalDtype):
        return [c for c in s.cat.categories if (s == c).any()]
    vals = list(pd.unique(s))
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return sorted(vals, key=float)
    return sorted(vals, key=lambda x: str(x))


def _pstr(p: float) -> str:
    return "< .001" if p < 0.001 else f"= {p:.3f}"


# =============================================================================================== design matrices
class _Design:
    """Covariates (numeric) and factors (dummy coded, lowest level = base, Stata i.var) -> design matrix."""

    def __init__(self, data: pd.DataFrame, covs: Sequence[str], factors: Sequence[str], notes: List[str],
                 const: bool = True, levels_from: Optional[pd.DataFrame] = None):
        src = data if levels_from is None else levels_from
        cols: List[np.ndarray] = []
        self.names: List[str] = []
        self.info: List[Dict[str, Any]] = []
        fac = [f for f in factors]
        order: List[Tuple[str, str]] = []
        for c in covs:
            if c in fac:
                continue
            s = data[c]
            if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
                order.append((c, "cont"))
            else:
                order.append((c, "fac"))
                notes.append(f"{c}: 숫자형이 아니어서 요인(더미 변수)으로 처리")
        order += [(f, "fac") for f in fac]
        for var, kind in order:
            if kind == "cont":
                cols.append(pd.to_numeric(data[var], errors="coerce").astype(float).to_numpy())
                self.names.append(var)
                self.info.append({"var": var, "kind": "cont"})
                continue
            levels = _sorted_levels(src[var])
            if len(levels) < 2:
                notes.append(f"{var}: 범주가 1개뿐이어서 제외")
                continue
            base = levels[0]
            for lv in levels[1:]:
                cols.append((data[var] == lv).to_numpy(dtype=float))
                self.names.append(f"{var}_{_lvl_str(lv)}")
                self.info.append({"var": var, "kind": "dummy", "level": lv, "base": base})
        if const:
            cols.append(np.ones(len(data)))
            self.names.append("_cons")
            self.info.append({"var": "_cons", "kind": "const"})
        self.X = np.column_stack(cols) if cols else np.empty((len(data), 0))

    def drop_collinear(self, notes: List[str], rows: Optional[np.ndarray] = None) -> None:
        """Omit columns that are linear combinations of earlier ones (constant checked first), like Stata _rmcoll."""
        X = self.X if rows is None else self.X[rows]
        p = X.shape[1]
        order = ([p - 1] if self.names and self.names[-1] == "_cons" else []) + \
                [j for j in range(p) if not (self.names[j] == "_cons" and j == p - 1)]
        basis: List[np.ndarray] = []
        keep = []
        for j in order:
            v = X[:, j].astype(float).copy()
            nv = np.linalg.norm(v)
            if nv == 0:
                continue
            for _ in range(2):
                for q in basis:
                    v -= (q @ v) * q
            r = np.linalg.norm(v)
            if r > 1e-9 * nv:
                basis.append(v / r)
                keep.append(j)
        keep_set = set(keep)
        dropped = [self.names[j] for j in range(p) if j not in keep_set]
        if dropped:
            notes.append(f"공선성으로 제외된 변수 (omitted because of collinearity): {', '.join(dropped)}")
            idx = [j for j in range(p) if j in keep_set]
            self.X = self.X[:, idx]
            self.names = [self.names[j] for j in idx]
            self.info = [self.info[j] for j in idx]

    def keep_columns(self, idx: Sequence[int]) -> None:
        idx = list(idx)
        self.X = self.X[:, idx]
        self.names = [self.names[j] for j in idx]
        self.info = [self.info[j] for j in idx]

    @property
    def slope_idx(self) -> List[int]:
        return [j for j, inf in enumerate(self.info) if inf["kind"] != "const"]

    def factor_groups(self) -> Dict[str, List[int]]:
        g: Dict[str, List[int]] = {}
        for j, inf in enumerate(self.info):
            if inf["kind"] == "dummy":
                g.setdefault(inf["var"], []).append(j)
        return g

    def base_notes(self) -> List[str]:
        seen, out = set(), []
        for inf in self.info:
            if inf["kind"] == "dummy" and inf["var"] not in seen:
                seen.add(inf["var"])
                out.append(f"{inf['var']}={_lvl_str(inf['base'])}")
        return out


# =============================================================================================== estimation engine
# Estimators are statsmodels models: Logit, Probit, GLM(Binomial, CLogLog), OrderedModel, MNLogit, OLS, and
# GenericLikelihoodModel subclasses for tobit, truncreg, heckman (ML) and heckprobit (statsmodels supplies the
# Newton optimiser, OIM / sandwich / cluster covariances). Glue here: Stata's small-sample factors, Wald tests,
# delta-method Jacobians (statsmodels numdiff) and parameter-scaled numerical steps.
import warnings

import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.numdiff import approx_fprime


def _col_scale(X: np.ndarray, mult: float = 1.0) -> np.ndarray:
    """Characteristic step scale of coefficients on the columns of X: mult / rms(column)."""
    rms = np.sqrt(np.mean(np.asarray(X, float) ** 2, axis=0)) if X.size else np.zeros(X.shape[1])
    return mult / np.where(rms > 0, rms, 1.0)


class _FitInfo:
    def __init__(self, ll: float, converged: bool, iterations: int):
        self.ll, self.converged, self.iterations = ll, converged, iterations


def _cov_kw(kind: str, groups: Optional[np.ndarray]) -> Dict[str, Any]:
    """statsmodels covariance arguments; Stata's small-sample factors are applied afterwards in _stata_cov."""
    if kind == "oim":
        return {}
    if kind == "robust":
        return {"cov_type": "HC0"}
    return {"cov_type": "cluster", "cov_kwds": {"groups": np.asarray(groups), "use_correction": False}}


def _stata_cov(res, kind: str, n: int, groups: Optional[np.ndarray], V: Optional[np.ndarray] = None):
    """Covariance with Stata's factors for ML estimators: N/(N-1) for vce(robust), G/(G-1) for vce(cluster)."""
    V = np.asarray(res.cov_params() if V is None else V, dtype=float)
    if kind == "oim":
        return V, {"factor": 1.0}
    if kind == "robust":
        c = n / (n - 1.0)
        return V * c, {"factor": c}
    G = int(len(np.unique(groups)))
    if G < 2:
        raise ValueError("군집 강건 표준오차: 군집 2개 이상 필요")
    c = G / (G - 1.0)
    return V * c, {"factor": c, "G": G}


def _converged(res) -> bool:
    rv = getattr(res, "mle_retvals", None) or {}
    return bool(rv.get("converged", True))


def _iterations(res) -> int:
    rv = getattr(res, "mle_retvals", None) or {}
    return int(rv.get("iterations", 0) or 0)


def _quiet():
    w = warnings.catch_warnings()
    w.__enter__()
    warnings.simplefilter("ignore")
    return w


class _LLModel(GenericLikelihoodModel):
    """statsmodels likelihood model with analytic per-observation scores (own log likelihood) and a Hessian from
    central differences of the analytic score (statsmodels numdiff), steps scaled by the regressors."""

    def __init__(self, endog, exog, extra_names, step_scale, **kw):
        super().__init__(endog, exog, extra_params_names=list(extra_names), **kw)
        self.step_scale = np.asarray(step_scale, dtype=float)

    def loglike(self, params):
        v = float(np.sum(self.loglikeobs(np.asarray(params, float))))
        return v if np.isfinite(v) else -1e300

    def score(self, params):
        return self.score_obs(np.asarray(params, float)).sum(axis=0)

    def hessian(self, params):
        H = approx_fprime(np.asarray(params, float), self.score, epsilon=1e-5 * self.step_scale, centered=True)
        return 0.5 * (H + H.T)


def _mle(model, start: np.ndarray, kind: str, groups: Optional[np.ndarray], maxiter: int = 300):
    """Fit a statsmodels likelihood model: Newton from the start values, BFGS then Newton if that fails."""
    kw = _cov_kw(kind, groups)
    w = _quiet()
    try:
        res = model.fit(start_params=np.asarray(start, float), method="newton", maxiter=maxiter, tol=1e-10, disp=0, **kw)
        if not _converged(res) or not np.isfinite(res.llf) or res.llf <= -1e299:
            r0 = model.fit(start_params=np.asarray(start, float), method="bfgs", maxiter=5000, gtol=1e-9, disp=0)
            res = model.fit(start_params=r0.params, method="newton", maxiter=maxiter, tol=1e-10, disp=0, **kw)
    finally:
        w.__exit__(None, None, None)
    return res


def _vce_note(kind: str, extra: Dict[str, Any], cluster: Optional[str]) -> str:
    if kind == "oim":
        return "표준오차: 관측 정보행렬(OIM) 기준 (Stata 기본 vce(oim))"
    if kind == "robust":
        return "표준오차: Huber–White 강건 표준오차, N/(N−1) 보정 (Stata vce(robust))"
    return f"표준오차: 군집 강건 표준오차, 군집 {extra.get('G')}개({cluster}), G/(G−1) 보정 (Stata vce(cluster {cluster}))"


def _wald(b: np.ndarray, V: np.ndarray, idx: Sequence[int]) -> Tuple[float, int, float]:
    idx = list(idx)
    if not idx:
        return float("nan"), 0, float("nan")
    bb = b[idx]
    VV = V[np.ix_(idx, idx)]
    try:
        W = float(bb @ np.linalg.solve(VV, bb))
        df = len(idx)
    except np.linalg.LinAlgError:
        W = float(bb @ np.linalg.pinv(VV) @ bb)
        df = int(np.linalg.matrix_rank(VV))
    return W, df, float(stats.chi2.sf(W, df))


def _num_jac(fn: Callable[[np.ndarray], np.ndarray], theta: np.ndarray, scale: Optional[np.ndarray] = None,
             h_rel: float = 1e-6) -> np.ndarray:
    """Jacobian for the delta method (statsmodels numdiff, central differences); steps follow the parameter scale."""
    theta = np.asarray(theta, float)
    f0 = np.atleast_1d(np.asarray(fn(theta), dtype=float))
    eps = h_rel * (np.asarray(scale, float) if scale is not None else np.maximum(1.0, np.abs(theta)))
    J = np.asarray(approx_fprime(theta, lambda t: np.atleast_1d(np.asarray(fn(t), dtype=float)), epsilon=eps,
                                 centered=True), dtype=float)
    return J.reshape(f0.size, theta.size)


def _groups(series: Optional[pd.Series]) -> Optional[np.ndarray]:
    if series is None:
        return None
    codes, _ = pd.factorize(series, sort=True)
    return codes.astype(int)


class _Inference:
    """z (or t with df) statistics, p values and confidence intervals."""

    def __init__(self, level: float, df_r: Optional[float] = None):
        self.level = level
        self.df_r = df_r
        a = 1.0 - level / 100.0
        self.q = float(stats.t.ppf(1 - a / 2, df_r)) if df_r else float(stats.norm.ppf(1 - a / 2))
        self.stat_name = "t" if df_r else "z"

    def stat_p(self, b: float, se: float) -> Tuple[float, float]:
        if not (se > 0) or not np.isfinite(se):
            return float("nan"), float("nan")
        z = b / se
        p = 2.0 * (stats.t.sf(abs(z), self.df_r) if self.df_r else stats.norm.sf(abs(z)))
        return float(z), float(p)

    def ci(self, b: float, se: float) -> Tuple[float, float]:
        return float(b - self.q * se), float(b + self.q * se)


def _coef_table(name: str, title: str, inf: _Inference, eq_col: Optional[str] = None,
                exp_title: Optional[str] = None) -> ResultsTable:
    t = ResultsTable(name=name, title=title)
    if eq_col:
        t.add_column("eq", eq_col, col_type="text")
    t.add_column("term", "변수", col_type="text")
    t.add_column("b", "계수 (B)", col_type="number")
    t.add_column("se", "표준오차", col_type="number")
    t.add_column("z", inf.stat_name, col_type="number")
    t.add_column("p", "p", col_type="number", fmt="pvalue")
    t.add_column("lo", f"{inf.level:g}% CI 하한", col_type="number")
    t.add_column("hi", f"{inf.level:g}% CI 상한", col_type="number")
    if exp_title:
        t.add_column("eb", exp_title, col_type="number")
        t.add_column("eb_se", f"{exp_title} 표준오차", col_type="number")
        t.add_column("eb_lo", f"{exp_title} CI 하한", col_type="number")
        t.add_column("eb_hi", f"{exp_title} CI 상한", col_type="number")
    return t


def _coef_row(inf: _Inference, term: str, b: float, se: float, eq: Optional[str] = None, exp: bool = False,
              test: bool = True) -> Dict[str, Any]:
    z, p = inf.stat_p(b, se) if test else ("", "")
    lo, hi = inf.ci(b, se)
    row = {"term": term, "b": float(b), "se": float(se), "z": z, "p": p, "lo": lo, "hi": hi}
    if eq is not None:
        row["eq"] = eq
    if exp:
        eb = float(np.exp(np.clip(b, -700, 700)))
        row.update({"eb": eb, "eb_se": eb * float(se), "eb_lo": float(np.exp(np.clip(lo, -700, 700))),
                    "eb_hi": float(np.exp(np.clip(hi, -700, 700)))})
    return row


def _finish_coef(t: ResultsTable, exp: bool, label, check_separation: bool = False) -> ResultsTable:
    if check_separation:
        bad = [str(r.get("term")) for r in t.rows
               if isinstance(r.get("se"), (int, float)) and (not np.isfinite(r["se"]) or r["se"] > 100)
               and isinstance(r.get("b"), (int, float)) and abs(r["b"]) > 5]
        if bad:
            t.add_note("매우 큰 계수·표준오차: " + ", ".join(dict.fromkeys(bad)) +
                       " — 완전·준완전 분리(특정 범주 조합에 결과 값 한쪽만 존재) 의심, 범주 통합 또는 변수 제외 검토")
    t.mark_coefficients(estimate="b", se="se", p="p", stat="z", ci=("lo", "hi"),
                        exp_estimate="eb" if exp else None, label=label)
    return t


def _fit_table(name: str, rows: List[Tuple[str, Any, str]], title: str = "모형 적합도 (Model fit)") -> ResultsTable:
    """Long two-column table (statistic, value) with a per-row format."""
    t = ResultsTable(name=name, title=title)
    t.add_column("stat", "항목", col_type="text")
    t.add_column("value", "값", col_type="number")
    for label, val, fmt in rows:
        r = {"stat": label, "value": val}
        if fmt:
            r["_rowfmt"] = fmt
        t.add_row(r)
    return t


def _me_table(name: str, title: str, inf: _Inference, outcome_col: bool = False) -> ResultsTable:
    t = ResultsTable(name=name, title=title)
    if outcome_col:
        t.add_column("outcome", "결과 범주", col_type="text")
    t.add_column("term", "변수", col_type="text")
    t.add_column("b", "dy/dx", col_type="number")
    t.add_column("se", "표준오차 (delta)", col_type="number")
    t.add_column("z", inf.stat_name, col_type="number")
    t.add_column("p", "p", col_type="number", fmt="pvalue")
    t.add_column("lo", f"{inf.level:g}% CI 하한", col_type="number")
    t.add_column("hi", f"{inf.level:g}% CI 상한", col_type="number")
    return t


ME_NOTE_AME = ("dy/dx: 표본 평균 한계효과(AME, Stata margins, dydx(*)). 연속변수는 미분, 요인 더미는 기준 범주 대비 "
               "이산 변화. 표준오차는 delta 방법(공변량 고정)")
ME_NOTE_MEM = ("dy/dx: 평균값에서의 한계효과(MEM, Stata margins, dydx(*) atmeans). 다른 변수(요인 더미 포함)를 표본 "
               "평균에 고정, 요인 더미는 기준 범주 대비 이산 변화. 표준오차는 delta 방법")


def _me_effects(theta: np.ndarray, X: np.ndarray, info: List[Dict[str, Any]], groups: Dict[str, List[int]],
                pred: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
                coef_of_col: Callable[[np.ndarray, int], float], at_means: bool) -> np.ndarray:
    """Marginal effects for a single-index prediction pred(theta, X) -> (value, d value / d eta)."""
    Xe = X.mean(axis=0, keepdims=True) if at_means else X
    val, deta = pred(theta, Xe)
    out = []
    for j, inf in enumerate(info):
        if inf["kind"] == "cont":
            out.append(float(np.mean(deta * coef_of_col(theta, j))))
        elif inf["kind"] == "dummy":
            g = groups[inf["var"]]
            X1, X0 = Xe.copy(), Xe.copy()
            X1[:, g] = 0.0
            X0[:, g] = 0.0
            X1[:, j] = 1.0
            out.append(float(np.mean(pred(theta, X1)[0] - pred(theta, X0)[0])))
    return np.asarray(out)


def _me_terms(info: List[Dict[str, Any]], names: List[str]) -> List[str]:
    return [names[j] for j, inf in enumerate(info) if inf["kind"] in ("cont", "dummy")]


# =============================================================================================== binary links
LINKS_BIN = ("logit", "probit", "cloglog")
_LINK_ALIASES = {"logistic": "logit", "normal": "probit", "complementary log-log": "cloglog", "cll": "cloglog"}


def _log_phi(t):
    return -0.5 * t * t - 0.5 * LOG_2PI


def _cdf(link: str, eta: np.ndarray):
    """F, f = F', f' for a binary/ordinal link."""
    if link == "logit":
        F = special.expit(eta)
        f = F * (1.0 - F)
        return F, f, f * (1.0 - 2.0 * F)
    if link == "probit":
        F = special.ndtr(eta)
        f = np.exp(_log_phi(eta))
        with np.errstate(invalid="ignore"):
            d = np.where(np.isfinite(eta), -eta * f, 0.0)
        return F, f, d
    with np.errstate(over="ignore"):
        mu = np.exp(np.minimum(eta, 700.0))
        F = -np.expm1(-mu)
        f = mu * np.exp(-mu)
        f = np.where(np.isfinite(f), f, 0.0)
        return F, f, f * (1.0 - mu)


def _sm_binary(y: np.ndarray, X: np.ndarray, link: str, kind: str = "oim", groups: Optional[np.ndarray] = None):
    """statsmodels Logit / Probit (analytic Hessian) or GLM Binomial-CLogLog fitted by Newton with the observed
    Hessian (Stata cloglog reports OIM standard errors; IRLS would give the expected information)."""
    kw = _cov_kw(kind, groups)
    w = _quiet()
    try:
        if link == "logit":
            res = sm.Logit(y, X).fit(method="newton", maxiter=200, tol=1e-12, disp=0, **kw)
        elif link == "probit":
            res = sm.Probit(y, X).fit(method="newton", maxiter=200, tol=1e-12, disp=0, **kw)
        else:
            fam = sm.families.Binomial(link=sm.families.links.CLogLog())
            res = sm.GLM(y, X, family=fam).fit(method="newton", optim_hessian="oim", maxiter=300, tol=1e-12, disp=0,
                                               **kw)
    finally:
        w.__exit__(None, None, None)
    return res


def _margins_lib(res, X: np.ndarray, info: List[Dict[str, Any]], groups_f: Dict[str, List[int]], theta: np.ndarray,
                 V: np.ndarray, c: float, at_means: bool, predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 scale: Optional[np.ndarray] = None):
    """Marginal effects as Stata margins, dydx(*): continuous regressors from statsmodels get_margeff (delta-method
    SE scaled by the Stata covariance factor c); factor dummies as discrete change from the base level with the
    other levels of that factor set to 0 (own glue; statsmodels' dummy=True holds the other dummies at their
    observed values, which differs from Stata for factors with 3+ levels). Returns (values, ses), terms x outcomes."""
    w = _quiet()
    try:
        me = res.get_margeff(at="mean" if at_means else "overall", method="dydx", dummy=False)
        mv, ms = np.asarray(me.margeff, float), np.asarray(me.margeff_se, float) * math.sqrt(c)
    finally:
        w.__exit__(None, None, None)
    if mv.ndim == 1:
        mv, ms = mv[:, None], ms[:, None]
    nonconst = [j for j, inf in enumerate(info) if inf["kind"] != "const"]
    pos = {j: i for i, j in enumerate(nonconst)}
    Xe = X.mean(axis=0, keepdims=True) if at_means else X
    vals, ses = [], []
    for j, inf in enumerate(info):
        if inf["kind"] == "cont":
            vals.append(mv[pos[j]])
            ses.append(ms[pos[j]])
        elif inf["kind"] == "dummy":
            g = groups_f[inf["var"]]
            X1, X0 = Xe.copy(), Xe.copy()
            X1[:, g] = 0.0
            X0[:, g] = 0.0
            X1[:, j] = 1.0
            def fn(t, X1=X1, X0=X0):
                d = np.asarray(predict(t, X1), float) - np.asarray(predict(t, X0), float)
                return np.atleast_1d(d.mean(axis=0))
            v = fn(theta)
            Jm = _num_jac(fn, theta, scale)
            vals.append(v)
            ses.append(np.sqrt(np.clip(np.diag(Jm @ V @ Jm.T), 0, None)))
    return np.asarray(vals).reshape(len(vals), -1), np.asarray(ses).reshape(len(ses), -1)


def _binary_outcome(s: pd.Series, dep: str, notes: List[str]) -> np.ndarray:
    vals = _sorted_levels(s)
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        fv = [float(v) for v in vals]
        if set(fv) <= {0.0, 1.0}:
            y = s.astype(float).to_numpy()
        elif len(vals) == 2:
            y = (s.astype(float) == fv[1]).to_numpy(dtype=float)
            notes.append(f"종속변수 코딩: {dep}={_lvl_str(vals[1])} → 1, {dep}={_lvl_str(vals[0])} → 0 "
                         f"(Stata는 0이 아닌 값을 모두 1로 처리)")
        elif 0.0 in fv:
            y = (s.astype(float) != 0).to_numpy(dtype=float)
            notes.append(f"종속변수 코딩: {dep}≠0 → 1, {dep}=0 → 0 (Stata 규칙)")
        else:
            raise ValueError(f"{dep}: 이항 종속변수 필요 (값 {len(vals)}개, 0 없음)")
    else:
        if len(vals) != 2:
            raise ValueError(f"{dep}: 이항 종속변수 필요 (범주 {len(vals)}개: {', '.join(map(str, vals[:6]))})")
        y = (s == vals[1]).to_numpy(dtype=float)
        notes.append(f"종속변수 코딩: {dep}='{vals[1]}' → 1, '{vals[0]}' → 0")
    if y.min() == y.max():
        raise ValueError(f"{dep}: 종속변수 값이 한 가지 — 추정 불가 (outcome does not vary)")
    return y


def _perfect_prediction(y: np.ndarray, des: _Design, notes: List[str]) -> np.ndarray:
    """Stata logit/probit rule for 0/1 regressors: x!=0 (or x==0) predicting the outcome perfectly ->
    the variable is dropped and those observations are not used. Returns the row mask kept."""
    keep = np.ones(len(y), dtype=bool)
    changed = True
    while changed:
        changed = False
        for j, inf in enumerate(list(des.info)):
            if inf["kind"] == "const":
                continue
            x = des.X[keep, j]
            if not np.all((x == 0) | (x == 1)):
                continue
            yy = y[keep]
            for val, txt in ((1.0, "!= 0"), (0.0, "== 0")):
                sel = x == val
                if sel.sum() == 0 or sel.sum() == len(x):
                    continue
                ys = yy[sel]
                if ys.min() == ys.max():
                    what = "성공" if ys[0] == 1 else "실패"
                    idx = np.where(keep)[0][sel]
                    keep[idx] = False
                    notes.append(f"{des.names[j]} {txt} → {what} 완전 예측: 변수 제외, 관측치 {len(idx)}개 미사용 "
                                 f"(Stata: predicts {'success' if ys[0] == 1 else 'failure'} perfectly)")
                    des.keep_columns([k for k in range(len(des.names)) if k != j])
                    changed = True
                    break
            if changed:
                break
    return keep


def _hosmer_lemeshow(y: np.ndarray, pr: np.ndarray, groups: int):
    """Hosmer–Lemeshow test with Stata `estat gof, group(#)` grouping (_pctile cut points, ties kept together)."""
    n = len(pr)
    xs = np.sort(pr)
    cuts = []
    for k in range(1, groups):
        w = n * k / groups
        wi = int(round(w))
        if abs(w - wi) < 1e-9:
            q = 0.5 * (xs[wi - 1] + xs[min(wi, n - 1)])
        else:
            q = xs[int(math.ceil(w)) - 1]
        cuts.append(q)
    grp = np.full(n, groups, dtype=int)
    for k in range(groups - 1, 0, -1):
        grp[pr <= cuts[k - 1]] = k
    rows = []
    chi = 0.0
    for g in sorted(set(grp.tolist())):
        sel = grp == g
        ng = int(sel.sum())
        o1 = float(y[sel].sum())
        e1 = float(pr[sel].sum())
        rows.append({"group": g, "cut": float(pr[sel].max()), "o1": o1, "e1": e1, "o0": ng - o1, "e0": ng - e1, "n": ng})
        pbar = e1 / ng
        chi += (o1 - e1) ** 2 / (ng * pbar * (1 - pbar))
    G = len(rows)
    df = G - 2
    return rows, chi, df, float(stats.chi2.sf(chi, df)) if df > 0 else float("nan")


# =============================================================================================== binaryReg
class BinaryReg(BaseAnalysis):
    """Binary choice models: logit / probit / cloglog (Stata)."""

    STATA = {"logit": "logit", "probit": "probit", "cloglog": "cloglog"}
    TITLES = {"logit": "이항 로짓 (logit)", "probit": "이항 프로빗 (probit)", "cloglog": "보완 로그-로그 (cloglog)"}

    def __init__(self, data: pd.DataFrame, dep: str, covs=None, factors=None, link: str = "logit",
                 odds_ratio: bool = False, ame: bool = False, mem: bool = False, class_table: bool = False,
                 cutoff: float = 0.5, hosmer: bool = False, hl_groups: int = 10, hl_table: bool = False,
                 roc: bool = False, cov_type: str = "oim", cluster: Optional[str] = None, ci_width: float = 95.0):
        link = resolve_choice("연결함수(link)", link, LINKS_BIN, _LINK_ALIASES, default="logit")
        cov_type = resolve_choice("표준오차 유형(covType)", cov_type, COV_TYPES, _COV_ALIASES, default="oim")
        covs, factors = _as_list(covs), _as_list(factors)
        if odds_ratio and link == "probit":
            raise ValueError("승산비(oddsRatio): probit에는 해당 없음 (logit의 OR, cloglog의 exp(b)만 가능)")
        if not (0.0 < float(cutoff) < 1.0):
            raise ValueError(f"분류 기준값(cutoff): 0과 1 사이 필요 (입력: {cutoff})")
        if int(hl_groups) < 3:
            raise ValueError(f"Hosmer–Lemeshow 집단 수(hlGroups): 3 이상 필요 (입력: {hl_groups})")
        if cov_type == "cluster" and not cluster:
            raise ValueError("군집 강건 표준오차(covType='cluster'): 군집 변수(clusterVar) 지정 필요")
        cluster = cluster if cov_type == "cluster" else None       # unused otherwise (no listwise deletion on it)
        super().__init__(name="binaryReg", title=f"{self.TITLES[link]} 회귀", dataset=data, options={
            "dep": dep, "covs": covs, "factors": factors, "link": link, "oddsRatio": bool(odds_ratio),
            "ame": bool(ame), "mem": bool(mem), "classTable": bool(class_table), "cutoff": float(cutoff),
            "hosmer": bool(hosmer), "hlGroups": int(hl_groups), "hlTable": bool(hl_table), "roc": bool(roc),
            "covType": cov_type, "clusterVar": cluster, "ciWidth": float(ci_width)})
        self.run()

    def run(self):
        o = self.options
        df, dep, covs, factors, link = self.dataset, o["dep"], o["covs"], o["factors"], o["link"]
        _check_columns(df, self.title, {"dep": dep, "covs/factors": covs + factors},
                       {"clusterVar": o["clusterVar"]} if o["clusterVar"] else {})
        notes: List[str] = []
        cols = [dep] + covs + [f for f in factors if f not in covs] + ([o["clusterVar"]] if o["clusterVar"] else [])
        data = df[list(dict.fromkeys(cols))].dropna()
        if len(data) == 0:
            raise ValueError(f"{self.title}: 결측 제외 후 관측치 없음")
        y = _binary_outcome(data[dep], dep, notes)
        des = _Design(data, covs, factors, notes)
        des.drop_collinear(notes)
        keep = _perfect_prediction(y, des, notes)
        if not keep.all():
            data, y, des.X = data[keep], y[keep], des.X[keep]
        X = des.X
        n, k = X.shape
        groups = _groups(data[o["clusterVar"]]) if o["covType"] == "cluster" else None
        res = _sm_binary(y, X, link, o["covType"], groups)
        b = np.asarray(res.params, float)
        eta = X @ b
        V, vx = _stata_cov(res, o["covType"], n, groups)
        se = np.sqrt(np.clip(np.diag(V), 0, None))
        fit = _FitInfo(float(res.llf), _converged(res), _iterations(res))
        inf = _Inference(o["ciWidth"])
        n1 = float(y.sum())
        ll0 = n1 * math.log(n1 / n) + (n - n1) * math.log(1 - n1 / n)
        slopes = des.slope_idx
        r2 = 1.0 - fit.ll / ll0
        aic = -2 * fit.ll + 2 * k
        bic = -2 * fit.ll + k * math.log(n)
        if o["covType"] == "oim":
            chi2, dfm = 2.0 * (fit.ll - ll0), len(slopes)
            pchi = float(stats.chi2.sf(chi2, dfm)) if dfm > 0 else float("nan")
            test = "LR"
        else:
            chi2, dfm, pchi = _wald(b, V, slopes)
            test = "Wald"
        ll_lab = "로그우도 (Log likelihood)" if o["covType"] == "oim" else "로그 유사우도 (Log pseudolikelihood)"
        fit_t = _fit_table("fit", [("관측치 수 (N)", n, "integer"), (ll_lab, fit.ll, ""),
                                   (f"{test} χ²", chi2, ""), ("자유도 (df)", dfm, "integer"), ("p", pchi, "pvalue"),
                                   ("McFadden 유사 R² (Pseudo R2)", r2, "zto"), ("AIC", aic, ""), ("BIC", bic, "")])
        fit_t.add_note(f"{test} χ²: 모든 기울기 = 0 검정" + (" (상수항 모형 대비 우도비)" if test == "LR" else ""))
        if not fit.converged:
            fit_t.add_note(f"수렴 실패: 반복 {fit.iterations}회 후 중단 — 추정치 해석 주의")
        self.add_table(fit_t)

        exp = bool(o["oddsRatio"])
        exp_title = {"logit": "승산비 (OR)", "cloglog": "exp(b)"}.get(link) if exp else None
        ct = _coef_table("coef", f"계수 추정치 (Coefficients) — {dep}", inf, exp_title=exp_title)
        for j, nm in enumerate(des.names):
            ct.add_row(_coef_row(inf, nm, b[j], se[j], exp=exp))
        for nt in notes:
            ct.add_note(nt)
        if des.base_notes():
            ct.add_note("요인 기준 범주: " + ", ".join(des.base_notes()))
        ct.add_note(_vce_note(o["covType"], vx, o["clusterVar"]))
        if exp and link == "logit":
            ct.add_note("승산비(OR) = exp(B), 표준오차는 delta 방법 exp(B)·SE (Stata logit, or)")
        self.add_table(_finish_coef(ct, exp, "term", check_separation=True))

        # margins: statsmodels get_margeff (continuous) + discrete change for factor dummies
        groups_f = des.factor_groups()
        predict = lambda theta, Xm: res.model.predict(theta, Xm)
        for flag, at_means, name, title, note in (
                ("ame", False, "ame", "평균 한계효과 (Average marginal effects) — Pr(" + dep + "=1)", ME_NOTE_AME),
                ("mem", True, "mem", "평균값에서의 한계효과 (Marginal effects at means) — Pr(" + dep + "=1)", ME_NOTE_MEM)):
            if not o[flag]:
                continue
            mv, ms = _margins_lib(res, X, des.info, groups_f, b, V, vx["factor"], at_means, predict, _col_scale(X))
            me, mse = mv[:, 0], ms[:, 0]
            mt = _me_table(name, title, inf)
            for nm, v, s in zip(_me_terms(des.info, des.names), me, mse):
                z, p = inf.stat_p(v, s)
                lo, hi = inf.ci(v, s)
                mt.add_row({"term": nm, "b": v, "se": s, "z": z, "p": p, "lo": lo, "hi": hi})
            mt.add_note(note)
            mt.mark_coefficients(estimate="b", se="se", p="p", stat="z", ci=("lo", "hi"), label="term")
            self.add_table(mt)

        pr = np.asarray(res.model.predict(b, X), float)
        if o["classTable"]:
            self._classification(y, pr, o["cutoff"])
        if o["hosmer"]:
            rows, chi, dfh, ph = _hosmer_lemeshow(y, pr, int(o["hlGroups"]))
            ht = _fit_table("hosmer", [("관측치 수", n, "integer"), ("집단 수", len(rows), "integer"),
                                       ("Hosmer–Lemeshow χ²", chi, ""), ("자유도 (df)", dfh, "integer"),
                                       ("p", ph, "pvalue")], title="Hosmer–Lemeshow 적합도 검정 (estat gof, group)")
            ht.add_note(f"예측확률 분위수로 {int(o['hlGroups'])}개 집단 구성, 동점은 같은 집단 (Stata estat gof, group({int(o['hlGroups'])}))")
            self.add_table(ht)
            if o["hlTable"]:
                gt = ResultsTable(name="hosmer_groups", title="Hosmer–Lemeshow 집단별 관측·기대 빈도")
                for c, ttl, ty in (("group", "집단", "integer"), ("cut", "예측확률 상한", "number"),
                                   ("o1", "관측 1", "number"), ("e1", "기대 1", "number"),
                                   ("o0", "관측 0", "number"), ("e0", "기대 0", "number"), ("n", "합계", "integer")):
                    gt.add_column(c, ttl, col_type=ty, fmt="zto" if c == "cut" else "")
                for r in rows:
                    gt.add_row(r)
                self.add_table(gt)
        auc = None
        if o["roc"]:
            from sklearn.metrics import roc_auc_score, roc_curve
            auc = float(roc_auc_score(y, pr))
            rt = _fit_table("roc", [("관측치 수", n, "integer"), ("ROC 곡선 아래 면적 (AUC)", auc, "zto")],
                            title="ROC 곡선 아래 면적 (lroc)")
            self.add_table(rt)
            try:
                from .plots import plot_roc_curve
                fpr, tpr, _ = roc_curve(y, pr)
                self.add_plot("ROC 곡선 (lroc)", plot_roc_curve(fpr, tpr, auc))
            except Exception:
                pass

        self.model_stats = {"N": n, "LL": fit.ll, "Chi2": (chi2, pchi), "df": dfm, "R2_McF": r2, "AIC": aic, "BIC": bic}
        self.syntax_python = _sdk_call("binaryReg", o, {"link": "logit", "oddsRatio": False, "ame": False, "mem": False,
                                                        "classTable": False, "cutoff": 0.5, "hosmer": False,
                                                        "hlGroups": 10, "hlTable": False, "roc": False,
                                                        "covType": "oim", "clusterVar": None, "ciWidth": 95.0})
        self.syntax_r = "\n".join(_r_lines("binaryReg", o))
        self.apa_narrative = (f"{self.TITLES[link]} 회귀모형(N = {n})의 {test} χ²({dfm}) = {chi2:.2f}, p {_pstr(pchi)}, "
                              f"McFadden 유사 R² = {r2:.3f}" + (f", AUC = {auc:.3f}" if auc is not None else "") + ".")

    def _classification(self, y, pr, cutoff):
        pos = pr >= cutoff
        D = y == 1
        a, bb = int(np.sum(pos & D)), int(np.sum(pos & ~D))
        c, d = int(np.sum(~pos & D)), int(np.sum(~pos & ~D))
        t = ResultsTable(name="classification", title=f"분류표 (estat classification, 기준값 {cutoff:g})")
        t.add_column("cls", "예측 분류", col_type="text")
        t.add_column("D", "실제 D (=1)", col_type="integer")
        t.add_column("nD", "실제 ~D (=0)", col_type="integer")
        t.add_column("tot", "합계", col_type="integer")
        t.add_row({"cls": "+ (예측 1)", "D": a, "nD": bb, "tot": a + bb})
        t.add_row({"cls": "− (예측 0)", "D": c, "nD": d, "tot": c + d})
        t.add_row({"cls": "합계", "D": a + c, "nD": bb + d, "tot": a + bb + c + d})
        t.add_note(f"예측확률 ≥ {cutoff:g} 이면 + 로 분류 (Stata estat classification)")
        self.add_table(t)
        div = lambda u, v: 100.0 * u / v if v else float("nan")
        rows = [("민감도 Pr(+|D)", div(a, a + c)), ("특이도 Pr(−|~D)", div(d, bb + d)),
                ("양성 예측도 Pr(D|+)", div(a, a + bb)), ("음성 예측도 Pr(~D|−)", div(d, c + d)),
                ("위양성률 Pr(+|~D)", div(bb, bb + d)), ("위음성률 Pr(−|D)", div(c, a + c)),
                ("분류 + 중 오분류 Pr(~D|+)", div(bb, a + bb)), ("분류 − 중 오분류 Pr(D|−)", div(c, c + d)),
                ("정분류율 (Correctly classified)", div(a + d, a + bb + c + d))]
        self.add_table(_fit_table("class_stats", [(r, v, "pc") for r, v in rows], title="분류 정확도 (%)"))


# =============================================================================================== ordinalReg
def _sm_ordered(yk: np.ndarray, X: np.ndarray, link: str, kind: str, groups: Optional[np.ndarray]):
    """statsmodels OrderedModel (logit / probit) fitted by Newton."""
    w = _quiet()
    try:
        mod = OrderedModel(yk, X, distr=link)
        res = mod.fit(method="newton", maxiter=200, disp=0, **_cov_kw(kind, groups))
        if not _converged(res):
            r0 = mod.fit(method="bfgs", maxiter=5000, disp=0)
            res = mod.fit(start_params=r0.params, method="newton", maxiter=200, disp=0, **_cov_kw(kind, groups))
    finally:
        w.__exit__(None, None, None)
    return res


def _ordered_stata_params(res, p: int, K: int, kind: str, n: int, groups: Optional[np.ndarray]):
    """OrderedModel thresholds (first cut, then log increments) -> Stata cut1..cutK-1 with the delta-method
    covariance; Stata's robust / cluster factors applied."""
    raw = np.asarray(res.params, float)
    Vraw, vx = _stata_cov(res, kind, n, groups)
    cuts = np.asarray(res.model.transform_threshold_params(raw), float)[1:-1]
    J = np.eye(p + K - 1)
    r = raw[p:]
    for k in range(K - 1):
        J[p + k, p:] = 0.0
        J[p + k, p] = 1.0
        for j in range(1, k + 1):
            J[p + k, p + j] = math.exp(r[j])
    return np.concatenate([raw[:p], cuts]), J @ Vraw @ J.T, vx


def _ordinal_probs(theta: np.ndarray, X: np.ndarray, K: int, link: str):
    """Category probabilities (n x K) and their derivatives with respect to eta."""
    p = X.shape[1]
    eta = X @ theta[:p]
    cc = np.concatenate([[-np.inf], theta[p:], [np.inf]])
    F = np.empty((X.shape[0], K + 1))
    f = np.empty((X.shape[0], K + 1))
    for j in range(K + 1):
        a = cc[j] - eta
        Fa, fa, _ = _cdf(link, a)
        F[:, j] = Fa
        f[:, j] = np.where(np.isfinite(a), fa, 0.0)
    P = F[:, 1:] - F[:, :-1]
    dP = -(f[:, 1:] - f[:, :-1])
    return P, dP


def _brant(y_k: np.ndarray, X: np.ndarray, K: int, names: List[str]):
    """Brant (1990) Wald test of parallel lines from K-1 cumulative binary logits (as in Stata/R brant)."""
    n, p = X.shape
    Xc = np.column_stack([np.ones(n), X])
    betas, pis, invs, bad = [], [], [], []
    for k in range(K - 1):
        z = (y_k > k).astype(float)
        f = _sm_binary(z, Xc, "logit")                      # statsmodels Logit for each cumulative split
        bse = np.asarray(f.bse, float)
        if not _converged(f) or not np.all(np.isfinite(bse)) or np.any(bse > 1e3):
            bad.append(k + 1)
        betas.append(np.asarray(f.params, float))
        pis.append(np.asarray(f.predict(Xc), float))
        invs.append(np.asarray(f.cov_params(), float))     # (X'WX)^-1
    m = K - 1
    Vs = np.zeros((m * p, m * p))
    for i in range(m):
        for j in range(i, m):
            if i == j:
                Vij = invs[i]
            else:
                w = pis[j] - pis[i] * pis[j]
                Vij = invs[i] @ ((Xc.T * w) @ Xc) @ invs[j]
            Vs[i * p:(i + 1) * p, j * p:(j + 1) * p] = Vij[1:, 1:]
            Vs[j * p:(j + 1) * p, i * p:(i + 1) * p] = Vij[1:, 1:].T
    bstar = np.concatenate([bk[1:] for bk in betas])
    D = np.zeros(((m - 1) * p, m * p))
    for r in range(m - 1):
        D[r * p:(r + 1) * p, :p] = np.eye(p)
        D[r * p:(r + 1) * p, (r + 1) * p:(r + 2) * p] = -np.eye(p)
    out = []

    def test(sel_cols):
        idx = [r * p + c for r in range(m - 1) for c in sel_cols]
        Db = (D @ bstar)[idx]
        DV = (D @ Vs @ D.T)[np.ix_(idx, idx)]
        W = float(Db @ np.linalg.solve(DV, Db))
        return W, len(idx), float(stats.chi2.sf(W, len(idx)))

    out.append(("전체 (All)",) + test(list(range(p))))
    for c in range(p):
        out.append((names[c],) + test([c]))
    return out, bad


class OrdinalReg(BaseAnalysis):
    """Ordered logit / probit (Stata ologit / oprobit)."""

    TITLES = {"logit": "순서형 로짓 (ologit)", "probit": "순서형 프로빗 (oprobit)"}

    def __init__(self, data: pd.DataFrame, dep: str, covs=None, factors=None, link: str = "logit",
                 odds_ratio: bool = False, pred_probs: bool = False, ame: bool = False, brant: bool = False,
                 cov_type: str = "oim", cluster: Optional[str] = None, ci_width: float = 95.0):
        link = resolve_choice("연결함수(link)", link, ("logit", "probit"), _LINK_ALIASES, default="logit")
        cov_type = resolve_choice("표준오차 유형(covType)", cov_type, COV_TYPES, _COV_ALIASES, default="oim")
        covs, factors = _as_list(covs), _as_list(factors)
        if odds_ratio and link != "logit":
            raise ValueError("승산비(oddsRatio): ologit(link='logit')에서만 가능")
        if brant and link != "logit":
            raise ValueError("Brant 검정(brant): ologit(link='logit')에서만 가능")
        if cov_type == "cluster" and not cluster:
            raise ValueError("군집 강건 표준오차(covType='cluster'): 군집 변수(clusterVar) 지정 필요")
        cluster = cluster if cov_type == "cluster" else None       # unused otherwise (no listwise deletion on it)
        super().__init__(name="ordinalReg", title=f"{self.TITLES[link]} 회귀", dataset=data, options={
            "dep": dep, "covs": covs, "factors": factors, "link": link, "oddsRatio": bool(odds_ratio),
            "predProbs": bool(pred_probs), "ame": bool(ame), "brant": bool(brant), "covType": cov_type,
            "clusterVar": cluster, "ciWidth": float(ci_width)})
        self.run()

    def run(self):
        o = self.options
        df, dep, covs, factors, link = self.dataset, o["dep"], o["covs"], o["factors"], o["link"]
        _check_columns(df, self.title, {"dep": dep, "covs/factors": covs + factors},
                       {"clusterVar": o["clusterVar"]} if o["clusterVar"] else {})
        notes: List[str] = []
        cols = [dep] + covs + [f for f in factors if f not in covs] + ([o["clusterVar"]] if o["clusterVar"] else [])
        data = df[list(dict.fromkeys(cols))].dropna()
        cats = _sorted_levels(data[dep])
        K = len(cats)
        if K < 3:
            raise ValueError(f"{dep}: 순서형 종속변수는 범주 3개 이상 필요 (범주 {K}개 — 이항이면 binaryReg 사용)")
        if K > 30:
            raise ValueError(f"{dep}: 범주 {K}개 — 순서형 모형 대상으로 보기 어려움 (30개 이하)")
        cmap = {c: i for i, c in enumerate(cats)}
        yk = data[dep].map(cmap).to_numpy(dtype=int)
        if not (pd.api.types.is_numeric_dtype(data[dep]) or isinstance(data[dep].dtype, pd.CategoricalDtype)):
            notes.append(f"범주 순서(문자 정렬): {' < '.join(map(str, cats))}")
        des = _Design(data, covs, factors, notes, const=True)
        des.drop_collinear(notes)
        des.keep_columns([j for j, inf in enumerate(des.info) if inf["kind"] != "const"])
        X = des.X
        n, p = X.shape
        groups = _groups(data[o["clusterVar"]]) if o["covType"] == "cluster" else None
        res = _sm_ordered(yk, X, link, o["covType"], groups)
        th, V, vx = _ordered_stata_params(res, p, K, o["covType"], n, groups)
        fit = _FitInfo(float(res.llf), _converged(res), _iterations(res))
        oscale = np.concatenate([_col_scale(X), np.ones(K - 1)])
        se = np.sqrt(np.clip(np.diag(V), 0, None))
        inf = _Inference(o["ciWidth"])
        counts = np.bincount(yk, minlength=K)
        ll0 = float(np.sum(counts * np.log(counts / n)))
        npar = p + K - 1
        r2 = 1.0 - fit.ll / ll0
        aic = -2 * fit.ll + 2 * npar
        bic = -2 * fit.ll + npar * math.log(n)
        if o["covType"] == "oim":
            chi2, dfm, test = 2.0 * (fit.ll - ll0), p, "LR"
            pchi = float(stats.chi2.sf(chi2, dfm))
        else:
            chi2, dfm, pchi = _wald(th, V, list(range(p)))
            test = "Wald"
        ll_lab = "로그우도 (Log likelihood)" if o["covType"] == "oim" else "로그 유사우도 (Log pseudolikelihood)"
        ft = _fit_table("fit", [("관측치 수 (N)", n, "integer"), ("범주 수", K, "integer"), (ll_lab, fit.ll, ""),
                                (f"{test} χ²", chi2, ""), ("자유도 (df)", dfm, "integer"), ("p", pchi, "pvalue"),
                                ("McFadden 유사 R² (Pseudo R2)", r2, "zto"), ("AIC", aic, ""), ("BIC", bic, "")])
        if not fit.converged:
            ft.add_note(f"수렴 실패: 반복 {fit.iterations}회 후 중단 — 추정치 해석 주의")
        self.add_table(ft)

        exp = bool(o["oddsRatio"])
        ct = _coef_table("coef", f"계수 추정치 (Coefficients) — {dep}", inf, exp_title="승산비 (OR)" if exp else None)
        for j, nm in enumerate(des.names):
            ct.add_row(_coef_row(inf, nm, th[j], se[j], exp=exp))
        for k in range(K - 1):
            row = _coef_row(inf, f"/cut{k + 1}", th[p + k], se[p + k], test=False)
            if exp:
                row.update({"eb": "", "eb_se": "", "eb_lo": "", "eb_hi": ""})
            ct.add_row(row)
        Fn = "Λ" if link == "logit" else "Φ"
        ct.add_note(f"모수화: Pr({dep} ≤ k) = {Fn}(cut_k − xb), 상수항 없음 (Stata {('ologit' if link == 'logit' else 'oprobit')})")
        ct.add_note("범주: " + ", ".join(f"{i + 1}={_lvl_str(c)}" for i, c in enumerate(cats)))
        for nt in notes:
            ct.add_note(nt)
        if des.base_notes():
            ct.add_note("요인 기준 범주: " + ", ".join(des.base_notes()))
        ct.add_note(_vce_note(o["covType"], vx, o["clusterVar"]))
        self.add_table(_finish_coef(ct, exp, "term", check_separation=True))

        if o["predProbs"]:
            fnp = lambda t: _ordinal_probs(t, X, K, link)[0].mean(axis=0)
            pm = fnp(th)
            J = _num_jac(fnp, th, oscale)
            pse = np.sqrt(np.clip(np.diag(J @ V @ J.T), 0, None))
            pt = ResultsTable(name="pred_probs", title="범주별 평균 예측확률 (margins, predict(outcome()))")
            for c, ttl, ty, fm in (("cat", "범주", "text", ""), ("obs", "관측 비율", "number", "zto"),
                                   ("b", "평균 예측확률", "number", "zto"), ("se", "표준오차 (delta)", "number", ""),
                                   ("z", "z", "number", ""), ("p", "p", "number", "pvalue"),
                                   ("lo", f"{inf.level:g}% CI 하한", "number", "zto"),
                                   ("hi", f"{inf.level:g}% CI 상한", "number", "zto")):
                pt.add_column(c, ttl, col_type=ty, fmt=fm)
            for k in range(K):
                z, pv = inf.stat_p(pm[k], pse[k])
                lo, hi = inf.ci(pm[k], pse[k])
                pt.add_row({"cat": _lvl_str(cats[k]), "obs": counts[k] / n, "b": pm[k], "se": pse[k], "z": z, "p": pv,
                            "lo": lo, "hi": hi})
            self.add_table(pt)

        if o["ame"]:
            groups_f = des.factor_groups()
            mt = _me_table("ame", f"범주별 평균 한계효과 (AME) — Pr({dep}=k)", inf, outcome_col=True)
            terms = _me_terms(des.info, des.names)
            for k in range(K):
                def pred(t, Xm, k=k):
                    P, dP = _ordinal_probs(t, Xm, K, link)
                    return P[:, k], dP[:, k]
                fn = lambda t, pr=pred: _me_effects(t, X, des.info, groups_f, pr, lambda tt, j: tt[j], False)
                me = fn(th)
                J = _num_jac(fn, th, oscale)
                mse = np.sqrt(np.clip(np.diag(J @ V @ J.T), 0, None))
                for nm, v, s in zip(terms, me, mse):
                    z, pv = inf.stat_p(v, s)
                    lo, hi = inf.ci(v, s)
                    mt.add_row({"outcome": _lvl_str(cats[k]), "term": nm, "b": v, "se": s, "z": z, "p": pv, "lo": lo, "hi": hi})
            mt.add_note(ME_NOTE_AME + " — 결과 범주별 (Stata margins, dydx(*) predict(outcome(k)))")
            mt.mark_coefficients(estimate="b", se="se", p="p", stat="z", ci=("lo", "hi"), label=["outcome", "term"])
            self.add_table(mt)

        if o["brant"]:
            res, bad = _brant(yk, X, K, des.names)
            bt = ResultsTable(name="brant", title="Brant 평행선 가정 검정 (brant, detail)")
            bt.add_column("term", "변수", col_type="text")
            bt.add_column("chi2", "χ²", col_type="number")
            bt.add_column("df", "자유도", col_type="integer")
            bt.add_column("p", "p", col_type="number", fmt="pvalue")
            for nm, W, d, pv in res:
                bt.add_row({"term": nm, "chi2": W, "df": d, "p": pv})
            bt.add_note("방법: 누적 이항 로짓 K−1개(y > k) 개별 추정, 계수 동일성 Wald 검정, 추정치 간 공분산은 "
                        "Brant(1990) 식 (Stata spost13 brant · R brant 패키지와 같은 방식)")
            bt.add_note("유의한 χ²: 평행선(비례 승산) 가정 위배 가능성")
            if bad:
                cut_txt = ", ".join(f"y > {_lvl_str(cats[k - 1])}" for k in bad)
                bt.add_note(f"이항 로짓 분할 {cut_txt}: 수렴 실패 또는 매우 큰 표준오차(완전·준완전 분리) — 검정값 해석 불가, "
                            f"범주 통합 후 재검정 필요 (Stata도 이 경우 값이 불안정)")
            self.add_table(bt)

        self.model_stats = {"N": n, "LL": fit.ll, "Chi2": (chi2, pchi), "df": dfm, "R2_McF": r2, "AIC": aic, "BIC": bic}
        self.syntax_python = _sdk_call("ordinalReg", o, {"link": "logit", "oddsRatio": False, "predProbs": False,
                                                         "ame": False, "brant": False, "covType": "oim",
                                                         "clusterVar": None, "ciWidth": 95.0})
        self.syntax_r = "\n".join(_r_lines("ordinalReg", o))
        self.apa_narrative = (f"{self.TITLES[link]} 회귀모형(N = {n}, 범주 {K}개)의 {test} χ²({dfm}) = {chi2:.2f}, "
                              f"p {_pstr(pchi)}, McFadden 유사 R² = {r2:.3f}.")


# =============================================================================================== multinomReg
def _sm_mnlogit(ycode: np.ndarray, X: np.ndarray, kind: str, groups: Optional[np.ndarray]):
    """statsmodels MNLogit (analytic Hessian) by Newton; code 0 is the base outcome."""
    w = _quiet()
    try:
        res = sm.MNLogit(ycode, X).fit(method="newton", maxiter=200, tol=1e-12, disp=0, **_cov_kw(kind, groups))
    finally:
        w.__exit__(None, None, None)
    return res


class MultinomReg(BaseAnalysis):
    """Multinomial logit (Stata mlogit)."""

    def __init__(self, data: pd.DataFrame, dep: str, covs=None, factors=None, base_outcome: Any = None,
                 rrr: bool = False, ame: bool = False, cov_type: str = "oim", cluster: Optional[str] = None,
                 ci_width: float = 95.0):
        cov_type = resolve_choice("표준오차 유형(covType)", cov_type, COV_TYPES, _COV_ALIASES, default="oim")
        covs, factors = _as_list(covs), _as_list(factors)
        if cov_type == "cluster" and not cluster:
            raise ValueError("군집 강건 표준오차(covType='cluster'): 군집 변수(clusterVar) 지정 필요")
        cluster = cluster if cov_type == "cluster" else None       # unused otherwise (no listwise deletion on it)
        super().__init__(name="multinomReg", title="다항 로짓 (mlogit) 회귀", dataset=data, options={
            "dep": dep, "covs": covs, "factors": factors, "baseOutcome": base_outcome, "rrr": bool(rrr),
            "ame": bool(ame), "covType": cov_type, "clusterVar": cluster, "ciWidth": float(ci_width)})
        self.run()

    def run(self):
        o = self.options
        df, dep, covs, factors = self.dataset, o["dep"], o["covs"], o["factors"]
        _check_columns(df, self.title, {"dep": dep, "covs/factors": covs + factors},
                       {"clusterVar": o["clusterVar"]} if o["clusterVar"] else {})
        notes: List[str] = []
        cols = [dep] + covs + [f for f in factors if f not in covs] + ([o["clusterVar"]] if o["clusterVar"] else [])
        data = df[list(dict.fromkeys(cols))].dropna()
        cats = _sorted_levels(data[dep])
        J = len(cats)
        if J < 3:
            raise ValueError(f"{dep}: 다항 종속변수는 범주 3개 이상 필요 (범주 {J}개 — 이항이면 binaryReg 사용)")
        if J > 30:
            raise ValueError(f"{dep}: 범주 {J}개 — 다항 모형 대상으로 보기 어려움 (30개 이하)")
        cmap = {c: i for i, c in enumerate(cats)}
        yj = data[dep].map(cmap).to_numpy(dtype=int)
        counts = np.bincount(yj, minlength=J)
        bo = o["baseOutcome"]
        if bo is None or (isinstance(bo, str) and not bo.strip()):
            base = int(np.argmax(counts))
            base_txt = "가장 빈도가 높은 범주 (Stata 기본)"
        else:
            match = [i for i, c in enumerate(cats) if _lvl_str(c) == _lvl_str(bo) or str(c) == str(bo)]
            if not match:
                try:
                    match = [i for i, c in enumerate(cats) if float(c) == float(bo)]
                except (TypeError, ValueError):
                    match = []
            if not match:
                raise ValueError(f"기준 범주(baseOutcome): '{bo}' 없음 (가능: {', '.join(_lvl_str(c) for c in cats)})")
            base = match[0]
            base_txt = "지정 (baseoutcome)"
        o["baseOutcome"] = None if base_txt.startswith("가장") else _lvl_str(cats[base])
        des = _Design(data, covs, factors, notes)
        des.drop_collinear(notes)
        X = des.X
        n, p = X.shape
        groups = _groups(data[o["clusterVar"]]) if o["covType"] == "cluster" else None
        nb = [j for j in range(J) if j != base]
        code = {c: i for i, c in enumerate([base] + nb)}            # statsmodels: code 0 = base outcome
        res = _sm_mnlogit(np.array([code[v] for v in yj]), X, o["covType"], groups)
        th = np.asarray(res.params, float).T.ravel()               # outcome-major: (outcome j, regressor)
        V, vx = _stata_cov(res, o["covType"], n, groups)
        fit = _FitInfo(float(res.llf), _converged(res), _iterations(res))
        se = np.sqrt(np.clip(np.diag(V), 0, None))
        inf = _Inference(o["ciWidth"])
        ll0 = float(np.sum(counts * np.log(counts / n)))
        npar = (J - 1) * p
        nb = [j for j in range(J) if j != base]
        slope_idx = [a * p + j for a in range(J - 1) for j in des.slope_idx]
        r2 = 1.0 - fit.ll / ll0
        aic = -2 * fit.ll + 2 * npar
        bic = -2 * fit.ll + npar * math.log(n)
        if o["covType"] == "oim":
            chi2, dfm, test = 2.0 * (fit.ll - ll0), len(slope_idx), "LR"
            pchi = float(stats.chi2.sf(chi2, dfm))
        else:
            chi2, dfm, pchi = _wald(th, V, slope_idx)
            test = "Wald"
        ll_lab = "로그우도 (Log likelihood)" if o["covType"] == "oim" else "로그 유사우도 (Log pseudolikelihood)"
        ft = _fit_table("fit", [("관측치 수 (N)", n, "integer"), ("범주 수", J, "integer"), (ll_lab, fit.ll, ""),
                                (f"{test} χ²", chi2, ""), ("자유도 (df)", dfm, "integer"), ("p", pchi, "pvalue"),
                                ("McFadden 유사 R² (Pseudo R2)", r2, "zto"), ("AIC", aic, ""), ("BIC", bic, "")])
        if not fit.converged:
            ft.add_note(f"수렴 실패: 반복 {fit.iterations}회 후 중단 — 추정치 해석 주의")
        self.add_table(ft)

        exp = bool(o["rrr"])
        ct = _coef_table("coef", f"계수 추정치 (Coefficients) — {dep}", inf, eq_col="결과 범주",
                         exp_title="상대위험비 (RRR)" if exp else None)
        for j in range(J):
            if j == base:
                row = {"eq": _lvl_str(cats[j]), "term": "(기준 범주 base outcome)", "b": "", "se": "", "z": "", "p": "",
                       "lo": "", "hi": ""}
                if exp:
                    row.update({"eb": "", "eb_se": "", "eb_lo": "", "eb_hi": ""})
                ct.add_row(row)
                continue
            a = nb.index(j)
            for c, nm in enumerate(des.names):
                ct.add_row(_coef_row(inf, nm, th[a * p + c], se[a * p + c], eq=_lvl_str(cats[j]), exp=exp))
        ct.add_note(f"기준 범주(base outcome): {dep}={_lvl_str(cats[base])} — {base_txt}")
        for nt in notes:
            ct.add_note(nt)
        if des.base_notes():
            ct.add_note("요인 기준 범주: " + ", ".join(des.base_notes()))
        ct.add_note(_vce_note(o["covType"], vx, o["clusterVar"]))
        if exp:
            ct.add_note("상대위험비(RRR) = exp(B), 표준오차는 delta 방법 (Stata mlogit, rrr)")
        self.add_table(_finish_coef(ct, exp, ["eq", "term"], check_separation=True))

        if o["ame"]:
            groups_f = des.factor_groups()
            mt = _me_table("ame", f"범주별 평균 한계효과 (AME) — Pr({dep}=j)", inf, outcome_col=True)
            terms = _me_terms(des.info, des.names)

            predict = lambda t, Xm: res.model.predict(np.asarray(t).reshape(J - 1, p).T, Xm)
            mv, ms = _margins_lib(res, X, des.info, groups_f, th, V, vx["factor"], False, predict,
                                  np.tile(_col_scale(X), J - 1))
            for k in range(J):
                for i, nm in enumerate(terms):
                    v, s_ = float(mv[i, code[k]]), float(ms[i, code[k]])
                    z, pv = inf.stat_p(v, s_)
                    lo, hi = inf.ci(v, s_)
                    mt.add_row({"outcome": _lvl_str(cats[k]), "term": nm, "b": v, "se": s_, "z": z, "p": pv,
                                "lo": lo, "hi": hi})
            mt.add_note(ME_NOTE_AME + " — 결과 범주별 (Stata margins, dydx(*) predict(outcome(j)))")
            mt.mark_coefficients(estimate="b", se="se", p="p", stat="z", ci=("lo", "hi"), label=["outcome", "term"])
            self.add_table(mt)

        self.model_stats = {"N": n, "LL": fit.ll, "Chi2": (chi2, pchi), "df": dfm, "R2_McF": r2, "AIC": aic, "BIC": bic}
        self.syntax_python = _sdk_call("multinomReg", o, {"baseOutcome": None, "rrr": False, "ame": False,
                                                          "covType": "oim", "clusterVar": None, "ciWidth": 95.0})
        self.syntax_r = "\n".join(_r_lines("multinomReg", dict(o, baseOutcome=_lvl_str(cats[base]))))
        self.apa_narrative = (f"다항 로짓 회귀모형(N = {n}, 범주 {J}개, 기준 범주 {_lvl_str(cats[base])})의 {test} "
                              f"χ²({dfm}) = {chi2:.2f}, p {_pstr(pchi)}, McFadden 유사 R² = {r2:.3f}.")




# =============================================================================================== censored / truncated
def _parse_limit(v: Any, which: str, y: np.ndarray) -> Optional[float]:
    if v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, bool) and not v):
        return None
    if isinstance(v, bool):
        return float(np.min(y)) if which == "ll" else float(np.max(y))
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("min", "ll", "minimum"):
            return float(np.min(y))
        if s in ("max", "ul", "maximum"):
            return float(np.max(y))
        if s in ("none", ".", "inf", "+inf", "-inf"):
            return None
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"{which}: 숫자 또는 '{'min' if which == 'll' else 'max'}' 필요 (입력: '{v}')")
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{which}: 숫자 또는 '{'min' if which == 'll' else 'max'}' 필요 (입력: {v!r})")
    return None if not np.isfinite(f) else f


def _limit_txt(L: Optional[float], U: Optional[float]) -> str:
    return f"하한 = {('−inf' if L is None else f'{L:g}')}, 상한 = {('+inf' if U is None else f'{U:g}')}"


def _ols_start(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    s = math.log(max(float(np.std(r)), 1e-8))
    return np.concatenate([b, [s]])


def _tobit_parts(theta: np.ndarray, X: np.ndarray, y: np.ndarray, L: Optional[float], U: Optional[float], want: str):
    k = X.shape[1]
    b, s = theta[:k], theta[k]
    sig = math.exp(s)
    eta = X @ b
    cl = np.zeros(len(y), bool) if L is None else (y <= L)
    cu = np.zeros(len(y), bool) if U is None else (y >= U)
    un = ~(cl | cu)
    z = (y - eta) / sig
    if want == "ll":
        ll = np.empty(len(y))
        ll[un] = -0.5 * z[un] ** 2 - s - 0.5 * LOG_2PI
        if L is not None:
            ll[cl] = special.log_ndtr((L - eta[cl]) / sig)
        if U is not None:
            ll[cu] = special.log_ndtr(-(U - eta[cu]) / sig)
        return ll
    de = np.zeros(len(y))
    ds = np.zeros(len(y))
    de[un] = z[un] / sig
    ds[un] = z[un] ** 2 - 1.0
    if L is not None and cl.any():
        a = (L - eta[cl]) / sig
        m = np.exp(_log_phi(a) - special.log_ndtr(a))
        de[cl] = -m / sig
        ds[cl] = -m * a
    if U is not None and cu.any():
        bb = (U - eta[cu]) / sig
        m = np.exp(_log_phi(bb) - special.log_ndtr(-bb))
        de[cu] = m / sig
        ds[cu] = m * bb
    return np.column_stack([X * de[:, None], ds])


def _log_diff_ndtr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """log(Phi(b) - Phi(a)) for a < b, stable in both tails."""
    out = np.empty(np.broadcast(a, b).shape)
    a = np.broadcast_to(a, out.shape)
    b = np.broadcast_to(b, out.shape)
    up = a > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        la, lb = special.log_ndtr(-a[up]), special.log_ndtr(-b[up])
        out[up] = la + np.log1p(-np.exp(lb - la))
        la, lb = special.log_ndtr(a[~up]), special.log_ndtr(b[~up])
        out[~up] = lb + np.log1p(-np.exp(la - lb))
    return out


def _trunc_parts(theta: np.ndarray, X: np.ndarray, y: np.ndarray, L: Optional[float], U: Optional[float], want: str):
    k = X.shape[1]
    b, s = theta[:k], theta[k]
    sig = math.exp(s)
    eta = X @ b
    z = (y - eta) / sig
    a = np.full(len(y), -np.inf) if L is None else (L - eta) / sig
    bb = np.full(len(y), np.inf) if U is None else (U - eta) / sig
    logD = _log_diff_ndtr(a, bb)
    if want == "ll":
        return -0.5 * z ** 2 - s - 0.5 * LOG_2PI - logD
    ra = np.where(np.isfinite(a), np.exp(_log_phi(np.where(np.isfinite(a), a, 0.0)) - logD), 0.0)
    rb = np.where(np.isfinite(bb), np.exp(_log_phi(np.where(np.isfinite(bb), bb, 0.0)) - logD), 0.0)
    de = z / sig + (rb - ra) / sig
    ds = z ** 2 - 1.0 + np.where(np.isfinite(bb), bb, 0.0) * rb - np.where(np.isfinite(a), a, 0.0) * ra
    return np.column_stack([X * de[:, None], ds])


class _TobitModel(_LLModel):
    """Censored-normal regression (Stata tobit) as a statsmodels GenericLikelihoodModel; params (beta, ln sigma)."""

    def __init__(self, y, X, L, U, step_scale):
        super().__init__(y, X, ["lnsigma"], step_scale)
        self.L, self.U = L, U

    def loglikeobs(self, params):
        return _tobit_parts(params, self.exog, self.endog, self.L, self.U, "ll")

    def score_obs(self, params):
        return _tobit_parts(params, self.exog, self.endog, self.L, self.U, "score")


class _TruncModel(_TobitModel):
    """Truncated-normal regression (Stata truncreg); params (beta, ln sigma)."""

    def loglikeobs(self, params):
        return _trunc_parts(params, self.exog, self.endog, self.L, self.U, "ll")

    def score_obs(self, params):
        return _trunc_parts(params, self.exog, self.endog, self.L, self.U, "score")


def _tobit_pred(kind: str, L: Optional[float], U: Optional[float], k: int):
    """Prediction (and d/d eta) after tobit: xb, e(ll,ul), pr(ll,ul), ystar(ll,ul)."""
    def fn(theta, Xm):
        eta = Xm @ theta[:k]
        sig = math.exp(theta[k])
        a = np.full(eta.shape, -np.inf) if L is None else (L - eta) / sig
        b = np.full(eta.shape, np.inf) if U is None else (U - eta) / sig
        Fa, Fb = special.ndtr(a), special.ndtr(b)
        fa = np.where(np.isfinite(a), np.exp(_log_phi(np.where(np.isfinite(a), a, 0.0))), 0.0)
        fb = np.where(np.isfinite(b), np.exp(_log_phi(np.where(np.isfinite(b), b, 0.0))), 0.0)
        afa = np.where(np.isfinite(a), a, 0.0) * fa
        bfb = np.where(np.isfinite(b), b, 0.0) * fb
        D = np.exp(_log_diff_ndtr(a, b))
        if kind == "xb":
            return eta, np.ones_like(eta)
        if kind == "pr":
            return D, (fa - fb) / sig
        if kind == "e":
            Nn = fa - fb
            return eta + sig * Nn / D, 1.0 + (afa - bfb) / D - (Nn / D) ** 2
        val = D * eta + sig * (fa - fb)
        if L is not None:
            val = val + L * Fa
        if U is not None:
            val = val + U * (1.0 - Fb)
        return val, D
    return fn


class TobitReg(BaseAnalysis):
    """Censored-normal regression (Stata tobit, ll()/ul())."""

    def __init__(self, data: pd.DataFrame, dep: str, covs=None, factors=None, ll: Any = None, ul: Any = None,
                 me_latent: bool = False, me_conditional: bool = False, me_prob: bool = False,
                 me_censored: bool = False, cov_type: str = "oim", cluster: Optional[str] = None,
                 ci_width: float = 95.0):
        cov_type = resolve_choice("표준오차 유형(covType)", cov_type, COV_TYPES, _COV_ALIASES, default="oim")
        covs, factors = _as_list(covs), _as_list(factors)
        if cov_type == "cluster" and not cluster:
            raise ValueError("군집 강건 표준오차(covType='cluster'): 군집 변수(clusterVar) 지정 필요")
        cluster = cluster if cov_type == "cluster" else None       # unused otherwise (no listwise deletion on it)
        super().__init__(name="tobitReg", title="토빗 회귀 (tobit)", dataset=data, options={
            "dep": dep, "covs": covs, "factors": factors, "ll": ll, "ul": ul, "meLatent": bool(me_latent),
            "meConditional": bool(me_conditional), "meProb": bool(me_prob), "meCensored": bool(me_censored),
            "covType": cov_type, "clusterVar": cluster, "ciWidth": float(ci_width)})
        self.run()

    def run(self):
        o = self.options
        df, dep, covs, factors = self.dataset, o["dep"], o["covs"], o["factors"]
        _check_columns(df, self.title, {"dep": dep, "covs/factors": covs + factors},
                       {"clusterVar": o["clusterVar"]} if o["clusterVar"] else {})
        notes: List[str] = []
        cols = [dep] + covs + [f for f in factors if f not in covs] + ([o["clusterVar"]] if o["clusterVar"] else [])
        data = df[list(dict.fromkeys(cols))].dropna()
        y = pd.to_numeric(data[dep], errors="coerce")
        if y.isna().any():
            raise ValueError(f"{dep}: 숫자형 종속변수 필요")
        y = y.to_numpy(dtype=float)
        L = _parse_limit(o["ll"], "ll", y)
        U = _parse_limit(o["ul"], "ul", y)
        if L is None and U is None:
            raise ValueError("토빗: 검열 하한 ll 또는 상한 ul 지정 필요 (예: ll=0, 또는 ll='min')")
        if L is not None and U is not None and not L < U:
            raise ValueError(f"토빗: ll({L:g}) < ul({U:g}) 필요")
        des = _Design(data, covs, factors, notes)
        des.drop_collinear(notes)
        X = des.X
        n, k = X.shape
        cl = np.zeros(n, bool) if L is None else (y <= L)
        cu = np.zeros(n, bool) if U is None else (y >= U)
        if (~(cl | cu)).sum() == 0:
            raise ValueError("토빗: 미검열 관측치 없음 — 추정 불가")
        start = _ols_start(X, y)
        tscale = np.concatenate([_col_scale(X, math.exp(start[-1])), [1.0]])
        groups = _groups(data[o["clusterVar"]]) if o["covType"] == "cluster" else None
        res = _mle(_TobitModel(y, X, L, U, tscale), start, o["covType"], groups)
        th = np.asarray(res.params, float)
        V, vx = _stata_cov(res, o["covType"], n, groups)
        fit = _FitInfo(float(res.llf), _converged(res), _iterations(res))
        # Stata tobit: e(df_r) = N - rank(e(V)) + 2, i.e. N - (number of slopes) with a full-rank VCE
        # (checked: 993 with N = 1000 and 7 slopes); a cluster VCE with few clusters has a lower rank
        dV = np.sqrt(np.clip(np.diag(V), 1e-300, None))
        rank = int(np.linalg.matrix_rank(V / np.outer(dV, dV), tol=1e-8))
        df_r = n - rank + 2
        inf = _Inference(o["ciWidth"], df_r=df_r)
        se = np.sqrt(np.clip(np.diag(V), 0, None))
        ones = np.ones((n, 1))
        start0 = _ols_start(ones, y)
        ll0 = float(_mle(_TobitModel(y, ones, L, U, np.array([math.exp(start0[-1]), 1.0])), start0, "oim", None).llf)
        slopes = des.slope_idx
        npar = k + 1
        r2 = 1.0 - fit.ll / ll0
        aic = -2 * fit.ll + 2 * npar
        bic = -2 * fit.ll + npar * math.log(n)
        rows = [("관측치 수 (N)", n, "integer"), ("미검열 (Uncensored)", int((~(cl | cu)).sum()), "integer"),
                ("좌측 검열 (Left-censored)", int(cl.sum()), "integer"),
                ("우측 검열 (Right-censored)", int(cu.sum()), "integer")]
        if o["covType"] == "oim":
            chi2, dfm = 2.0 * (fit.ll - ll0), len(slopes)
            pchi = float(stats.chi2.sf(chi2, dfm)) if dfm else float("nan")
            rows += [("로그우도 (Log likelihood)", fit.ll, ""), ("LR χ²", chi2, ""), ("자유도 (df)", dfm, "integer"),
                     ("p", pchi, "pvalue")]
            test_txt = f"LR χ²({dfm}) = {chi2:.2f}, p {_pstr(pchi)}"
            self.model_stats = {"Chi2": (chi2, pchi), "df": dfm}
        else:
            W, dfm, _ = _wald(th, V, slopes)
            F = W / dfm
            pF = float(stats.f.sf(F, dfm, df_r))
            rows += [("로그 유사우도 (Log pseudolikelihood)", fit.ll, ""), ("F", F, ""), ("자유도 1", dfm, "integer"),
                     ("자유도 2", df_r, "integer"), ("p", pF, "pvalue")]
            test_txt = f"F({dfm}, {df_r}) = {F:.2f}, p {_pstr(pF)}"
            self.model_stats = {"F": (F, pF), "df": dfm}
        rows += [("McFadden 유사 R² (Pseudo R2)", r2, "zto"), ("AIC", aic, ""), ("BIC", bic, "")]
        ft = _fit_table("fit", rows)
        ft.add_note(f"검열 한계: {_limit_txt(L, U)} (y ≤ 하한 → 좌측 검열, y ≥ 상한 → 우측 검열)")
        if not fit.converged:
            ft.add_note(f"수렴 실패: 반복 {fit.iterations}회 후 중단 — 추정치 해석 주의")
        self.add_table(ft)

        ct = _coef_table("coef", f"계수 추정치 (Coefficients) — {dep}", inf)
        for j, nm in enumerate(des.names):
            ct.add_row(_coef_row(inf, nm, th[j], se[j]))
        sig = math.exp(th[k])
        se_ls = se[k]
        q = inf.q
        ct.add_row({"term": f"var(e.{dep})", "b": sig ** 2, "se": 2 * sig ** 2 * se_ls, "z": "", "p": "",
                    "lo": sig ** 2 * math.exp(-2 * q * se_ls), "hi": sig ** 2 * math.exp(2 * q * se_ls)})
        ct.add_row({"term": "/sigma", "b": sig, "se": sig * se_ls, "z": "", "p": "",
                    "lo": sig * math.exp(-q * se_ls), "hi": sig * math.exp(q * se_ls)})
        ct.add_note(f"t 통계량·p·CI: 자유도 {df_r} t 분포 (Stata tobit은 t 보고)")
        ct.add_note("var(e.y)·sigma: 표준오차는 delta 방법, CI는 로그 척도에서 구한 뒤 변환")
        for nt in notes:
            ct.add_note(nt)
        if des.base_notes():
            ct.add_note("요인 기준 범주: " + ", ".join(des.base_notes()))
        ct.add_note(_vce_note(o["covType"], vx, o["clusterVar"]))
        self.add_table(_finish_coef(ct, False, "term"))

        groups_f = des.factor_groups()
        inf_me = _Inference(o["ciWidth"])          # Stata margins after tobit reports z
        lims = f"{'' if L is None else f'{L:g}'},{'.' if U is None else f'{U:g}'}"
        if L is None:
            lims = f".,{U:g}"
        specs = [("meLatent", "xb", "me_xb", f"한계효과 — 잠재변수 E({dep}*) = xb (margins, dydx(*) predict(xb))"),
                 ("meConditional", "e", "me_cond", f"한계효과 — 조건부 기대값 E({dep} | 검열 범위 안) (predict(e({lims})))"),
                 ("meProb", "pr", "me_pr", f"한계효과 — 미검열 확률 Pr(검열 범위 안) (predict(pr({lims})))"),
                 ("meCensored", "ystar", "me_ystar", f"한계효과 — 관측 결과 기대값 E({dep}) (predict(ystar({lims})))")]
        for key, kind, name, title in specs:
            if not o[key]:
                continue
            pred = _tobit_pred(kind, L, U, k)
            fn = lambda t, pr=pred: _me_effects(t, X, des.info, groups_f, pr, lambda tt, j: tt[j], False)
            me = fn(th)
            Jm = _num_jac(fn, th, tscale)
            mse = np.sqrt(np.clip(np.diag(Jm @ V @ Jm.T), 0, None))
            mt = _me_table(name, title, inf_me)
            for nm, v, s_ in zip(_me_terms(des.info, des.names), me, mse):
                z, pv = inf_me.stat_p(v, s_)
                lo, hi = inf_me.ci(v, s_)
                mt.add_row({"term": nm, "b": v, "se": s_, "z": z, "p": pv, "lo": lo, "hi": hi})
            mt.add_note(ME_NOTE_AME)
            mt.mark_coefficients(estimate="b", se="se", p="p", stat="z", ci=("lo", "hi"), label="term")
            self.add_table(mt)

        self.model_stats.update({"N": n, "LL": fit.ll, "R2_McF": r2, "AIC": aic, "BIC": bic})
        self.syntax_python = _sdk_call("tobitReg", o, {"meLatent": False, "meConditional": False, "meProb": False,
                                                       "meCensored": False, "covType": "oim", "clusterVar": None,
                                                       "ciWidth": 95.0})
        self.syntax_r = "\n".join(_r_lines("tobitReg", o))
        self.apa_narrative = (f"토빗 회귀모형(N = {n}, 좌측 검열 {int(cl.sum())}개, 우측 검열 {int(cu.sum())}개)의 "
                              f"{test_txt}, McFadden 유사 R² = {r2:.3f}, σ = {sig:.3f}.")


class TruncReg(BaseAnalysis):
    """Truncated-normal regression (Stata truncreg, ll()/ul())."""

    def __init__(self, data: pd.DataFrame, dep: str, covs=None, factors=None, ll: Any = None, ul: Any = None,
                 cov_type: str = "oim", cluster: Optional[str] = None, ci_width: float = 95.0):
        cov_type = resolve_choice("표준오차 유형(covType)", cov_type, COV_TYPES, _COV_ALIASES, default="oim")
        covs, factors = _as_list(covs), _as_list(factors)
        if cov_type == "cluster" and not cluster:
            raise ValueError("군집 강건 표준오차(covType='cluster'): 군집 변수(clusterVar) 지정 필요")
        cluster = cluster if cov_type == "cluster" else None       # unused otherwise (no listwise deletion on it)
        super().__init__(name="truncReg", title="절단 회귀 (truncreg)", dataset=data, options={
            "dep": dep, "covs": covs, "factors": factors, "ll": ll, "ul": ul, "covType": cov_type,
            "clusterVar": cluster, "ciWidth": float(ci_width)})
        self.run()

    def run(self):
        o = self.options
        df, dep, covs, factors = self.dataset, o["dep"], o["covs"], o["factors"]
        _check_columns(df, self.title, {"dep": dep, "covs/factors": covs + factors},
                       {"clusterVar": o["clusterVar"]} if o["clusterVar"] else {})
        notes: List[str] = []
        cols = [dep] + covs + [f for f in factors if f not in covs] + ([o["clusterVar"]] if o["clusterVar"] else [])
        data = df[list(dict.fromkeys(cols))].dropna()
        y_all = pd.to_numeric(data[dep], errors="coerce")
        if y_all.isna().any():
            raise ValueError(f"{dep}: 숫자형 종속변수 필요")
        y_all = y_all.to_numpy(dtype=float)
        L = _parse_limit(o["ll"], "ll", y_all)
        U = _parse_limit(o["ul"], "ul", y_all)
        if L is None and U is None:
            raise ValueError("절단 회귀: 절단 하한 ll 또는 상한 ul 지정 필요 (예: ll=0)")
        if L is not None and U is not None and not L < U:
            raise ValueError(f"절단 회귀: ll({L:g}) < ul({U:g}) 필요")
        inside = np.ones(len(y_all), bool)
        if L is not None:
            inside &= y_all > L
        if U is not None:
            inside &= y_all < U
        ntr = int((~inside).sum())
        data = data[inside]
        y = y_all[inside]
        if len(y) == 0:
            raise ValueError("절단 회귀: 절단 범위 안 관측치 없음")
        des = _Design(data, covs, factors, notes)
        des.drop_collinear(notes)
        X = des.X
        n, k = X.shape
        start = _ols_start(X, y)
        groups = _groups(data[o["clusterVar"]]) if o["covType"] == "cluster" else None
        res = _mle(_TruncModel(y, X, L, U, np.concatenate([_col_scale(X, math.exp(start[-1])), [1.0]])), start,
                   o["covType"], groups)
        th = np.asarray(res.params, float)
        V, vx = _stata_cov(res, o["covType"], n, groups)
        fit = _FitInfo(float(res.llf), _converged(res), _iterations(res))
        sig = math.exp(th[k])
        J = np.eye(k + 1)
        J[k, k] = sig
        Vs = J @ V @ J.T               # covariance of (beta, sigma)
        se = np.sqrt(np.clip(np.diag(Vs), 0, None))
        inf = _Inference(o["ciWidth"])
        W, dfm, pW = _wald(th, V, des.slope_idx)
        npar = k + 1
        aic = -2 * fit.ll + 2 * npar
        bic = -2 * fit.ll + npar * math.log(n)
        ll_lab = "로그우도 (Log likelihood)" if o["covType"] == "oim" else "로그 유사우도 (Log pseudolikelihood)"
        ft = _fit_table("fit", [("관측치 수 (N)", n, "integer"), ("절단으로 제외된 관측치", ntr, "integer"),
                                (ll_lab, fit.ll, ""), ("Wald χ²", W, ""), ("자유도 (df)", dfm, "integer"),
                                ("p", pW, "pvalue"), ("AIC", aic, ""), ("BIC", bic, "")])
        ft.add_note(f"절단 한계: {_limit_txt(L, U)} — 범위 밖(y ≤ 하한 또는 y ≥ 상한) 관측치 {ntr}개 제외 "
                    f"(Stata: {ntr} obs. truncated)")
        if not fit.converged:
            ft.add_note(f"수렴 실패: 반복 {fit.iterations}회 후 중단 — 추정치 해석 주의")
        self.add_table(ft)
        ct = _coef_table("coef", f"계수 추정치 (Coefficients) — {dep}", inf)
        for j, nm in enumerate(des.names):
            ct.add_row(_coef_row(inf, nm, th[j], se[j]))
        ct.add_row(_coef_row(inf, "/sigma", sig, se[k]))
        for nt in notes:
            ct.add_note(nt)
        if des.base_notes():
            ct.add_note("요인 기준 범주: " + ", ".join(des.base_notes()))
        ct.add_note(_vce_note(o["covType"], vx, o["clusterVar"]))
        self.add_table(_finish_coef(ct, False, "term"))
        self.model_stats = {"N": n, "LL": fit.ll, "Chi2": (W, pW), "df": dfm, "AIC": aic, "BIC": bic}
        self.syntax_python = _sdk_call("truncReg", o, {"covType": "oim", "clusterVar": None, "ciWidth": 95.0})
        self.syntax_r = "\n".join(_r_lines("truncReg", o))
        self.apa_narrative = (f"절단 회귀모형(N = {n}, 절단 제외 {ntr}개)의 Wald χ²({dfm}) = {W:.2f}, p {_pstr(pW)}, "
                              f"σ = {sig:.3f}.")


# =============================================================================================== sample selection
def _bvn_upper(dh: np.ndarray, dk: np.ndarray, r: float) -> np.ndarray:
    """P(X > dh, Y > dk) for a standard bivariate normal with correlation r (Genz 2004, TVPACK BVND), vectorised."""
    h = np.asarray(dh, float)
    k = np.asarray(dk, float).copy()
    if r == 0.0:
        return special.ndtr(-h) * special.ndtr(-k)
    ng = 6 if abs(r) < 0.3 else (12 if abs(r) < 0.75 else 20)
    xg, wg = np.polynomial.legendre.leggauss(ng)
    tp = 2.0 * math.pi
    hk = h * k
    if abs(r) < 0.925:
        hs = (h * h + k * k) / 2.0
        asr = math.asin(r)
        sn = np.sin(asr * (xg + 1.0) / 2.0)
        e = np.exp((np.outer(hk, sn) - hs[:, None]) / (1.0 - sn * sn)[None, :])
        return np.clip(e @ wg * asr / (2.0 * tp) + special.ndtr(-h) * special.ndtr(-k), 0.0, 1.0)
    if r < 0:
        k = -k
        hk = -hk
    bvn = np.zeros_like(h)
    if abs(r) < 1:
        as_ = (1.0 - r) * (1.0 + r)
        a = math.sqrt(as_)
        bs = (h - k) ** 2
        c = (4.0 - hk) / 8.0
        d = (12.0 - hk) / 16.0
        with np.errstate(over="ignore", invalid="ignore"):
            bvn = a * np.exp(-(bs / as_ + hk) / 2.0) * (1.0 - c * (bs - as_) * (1.0 - d * bs / 5.0) / 3.0
                                                       + c * d * as_ * as_ / 5.0)
            b = np.sqrt(bs)
            term = np.exp(-hk / 2.0) * math.sqrt(tp) * special.ndtr(-b / a) * b * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0)
            bvn = bvn - np.where(hk > -160, term, 0.0)
            a2 = a / 2.0
            for xi, wi in zip(xg, wg):
                xs = (a2 * (xi + 1.0)) ** 2
                rs = math.sqrt(1.0 - xs)
                add = a2 * wi * (np.exp(-bs / (2.0 * xs) - hk / (1.0 + rs)) / rs
                                 - np.exp(-(bs / xs + hk) / 2.0) * (1.0 + c * xs * (1.0 + d * xs)))
                bvn = bvn + np.where(np.isfinite(add), add, 0.0)
        bvn = -bvn / tp
    if r > 0:
        bvn = bvn + special.ndtr(-np.maximum(h, k))
    else:
        bvn = -bvn + np.maximum(0.0, special.ndtr(-h) - special.ndtr(-k))
    return np.clip(bvn, 0.0, 1.0)


def _bvn_cdf(a: np.ndarray, b: np.ndarray, r: float) -> np.ndarray:
    """Phi2(a, b; r) = P(X < a, Y < b)."""
    return _bvn_upper(-np.asarray(a, float), -np.asarray(b, float), r)


def _heck_parts(theta, X, Z, y, s, want):
    kx, kz = X.shape[1], Z.shape[1]
    b, g, tau, ls = theta[:kx], theta[kx:kx + kz], theta[kx + kz], theta[kx + kz + 1]
    rho, sig, r1 = math.tanh(tau), math.exp(ls), 1.0 / math.cosh(tau)
    zg = Z @ g
    sel = s > 0.5
    e = (y[sel] - X[sel] @ b) / sig
    A = (zg[sel] + rho * e) / r1
    if want == "ll":
        ll = np.empty(len(s))
        ll[~sel] = special.log_ndtr(-zg[~sel])
        ll[sel] = special.log_ndtr(A) - 0.5 * e * e - ls - 0.5 * LOG_2PI
        return ll
    n = len(s)
    S = np.zeros((n, kx + kz + 2))
    lam0 = np.exp(_log_phi(zg[~sel]) - special.log_ndtr(-zg[~sel]))
    S[~sel, kx:kx + kz] = -Z[~sel] * lam0[:, None]
    m = np.exp(_log_phi(A) - special.log_ndtr(A))
    S[sel, :kx] = X[sel] * ((e - m * rho / r1) / sig)[:, None]
    S[sel, kx:kx + kz] = Z[sel] * (m / r1)[:, None]
    S[sel, kx + kz] = m * (e + rho * zg[sel]) / r1
    S[sel, kx + kz + 1] = -m * rho * e / r1 + e * e - 1.0
    return S


def _heckprob_parts(theta, X, Z, y, s, want):
    kx, kz = X.shape[1], Z.shape[1]
    b, g, tau = theta[:kx], theta[kx:kx + kz], theta[kx + kz]
    rho = math.tanh(tau)
    zg = Z @ g
    xb = X @ b
    sel = s > 0.5
    n = len(s)
    q = 2.0 * y - 1.0
    ll = np.zeros(n)
    ll[~sel] = special.log_ndtr(-zg[~sel])
    P = np.ones(n)
    for qq in (1.0, -1.0):
        m = sel & (q == qq)
        if m.any():
            P[m] = _bvn_cdf(qq * xb[m], zg[m], qq * rho)
    with np.errstate(divide="ignore"):
        ll[sel] = np.log(P[sel])
    if want == "ll":
        return ll
    S = np.zeros((n, kx + kz + 1))
    lam0 = np.exp(_log_phi(zg[~sel]) - special.log_ndtr(-zg[~sel]))
    S[~sel, kx:kx + kz] = -Z[~sel] * lam0[:, None]
    a = q[sel] * xb[sel]
    bb = zg[sel]
    r = q[sel] * rho
    sr = np.sqrt(1.0 - r * r)
    Ps = P[sel]
    w1 = np.exp(_log_phi(a)) * special.ndtr((bb - r * a) / sr) / Ps
    w2 = np.exp(_log_phi(bb)) * special.ndtr((a - r * bb) / sr) / Ps
    w3 = np.exp(-(a * a - 2 * r * a * bb + bb * bb) / (2 * (1 - r * r))) / (2 * math.pi * sr) / Ps
    S[sel, :kx] = X[sel] * (q[sel] * w1)[:, None]
    S[sel, kx:kx + kz] = Z[sel] * w2[:, None]
    S[sel, kx + kz] = q[sel] * w3 * (1.0 - rho * rho)
    return S


class _HeckmanModel(_LLModel):
    """Heckman selection model (Stata heckman, ML) as a statsmodels GenericLikelihoodModel;
    params (beta, gamma, athrho, ln sigma)."""

    def __init__(self, y, X, Z, s, step_scale, znames):
        super().__init__(y, X, list(znames) + ["athrho", "lnsigma"], step_scale)
        self.Z, self.s = Z, s

    def loglikeobs(self, params):
        return _heck_parts(params, self.exog, self.Z, self.endog, self.s, "ll")

    def score_obs(self, params):
        return _heck_parts(params, self.exog, self.Z, self.endog, self.s, "score")


class _HeckProbitModel(_HeckmanModel):
    """Probit with sample selection (Stata heckprobit); params (beta, gamma, athrho)."""

    def __init__(self, y, X, Z, s, step_scale, znames):
        _LLModel.__init__(self, y, X, list(znames) + ["athrho"], step_scale)
        self.Z, self.s = Z, s

    def loglikeobs(self, params):
        return _heckprob_parts(params, self.exog, self.Z, self.endog, self.s, "ll")

    def score_obs(self, params):
        return _heckprob_parts(params, self.exog, self.Z, self.endog, self.s, "score")


class _SelectionData:
    """Estimation sample for selection models: selection rule, outcome and selection designs."""

    def __init__(self, df, label, dep, covs, factors, sdep, scovs, sfactors, cluster, notes):
        _check_columns(df, label, {"dep": dep, "covs/factors": covs + factors, "selectCovs": scovs + sfactors},
                       {k: v for k, v in (("selectDep", sdep), ("clusterVar", cluster)) if v})
        svars = list(dict.fromkeys(scovs + [f for f in sfactors if f not in scovs] + ([sdep] if sdep else [])
                                   + ([cluster] if cluster else [])))
        data = df[list(dict.fromkeys(svars + [dep] + covs + factors))].copy()
        data = data[data[svars].notna().all(axis=1)]
        if sdep:
            sv = pd.to_numeric(data[sdep], errors="coerce")
            if sv.isna().any():
                raise ValueError(f"{sdep}: 숫자형 선택 지표 필요 (0 = 미선택, 0이 아닌 값 = 선택)")
            s = (sv != 0).to_numpy(dtype=float)
            self.rule = f"{sdep} ≠ 0 이면 선택 (Stata select({sdep} = ...))"
            self.sname = sdep
        else:
            s = data[dep].notna().to_numpy(dtype=float)
            self.rule = f"{dep} 비결측이면 선택 (Stata select(varlist))"
            self.sname = "select"
        out_vars = [dep] + covs + factors
        bad = (s > 0.5) & data[out_vars].isna().any(axis=1).to_numpy()
        if bad.any():
            notes.append(f"선택된 관측치 중 결과식 변수 결측 {int(bad.sum())}개 제외")
            data, s = data[~bad], s[~bad]
        self.data, self.s = data, s
        sel = s > 0.5
        if sel.sum() == 0 or sel.sum() == len(s):
            raise ValueError(f"{label}: 선택·미선택 관측치가 모두 있어야 함 (선택 {int(sel.sum())}, 미선택 {int((~sel).sum())})")
        filled = data.copy()
        for c in out_vars:
            if pd.api.types.is_numeric_dtype(filled[c]):
                filled.loc[~sel, c] = filled.loc[~sel, c].fillna(0)
        self.dx = _Design(filled, covs, factors, notes, levels_from=data[sel])
        self.dx.drop_collinear(notes, rows=sel)
        self.dz = _Design(data, scovs, sfactors, notes)
        self.dz.drop_collinear(notes)
        yv = pd.to_numeric(filled[dep], errors="coerce").to_numpy(dtype=float)
        yv[~sel] = 0.0
        if np.isnan(yv[sel]).any():
            raise ValueError(f"{dep}: 숫자형 종속변수 필요")
        self.y = yv
        self.X = np.nan_to_num(self.dx.X)
        self.Z = self.dz.X
        self.groups = _groups(data[cluster]) if cluster else None


def _heckman_twostep(sd: _SelectionData):
    X, Z, y, s = sd.X, sd.Z, sd.y, sd.s
    sel = s > 0.5
    pf = _sm_binary(s, Z, "probit")                       # step 1: statsmodels Probit (selection)
    g = np.asarray(pf.params, float)
    Vg = np.asarray(pf.cov_params(), float)
    zg = Z @ g
    lam = np.exp(_log_phi(zg) - special.log_ndtr(zg))     # inverse Mills ratio
    delta = lam * (lam + zg)
    Xs = np.column_stack([X[sel], lam[sel]])
    ys = y[sel]
    ols = sm.OLS(ys, Xs).fit()                            # step 2: statsmodels OLS with the Mills ratio
    coef = np.asarray(ols.params, float)
    e = ys - Xs @ coef
    n1 = int(sel.sum())
    bl = coef[-1]
    sig2 = float(e @ e) / n1 + bl * bl * float(np.mean(delta[sel]))
    sig = math.sqrt(sig2)
    rho = bl / sig
    truncated = None
    if abs(rho) > 1:
        truncated = rho
        rho = float(np.sign(rho))
        sig = bl / rho
    D = delta[sel]
    XtX_inv = np.linalg.inv(Xs.T @ Xs)
    Zs = Z[sel]
    Q = rho * rho * (Xs.T * D) @ Zs @ Vg @ Zs.T @ (Xs * D[:, None])
    V = sig * sig * XtX_inv @ ((Xs.T * (1.0 - rho * rho * D)) @ Xs + Q) @ XtX_inv
    return {"beta": coef[:-1], "bl": bl, "V": V, "gamma": g, "Vg": Vg, "sigma": sig, "rho": rho,
            "truncated": truncated, "probit_ll": float(pf.llf), "n1": n1}


class HeckmanSel(BaseAnalysis):
    """Heckman sample-selection model: ML (Stata heckman) or Heckman two-step (heckman, twostep)."""

    def __init__(self, data: pd.DataFrame, dep: str, covs=None, factors=None, select_dep: Optional[str] = None,
                 select_covs=None, select_factors=None, method: str = "ml", cov_type: str = "oim",
                 cluster: Optional[str] = None, ci_width: float = 95.0):
        method = resolve_choice("추정 방법(method)", method, ("ml", "twostep"),
                                {"mle": "ml", "2step": "twostep", "two-step": "twostep", "heckit": "twostep"}, "ml")
        cov_type = resolve_choice("표준오차 유형(covType)", cov_type, COV_TYPES, _COV_ALIASES, default="oim")
        if method == "twostep" and cov_type != "oim":
            raise ValueError("heckman twostep: 강건·군집 표준오차 미지원 (Stata도 twostep에서는 vce(conventional)·"
                             "bootstrap·jackknife만 허용) — covType='oim' 사용")
        if cov_type == "cluster" and not cluster:
            raise ValueError("군집 강건 표준오차(covType='cluster'): 군집 변수(clusterVar) 지정 필요")
        cluster = cluster if cov_type == "cluster" else None       # unused otherwise (no listwise deletion on it)
        covs, factors = _as_list(covs), _as_list(factors)
        scovs, sfactors = _as_list(select_covs), _as_list(select_factors)
        super().__init__(name="heckman", title="Heckman 표본선택 모형 (heckman)" + (" — 2단계 추정" if method == "twostep" else ""),
                         dataset=data, options={
                             "dep": dep, "covs": covs, "factors": factors, "selectDep": select_dep,
                             "selectCovs": scovs, "selectFactors": sfactors, "method": method, "covType": cov_type,
                             "clusterVar": cluster, "ciWidth": float(ci_width)})
        self.run()

    def run(self):
        o = self.options
        notes: List[str] = []
        sd = _SelectionData(self.dataset, self.title, o["dep"], o["covs"], o["factors"], o["selectDep"],
                            o["selectCovs"], o["selectFactors"], o["clusterVar"] if o["covType"] == "cluster" else None,
                            notes)
        X, Z, y, s = sd.X, sd.Z, sd.y, sd.s
        n, kx, kz = len(s), X.shape[1], Z.shape[1]
        n1 = int(s.sum())
        inf = _Inference(o["ciWidth"])
        dep = o["dep"]
        eq_out, eq_sel = dep, sd.sname
        two = _heckman_twostep(sd)
        ct = _coef_table("coef", f"계수 추정치 (Coefficients) — 결과식 {dep}, 선택식 {sd.sname}", inf, eq_col="식")
        fit_rows = [("관측치 수 (N)", n, "integer"), ("선택 (Selected)", n1, "integer"),
                    ("미선택 (Nonselected)", n - n1, "integer")]
        if o["method"] == "twostep":
            V = two["V"]
            se = np.sqrt(np.clip(np.diag(V), 0, None))
            seg = np.sqrt(np.clip(np.diag(two["Vg"]), 0, None))
            W, dfm, pW = _wald(two["beta"], V[:kx, :kx], sd.dx.slope_idx)
            fit_rows += [("Wald χ²", W, ""), ("자유도 (df)", dfm, "integer"), ("p", pW, "pvalue")]
            for j, nm in enumerate(sd.dx.names):
                ct.add_row(_coef_row(inf, nm, two["beta"][j], se[j], eq=eq_out))
            for j, nm in enumerate(sd.dz.names):
                ct.add_row(_coef_row(inf, nm, two["gamma"][j], seg[j], eq=eq_sel))
            ct.add_row(_coef_row(inf, "lambda", two["bl"], se[kx], eq="/mills"))
            ct.add_row({"eq": "", "term": "rho", "b": two["rho"], "se": "", "z": "", "p": "", "lo": "", "hi": ""})
            ct.add_row({"eq": "", "term": "sigma", "b": two["sigma"], "se": "", "z": "", "p": "", "lo": "", "hi": ""})
            ct.add_note("2단계 추정(Heckman 1979): 1단계 probit 선택식 → 역밀스비(λ) 포함 OLS. 결과식 표준오차는 "
                        "Heckman·Greene 보정 공분산 (Stata heckman, twostep)")
            ct.add_note("lambda = ρσ의 2단계 추정치(역밀스비 계수), rho·sigma는 표준오차 없이 보고")
            if two["truncated"] is not None:
                ct.add_note(f"2단계 rho 추정치 {two['truncated']:.4f}가 [−1, 1] 밖이어서 {two['rho']:g}로 절단, "
                            f"sigma = lambda/rho (Stata 방식)")
            self.model_stats = {"N": n, "Chi2": (W, pW), "df": dfm, "rho": two["rho"], "lambda": two["bl"]}
            lam_txt = f"λ = {two['bl']:.3f} (SE = {se[kx]:.3f})"
            ll_txt = ""
        else:
            rho0 = float(np.clip(two["rho"], -0.9, 0.9))
            th0 = np.concatenate([two["beta"], two["gamma"], [math.atanh(rho0)], [math.log(max(two["sigma"], 1e-6))]])
            hscale = np.concatenate([_col_scale(X[s > 0.5], two["sigma"]), _col_scale(Z), [1.0, 1.0]])
            res = _mle(_HeckmanModel(y, X, Z, s, hscale, [f"s_{nm}" for nm in sd.dz.names]), th0, o["covType"], sd.groups)
            th = np.asarray(res.params, float)
            V, vx = _stata_cov(res, o["covType"], n, sd.groups)
            fit = _FitInfo(float(res.llf), _converged(res), _iterations(res))
            se = np.sqrt(np.clip(np.diag(V), 0, None))
            W, dfm, pW = _wald(th, V, sd.dx.slope_idx)
            ia, il = kx + kz, kx + kz + 1
            tau, ls = th[ia], th[il]
            rho, sig = math.tanh(tau), math.exp(ls)
            lam = rho * sig
            gl = np.array([sig * (1 - rho * rho), lam])         # d lambda / d (athrho, lnsigma)
            se_lam = float(math.sqrt(max(gl @ V[np.ix_([ia, il], [ia, il])] @ gl, 0.0)))
            npar = kx + kz + 2
            aic, bic = -2 * fit.ll + 2 * npar, -2 * fit.ll + npar * math.log(n)
            ll_lab = "로그우도 (Log likelihood)" if o["covType"] == "oim" else "로그 유사우도 (Log pseudolikelihood)"
            fit_rows += [(ll_lab, fit.ll, ""), ("Wald χ²", W, ""), ("자유도 (df)", dfm, "integer"),
                         ("p", pW, "pvalue"), ("AIC", aic, ""), ("BIC", bic, "")]
            for j, nm in enumerate(sd.dx.names):
                ct.add_row(_coef_row(inf, nm, th[j], se[j], eq=eq_out))
            for j, nm in enumerate(sd.dz.names):
                ct.add_row(_coef_row(inf, nm, th[kx + j], se[kx + j], eq=eq_sel))
            ct.add_row(_coef_row(inf, "/athrho", tau, se[ia], eq=""))
            ct.add_row(_coef_row(inf, "/lnsigma", ls, se[il], eq=""))
            lo, hi = inf.ci(tau, se[ia])
            ct.add_row({"eq": "", "term": "rho", "b": rho, "se": (1 - rho * rho) * se[ia], "z": "", "p": "",
                        "lo": math.tanh(lo), "hi": math.tanh(hi)})
            lo, hi = inf.ci(ls, se[il])
            ct.add_row({"eq": "", "term": "sigma", "b": sig, "se": sig * se[il], "z": "", "p": "",
                        "lo": math.exp(lo), "hi": math.exp(hi)})
            lo, hi = inf.ci(lam, se_lam)
            ct.add_row({"eq": "", "term": "lambda", "b": lam, "se": se_lam, "z": "", "p": "", "lo": lo, "hi": hi})
            ct.add_note("최대우도 추정(Stata heckman 기본). 보조모수 /athrho = atanh(ρ), /lnsigma = ln σ로 추정 후 "
                        "rho·sigma·lambda(= ρσ) 변환, 표준오차는 delta 방법, rho·sigma CI는 변환 척도에서 구해 역변환")
            ct.add_note(_vce_note(o["covType"], vx, o["clusterVar"]))
            if not fit.converged:
                ct.add_note(f"수렴 실패: 반복 {fit.iterations}회 후 중단 — 추정치 해석 주의")
            # test of independent equations
            if o["covType"] == "oim":
                ols_b = np.linalg.lstsq(X[s > 0.5], y[s > 0.5], rcond=None)[0]
                rss = float(np.sum((y[s > 0.5] - X[s > 0.5] @ ols_b) ** 2))
                ll_ols = -0.5 * n1 * (LOG_2PI + math.log(rss / n1) + 1.0)
                chi_i = 2.0 * (fit.ll - (two["probit_ll"] + ll_ols))
                it_name = "LR"
            else:
                chi_i = (tau / se[ia]) ** 2
                it_name = "Wald"
            p_i = float(stats.chi2.sf(max(chi_i, 0.0), 1))
            self.add_table(_fit_table("fit", fit_rows))
            it = _fit_table("indep", [(f"{it_name} χ²(1)", chi_i, ""), ("p", p_i, "pvalue")],
                            title=f"두 식의 독립성 검정 (rho = 0, {it_name} test of indep. eqns.)")
            it.add_note("LR: heckman 로그우도와 (선택식 probit + 결과식 OLS) 로그우도 합의 비교" if it_name == "LR"
                        else "Wald: /athrho = 0 검정 (강건·군집 표준오차 사용 시 Stata 방식)")
            self.model_stats = {"N": n, "LL": fit.ll, "Chi2": (W, pW), "df": dfm, "AIC": aic, "BIC": bic,
                                "rho": rho, "lambda": lam}
            lam_txt = f"ρ = {rho:.3f}, λ = {lam:.3f} (SE = {se_lam:.3f}), 독립성 {it_name} χ²(1) = {chi_i:.2f}, p {_pstr(p_i)}"
            ll_txt = "ML"
        if o["method"] == "twostep":
            self.add_table(_fit_table("fit", fit_rows))
        ct.add_note(f"선택 규칙: {sd.rule}")
        for nt in notes:
            ct.add_note(nt)
        bn = sd.dx.base_notes() + sd.dz.base_notes()
        if bn:
            ct.add_note("요인 기준 범주: " + ", ".join(dict.fromkeys(bn)))
        self.add_table(_finish_coef(ct, False, ["eq", "term"]))
        if o["method"] == "ml":
            self.add_table(it)
        self.syntax_python = _sdk_call("heckman", o, {"method": "ml", "covType": "oim", "clusterVar": None,
                                                      "ciWidth": 95.0, "selectDep": None, "factors": [],
                                                      "selectFactors": []})
        self.syntax_r = "\n".join(_r_lines("heckman", o))
        W_ = self.model_stats["Chi2"]
        self.apa_narrative = (f"Heckman 표본선택 모형({'최대우도' if ll_txt else '2단계'} 추정, N = {n}, 선택 {n1}개)의 "
                              f"결과식 Wald χ²({self.model_stats['df']}) = {W_[0]:.2f}, p {_pstr(W_[1])}, {lam_txt}.")


class HeckProbit(BaseAnalysis):
    """Probit with sample selection (Stata heckprobit, ML)."""

    def __init__(self, data: pd.DataFrame, dep: str, covs=None, factors=None, select_dep: Optional[str] = None,
                 select_covs=None, select_factors=None, cov_type: str = "oim", cluster: Optional[str] = None,
                 ci_width: float = 95.0):
        cov_type = resolve_choice("표준오차 유형(covType)", cov_type, COV_TYPES, _COV_ALIASES, default="oim")
        if cov_type == "cluster" and not cluster:
            raise ValueError("군집 강건 표준오차(covType='cluster'): 군집 변수(clusterVar) 지정 필요")
        cluster = cluster if cov_type == "cluster" else None       # unused otherwise (no listwise deletion on it)
        super().__init__(name="heckprobit", title="표본선택 프로빗 (heckprobit)", dataset=data, options={
            "dep": dep, "covs": _as_list(covs), "factors": _as_list(factors), "selectDep": select_dep,
            "selectCovs": _as_list(select_covs), "selectFactors": _as_list(select_factors), "covType": cov_type,
            "clusterVar": cluster, "ciWidth": float(ci_width)})
        self.run()

    def run(self):
        o = self.options
        notes: List[str] = []
        sd = _SelectionData(self.dataset, self.title, o["dep"], o["covs"], o["factors"], o["selectDep"],
                            o["selectCovs"], o["selectFactors"], o["clusterVar"] if o["covType"] == "cluster" else None,
                            notes)
        X, Z, y, s = sd.X, sd.Z, sd.y, sd.s
        sel = s > 0.5
        yv = y[sel]
        if not set(np.unique(yv)) <= {0.0, 1.0}:
            yn = _binary_outcome(pd.Series(yv), o["dep"], notes)
            y = y.copy()
            y[sel] = yn
            yv = yn
        if yv.min() == yv.max():
            raise ValueError(f"{o['dep']}: 선택된 관측치에서 결과 값이 한 가지 — 추정 불가")
        n, kx, kz = len(s), X.shape[1], Z.shape[1]
        n1 = int(sel.sum())
        pz = _sm_binary(s, Z, "probit")                      # statsmodels Probit: selection equation
        px = _sm_binary(y[sel], X[sel], "probit")            # statsmodels Probit: outcome on the selected
        hscale = np.concatenate([_col_scale(X[sel]), _col_scale(Z), [1.0]])
        model = _HeckProbitModel(y, X, Z, s, hscale, [f"s_{nm}" for nm in sd.dz.names])
        best = None
        for r0 in (0.0, 0.5, -0.5):
            f = _mle(model, np.concatenate([np.asarray(px.params), np.asarray(pz.params), [math.atanh(r0)]]),
                     o["covType"], sd.groups)
            if best is None or (_converged(f) and (not _converged(best) or f.llf > best.llf + 1e-9)):
                best = f
            if _converged(f) and r0 == 0.0:
                break
        res = best
        th = np.asarray(res.params, float)
        fit = _FitInfo(float(res.llf), _converged(res), _iterations(res))
        V, vx = _stata_cov(res, o["covType"], n, sd.groups)
        se = np.sqrt(np.clip(np.diag(V), 0, None))
        inf = _Inference(o["ciWidth"])
        W, dfm, pW = _wald(th, V, sd.dx.slope_idx)
        ia = kx + kz
        tau = th[ia]
        rho = math.tanh(tau)
        npar = kx + kz + 1
        aic, bic = -2 * fit.ll + 2 * npar, -2 * fit.ll + npar * math.log(n)
        ll_lab = "로그우도 (Log likelihood)" if o["covType"] == "oim" else "로그 유사우도 (Log pseudolikelihood)"
        self.add_table(_fit_table("fit", [("관측치 수 (N)", n, "integer"), ("선택 (Selected)", n1, "integer"),
                                          ("미선택 (Nonselected)", n - n1, "integer"), (ll_lab, fit.ll, ""),
                                          ("Wald χ²", W, ""), ("자유도 (df)", dfm, "integer"), ("p", pW, "pvalue"),
                                          ("AIC", aic, ""), ("BIC", bic, "")]))
        ct = _coef_table("coef", f"계수 추정치 (Coefficients) — 결과식 {o['dep']}, 선택식 {sd.sname}", inf, eq_col="식")
        for j, nm in enumerate(sd.dx.names):
            ct.add_row(_coef_row(inf, nm, th[j], se[j], eq=o["dep"]))
        for j, nm in enumerate(sd.dz.names):
            ct.add_row(_coef_row(inf, nm, th[kx + j], se[kx + j], eq=sd.sname))
        ct.add_row(_coef_row(inf, "/athrho", tau, se[ia], eq=""))
        lo, hi = inf.ci(tau, se[ia])
        ct.add_row({"eq": "", "term": "rho", "b": rho, "se": (1 - rho * rho) * se[ia], "z": "", "p": "",
                    "lo": math.tanh(lo), "hi": math.tanh(hi)})
        ct.add_note("최대우도 추정(이변량 정규 확률: Genz 알고리즘). /athrho = atanh(ρ), rho 표준오차는 delta 방법")
        ct.add_note(f"선택 규칙: {sd.rule}")
        for nt in notes:
            ct.add_note(nt)
        bn = sd.dx.base_notes() + sd.dz.base_notes()
        if bn:
            ct.add_note("요인 기준 범주: " + ", ".join(dict.fromkeys(bn)))
        ct.add_note(_vce_note(o["covType"], vx, o["clusterVar"]))
        if not fit.converged:
            ct.add_note(f"수렴 실패: 반복 {fit.iterations}회 후 중단 — 추정치 해석 주의")
        self.add_table(_finish_coef(ct, False, ["eq", "term"], check_separation=True))
        if o["covType"] == "oim":
            chi_i = 2.0 * (fit.ll - (float(pz.llf) + float(px.llf)))
            it_name = "LR"
        else:
            chi_i = (tau / se[ia]) ** 2
            it_name = "Wald"
        p_i = float(stats.chi2.sf(max(chi_i, 0.0), 1))
        it = _fit_table("indep", [(f"{it_name} χ²(1)", chi_i, ""), ("p", p_i, "pvalue")],
                        title=f"두 식의 독립성 검정 (rho = 0, {it_name} test of indep. eqns.)")
        it.add_note("LR: heckprobit 로그우도와 (선택식 probit + 결과식 probit) 로그우도 합의 비교" if it_name == "LR"
                    else "Wald: /athrho = 0 검정")
        self.add_table(it)
        self.model_stats = {"N": n, "LL": fit.ll, "Chi2": (W, pW), "df": dfm, "AIC": aic, "BIC": bic, "rho": rho}
        self.syntax_python = _sdk_call("heckprobit", o, {"covType": "oim", "clusterVar": None, "ciWidth": 95.0,
                                                         "selectDep": None, "factors": [], "selectFactors": []})
        self.syntax_r = "\n".join(_r_lines("heckprobit", o))
        self.apa_narrative = (f"표본선택 프로빗 모형(N = {n}, 선택 {n1}개)의 결과식 Wald χ²({dfm}) = {W:.2f}, "
                              f"p {_pstr(pW)}, ρ = {rho:.3f}, 독립성 {it_name} χ²(1) = {chi_i:.2f}, p {_pstr(p_i)}.")

# =============================================================================================== code text helpers
IDS = ("binaryReg", "ordinalReg", "multinomReg", "tobitReg", "truncReg", "heckman", "heckprobit")
ALIASES = {
    "binary_regression": "binaryReg", "binreg": "binaryReg", "binary_choice": "binaryReg",
    "ordinal_regression": "ordinalReg", "ordered_regression": "ordinalReg", "ordreg": "ordinalReg",
    "mlogit": "multinomReg", "multinomial": "multinomReg", "multinomial_logit": "multinomReg",
    "tobit": "tobitReg", "censored_regression": "tobitReg",
    "truncreg": "truncReg", "truncated_regression": "truncReg",
    "heckman_selection": "heckman", "sample_selection": "heckman",
    "heckprob": "heckprobit", "selection_probit": "heckprobit",
}


def _sdk_call(t: str, o: Dict[str, Any], defaults: Dict[str, Any]) -> str:
    parts = [f"{t!r}"]
    for k, v in o.items():
        if k in defaults and (v == defaults[k] or (v in (None, "", []) and defaults[k] in (None, "", []))):
            continue
        if v in (None, "", []) and k not in ("dep",):
            continue
        parts.append(f"{k}={v!r}")
    return f"sess.run({', '.join(parts)})"


def _r_lines(t: str, o: Dict[str, Any]) -> List[str]:
    try:
        return r_code(t, o) or []
    except Exception:
        return []


def _norm(p: Dict[str, Any]) -> Dict[str, Any]:
    """Raw UI / SDK params -> canonical option names used by the code export."""
    try:
        ct = _cov_type(p)
    except ValueError:
        ct = str(_opt(p, "covType", "cov_type", "oim"))
    lv = _opt(p, "ll", "ll", None)
    uv = _opt(p, "ul", "ul", None)
    return {
        "dep": _as_single(p.get("dep")), "covs": _as_list(p.get("covs")), "factors": _as_list(p.get("factors")),
        "link": str(_opt(p, "link", "link", "logit")).lower(),
        "oddsRatio": _flag(p, "oddsRatio", "odds_ratio", False) or _flag(p, "or", "or_", False),
        "ame": _flag(p, "ame", "ame", False), "mem": _flag(p, "mem", "mem", False),
        "classTable": _flag(p, "classTable", "class_table", False), "cutoff": float(_opt(p, "cutoff", "cutoff", 0.5)),
        "hosmer": _flag(p, "hosmer", "hosmer", False), "hlGroups": int(_opt(p, "hlGroups", "hl_groups", 10)),
        "hlTable": _flag(p, "hlTable", "hl_table", False), "roc": _flag(p, "roc", "roc", False),
        "predProbs": _flag(p, "predProbs", "pred_probs", False), "brant": _flag(p, "brant", "brant", False),
        "baseOutcome": _opt(p, "baseOutcome", "base_outcome", None), "rrr": _flag(p, "rrr", "rrr", False),
        "ll": lv, "ul": uv,
        "meLatent": _flag(p, "meLatent", "me_latent", False), "meConditional": _flag(p, "meConditional", "me_conditional", False),
        "meProb": _flag(p, "meProb", "me_prob", False), "meCensored": _flag(p, "meCensored", "me_censored", False),
        "selectDep": _as_single(_opt(p, "selectDep", "select_dep", None)),
        "selectCovs": _as_list(_opt(p, "selectCovs", "select_covs", None)),
        "selectFactors": _as_list(_opt(p, "selectFactors", "select_factors", None)),
        "method": str(_opt(p, "method", "method", "ml")).lower(),
        "covType": ct, "clusterVar": _cluster_var(p), "ciWidth": float(_opt(p, "ciWidth", "ci_width", 95)),
    }


def _st_x(covs: List[str], factors: List[str]) -> str:
    return " ".join(list(covs) + [f"i.{f}" for f in factors if f not in covs])


def _st_opts(opts: List[str]) -> str:
    opts = [x for x in opts if x]
    return (", " + " ".join(opts)) if opts else ""


def _st_vce(o: Dict[str, Any]) -> List[str]:
    out = []
    if o["covType"] == "robust":
        out.append("vce(robust)")
    elif o["covType"] == "cluster":
        out.append(f"vce(cluster {o['clusterVar']})")
    if abs(o["ciWidth"] - 95.0) > 1e-9:
        out.append(f"level({o['ciWidth']:g})")
    return out


def _st_lim(v: Any, which: str) -> Optional[str]:
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    if isinstance(v, str) and v.strip().lower() in ("min", "max", "ll", "ul", "minimum", "maximum"):
        return which
    if isinstance(v, bool):
        return which if v else None
    return f"{which}({float(v):g})"


def _st_lim_pair(o: Dict[str, Any]) -> str:
    def one(v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return "."
        if isinstance(v, str) and v.strip().lower() in ("min", "max", "ll", "ul"):
            return None
        return f"{float(v):g}"
    a, b = one(o["ll"]), one(o["ul"])
    if a is None or b is None:
        return ""
    return f"{a},{b}"


def stata_code(t: str, params: Dict[str, Any]) -> Optional[List[str]]:
    t = ALIASES.get(t, t)
    if t not in IDS:
        return None
    o = _norm(params or {})
    dep, xs, vce = o["dep"], _st_x(o["covs"], o["factors"]), _st_vce(o)
    if t == "binaryReg":
        link = o["link"] if o["link"] in LINKS_BIN else "logit"
        opts = (["or" if link == "logit" else "eform"] if o["oddsRatio"] and link != "probit" else []) + vce
        out = [f"{link} {dep} {xs}{_st_opts(opts)}"]
        if o["ame"]:
            out.append("margins, dydx(*)")
        if o["mem"]:
            out.append("margins, dydx(*) atmeans")
        if o["classTable"]:
            out.append(f"estat classification, cutoff({o['cutoff']:g})")
        if o["hosmer"]:
            out.append(f"estat gof, group({o['hlGroups']})" + (" table" if o["hlTable"] else ""))
        if o["roc"]:
            out.append("lroc")
        return out
    if t == "ordinalReg":
        cmd = "oprobit" if o["link"] == "probit" else "ologit"
        out = [f"{cmd} {dep} {xs}{_st_opts((['or'] if o['oddsRatio'] and cmd == 'ologit' else []) + vce)}"]
        if o["predProbs"]:
            out.append("margins")
        if o["ame"]:
            out.append("margins, dydx(*)")
        if o["brant"]:
            out += ["* Brant 검정: spost13 사용자 명령 (net install spost13_ado, from(https://jslsoc.sitehost.iu.edu/stata))",
                    "brant, detail"]
        return out
    if t == "multinomReg":
        bo = o["baseOutcome"]
        opts = ([f"baseoutcome({bo})"] if bo not in (None, "") else []) + (["rrr"] if o["rrr"] else []) + vce
        out = [f"mlogit {dep} {xs}{_st_opts(opts)}"]
        if bo not in (None, "") and not re.fullmatch(r"-?\d+(\.\d+)?", str(bo)):
            out.insert(0, f"* baseoutcome()에는 숫자 값 필요: 문자 범주는 encode 후 해당 값 지정 ({bo})")
        if o["ame"]:
            out.append("margins, dydx(*)")
        return out
    if t == "tobitReg":
        lims = [x for x in (_st_lim(o["ll"], "ll"), _st_lim(o["ul"], "ul")) if x]
        out = [f"tobit {dep} {xs}{_st_opts(lims + vce)}"]
        pair = _st_lim_pair(o)
        if o["meLatent"]:
            out.append("margins, dydx(*) predict(xb)")
        for key, fn in (("meConditional", "e"), ("meProb", "pr"), ("meCensored", "ystar")):
            if o[key]:
                out.append(f"margins, dydx(*) predict({fn}({pair}))" if pair else
                           f"* margins, dydx(*) predict({fn}(ll,ul)): ll·ul에 검열 한계값(숫자) 지정")
        return out
    if t == "truncReg":
        lims = [x for x in (_st_lim(o["ll"], "ll"), _st_lim(o["ul"], "ul")) if x]
        return [f"truncreg {dep} {xs}{_st_opts(lims + vce)}"]
    zs = _st_x(o["selectCovs"], o["selectFactors"])
    sel = f"select({o['selectDep']} = {zs})" if o["selectDep"] else f"select({zs})"
    if t == "heckman":
        return [f"heckman {dep} {xs}{_st_opts([sel] + (['twostep'] if o['method'] == 'twostep' else []) + vce)}"]
    return [f"heckprobit {dep} {xs}{_st_opts([sel] + vce)}"]


def _r_rhs(covs: List[str], factors: List[str]) -> str:
    terms = list(covs) + [f"factor({f})" for f in factors if f not in covs]
    return " + ".join(terms) if terms else "1"


def _r_vcov_lines(o: Dict[str, Any], fit: str = "fit") -> List[str]:
    if o["covType"] == "robust":
        return [f"lmtest::coeftest({fit}, vcov = sandwich::sandwich({fit}) * nobs({fit}) / (nobs({fit}) - 1))"
                "  # Stata vce(robust): N/(N-1)"]
    if o["covType"] == "cluster":
        return [f"lmtest::coeftest({fit}, vcov = sandwich::vcovCL({fit}, cluster = ~ {o['clusterVar']}, type = \"HC0\", "
                f"cadjust = TRUE))  # Stata vce(cluster): G/(G-1)"]
    return []


def _r_lim(v: Any, dep: str, which: str) -> str:
    if v is None or (isinstance(v, str) and not v.strip()):
        return "-Inf" if which == "left" else "Inf"
    if isinstance(v, str) and v.strip().lower() in ("min", "max", "ll", "ul"):
        return f"{'min' if which == 'left' else 'max'}(data${dep}, na.rm = TRUE)"
    return f"{float(v):g}"


def r_code(t: str, params: Dict[str, Any]) -> Optional[List[str]]:
    t = ALIASES.get(t, t)
    if t not in IDS:
        return None
    o = _norm(params or {})
    dep, rhs = o["dep"], _r_rhs(o["covs"], o["factors"])
    if t == "binaryReg":
        link = o["link"] if o["link"] in LINKS_BIN else "logit"
        out = [f"fit <- glm({dep} ~ {rhs}, family = binomial(link = \"{link}\"), data = data)", "summary(fit)"]
        if link != "logit":
            out.append("# glm 표준오차·sandwich bread는 기대 정보행렬 기준 — Stata(관측 정보행렬)와 소수 자리 차이 가능")
        if o["oddsRatio"] and link != "probit":
            out.append("exp(cbind(estimate = coef(fit), confint.default(fit)))")
        out += _r_vcov_lines(o)
        if o["ame"] or o["mem"]:
            out.append("# 한계효과: install.packages(\"marginaleffects\")")
        if o["ame"]:
            out.append("marginaleffects::avg_slopes(fit)")
        if o["mem"]:
            out.append("marginaleffects::slopes(fit, newdata = \"mean\")")
        if o["hosmer"]:
            out.append(f"ResourceSelection::hoslem.test(fit$y, fitted(fit), g = {o['hlGroups']})  # 집단 경계 규칙이 Stata와 다를 수 있음")
        if o["roc"]:
            out.append("pROC::auc(pROC::roc(fit$y, fitted(fit)))")
        return out
    if t == "ordinalReg":
        meth = "probit" if o["link"] == "probit" else "logistic"
        out = [f"fit <- MASS::polr(factor({dep}, ordered = TRUE) ~ {rhs}, data = data, method = \"{meth}\", Hess = TRUE)",
               "summary(fit)  # zeta = Stata cut, 계수 부호 Stata와 같음"]
        if o["oddsRatio"] and meth == "logistic":
            out.append("exp(coef(fit))")
        if o["brant"]:
            out.append("brant::brant(fit)  # 전체 검정은 R brant의 공분산 조립 방식 차이로 값이 다를 수 있음")
        if o["covType"] != "oim":
            out.append("# 강건·군집 표준오차: sandwich::vcovCL(fit, ...)는 polr 지원 제한 — ordinal::clm 등으로 별도 계산")
        return out
    if t == "multinomReg":
        bo = o["baseOutcome"]
        out = []
        if bo not in (None, ""):
            out.append(f"data${dep} <- relevel(factor(data${dep}), ref = \"{bo}\")")
        else:
            out.append(f"data${dep} <- relevel(factor(data${dep}), ref = names(which.max(table(data${dep}))))  # Stata 기본: 최빈 범주")
        out += [f"fit <- nnet::multinom({dep} ~ {rhs}, data = data, Hess = TRUE, maxit = 1000)", "summary(fit)"]
        if o["rrr"]:
            out.append("exp(coef(fit))")
        return out
    if t == "tobitReg":
        return [f"fit <- AER::tobit({dep} ~ {rhs}, left = {_r_lim(o['ll'], dep, 'left')}, right = {_r_lim(o['ul'], dep, 'right')}, data = data)",
                "summary(fit)  # Log(scale) = ln sigma"] + _r_vcov_lines(o)
    if t == "truncReg":
        L, U = o["ll"], o["ul"]
        if L not in (None, "") and U not in (None, ""):
            return [f"# 양쪽 절단(ll·ul 동시)은 truncreg 패키지 미지원 — 한쪽만 지정해 비교하거나 우도 직접 작성",
                    f"fit <- truncreg::truncreg({dep} ~ {rhs}, data = subset(data, {dep} > {_r_lim(L, dep, 'left')}), "
                    f"point = {_r_lim(L, dep, 'left')}, direction = \"left\", method = \"NR\")"]
        if L not in (None, ""):
            pt, d, cond = _r_lim(L, dep, "left"), "left", ">"
        else:
            pt, d, cond = _r_lim(U, dep, "right"), "right", "<"
        return [f"fit <- truncreg::truncreg({dep} ~ {rhs}, data = subset(data, {dep} {cond} {pt}), point = {pt}, "
                f"direction = \"{d}\", method = \"NR\")", "summary(fit)"]
    zr = _r_rhs(o["selectCovs"], o["selectFactors"])
    out = []
    if o["selectDep"]:
        sdep = o["selectDep"]
        out.append(f"data$.sel <- data${sdep} != 0")
    else:
        out.append(f"data$.sel <- !is.na(data${dep})")
    if t == "heckman":
        meth = "2step" if o["method"] == "twostep" else "ml"
        out += [f"fit <- sampleSelection::selection(.sel ~ {zr}, {dep} ~ {rhs}, data = data, method = \"{meth}\")",
                "summary(fit)  # ML: sigma·rho 직접 모수화 (Stata /lnsigma·/athrho와 변환 관계)"]
        return out
    out += [f"data$.y <- as.logical(data${dep})",
            f"fit <- sampleSelection::selection(.sel ~ {zr}, .y ~ {rhs}, data = data, method = \"ml\")",
            "summary(fit)  # 이항 결과: heckprobit과 같은 모형, 표준오차는 BHHH 기준일 수 있음"]
    return out


# =============================================================================================== dispatch
def dispatch(t: str, df: pd.DataFrame, p: Dict[str, Any]):
    t = ALIASES.get(t, t)
    if t not in IDS:
        return None
    p = p or {}
    dep = _as_single(p.get("dep"))
    covs, factors = _as_list(p.get("covs")), _as_list(p.get("factors"))
    common = dict(cov_type=_cov_type(p), cluster=_cluster_var(p), ci_width=_level(p))
    if t == "binaryReg":
        return BinaryReg(df, dep, covs, factors, link=_opt(p, "link", "link", "logit"),
                         odds_ratio=_flag(p, "oddsRatio", "odds_ratio", False) or _flag(p, "or", "or_", False),
                         ame=_flag(p, "ame", "ame", False), mem=_flag(p, "mem", "mem", False),
                         class_table=_flag(p, "classTable", "class_table", False),
                         cutoff=float(_opt(p, "cutoff", "cutoff", 0.5)), hosmer=_flag(p, "hosmer", "hosmer", False),
                         hl_groups=int(_opt(p, "hlGroups", "hl_groups", 10)), hl_table=_flag(p, "hlTable", "hl_table", False),
                         roc=_flag(p, "roc", "roc", False), **common)
    if t == "ordinalReg":
        return OrdinalReg(df, dep, covs, factors, link=_opt(p, "link", "link", "logit"),
                          odds_ratio=_flag(p, "oddsRatio", "odds_ratio", False) or _flag(p, "or", "or_", False),
                          pred_probs=_flag(p, "predProbs", "pred_probs", False), ame=_flag(p, "ame", "ame", False),
                          brant=_flag(p, "brant", "brant", False), **common)
    if t == "multinomReg":
        return MultinomReg(df, dep, covs, factors, base_outcome=_opt(p, "baseOutcome", "base_outcome", None),
                           rrr=_flag(p, "rrr", "rrr", False), ame=_flag(p, "ame", "ame", False), **common)
    if t == "tobitReg":
        return TobitReg(df, dep, covs, factors, ll=_opt(p, "ll", "ll", None), ul=_opt(p, "ul", "ul", None),
                        me_latent=_flag(p, "meLatent", "me_latent", False),
                        me_conditional=_flag(p, "meConditional", "me_conditional", False),
                        me_prob=_flag(p, "meProb", "me_prob", False), me_censored=_flag(p, "meCensored", "me_censored", False),
                        **common)
    if t == "truncReg":
        return TruncReg(df, dep, covs, factors, ll=_opt(p, "ll", "ll", None), ul=_opt(p, "ul", "ul", None), **common)
    sel = dict(select_dep=_as_single(_opt(p, "selectDep", "select_dep", None)),
               select_covs=_as_list(_opt(p, "selectCovs", "select_covs", None)),
               select_factors=_as_list(_opt(p, "selectFactors", "select_factors", None)))
    if t == "heckman":
        return HeckmanSel(df, dep, covs, factors, method=_opt(p, "method", "method", "ml"), **sel, **common)
    return HeckProbit(df, dep, covs, factors, **sel, **common)


# =============================================================================================== Stata input rules
_MODEL_CMDS = {"logit": "binaryReg", "logistic": "binaryReg", "probit": "binaryReg", "cloglog": "binaryReg",
               "ologit": "ordinalReg", "oprobit": "ordinalReg", "mlogit": "multinomReg", "tobit": "tobitReg",
               "truncreg": "truncReg", "heckman": "heckman", "heckprobit": "heckprobit"}


class _Unsupported(Exception):
    pass


def _split_top_comma(s: str) -> Tuple[str, str]:
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            return s[:i], s[i + 1:]
    return s, ""


def _parse_opt_list(txt: str) -> List[Tuple[str, Optional[str]]]:
    out, i, n = [], 0, len(txt or "")
    while i < n:
        while i < n and txt[i] in " \t,":
            i += 1
        j = i
        while j < n and txt[j] not in " \t,(":
            j += 1
        name = txt[i:j]
        if not name:
            break
        val = None
        if j < n and txt[j] == "(":
            depth, k = 0, j
            while k < n:
                if txt[k] == "(":
                    depth += 1
                elif txt[k] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            if depth != 0:
                raise _Unsupported(f"옵션 괄호 불일치: {name}")
            val = txt[j + 1:k].strip()
            j = k + 1
        out.append((name, val))
        i = j
    return out


def _canon(name: str, table: Sequence[Tuple[str, str]]) -> Optional[str]:
    low = name.lower()
    for full, ab in table:
        if low == full or (len(low) >= len(ab) and full.startswith(low)):
            return full
    return None


_COMMON_OPTS = [("vce", "vce"), ("robust", "r"), ("cluster", "cl"), ("level", "l"), ("nolog", "nolog"),
                ("log", "log")]
_CMD_OPTS = {
    "logit": [("or", "or")], "logistic": [("coef", "coef")], "probit": [], "cloglog": [("eform", "eform")],
    "ologit": [("or", "or")], "oprobit": [], "mlogit": [("baseoutcome", "b"), ("rrr", "rrr")],
    "tobit": [("ll", "ll"), ("ul", "ul")], "truncreg": [("ll", "ll"), ("ul", "ul")],
    "heckman": [("select", "sel"), ("twostep", "two")], "heckprobit": [("select", "sel")],
}


def _parse_varlist(txt: str, what: str) -> Tuple[List[str], List[str]]:
    covs, facs = [], []
    for tok in txt.split():
        if "#" in tok:
            raise _Unsupported(f"상호작용 표기(#·##) 미지원: {tok} — 곱 변수를 만든 뒤 지정")
        m = re.fullmatch(r"i\.\(([^)]+)\)", tok)
        if m:
            facs += m.group(1).split()
            continue
        if re.fullmatch(r"i\.[^\W\d]\w*", tok):
            facs.append(tok[2:])
        elif re.fullmatch(r"c\.[^\W\d]\w*", tok):
            covs.append(tok[2:])
        elif re.match(r"(?i)^(ib|b)[^.]*\.", tok) or re.match(r"(?i)^i\d", tok):
            raise _Unsupported(f"기준 범주 지정({tok}) 미지원 — 가장 작은 값이 기준 범주(Stata 기본)")
        elif re.fullmatch(r"[^\W\d]\w*", tok):
            covs.append(tok)
        else:
            raise _Unsupported(f"{what} 변수 표기 미지원: {tok} (와일드카드·시계열 연산자·범위 표기 제외)")
    return covs, facs


def _vce_opt(val: Optional[str]) -> Tuple[str, Optional[str]]:
    toks = (val or "").split()
    if not toks:
        raise _Unsupported("vce() 값 없음")
    k = toks[0].lower()
    if k in ("oim", "conventional", "conv"):
        return "oim", None
    if "robust".startswith(k) and len(k) >= 1:
        return "robust", None
    if "cluster".startswith(k) and len(k) >= 2:
        if len(toks) != 2:
            raise _Unsupported("vce(cluster 변수): 군집 변수 1개 필요")
        return "cluster", toks[1]
    raise _Unsupported(f"vce({val}) 미지원 (oim·robust·cluster만 가능)")


def _py_call(t: str, kw: Dict[str, Any]) -> str:
    parts = [repr(t)] + [f"{k}={v!r}" for k, v in kw.items()]
    return f"sess.run({', '.join(parts)})"


def _transpile_model(cmd: str, rest: str, line: str) -> Optional[str]:
    body, opt_txt = _split_top_comma(rest)
    if re.search(r"(^|\s)(if|in)\s", body + " ") or "[" in body:
        raise _Unsupported("if·in 한정자와 가중치([pw=] 등)는 미지원 — 자료 처리(keep if 등)로 먼저 걸러낸 뒤 실행")
    toks = body.split()
    if not toks:
        raise _Unsupported("종속변수 필요")
    dep = toks[0]
    if not re.fullmatch(r"[^\W\d]\w*", dep):
        raise _Unsupported(f"종속변수 표기 미지원: {dep}")
    covs, facs = _parse_varlist(" ".join(toks[1:]), "독립")
    opts = _parse_opt_list(opt_txt)
    table = _COMMON_OPTS + _CMD_OPTS[cmd]
    # plain logit/logistic stay with the existing logistic rule (logRegBin)
    if cmd in ("logit", "logistic") and not facs and all(_canon(nm, table) == "or" for nm, _ in opts) \
            and (cmd == "logit" or not opts):
        return None
    if not covs and not facs and cmd not in ("heckman", "heckprobit"):
        raise _Unsupported("독립변수 1개 이상 필요 (상수항만 있는 모형 미지원)")
    t = _MODEL_CMDS[cmd]
    kw: Dict[str, Any] = {"dep": dep}
    if covs:
        kw["covs"] = covs
    if facs:
        kw["factors"] = facs
    if cmd in ("probit", "cloglog"):
        kw["link"] = cmd
    if cmd == "oprobit":
        kw["link"] = "probit"
    if cmd == "logistic":
        kw["oddsRatio"] = True
    for nm, val in opts:
        c = _canon(nm, table)
        if c is None:
            raise _Unsupported(f"{cmd} 옵션 미지원: {nm}{'(' + val + ')' if val is not None else ''}")
        if c in ("nolog", "log"):
            continue
        if c == "vce":
            ct, cl = _vce_opt(val)
            kw["covType"] = ct
            if cl:
                kw["clusterVar"] = cl
        elif c == "robust":
            kw["covType"] = "robust"
        elif c == "cluster":
            if not val:
                raise _Unsupported("cluster(변수) 지정 필요")
            kw["covType"], kw["clusterVar"] = "cluster", val.strip()
        elif c == "level":
            try:
                kw["ciWidth"] = float(val)
            except (TypeError, ValueError):
                raise _Unsupported(f"level({val}) 해석 불가")
        elif c in ("or", "eform"):
            kw["oddsRatio"] = True
        elif c == "coef":
            kw["oddsRatio"] = False
        elif c == "rrr":
            kw["rrr"] = True
        elif c == "baseoutcome":
            if not val:
                raise _Unsupported("baseoutcome(#) 값 필요")
            bo = val.strip().lstrip("#")
            if val.strip().startswith("#"):
                raise _Unsupported("baseoutcome(#k) 순번 지정 미지원 — 범주 값 지정")
            kw["baseOutcome"] = bo
        elif c in ("ll", "ul"):
            if val is None or val == "":
                kw[c] = "min" if c == "ll" else "max"
            else:
                try:
                    kw[c] = float(val)
                except ValueError:
                    raise _Unsupported(f"{c}({val}): 숫자 한계만 지원 (변수 한계 미지원)")
        elif c == "twostep":
            kw["method"] = "twostep"
        elif c == "select":
            if not val:
                raise _Unsupported("select() 내용 필요")
            if "=" in val:
                sdep, zs = val.split("=", 1)
                sdep = sdep.strip()
                if not re.fullmatch(r"[^\W\d]\w*", sdep):
                    raise _Unsupported(f"select() 선택 지표 변수 표기 미지원: {sdep}")
                kw["selectDep"] = sdep
            else:
                zs = val
            zc, zf = _parse_varlist(zs.split(",")[0], "선택식")
            if "," in zs:
                raise _Unsupported("select() 안의 옵션(noconstant·offset) 미지원")
            if zc:
                kw["selectCovs"] = zc
            if zf:
                kw["selectFactors"] = zf
    if t in ("heckman", "heckprobit") and "selectCovs" not in kw and "selectFactors" not in kw:
        raise _Unsupported(f"{cmd}: select() 선택식 지정 필요")
    if t in ("tobitReg", "truncReg") and "ll" not in kw and "ul" not in kw:
        raise _Unsupported(f"{cmd}: ll() 또는 ul() 지정 필요")
    if kw.get("covType") in ("robust", "cluster") and t == "heckman" and kw.get("method") == "twostep":
        raise _Unsupported("heckman twostep: vce(robust)·vce(cluster) 미지원 (Stata와 같음)")
    return _py_call(t, kw)


_MARGINS_OPTS = [("dydx", "dydx"), ("atmeans", "atmeans"), ("predict", "predict"), ("noatlegend", "noatlegend"),
                 ("post", "post")]


def _check_followup(cmd: str, rest: str) -> None:
    """Syntax check at conversion time (the model-dependent part is checked when the line runs)."""
    body, opt_txt = _split_top_comma(rest)
    opts = _parse_opt_list(opt_txt)
    if cmd == "margins":
        if body.strip():
            raise _Unsupported(f"margins 변수 목록({body.strip()}) 미지원 — dydx(*)·atmeans·predict()만 지원")
        for nm, val in opts:
            c = _canon(nm, _MARGINS_OPTS)
            if c is None:
                raise _Unsupported(f"margins 옵션 미지원: {nm}{'(' + val + ')' if val is not None else ''}")
            if c == "post":
                raise _Unsupported("margins, post 미지원 (결과 저장 없이 표로만 표시)")
    elif cmd == "estat classification":
        for nm, val in opts:
            if _canon(nm, [("cutoff", "cut")]) is None:
                raise _Unsupported(f"estat classification 옵션 미지원: {nm}")
    elif cmd == "estat gof":
        for nm, val in opts:
            if _canon(nm, [("group", "g"), ("table", "t")]) is None:
                raise _Unsupported(f"estat gof 옵션 미지원: {nm}")
    elif cmd in ("lroc", "brant"):
        for nm, val in opts:
            if _canon(nm, [("nograph", "nog"), ("detail", "d")]) is None:
                raise _Unsupported(f"{cmd} 옵션 미지원: {nm}")


def _followup_line(stata_line: str) -> str:
    return ("from pyjamovi.analyses.limdep_models import stata_followup as _ld_follow; "
            f"_ld_follow(sess, {stata_line!r})")


def transpile(line: str) -> Optional[str]:
    s = (line or "").strip()
    m = re.match(r"^([A-Za-z]+)\b(.*)$", s)
    if not m:
        return None
    cmd, rest = m.group(1).lower(), m.group(2)
    try:
        if cmd == "margins":
            _check_followup("margins", rest)
            return _followup_line(s)
        if cmd == "estat":
            m2 = re.match(r"^\s*([A-Za-z]+)\b(.*)$", rest)
            sub = m2.group(1).lower() if m2 else ""
            if sub and "classification".startswith(sub) and len(sub) >= 5:
                _check_followup("estat classification", m2.group(2))
                return _followup_line(s)
            if sub == "gof" and re.search(r",.*\bg(r|ro|rou|roup)?\s*\(", m2.group(2) or ""):
                _check_followup("estat gof", m2.group(2))
                return _followup_line(s)
            return None
        if cmd in ("lroc", "brant"):
            _check_followup(cmd, rest)
            return _followup_line(s)
        if cmd not in _MODEL_CMDS:
            return None
        return _transpile_model(cmd, rest, s)
    except _Unsupported as e:
        return f"raise NotImplementedError({(str(e) + ' — ' + s)!r})"


def _lim_val(txt: str) -> Optional[float]:
    t = txt.strip()
    if t in (".", ""):
        return None
    return float(t)


def _rerun_in_place(last: BaseAnalysis, t: str, opts: Dict[str, Any]) -> BaseAnalysis:
    """Re-estimate on the model's own data with extra post-estimation output and update the existing result
    object, so the result sheet keeps one card per model (Stata: post-estimation commands use the stored model)."""
    from ..registry import dispatch_analysis
    new = dispatch_analysis(t, last.dataset, opts)
    for attr in ("title", "tables", "plots", "plot_vectors", "options", "syntax_python", "syntax_r", "apa_narrative",
                 "model_stats"):
        setattr(last, attr, getattr(new, attr))
    return last


def stata_followup(sess, stata_line: str):
    """Run a Stata post-estimation line (margins, estat classification, estat gof, lroc, brant) on the latest
    limited-dependent-variable model of the session: the model is re-estimated with the matching option switched on
    and its existing result is updated (no extra result card)."""
    s = stata_line.strip()
    hist = getattr(sess, "history", None) or []
    last = hist[-1] if hist else None
    if last is not None and getattr(last, "name", None) not in IDS and s.lower().startswith("margins") \
            and "margins" in (getattr(last, "options", None) or {}):
        # another extension model with its own margins switch (for example countReg): reuse or switch it on
        if last.options.get("margins"):
            return last
        return _rerun_in_place(last, last.name, dict(last.options, margins=True))
    if last is None or getattr(last, "name", None) not in IDS:
        raise NotImplementedError(f"{s}: 직전 추정 결과가 제한종속변수 모형(logit·probit·ologit·mlogit·tobit 등)이 아님 — "
                                  f"이 명령은 해당 모형 바로 뒤에서만 변환")
    t = last.name
    opts = dict(last.options)
    m = re.match(r"^([A-Za-z]+)\b(.*)$", s)
    cmd, rest = m.group(1).lower(), m.group(2)
    note = None
    if cmd == "margins":
        _, opt_txt = _split_top_comma(rest)
        ol = {(_canon(nm, _MARGINS_OPTS) or nm): val for nm, val in _parse_opt_list(opt_txt)}
        dydx = ol.get("dydx")
        pred = (ol.get("predict") or "").strip()
        at_means = "atmeans" in ol
        if dydx is not None and dydx.strip() not in ("*", "_all"):
            note = f"Stata dydx({dydx.strip()}): 모든 변수의 한계효과 표시"
        if dydx is None:
            if t == "ordinalReg" and not at_means and (not pred or pred.startswith("outcome") or pred.startswith("pr")):
                opts["predProbs"] = True
            else:
                raise NotImplementedError(f"{s}: 평균 예측값 margins는 ologit·oprobit의 범주별 확률만 지원")
        elif t == "binaryReg":
            if pred and pred.lower() not in ("pr", "p"):
                raise NotImplementedError(f"{s}: predict({pred}) 미지원 — Pr(y=1) 한계효과만 지원")
            opts["mem" if at_means else "ame"] = True
        elif t in ("ordinalReg", "multinomReg"):
            if at_means:
                raise NotImplementedError(f"{s}: {t}의 atmeans 한계효과 미지원 — 평균 한계효과(AME)만 지원")
            if pred and not pred.lower().startswith(("outcome", "pr")):
                raise NotImplementedError(f"{s}: predict({pred}) 미지원")
            if pred:
                note = "predict(outcome()) 지정과 관계없이 모든 결과 범주의 한계효과 표시"
            opts["ame"] = True
        elif t == "tobitReg":
            if at_means:
                raise NotImplementedError(f"{s}: tobit atmeans 한계효과 미지원 — 평균 한계효과(AME)만 지원")
            pm = re.fullmatch(r"(?i)(xb|e|pr|ystar)\s*(?:\(\s*([^,]*)\s*,\s*([^)]*)\s*\))?", pred or "xb")
            if not pm:
                raise NotImplementedError(f"{s}: predict({pred}) 미지원 — xb·e(a,b)·pr(a,b)·ystar(a,b)만 가능")
            kind = pm.group(1).lower()
            if kind != "xb":
                if pm.group(2) is None:
                    raise NotImplementedError(f"{s}: predict({kind}(a,b)) 한계값 지정 필요")
                y = pd.to_numeric(last.dataset[opts["dep"]], errors="coerce").dropna().to_numpy()
                L, U = _parse_limit(opts.get("ll"), "ll", y), _parse_limit(opts.get("ul"), "ul", y)
                a, b = _lim_val(pm.group(2)), _lim_val(pm.group(3))
                if a != L or b != U:
                    raise NotImplementedError(f"{s}: predict 한계({pm.group(2)},{pm.group(3)})가 모형의 검열 한계"
                                              f"({'.' if L is None else f'{L:g}'},{'.' if U is None else f'{U:g}'})와 다름 — "
                                              f"모형 한계에서만 계산")
            opts[{"xb": "meLatent", "e": "meConditional", "pr": "meProb", "ystar": "meCensored"}[kind]] = True
        else:
            raise NotImplementedError(f"{s}: {t} 이후 margins 미지원")
    elif cmd == "estat":
        m2 = re.match(r"^\s*([A-Za-z]+)\b(.*)$", rest)
        sub, tail = m2.group(1).lower(), m2.group(2)
        if t != "binaryReg":
            raise NotImplementedError(f"{s}: logit·probit·cloglog 이후에만 가능")
        ol = dict(_parse_opt_list(_split_top_comma(tail)[1]))
        if sub.startswith("class"):
            opts["classTable"] = True
            for k, v in ol.items():
                if _canon(k, [("cutoff", "cut")]) and v:
                    opts["cutoff"] = float(v)
        else:
            opts["hosmer"] = True
            for k, v in ol.items():
                c = _canon(k, [("group", "g"), ("table", "t")])
                if c == "group" and v:
                    opts["hlGroups"] = int(float(v))
                elif c == "table":
                    opts["hlTable"] = True
    elif cmd == "lroc":
        if t != "binaryReg":
            raise NotImplementedError(f"{s}: logit·probit·cloglog 이후에만 가능")
        opts["roc"] = True
    elif cmd == "brant":
        if t != "ordinalReg" or opts.get("link") != "logit":
            raise NotImplementedError(f"{s}: ologit 이후에만 가능")
        opts["brant"] = True
    else:
        raise NotImplementedError(f"{s}: 변환 규칙 없음")
    res = _rerun_in_place(last, t, opts)
    if note:
        for tb in res.tables:
            if tb.name.startswith("ame") or tb.name.startswith("me_") or tb.name == "mem":
                tb.add_note(note)
    return res


EXTENSION = {
    "catalog": [
        {"id": "binaryReg", "group": "Regression", "title": "Binary Choice (logit / probit / cloglog)", "module": "openstat",
         "icon": "analysis-regression.svg"},
        {"id": "ordinalReg", "group": "Regression", "title": "Ordered Logit / Probit (ologit / oprobit)", "module": "openstat",
         "icon": "analysis-regression.svg"},
        {"id": "multinomReg", "group": "Regression", "title": "Multinomial Logit (mlogit)", "module": "openstat",
         "icon": "analysis-regression.svg"},
        {"id": "tobitReg", "group": "Regression", "title": "Tobit Censored Regression (tobit)", "module": "openstat",
         "icon": "analysis-regression.svg"},
        {"id": "truncReg", "group": "Regression", "title": "Truncated Regression (truncreg)", "module": "openstat",
         "icon": "analysis-regression.svg"},
        {"id": "heckman", "group": "Regression", "title": "Heckman Sample Selection (heckman ML / twostep)", "module": "openstat",
         "icon": "analysis-regression.svg"},
        {"id": "heckprobit", "group": "Regression", "title": "Probit with Sample Selection (heckprobit)", "module": "openstat",
         "icon": "analysis-regression.svg"},
    ],
    "aliases": dict(ALIASES),
    "dispatch": dispatch,
    "stata": stata_code,
    "r": r_code,
    "transpile": transpile,
}
