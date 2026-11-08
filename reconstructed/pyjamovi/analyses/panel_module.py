"""
OpenStat Studio — linear panel-data models (Stata xtreg family) and panel diagnostics.
License: MIT

Computational core = established libraries; this module adds Stata-style options, tables and the reported
quantities the libraries do not give.

Estimators (option ``modelType``)        library call                                   Stata
    fixed   within, grand mean added      linearmodels PanelOLS(entity_effects=True)      xtreg, fe
    twoway  within + time dummies         PanelOLS(entity_effects=True) + dummy columns   xtreg y x i.t, fe
    random  GLS random effects            linearmodels RandomEffects (default components) xtreg, re
            reMethod='sa'                 Swamy–Arora (Stata formula, glue) + PooledOLS    xtreg, re sa
    between panel means (wls option)      linearmodels BetweenOLS(reweight=)              xtreg, be [wls]
    fd      first differences (time gaps) linearmodels PooledOLS on D. data (glue: D.)    regress D.y D.x
    pooled  OLS                           linearmodels PooledOLS                          regress y x
    pa      population-averaged GEE        statsmodels GEE (Exchangeable / Independence /  xtreg, pa corr()
                                          Unstructured; AR(1) with Stata's moment estimator)

Standard errors (``covType``): linearmodels covariance estimators with debiased=True / group_debias=True, which give
Stata's factors (conventional; regress HC1; cluster (N-1)/(N-K)*G/(G-1)); xtreg fe/re vce(robust) = cluster on the
panel variable. Driscoll–Kraay: linearmodels kernel (Bartlett) covariance of the regression of the transformed data
times T/(T-1)*(N-1)/(N-K) = xtscc and StataNow xtreg, fe vce(dkraay bartlett #). t/z, df and p-values follow Stata.

Diagnostics: Hausman (linearmodels RandomEffectsResults.hausman), Pesaran CD (linearmodels pesaran_cd),
F test u_i = 0 (linearmodels f_pooled), Wooldridge serial correlation (statsmodels OLS with cluster VCE, xtserial
recipe); implemented from the published formulas — no library found (searched linearmodels 7.1, statsmodels 0.15,
arch 8.0): Breusch–Pagan (1980) /
LM xttest0, modified Wald xttest3, Stata's R² between/overall, sigma_u, rho, corr(u_i, Xb), theta percentiles.

Verified against Stata 19.5 (xtreg, regress, xtgee, hausman, xttest0 and the xtserial/xtcsd/xttest3 recipes written
with built-in commands) and R plm; reference values are hard-coded in tests/test_panel_extended.py.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseAnalysis, ResultsTable, resolve_choice

# ----------------------------------------------------------------------------- option helpers
MODEL_TYPES = ("fixed", "twoway", "random", "cre", "between", "fd", "pooled", "pa")
MODEL_ALIASES = {"fe": "fixed", "within": "fixed", "entity": "fixed", "re": "random", "gls": "random",
                 "be": "between", "pooling": "pooled", "ols": "pooled", "pool": "pooled", "regress": "pooled",
                 "firstdiff": "fd", "first_difference": "fd", "fdiff": "fd", "gee": "pa", "twoways": "twoway",
                 "fe2": "twoway", "twfe": "twoway", "mundlak": "cre", "correlated": "cre", "cre_re": "cre"}
COV_TYPES = ("unadjusted", "robust", "clustered", "dk")
COV_ALIASES = {"conventional": "unadjusted", "ols": "unadjusted", "classical": "unadjusted", "homoskedastic": "unadjusted",
               "cluster": "clustered", "dkraay": "dk", "driscoll": "dk", "driscoll-kraay": "dk", "scc": "dk",
               "xtscc": "dk", "hc1": "robust"}
RE_METHODS = ("default", "sa")
PA_CORRS = ("exchangeable", "independent", "ar1", "unstructured")
PA_ALIASES = {"exch": "exchangeable", "exc": "exchangeable", "ind": "independent", "indep": "independent",
              "ar": "ar1", "ar 1": "ar1", "ar(1)": "ar1", "unstr": "unstructured", "uns": "unstructured"}

MODEL_LABEL = {
    "fixed": "고정효과 (집단 내 추정)",
    "twoway": "이원 고정효과 (개체 + 시점)",
    "random": "확률효과 (GLS)",
    "cre": "상관 확률효과 (Mundlak)",
    "between": "집단 간 추정",
    "fd": "1차 차분",
    "pooled": "통합 OLS",
    "pa": "모집단 평균 (GEE)",
}
STATA_CMD = {"fixed": "xtreg, fe", "twoway": "xtreg i.t, fe", "random": "xtreg, re", "cre": "xtreg, cre", "between": "xtreg, be",
             "fd": "regress D.y D.x", "pooled": "regress", "pa": "xtreg, pa"}
COV_LABEL = {"unadjusted": "일반(동분산 가정)", "robust": "강건", "clustered": "군집-강건", "dk": "Driscoll–Kraay"}
HAUSMAN_SIGMA = ("none", "more", "less")
PREDICT_STATS = ("xb", "u", "e", "ue", "xbu", "stdp")
PREDICT_LABEL = {"xb": "선형 예측 xb (상수 포함)", "u": "개체효과 u_i", "e": "개별 오차 e_it", "ue": "u_i + e_it (= y − xb)",
                 "xbu": "xb + u_i", "stdp": "선형 예측의 표준오차"}
PREDICT_MODELS = ("fixed", "twoway", "random", "cre")


def _opt(p: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in p and p[k] is not None and p[k] != "" and p[k] != []:
            return p[k]
    return default


def _flag(p: Dict[str, Any], *keys: str, default: bool = False) -> bool:
    v = _opt(p, *keys, default=default)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


def _as_list(val: Any) -> List[str]:
    if val is None or val == "":
        return []
    if isinstance(val, str):
        return [v.strip() for v in re.split(r"[,\s]+", val) if v.strip()]
    return [str(v) for v in val if v is not None and str(v) != ""]


def _as_single(val: Any) -> Any:
    if isinstance(val, (list, tuple)):
        return val[0] if len(val) > 0 else None
    if val == "":
        return None
    return val


# ----------------------------------------------------------------------------- data preparation
def time_positions(t: pd.Series, delta: float = 1.0) -> Tuple[np.ndarray, str]:
    """Integer time positions (Stata xtset with delta) and a note when the variable is not numeric."""
    note = ""
    s = t
    if pd.api.types.is_datetime64_any_dtype(s):
        for freq, lab in (("Y", "연"), ("Q", "분기"), ("M", "월"), ("D", "일")):
            per = s.dt.to_period(freq)
            if per.nunique() == s.nunique():
                return np.asarray(per.array.asi8, dtype=np.int64), f"날짜형 시점 변수를 {lab} 단위 번호로 변환"
        codes = pd.Categorical(s).codes
        return codes.astype(np.int64), "날짜형 시점 변수를 정렬 순서 번호로 변환 (간격 1 가정)"
    if not pd.api.types.is_numeric_dtype(s):
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().all():
            s = num
        else:
            cats = sorted(s.astype(str).unique())
            codes = pd.Categorical(s.astype(str), categories=cats).codes
            return codes.astype(np.int64), "문자형 시점 변수를 정렬 순서 번호로 변환 (간격 1 가정)"
    v = s.astype(float).to_numpy()
    d = float(delta or 1.0)
    if d <= 0:
        raise ValueError("시점 간격(delta)은 양수")
    base = np.nanmin(v)
    pos = (v - base) / d
    if not np.allclose(pos, np.round(pos), atol=1e-8):
        raise ValueError(f"시점 값이 간격 delta={d:g}의 정수배가 아님 (Stata xtset의 delta() 확인)")
    return np.round(pos).astype(np.int64), note


class PanelData:
    """Listwise-complete, sorted panel arrays."""

    def __init__(self, df: pd.DataFrame, entity: str, time: str, dep: str, covs: List[str],
                 extra: Sequence[str] = (), delta: float = 1.0, label: str = "패널 회귀"):
        if not entity or not time or not dep:
            miss = [n for n, v in (("개체 변수(entity)", entity), ("시점 변수(time)", time), ("종속변수(dep)", dep)) if not v]
            raise ValueError(f"{label}: 지정 필요 — {', '.join(miss)} (Stata 명령 입력 시 xtset 개체변수 시점변수 먼저 실행). "
                             f"사용 가능한 변수: {list(df.columns)}")
        cols = [entity, time, dep] + list(covs) + [c for c in extra if c]
        absent = [c for c in dict.fromkeys(cols) if c not in df.columns]
        if absent:
            raise ValueError(f"{label}: 데이터에 없는 변수 — {', '.join(absent)}. 사용 가능한 변수: {list(df.columns)}")
        if dep in covs:
            raise ValueError(f"{label}: 종속변수 '{dep}'가 설명변수에도 포함됨")
        if entity in covs or time in covs:
            raise ValueError(f"{label}: 개체·시점 변수는 설명변수로 지정할 수 없음")
        sub = df[list(dict.fromkeys(cols))].copy()
        for c in [dep] + list(covs):
            if not pd.api.types.is_numeric_dtype(sub[c]):
                conv = pd.to_numeric(sub[c], errors="coerce")
                if conv.notna().sum() < sub[c].notna().sum():
                    raise ValueError(f"{label}: '{c}'는 숫자형 변수 필요 (문자 값 포함)")
                sub[c] = conv
        sub = sub.dropna()
        if sub.empty:
            raise ValueError(f"{label}: 결측 제외 후 관측치 없음")
        self.n_dropped = len(df) - len(sub)
        tpos, self.time_note = time_positions(sub[time], delta)
        sub["_tpos"] = tpos
        if sub.duplicated([entity, "_tpos"]).any():
            dup = sub[sub.duplicated([entity, "_tpos"], keep=False)].head(3)
            raise ValueError(f"{label}: 같은 개체·시점 관측치 중복 (Stata: repeated time values within panel) — 예: "
                             f"{dup[[entity, time]].to_dict('records')}")
        sub = sub.sort_values([entity, "_tpos"], kind="mergesort")
        self.row_index = sub.index.to_numpy()      # row labels of the input data (for predictions)
        sub = sub.reset_index(drop=True)
        self.df = sub
        self.entity, self.time, self.dep, self.covs = entity, time, dep, list(covs)
        self.g, self.entity_labels = pd.factorize(sub[entity], sort=True)
        self.tpos = sub["_tpos"].to_numpy(np.int64)
        self.y = sub[dep].to_numpy(float)
        self.X = sub[list(covs)].to_numpy(float) if covs else np.zeros((len(sub), 0))
        self.N = len(sub)
        self.n = int(self.g.max()) + 1
        self.Ti = np.bincount(self.g, minlength=self.n)
        self.time_values = sub[time].to_numpy()

    def group_mean(self, a: np.ndarray, g: Optional[np.ndarray] = None, Ti: Optional[np.ndarray] = None) -> np.ndarray:
        g = self.g if g is None else g
        Ti = self.Ti if Ti is None else Ti
        n = len(Ti)
        s = np.zeros((n,) + a.shape[1:])
        np.add.at(s, g, a)
        return s / (Ti if a.ndim == 1 else Ti[:, None])

    def time_dummies(self) -> Tuple[np.ndarray, List[str]]:
        """Indicators for every period except the first (Stata i.t base level)."""
        uniq = np.unique(self.tpos)
        lab = {}
        for tp, tv in zip(self.tpos, self.time_values):
            lab.setdefault(int(tp), tv)
        D = np.column_stack([(self.tpos == u).astype(float) for u in uniq[1:]]) if len(uniq) > 1 else np.zeros((self.N, 0))
        names = [f"{_fmt_val(lab[int(u)])}.{self.time}" for u in uniq[1:]]
        return D, names


def _fmt_val(v: Any) -> str:
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(v).date())
    return str(v)


def lag_index(g: np.ndarray, tpos: np.ndarray, lag: int = 1) -> np.ndarray:
    """Row index of (i, t - lag) for each row, -1 when that observation is absent (gaps follow the time values)."""
    key = pd.Series(np.arange(len(g)), index=pd.MultiIndex.from_arrays([g, tpos]))
    want = pd.MultiIndex.from_arrays([g, tpos - lag])
    idx = key.reindex(want).to_numpy()
    return np.where(np.isnan(idx), -1, idx).astype(np.int64)


# ----------------------------------------------------------------------------- linear algebra helpers
def independent_columns(M: np.ndarray, protect: Sequence[int] = (), tol: float = 1e-9) -> List[int]:
    """Columns kept (original order), dropping those collinear with columns entering earlier.

    Protected columns (for example the constant) enter first, the rest in order — as Stata drops the later
    variable of a collinear set and keeps _cons."""
    order = list(protect) + [j for j in range(M.shape[1]) if j not in protect]
    keep: List[int] = []
    Q = np.zeros((M.shape[0], 0))
    for j in order:
        v = np.asarray(M[:, j], float)
        nv = float(np.linalg.norm(v))
        if nv == 0.0 or not np.isfinite(nv):
            continue
        r = v
        for _ in range(2):          # re-orthogonalize once for numerical stability
            if Q.shape[1]:
                r = r - Q @ (Q.T @ r)
        nr = float(np.linalg.norm(r))
        if nr > tol * nv:
            keep.append(j)
            Q = np.column_stack([Q, r / nr])
    return sorted(keep)


def _panel_index(g: np.ndarray, tpos: np.ndarray) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([np.asarray(g, np.int64), np.asarray(tpos, np.int64)], names=["entity", "time"])


def dk_lag_default(T: int) -> int:
    """xtscc default lag: floor(4 (T/100)^(2/9))."""
    return int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))


def stata_percentile(x: np.ndarray, p: float) -> float:
    """Stata summarize, detail percentile definition."""
    xs = np.sort(np.asarray(x, float))
    n = len(xs)
    if n == 0:
        return float("nan")
    h = n * p / 100.0
    k = int(np.floor(h))
    if h == k:
        if k <= 0:
            return float(xs[0])
        if k >= n:
            return float(xs[-1])
        return float((xs[k - 1] + xs[k]) / 2.0)
    return float(xs[min(int(np.ceil(h)) - 1, n - 1)])


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ----------------------------------------------------------------------------- fit container
class Fit:
    """Coefficients, VCE and reporting statistics of one estimator (estimates and VCE come from the library result)."""

    def __init__(self, **kw):
        self.names: List[str] = []
        self.b = np.zeros(0)
        self.V = np.zeros((0, 0))
        self.dist = "t"            # "t" or "z"
        self.df_r: Optional[float] = None
        self.vce_label = ""
        self.notes: List[str] = []
        self.stats: Dict[str, Any] = {}
        self.omitted: List[str] = []
        self.lib = None            # linearmodels / statsmodels results object
        self.lib_name = ""
        self.cd_result = None      # results object whose pesaran_cd matches Stata xtcsd for this model
        self.resid_e: Optional[np.ndarray] = None   # idiosyncratic residual e_it per row (Stata predict, e)
        self.rows: Optional[np.ndarray] = None      # rows of PanelData used
        self.__dict__.update(kw)

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.V), 0, None))

    def stat_p_ci(self, level: float = 0.95):
        se = self.se
        with np.errstate(divide="ignore", invalid="ignore"):
            st = np.where(se > 0, self.b / se, np.nan)
        if self.dist == "z" or self.df_r is None:
            p = 2 * stats.norm.sf(np.abs(st))
            crit = stats.norm.ppf(0.5 + level / 2)
        else:
            p = 2 * stats.t.sf(np.abs(st), self.df_r)
            crit = stats.t.ppf(0.5 + level / 2, self.df_r)
        return st, p, self.b - crit * se, self.b + crit * se

    def wald(self, idx: Sequence[int]):
        """Joint test that the coefficients idx are zero: (F, df1, df2, p) for t-based fits, (chi2, df, None, p) for z.

        With a cluster-robust VCE of rank below the number of tested coefficients (more coefficients than
        clusters - 1) the statistic is not reported, as in Stata."""
        idx = list(idx)
        if not idx:
            return None
        bs = self.b[idx]
        Vs = self.V[np.ix_(idx, idx)]
        q = len(idx)
        ev = np.linalg.eigvalsh((Vs + Vs.T) / 2)
        if np.sum(ev > max(ev.max(), 0) * 1e-12) < q:
            return (None, q, (None if self.dist == "z" or self.df_r is None or getattr(self, "force_chi2", False)
                              else self.df_r), None)
        W = float(bs @ np.linalg.solve(Vs, bs))
        if self.dist == "z" or self.df_r is None or getattr(self, "force_chi2", False):
            return W, q, None, float(stats.chi2.sf(W, q))
        F = W / q
        return F, q, self.df_r, float(stats.f.sf(F, q, self.df_r))


# ----------------------------------------------------------------------------- library calls (linearmodels)
def _lm_cov_kwargs(cov: str, robust_is_cluster: bool, idx: pd.MultiIndex, entity: np.ndarray,
                   cluster: Optional[np.ndarray]) -> Dict[str, Any]:
    """Stata vce() -> linearmodels fit options (debiased / group_debias reproduce Stata's small-sample factors)."""
    if cov == "unadjusted":
        return {"cov_type": "unadjusted", "debiased": True}
    if cov == "robust" and not robust_is_cluster:
        return {"cov_type": "robust", "debiased": True}
    groups = entity if (cov == "robust" or cluster is None) else cluster
    codes = pd.factorize(np.asarray(groups))[0]
    if codes.max() + 1 < 2:
        raise ValueError("군집-강건 표준오차: 군집 2개 이상 필요")
    return {"cov_type": "clustered", "clusters": pd.DataFrame({"cl": codes}, index=idx), "group_debias": True,
            "debiased": True}


def _check_nested(entity: np.ndarray, cluster: Optional[np.ndarray]):
    if cluster is None:
        return
    chk = pd.DataFrame({"e": entity, "c": cluster}).groupby("e")["c"].nunique()
    if (chk > 1).any():
        raise ValueError("군집 변수: 패널(개체)이 군집 안에 포함되지 않음 — Stata xtreg도 오류 처리 "
                         "(panels are not nested within clusters)")


def _dk_on_transformed(yt: np.ndarray, Zt: np.ndarray, names: List[str], g: np.ndarray, tpos: np.ndarray,
                       lag: Optional[int]):
    """Driscoll–Kraay VCE as xtscc / Stata xtreg, fe vce(dkraay): linearmodels kernel (Bartlett) covariance of the
    pooled regression of the transformed data, times T/(T-1)*(N-1)/(N-K)."""
    from linearmodels.panel import PooledOLS
    uniq = np.unique(tpos)
    T = len(uniq)
    if T < 3:
        raise ValueError("Driscoll–Kraay 표준오차: 시점 3개 이상 필요")
    if T < uniq.max() - uniq.min() + 1:
        raise ValueError("Driscoll–Kraay 표준오차: 모든 개체에 공통으로 빠진 시점이 있음 (xtscc: timevar is not regularly spaced)")
    m = dk_lag_default(T) if lag is None else int(abs(lag))
    if m >= T:
        raise ValueError(f"Driscoll–Kraay 최대 시차({m})는 시점 수({T})보다 작아야 함")
    idx = _panel_index(g, tpos)
    mod = PooledOLS(pd.Series(yt, index=idx), pd.DataFrame(Zt, index=idx, columns=names))
    res = mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=m, debiased=False)
    N, K = Zt.shape
    V = res.cov.loc[names, names].to_numpy() * (T / (T - 1.0)) * ((N - 1.0) / (N - K))
    return res, V, T, m


def _order(res, names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    b = res.params.loc[names].to_numpy(float)
    V = res.cov.loc[names, names].to_numpy(float)
    return b, V


def _design(pd_: PanelData, add_time: bool) -> Tuple[np.ndarray, List[str]]:
    X, names = pd_.X, list(pd_.covs)
    if add_time:
        D, dn = pd_.time_dummies()
        X = np.column_stack([X, D]) if D.shape[1] else X
        names = names + dn
    return X, names


def _within(pd_: PanelData, a: np.ndarray) -> np.ndarray:
    """x - xbar_i + xbar (Stata xtreg, fe transform) — used for the collinearity check and the DK regression."""
    m = pd_.group_mean(a)
    return a - m[pd_.g] + a.mean(0)


def _kept_within(pd_: PanelData, X: np.ndarray, names: List[str]):
    Xt = _within(pd_, X) if X.shape[1] else X
    keep = independent_columns(np.column_stack([np.ones(pd_.N), Xt]), protect=[0])
    kept = [j - 1 for j in keep if j > 0]
    return kept, [names[j] for j in range(len(names)) if j not in kept]


def fit_within(pd_: PanelData, X: np.ndarray, names: List[str], cov: str = "unadjusted", cluster=None,
               dk_lag=None) -> Fit:
    """xtreg, fe — linearmodels PanelOLS(entity_effects=True) with a constant (Stata's grand-mean parametrization)."""
    from linearmodels.panel import PanelOLS
    N, n = pd_.N, pd_.n
    kept, omitted = _kept_within(pd_, X, names)
    nk = [names[j] for j in kept]
    k = len(kept)
    if N - n - k <= 0:
        raise ValueError("고정효과 모형: 잔차 자유도 부족 (관측치 수가 집단 수 + 설명변수 수보다 많아야 함)")
    idx = _panel_index(pd_.g, pd_.tpos)
    cols = [f"x{j}" for j in range(k)] + ["_cons"]
    Xk = pd.DataFrame(np.column_stack([X[:, kept], np.ones(N)]), index=idx, columns=cols)
    y = pd.Series(pd_.y, index=idx)
    mod = PanelOLS(y, Xk, entity_effects=True)
    fit = Fit(names=nk + ["Intercept"], omitted=omitted, rows=np.arange(N), lib_name="linearmodels.PanelOLS")
    if cov == "dk":
        base = mod.fit(cov_type="unadjusted", debiased=True)
        Zt = np.column_stack([_within(pd_, X[:, kept]) if k else np.zeros((N, 0)), np.ones(N)])
        yt = _within(pd_, pd_.y[:, None])[:, 0]
        _, V, T, m = _dk_on_transformed(yt, Zt, cols, pd_.g, pd_.tpos, dk_lag)
        fit.b, fit.V = _order(base, cols)[0], V
        fit.df_r, fit.vce_label = T - 1.0, f"Driscoll–Kraay (최대 시차 {m})"
        fit.stats.update({"dk_lag": m, "dk_T": T})
        res = base
    else:
        if cov == "clustered":
            _check_nested(pd_.g, cluster)
        res = mod.fit(**_lm_cov_kwargs(cov, True, idx, pd_.g, cluster))
        fit.b, fit.V = _order(res, cols)
        if cov == "unadjusted":
            fit.df_r, fit.vce_label = float(res.df_resid), "일반"
        else:
            G = int(pd.factorize(pd_.g if (cov == "robust" or cluster is None) else cluster)[0].max() + 1)
            fit.df_r, fit.vce_label = G - 1.0, f"군집-강건 (군집 {G}개)"
            fit.stats["n_clusters"] = G
    fit.lib, fit.cd_result = res, res
    slopes = fit.b[:k]
    yb = pd_.group_mean(pd_.y)
    Xb = pd_.group_mean(X[:, kept]) if k else np.zeros((n, 0))
    xb = X[:, kept] @ slopes
    u = yb - fit.b[-1] - Xb @ slopes                       # Stata u_i = ybar_i - a - xbar_i b
    ssr = float(res.resid_ss)
    s2 = ssr / (N - n - k)
    sigma_u = float(np.std(u, ddof=1)) if n > 1 else float("nan")
    sigma_e = float(np.sqrt(s2))
    fit.stats.update({
        "r2_w": float(res.rsquared_within),
        "r2_b": _corr(Xb @ slopes, yb) ** 2 if k else float("nan"),     # Stata: corr(xbar_i b, ybar_i)^2
        "r2_o": _corr(xb, pd_.y) ** 2 if k else float("nan"),           # Stata: corr(x_it b, y_it)^2
        "sigma_u": sigma_u, "sigma_e": sigma_e,
        "rho": sigma_u ** 2 / (sigma_u ** 2 + sigma_e ** 2),
        "corr_u_xb": _corr(u[pd_.g], xb) if k else float("nan"),
        "ssr": ssr, "df_r_conv": N - n - k, "k": k,
    })
    fp = res.f_pooled                                        # F test that all u_i = 0 (linearmodels)
    fit.stats["F_u"] = (float(fp.stat), int(fp.df), int(fp.df_denom), float(fp.pval))
    fit.resid_e = np.asarray(res.idiosyncratic).ravel()      # e_it = y - a - xb - u_i
    fit.u = u
    return fit


def re_components(pd_: PanelData, X: np.ndarray, method: str = "default"):
    """sigma_e^2 and sigma_u^2 of xtreg, re from the library within and between regressions.

    default: max(0, SSR_b/(n-K) - s_e^2/T_harmonic)  (= linearmodels RandomEffects)
    sa:      max(0, (SSR*_b - (n-K) s_e^2)/(N - tr)) with SSR*_b = sum T_i e_b,i^2 from the unweighted between
             regression ([XT] xtreg Methods and formulas; linearmodels' small_sample uses the T_i-weighted one)."""
    from linearmodels.panel import BetweenOLS, PanelOLS
    N, n = pd_.N, pd_.n
    idx = _panel_index(pd_.g, pd_.tpos)
    kept, _ = _kept_within(pd_, X, [str(j) for j in range(X.shape[1])])
    y = pd.Series(pd_.y, index=idx)
    fe = PanelOLS(y, pd.DataFrame(np.column_stack([X[:, kept], np.ones(N)]), index=idx), entity_effects=True).fit()
    s2e = float(fe.resid_ss) / (N - n - len(kept))
    Zb = np.column_stack([pd_.group_mean(X) if X.shape[1] else np.zeros((n, 0)), np.ones(n)])
    kb = independent_columns(Zb, protect=[Zb.shape[1] - 1])
    Kb = len(kb)
    if n - Kb <= 0:
        raise ValueError("확률효과 모형: 집단 수가 설명변수 수(상수 포함)보다 많아야 함 (집단 간 회귀)")
    cols = [f"x{j}" for j in kb]
    Xfull = np.column_stack([X, np.ones(N)])[:, kb]
    be = BetweenOLS(y, pd.DataFrame(Xfull, index=idx, columns=cols)).fit()
    eb = pd_.group_mean(pd_.y) - Zb[:, kb] @ be.params.to_numpy()
    if method == "sa":
        XF = Zb[:, kb]
        A = (XF * pd_.Ti[:, None]).T @ XF
        B = (XF * (pd_.Ti ** 2)[:, None]).T @ XF
        ctr = float(np.trace(np.linalg.solve(A, B)))
        raw = (float(np.sum(pd_.Ti * eb ** 2)) - (n - Kb) * s2e) / (N - ctr)
    else:
        Tbar = n / float(np.sum(1.0 / pd_.Ti))
        raw = float(eb @ eb) / (n - Kb) - s2e / Tbar
    return s2e, max(0.0, raw), raw


def fit_random(pd_: PanelData, X: np.ndarray, names: List[str], cov: str = "unadjusted", cluster=None,
               dk_lag=None, method: str = "default") -> Fit:
    """xtreg, re — linearmodels RandomEffects (Stata default variance components); sa / DK: GLS transform with the
    Stata components and linearmodels PooledOLS on the transformed data."""
    from linearmodels.panel import PooledOLS, RandomEffects
    N = pd_.N
    Z0 = np.column_stack([X, np.ones(N)])
    keep = independent_columns(Z0, protect=[Z0.shape[1] - 1])
    kept = [j for j in keep if j < X.shape[1]]
    omitted = [names[j] for j in range(len(names)) if j not in kept]
    k = len(kept)
    idx = _panel_index(pd_.g, pd_.tpos)
    cols = [f"x{j}" for j in range(k)] + ["_cons"]
    y = pd.Series(pd_.y, index=idx)
    Xk = X[:, kept]
    s2e, s2u, raw = re_components(pd_, Xk, method)
    th = 1 - np.sqrt(s2e / (pd_.Ti * s2u + s2e))
    thr = th[pd_.g]
    yb = pd_.group_mean(pd_.y)
    Xb = pd_.group_mean(Xk) if k else np.zeros((pd_.n, 0))
    fit = Fit(names=[names[j] for j in kept] + ["Intercept"], omitted=omitted, rows=np.arange(N), dist="z")
    if cov == "clustered":
        _check_nested(pd_.g, cluster)
    if method == "default" and cov != "dk":
        res = RandomEffects(y, pd.DataFrame(np.column_stack([Xk, np.ones(N)]), index=idx, columns=cols)).fit(
            **_lm_cov_kwargs(cov, True, idx, pd_.g, cluster))
        fit.lib_name = "linearmodels.RandomEffects"
        fit.cd_result = res
        vd = res.variance_decomposition
        s2u, s2e = float(vd["Effects"]), float(vd["Residual"])
    else:
        ys = pd_.y - thr * yb[pd_.g]
        Zs = np.column_stack([Xk - thr[:, None] * Xb[pd_.g] if k else np.zeros((N, 0)), 1 - thr])
        mod = PooledOLS(pd.Series(ys, index=idx), pd.DataFrame(Zs, index=idx, columns=cols))
        fit.lib_name = "linearmodels.PooledOLS (GLS 변환 자료)"
        if cov == "dk":
            res, V, T, m = _dk_on_transformed(ys, Zs, cols, pd_.g, pd_.tpos, dk_lag)
            fit.stats.update({"dk_lag": m, "dk_T": T})
        else:
            res = mod.fit(**_lm_cov_kwargs(cov, True, idx, pd_.g, cluster))
    b, V_lib = _order(res, cols)
    fit.b = b
    fit.V = V if cov == "dk" else V_lib
    fit.lib = res
    if cov == "dk":
        fit.dist, fit.df_r, fit.vce_label = "t", fit.stats["dk_T"] - 1.0, f"Driscoll–Kraay (최대 시차 {fit.stats['dk_lag']})"
        fit.force_chi2 = True       # xtscc, re: Wald chi2 = F * df_m with t tests on T-1 df
    elif cov == "unadjusted":
        fit.vce_label = "일반"
    else:
        G = int(pd.factorize(pd_.g if (cov == "robust" or cluster is None) else cluster)[0].max() + 1)
        fit.vce_label = f"군집-강건 (군집 {G}개)"
        fit.stats["n_clusters"] = G
    slopes = b[:-1]
    xb = Xk @ slopes
    resid = pd_.y - b[-1] - xb
    rb = pd_.group_mean(resid)
    u_hat = pd_.Ti * s2u / (pd_.Ti * s2u + s2e) * rb
    fit.stats.update({
        "r2_w": _corr((Xk - Xb[pd_.g]) @ slopes, pd_.y - yb[pd_.g]) ** 2 if k else float("nan"),
        "r2_b": _corr(Xb @ slopes, yb) ** 2 if k else float("nan"),
        "r2_o": _corr(xb, pd_.y) ** 2 if k else float("nan"),
        "sigma_u": float(np.sqrt(s2u)), "sigma_e": float(np.sqrt(s2e)), "rho": float(s2u / (s2u + s2e)),
        "s2u_raw": raw, "theta": th, "theta_obs": thr, "k": k, "method": method,
    })
    fit.resid_e = resid - u_hat[pd_.g]
    fit.u = u_hat
    s2_gls = getattr(res, "s2", None)            # Stata e(rmse)^2 of xtreg, re: transformed-regression SSR/(N-K)
    fit.stats["rmse"] = float(np.sqrt(float(s2_gls))) if s2_gls is not None else float("nan")
    return fit


# ----------------------------------------------------------------------------- correlated random effects (xtreg, cre)
MEAN_PREFIX = "xt_means:"


def cre_design(pd_: PanelData, X: np.ndarray, names: List[str]) -> Tuple[np.ndarray, List[str], List[str]]:
    """[x_it, xbar_i] with panel means of the time-varying regressors (estimation-sample means, Wooldridge 2019);
    time-invariant regressors keep no mean column (their mean equals the variable, as Stata omits it)."""
    tv = []
    for j in range(X.shape[1]):
        dev = X[:, j] - pd_.group_mean(X[:, j])[pd_.g]
        if np.max(np.abs(dev)) > 1e-10 * max(1.0, float(np.max(np.abs(X[:, j])))):
            tv.append(j)
    M = pd_.group_mean(X[:, tv])[pd_.g] if tv else np.zeros((pd_.N, 0))
    return np.column_stack([X, M]), list(names) + [MEAN_PREFIX + names[j] for j in tv], [names[j] for j in tv]


def fit_cre(pd_: PanelData, X: np.ndarray, names: List[str], cov: str = "unadjusted", cluster=None,
            method: str = "default") -> Fit:
    """xtreg, cre — random-effects GLS of y on x_it, xbar_i, z_i (Mundlak 1978; Wooldridge 2019; Stata [XT] xtreg
    Methods and formulas, xtreg, cre), computed by fit_random on the extended design; Mundlak test H0: gamma = 0."""
    if cov == "dk":
        raise ValueError("상관 확률효과(cre)에는 Driscoll–Kraay 표준오차 미지원 (Stata xtreg, cre: conventional·robust·cluster)")
    Xa, na, tv = cre_design(pd_, X, names)
    if not tv:
        raise ValueError("상관 확률효과(cre): 시점에 따라 변하는 설명변수 1개 이상 필요 (패널 평균이 모두 변수 자체와 같음)")
    fit = fit_random(pd_, Xa, na, cov, cluster, None, method)
    col = {nm: j for j, nm in enumerate(na)}
    mean_i = [i for i, nm in enumerate(fit.names) if nm.startswith(MEAN_PREFIX)]
    xit_i = [i for i, nm in enumerate(fit.names) if not nm.startswith(MEAN_PREFIX) and nm != "Intercept"]
    fit.stats["test_idx"] = xit_i
    fit.stats["mundlak"] = fit.wald(mean_i)
    a = Xa[:, [col[fit.names[i]] for i in xit_i]] @ fit.b[xit_i] if xit_i else np.zeros(pd_.N)
    # Stata e(corr) of xtreg, cre (xtreg_cre.ado): corr(x_it b, sum_j xbar_ij) — the panel means enter unweighted
    fit.stats["corr_xit_means"] = _corr(a, pd_.group_mean(X).sum(axis=1)[pd_.g])
    fit.stats["cre_tv"] = tv
    fit.stats["cre_ti"] = [nm for nm in names if nm not in tv]
    fit.X_used = Xa
    fit.names_used = na
    return fit


def wald_pivot(b: np.ndarray, V: np.ndarray, rel_tol: float = 1e-10) -> Tuple[float, int, List[int]]:
    """Wald statistic b' V_S^-1 b over a full-rank subset S of the restrictions, as Stata test reports it when the
    VCE is singular ("Constraint # dropped"): S is chosen greedily by the largest remaining (Schur-complement) variance,
    i.e. a pivoted Cholesky factorization of V (Golub & Van Loan 2013, sec. 4.2.8). Returns (W, |S|, dropped 1-based)."""
    q = len(b)
    A = (np.asarray(V, float) + np.asarray(V, float).T) / 2
    d0 = float(np.max(np.diag(A))) if q else 0.0
    avail, kept = list(range(q)), []
    while avail and d0 > 0:
        dg = np.array([A[j, j] for j in avail])
        k = int(np.argmax(dg))
        if dg[k] <= rel_tol * d0:
            break
        j = avail.pop(k)
        kept.append(j)
        c = A[:, j].copy()
        A = A - np.outer(c, c) / c[j]
    kept.sort()
    if not kept:
        return float("nan"), 0, list(range(1, q + 1))
    bs = b[kept]
    W = float(bs @ np.linalg.solve(V[np.ix_(kept, kept)], bs))
    return W, len(kept), [j + 1 for j in range(q) if j not in kept]


def fit_between(pd_: PanelData, X: np.ndarray, names: List[str], cov: str = "unadjusted", cluster=None,
                wls: bool = False) -> Fit:
    """xtreg, be — linearmodels BetweenOLS (reweight=True for Stata's wls)."""
    from linearmodels.panel import BetweenOLS
    n, N = pd_.n, pd_.N
    if cov == "dk":
        raise ValueError("집단 간 추정(be)에는 Driscoll–Kraay 표준오차를 적용할 수 없음")
    Zb = np.column_stack([pd_.group_mean(X), np.ones(n)])
    keep = independent_columns(Zb, protect=[Zb.shape[1] - 1])
    kept = [j for j in keep if j < X.shape[1]]
    omitted = [names[j] for j in range(len(names)) if j not in kept]
    if n - len(kept) - 1 <= 0:
        raise ValueError("집단 간 추정: 집단 수가 설명변수 수(상수 포함)보다 많아야 함")
    idx = _panel_index(pd_.g, pd_.tpos)
    cols = [f"x{j}" for j in range(len(kept))] + ["_cons"]
    mod = BetweenOLS(pd.Series(pd_.y, index=idx), pd.DataFrame(np.column_stack([X[:, kept], np.ones(N)]), index=idx, columns=cols))
    kw: Dict[str, Any] = {"reweight": bool(wls)}
    if cov == "unadjusted":
        kw.update(cov_type="unadjusted", debiased=True)
    elif cov == "robust":
        kw.update(cov_type="robust", debiased=True)
    else:
        groups = pd_.g if cluster is None else cluster
        _check_nested(pd_.g, cluster)
        kw.update(cov_type="clustered", clusters=pd.DataFrame({"cl": pd.factorize(np.asarray(groups))[0]}, index=idx),
                  group_debias=True, debiased=True)
    res = mod.fit(**kw)
    fit = Fit(names=[names[j] for j in kept] + ["Intercept"], omitted=omitted, lib=res, lib_name="linearmodels.BetweenOLS")
    fit.b, fit.V = _order(res, cols)
    fit.df_r = float(res.df_resid) if cov != "clustered" else float(pd.factorize(np.asarray(pd_.g if cluster is None else cluster))[0].max())
    fit.vce_label = {"unadjusted": "일반", "robust": "강건 (HC1)", "clustered": "군집-강건"}[cov]
    if cov in ("robust", "clustered"):
        fit.notes.append("Stata xtreg, be는 vce(robust)·vce(cluster) 미지원 — 집단 평균 회귀에 "
                         + ("HC1 강건 표준오차" if cov == "robust" else "군집-강건 표준오차") + " 적용")
    slopes = fit.b[:-1]
    Xk = X[:, kept]
    yb = pd_.group_mean(pd_.y)
    Xb = pd_.group_mean(Xk) if kept else np.zeros((n, 0))
    fit.stats.update({
        "r2_b": float(res.rsquared),
        "r2_w": _corr((Xk - Xb[pd_.g]) @ slopes, pd_.y - yb[pd_.g]) ** 2 if kept else float("nan"),
        "r2_o": _corr(Xk @ slopes, pd_.y) ** 2 if kept else float("nan"),
        "rmse": float(np.sqrt(res.s2)), "k": len(kept), "wls": wls,
    })
    return fit


def fit_ols_panel(pd_: PanelData, X: np.ndarray, names: List[str], cov: str = "unadjusted", cluster=None,
                  dk_lag=None, diff: bool = False, constant: bool = True) -> Fit:
    """regress y x (pooled) or regress D.y D.x (fd) — linearmodels PooledOLS; differences follow the time variable."""
    from linearmodels.panel import PooledOLS
    y, g, tpos = pd_.y, pd_.g, pd_.tpos
    rows = np.arange(pd_.N)
    if diff:
        li = lag_index(g, tpos, 1)
        rows = np.where(li >= 0)[0]
        if len(rows) == 0:
            raise ValueError("1차 차분: 연속된 시점 쌍이 없음 (시점 간격·delta 확인)")
        y = pd_.y[rows] - pd_.y[li[rows]]
        X = X[rows] - X[li[rows]]
        g, tpos = g[rows], tpos[rows]
    N = len(y)
    Z = np.column_stack([X, np.ones(N)]) if constant else X
    keep = independent_columns(Z, protect=[Z.shape[1] - 1] if constant else [])
    kept = [j for j in keep if j < X.shape[1]]
    omitted = [names[j] for j in range(len(names)) if j not in kept]
    cols = [f"x{j}" for j in range(len(kept))] + (["_cons"] if constant else [])
    if not cols:
        raise ValueError("추정할 계수가 없음")
    Zk = np.column_stack([X[:, kept]] + ([np.ones(N)] if constant else []))
    K = Zk.shape[1]
    if N - K <= 0:
        raise ValueError("관측치 수가 계수 수보다 많아야 함")
    idx = _panel_index(g, tpos)
    mod = PooledOLS(pd.Series(y, index=idx), pd.DataFrame(Zk, index=idx, columns=cols))
    fit = Fit(names=[names[j] for j in kept] + (["Intercept"] if constant else []), omitted=omitted, rows=rows,
              lib_name="linearmodels.PooledOLS")
    clus = None if cluster is None else np.asarray(cluster)[rows]
    if cov == "dk":
        res, V, T, m = _dk_on_transformed(y, Zk, cols, g, tpos, dk_lag)
        fit.b, fit.V = _order(res, cols)[0], V
        fit.df_r, fit.vce_label = T - 1.0, f"Driscoll–Kraay (최대 시차 {m})"
        fit.stats.update({"dk_lag": m, "dk_T": T})
    else:
        res = mod.fit(**_lm_cov_kwargs(cov, False, idx, g, clus))
        fit.b, fit.V = _order(res, cols)
        if cov == "clustered":
            G = int(pd.factorize(g if clus is None else clus)[0].max() + 1)
            fit.df_r, fit.vce_label = G - 1.0, f"군집-강건 (군집 {G}개)"
            fit.stats["n_clusters"] = G
        else:
            fit.df_r = float(N - K)
            fit.vce_label = "일반" if cov == "unadjusted" else "강건 (HC1)"
    fit.lib, fit.cd_result = res, res
    e = y - Zk @ fit.b
    ssr = float(e @ e)
    tss = float(((y - y.mean()) ** 2).sum()) if constant else float((y ** 2).sum())
    r2 = 1 - ssr / tss if tss > 0 else float("nan")
    fit.stats.update({"r2": r2, "r2_adj": 1 - (1 - r2) * (N - (1 if constant else 0)) / (N - K),
                      "rmse": float(np.sqrt(ssr / (N - K))), "N": N, "groups": int(len(np.unique(g))),
                      "k": len(kept), "ssr": ssr})
    fit.resid_e = e
    fit.g_rows, fit.t_rows = g, tpos
    return fit


# ----------------------------------------------------------------------------- population-averaged GEE (statsmodels)
def _stata_ar1_struct():
    """statsmodels CovStruct with Stata xtgee's corr(ar 1) moment estimator:
    alpha = [sum r_t r_t+1 / sum(n_i - 1)] / [sum r^2 / N] over panels with n_i > 1 (checked against Stata 19)."""
    from statsmodels.genmod.cov_struct import CovStruct

    class StataAR1(CovStruct):
        def __init__(self):
            super().__init__()
            self.dep_params = 0.0

        def update(self, params):
            num = den = npair = nobs = 0.0
            for i in range(self.model.num_group):
                r = self.model.endog_li[i] - self.model.cached_means[i][0]
                if len(r) < 2:
                    continue
                num += float(np.sum(r[1:] * r[:-1]))
                den += float(np.sum(r * r))
                npair += len(r) - 1
                nobs += len(r)
            self.dep_params = (num / npair) / (den / nobs) if npair and den > 0 else 0.0

        def covariance_matrix(self, expval, index):
            k = len(expval)
            ix = np.arange(k)
            return self.dep_params ** np.abs(ix[:, None] - ix[None, :]), True

        def summary(self):
            return f"Stata AR(1) correlation: {self.dep_params:.6f}"

    return StataAR1()


def fit_pa(pd_: PanelData, X: np.ndarray, names: List[str], cov: str = "unadjusted", corr: str = "exchangeable",
           tol: float = 1e-8, maxit: int = 200) -> Fit:
    """xtreg, pa / xtgee family(gaussian) link(identity) — statsmodels GEE, scale = Pearson chi2 / N (ddof_scale=0)."""
    import statsmodels.api as sm
    from statsmodels.genmod import cov_struct as cs
    if cov == "dk":
        raise ValueError("모집단 평균(pa) 모형에는 Driscoll–Kraay 표준오차를 적용할 수 없음")
    g, tpos, Ti = pd_.g, pd_.tpos, pd_.Ti
    y = pd_.y
    rows = np.arange(pd_.N)
    if corr == "ar1":
        same = np.zeros(pd_.N, bool)
        same[1:] = g[1:] == g[:-1]
        if (same & (np.diff(np.r_[tpos[0], tpos]) != 1)).any():
            raise ValueError("pa corr(ar1): 시점 간격이 일정하지 않은 개체가 있음 (Stata xtgee도 force 없이는 추정하지 않음)")
        rows = np.where(Ti[g] > 1)[0]      # Stata drops panels with too few observations for lag structures
    Xr, yr, gr, tr = X[rows], y[rows], g[rows], tpos[rows]
    N = len(yr)
    Z = np.column_stack([Xr, np.ones(N)])
    keep = independent_columns(Z, protect=[Z.shape[1] - 1])
    kept = [j for j in keep if j < X.shape[1]]
    omitted = [names[j] for j in range(len(names)) if j not in kept]
    Zk = np.column_stack([Xr[:, kept], np.ones(N)])
    struct = {"exchangeable": cs.Exchangeable, "independent": cs.Independence, "unstructured": cs.Unstructured}.get(corr)
    struct = _stata_ar1_struct() if corr == "ar1" else struct()
    kw = {"time": np.searchsorted(np.unique(tr), tr)} if corr == "unstructured" else {}
    mod = sm.GEE(yr, Zk, groups=gr, cov_struct=struct, family=sm.families.Gaussian(), **kw)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(maxiter=maxit, ctol=tol, ddof_scale=0, scale=None,
                      cov_type="naive" if cov == "unadjusted" else "robust")
    n = int(len(np.unique(gr)))
    V = np.asarray(res.cov_params())
    if cov != "unadjusted":
        V = V * (n / (n - 1.0))                         # [P] robust: clustered on panels, factor M/(M-1)
        if cov == "clustered":
            pass
    fit = Fit(names=[names[j] for j in kept] + ["Intercept"], omitted=omitted, rows=rows, dist="z",
              b=np.asarray(res.params, float), V=V, lib=res, lib_name="statsmodels GEE")
    fit.vce_label = "일반" if cov == "unadjusted" else f"강건 (패널 군집 {n}개)"
    if cov == "clustered":
        fit.notes.append("Stata xtgee의 강건 표준오차는 패널 군집 기준 — 군집 변수 지정은 무시하고 패널 군집 적용")
    dep = mod.cov_struct.dep_params
    converged = bool(getattr(res, "converged", True))
    fit.stats.update({"scale": float(res.scale), "alpha": dep, "corr": corr, "converged": converged,
                      "N": N, "groups": n, "Ti": np.bincount(pd.factorize(gr)[0]), "k": len(kept)})
    if not converged:
        fit.notes.append(f"GEE 반복 {maxit}회 내 수렴 기준 미충족 — 결과 해석 주의")
    fit.resid_e = yr - Zk @ fit.b
    return fit


# ----------------------------------------------------------------------------- diagnostics
def hausman_test(pd_: PanelData, X: np.ndarray, names: List[str], sigma: str = "none") -> Dict[str, Any]:
    """hausman fe re — linearmodels RandomEffectsResults.hausman (each model's conventional VCE, generalized inverse).

    sigma='more' / 'less' (Stata hausman sigmamore / sigmaless): both VCEs use the disturbance SD of the efficient
    (RE: e(rmse)) or of the consistent (FE: e(sigma_e)) estimator, V_b*(s_B/s_b)^2 or V_B*(s_b/s_B)^2 ([R] hausman)."""
    from linearmodels.panel import PanelOLS, RandomEffects
    N = pd_.N
    kept, _ = _kept_within(pd_, X, names)
    cols = [names[j] for j in kept]
    idx = _panel_index(pd_.g, pd_.tpos)
    y = pd.Series(pd_.y, index=idx)
    Xd = pd.DataFrame(np.column_stack([X[:, kept], np.ones(N)]), index=idx, columns=cols + ["_cons"])
    import warnings
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        fe = PanelOLS(y, Xd, entity_effects=True).fit()
        try:
            re_ = RandomEffects(y, Xd).fit()
        except ValueError as e:
            raise ValueError("확률효과 모형 추정 불가 — 집단 간 회귀의 잔차 자유도 부족 (집단 수 ≤ 설명변수 수, "
                             f"시점 더미 포함 시 흔함): {e}")
        h = re_.hausman(fe)
    if sigma in ("more", "less"):
        terms = list(h.estimates.index)
        b0, b1 = fe.params[terms].to_numpy(), re_.params[terms].to_numpy()
        V0, V1 = fe.cov.loc[terms, terms].to_numpy(), re_.cov.loc[terms, terms].to_numpy()
        s0, s1 = float(np.sqrt(fe.s2)), float(np.sqrt(re_.s2))
        if sigma == "more":
            V0 = (s1 / s0) ** 2 * V0
        else:
            V1 = (s0 / s1) ** 2 * V1
        Vd = V0 - V1
        Vs = (Vd + Vd.T) / 2
        ev = np.linalg.eigvalsh(Vs)
        tol = max(1e-300, float(np.max(np.abs(ev)))) * 1e-12
        rank = int(np.sum(np.abs(ev) > tol))
        d = b0 - b1
        chi2 = float(d @ np.linalg.pinv(Vs) @ d)
        return {"chi2": chi2, "df": rank, "p": float(stats.chi2.sf(chi2, rank)) if chi2 >= 0 else float("nan"),
                "pos_def": bool(np.all(ev > tol)), "terms": terms, "b_fe": b0, "b_re": b1,
                "se_diff": np.sqrt(np.clip(np.diag(Vs), 0, None)), "sigma": sigma, "s_fe": s0, "s_re": s1}
    est = h.estimates
    terms = list(est.index)
    Vd = fe.cov.loc[terms, terms].to_numpy() - re_.cov.loc[terms, terms].to_numpy()
    ev = np.linalg.eigvalsh((Vd + Vd.T) / 2)
    pd_ok = bool(np.all(ev > 1e-12 * max(1e-300, np.max(np.abs(ev)))))
    st = h.statistic
    chi2 = float(getattr(st, "stat", np.nan))
    valid = np.isfinite(chi2)
    if not valid:      # linearmodels returns an InvalidTestStatistic when chi2 < 0; Stata prints the negative value
        b0, b1 = est["b0"].to_numpy(), est["b1"].to_numpy()
        chi2 = float((b0 - b1) @ np.linalg.pinv(Vd) @ (b0 - b1))
    return {"chi2": chi2, "df": int(getattr(st, "df", len(terms)) or len(terms)),
            "p": float(st.pval) if valid else float("nan"), "pos_def": pd_ok, "terms": terms,
            "b_fe": est["b0"].to_numpy(), "b_re": est["b1"].to_numpy(), "se_diff": est["Std. Err."].to_numpy()}


def bp_lm_test(pd_: PanelData, X: np.ndarray) -> Dict[str, Any]:
    """xttest0 (Breusch & Pagan 1980; no library found: linearmodels, statsmodels, arch): Breusch–Pagan LM with the
    Baltagi–Li unbalanced form on linearmodels PooledOLS residuals; 50:50 chi2(0)/chi2(1) mixture."""
    from linearmodels.panel import PooledOLS
    N, n = pd_.N, pd_.n
    Z = np.column_stack([X, np.ones(N)])
    keep = independent_columns(Z, protect=[Z.shape[1] - 1])
    idx = _panel_index(pd_.g, pd_.tpos)
    po = PooledOLS(pd.Series(pd_.y, index=idx), pd.DataFrame(Z[:, keep], index=idx)).fit()
    v = np.asarray(po.resids).ravel()
    s = np.zeros(n)
    np.add.at(s, pd_.g, v)
    A1 = 1 - float(np.sum(s ** 2)) / float(v @ v)
    lm = (N ** 2 / 2.0) * A1 ** 2 / (float(np.sum(pd_.Ti ** 2)) - N)
    s2e, s2u, raw = re_components(pd_, X, "default")
    if raw < 0:
        lm = 0.0
    p = 1.0 if lm <= 0 else 0.5 * float(stats.chi2.sf(lm, 1))
    return {"lm": lm, "p": p, "var_y": float(np.var(pd_.y, ddof=1)), "var_e": s2e, "var_u": s2u}


def wooldridge_serial(pd_: PanelData, X: np.ndarray) -> Dict[str, Any]:
    """xtserial (Drukker 2003) — statsmodels OLS: regress D.y D.x, nocons; regress e L.e, nocons, cluster(id)
    (use_correction = Stata regress factor); F = ((b + 0.5)/se)^2 with G-1 df. Lags follow the time variable."""
    import statsmodels.api as sm
    li = lag_index(pd_.g, pd_.tpos, 1)
    rows = np.where(li >= 0)[0]
    if len(rows) < 3:
        raise ValueError("Wooldridge 검정: 1차 차분 관측치 부족")
    dy = pd_.y[rows] - pd_.y[li[rows]]
    dX = X[rows] - X[li[rows]]
    keep = independent_columns(dX) if dX.shape[1] else []
    e = sm.OLS(dy, dX[:, keep]).fit().resid if keep else dy
    g, tp = pd_.g[rows], pd_.tpos[rows]
    li2 = lag_index(g, tp, 1)
    r2 = np.where(li2 >= 0)[0]
    if len(r2) < 3:
        raise ValueError("Wooldridge 검정: 잔차의 시차 관측치 부족 (개체별 연속 시점 3개 이상 필요)")
    gg = g[r2]
    res = sm.OLS(e[r2], e[li2[r2]][:, None]).fit(cov_type="cluster", cov_kwds={"groups": gg, "use_correction": True,
                                                                              "df_correction": True})
    rho = float(res.params[0])
    se = float(res.bse[0])
    G = int(len(np.unique(gg)))
    F = ((rho + 0.5) / se) ** 2
    return {"F": F, "df1": 1, "df2": G - 1, "p": float(stats.f.sf(F, 1, G - 1)), "corr": rho, "N": len(r2), "G": G}


def pesaran_cd(result) -> Dict[str, Any]:
    """xtcsd, pesaran — linearmodels PanelResults.pesaran_cd (pairwise Pearson correlations on common periods)."""
    t = result.pesaran_cd
    if not hasattr(t, "stat") or not np.isfinite(getattr(t, "stat", np.nan)):
        raise ValueError(f"Pesaran CD 검정 적용 불가: {getattr(t, 'reason', t)}")
    return {"cd": float(t.stat), "p": float(t.pval)}


def mean_abs_corr(e: np.ndarray, g: np.ndarray, tpos: np.ndarray) -> float:
    """xtcsd, abs: average absolute off-diagonal residual correlation (pandas pairwise correlations)."""
    wide = pd.DataFrame({"g": g, "t": tpos, "e": e}).pivot(index="t", columns="g", values="e")
    C = wide.corr(min_periods=2).to_numpy()
    iu = np.triu_indices_from(C, 1)
    v = np.abs(C[iu])
    return float(np.nanmean(v))


def modified_wald(e: np.ndarray, g: np.ndarray, n_groups: int) -> Dict[str, Any]:
    """xttest3 (Greene 2000; Baum 2001, v1.0.8 formula — no library found): sum_i (s_i^2 - s^2)^2 / V_i, chi2(N_g)."""
    codes, _ = pd.factorize(g)
    n = int(codes.max()) + 1
    Ti = np.bincount(codes, minlength=n).astype(float)
    e2 = e ** 2
    si = np.zeros(n)
    np.add.at(si, codes, e2)
    si /= Ti
    dev = (e2 - si[codes]) ** 2
    v = np.zeros(n)
    np.add.at(v, codes, dev)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = v / Ti / (Ti - 1)
    ok = np.isfinite(v) & (v >= 1e-12)
    s2 = float(np.var(e, ddof=0))
    W = float(np.sum((si[ok] - s2) ** 2 / v[ok]))
    df = int(n_groups)
    return {"chi2": W, "df": df, "p": float(stats.chi2.sf(W, df)), "excluded": int(np.sum(~ok))}


# ----------------------------------------------------------------------------- analysis
class PanelAnalysis(BaseAnalysis):
    """Linear panel-data regression with Stata xtreg conventions."""

    def __init__(
        self,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        dep: str,
        covs: Union[str, List[str]],
        model_type: str = "fixed",
        cov_type: str = "unadjusted",
        hausman: bool = True,
        plot_trends: bool = True,
        cluster: Optional[str] = None,
        dk_lags: Optional[int] = None,
        re_method: str = "default",
        be_wls: bool = False,
        fd_constant: bool = True,
        pa_corr: str = "exchangeable",
        bp_test: bool = False,
        serial_test: bool = False,
        cd_test: bool = False,
        het_test: bool = False,
        show_time_effects: bool = True,
        delta: float = 1.0,
        hausman_sigma: str = "none",
        time_test: bool = True,
    ):
        cov_list = _as_list(covs) if isinstance(covs, str) else list(covs or [])
        mt = resolve_choice("추정 방법(modelType)", model_type, MODEL_TYPES, MODEL_ALIASES, default="fixed")
        ct = resolve_choice("표준오차(covType)", cov_type, COV_TYPES, COV_ALIASES, default="unadjusted")
        rm = resolve_choice("확률효과 분산성분(reMethod)", re_method, RE_METHODS, {"swar": "sa", "swamy": "sa", "consistent": "default"},
                            default="default")
        pc = resolve_choice("작업 상관 구조(paCorr)", pa_corr, PA_CORRS, PA_ALIASES, default="exchangeable")
        hs = resolve_choice("Hausman 분산 기준(hausmanSigma)", hausman_sigma or "none", HAUSMAN_SIGMA,
                            {"sigmamore": "more", "sigmaless": "less", "": "none", "default": "none"}, default="none")
        if not cov_list:
            raise ValueError("패널 회귀: 설명변수(covs) 1개 이상 지정 필요")
        if dk_lags is not None and dk_lags != "":
            dk_lags = int(dk_lags)
            if dk_lags < 0:
                raise ValueError("Driscoll–Kraay 최대 시차(dkLags)는 0 이상 정수")
        else:
            dk_lags = None
        super().__init__(
            name="panel",
            title=f"패널 회귀 — {MODEL_LABEL[mt]} ({STATA_CMD[mt]})",
            dataset=data,
            options={
                "entity": entity_col, "time": time_col, "dep": dep, "covs": cov_list,
                "modelType": mt, "covType": ct, "cluster": cluster or "", "dkLags": dk_lags,
                "reMethod": rm, "beWls": bool(be_wls), "fdConstant": bool(fd_constant), "paCorr": pc,
                "hausman": bool(hausman), "bpTest": bool(bp_test), "serialTest": bool(serial_test),
                "cdTest": bool(cd_test), "hetTest": bool(het_test), "showTimeEffects": bool(show_time_effects),
                "delta": float(delta or 1.0), "plotTrends": bool(plot_trends), "hausmanSigma": hs,
                "timeTest": bool(time_test),
            },
        )
        self.entity_col, self.time_col, self.dep, self.covs = entity_col, time_col, dep, cov_list
        self.model_type, self.cov_type = mt, ct
        self.syntax_python = self._py_syntax()
        self.syntax_r = "\n".join(r_code("panel", self.options) or [])
        self.run()

    # ------------------------------------------------------------------ syntax
    def _py_syntax(self) -> str:
        o = self.options
        defaults = {"cluster": "", "dkLags": None, "reMethod": "default", "beWls": False, "fdConstant": True,
                    "paCorr": "exchangeable", "hausman": True, "bpTest": False, "serialTest": False, "cdTest": False,
                    "hetTest": False, "showTimeEffects": True, "delta": 1.0, "plotTrends": True,
                    "hausmanSigma": "none", "timeTest": True}
        parts = [f"entity={o['entity']!r}", f"time={o['time']!r}", f"dep={o['dep']!r}", f"covs={o['covs']!r}",
                 f"modelType={o['modelType']!r}", f"covType={o['covType']!r}"]
        for k, dv in defaults.items():
            if o.get(k) != dv:
                parts.append(f"{k}={o[k]!r}")
        return f"res = sess.run('panel', {', '.join(parts)})"

    # ------------------------------------------------------------------ run
    def run(self):
        o = self.options
        mt, ct = self.model_type, self.cov_type
        clus_name = o.get("cluster") or None
        if ct == "clustered" and not clus_name:
            clus_name = None
        extra = [clus_name] if clus_name and clus_name != self.entity_col else []
        P = PanelData(self.dataset, self.entity_col, self.time_col, self.dep, self.covs, extra=extra,
                      delta=o["delta"], label="패널 회귀")
        self.panel = P
        cluster = None
        if ct == "clustered" and clus_name and clus_name != self.entity_col:
            cluster = P.df[clus_name].to_numpy()
        X, names = _design(P, mt == "twoway")
        n_time_dummies = len(names) - len(self.covs)
        if mt in ("fixed", "twoway"):
            fit = fit_within(P, X, names, ct, cluster, o["dkLags"])
        elif mt == "random":
            fit = fit_random(P, X, names, ct, cluster, o["dkLags"], o["reMethod"])
        elif mt == "cre":
            fit = fit_cre(P, X, names, ct, cluster, o["reMethod"])
        elif mt == "between":
            fit = fit_between(P, X, names, ct, cluster, o["beWls"])
        elif mt == "fd":
            fit = fit_ols_panel(P, X, names, ct, cluster, o["dkLags"], diff=True, constant=o["fdConstant"])
        elif mt == "pooled":
            fit = fit_ols_panel(P, X, names, ct, cluster, o["dkLags"], diff=False, constant=True)
        else:
            fit = fit_pa(P, X, names, ct, o["paCorr"])
        if not hasattr(fit, "X_used"):
            fit.X_used, fit.names_used = X, names
        self.fit = fit
        self._summary_table(P, fit)
        self._coef_tables(P, fit, n_time_dummies)
        if mt == "twoway" and o["timeTest"]:
            self._safe(lambda: self._time_test_table(fit), "시점 효과 결합검정")
        if mt == "cre":
            self._mundlak_table(fit)
        if mt in ("fixed", "twoway"):
            self._u_test_table(fit)
        if mt in ("random", "cre"):
            self._re_variance_table(P, fit)
        if mt == "pa":
            self._pa_corr_table(fit)
        # diagnostics
        diag_X = X
        if o["hausman"]:
            self._safe(lambda: self._hausman_table(P, diag_X, names), "Hausman 검정")
        if o["bpTest"]:
            self._safe(lambda: self._bp_table(P, diag_X), "Breusch–Pagan LM 검정")
        if o["serialTest"]:
            self._safe(lambda: self._serial_table(P), "Wooldridge 자기상관 검정")
        if o["cdTest"]:
            self._safe(lambda: self._cd_table(P, fit, diag_X, names), "Pesaran CD 검정")
        if o["hetTest"]:
            self._safe(lambda: self._het_table(P, fit, diag_X, names), "수정 Wald 이분산 검정")
        if o["plotTrends"]:
            try:
                from .plots import plot_panel_trends
                self.add_plot("개체별 추이 (Panel trajectories)", plot_panel_trends(P.df, self.entity_col, self.time_col, self.dep))
            except Exception:
                pass
        self._narrative(P, fit)

    def _safe(self, fn, label: str):
        try:
            fn()
        except Exception as e:  # diagnostics failing must not hide the estimates; report the reason
            t = ResultsTable(name=f"diag_error_{len(self.tables)}", title=f"{label} — 계산 불가")
            t.add_column("msg", "사유", col_type="text")
            t.add_row({"msg": str(e)})
            self.add_table(t)

    # ------------------------------------------------------------------ tables
    def _summary_table(self, P: PanelData, fit: Fit):
        mt = self.model_type
        t = ResultsTable(name="summary", title=f"모형 요약 ({STATA_CMD[mt]})")
        t.add_column("metric", "항목", col_type="text")
        t.add_column("val", "값", col_type="number")
        rows: List[Tuple[str, Any, str]] = []
        N_used = fit.stats.get("N", P.N if fit.rows is None else len(fit.rows))
        if mt == "fd":
            Ti_used = np.bincount(pd.factorize(fit.g_rows)[0])
        elif mt == "pa":
            Ti_used = fit.stats["Ti"]
        else:
            Ti_used = P.Ti
        rows.append(("관측치 수 (N)", N_used, "integer"))
        rows.append(("집단 수", len(Ti_used), "integer"))
        rows.append(("집단당 관측치 최소", int(Ti_used.min()), "integer"))
        rows.append(("집단당 관측치 평균", float(Ti_used.mean()), ""))
        rows.append(("집단당 관측치 최대", int(Ti_used.max()), "integer"))
        st = fit.stats
        slope_idx = [i for i, nm in enumerate(fit.names) if nm != "Intercept"]
        test = fit.wald(fit.stats.get("test_idx", slope_idx))
        if mt in ("fixed", "twoway", "random", "cre", "between"):
            rows += [("R² 집단 내 (within)", st.get("r2_w"), "zto"), ("R² 집단 간 (between)", st.get("r2_b"), "zto"),
                     ("R² 전체 (overall)", st.get("r2_o"), "zto")]
        if mt in ("fd", "pooled"):
            rows += [("R²", st.get("r2"), "zto"), ("수정 R²", st.get("r2_adj"), "zto"), ("Root MSE", st.get("rmse"), "")]
        if mt in ("fixed", "twoway"):
            rows += [("sigma_u", st["sigma_u"], ""), ("sigma_e", st["sigma_e"], ""),
                     ("rho (u_i 분산 비중)", st["rho"], "zto"), ("corr(u_i, Xb)", st["corr_u_xb"], "")]
        if mt in ("random", "cre"):
            rows += [("sigma_u", st["sigma_u"], ""), ("sigma_e", st["sigma_e"], ""),
                     ("rho (u_i 분산 비중)", st["rho"], "zto")]
        if mt == "cre":
            rows.append(("corr(xit_vars*b, xt_means) — Stata e(corr)", st["corr_xit_means"], ""))
        if mt == "between":
            rows.append(("sd(u_i + avg(e_i.))", st["rmse"], ""))
        if mt == "pa":
            rows += [("척도 모수 (scale)", st["scale"], "")]
        if test is not None:
            if test[2] is None:
                rows += [(f"Wald χ²({test[1]})", test[0], ""), ("Prob > χ²", test[3], "pvalue")]
            else:
                rows += [(f"F({test[1]}, {int(test[2])})", test[0], ""), ("Prob > F", test[3], "pvalue")]
        for lab, v, fmt in rows:
            r = {"metric": lab, "val": None if v is None or (isinstance(v, float) and np.isnan(v)) else v}
            if fmt:
                r["_rowfmt"] = fmt
            t.add_row(r)
        t.add_note(f"추정: {MODEL_LABEL[mt]} — Stata {STATA_CMD[mt]}")
        if mt in ("random", "cre"):
            t.add_note("분산성분: " + ("Swamy–Arora 소표본 추정 (Stata sa 옵션, Baltagi–Chang)" if st["method"] == "sa"
                                    else "Stata 기본 일치 추정 (집단 간 회귀 SSR, 조화평균 T)")
                       + ("; corr(u_i, X) = 0 가정" if mt == "random" else ""))
        if mt == "cre":
            t.add_note("계산 방식: Mundlak (1978)·Wooldridge (2019) 상관 확률효과 — 시변 설명변수의 패널 평균(추정 표본 기준)을 "
                       "추가한 확률효과 GLS (Stata [XT] xtreg, cre Methods and formulas); 시변 변수 계수는 고정효과 추정치와 같음")
            if st.get("cre_ti"):
                t.add_note("시불변 변수(패널 평균 제외): " + ", ".join(st["cre_ti"]))
            t.add_note("모형 Wald 검정: xit_vars 계수(시변·시불변 변수) = 0, 패널 평균 계수 제외 (Stata 동일)")
            t.add_note("corr(xit_vars*b, xt_means): Stata e(corr)와 같은 계산 — x_it·b와 패널 평균 변수 합(Σ x̄_j)의 상관 "
                       "(Stata 출력 표기는 xt_means*γ이나 계산에는 γ를 곱하지 않음, xtreg_cre.ado)")
        if mt == "fd":
            t.add_note("1차 차분은 시점 값 기준 (t-1 시점이 없으면 차분 결측, Stata D. 연산자와 동일)"
                       + ("; 상수항 포함 (regress D.y D.x)" if self.options["fdConstant"] else "; 상수항 제외 (noconstant)"))
        if mt == "between" and st.get("wls"):
            t.add_note("가중최소제곱 (Stata wls 옵션, 가중치 T_i)")
        if P.time_note:
            t.add_note(P.time_note)
        if P.n_dropped:
            t.add_note(f"결측으로 제외된 행: {P.n_dropped}")
        self.add_table(t)
        ms: Dict[str, Any] = {"N": int(N_used), "Groups": int(len(Ti_used)),
                              "Periods": int(len(np.unique(P.tpos)))}
        for k_, key in (("r2_w", "R2_within"), ("r2_b", "R2_between"), ("r2_o", "R2_overall"),
                        ("r2", "R2"), ("r2_adj", "R2_adj")):
            if st.get(k_) is not None and not (isinstance(st.get(k_), float) and np.isnan(st[k_])):
                ms[key] = float(st[k_])
        if mt in ("fixed", "twoway", "random", "cre"):
            ms["Var_between"] = float(st["sigma_u"] ** 2)
            ms["Var_within"] = float(st["sigma_e"] ** 2)
        if test is not None and test[0] is not None:
            ms["Chi2" if test[2] is None else "F"] = (float(test[0]), float(test[3]))
        elif test is not None:
            t.add_note("군집 수에 비해 계수가 많아 분산행렬 계수(rank) 부족 — 모형 결합검정 미보고 (Stata 동일)")
        self.model_stats = ms

    def _coef_tables(self, P: PanelData, fit: Fit, n_time_dummies: int):
        st, p, lo, hi = fit.stat_p_ci()
        lab_stat = "z" if fit.dist == "z" or fit.df_r is None else "t"
        time_names = set(_design(P, True)[1][len(self.covs):]) if self.model_type == "twoway" else set()
        t = ResultsTable(name="coef", title="계수 추정치 (Coefficients)")
        t.add_column("var", "변수", col_type="text")
        t.add_column("b", "추정치", col_type="number")
        t.add_column("se", "표준오차", col_type="number")
        t.add_column("t", lab_stat, col_type="number")
        t.add_column("p", "p", col_type="number", fmt="pvalue")
        t.add_column("cil", "95% CI 하한", col_type="number")
        t.add_column("ciu", "95% CI 상한", col_type="number")
        tt = None
        if time_names and self.options["showTimeEffects"]:
            tt = ResultsTable(name="time_effects", title=f"시점 더미 계수 (i.{self.time_col})")
            for c in t.columns:
                tt.columns.append(dict(c))
        mtab = None
        if self.model_type == "cre":
            t.title = "계수 추정치 — 설명변수 (xit_vars)"
            mtab = ResultsTable(name="cre_means", title="계수 추정치 — 패널 평균 (xt_means)")
            for c in t.columns:
                mtab.columns.append(dict(c))
        prefix = "D." if self.model_type == "fd" else ""
        for i, nm in enumerate(fit.names):
            row = {"var": nm if nm == "Intercept" else prefix + nm, "b": float(fit.b[i]), "se": float(fit.se[i]),
                   "t": float(st[i]), "p": float(p[i]), "cil": float(lo[i]), "ciu": float(hi[i])}
            if nm in time_names:
                if tt is not None:
                    tt.add_row(row)
            elif mtab is not None and nm.startswith(MEAN_PREFIX):
                row["var"] = nm[len(MEAN_PREFIX):]
                mtab.add_row(row)
            else:
                t.add_row(row)
        t.add_note(f"표준오차: {fit.vce_label}" + self._vce_note(fit))
        if lab_stat == "t" and fit.df_r is not None:
            t.add_note(f"t 검정 자유도 {int(fit.df_r)}")
        if fit.omitted:
            t.add_note("공선성으로 제외 (Stata omitted): " + ", ".join(fit.omitted))
        for nt in fit.notes:
            t.add_note(nt)
        t.mark_coefficients(estimate="b", se="se", p="p", stat="t", ci=("cil", "ciu"), label="var")
        self.add_table(t)
        if mtab is not None and mtab.rows:
            mtab.add_note("패널 평균 x̄_i의 계수 γ (Mundlak 회귀); 표준오차 " + fit.vce_label)
            self.add_table(mtab)
        if tt is not None and tt.rows:
            tt.add_note(f"기준 시점: 첫 시점 (Stata i.{self.time_col} 기준 범주); 표준오차 {fit.vce_label}")
            self.add_table(tt)
        elif time_names and not self.options["showTimeEffects"]:
            t.add_note(f"시점 더미 {n_time_dummies}개 포함 (표시 생략)")

    def _vce_note(self, fit: Fit) -> str:
        mt, ct = self.model_type, self.cov_type
        if ct == "robust" and mt in ("fixed", "twoway", "random", "cre"):
            return " — Stata xtreg의 vce(robust)는 패널 변수 군집과 동일, 소표본 보정 (N-1)/(N-K)·G/(G-1)"
        if ct == "robust" and mt in ("fd", "pooled"):
            return " — Stata regress vce(robust), 보정 N/(N-K)"
        if ct == "clustered" and mt != "pa":
            return " — 소표본 보정 (N-1)/(N-K)·G/(G-1) (Stata vce(cluster))"
        if ct == "dk":
            return " — xtscc 방식, 보정 T/(T-1)·(N-1)/(N-K)"
        if ct in ("robust", "clustered") and mt == "pa":
            return " — 보정 G/(G-1) (Stata [P] robust 관례)"
        return ""

    def _time_test_table(self, fit: Fit):
        """testparm i.t after xtreg y x i.t, fe."""
        idx = [i for i, nm in enumerate(fit.names) if nm.endswith("." + self.time_col)]
        if not idx:
            return
        W, q, dropped = wald_pivot(fit.b[idx], fit.V[np.ix_(idx, idx)])
        t = ResultsTable(name="time_test", title=f"시점 효과 결합검정 (testparm i.{self.time_col})")
        z = fit.dist == "z" or fit.df_r is None
        t.add_column("stat", "χ²" if z else "F", col_type="number")
        t.add_column("df1", "df1", col_type="integer")
        if not z:
            t.add_column("df2", "df2", col_type="integer")
        t.add_column("p", "p", col_type="number", fmt="pvalue")
        if q == 0 or not np.isfinite(W):
            t.add_note("검정 불가: 시점 더미 분산행렬 계수(rank) 0")
            self.add_table(t)
            return
        if z:
            t.add_row({"stat": W, "df1": q, "p": float(stats.chi2.sf(W, q))})
        else:
            F = W / q
            t.add_row({"stat": F, "df1": q, "df2": int(fit.df_r), "p": float(stats.f.sf(F, q, fit.df_r))})
        t.add_note(f"H0: 시점 더미 {len(idx)}개 계수 = 0; 표준오차 {fit.vce_label} 기준 Wald 검정 (Stata testparm)")
        if dropped:
            t.add_note(f"분산행렬 계수 부족으로 제약 {len(dropped)}개 제외 (Stata 'Constraint # dropped'): "
                       + ", ".join(str(d) for d in dropped) + " — 남은 분산이 큰 제약부터 선택하는 피벗 콜레스키 방식")
        self.add_table(t)

    def _mundlak_table(self, fit: Fit):
        mu = fit.stats.get("mundlak")
        t = ResultsTable(name="mundlak", title="Mundlak 검정 (xt_means = 0)")
        t.add_column("chi2", "χ²", col_type="number")
        t.add_column("df", "df", col_type="integer")
        t.add_column("p", "p", col_type="number", fmt="pvalue")
        t.add_column("decision", "판단", col_type="text")
        if mu is None or mu[0] is None:
            t.add_note("검정 불가: 패널 평균 계수 분산행렬의 계수(rank) 부족")
        else:
            dec = "확률효과 가정(corr(u_i, x) = 0) 기각 — 고정효과·상관 확률효과 선호 (p < .05)" if mu[3] < 0.05 \
                else "확률효과 가정 기각 안 됨 (p ≥ .05)"
            t.add_row({"chi2": mu[0], "df": mu[1], "p": mu[3], "decision": dec})
        t.add_note("H0: 패널 평균 계수 γ = 0 (Mundlak 1978); 본 모형의 표준오차(" + fit.vce_label + ") 사용 — "
                   "군집 강건 분산에서도 적용 가능한 Hausman 대안 검정 (Wooldridge 2010, 10.7.3; Stata xtreg, cre)")
        self.add_table(t)

    def predictions(self, which: Sequence[str] = PREDICT_STATS) -> pd.DataFrame:
        """Stata predict after xtreg (fe, re, cre): xb, u, e, ue, xbu, stdp — indexed by the input-data rows."""
        mt, fit, P = self.model_type, self.fit, self.panel
        if mt not in PREDICT_MODELS:
            raise ValueError(f"예측값 저장: 고정효과·이원 고정효과·확률효과·상관 확률효과 모형만 지원 (현재: {MODEL_LABEL[mt]})")
        bad = [w for w in which if w not in PREDICT_STATS]
        if bad:
            raise ValueError(f"예측 통계 미지원: {bad} — 가능: {list(PREDICT_STATS)}")
        col = {nm: j for j, nm in enumerate(fit.names_used)}
        slope = [i for i, nm in enumerate(fit.names) if nm != "Intercept"]
        Z = np.column_stack([fit.X_used[:, [col[fit.names[i]] for i in slope]], np.ones(P.N)])
        order = slope + [fit.names.index("Intercept")]
        b, V = fit.b[order], fit.V[np.ix_(order, order)]
        xb = Z @ b
        u = np.asarray(fit.u, float)[P.g]
        # stdp: after xtreg, cre Stata uses the first equation only (xit_vars and _cons, without xt_means)
        sel = [j for j, i in enumerate(order) if not fit.names[i].startswith(MEAN_PREFIX)]
        Zs, Vs = Z[:, sel], V[np.ix_(sel, sel)]
        out = {"xb": xb, "u": u, "ue": P.y - xb, "e": P.y - xb - u, "xbu": xb + u,
               "stdp": np.sqrt(np.clip(np.einsum("ij,jk,ik->i", Zs, Vs, Zs), 0, None))}
        return pd.DataFrame({w: out[w] for w in which}, index=P.row_index)

    def _u_test_table(self, fit: Fit):
        Fu = fit.stats.get("F_u")
        if not Fu:
            return
        t = ResultsTable(name="u_test", title="개체효과 F 검정 (F test that all u_i = 0)")
        t.add_column("F", "F", col_type="number")
        t.add_column("df1", "df1", col_type="integer")
        t.add_column("df2", "df2", col_type="integer")
        t.add_column("p", "p", col_type="number", fmt="pvalue")
        t.add_row({"F": Fu[0], "df1": int(Fu[1]), "df2": int(Fu[2]), "p": Fu[3]})
        t.add_note("H0: 모든 개체효과 u_i = 0 (통합 OLS 대비 고정효과 검정)")
        if self.cov_type != "unadjusted":
            t.add_note("동분산 가정 F 검정 — Stata는 vce(robust)·vce(cluster) 지정 시 이 줄을 표시하지 않음")
        self.add_table(t)

    def _re_variance_table(self, P: PanelData, fit: Fit):
        st = fit.stats
        t = ResultsTable(name="variance", title="분산성분과 θ (Variance components)")
        t.add_column("item", "항목", col_type="text")
        t.add_column("val", "값", col_type="number")
        t.add_row({"item": "sigma_u", "val": st["sigma_u"]})
        t.add_row({"item": "sigma_e", "val": st["sigma_e"]})
        t.add_row({"item": "rho", "val": st["rho"]})
        th = st["theta"]
        if np.allclose(th, th[0]):
            t.add_row({"item": "theta", "val": float(th[0])})
        else:
            tob = st["theta_obs"]
            for lab, pct in (("theta 최소", None), ("theta 5%", 5), ("theta 중앙값", 50), ("theta 95%", 95), ("theta 최대", None)):
                if pct is None:
                    v = float(tob.min()) if "최소" in lab else float(tob.max())
                else:
                    v = stata_percentile(tob, pct)
                t.add_row({"item": lab, "val": v})
            t.add_note("불균형 패널: θ_i 분포 (관측치 기준 백분위수, Stata summarize, detail 정의)")
        if st["s2u_raw"] < 0:
            t.add_note("개체 분산 추정치가 음수여서 0으로 절단 (θ = 0, 통합 OLS와 동일)")
        self.add_table(t)

    def _pa_corr_table(self, fit: Fit):
        st = fit.stats
        t = ResultsTable(name="pa_corr", title=f"작업 상관 구조 (corr({st['corr']}))")
        t.add_column("item", "항목", col_type="text")
        t.add_column("val", "값", col_type="number")
        a = st["alpha"]
        if st["corr"] == "independent":
            t.add_row({"item": "상관 (독립 가정)", "val": 0.0})
        elif st["corr"] in ("exchangeable", "ar1"):
            t.add_row({"item": "α (동일 개체 내 상관)" if st["corr"] == "exchangeable" else "ρ (AR(1) 1차 상관)", "val": float(a)})
        else:
            m = a.shape[0]
            for i in range(m):
                for j in range(i + 1, m):
                    t.add_row({"item": f"R[{i + 1},{j + 1}]", "val": float(a[i, j])})
            t.add_note("비구조 상관의 행·열 번호는 시점 순서 (Stata와의 수치 대조 미실시)")
        t.add_row({"item": "척도 모수 φ (Pearson χ²/N)", "val": st["scale"]})
        t.add_note("Stata xtgee family(gaussian) link(identity) 기본값: 척도 = Pearson χ²/N, 적률 추정")
        self.add_table(t)

    def _hausman_table(self, P: PanelData, X: np.ndarray, names: List[str]):
        sig = self.options.get("hausmanSigma", "none")
        h = hausman_test(P, X, names, sig)
        t = ResultsTable(name="hausman", title="Hausman 검정 (Hausman Specification Test: FE vs RE)")
        t.add_column("chi2", "χ²", col_type="number")
        t.add_column("df", "df", col_type="integer")
        t.add_column("p", "p", col_type="number", fmt="pvalue")
        t.add_column("decision", "판단", col_type="text")
        if h["chi2"] < 0:
            dec = "χ² < 0: 점근 가정 미충족 (Stata 안내: suest 등 대안 검정)"
        else:
            dec = "고정효과 선호 (p < .05)" if h["p"] < 0.05 else "확률효과 일관성 기각 안 됨 (p ≥ .05)"
        t.add_row({"chi2": h["chi2"], "df": h["df"], "p": h["p"], "decision": dec})
        if sig == "more":
            t.add_note(f"sigmamore: 두 분산행렬을 확률효과 모형의 오차 표준편차(e(rmse) = {h['s_re']:.6g})로 계산 "
                       "— V_b·(s_B/s_b)², Stata [R] hausman")
        elif sig == "less":
            t.add_note(f"sigmaless: 두 분산행렬을 고정효과 모형의 오차 표준편차(sigma_e = {h['s_fe']:.6g})로 계산 "
                       "— V_B·(s_b/s_B)², Stata [R] hausman")
        t.add_note("Stata hausman fe re: 두 모형의 일반 분산(동분산 가정)으로 계산, (V_b − V_B)의 일반화 역행렬, df = rank(V_b − V_B)")
        if not h["pos_def"]:
            t.add_note("V_b − V_B가 양정치 아님 (Stata: V_b-V_B is not positive definite)")
        if self.cov_type != "unadjusted":
            t.add_note("본 모형의 강건 표준오차와 무관하게 일반 분산으로 재추정 (Stata hausman은 강건 분산 미지원; "
                       "군집 강건 비교는 상관 확률효과(xtreg, cre)의 Mundlak 검정)")
        if self.model_type == "twoway":
            t.add_note("이원 모형: 두 추정에 같은 시점 더미 포함")
        self.add_table(t)
        ct = ResultsTable(name="hausman_coef", title="Hausman 검정 계수 비교")
        ct.add_column("var", "변수", col_type="text")
        ct.add_column("b", "(b) 고정효과", col_type="number")
        ct.add_column("B", "(B) 확률효과", col_type="number")
        ct.add_column("d", "b − B", col_type="number")
        ct.add_column("se", "sqrt(diag(V_b − V_B))", col_type="number")
        for nm, bf, br, sd in zip(h["terms"], h["b_fe"], h["b_re"], h["se_diff"]):
            ct.add_row({"var": nm, "b": float(bf), "B": float(br), "d": float(bf - br), "se": float(sd)})
        self.add_table(ct)

    def _bp_table(self, P: PanelData, X: np.ndarray):
        r = bp_lm_test(P, X)
        t = ResultsTable(name="bp_lm", title="Breusch–Pagan LM 확률효과 검정 (xttest0)")
        t.add_column("chi2", "chibar2(01)", col_type="number")
        t.add_column("p", "Prob > chibar2", col_type="number", fmt="pvalue")
        t.add_column("var_u", "Var(u)", col_type="number")
        t.add_column("var_e", "Var(e)", col_type="number")
        t.add_row({"chi2": r["lm"], "p": r["p"], "var_u": r["var_u"], "var_e": r["var_e"]})
        t.add_note("H0: Var(u) = 0 (통합 OLS 적합); 통합 OLS 잔차 기준, Baltagi–Li 불균형 패널 공식")
        t.add_note("귀무분포: χ²(0)과 χ²(1)의 50:50 혼합 (단측 검정)")
        self.add_table(t)

    def _serial_table(self, P: PanelData):
        r = wooldridge_serial(P, P.X)
        t = ResultsTable(name="serial", title="Wooldridge 1차 자기상관 검정 (xtserial)")
        t.add_column("F", "F", col_type="number")
        t.add_column("df1", "df1", col_type="integer")
        t.add_column("df2", "df2", col_type="integer")
        t.add_column("p", "p", col_type="number", fmt="pvalue")
        t.add_column("corr", "차분 잔차 1차 계수", col_type="number")
        t.add_row({"F": r["F"], "df1": r["df1"], "df2": r["df2"], "p": r["p"], "corr": r["corr"]})
        t.add_note("H0: 1차 자기상관 없음 (차분 잔차 계수 = −0.5); regress D.y D.x, nocons 후 잔차를 시차 잔차에 회귀, 패널 군집 강건 분산")
        if self.model_type == "twoway":
            t.add_note("시점 더미는 포함하지 않음 (xtserial depvar indepvars 형식)")
        self.add_table(t)

    def _cd_source(self, P: PanelData, fit: Fit, X: np.ndarray, names: List[str]):
        """Results object whose residuals xtcsd would use, and a label."""
        mt = self.model_type
        if fit.cd_result is not None:
            src = {"fixed": "현재 모형의 e_it 잔차 (Stata predict, e)", "twoway": "현재 모형의 e_it 잔차 (Stata predict, e)",
                   "random": "현재 모형의 e_it 잔차 (Stata predict, e)", "cre": "현재 모형의 e_it 잔차 (Stata predict, e)",
                   "fd": "1차 차분 회귀 잔차",
                   "pooled": "현재 모형 잔차 (y − Xb)"}[mt]
            return fit.cd_result, fit.resid_e, (fit.g_rows if mt == "fd" else P.g), (fit.t_rows if mt == "fd" else P.tpos), src
        fe = fit_within(P, X, names)
        why = {"between": "집단 간 추정에는 xtcsd 미적용", "pa": "모집단 평균 모형", "random": "확률효과 sa·DK 경로"}.get(mt, "")
        return fe.cd_result, fe.resid_e, P.g, P.tpos, f"고정효과 모형 e_it 잔차 ({why})"

    def _cd_table(self, P: PanelData, fit: Fit, X: np.ndarray, names: List[str]):
        res, e, g, tp, src = self._cd_source(P, fit, X, names)
        r = pesaran_cd(res)
        t = ResultsTable(name="cd", title="Pesaran CD 횡단면 의존성 검정 (xtcsd, pesaran)")
        t.add_column("cd", "CD", col_type="number")
        t.add_column("p", "p", col_type="number", fmt="pvalue")
        t.add_column("abs", "평균 |상관|", col_type="number")
        t.add_column("n", "개체 수", col_type="integer")
        t.add_row({"cd": r["cd"], "p": r["p"], "abs": mean_abs_corr(e, g, tp), "n": int(len(np.unique(g)))})
        t.add_note("H0: 개체 간 잔차 독립; 개체 쌍별 공통 시점 Pearson 상관, CD ~ N(0,1) 양측 검정 (linearmodels pesaran_cd)")
        t.add_note(f"잔차: {src}")
        if self.model_type == "twoway":
            t.add_note("시점 더미가 있으면 잔차가 시점별로 중심화되어 CD가 하향 편의 (linearmodels 문서 안내)")
        self.add_table(t)

    def _het_table(self, P: PanelData, fit: Fit, X: np.ndarray, names: List[str]):
        if self.model_type in ("fixed", "twoway"):
            e, note = fit.resid_e, ""
        else:
            fe = fit_within(P, X, names)
            e, note = fe.resid_e, "고정효과 모형 잔차로 계산 (xttest3는 xtreg, fe 후 명령)"
        note = (note + "; " if note else "") + "계산 방식: Greene (2000) 수정 Wald, xttest3 v1.0.8 공식 (해당 파이썬 라이브러리 없음)"
        r = modified_wald(e, P.g, P.n)
        t = ResultsTable(name="het", title="집단별 이분산 수정 Wald 검정 (xttest3)")
        t.add_column("chi2", "χ²", col_type="number")
        t.add_column("df", "df", col_type="integer")
        t.add_column("p", "p", col_type="number", fmt="pvalue")
        t.add_row({"chi2": r["chi2"], "df": r["df"], "p": r["p"]})
        t.add_note("H0: 모든 개체의 오차분산 동일 (sigma_i² = sigma²); Greene (2000) 수정 Wald")
        if r["excluded"]:
            t.add_note(f"관측치 1개 또는 분산 0인 개체 {r['excluded']}개는 합에서 제외 (자유도는 집단 수 유지, xttest3 동일)")
        if note:
            t.add_note(note)
        self.add_table(t)

    def _narrative(self, P: PanelData, fit: Fit):
        mt = self.model_type
        slope_idx = [i for i, nm in enumerate(fit.names) if nm != "Intercept"]
        test = fit.wald(fit.stats.get("test_idx", slope_idx))
        n_g = self.model_stats.get("Groups", P.n)
        txt = f"{MODEL_LABEL[mt]} 패널 회귀 (개체 {n_g}개, 관측치 {self.model_stats.get('N')}개)"
        if test is not None and test[0] is not None:
            pv = "< .001" if test[3] < 0.001 else f"= {test[3]:.3f}"
            if test[2] is None:
                txt += f", Wald χ²({test[1]}) = {test[0]:.2f}, p {pv}"
            else:
                txt += f", F({test[1]}, {int(test[2])}) = {test[0]:.2f}, p {pv}"
        if "r2_w" in fit.stats and not np.isnan(fit.stats.get("r2_w", np.nan)):
            txt += f", 집단 내 R² = {fit.stats['r2_w']:.3f}"
        elif "r2" in fit.stats:
            txt += f", R² = {fit.stats['r2']:.3f}"
        mu = self.get_table("mundlak")
        if mu is not None and mu.rows:
            mp = mu.rows[0]["p"]
            txt += f"; Mundlak χ²({mu.rows[0]['df']}) = {mu.rows[0]['chi2']:.2f}, " + ("p < .001" if mp < 0.001 else f"p = {mp:.3f}")
        h = self.get_table("hausman")
        if h is not None and h.rows:
            hr = h.rows[0]
            hp = hr["p"]
            hps = "p < .001" if (hp == hp and hp < 0.001) else f"p = {hp:.3f}"
            txt += f"; Hausman χ²({hr['df']}) = {hr['chi2']:.2f}, {hps}"
        self.apa_narrative = txt + "."


# ----------------------------------------------------------------------------- registry glue
def _params_to_kwargs(df: pd.DataFrame, p: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        entity_col=_as_single(_opt(p, "entity", "entity_col", "idVar", "panelVar", "panel_var")),
        time_col=_as_single(_opt(p, "time", "time_col", "timeVar", "time_var")),
        dep=_as_single(_opt(p, "dep", "depvar")),
        covs=_as_list(_opt(p, "covs", "indeps", "indepvars", default=[])),
        model_type=str(_opt(p, "modelType", "model_type", "model", "estimator", default="fixed")),
        cov_type=str(_opt(p, "covType", "cov_type", "vce", default="unadjusted")),
        hausman=_flag(p, "hausman", default=True),
        plot_trends=_flag(p, "plotTrends", "plot_trends", default=True),
        cluster=_as_single(_opt(p, "cluster", "clusterVar", "cluster_var")),
        dk_lags=_opt(p, "dkLags", "dk_lags", "lag"),
        re_method=str(_opt(p, "reMethod", "re_method", default="default")),
        be_wls=_flag(p, "beWls", "be_wls", "wls", default=False),
        fd_constant=_flag(p, "fdConstant", "fd_constant", default=True),
        pa_corr=str(_opt(p, "paCorr", "pa_corr", "corr", default="exchangeable")),
        bp_test=_flag(p, "bpTest", "bp_test", "xttest0", default=False),
        serial_test=_flag(p, "serialTest", "serial_test", "xtserial", default=False),
        cd_test=_flag(p, "cdTest", "cd_test", "xtcsd", default=False),
        het_test=_flag(p, "hetTest", "het_test", "xttest3", default=False),
        show_time_effects=_flag(p, "showTimeEffects", "show_time_effects", default=True),
        delta=float(_opt(p, "delta", default=1.0)),
        hausman_sigma=str(_opt(p, "hausmanSigma", "hausman_sigma", default="none") or "none"),
        time_test=_flag(p, "timeTest", "time_test", "testparm", default=True),
    )


def dispatch(t: str, df: pd.DataFrame, params: Dict[str, Any]):
    if t != "panel":
        return None
    return PanelAnalysis(df, **_params_to_kwargs(df, params or {}))


def predict_columns(df: pd.DataFrame, stats_map: Dict[str, str], params: Dict[str, Any]) -> pd.DataFrame:
    """Data step of sess.panel_predict: refit the panel model and add Stata predict statistics as new variables
    ({statistic: new variable}); rows outside the estimation sample get missing values (Stata predict, e(sample))."""
    if not stats_map:
        raise ValueError("예측값 저장: {통계: 새 변수 이름} 지정 필요 (예: {'u': 'u_hat'})")
    kw = _params_to_kwargs(df, params or {})
    kw.update(hausman=False, plot_trends=False, bp_test=False, serial_test=False, cd_test=False, het_test=False)
    res = PanelAnalysis(df, **kw)
    pred = res.predictions(list(stats_map))
    out = df.copy()
    for stat, name in stats_map.items():
        if not re.fullmatch(r"[^\W\d]\w*", str(name)):
            raise ValueError(f"예측값 저장: 변수 이름 '{name}' 사용 불가 (문자·숫자·밑줄, 첫 글자는 문자)")
        out[name] = pred[stat].reindex(df.index)
    return out


def predict_stata_lines(stats_map: Dict[str, str], params: Dict[str, Any]) -> List[str]:
    p = dict(params or {}, hausman=False, bpTest=False, serialTest=False, cdTest=False, hetTest=False, timeTest=False)
    lines = stata_code("panel", p) or []
    if lines:
        lines[-1] = "quietly " + lines[-1]
    return lines + [f"predict double {name}, {stat}" for stat, name in stats_map.items()]


def predict_r_lines(stats_map: Dict[str, str], params: Dict[str, Any]) -> List[str]:
    p = dict(params or {}, hausman=False, bpTest=False, serialTest=False, cdTest=False, hetTest=False, timeTest=False)
    lines = [ln for ln in (r_code("panel", p) or []) if not ln.startswith(("summary(", "lmtest::"))]
    want = ", ".join(f"{k}→{v}" for k, v in stats_map.items())
    return lines + [f"# 예측값({want}): fitted(fit)·residuals(fit)·plm::fixef(fit, type = \"dmean\") 등으로 계산 — "
                    "Stata predict u는 관측치 가중 평균 기준이므로 불균형 패널에서는 정의 확인 필요"]


# ----------------------------------------------------------------------------- code export
def xtset_lines(e: str, tm: str, delta: float = 1.0) -> List[str]:
    """xtset that also works when the panel identifier is a string variable."""
    d = f", delta({float(delta):g})" if delta and float(delta) != 1.0 else ""
    return [f"capture confirm numeric variable {e}",
            f"if _rc encode {e}, generate(_panel_id)",
            f"if _rc xtset _panel_id {tm}{d}",
            f"else xtset {e} {tm}{d}"]


def stata_code(t: str, p: Dict[str, Any]) -> Optional[List[str]]:
    if t != "panel":
        return None
    kw = _params_to_kwargs(pd.DataFrame(), p)
    e, tm, y, x = kw["entity_col"], kw["time_col"], kw["dep"], " ".join(kw["covs"])
    mt = resolve_choice("modelType", kw["model_type"], MODEL_TYPES, MODEL_ALIASES, default="fixed")
    ct = resolve_choice("covType", kw["cov_type"], COV_TYPES, COV_ALIASES, default="unadjusted")
    cl = kw["cluster"] or e
    out = xtset_lines(e, tm, kw["delta"])
    lag = f" lag({kw['dk_lags']})" if kw["dk_lags"] not in (None, "") else ""
    if ct == "dk":
        if mt in ("fixed", "twoway"):
            rhs = x + (f" i.{tm}" if mt == "twoway" else "")
            if kw["dk_lags"] in (None, ""):
                out += [f"quietly levelsof {tm}", "local dklag = floor(4*(r(r)/100)^(2/9))   // xtscc 기본 시차"]
                m_ = "`dklag'"
            else:
                m_ = str(kw["dk_lags"])
            out += ["* StataNow 내장 Driscoll–Kraay (이전 버전: ssc install xtscc 후 xtscc ..., fe lag(#))",
                    f"xtreg {y} {rhs}, fe vce(dkraay bartlett {m_})"]
        elif mt == "random":
            out += ["* ssc install xtscc", f"xtscc {y} {x}, re{lag}"]
        elif mt == "pooled":
            out += ["* ssc install xtscc", f"xtscc {y} {x}, pooled{lag}"]
        elif mt == "fd":
            out += ["* ssc install xtscc", f"xtscc D.{y} " + " ".join(f"D.{v}" for v in kw["covs"]) + f", pooled{lag}"]
        else:
            out.append("* Driscoll–Kraay 표준오차는 이 추정 방법에서 지원하지 않음")
    else:
        if mt in ("fixed", "twoway", "random", "pa"):
            vce = {"robust": " vce(robust)", "clustered": f" vce(cluster {cl})"}.get(ct, "")
        else:
            vce = {"robust": " vce(robust)", "clustered": f" vce(cluster {cl})"}.get(ct, "")
        if mt == "fixed":
            out.append(f"xtreg {y} {x}, fe{vce}")
        elif mt == "twoway":
            out.append(f"xtreg {y} {x} i.{tm}, fe{vce}")
            if kw["time_test"]:
                out.append(f"testparm i.{tm}")
        elif mt == "random":
            out.append(f"xtreg {y} {x}, re{' sa' if kw['re_method'] == 'sa' else ''}{vce} theta")
        elif mt == "cre":
            out.append(f"xtreg {y} {x}, cre{' sa' if kw['re_method'] == 'sa' else ''}{vce}   // Stata 19 이상")
        elif mt == "between":
            if ct != "unadjusted":
                out.append("* xtreg, be는 vce(robust)·vce(cluster) 미지원 — 일반 표준오차로 출력")
            out.append(f"xtreg {y} {x}, be{' wls' if kw['be_wls'] else ''}")
        elif mt == "fd":
            out.append(f"regress D.{y} " + " ".join(f"D.{v}" for v in kw["covs"])
                       + ("" if kw["fd_constant"] else ", noconstant") + (("," if kw["fd_constant"] else "") + vce if vce else ""))
        elif mt == "pooled":
            out.append(f"regress {y} {x}" + ("," + vce if vce else ""))
        else:
            corr = {"ar1": "ar 1"}.get(kw["pa_corr"], kw["pa_corr"])
            vce_pa = " vce(robust)" if ct in ("robust", "clustered") else ""
            out.append(f"xtreg {y} {x}, pa corr({corr}){vce_pa}")
    if kw["hausman"]:
        rhs = x + (f" i.{tm}" if mt == "twoway" else "")
        hopt = {"more": ", sigmamore", "less": ", sigmaless"}.get(kw["hausman_sigma"], "")
        out += [f"quietly xtreg {y} {rhs}, fe", "estimates store fe", f"quietly xtreg {y} {rhs}, re",
                "estimates store re", f"hausman fe re{hopt}"]
    if kw["bp_test"]:
        rhs = x + (f" i.{tm}" if mt == "twoway" else "")
        out += [f"quietly xtreg {y} {rhs}, re", "xttest0"]
    if kw["serial_test"]:
        out += ["* ssc install xtserial", f"xtserial {y} {x}"]
    if kw["cd_test"]:
        cmd = "re" if mt == "random" else "fe"
        rhs = x + (f" i.{tm}" if mt == "twoway" else "")
        out += ["* ssc install xtcsd", f"quietly xtreg {y} {rhs}, {cmd}", "xtcsd, pesaran abs"]
    if kw["het_test"]:
        rhs = x + (f" i.{tm}" if mt == "twoway" else "")
        out += ["* ssc install xttest3", f"quietly xtreg {y} {rhs}, fe", "xttest3"]
    return out


def r_code(t: str, p: Dict[str, Any]) -> Optional[List[str]]:
    if t != "panel":
        return None
    kw = _params_to_kwargs(pd.DataFrame(), p)
    e, tm, y = kw["entity_col"], kw["time_col"], kw["dep"]
    mt = resolve_choice("modelType", kw["model_type"], MODEL_TYPES, MODEL_ALIASES, default="fixed")
    ct = resolve_choice("covType", kw["cov_type"], COV_TYPES, COV_ALIASES, default="unadjusted")
    f = f"{y} ~ " + " + ".join(kw["covs"])
    q = lambda s: '"' + str(s) + '"'
    out = [f"pdata <- plm::pdata.frame(data, index = c({q(e)}, {q(tm)}))"]
    if mt == "pa":
        corr = {"exchangeable": "exchangeable", "independent": "independence", "ar1": "ar1", "unstructured": "unstructured"}[kw["pa_corr"]]
        out += [f"# 모집단 평균(GEE) 모형: geepack 패키지 필요 (install.packages(\"geepack\"))",
                f"data <- data[order(data${e}, data${tm}), ]",
                f"fit <- geepack::geeglm({f}, id = {e}, data = data, family = gaussian, corstr = {q(corr)})",
                "summary(fit)  # geepack 기본 표준오차는 강건(sandwich) 분산"]
    else:
        model = {"fixed": "within", "twoway": "within", "random": "random", "cre": "random", "between": "between",
                 "fd": "fd", "pooled": "pooling"}[mt]
        extra = ', effect = "twoways"' if mt == "twoway" else ""
        fm = f
        if mt == "cre":
            fm = f + " + " + " + ".join(f"Between({v})" for v in kw["covs"])
            out.append("# 상관 확률효과(Mundlak): 패널 평균 Between(x) 추가 — 시불변 변수의 Between()은 제외할 것")
        if mt in ("random", "cre"):
            extra += ', random.method = "swar"'
        out.append(f"fit <- plm::plm({fm}, data = pdata, model = {q(model)}{extra})")
        if ct == "unadjusted":
            out.append("summary(fit)")
        elif ct == "dk":
            lag = kw["dk_lags"]
            out.append(f"lmtest::coeftest(fit, vcov = plm::vcovSCC(fit, type = \"HC0\"{'' if lag in (None, '') else f', maxlag = {lag}'}))"
                       "  # Stata xtscc는 추가로 T/(T-1)*(N-1)/(N-K) 보정")
        elif ct == "clustered" and kw["cluster"] and kw["cluster"] != e:
            out.append(f"lmtest::coeftest(fit, vcov = sandwich::vcovCL(fit, cluster = ~{kw['cluster']}, type = \"HC1\"))")
        elif ct == "robust" and mt in ("fd", "pooled", "between"):
            out.append("lmtest::coeftest(fit, vcov = plm::vcovHC(fit, method = \"white1\", type = \"HC1\"))")
        else:
            out.append("lmtest::coeftest(fit, vcov = plm::vcovHC(fit, method = \"arellano\", type = \"HC1\"))"
                       "  # Stata 소표본 보정과 차이 있을 수 있음")
        if mt == "cre":
            out.append("lmtest::waldtest(fit, " + ", ".join(f'"Between({v})"' for v in kw["covs"])
                       + ")  # Mundlak 검정 (강건 분산이면 vcov = 인수 지정)")
        if mt == "twoway" and kw["time_test"]:
            out.append(f"plm::pFtest(fit, plm::plm({f}, data = pdata, model = \"within\"))  # 시점 효과 F 검정 (일반 분산 testparm과 같음)")
        if mt in ("random", "cre") and kw["re_method"] == "default":
            out.append("# 불균형 패널에서 Stata 기본 분산성분(조화평균 T)은 plm swar(= Stata sa)와 다름")
    if kw["hausman"]:
        efx = ', effect = "twoways"' if mt == "twoway" else ""
        out += [f"fe <- plm::plm({f}, data = pdata, model = \"within\"{efx})",
                f"re <- plm::plm({f}, data = pdata, model = \"random\"{efx})", "plm::phtest(fe, re)"
                + ("  # sigmamore·sigmaless 옵션에 해당하는 인수 없음" if kw["hausman_sigma"] != "none" else "")]
    if kw["bp_test"]:
        out.append(f"plm::plmtest(plm::plm({f}, data = pdata, model = \"pooling\"), type = \"bp\")")
    if kw["serial_test"]:
        out.append(f"plm::pwfdtest({f}, data = pdata)  # Stata xtserial과 같은 검정, 분산 보정 차이 가능")
    if kw["cd_test"]:
        out.append(f"plm::pcdtest(plm::plm({f}, data = pdata, model = {q('random' if mt == 'random' else 'within')}), test = \"cd\")")
    if kw["het_test"]:
        out.append("# xttest3(수정 Wald 집단별 이분산)에 해당하는 plm 함수 없음")
    return out


# ----------------------------------------------------------------------------- Stata command input
def _split_opts(opt_txt: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    i, s = 0, opt_txt or ""
    while i < len(s):
        while i < len(s) and s[i] in " \t,":
            i += 1
        j = i
        while j < len(s) and s[j] not in " \t,(":
            j += 1
        name = s[i:j].lower()
        val = ""
        if j < len(s) and s[j] == "(":
            depth, k = 1, j + 1
            while k < len(s) and depth:
                depth += {"(": 1, ")": -1}.get(s[k], 0)
                k += 1
            val = s[j + 1:k - 1].strip()
            j = k
        if name:
            out[name] = val
        i = j
    return out


_LAST_PANEL = "next(a for a in reversed(sess.history) if getattr(a, 'name', '') == 'panel')"


def _err(msg: str) -> str:
    return f"raise NotImplementedError({msg!r})"


def _xt_index_args() -> str:
    return ("entity=getattr(sess, 'panel_spec', {}).get('idVar'), "
            "time=getattr(sess, 'panel_spec', {}).get('timeVar')")


def _parse_vce(v: str, line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """vce() text -> (covType, cluster, error)."""
    toks = v.split()
    if not toks:
        return None, None, None
    k = toks[0].lower()
    if k in ("conventional", "ols", "gmm"):
        return "unadjusted", None, None
    if "robust".startswith(k) and len(k) >= 1:
        return "robust", None, None
    if "cluster".startswith(k) and len(k) >= 2:
        if len(toks) != 2:
            return None, None, _err("vce(cluster): 군집 변수 1개만 지원 — " + line)
        return "clustered", toks[1], None
    return None, None, _err(f"vce({v}) 미지원 — " + line)


def transpile(line: str) -> Optional[str]:
    s = line.strip()
    low = s.lower()
    m = re.match(r"^(xtreg|xtgee)\s+([^,]+?)\s*(?:,\s*(.*))?$", s, re.IGNORECASE)
    if m:
        cmd = m.group(1).lower()
        toks = m.group(2).split()
        if len(toks) < 2:
            return _err(f"Stata {cmd}: 종속변수와 설명변수 필요 — " + s)
        dep, rhs = toks[0], toks[1:]
        opts = _split_opts(m.group(3) or "")
        mt = None
        time_dummy = False
        covs = []
        for v in rhs:
            mm = re.fullmatch(r"i\.(\w+)", v, re.IGNORECASE)
            if mm:
                time_dummy = mm.group(1)
                continue
            if re.search(r"[.#]", v):
                return _err("요인변수·시계열 연산자(i.·c.·L.·D.·#) 미지원: 변수를 만든 뒤 지정 — " + s)
            covs.append(v)
        if cmd == "xtgee":
            fam = opts.pop("family", opts.pop("fam", "gaussian")).lower()
            link = opts.pop("link", "identity").lower()
            if not fam.startswith("gau") or not link.startswith("iden"):
                return _err("xtgee: family(gaussian) link(identity)만 지원 — " + s)
            mt = "pa"
        for key in ("fe", "re", "be", "pa", "mle", "cre"):
            if key in opts:
                if key == "mle":
                    return _err("xtreg, mle 미지원 (확률효과는 re 옵션: GLS) — " + s)
                mt = {"fe": "fixed", "re": "random", "be": "between", "pa": "pa", "cre": "cre"}[key]
                opts.pop(key)
        mt = mt or "random"     # Stata xtreg default = re
        kw = [f"dep={dep!r}", f"covs={covs!r}"]
        idx_args = _xt_index_args()
        if time_dummy:
            if mt != "fixed":
                return _err("i.시점 더미는 xtreg, fe와 함께만 지원 (이원 고정효과) — " + s)
            mt = "twoway"
            idx_args = f"entity=getattr(sess, 'panel_spec', {{}}).get('idVar'), time={time_dummy!r}"
        kw.append(f"modelType={mt!r}")
        ct = None
        if "vce" in opts:
            ct, cl, err = _parse_vce(opts.pop("vce"), s)
            if err:
                return err
            if cl:
                kw.append(f"cluster={cl!r}")
        if "robust" in opts:
            opts.pop("robust")
            ct = "robust"
        if "cluster" in opts:
            ct = "clustered"
            kw.append(f"cluster={opts.pop('cluster')!r}")
        if ct:
            kw.append(f"covType={ct!r}")
        if "sa" in opts:
            opts.pop("sa")
            kw.append("reMethod='sa'")
        if "wls" in opts:
            opts.pop("wls")
            kw.append("beWls=True")
        opts.pop("theta", None)
        for ck in ("corr", "correlation", "c"):
            if ck in opts:
                cv = opts.pop(ck).lower().replace(" ", "")
                cmap = {"exchangeable": "exchangeable", "exc": "exchangeable", "independent": "independent",
                        "ind": "independent", "ar1": "ar1", "unstructured": "unstructured", "uns": "unstructured"}
                cv2 = next((v for k_, v in cmap.items() if cv.startswith(k_)), None)
                if cv2 is None:
                    return _err(f"corr({cv}) 미지원 (exchangeable·independent·ar 1·unstructured) — " + s)
                kw.append(f"paCorr={cv2!r}")
        if opts:
            return _err(f"{cmd} 옵션 미지원: {', '.join(opts)} — " + s)
        return f"res = sess.run('panel', {idx_args}, {', '.join(kw)})"
    m = re.match(r"^xtscc\s+([^,]+?)\s*(?:,\s*(.*))?$", s, re.IGNORECASE)
    if m:
        toks = m.group(1).split()
        opts = _split_opts(m.group(2) or "")
        if len(toks) < 2 or any(re.search(r"[.#]", v) for v in toks):
            return _err("xtscc: 종속변수·설명변수(연산자 없이) 필요 — " + s)
        mt = "fixed" if "fe" in opts else "random" if "re" in opts else "pooled"
        for k_ in ("fe", "re", "pooled", "pool"):
            opts.pop(k_, None)
        kw = [f"dep={toks[0]!r}", f"covs={toks[1:]!r}", f"modelType={mt!r}", "covType='dk'"]
        if "lag" in opts:
            kw.append(f"dkLags={int(opts.pop('lag'))}")
        if opts:
            return _err(f"xtscc 옵션 미지원: {', '.join(opts)} — " + s)
        return f"res = sess.run('panel', {_xt_index_args()}, {', '.join(kw)})"
    diag = {"xttest0": ("bpTest", "Breusch–Pagan LM"), "xttest3": ("hetTest", "수정 Wald 이분산"),
            "xtcsd": ("cdTest", "Pesaran CD")}
    m = re.match(r"^(xttest0|xttest3|xtcsd)\b(.*)$", s, re.IGNORECASE)
    if m:
        key, lab = diag[m.group(1).lower()]
        rest = m.group(2).strip().lstrip(",").strip().lower()
        if m.group(1).lower() == "xtcsd" and rest and not set(rest.replace(",", " ").split()) <= {"pesaran", "pes", "abs", "show"}:
            return _err("xtcsd: pesaran 검정만 지원 (frees·friedman 미지원) — " + s)
        return (f"res = sess.run('panel', **dict({_LAST_PANEL}.options, {key}=True))"
                f"  # [Stata] {s} → 직전 패널 모형에 {lab} 검정 추가")
    m = re.match(r"^xtserial\s+(\S+)\s+([^,]+?)\s*(?:,\s*(.*))?$", s, re.IGNORECASE)
    if m:
        return (f"res = sess.run('panel', {_xt_index_args()}, dep={m.group(1)!r}, covs={m.group(2).split()!r}, "
                f"modelType='fd', serialTest=True, hausman=False)  # [Stata] {s}")
    m = re.match(r"^hausman\s+fe\s+re\b\s*(?:,\s*(.*))?$", low)
    if m:
        hopts = set((m.group(1) or "").replace(",", " ").split())
        sig = "more" if "sigmamore" in hopts else "less" if "sigmaless" in hopts else "none"
        other = hopts - {"sigmamore", "sigmaless", "constant", "alleqs"}
        if other:
            return _err(f"hausman 옵션 미지원: {', '.join(sorted(other))} (sigmamore·sigmaless 지원) — " + s)
        if sig != "none":
            return (f"res = sess.run('panel', **dict({_LAST_PANEL}.options, hausman=True, hausmanSigma={sig!r}))"
                    f"  # [Stata] {s} → Hausman 검정은 선형 패널 모형 결과에 포함")
        return ("_h = next(a for a in reversed(sess.history) if getattr(a, 'name', '') in ('panel', 'panelNonlinear'))\n"
                f"res = sess.run(_h.name, **dict(_h.options, hausman=True))"
                f"  # [Stata] {s} → 직전 패널 모형(xtreg·xtlogit·xtpoisson) 결과에 Hausman 검정 포함")
    m = re.match(r"^testparm\s+i\.(\w+)\s*$", s, re.IGNORECASE)
    if m:
        return (f"res = sess.run('panel', **dict({_LAST_PANEL}.options, timeTest=True))"
                f"  # [Stata] {s} → 이원 고정효과(xtreg ... i.{m.group(1)}, fe) 결과의 시점 효과 결합검정")
    m = re.match(r"^predict\s+(?:double\s+|float\s+)?([A-Za-z_]\w*)\s*(?:,\s*(\w+)\s*)?$", s, re.IGNORECASE)
    if m:
        stat = (m.group(2) or "xb").lower()
        if stat not in PREDICT_STATS:
            return _err(f"predict 옵션 미지원: {stat} (가능: {', '.join(PREDICT_STATS)}) — " + s)
        return f"sess.predict({m.group(1)!r}, {stat!r})  # [Stata] {s}"
    return None


EXTENSION = {
    "catalog": [{"id": "panel", "group": "Linear Models",
                 "title": "Panel Regression (xtreg fe/re/be/fd/pa, robust·cluster·Driscoll–Kraay, diagnostics)",
                 "module": "openstat", "icon": "analysis-linearmodels.svg"}],
    "aliases": {"xtreg": "panel", "panel_regression": "panel", "panelRegression": "panel", "xtscc": "panel",
                "xtgee": "panel", "plm": "panel"},
    "dispatch": dispatch,
    "stata": stata_code,
    "r": r_code,
    "transpile": transpile,
}
