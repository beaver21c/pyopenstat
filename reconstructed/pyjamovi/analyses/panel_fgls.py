"""
OpenStat Studio — linear panel estimators with non-spherical disturbances (Stata xtgls · xtpcse · xtregar) and the
Hausman–Taylor / Amemiya–MaCurdy instrumental-variables random-effects estimator (Stata xthtaylor).
License: MIT

Two analysis ids:

    panelGLS        option ``method`` = xtgls | xtpcse | xtregar; the name typed (xtgls, xtpcse, xtregar) presets it
    hausmanTaylor   Stata xthtaylor (alias ``xthtaylor``)

No Python library offers these estimators (searched linearmodels 7.1: PanelOLS, RandomEffects, IV2SLS;
statsmodels 0.15: GLS, WLS, GLSAR).  The regressions run in statsmodels / linearmodels; the transformations, the
disturbance-covariance estimators and the small-sample conventions follow the cited papers and Stata's
Methods and formulas:

estimator / quantity                     computation                                                         Stata
xtgls  β, V = (X'Ω⁻¹X)⁻¹                 statsmodels WLS (diagonal Σ) or OLS on data whitened by Σ⁻¹ᐟ² ⊗ I    xtgls
       Σ̂ (iid / heteroskedastic /         Σ̂ᵢⱼ = eᵢ'eⱼ / T (Greene 2018 ch. 11; Stata [XT] xtgls Methods and     panels()
       correlated), igls, LL, nmk        formulas); generalized inverse in Stata's syminv() order when T < m
       AR(1) common / panel-specific      Prais–Winsten per panel, ρ̂ by the [TS] prais rhotype formulas,       corr() rhotype()
                                          common ρ̂ = Σ Tᵢ ρ̂ᵢ / Σ Tᵢ
xtpcse β (OLS or Prais–Winsten)           statsmodels OLS on the (transformed) data                             xtpcse
       PCSE V = (X'X)⁻¹X'ΩX(X'X)⁻¹        Beck & Katz (1995); Stata [XT] xtpcse Methods and formulas            hetonly
       Σ̂ casewise / pairwise             (Σ̂ᵢⱼ over common periods, ρ̂ bounded to [−1, 1], common ρ̂ weighted    independent
                                          by the pairs used, np1)                                               casewise pairwise np1
xtregar ρ̂ (dw regress freg tscorr theil   iterated Prais–Winsten on within-demeaned data ([TS] prais) or the    xtregar rhotype()
       nagar onestep), rhof, twostep      Baltagi & Wu (1999) one-step estimator                                rhof() twostep
       FE within estimator                Baltagi & Wu (1999) Cᵢ(ρ) transformation for unequally spaced data,   xtregar, fe
                                          statsmodels OLS on the within-transformed data, Stata df conventions
       RE GLS                             Baltagi & Wu (1999) variance components and θᵢ transformation,       xtregar, re
                                          statsmodels OLS
       LBI test                           Bhargava, Franzini & Narendranathan (1982) modified DW and           lbi
                                          Baltagi–Wu (1999) LBI
xthtaylor                                 Hausman & Taylor (1981), Amemiya & MaCurdy (1986); within step         xthtaylor
                                          statsmodels OLS, IV steps linearmodels IV2SLS; σ²ᵤ, θᵢ as in           amacurdy small
                                          Stata [XT] xthtaylor Methods and formulas                             vce(robust)
Hausman FE vs HT                         Hausman (1978) with the SVD generalized inverse of [R] hausman          hausman fe ht

Verified against Stata 19.5 (StataNow MP) on sample_data/Grunfeld_Panel.csv (balanced, an unbalanced variant with
internal gaps and one with few gaps) and sample_data/Panel_HT_Wage.csv; reference values are hard-coded in
tests/test_panel_fgls.py, Stata do-files and logs kept outside the repository.

Known differences and scope
    xtregar   Stata stores the Cᵢ(ρ)-transformed variables in float precision; this module keeps double precision
              (relative differences ~1e-7).  Stata's re header shows Wald χ² with df = K + 1 (constant counted) and
              computes Prob with that df; the table follows Stata and the note gives p for the K slopes.
    xtgls     corr(unstructured) (StataNow), weights and level() are not offered; the log likelihood is not reported
              when Σ̂ is singular (Stata prints a value driven by det(Σ̂) ≈ 0).
    xthtaylor hausman between Hausman–Taylor and Amemiya–MaCurdy is not offered; vce(cluster other) is not offered.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseAnalysis, ResultsTable, resolve_choice
from .panel_module import (PanelData, _corr, _err, _split_opts, _xt_index_args, independent_columns, lag_index,
                           stata_percentile, xtset_lines)

# ----------------------------------------------------------------------------- option vocabularies
METHODS = ("xtgls", "xtpcse", "xtregar")
METHOD_LABEL = {"xtgls": "패널 FGLS (xtgls)", "xtpcse": "패널 수정 표준오차 PCSE (xtpcse)",
                "xtregar": "AR(1) 오차 고정·확률효과 (xtregar)"}
PANELS = ("iid", "heteroskedastic", "correlated")
CORRS = ("independent", "ar1", "psar1")
PCSE_TYPES = ("correlated", "hetonly", "independent")
RHOTYPES = ("regress", "freg", "tscorr", "dw", "theil", "nagar", "onestep")
RHO_ALLOWED = {"xtgls": ("regress", "dw", "freg", "nagar", "theil", "tscorr"),
               "xtpcse": ("regress", "freg", "tscorr", "dw"),
               "xtregar": RHOTYPES}
RHO_DEFAULT = {"xtgls": "regress", "xtpcse": "regress", "xtregar": "dw"}
ALIAS_PRESETS = {"xtgls": "xtgls", "xtpcse": "xtpcse", "xtregar": "xtregar", "panelgls": None}
PANELS_LABEL = {"iid": "동분산 (panels(iid))", "heteroskedastic": "패널별 이분산 (panels(heteroskedastic))",
                "correlated": "이분산 + 패널 간 동시 상관 (panels(correlated))"}
CORR_LABEL = {"independent": "자기상관 없음", "ar1": "공통 AR(1)", "psar1": "패널별 AR(1)"}
PCSE_LABEL = {"correlated": "패널 간 동시 상관 + 이분산 (PCSE)", "hetonly": "패널별 이분산만 (hetonly)",
              "independent": "독립·동분산 (independent)"}
RHO_LABEL = {"regress": "시차 회귀", "freg": "선행 회귀", "tscorr": "시계열 자기상관", "dw": "Durbin–Watson",
             "theil": "Theil", "nagar": "Nagar", "onestep": "Baltagi–Wu 1단계"}


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
    return None if val == "" else val


def _prefix_choice(label: str, value: Any, allowed: Sequence[str], aliases: Optional[Dict[str, str]] = None,
                   default: Optional[str] = None) -> str:
    """Select option with Stata's minimal-abbreviation rule (panels(h), corr(ps)) on top of resolve_choice."""
    if isinstance(value, str) and value.strip():
        low = value.strip().lower()
        if aliases and low in aliases:
            return aliases[low]
        hits = [a for a in allowed if a.startswith(low)]
        if len(hits) == 1:
            return hits[0]
    return resolve_choice(label, value, allowed, aliases, default=default)


_PANEL_ALIASES = {"hetero": "heteroskedastic", "heteroscedastic": "heteroskedastic", "het": "heteroskedastic",
                  "h": "heteroskedastic", "corr": "correlated", "c": "correlated", "homoskedastic": "iid", "i": "iid"}
_CORR_ALIASES = {"ind": "independent", "i": "independent", "none": "independent", "ar": "ar1", "ar(1)": "ar1",
                 "common": "ar1", "ps": "psar1", "p": "psar1", "panel": "psar1"}
_PCSE_ALIASES = {"pcse": "correlated", "default": "correlated", "het": "hetonly", "hetero": "hetonly",
                 "heteroskedastic": "hetonly", "indep": "independent", "ind": "independent"}
_RHO_ALIASES = {"reg": "regress", "durbin": "dw", "ts": "tscorr", "one": "onestep"}


def normalize_options(params: Dict[str, Any], preset: Optional[str] = None) -> Dict[str, Any]:
    """UI / sess.run parameters -> validated canonical options of panelGLS (unknown select values raise)."""
    p = params or {}
    m_raw = _opt(p, "method", "estimator", "command")
    if m_raw is None:
        method = preset or "xtgls"
    else:
        method = _prefix_choice("추정 명령(method)", m_raw, METHODS)
        if preset and method != preset:
            raise ValueError(f"분석 이름과 추정 명령 불일치: 분석 이름은 '{preset}', method='{m_raw}' — 둘 중 하나만 지정")
    panels = _prefix_choice("패널 간 오차 구조(panels)", _opt(p, "panels", "panel", default="iid"), PANELS, _PANEL_ALIASES)
    corr = _prefix_choice("패널 내 자기상관(corr)", _opt(p, "corr", "correlation", "c", default="independent"), CORRS,
                          _CORR_ALIASES)
    pcse = _prefix_choice("PCSE 공분산 구조(pcse)", _opt(p, "pcse", "pcseType", default="correlated"), PCSE_TYPES,
                          _PCSE_ALIASES)
    if _flag(p, "hetonly"):
        pcse = "hetonly"
    if _flag(p, "independent"):
        if pcse == "hetonly":
            raise ValueError("hetonly와 independent는 함께 지정할 수 없음 (Stata xtpcse 규칙과 동일)")
        pcse = "independent"
    if _flag(p, "pairwise") and _flag(p, "casewise"):
        raise ValueError("casewise와 pairwise는 함께 지정할 수 없음 (Stata xtpcse 규칙과 동일)")
    model = _prefix_choice("모형(model)", _opt(p, "model", "effects", default="re"), ("fe", "re"),
                           {"fixed": "fe", "random": "re", "within": "fe", "gls": "re"})
    rt_raw = _opt(p, "rhotype", "rhoType", "rho_type")
    if isinstance(rt_raw, str) and rt_raw.strip().lower() in ("default", "auto", "기본"):
        rt_raw = None                           # form value for "the command's default method"
    rhof = _opt(p, "rhof", "rhoFixed", "rho_fixed")
    if rhof is not None:
        try:
            rhof = float(rhof)
        except (TypeError, ValueError):
            raise ValueError(f"고정 rho(rhof): 숫자 필요 (입력 '{rhof}')")
    if rt_raw is None:
        rhotype = RHO_DEFAULT[method]
    else:
        rhotype = _prefix_choice("rho 추정 방법(rhotype)", rt_raw, RHOTYPES, _RHO_ALIASES)
        if rhotype not in RHO_ALLOWED[method]:
            raise ValueError(f"{method}: rhotype({rhotype}) 사용 불가 (가능: {', '.join(RHO_ALLOWED[method])})")
    iterate = int(_opt(p, "iterate", "maxIter", "max_iter", default=300))
    if iterate < 1:
        raise ValueError("반복 횟수(iterate)는 1 이상 정수")
    tol = float(_opt(p, "tolerance", "tol", default=1e-7))
    if not tol > 0:
        raise ValueError("수렴 기준(tolerance)은 양수")
    constant = _flag(p, "constant", default=not _flag(p, "noconstant", "nocons"))
    o = {
        "entity": _as_single(_opt(p, "entity", "entity_col", "idVar", "panelVar", "panel_var")),
        "time": _as_single(_opt(p, "time", "time_col", "timeVar", "time_var")),
        "dep": _as_single(_opt(p, "dep", "depvar")),
        "covs": _as_list(_opt(p, "covs", "indeps", "indepvars", default=[])),
        "method": method, "panels": panels, "corr": corr, "rhotype": rhotype,
        "rhotypeSet": rt_raw is not None,
        "igls": _flag(p, "igls"), "iterate": iterate, "tolerance": tol, "force": _flag(p, "force"),
        "nmk": _flag(p, "nmk"), "pcse": pcse, "pairwise": _flag(p, "pairwise"), "np1": _flag(p, "np1"),
        "model": model, "rhof": rhof, "twostep": _flag(p, "twostep", "twoStep"), "lbi": _flag(p, "lbi"),
        "constant": constant, "delta": float(_opt(p, "delta", default=1.0)),
    }
    if method in ("xtgls", "xtpcse") and o["rhotypeSet"] and corr == "independent":
        o["rhotypeSet"] = False                 # no autocorrelation parameter to estimate (Stata xtgls: error 198)
        o["rhotype"] = RHO_DEFAULT[method]
    if method == "xtregar":
        if rhof is not None and o["rhotypeSet"]:
            raise ValueError("xtregar: rhotype()와 rhof()는 함께 지정할 수 없음 (Stata 동일)")
        if rhof is not None and abs(rhof) >= 1:
            raise ValueError("xtregar: |rhof| < 1 필요 (Stata 동일)")
        if rhotype == "onestep" and o["twostep"]:
            raise ValueError("xtregar: onestep과 twostep은 함께 지정할 수 없음 (Stata 동일)")
        if not constant:
            raise ValueError("xtregar: 상수항 제외(noconstant) 옵션 없음 (Stata 동일)")
    return o


# ----------------------------------------------------------------------------- shared numerics
def _ols(y: np.ndarray, X: np.ndarray):
    """statsmodels OLS without an added constant (the design carries its own constant column)."""
    import statsmodels.api as sm
    return sm.OLS(y, X).fit()


def _rho_ar1(e: np.ndarray, e_lag: np.ndarray, e_lead: np.ndarray, method: str, k: int) -> Tuple[float, int]:
    """AR(1) coefficient of one residual series by the [TS] prais rhotype formulas; lags/leads are NaN at panel
    starts and time gaps.  Returns (rho, pairs used); rho is NaN where Stata returns missing."""
    has = ~np.isnan(e_lag)
    m = int(has.sum())
    n = len(e)
    with np.errstate(divide="ignore", invalid="ignore"):
        if method == "regress":
            den = float(np.sum(e_lag[has] ** 2))
            rho = float(np.sum(e[has] * e_lag[has])) / den if den > 0 else np.nan
        elif method == "freg":
            hf = ~np.isnan(e_lead)
            den = float(np.sum(e_lead[hf] ** 2))
            rho = float(np.sum(e[hf] * e_lead[hf])) / den if den > 0 else np.nan
            m = int(hf.sum())
        else:
            sse = float(np.sum(e ** 2))
            if sse <= 0:
                return np.nan, m
            if method in ("dw", "nagar"):
                dwv = float(np.sum((e[has] - e_lag[has]) ** 2)) / sse
                rho = 1 - dwv / 2 if method == "dw" else (n ** 2 * (1 - dwv / 2) + k ** 2) / (n ** 2 - k ** 2)
            else:
                cov = float(np.sum(e[has] * e_lag[has]))
                rho = cov / sse if method == "tscorr" else ((n - k) * cov) / (m * sse) if m else np.nan
    return (float(rho) if np.isfinite(rho) else np.nan), m


def _shift(a: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = np.full(len(idx), np.nan)
    ok = idx >= 0
    out[ok] = a[idx[ok]]
    return out


def _pw(A: np.ndarray, prev: np.ndarray, rho_obs: np.ndarray) -> np.ndarray:
    """Prais–Winsten transformation: a_t − ρ a_{t−1} where the previous observation exists, √(1−ρ²) a_t otherwise."""
    A2 = A if A.ndim == 2 else A[:, None]
    out = np.empty_like(A2, dtype=float)
    has = prev >= 0
    r = rho_obs[:, None]
    with np.errstate(invalid="ignore"):
        out[~has] = A2[~has] * np.sqrt(1 - r[~has] ** 2)
    out[has] = A2[has] - r[has] * A2[prev[has]]
    return out if A.ndim == 2 else out[:, 0]


def _syminv(S: np.ndarray, tol: float = 1e-10) -> Tuple[np.ndarray, int]:
    """Generalized inverse of a symmetric positive-semidefinite matrix as Stata syminv()/invsym() returns it: columns
    are swept largest remaining pivot first; columns whose remaining pivot is ≤ tol × original diagonal are set to 0
    (rows and columns) and the remaining block is inverted.  Returns (inverse, number of zeroed columns)."""
    A = np.array(S, dtype=float)
    n = A.shape[0]
    orig = np.abs(np.diag(A)).copy()
    rem = list(range(n))
    kept: List[int] = []
    while rem:
        d = np.array([A[j, j] for j in rem])
        ok = d > tol * np.where(orig[rem] > 0, orig[rem], 1.0)
        if not ok.any():
            break
        j = rem[int(np.argmax(np.where(ok, d, -np.inf)))]
        A = A - np.outer(A[:, j], A[j, :]) / A[j, j]
        kept.append(j)
        rem.remove(j)
    kept.sort()
    out = np.zeros((n, n))
    if kept:
        ix = np.ix_(kept, kept)
        out[ix] = np.linalg.inv(np.asarray(S, float)[ix])
    return out, n - len(kept)


def _wald(b: np.ndarray, V: np.ndarray, idx: Sequence[int]) -> Tuple[Optional[float], int]:
    idx = list(idx)
    if not idx:
        return None, 0
    Vs = V[np.ix_(idx, idx)]
    try:
        W = float(b[idx] @ np.linalg.solve(Vs, b[idx]))
    except np.linalg.LinAlgError:
        W = float(b[idx] @ np.linalg.pinv(Vs) @ b[idx])
    return W, len(idx)


def _mreldif(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / (np.abs(b) + 1.0))) if len(a) else 0.0


def _gaps(g: np.ndarray, tpos: np.ndarray) -> int:
    """Number of internal gaps (Stata tsreport, panel)."""
    same = g[1:] == g[:-1]
    return int(np.sum(same & (np.diff(tpos) > 1)))


class Est:
    """Coefficients, VCE and reported statistics of one estimator."""

    def __init__(self):
        self.names: List[str] = []
        self.b = np.zeros(0)
        self.V = np.zeros((0, 0))
        self.dist = "z"
        self.df_r: Optional[float] = None
        self.stats: Dict[str, Any] = {}
        self.notes: List[str] = []
        self.omitted: List[str] = []
        self.blocks: Optional[List[str]] = None
        self.lib = ""

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.V), 0, None))


def _design(P: PanelData, constant: bool) -> Tuple[np.ndarray, List[str], List[str]]:
    """Regressors (+ _cons) without collinear columns; later columns of a collinear set are dropped and _cons kept,
    as Stata _rmcoll does."""
    X = P.X
    names = list(P.covs)
    if constant:
        M = np.column_stack([X, np.ones(P.N)])
        keep = independent_columns(M, protect=[M.shape[1] - 1])
        kept = [j for j in keep if j < X.shape[1]]
        omitted = [names[j] for j in range(len(names)) if j not in kept]
        return np.column_stack([X[:, kept], np.ones(P.N)]), [names[j] for j in kept] + ["_cons"], omitted
    keep = independent_columns(X)
    omitted = [names[j] for j in range(len(names)) if j not in keep]
    return X[:, keep], [names[j] for j in keep], omitted


def _subset_panel(P: PanelData, rows: np.ndarray) -> None:
    """Restrict the panel arrays to `rows` in place (keeps the sort order)."""
    codes = P.g[rows]
    uniq = np.unique(codes)
    P.entity_labels = P.entity_labels[uniq]
    P.g = np.searchsorted(uniq, codes)
    P.df = P.df.iloc[rows].reset_index(drop=True)
    if getattr(P, "row_index", None) is not None:
        P.row_index = P.row_index[rows]
    P.tpos, P.y, P.X = P.tpos[rows], P.y[rows], P.X[rows]
    P.time_values = P.time_values[rows]
    P.N = len(rows)
    P.n = len(uniq)
    P.Ti = np.bincount(P.g, minlength=P.n)


# ----------------------------------------------------------------------------- xtgls
def fit_xtgls(P: PanelData, o: Dict[str, Any], time_given: bool = True) -> Est:
    """Stata xtgls: two-step (three-step with AR(1)) or iterated FGLS; see [XT] xtgls Methods and formulas."""
    est = Est()
    panels, corr, rt = o["panels"], o["corr"], o["rhotype"]
    if corr != "independent" or panels == "correlated":
        if not time_given:
            raise ValueError("xtgls: corr(ar1)·corr(psar1)·panels(correlated)에는 시점 변수 필요 (xtset 개체 시점)")
    if corr != "independent":
        single = P.Ti[P.g] == 1
        if single.any():
            est.notes.append(f"관측치 {int(single.sum())}개 제외: 집단 내 관측치 1개 (Stata 동일)")
            _subset_panel(P, np.flatnonzero(~single))
    X, names, omitted = _design(P, o["constant"])
    est.omitted = omitted
    g, y, N, m = P.g, P.y, P.N, P.n
    first = np.r_[True, g[1:] != g[:-1]]
    pos = np.arange(N) - np.flatnonzero(first)[g]               # order within panel (force: lags follow it)
    if corr != "independent" and not o["force"]:
        d = np.diff(P.tpos)[~first[1:]]
        if len(d) and (d.min() != 1 or d.max() != 1):
            raise ValueError("xtgls: 시점 간격이 일정하지 않음 (Stata: time variable is not regularly spaced) — "
                             "force 옵션 지정 시 패널 내 관측 순서를 시차로 사용")
    uses_time = time_given and (corr != "independent" or panels == "correlated")
    ttemp = pd.factorize(P.tpos, sort=True)[0] if (uses_time and not o["force"]) else pos
    n_t = int(ttemp.max()) + 1
    if corr != "independent" and n_t < 2:
        corr = "independent"
    if panels == "correlated" and m == 1:
        panels = "heteroskedastic"
    if panels == "correlated":
        sets = {tuple(ttemp[g == i]) for i in range(m)}
        if len(sets) > 1:
            raise ValueError("xtgls panels(correlated): 모든 패널의 관측 시점이 같아야 함 (Stata: panels must be balanced)")
    prev = np.where(first, -1, np.arange(N) - 1)
    nxt = np.where(np.r_[g[1:] != g[:-1], True], -1, np.arange(N) + 1)
    k_rho = X.shape[1] - (1 if o["constant"] else 0)            # _crcar1 k(): number of regressors
    rhos = None
    ys, Xs = y, X
    if corr != "independent":
        e0 = _ols(y, X).resid
        el, ef = _shift(e0, prev), _shift(e0, nxt)
        rr = np.array([_rho_ar1(e0[g == i], el[g == i], ef[g == i], rt, k_rho)[0] for i in range(m)])
        if corr == "ar1":
            ok = np.isfinite(rr)
            rho = float(np.sum(rr[ok] * P.Ti[ok]) / np.sum(P.Ti[ok]))
            rho_obs = np.full(N, rho)
            rhos = np.array([rho])
        else:
            bad = np.flatnonzero(~np.isfinite(rr))
            if len(bad):
                raise ValueError(f"xtgls corr(psar1): 패널 {P.entity_labels[bad[0]]}의 rho_i 계산 불가 "
                                 f"(rhotype {rt}) — Stata 동일 오류")
            rho_obs = rr[g]
            rhos = rr
        ys, Xs = _pw(y, prev, rho_obs), _pw(X, prev, rho_obs)
    valid = np.isfinite(ys) & np.all(np.isfinite(Xs), axis=1)
    if not valid.all():
        est.notes.append(f"|rho| > 1인 패널의 첫 관측치 {int((~valid).sum())}개는 Prais–Winsten 변환값이 결측 — 추정에서 제외 "
                         "(Stata 동일; 보고 관측치 수는 표본 전체)")
    gv = g[valid]
    tv = ttemp[valid]
    # GLS loop (Stata: residuals → Σ̂ → GLS; igls repeats until the relative change of β ≤ tolerance)
    e = np.full(N, np.nan)
    e[valid] = _ols(ys[valid], Xs[valid]).resid
    it, diff, b, bold, V = 0, o["tolerance"] + 1.0, None, None, None
    Sigma = None
    while it < o["iterate"] and diff > o["tolerance"]:
        if not o["igls"]:
            diff = 0.0
        if it > 0:
            e = ys - Xs @ b
        Sigma = _xtgls_sigma(e, g, ttemp, m, n_t, panels)
        b, V = _xtgls_gls(ys[valid], Xs[valid], gv, tv, m, n_t, panels, Sigma)
        if it == 0:
            bold = b.copy()
        else:
            diff = _mreldif(b, bold)
            bold = b.copy()
        it += 1
    converged = diff <= o["tolerance"]
    if not converged:
        est.notes.append(f"수렴 미달: 반복 {it}회 후 계수 상대 변화 {diff:.3g} > tolerance {o['tolerance']:g} "
                         "(Stata: convergence not achieved) — 마지막 반복 결과 보고")
    ehat = ys - Xs @ b
    ll = None
    if corr == "independent" and (o["igls"] or panels == "iid"):
        if panels == "correlated":
            sign, logdet = np.linalg.slogdet(Sigma)
            Sinv_ll, n0 = _syminv(Sigma)
            E = ehat.reshape(m, n_t)
            quad = float(np.einsum("it,ij,jt->", E, Sinv_ll, E))
            if sign > 0 and n0 == 0:
                ll = -0.5 * (N * np.log(2 * np.pi) + n_t * logdet + quad)
            else:
                est.notes.append("Σ̂ 특이행렬 — 로그우도 미보고 (Stata 출력 로그우도는 det(Σ̂)의 반올림 오차에 좌우)")
        else:
            s_obs = np.diag(Sigma)[g]
            ll = -0.5 * (N * np.log(2 * np.pi) + float(np.sum(np.log(s_obs))) + float(np.sum(ehat ** 2 / s_obs)))
    rank = int(np.sum(np.diag(V) > 0))
    if o["nmk"]:
        nv = int(valid.sum())
        V = V * nv / (nv - rank)
    est.names, est.b, est.V, est.dist = names, b, V, "z"
    slope = [j for j, nm in enumerate(names) if nm != "_cons"]
    W, df = _wald(b, V, slope)
    n_cv = 1 if panels == "iid" else m if panels == "heteroskedastic" else m * (m + 1) // 2
    est.stats.update({
        "N": N, "N_g": m, "N_t": n_t, "Ti": P.Ti.copy(), "chi2": W, "df": df,
        "p": float(stats.chi2.sf(W, df)) if W is not None else None, "ll": ll, "n_cf": rank, "n_cv": n_cv,
        "n_cr": 0 if corr == "independent" else (1 if corr == "ar1" else m), "Sigma": Sigma, "rho": rhos,
        "iterations": it, "converged": converged, "panels": panels, "corr": corr,
        "balanced": bool(P.Ti.min() == P.Ti.max()),
        "generalized_inverse": bool(panels == "correlated" and n_t < m),
    })
    est.lib = ("statsmodels WLS" if panels != "correlated" else "statsmodels OLS (Σ̂⁻¹ᐟ² ⊗ I 백색화 자료)")
    if panels == "correlated" and n_t < m:
        est.notes.append(f"시점 수({n_t}) < 패널 수({m}): Σ̂가 특이행렬 — 일반화 역행렬 기반 결과 (Stata 동일 경고)")
    return est


def _xtgls_sigma(e: np.ndarray, g: np.ndarray, ttemp: np.ndarray, m: int, n_t: int, panels: str) -> np.ndarray:
    """Σ̂ from residuals (NaN = observation not used): iid σ̂² = mean e², heteroskedastic σ̂ᵢ² = mean eᵢ²,
    correlated Σ̂ᵢⱼ = Σ_t eᵢₜeⱼₜ / T over periods complete for all panels ([XT] xtgls Methods and formulas)."""
    ok = np.isfinite(e)
    if panels == "iid":
        return np.eye(m) * float(np.mean(e[ok] ** 2))
    if panels == "heteroskedastic":
        s = np.bincount(g[ok], weights=e[ok] ** 2, minlength=m) / np.bincount(g[ok], minlength=m)
        return np.diag(s)
    E = np.full((m, n_t), np.nan)
    E[g, ttemp] = e
    E = E[:, np.all(np.isfinite(E), axis=0)]
    return E @ E.T / n_t


def _xtgls_gls(ys, Xs, g, ttemp, m, n_t, panels, Sigma) -> Tuple[np.ndarray, np.ndarray]:
    """β̂ = (X'Ω̂⁻X)⁻¹X'Ω̂⁻y, V = (X'Ω̂⁻X)⁻¹ with Ω̂ = Σ̂ ⊗ I: statsmodels WLS for diagonal Σ̂, statsmodels OLS on
    the Σ̂⁻¹ᐟ²-whitened periods for panels(correlated); Σ̂⁻ = Stata-order generalized inverse when Σ̂ is singular."""
    import statsmodels.api as sm
    if panels != "correlated":
        w = 1.0 / np.diag(Sigma)[g]
        res = sm.WLS(ys, Xs, weights=w).fit()
        return np.asarray(res.params, float), np.asarray(res.normalized_cov_params, float)
    Sinv, _ = _syminv(Sigma)
    p = Xs.shape[1]
    if len(ys) == m * n_t:
        lam, Q = np.linalg.eigh((Sinv + Sinv.T) / 2)
        C = np.sqrt(np.clip(lam, 0, None))[:, None] * Q.T          # C'C = Σ̂⁻
        Ym = np.zeros((m, n_t))
        Xm = np.zeros((m, n_t, p))
        Ym[g, ttemp] = ys
        Xm[g, ttemp] = Xs
        yw = (C @ Ym).ravel()
        Xw = np.einsum("ij,jtp->itp", C, Xm).reshape(m * n_t, p)
        res = sm.OLS(yw, Xw).fit()
        return np.asarray(res.params, float), np.asarray(res.normalized_cov_params, float)
    # observations missing after the AR(1) transformation: period blocks of Σ̂⁻ (Stata glsaccum)
    XWX, XWy = np.zeros((p, p)), np.zeros(p)
    for t in range(n_t):
        r = np.flatnonzero(ttemp == t)
        W = Sinv[np.ix_(g[r], g[r])]
        XWX += Xs[r].T @ W @ Xs[r]
        XWy += Xs[r].T @ W @ ys[r]
    V, _ = _syminv(XWX)
    return V @ XWy, V


# ----------------------------------------------------------------------------- xtpcse
def fit_xtpcse(P: PanelData, o: Dict[str, Any]) -> Est:
    """Stata xtpcse: OLS or Prais–Winsten coefficients with panel-corrected standard errors (Beck & Katz 1995)."""
    est = Est()
    corr, rt, pcse = o["corr"], o["rhotype"], o["pcse"]
    X, names, omitted = _design(P, o["constant"])
    est.omitted = omitted
    g, y, N, m, tpos = P.g, P.y, P.N, P.n, P.tpos
    k = X.shape[1]
    prev, nxt = lag_index(g, tpos, 1), lag_index(g, tpos, -1)
    gaps = _gaps(g, tpos)
    rhos = None
    res0 = _ols(y, X)
    ys, Xs = y, X
    if corr != "independent":
        e0 = np.asarray(res0.resid)
        el, ef = _shift(e0, prev), _shift(e0, nxt)
        rr, mm = np.empty(m), np.empty(m)
        for i in range(m):
            s = g == i
            rr[i], mm[i] = _rho_ar1(e0[s], el[s], ef[s], rt, k)
        bounded = np.isfinite(rr) & (np.abs(rr) > 1)
        if bounded.any():
            rr = np.where(bounded, np.clip(rr, -1, 1), rr)
            est.notes.append("rho 추정치가 [−1, 1] 밖인 패널은 경계값으로 제한 (Stata 동일 안내)")
        if corr == "ar1":
            ok = np.isfinite(rr)
            w = mm[ok] + (1 if o["np1"] else 0)
            if np.sum(w) == 0:
                raise ValueError("xtpcse corr(ar1): rho 추정에 쓸 연속 관측치 없음")
            rho = float(np.sum(w * rr[ok]) / np.sum(w))
            rho_obs = np.full(N, rho)
            rhos = np.array([rho])
        else:
            miss = ~np.isfinite(rr)
            if miss.any():
                est.notes.append("rho_i 계산 불가 패널은 0으로 가정: " + ", ".join(str(v) for v in P.entity_labels[miss]))
                rr = np.where(miss, 0.0, rr)
            rho_obs = rr[g]
            rhos = rr
        ys, Xs = _pw(y, prev, rho_obs), _pw(X, prev, rho_obs)
        if gaps:
            est.notes.append(f"시계열 공백 {gaps}곳: rho 계산과 Prais–Winsten 변환은 공백마다 다시 시작 (Stata 동일)")
    res = _ols(ys, Xs)
    b = np.asarray(res.params, float)
    e = np.asarray(res.resid, float)
    rss = float(np.sum(e ** 2))
    tss = float(np.sum((ys - ys.mean()) ** 2)) if o["constant"] else float(np.sum(ys ** 2))
    XtXi = np.asarray(res.normalized_cov_params, float)
    times = np.unique(tpos)
    balanced = len({tuple(tpos[g == i]) for i in range(m)}) == 1
    n_sigma = None
    if pcse == "hetonly":
        s = np.bincount(g, weights=e ** 2, minlength=m) / P.Ti
        Sigma = np.diag(s)
        XOX = (Xs * s[g][:, None]).T @ Xs
    elif pcse == "independent":
        s = float(np.mean(e ** 2))
        Sigma = np.eye(m) * s
        XOX = s * (Xs.T @ Xs)
    else:
        tix = np.searchsorted(times, tpos)
        Emat = np.full((m, len(times)), np.nan)
        Emat[g, tix] = e
        present = ~np.isnan(Emat)
        if balanced or not o["pairwise"]:
            common = present.all(axis=0)
            if not common.any():
                raise ValueError("xtpcse: 모든 패널에 공통인 시점 없음 — casewise로 Σ 추정 불가 (Stata 동일 오류); "
                                 "pairwise 옵션 사용")
            Ec = Emat[:, common]
            n_sigma = int(common.sum())
            Sigma = Ec @ Ec.T / n_sigma
            if not balanced and n_sigma < P.Ti.mean() / 2:
                est.notes.append(f"Σ 추정에 쓴 패널당 관측치 {n_sigma} < 패널당 평균 관측치의 절반 — pairwise 옵션 검토 "
                                 "(Stata 동일 안내)")
        else:
            E0 = np.where(present, Emat, 0.0)
            P1 = present.astype(float)
            cnt = P1 @ P1.T
            with np.errstate(invalid="ignore", divide="ignore"):
                Sigma = np.where(cnt > 0, (E0 @ E0.T) / cnt, 0.0)
            if (cnt == 0).any():
                est.notes.append("공통 시점이 없는 패널 쌍의 공분산은 0으로 가정 (Stata 동일 안내)")
        XOX = np.zeros((k, k))
        for t in range(len(times)):
            rows = np.flatnonzero(tix == t)
            if len(rows):
                Xt = Xs[rows]
                XOX += Xt.T @ Sigma[np.ix_(g[rows], g[rows])] @ Xt
    V = XtXi @ XOX @ XtXi
    if o["nmk"]:
        V = V * N / (N - k)
    V = 0.5 * (V + V.T)
    est.names, est.b, est.V, est.dist = names, b, V, "z"
    slope = [j for j, nm in enumerate(names) if nm != "_cons"]
    W, df = _wald(b, V, slope)
    est.stats.update({
        "N": N, "N_g": m, "Ti": P.Ti.copy(), "chi2": W, "df": df, "p": float(stats.chi2.sf(W, df)) if W is not None else None,
        "r2": 1 - rss / tss if tss > 0 else np.nan, "rmse": float(np.sqrt(rss / (N - k))), "rss": rss, "n_cf": k,
        "n_cv": m if pcse == "hetonly" else 1 if pcse == "independent" else m * (m + 1) // 2,
        "n_cr": 0 if corr == "independent" else (1 if corr == "ar1" else m), "rho": rhos, "Sigma": Sigma,
        "n_sigma": n_sigma, "N_gaps": gaps, "balanced": balanced, "corr": corr, "pcse": pcse,
        "missmeth": ("pairwise" if (o["pairwise"] and not balanced and pcse == "correlated") else "casewise"),
    })
    est.lib = "statsmodels OLS"
    return est


# ----------------------------------------------------------------------------- xtregar
def _bw_transform(A: np.ndarray, first: np.ndarray, d: np.ndarray, rho: float) -> np.ndarray:
    """Baltagi–Wu (1999) Cᵢ(ρ) transformation for unequally spaced panels (d = t_ij − t_i,j−1 in delta units)."""
    A2 = A if A.ndim == 2 else A[:, None]
    out = np.empty_like(A2, dtype=float)
    c = np.sqrt(1 - rho ** 2)
    out[first] = c * A2[first]
    nf = ~first
    dd = d[nf][:, None]
    den = np.sqrt(1 - rho ** (2 * dd))
    out[nf] = c * (A2[nf] / den - A2[np.flatnonzero(nf) - 1] * (rho ** dd) / den)
    return out if A.ndim == 2 else out[:, 0]


def _bw_constant(first: np.ndarray, d: np.ndarray, rho: float) -> np.ndarray:
    """g_i = Cᵢ(ρ)·1: √(1−ρ²) at a panel's first observation, √(1−ρ²)(1−ρ^d)/√(1−ρ^{2d}) after a step of d periods."""
    c = np.sqrt(1 - rho ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = c * (1 - rho ** d) / np.sqrt(1 - rho ** (2 * d))
    return np.where(first, c, out)


def _demean(P: PanelData, A: np.ndarray) -> np.ndarray:
    return A - P.group_mean(A)[P.g]


def _prais_rho(yd: np.ndarray, Xd: np.ndarray, prev: np.ndarray, nxt: np.ndarray, method: str, twostep: bool,
               iterate: int = 100, tol: float = 1e-6) -> Tuple[float, int]:
    """[TS] prais (nocons) on the within-demeaned data: ρ̂ from the residuals, Prais–Winsten restarted at gaps,
    iterated until |Δρ| ≤ 1e-6 (twostep: one round)."""
    k = 1 + Xd.shape[1]                     # prais counts the dependent variable in k()
    b = np.asarray(_ols(yd, Xd).params, float)
    rho, last, it = 0.0, 2.0, 0
    maxit = 1 if twostep else iterate
    while it < maxit and abs(rho - last) > tol:
        last = rho
        e = yd - Xd @ b
        rho = _rho_ar1(e, _shift(e, prev), _shift(e, nxt), method, k)[0]
        if not np.isfinite(rho):
            raise ValueError(f"xtregar: rhotype({method})로 rho 계산 불가")
        ro = np.full(len(yd), rho)
        b = np.asarray(_ols(_pw(yd, prev, ro), _pw(Xd, prev, ro)).params, float)
        it += 1
    return float(rho), it


def _lbi(P: PanelData, Xk: np.ndarray) -> Dict[str, float]:
    """Bhargava–Franzini–Narendranathan modified DW (d1) and Baltagi–Wu LBI (Baltagi & Wu 1999; [XT] xtregar)."""
    yd = _demean(P, P.y)
    Xd = _demean(P, Xk) if Xk.shape[1] else Xk
    Z = np.column_stack([Xd, np.ones(P.N)])
    z = np.asarray(_ols(yd, Z).resid, float)
    g, tpos = P.g, P.tpos
    first = np.r_[True, g[1:] != g[:-1]]
    last = np.r_[g[1:] != g[:-1], True]
    S = float(np.sum(z ** 2))
    I = np.zeros(P.N, bool)
    I[1:] = (~first[1:]) & (np.diff(tpos) == 1)
    zl = np.r_[0.0, z[:-1]]
    d1 = float(np.sum(((z - zl * I) ** 2)[~first])) / S
    Inext = np.r_[I[1:], False]
    d2 = float(np.sum((z ** 2 * (1 - Inext))[~last])) / S
    d3 = float(np.sum(z[first] ** 2)) / S
    d4 = float(np.sum(z[last] ** 2)) / S
    return {"d1": d1, "LBI": d1 + d2 + d3 + d4, "N_LBI": P.N}


def fit_xtregar(P: PanelData, o: Dict[str, Any]) -> Est:
    """Stata xtregar fe / re: Baltagi & Wu (1999) estimators for (unequally spaced) panels with AR(1) disturbances."""
    est = Est()
    g, tpos, N, m = P.g, P.tpos, P.N, P.n
    X0, names0, omitted0 = _design(P, True)              # levels collinearity (re)
    Xk = X0[:, :-1]
    xnames = names0[:-1]
    Xd_all = _demean(P, Xk) if Xk.shape[1] else Xk
    keep_d = independent_columns(Xd_all) if Xk.shape[1] else []
    if o["model"] == "fe":
        dropped = [xnames[j] for j in range(len(xnames)) if j not in keep_d]
        Xk, xnames = Xk[:, keep_d], [xnames[j] for j in keep_d]
        est.omitted = omitted0 + dropped
        Xd = Xd_all[:, keep_d]
    else:
        est.omitted = omitted0
        Xd = Xd_all[:, keep_d]
    yd = _demean(P, P.y)
    prev, nxt = lag_index(g, tpos, 1), lag_index(g, tpos, -1)
    rt = o["rhotype"]
    if o["rhof"] is not None:
        rho, rho_it = float(o["rhof"]), 0
    elif rt == "onestep":
        e = np.asarray(_ols(yd, Xd).resid, float) if Xd.shape[1] else yd
        el = _shift(e, prev)
        has = ~np.isnan(el)
        rho = (float(np.sum(e[has] * el[has])) / int(has.sum())) / (float(np.sum(e ** 2)) / N)
        rho_it = 1
    else:
        rho, rho_it = _prais_rho(yd, Xd, prev, nxt, rt, o["twostep"])
    if not abs(rho) < 1:
        raise ValueError(f"xtregar: 추정 rho = {rho:.4f} (|rho| ≥ 1) — rhotype(tscorr) 사용 권장 (Stata 기술 노트)")
    first = np.r_[True, g[1:] != g[:-1]]
    d = np.r_[0, np.diff(tpos)].astype(float)
    ys = _bw_transform(P.y, first, d, rho)
    Xs = _bw_transform(Xk, first, d, rho) if Xk.shape[1] else Xk
    cs = _bw_transform(np.ones(N), first, d, rho)
    gaps = _gaps(g, tpos)
    if gaps:
        est.notes.append(f"시점 간격이 다른 관측 {gaps}곳: Baltagi–Wu 변환이 간격을 반영 (Stata 동일)")
    est.stats.update({"rho_ar": rho, "rhotype": "fixed" if o["rhof"] is not None else rt, "rho_iter": rho_it})
    if o["model"] == "fe":
        _xtregar_fe(P, o, est, ys, Xs, cs, Xk, xnames, first, rho)
    else:
        _xtregar_re(P, o, est, ys, Xs, cs, X0, names0, first, rho, gi=_bw_constant(first, d, rho))
    if o["lbi"]:
        kd = Xk if o["model"] == "fe" else X0[:, :-1][:, keep_d]
        est.stats.update(_lbi(P, kd))
    return est


def _xtregar_fe(P, o, est, ys, Xs, cs, Xk, xnames, first, rho):
    import statsmodels.api as sm
    keep = ~first
    g = pd.factorize(P.g[keep], sort=True)[0]
    Ng = int(g.max()) + 1
    Ti = np.bincount(g)
    NT = int(keep.sum())
    yk, Xk_s, ck = ys[keep], Xs[keep], cs[keep]
    rrss = float(np.sum(_ols(yk, np.column_stack([Xk_s, ck])).resid ** 2))
    Ti_obs = Ti[g]

    def fev(a):
        s = np.zeros((Ng,) + a.shape[1:])
        np.add.at(s, g, a)
        mean_i = s / (Ti if a.ndim == 1 else Ti[:, None])
        return a + a.mean(axis=0) - mean_i[g]

    yf = fev(yk)
    Xf = fev(Xk_s) if Xk_s.shape[1] else Xk_s
    res = sm.OLS(yf, np.column_stack([Xf, np.ones(NT)])).fit()
    K = Xf.shape[1] + 1
    df_m = K - 1 + Ng - 1
    df_b = K - 1
    df_r = NT - df_m - 1
    rss, mss, r2 = float(res.ssr), float(res.ess), float(res.rsquared)
    rmse = float(np.sqrt(rss / (NT - K)))
    F = float(res.fvalue) * df_r / (NT - K) if K > 1 else np.nan
    F_f = df_r / (Ng - 1) * (rrss - rss) / rss if Ng > 1 else np.nan
    sigma_e = float(np.sqrt(rmse ** 2 * (NT - df_b - 1) / (NT - df_m - 1)))
    V = np.asarray(res.cov_params(), float) * (NT - K) / (NT - Ng - K + 1)
    b = np.asarray(res.params, float)
    # corr(u_i, xb), R² between / overall, sigma_u on the untransformed data without each panel's first observation
    yo = P.y[keep]
    xb = Xk[keep] @ b[:-1] + b[-1] if Xk.shape[1] else np.full(NT, b[-1])
    ybar = np.bincount(g, weights=yo) / Ti
    xbbar = np.bincount(g, weights=xb) / Ti
    u = ybar - xbbar
    sigma_u = float(np.std(u, ddof=1)) if Ng > 1 else np.nan
    b_rep = b.copy()
    b_rep[-1] = b[-1] / (1 - rho)
    est.names, est.b, est.V, est.dist, est.df_r = xnames + ["_cons"], b_rep, V, "t", float(df_r)
    est.stats.update({
        "N": NT, "N_g": Ng, "Ti": Ti, "df_m": df_m, "df_b": df_b, "df_r": df_r, "df_a": Ng - 1, "mss": mss, "rss": rss,
        "r2_w": r2, "r2_a": 1 - (NT - 1) / df_r * (1 - r2), "rmse": rmse, "F": F,
        "F_p": float(stats.f.sf(F, df_b, df_r)) if np.isfinite(F) else None,
        "F_f": F_f, "F_f_p": float(stats.f.sf(F_f, Ng - 1, df_r)) if np.isfinite(F_f) else None,
        "sigma_e": sigma_e, "sigma_u": sigma_u, "rho_fov": sigma_u ** 2 / (sigma_e ** 2 + sigma_u ** 2),
        "corr": _corr(u[g], xb), "r2_b": _corr(xbbar[g], ybar[g]) ** 2, "r2_o": _corr(xb, yo) ** 2,
        "Tbar": float(len(Ti) / np.sum(1.0 / Ti)), "ll": float(res.llf), "model": "fe",
    })
    est.notes.append(f"각 패널의 첫 관측치는 Cochrane–Orcutt 변환으로 제외: 관측치 {P.N} → {NT} (Stata 동일)")
    est.notes.append("_cons = 변환 자료 상수항 ÷ (1 − rho_ar); 표준오차는 변환 자료 기준 (Stata 동일)")
    est.lib = "statsmodels OLS"


def _xtregar_re(P, o, est, ys, Xs, cs, X0, names0, first, rho, gi=None):
    g, N, m = P.g, P.N, P.n
    Z = np.column_stack([Xs, cs])
    mu = np.asarray(_ols(ys, Z).resid, float)
    cs = cs if gi is None else gi                       # g_i = Cᵢ(ρ)·1 (Baltagi & Wu 1999)
    gg = np.bincount(g, weights=cs ** 2, minlength=m)
    mug = np.bincount(g, weights=mu * cs, minlength=m)
    s2w = float(np.sum(mug ** 2 / gg))
    s2e = (float(np.sum(mu ** 2)) - s2w) / float(np.sum(P.Ti - 1))
    s2u = max(0.0, (s2w - m * s2e) / float(np.sum(gg)))
    om = gg * s2u + s2e
    theta = 1 - np.sqrt(s2e / om)
    th_obs = theta[g]

    def gls(a):
        A = a if a.ndim == 2 else a[:, None]
        sz = np.zeros((m, A.shape[1]))
        np.add.at(sz, g, A * cs[:, None])
        out = A - th_obs[:, None] * cs[:, None] * (sz[g] / gg[g][:, None])
        return out if a.ndim == 2 else out[:, 0]

    res = _ols(gls(ys), gls(Z))
    b = np.asarray(res.params, float)
    V = np.asarray(res.cov_params(), float)
    names = names0
    xb = X0 @ b
    ybar = P.group_mean(P.y)
    xbbar = P.group_mean(xb)
    slope = list(range(len(names) - 1))
    W, _ = _wald(b, V, slope)
    df_m = len(names)
    Tcon = bool(P.Ti.min() == P.Ti.max())
    est.names, est.b, est.V, est.dist = names, b, V, "z"
    est.stats.update({
        "N": N, "N_g": m, "Ti": P.Ti.copy(), "sigma_u": float(np.sqrt(s2u)), "sigma_e": float(np.sqrt(s2e)),
        "rho_fov": s2u / (s2e + s2u), "chi2": W, "df_m": df_m, "chi2_df": len(slope),
        "p_stata": float(stats.chi2.sf(W, df_m)) if W is not None else None,
        "p": float(stats.chi2.sf(W, len(slope))) if W is not None else None,
        "r2_o": _corr(xb, P.y) ** 2, "r2_b": _corr(xbbar, ybar) ** 2,
        "r2_w": _corr(xb - xbbar[g], P.y - ybar[g]) ** 2, "Tcon": Tcon,
        "thta_min": float(np.min(th_obs)), "thta_5": stata_percentile(th_obs, 5), "thta_50": stata_percentile(th_obs, 50),
        "thta_95": stata_percentile(th_obs, 95), "thta_max": float(np.max(th_obs)),
        "Tbar": float(m / np.sum(1.0 / P.Ti)), "model": "re",
    })
    est.lib = "statsmodels OLS"


# ----------------------------------------------------------------------------- xthtaylor
def ht_options(params: Dict[str, Any]) -> Dict[str, Any]:
    p = params or {}
    vce = resolve_choice("표준오차(vce)", _opt(p, "vce", "covType", default="conventional"), ("conventional", "robust"),
                         {"oim": "conventional", "unadjusted": "conventional", "cluster": "robust",
                          "clustered": "robust", "r": "robust"})
    return {
        "entity": _as_single(_opt(p, "entity", "entity_col", "idVar", "panelVar")),
        "time": _as_single(_opt(p, "time", "time_col", "timeVar")),
        "dep": _as_single(_opt(p, "dep", "depvar")),
        "covs": _as_list(_opt(p, "covs", "indeps", "indepvars", default=[])),
        "endog": _as_list(_opt(p, "endog", "endogenous", default=[])),
        "constantVars": _as_list(_opt(p, "constantVars", "constant_vars", "timeInvariant", "ti", default=[])),
        "varying": _as_list(_opt(p, "varying", "varyingVars", "timeVarying", default=[])),
        "amacurdy": _flag(p, "amacurdy", "am"), "small": _flag(p, "small"),
        "constant": _flag(p, "constant", default=not _flag(p, "noconstant", "nocons")),
        "hausman": _flag(p, "hausman", default=True), "vce": vce,
        "delta": float(_opt(p, "delta", default=1.0)),
    }


def _time_invariant(P: PanelData, a: np.ndarray) -> bool:
    """xthtaylor rule: running within-panel mean minus the value is 0 (|·| < 1e-12) for every observation."""
    s = pd.Series(a)
    cm = s.groupby(P.g).cumsum() / (s.groupby(P.g).cumcount() + 1)
    d = (cm - s).to_numpy()
    return bool(abs(d.min()) < 1e-12 and abs(d.max()) < 1e-12)


def fit_xthtaylor(P: PanelData, o: Dict[str, Any]) -> Est:
    """Hausman & Taylor (1981) / Amemiya & MaCurdy (1986) estimator; Stata [XT] xthtaylor Methods and formulas."""
    from linearmodels.iv import IV2SLS
    est = Est()
    covs = list(P.covs)
    endog = list(dict.fromkeys(o["endog"]))
    if not endog:
        raise ValueError("xthtaylor: 내생 변수(endog) 지정 필요 (Stata endog() 필수)")
    bad = [v for v in endog + o["constantVars"] + o["varying"] if v not in covs]
    if bad:
        raise ValueError(f"xthtaylor: endog()·constant()·varying() 변수가 설명변수 목록에 없음 — {', '.join(bad)}. "
                         f"설명변수: {covs}")
    if o["constantVars"] and o["varying"]:
        raise ValueError("xthtaylor: constant()와 varying()은 함께 지정할 수 없음 (Stata 동일)")
    ti = {v: _time_invariant(P, P.X[:, j]) for j, v in enumerate(covs)}
    inv = [v for v in covs if ti[v]]
    var = [v for v in covs if not ti[v]]
    if o["constantVars"]:
        d1 = [v for v in o["constantVars"] if v not in inv]
        if d1:
            raise ValueError(f"xthtaylor: {', '.join(d1)} — 패널 내 시간 불변 아님 (constant() 지정 오류)")
        d2 = [v for v in inv if v not in o["constantVars"]]
        if d2:
            raise ValueError(f"xthtaylor: {', '.join(d2)} — 시간 불변 변수이나 constant()에 없음 (Stata: not included in constant())")
    if o["varying"]:
        d1 = [v for v in o["varying"] if v not in var]
        if d1:
            raise ValueError(f"xthtaylor: {', '.join(d1)} — 패널 내 시간 변동 없음 (varying() 지정 오류)")
        d2 = [v for v in var if v not in o["varying"]]
        if d2:
            raise ValueError(f"xthtaylor: {', '.join(d2)} — 시간 변동 변수이나 varying()에 없음 (Stata: not included in varying())")
    # collinearity (Stata _rmcoll with the constant)
    M = np.column_stack([np.ones(P.N), P.X]) if o["constant"] else P.X
    keep = independent_columns(M, protect=[0] if o["constant"] else [])
    kept = [covs[j - (1 if o["constant"] else 0)] for j in keep if (j > 0 or not o["constant"])]
    est.omitted = [v for v in covs if v not in kept]
    exog = [v for v in kept if v not in endog]
    x1 = [v for v in exog if v in var]
    x2 = [v for v in kept if v in endog and v in var]
    z1 = [v for v in exog if v in inv]
    z2 = [v for v in kept if v in endog and v in inv]
    if not x1 or not z1:
        msg = []
        if not x1:
            msg.append("시간 변동 외생 변수 없음 (There are no time-varying exogenous variables)")
        if not z1:
            msg.append("시간 불변 외생 변수 없음 (There are no time-invariant exogenous variables)")
        raise ValueError("xthtaylor: " + "; ".join(msg) + " — 공선성으로 제외되었을 수 있음 (Stata 동일 오류)")
    k1, g2 = len(x1), len(z2)
    col = {v: P.X[:, covs.index(v)] for v in covs}
    g, N, n = P.g, P.N, P.n
    Ti = P.Ti
    if not o["amacurdy"] and k1 < g2:
        raise ValueError(f"xthtaylor: 식별 불가 — 시간 변동 외생 변수 수 k1={k1} < 시간 불변 내생 변수 수 g2={g2} "
                         "(Stata: The model is underidentified (k1<g2))")
    if o["amacurdy"]:
        first_t = P.tpos[np.r_[True, g[1:] != g[:-1]]]
        if first_t.min() != first_t.max():
            raise ValueError("xthtaylor amacurdy: 모든 패널이 같은 시점에서 시작해야 함 (Stata: All panels must start at "
                             "the same time period)")
        if _gaps(g, P.tpos):
            raise ValueError("xthtaylor amacurdy: 시계열 공백 불허 (Stata 동일)")
        if Ti.min() != Ti.max():
            raise ValueError("xthtaylor amacurdy: 균형 패널만 가능 (Stata: amacurdy only works with balanced data)")
        if Ti.max() * k1 < g2:
            raise ValueError("xthtaylor amacurdy: 식별 불가 (T·k1 < g2)")
    balanced = bool(Ti.min() == Ti.max())
    Tbar = float(Ti.max()) if balanced else float(n / np.sum(1.0 / Ti))
    mean = lambda a: P.group_mean(a)[g]
    y = P.y
    Xtv = np.column_stack([col[v] for v in x1 + x2])
    Xm = P.group_mean(Xtv)[g]
    Xdm = Xtv - Xm
    ym = mean(y)
    bw = np.asarray(_ols(y - ym, Xdm).params, float)
    rss_w = float(np.sum((y - ym - Xdm @ bw) ** 2))
    s2e = rss_w / (N - n)
    d = ym - Xm @ bw
    xbw = Xtv @ bw
    Zx = np.column_stack([col[v] for v in z1 + z2])
    c1 = np.ones((N, 1))
    iv1 = IV2SLS(d, np.column_stack([c1, np.column_stack([col[v] for v in z1])]),
                 np.column_stack([col[v] for v in z2]) if z2 else None,
                 np.column_stack([col[v] for v in x1]) if z2 else None).fit(cov_type="unadjusted")
    zg = np.column_stack([c1, Zx]) @ np.asarray(iv1.params, float)
    r = y - xbw - zg
    rbar = P.group_mean(r)
    s21 = float(np.sum(Ti * rbar ** 2)) / n
    s2u = max(0.0, (s21 - s2e) / Tbar)
    if s2u == 0:
        theta = np.zeros(N)
    elif balanced:
        theta = np.full(N, 1 - np.sqrt(s2e / s21))
    else:
        theta = 1 - np.sqrt(s2e / (Ti[g] * s2u + s2e))
    yg = y - theta * ym
    Xg = Xtv - theta[:, None] * Xm
    Zg = Zx * (1 - theta)[:, None]
    endog_g = np.column_stack([Xg, Zg])
    if o["amacurdy"]:
        T = int(Ti.max())
        start = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])[g]          # first row of each observation's panel
        X1 = np.column_stack([col[v] for v in x1])
        inst_extra = np.column_stack([X1[start + t, j] for j in range(k1) for t in range(T)])
    else:
        inst_extra = Xm[:, :k1]
    inst = np.column_stack([Xdm, inst_extra, np.column_stack([col[v] for v in z1])])
    exog_g = (1 - theta)[:, None] if o["constant"] else None
    names = x1 + x2 + z1 + z2 + (["_cons"] if o["constant"] else [])
    mod = IV2SLS(yg, exog_g, endog_g, inst)
    if o["vce"] == "robust":
        res = mod.fit(cov_type="clustered", clusters=np.asarray(g), debiased=True)
    else:
        res = mod.fit(cov_type="unadjusted", debiased=True)
    pr = np.asarray(res.params, float)
    Vr = np.asarray(res.cov, float)
    # linearmodels orders exog first, then endog
    k_ex = 1 if o["constant"] else 0
    order = list(range(k_ex, len(pr))) + list(range(k_ex))
    b = pr[order]
    V = Vr[np.ix_(order, order)]
    K = len(b)
    df_r = N - K
    slope = [j for j, nm in enumerate(names) if nm != "_cons"]
    if o["constant"]:
        W, df = _wald(b, V, slope)
    else:
        W, df = _wald(b, V, list(range(K)))
    if o["vce"] == "robust":
        W = float(b @ np.linalg.pinv(V) @ b)              # Stata: e(b)*syminv(e(V))*e(b)' over all coefficients
    elif not o["constant"]:
        W = None                                          # Stata noconstant: F = . and chi2 = F·df not reported
    est.names, est.b, est.V = names, b, V
    est.dist = "t" if o["small"] else "z"
    est.df_r = float(df_r) if o["small"] else None
    est.blocks = (["TVexogenous"] * len(x1) + ["TVendogenous"] * len(x2) + ["TIexogenous"] * len(z1)
                  + ["TIendogenous"] * len(z2) + ([""] if o["constant"] else []))
    est.stats.update({
        "N": N, "N_g": n, "Ti": Ti.copy(), "sigma_u": float(np.sqrt(s2u)), "sigma_e": float(np.sqrt(s2e)),
        "rho": s2u / (s2u + s2e), "chi2": W, "df_m": df, "p": float(stats.chi2.sf(W, df)) if W is not None else None,
        "F": W / df if (W is not None and df and o["vce"] != "robust") else None,
        "df_r": df_r if o["small"] else None,
        "F_p": float(stats.f.sf(W / df, df, df_r)) if (W is not None and df and o["vce"] != "robust") else None,
        "Tbar": Tbar, "Tcon": balanced, "x1": x1, "x2": x2, "z1": z1, "z2": z2, "k1": k1, "g2": g2,
        "theta": theta,
    })
    est.lib = "linearmodels IV2SLS"
    return est


def hausman_fe_ht(P: PanelData, ht: Est) -> Dict[str, Any]:
    """hausman fe ht: b = xtreg, fe (linearmodels PanelOLS via panel_module.fit_within), B = xthtaylor; statistic
    (b−B)'(V_b−V_B)⁻(b−B) with the SVD generalized inverse and df = rank, as Stata [R] hausman."""
    from .panel_module import fit_within
    tv = ht.stats["x1"] + ht.stats["x2"]
    cols = [P.covs.index(v) for v in tv]
    fe = fit_within(P, P.X[:, cols], tv)
    common = [v for v in tv if v in fe.names]
    ib = [fe.names.index(v) for v in common]
    iB = [ht.names.index(v) for v in common]
    q = fe.b[ib] - ht.b[iB]
    Vd = fe.V[np.ix_(ib, ib)] - ht.V[np.ix_(iB, iB)]
    U, w, Vt = np.linalg.svd(Vd)
    tol = w.max() * len(w) * 1e-9 if len(w) else 0.0
    winv = np.where(w > tol, 1.0 / np.where(w > tol, w, 1.0), 0.0)
    Vinv = Vt.T @ np.diag(winv) @ U.T
    chi2 = float(q @ Vinv @ q)
    rank = int(np.sum(w > tol))
    eig = np.linalg.eigvalsh((Vd + Vd.T) / 2)
    return {"chi2": chi2, "df": rank, "p": float(stats.chi2.sf(chi2, rank)) if rank else None,
            "df_overid": ht.stats["k1"] - ht.stats["g2"], "posdef": bool(eig.min() > 0), "names": common,
            "b": fe.b[ib], "B": ht.b[iB], "se": np.sqrt(np.clip(np.diag(Vd), 0, None))}


# ----------------------------------------------------------------------------- analyses
def _coef_rows(t: ResultsTable, est: Est, level: float = 95.0, blocks: bool = False):
    se = est.se
    with np.errstate(divide="ignore", invalid="ignore"):
        st = np.where(se > 0, est.b / se, np.nan)
    if est.dist == "t" and est.df_r is not None:
        p = 2 * stats.t.sf(np.abs(st), est.df_r)
        crit = stats.t.ppf(0.5 + level / 200, est.df_r)
    else:
        p = 2 * stats.norm.sf(np.abs(st))
        crit = stats.norm.ppf(0.5 + level / 200)
    for i, nm in enumerate(est.names):
        row = {"var": nm, "b": float(est.b[i]), "se": float(se[i]), "t": float(st[i]), "p": float(p[i]),
               "cil": float(est.b[i] - crit * se[i]), "ciu": float(est.b[i] + crit * se[i])}
        if blocks:
            row = {"block": est.blocks[i] if est.blocks else "", **row}
        t.add_row(row)


def _coef_table(est: Est, se_label: str = "표준오차", blocks: bool = False) -> ResultsTable:
    t = ResultsTable(name="coef", title="계수 추정치 (Coefficients)")
    if blocks:
        t.add_column("block", "구분", col_type="text")
    t.add_column("var", "변수", col_type="text")
    t.add_column("b", "추정치", col_type="number")
    t.add_column("se", se_label, col_type="number")
    t.add_column("t", "t" if est.dist == "t" and est.df_r is not None else "z", col_type="number")
    t.add_column("p", "p", col_type="number", fmt="pvalue")
    t.add_column("cil", "95% CI 하한", col_type="number")
    t.add_column("ciu", "95% CI 상한", col_type="number")
    _coef_rows(t, est, blocks=blocks)
    if est.omitted:
        t.add_note("공선성으로 제외 (Stata omitted): " + ", ".join(est.omitted))
    t.mark_coefficients(estimate="b", se="se", p="p", stat="t", ci=("cil", "ciu"), label="var")
    return t


def _summary(rows: List[Tuple[str, Any, str]], title: str) -> ResultsTable:
    t = ResultsTable(name="summary", title=title)
    t.add_column("metric", "항목", col_type="text")
    t.add_column("val", "값", col_type="number")
    for lab, v, fmt in rows:
        r = {"metric": lab, "val": None if v is None or (isinstance(v, float) and np.isnan(v)) else v}
        if fmt:
            r["_rowfmt"] = fmt
        t.add_row(r)
    return t


def _group_rows(Ti: np.ndarray) -> List[Tuple[str, Any, str]]:
    return [("집단당 관측치 최소", int(Ti.min()), "integer"), ("집단당 관측치 평균", float(Ti.mean()), ""),
            ("집단당 관측치 최대", int(Ti.max()), "integer")]


def _pv(p: Optional[float]) -> str:
    if p is None or not np.isfinite(p):
        return ""
    return "p < .001" if p < 0.001 else f"p = {p:.3f}"


class PanelGLSAnalysis(BaseAnalysis):
    """xtgls / xtpcse / xtregar with Stata conventions."""

    def __init__(self, data: pd.DataFrame, **opts):
        o = normalize_options(opts)
        if not o["covs"]:
            raise ValueError(f"{o['method']}: 설명변수(covs) 1개 이상 지정 필요. 사용 가능한 변수: {list(data.columns)}")
        super().__init__(name="panelGLS", title=METHOD_LABEL[o["method"]], dataset=data, options=o)
        self.syntax_python = python_call(o)
        self.syntax_r = "\n".join(r_code("panelGLS", o) or [])
        self.run()

    def run(self):
        o = self.options
        df = self.dataset
        time_given = bool(o["time"])
        tcol = o["time"]
        if not time_given and o["method"] == "xtgls" and o["entity"] in (df.columns if hasattr(df, "columns") else []):
            df = df.copy()
            tcol = "_order_in_panel"
            df[tcol] = df.groupby(o["entity"]).cumcount()
        P = PanelData(df, o["entity"], tcol, o["dep"], o["covs"], delta=o["delta"], label=o["method"])
        self.panel = P
        if o["method"] == "xtgls":
            est = fit_xtgls(P, o, time_given)
        elif o["method"] == "xtpcse":
            est = fit_xtpcse(P, o)
        else:
            est = fit_xtregar(P, o)
        self.est = est
        getattr(self, f"_tables_{o['method']}")(P, est)
        if P.n_dropped:
            self.tables[0].add_note(f"결측으로 제외된 행: {P.n_dropped}")
        if P.time_note:
            self.tables[0].add_note(P.time_note)
        self._narrative(est)

    # ------------------------------------------------------------------ xtgls
    def _tables_xtgls(self, P, est):
        o, st = self.options, est.stats
        rows = [("관측치 수 (N)", st["N"], "integer"), ("집단 수", st["N_g"], "integer"),
                ("시점 수", st["N_t"], "integer")]
        if not st["balanced"]:
            rows += _group_rows(st["Ti"])
        rows += [("추정 공분산 수", st["n_cv"], "integer"), ("추정 자기상관 수", st["n_cr"], "integer"),
                 ("추정 계수 수", st["n_cf"], "integer")]
        if st["chi2"] is not None:
            rows += [(f"Wald χ²({st['df']})", st["chi2"], ""), ("Prob > χ²", st["p"], "pvalue")]
        if st["ll"] is not None:
            rows.append(("로그우도 (Log likelihood)", st["ll"], ""))
        if st["corr"] == "ar1":
            rows.append(("rho (공통 AR(1))", float(st["rho"][0]), ""))
        if o["igls"]:
            rows.append(("반복 횟수", st["iterations"], "integer"))
        t = _summary(rows, "모형 요약 (xtgls)")
        t.add_note(f"계수: 일반화최소제곱(GLS); 패널: {PANELS_LABEL[st['panels']]}; 자기상관: {CORR_LABEL[st['corr']]}"
                   + (f" (rhotype {o['rhotype']})" if st["corr"] != "independent" else ""))
        t.add_note("추정: " + ("반복 GLS (igls)" if o["igls"] else ("3단계 FGLS" if st["corr"] != "independent" else "2단계 FGLS"))
                   + "; 분산 분모 " + ("N − k (nmk)" if o["nmk"] else "N (Stata 기본)"))
        if st["ll"] is None:
            t.add_note("로그우도: corr(independent)에서 panels(iid) 또는 igls일 때만 보고 (Stata 동일)")
        t.add_note("계산 방식: Stata [XT] xtgls Methods and formulas (Σ̂ᵢⱼ = eᵢ'eⱼ/T, Prais–Winsten AR(1)); "
                   f"라이브러리: {est.lib}")
        self.add_table(t)
        ct = _coef_table(est)
        for nt in est.notes:
            ct.add_note(nt)
        self.add_table(ct)
        self._panel_table(P, est)
        ms = {"N": int(st["N"]), "Groups": int(st["N_g"])}
        if st["chi2"] is not None:
            ms["Chi2"] = (float(st["chi2"]), float(st["p"]))
        if st["ll"] is not None:
            ms["LL"] = float(st["ll"])
        self.model_stats = ms

    def _panel_table(self, P, est):
        st = est.stats
        Sigma, rho = st.get("Sigma"), st.get("rho")
        show_var = st.get("panels", st.get("pcse")) in ("heteroskedastic", "correlated", "hetonly")
        show_rho = rho is not None and len(rho) > 1
        if not (show_var or show_rho) or Sigma is None:
            return
        t = ResultsTable(name="panel_params", title="패널별 분산·자기상관 (e(Sigma) 대각·rho_i)")
        t.add_column("panel", "패널", col_type="text")
        if show_var:
            t.add_column("sigma2", "오차 분산 σ²ᵢ", col_type="number")
        if show_rho:
            t.add_column("rho", "rho_i", col_type="number")
        for i in range(P.n):
            r = {"panel": str(P.entity_labels[i])}
            if show_var:
                r["sigma2"] = float(Sigma[i, i])
            if show_rho:
                r["rho"] = float(rho[i])
            t.add_row(r)
        self.add_table(t)

    # ------------------------------------------------------------------ xtpcse
    def _tables_xtpcse(self, P, est):
        o, st = self.options, est.stats
        rows = [("관측치 수 (N)", st["N"], "integer"), ("집단 수", st["N_g"], "integer")] + _group_rows(st["Ti"])
        rows += [("추정 공분산 수", st["n_cv"], "integer"), ("추정 자기상관 수", st["n_cr"], "integer"),
                 ("추정 계수 수", st["n_cf"], "integer"), ("R²", st["r2"], "zto")]
        if st["chi2"] is not None:
            rows += [(f"Wald χ²({st['df']})", st["chi2"], ""), ("Prob > χ²", st["p"], "pvalue")]
        if st["corr"] == "ar1":
            rows.append(("rho (공통 AR(1))", float(st["rho"][0]), ""))
        t = _summary(rows, "모형 요약 (xtpcse)")
        bal = "균형" if st["balanced"] else "불균형"
        t.add_note(f"계수: {'Prais–Winsten 회귀' if st['corr'] != 'independent' else 'OLS'}; 패널: {PCSE_LABEL[st['pcse']]} "
                   f"({bal}); 자기상관: {CORR_LABEL[st['corr']]}"
                   + (f" (rhotype {o['rhotype']}{', np1' if o['np1'] else ''})" if st["corr"] != "independent" else ""))
        if st["pcse"] == "correlated" and not st["balanced"]:
            t.add_note(f"Σ 추정: {'pairwise (패널 쌍별 공통 시점)' if st['missmeth'] == 'pairwise' else 'casewise (모든 패널 공통 시점'}"
                       + (f" {st['n_sigma']}개)" if st["missmeth"] == "casewise" else ""))
        t.add_note("계산 방식: Beck & Katz (1995), Stata [XT] xtpcse Methods and formulas — V = (X'X)⁻¹X'ΩX(X'X)⁻¹"
                   + ("; 분모 N − k (nmk)" if o["nmk"] else "") + f"; 라이브러리: {est.lib}")
        self.add_table(t)
        lab = {"correlated": "패널 수정 표준오차", "hetonly": "이분산 수정 표준오차", "independent": "독립 수정 표준오차"}
        ct = _coef_table(est, lab[st["pcse"]])
        for nt in est.notes:
            ct.add_note(nt)
        self.add_table(ct)
        self._panel_table(P, est)
        ms = {"N": int(st["N"]), "Groups": int(st["N_g"]), "R2": float(st["r2"])}
        if st["chi2"] is not None:
            ms["Chi2"] = (float(st["chi2"]), float(st["p"]))
        self.model_stats = ms

    # ------------------------------------------------------------------ xtregar
    def _tables_xtregar(self, P, est):
        o, st = self.options, est.stats
        fe = st["model"] == "fe"
        rows = [("관측치 수 (N)", st["N"], "integer"), ("집단 수", st["N_g"], "integer")] + _group_rows(st["Ti"])
        rows += [("R² 집단 내 (within)", st["r2_w"], "zto"), ("R² 집단 간 (between)", st["r2_b"], "zto"),
                 ("R² 전체 (overall)", st["r2_o"], "zto")]
        if fe:
            rows += [(f"F({st['df_b']}, {st['df_r']})", st["F"], ""), ("Prob > F", st["F_p"], "pvalue"),
                     ("corr(u_i, Xb)", st["corr"], "")]
        else:
            rows += [(f"Wald χ²({st['df_m']})", st["chi2"], ""), ("Prob > χ²", st["p_stata"], "pvalue")]
        t = _summary(rows, f"모형 요약 (xtregar, {st['model']})")
        t.add_note(("FE 집단 내 추정" if fe else "RE GLS 추정 (Baltagi–Wu)") + " — AR(1) 오차; rho 추정: "
                   + ("고정값 rhof" if st["rhotype"] == "fixed" else f"{RHO_LABEL[st['rhotype']]} (rhotype {st['rhotype']})"
                      + ("; 2단계" if o["twostep"] else "")))
        if not fe:
            t.add_note(f"Stata 표시 관례: Wald χ² 자유도 = 상수항 포함 계수 수 {st['df_m']}, Prob도 같은 자유도로 계산; "
                       f"기울기 계수 {st['chi2_df']}개 기준 p = {st['p']:.4g}")
            t.add_note("corr(u_i, Xb) = 0 가정")
        t.add_note("계산 방식: Baltagi & Wu (1999), Stata [XT] xtregar Methods and formulas; 라이브러리: statsmodels OLS")
        self.add_table(t)
        ct = _coef_table(est)
        for nt in est.notes:
            ct.add_note(nt)
        if fe:
            ct.add_note(f"t 검정 자유도 {int(est.df_r)}")
        self.add_table(ct)
        a = ResultsTable(name="ancillary", title="오차 구성요소 (rho_ar · sigma_u · sigma_e · rho_fov)")
        a.add_column("item", "항목", col_type="text")
        a.add_column("val", "값", col_type="number")
        items = [("rho_ar", st["rho_ar"]), ("sigma_u", st["sigma_u"]), ("sigma_e", st["sigma_e"]),
                 ("rho_fov (u_i 분산 비중)", st["rho_fov"])]
        if not fe and st["Tcon"]:
            items.append(("theta", st["thta_min"]))
        for k_, v in items:
            a.add_row({"item": k_, "val": v})
        a.add_note("sigma_e: " + ("변환 자료 집단 내 회귀 잔차 표준편차 (자유도 N − N_g − K + 1)" if fe
                                  else "Baltagi–Wu 분산성분 η_it 표준편차"))
        self.add_table(a)
        if not fe and not st["Tcon"]:
            th = ResultsTable(name="theta", title="theta 분포 (불균형 패널)")
            for c_ in ("min", "p5", "p50", "p95", "max"):
                th.add_column(c_, {"min": "최소", "p5": "5%", "p50": "중앙값", "p95": "95%", "max": "최대"}[c_], col_type="number")
            th.add_row({"min": st["thta_min"], "p5": st["thta_5"], "p50": st["thta_50"], "p95": st["thta_95"],
                        "max": st["thta_max"]})
            self.add_table(th)
        if fe:
            u = ResultsTable(name="u_test", title="개체효과 F 검정 (F test that all u_i = 0)")
            u.add_column("F", "F", col_type="number")
            u.add_column("df1", "df1", col_type="integer")
            u.add_column("df2", "df2", col_type="integer")
            u.add_column("p", "p", col_type="number", fmt="pvalue")
            u.add_row({"F": st["F_f"], "df1": st["df_a"], "df2": st["df_r"], "p": st["F_f_p"]})
            self.add_table(u)
        if o["lbi"]:
            lb = ResultsTable(name="lbi", title="AR(1) 검정 통계량 (rho = 0)")
            lb.add_column("stat", "통계량", col_type="text")
            lb.add_column("val", "값", col_type="number")
            lb.add_row({"stat": "수정 Bhargava et al. Durbin–Watson", "val": st["d1"]})
            lb.add_row({"stat": "Baltagi–Wu LBI", "val": st["LBI"]})
            lb.add_note(f"p값 미보고 (Stata 동일, 분포가 표본 구조에 의존); 사용 관측치 {st['N_LBI']}")
            lb.add_note("계산 방식: Bhargava, Franzini & Narendranathan (1982), Baltagi & Wu (1999) — 집단 평균 제거 잔차")
            self.add_table(lb)
        ms = {"N": int(st["N"]), "Groups": int(st["N_g"]), "R2_within": float(st["r2_w"]),
              "R2_between": float(st["r2_b"]), "R2_overall": float(st["r2_o"])}
        if fe and np.isfinite(st["F"]):
            ms["F"] = (float(st["F"]), float(st["F_p"]))
        if not fe and st["chi2"] is not None:
            ms["Chi2"] = (float(st["chi2"]), float(st["p_stata"]))
        self.model_stats = ms

    def _narrative(self, est):
        o, st = self.options, est.stats
        txt = f"{METHOD_LABEL[o['method']]} (집단 {st['N_g']}개, 관측치 {st['N']}개)"
        if st.get("chi2") is not None and o["method"] != "xtregar":
            txt += f", Wald χ²({st['df']}) = {st['chi2']:.2f}, {_pv(st['p'])}"
        elif o["method"] == "xtregar" and st["model"] == "fe" and np.isfinite(st["F"]):
            txt += f", F({st['df_b']}, {st['df_r']}) = {st['F']:.2f}, {_pv(st['F_p'])}, rho_ar = {st['rho_ar']:.3f}"
        elif o["method"] == "xtregar":
            txt += f", Wald χ²({st['df_m']}) = {st['chi2']:.2f}, rho_ar = {st['rho_ar']:.3f}"
        self.apa_narrative = txt + "."


class HausmanTaylorAnalysis(BaseAnalysis):
    """Stata xthtaylor (Hausman–Taylor; amacurdy = Amemiya–MaCurdy)."""

    def __init__(self, data: pd.DataFrame, **opts):
        o = ht_options(opts)
        extra = [v for v in o["endog"] if v not in o["covs"]]
        if extra:                                # Stata requires endog() ⊂ varlist; the form lists them separately
            o["covs"] = o["covs"] + extra
        if not o["covs"]:
            raise ValueError(f"xthtaylor: 설명변수(covs) 지정 필요. 사용 가능한 변수: {list(data.columns)}")
        title = "Amemiya–MaCurdy 추정 (xthtaylor, amacurdy)" if o["amacurdy"] else "Hausman–Taylor 추정 (xthtaylor)"
        super().__init__(name="hausmanTaylor", title=title, dataset=data, options=o)
        self.syntax_python = ht_python_call(o)
        self.syntax_r = "\n".join(r_code("hausmanTaylor", o) or [])
        self.run()

    def run(self):
        o = self.options
        if o["amacurdy"] and not o["time"]:
            raise ValueError("xthtaylor amacurdy: 시점 변수 필요 (xtset 개체 시점)")
        df = self.dataset
        tcol = o["time"]
        if not tcol and o["entity"] in df.columns:
            df = df.copy()
            tcol = "_order_in_panel"
            df[tcol] = df.groupby(o["entity"]).cumcount()
        P = PanelData(df, o["entity"], tcol, o["dep"], o["covs"], delta=o["delta"], label="xthtaylor")
        self.panel = P
        est = fit_xthtaylor(P, o)
        self.est = est
        st = est.stats
        rows = [("관측치 수 (N)", st["N"], "integer"), ("집단 수", st["N_g"], "integer")] + _group_rows(st["Ti"])
        if o["small"] and st["F"] is not None:
            rows += [(f"F({st['df_m']}, {st['df_r']})", st["F"], ""), ("Prob > F", st["F_p"], "pvalue")]
        if st["chi2"] is not None:
            rows += [(f"Wald χ²({st['df_m']})", st["chi2"], ""), ("Prob > χ²", st["p"], "pvalue")]
        t = _summary(rows, "모형 요약 (" + ("Amemiya–MaCurdy" if o["amacurdy"] else "Hausman–Taylor") + ")")
        if st["chi2"] is None:
            t.add_note("상수항 제외(noconstant): 모형 Wald 검정 미보고 (Stata 동일)")
        t.add_note("확률효과 u_i ~ i.i.d.; 내생 변수는 u_i와 상관 허용, e_it와는 무상관 가정")
        t.add_note("도구변수: 시간 변동 변수의 집단 내 편차, " + ("X1의 각 시점 값 (Amemiya–MaCurdy)" if o["amacurdy"]
                                                      else "X1의 집단 평균") + ", 시간 불변 외생 변수 Z1")
        t.add_note("계산 방식: Hausman & Taylor (1981), Amemiya & MaCurdy (1986), Stata [XT] xthtaylor Methods and "
                   "formulas; 라이브러리: statsmodels OLS (집단 내 추정), linearmodels IV2SLS (도구변수 단계)")
        if o["vce"] == "robust":
            t.add_note("표준오차: 군집-강건 (패널 변수 군집, 보정 (N−1)/(N−K)·G/(G−1)); Wald χ²는 상수항 포함 전체 계수 "
                       "(Stata vce(robust) 동일)")
        if P.n_dropped:
            t.add_note(f"결측으로 제외된 행: {P.n_dropped}")
        self.add_table(t)
        ct = _coef_table(est, blocks=True)
        ct.add_note("TV = 시간 변동(time varying), TI = 시간 불변(time invariant)")
        if o["small"]:
            ct.add_note(f"소표본 통계량 (small): t 자유도 {st['df_r']}")
        self.add_table(ct)
        a = ResultsTable(name="ancillary", title="분산성분 (sigma_u · sigma_e · rho)")
        a.add_column("item", "항목", col_type="text")
        a.add_column("val", "값", col_type="number")
        for k_, v in (("sigma_u", st["sigma_u"]), ("sigma_e", st["sigma_e"]), ("rho (u_i 분산 비중)", st["rho"])):
            a.add_row({"item": k_, "val": v})
        a.add_note("sigma_e² = 집단 내 회귀 RSS/(N − n); sigma_u² = (s² − sigma_e²)/T̄ (T̄: 불균형 시 조화평균)")
        self.add_table(a)
        if o["hausman"]:
            self._safe(lambda: self._hausman(P, est), "Hausman 검정 (xtreg, fe 대 xthtaylor)")
        ms = {"N": int(st["N"]), "Groups": int(st["N_g"])}
        if st["chi2"] is not None:
            ms["Chi2"] = (float(st["chi2"]), float(st["p"]))
        self.model_stats = ms
        txt = f"{'Amemiya–MaCurdy' if o['amacurdy'] else 'Hausman–Taylor'} 추정 (집단 {st['N_g']}개, 관측치 {st['N']}개)"
        if st["chi2"] is not None:
            txt += f", Wald χ²({st['df_m']}) = {st['chi2']:.2f}, {_pv(st['p'])}"
        txt += f", sigma_u = {st['sigma_u']:.3f}, sigma_e = {st['sigma_e']:.3f}"
        h = self.get_table("hausman")
        if h is not None and h.rows:
            r0 = h.rows[0]
            txt += f"; Hausman χ²({r0['df']}) = {r0['chi2']:.2f}, {_pv(r0['p'])}"
        self.apa_narrative = txt + "."

    def _safe(self, fn, label: str):
        try:
            fn()
        except Exception as e:  # the diagnostic failing must not hide the estimates
            t = ResultsTable(name=f"diag_error_{len(self.tables)}", title=f"{label} — 계산 불가")
            t.add_column("msg", "사유", col_type="text")
            t.add_row({"msg": str(e)})
            self.add_table(t)

    def _hausman(self, P, est):
        h = hausman_fe_ht(P, est)
        t = ResultsTable(name="hausman", title="Hausman 검정 (xtreg, fe 대 xthtaylor)")
        t.add_column("chi2", "χ²", col_type="number")
        t.add_column("df", "자유도", col_type="integer")
        t.add_column("p", "p", col_type="number", fmt="pvalue")
        t.add_column("basis", "자유도 기준", col_type="text")
        t.add_row({"chi2": h["chi2"], "df": h["df"], "p": h["p"], "basis": "차이 분산행렬 계수(rank) — Stata 기본"})
        if h["df_overid"] > 0 and h["df_overid"] != h["df"]:
            t.add_row({"chi2": h["chi2"], "df": h["df_overid"], "p": float(stats.chi2.sf(h["chi2"], h["df_overid"])),
                       "basis": "과대식별 제약 수 k1 − g2 (Stata 설명서 예: hausman fe ht, df(#))"})
        t.add_note("H0: 계수 차이가 체계적이지 않음 (xthtaylor 도구변수 타당); 비교 계수: " + ", ".join(h["names"]))
        if not h["posdef"]:
            t.add_note("V_b − V_B 양정치 아님 (Stata 동일 안내) — 일반화 역행렬 사용")
        t.add_note("계산 방식: Hausman (1978), Stata [R] hausman (SVD 일반화 역행렬, 각 모형의 일반 분산)")
        self.add_table(t)


# ----------------------------------------------------------------------------- registry glue
def _preset(params: Dict[str, Any]) -> Optional[str]:
    name = params.get("_alias") or params.get("_analysis")
    return ALIAS_PRESETS.get(str(name).strip().lower()) if name else None


def dispatch(t: str, df: pd.DataFrame, params: Dict[str, Any]):
    p = dict(params or {})
    preset = _preset(p)
    for k in ("_alias", "_analysis"):
        p.pop(k, None)
    if t == "panelGLS":
        o = normalize_options(p, preset=preset)
        return PanelGLSAnalysis(df, **_to_kwargs(o))
    if t == "hausmanTaylor":
        return HausmanTaylorAnalysis(df, **p)
    return None


def _to_kwargs(o: Dict[str, Any]) -> Dict[str, Any]:
    kw = {k: v for k, v in o.items() if k != "rhotypeSet"}
    if not o["rhotypeSet"]:
        kw.pop("rhotype")
    return kw


_GLS_DEFAULTS = {"panels": "iid", "corr": "independent", "igls": False, "iterate": 300, "tolerance": 1e-7,
                 "force": False, "nmk": False, "pcse": "correlated", "pairwise": False, "np1": False, "model": "re",
                 "rhof": None, "twostep": False, "lbi": False, "constant": True, "delta": 1.0}
_GLS_RELEVANT = {"xtgls": ("panels", "corr", "igls", "iterate", "tolerance", "force", "nmk", "constant", "delta"),
                 "xtpcse": ("corr", "pcse", "pairwise", "np1", "nmk", "constant", "delta"),
                 "xtregar": ("model", "rhof", "twostep", "lbi", "delta")}


def python_call(o: Dict[str, Any]) -> str:
    kw = [f"entity={o['entity']!r}", f"time={o['time']!r}", f"dep={o['dep']!r}", f"covs={list(o['covs'])!r}",
          f"method={o['method']!r}"]
    for k in _GLS_RELEVANT[o["method"]]:
        if o.get(k) != _GLS_DEFAULTS[k]:
            kw.append(f"{k}={o[k]!r}")
    if o.get("rhotypeSet"):
        kw.append(f"rhotype={o['rhotype']!r}")
    return f"res = sess.run('panelGLS', {', '.join(kw)})"


def ht_python_call(o: Dict[str, Any]) -> str:
    kw = [f"entity={o['entity']!r}", f"time={o['time']!r}", f"dep={o['dep']!r}", f"covs={list(o['covs'])!r}",
          f"endog={list(o['endog'])!r}"]
    for k, dv in (("constantVars", []), ("varying", []), ("amacurdy", False), ("small", False), ("constant", True),
                  ("hausman", True), ("vce", "conventional"), ("delta", 1.0)):
        if o.get(k) != dv:
            kw.append(f"{k}={o[k]!r}")
    return f"res = sess.run('hausmanTaylor', {', '.join(kw)})"


# ----------------------------------------------------------------------------- code export
def _resolve(t: str, params: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    low = str(t).strip().lower()
    try:
        if t == "panelGLS" or low in ("xtgls", "xtpcse", "xtregar", "panelgls"):
            p = dict(params or {})
            for k in ("_alias", "_analysis"):
                p.pop(k, None)
            if "rhotypeSet" in p and not p.get("rhotypeSet"):
                p.pop("rhotype", None)
            p.pop("rhotypeSet", None)
            return "panelGLS", normalize_options(p, preset=ALIAS_PRESETS.get(low))
        if t == "hausmanTaylor" or low in ("xthtaylor", "hausmantaylor"):
            return "hausmanTaylor", ht_options(params)
    except ValueError:
        return None
    return None


def stata_code(t: str, params: Dict[str, Any]) -> Optional[List[str]]:
    r = _resolve(t, params)
    if r is None:
        return None
    kind, o = r
    e, tm, y, x = o["entity"], o["time"], o["dep"], " ".join(o["covs"])
    out = xtset_lines(e, tm, o["delta"]) if tm else [f"xtset {e}"]
    if kind == "hausmanTaylor":
        opts = [f"endog({' '.join(o['endog'])})"]
        if o["constantVars"]:
            opts.append(f"constant({' '.join(o['constantVars'])})")
        if o["varying"]:
            opts.append(f"varying({' '.join(o['varying'])})")
        for k in ("amacurdy", "small"):
            if o[k]:
                opts.append(k)
        if not o["constant"]:
            opts.append("noconstant")
        if o["vce"] == "robust":
            opts.append("vce(robust)")
        cmd = f"xthtaylor {y} {x}, {' '.join(opts)}"
        if o["hausman"]:
            out += [f"quietly xtreg {y} {x}, fe", "estimates store fe", f"quietly {cmd}", "estimates store ht",
                    cmd, "hausman fe ht"]
        else:
            out.append(cmd)
        return out
    m = o["method"]
    opts: List[str] = []
    if m == "xtgls":
        if o["panels"] != "iid":
            opts.append(f"panels({'hetero' if o['panels'] == 'heteroskedastic' else 'correlated'})")
        if o["corr"] != "independent":
            opts.append(f"corr({o['corr']})")
            if o["rhotypeSet"]:
                opts.append(f"rhotype({o['rhotype']})")
        if o["igls"]:
            opts.append("igls")
            if o["iterate"] != 300:
                opts.append(f"iterate({o['iterate']})")
            if o["tolerance"] != 1e-7:
                opts.append(f"tolerance({o['tolerance']:g})")
        for k in ("force", "nmk"):
            if o[k]:
                opts.append(k)
        if not o["constant"]:
            opts.append("noconstant")
    elif m == "xtpcse":
        if o["corr"] != "independent":
            opts.append(f"correlation({o['corr']})")
            if o["rhotypeSet"]:
                opts.append(f"rhotype({o['rhotype']})")
            if o["np1"]:
                opts.append("np1")
        if o["pcse"] != "correlated":
            opts.append(o["pcse"])
        elif o["pairwise"]:
            opts.append("pairwise")
        if o["nmk"]:
            opts.append("nmk")
        if not o["constant"]:
            opts.append("noconstant")
    else:
        opts.append(o["model"])
        if o["rhof"] is not None:
            opts.append(f"rhof({o['rhof']:g})")
        elif o["rhotypeSet"]:
            opts.append(f"rhotype({o['rhotype']})")
        if o["twostep"]:
            opts.append("twostep")
        if o["lbi"]:
            opts.append("lbi")
    out.append(f"{m} {y} {x}" + (", " + " ".join(opts) if opts else ""))
    return out


def r_code(t: str, params: Dict[str, Any]) -> Optional[List[str]]:
    r = _resolve(t, params)
    if r is None:
        return None
    kind, o = r
    e, tm, y = o["entity"], o["time"], o["dep"]
    q = lambda s: '"' + str(s) + '"'
    f = f"{y} ~ " + " + ".join(o["covs"])
    if kind == "hausmanTaylor":
        covs = o["covs"] + [v for v in o["endog"] if v not in o["covs"]]
        f = f"{y} ~ " + " + ".join(covs)
        exog = [v for v in covs if v not in o["endog"]]
        if o["constantVars"]:
            x2 = [v for v in o["endog"] if v not in o["constantVars"]]
        elif o["varying"]:
            x2 = [v for v in o["endog"] if v in o["varying"]]
        else:
            x2 = None
        out = [f"pdata <- plm::pdata.frame(data, index = c({q(e)}, {q(tm)}))",
               "# 3부분 공식: 회귀식 | 외생 변수(X1, Z1) | 시간 변동 내생 변수(X2); 시간 불변 내생 변수(Z2)는 나머지 도구로 식별"]
        if x2 is None:
            out.append(f"# 내생 변수 {', '.join(o['endog'])} 중 시간 변동 변수만 세 번째 부분에 지정 (constant() 미지정으로 자동 구분 불가)")
        out += [f"fit <- plm::plm({f} | {' + '.join(exog)} | {' + '.join(x2) if x2 is not None else ' + '.join(o['endog'])}, "
                f"data = pdata, model = \"random\", random.method = \"ht\", "
                f"inst.method = {q('am' if o['amacurdy'] else 'baltagi')})",
                "summary(fit)",
               "# 차이: plm의 분산성분은 Stata xthtaylor와 같은 Hausman–Taylor 공식이나 불균형 패널의 T̄·자유도 처리는 "
               "다를 수 있음"]
        if o["hausman"]:
            out += [f"fe <- plm::plm({f}, data = pdata, model = \"within\")", "plm::phtest(fe, fit)"]
        return out
    m = o["method"]
    out = [f"data <- data[order(data${e}, data${tm}), ]"] if tm else []
    if m == "xtgls":
        if o["panels"] == "correlated" or o["corr"] == "psar1":
            ac = {"independent": "none", "ar1": "ar1", "psar1": "psar1"}[o["corr"]]
            pm = "parks" if o["panels"] == "correlated" else ("phet" if o["panels"] == "heteroskedastic" else "none")
            out += ["# panelAR 패키지 (Parks–Kmenta FGLS 포함)",
                    f"fit <- panelAR::panelAR({f}, data = data, panelVar = {q(e)}, timeVar = {q(tm)}, "
                    f"autoCorr = {q(ac)}, panelCorrMethod = {q(pm)}, rho.na.rm = TRUE)",
                    "summary(fit)",
                    "# 차이: panelAR의 rho 기본 추정 방식·분산 분모는 Stata xtgls와 다를 수 있음 (Stata 기본 분모 N)"]
        else:
            w = ", weights = nlme::varIdent(form = ~ 1 | " + e + ")" if o["panels"] == "heteroskedastic" else ""
            cr = f", correlation = nlme::corAR1(form = ~ {tm} | {e})" if o["corr"] == "ar1" else ""
            out += [f"fit <- nlme::gls({f}, data = data{w}{cr}, method = \"ML\")", "summary(fit)",
                    "# 차이: nlme::gls는 최대우도 동시 추정 — Stata xtgls 2·3단계 FGLS와 계수가 다름; "
                    "corr(independent)에서 igls 결과(MLE)와 대응"]
    elif m == "xtpcse":
        if o["corr"] == "independent" and o["pcse"] == "correlated":
            out += [f"ols <- lm({f}, data = data)",
                    f"pc <- pcse::pcse(ols, groupN = data${e}, groupT = data${tm}, pairwise = {'TRUE' if o['pairwise'] else 'FALSE'})",
                    "summary(pc)",
                    "# 차이: pcse 패키지는 균형 여부에 따라 casewise/pairwise 처리 방식이 Stata와 다를 수 있음"]
        else:
            pm = {"correlated": "pcse", "hetonly": "phet", "independent": "none"}[o["pcse"]]
            ac = {"independent": "none", "ar1": "ar1", "psar1": "psar1"}[o["corr"]]
            out += ["# panelAR 패키지",
                    f"fit <- panelAR::panelAR({f}, data = data, panelVar = {q(e)}, timeVar = {q(tm)}, "
                    f"autoCorr = {q(ac)}, panelCorrMethod = {q(pm)}, rho.na.rm = TRUE)",
                    "summary(fit)",
                    "# 차이: panelAR의 공통 rho 가중치·rho 경계 처리는 Stata xtpcse(np1·[−1,1] 제한)와 다를 수 있음"]
    else:
        out += [f"pdata <- plm::pdata.frame(data, index = c({q(e)}, {q(tm)}))",
                "# Baltagi–Wu AR(1) 고정·확률효과 추정량의 R 대응 함수 없음 (plm 미지원) — 검정 통계량만 대응",
                f"plm::pbnftest(plm::plm({f}, data = pdata, model = \"within\"))              # 수정 BFN Durbin–Watson",
                f"plm::pbnftest(plm::plm({f}, data = pdata, model = \"within\"), test = \"lbi\")  # Baltagi–Wu LBI"]
    return out


# ----------------------------------------------------------------------------- Stata command input
def _parse_cmd(s: str, cmd: str):
    m = re.match(rf"^{cmd}\s+([^,]+?)\s*(?:,\s*(.*))?$", s, re.IGNORECASE)
    if not m:
        return None, None, _err(f"{cmd}: 종속변수와 설명변수 필요 — " + s)
    body = m.group(1)
    if re.search(r"\s(if|in)\s", " " + body + " ") or re.search(r"[\[\]]", body):
        return None, None, _err(f"{cmd}: if·in 조건·가중치 미지원 — sess.filter_rows 등으로 표본을 먼저 정한 뒤 실행 — " + s)
    toks = body.split()
    if any(re.search(r"[.#]", v) for v in toks):
        return None, None, _err("요인변수·시계열 연산자(i.·L.·D.·#) 미지원: 변수를 만든 뒤 지정 — " + s)
    if len(toks) < 2:
        return None, None, _err(f"{cmd}: 종속변수와 설명변수 필요 — " + s)
    return toks, _split_opts(m.group(2) or ""), None


def _abbr(opts: Dict[str, str], full: str, minlen: int) -> Optional[str]:
    """Pop a Stata option given with any abbreviation of at least `minlen` letters."""
    for k in list(opts):
        if len(k) >= minlen and full.startswith(k):
            return opts.pop(k)
    return None


def transpile(line: str) -> Optional[str]:
    s = line.strip()
    head = s.split()[0].lower() if s.split() else ""
    if head in ("xtgls", "xtpcse", "xtregar"):
        toks, opts, err = _parse_cmd(s, head)
        if err:
            return err
        kw = [f"dep={toks[0]!r}", f"covs={toks[1:]!r}"]
        if _abbr(opts, "noconstant", 3) is not None:
            if head == "xtregar":
                return _err("xtregar: noconstant 옵션 없음 — " + s)
            kw.append("constant=False")
        rt = _abbr(opts, "rhotype", 4 if head == "xtregar" else 3)
        if rt is not None:
            kw.append(f"rhotype={rt.strip().lower()!r}")
        if head == "xtgls":
            v = _abbr(opts, "panels", 1)
            if v is not None:
                kw.append(f"panels={v.strip().lower()!r}")
            v = _abbr(opts, "corr", 1)
            if v is not None:
                v = v.strip().lower()
                if v and "unstructured".startswith(v) and len(v) >= 1 and v[0] == "u":
                    return _err("xtgls corr(unstructured) (StataNow) 미지원 — " + s)
                kw.append(f"corr={v!r}")
            for flag in ("igls", "force", "nmk"):
                if flag in opts:
                    opts.pop(flag)
                    kw.append(f"{flag}=True")
            v = _abbr(opts, "iterate", 4)
            if v is not None:
                kw.append(f"iterate={int(v)}")
            v = _abbr(opts, "tolerance", 3)
            if v is not None:
                kw.append(f"tolerance={float(v)!r}")
            for k in ("nolog", "log"):
                opts.pop(k, None)
        elif head == "xtpcse":
            v = _abbr(opts, "correlation", 1)
            if v is not None:
                kw.append(f"corr={v.strip().lower()!r}")
            if _abbr(opts, "hetonly", 3) is not None:
                kw.append("pcse='hetonly'")
            if _abbr(opts, "independent", 1) is not None:
                kw.append("pcse='independent'")
            if _abbr(opts, "pairwise", 1) is not None:
                kw.append("pairwise=True")
            _abbr(opts, "casewise", 2)
            for flag in ("np1", "nmk"):
                if flag in opts:
                    opts.pop(flag)
                    kw.append(f"{flag}=True")
            _abbr(opts, "detail", 1)
        else:
            for k in ("fe", "re"):
                if k in opts:
                    opts.pop(k)
                    kw.append(f"model={k!r}")
            v = opts.pop("rhof", None)
            if v is not None:
                kw.append(f"rhof={float(v)!r}")
            if _abbr(opts, "twostep", 3) is not None:
                kw.append("twostep=True")
            if "lbi" in opts:
                opts.pop("lbi")
                kw.append("lbi=True")
        if opts:
            return _err(f"{head} 옵션 미지원: {', '.join(opts)} — " + s)
        if head == "xtgls" and rt is not None and not any(k.startswith("corr=") and "'i" not in k for k in kw):
            return _err("xtgls: rhotype()는 corr(ar1)·corr(psar1)와 함께 지정 (Stata: rhotype() not allowed with default "
                        "corr(independent)) — " + s)
        return f"res = sess.run({head!r}, {_xt_index_args()}, {', '.join(kw)})"
    if head == "xthtaylor":
        toks, opts, err = _parse_cmd(s, head)
        if err:
            return err
        kw = [f"dep={toks[0]!r}", f"covs={toks[1:]!r}"]
        v = _abbr(opts, "endog", 1)
        if v is None:
            return _err("xthtaylor: endog() 필수 — " + s)
        kw.append(f"endog={v.split()!r}")
        v = _abbr(opts, "constant", 4)
        if v is not None:
            kw.append(f"constantVars={v.split()!r}")
        v = _abbr(opts, "varying", 1)
        if v is not None:
            kw.append(f"varying={v.split()!r}")
        if _abbr(opts, "amacurdy", 2) is not None:
            kw.append("amacurdy=True")
        if "small" in opts:
            opts.pop("small")
            kw.append("small=True")
        if _abbr(opts, "noconstant", 6) is not None:
            kw.append("constant=False")
        v = opts.pop("vce", None)
        if v is not None:
            vv = v.strip().lower()
            if vv in ("robust", "r", "rob", "robu", "robus"):
                kw.append("vce='robust'")
            elif vv.startswith("conv") or vv in ("oim", "ols"):
                pass
            elif vv.split()[0].startswith("cl") and len(vv.split()) == 2:
                return _err("xthtaylor vce(cluster 변수): 패널 변수 외 군집 미지원 — vce(robust)는 패널 군집과 동일 — " + s)
            else:
                return _err(f"xthtaylor vce({v}) 미지원 — " + s)
        if opts:
            return _err(f"xthtaylor 옵션 미지원: {', '.join(opts)} — " + s)
        return f"res = sess.run('xthtaylor', {_xt_index_args()}, {', '.join(kw)})"
    m = re.match(r"^hausman\s+(\w+)\s+(\w+)\s*(?:,.*)?$", s, re.IGNORECASE)
    is_ht = lambda n: bool(re.search(r"(^|_)(ht|xtht|htaylor|xthtaylor)\d*$|taylor", n, re.IGNORECASE))
    is_am = lambda n: bool(re.fullmatch(r"(.*_)?am|amacurdy", n, re.IGNORECASE))
    if m and (is_am(m.group(1)) or is_am(m.group(2))) and (is_ht(m.group(1)) or is_ht(m.group(2))):
        return _err("Hausman–Taylor 대 Amemiya–MaCurdy Hausman 검정 미지원 (xtreg, fe 대 xthtaylor만 지원) — " + s)
    if m and (is_ht(m.group(1)) or is_ht(m.group(2))):
        return (f"res = sess.run('hausmanTaylor', **dict(next(a for a in reversed(sess.history) "
                f"if getattr(a, 'name', '') == 'hausmanTaylor').options, hausman=True))"
                f"  # [Stata] {s} → xtreg, fe 대 xthtaylor Hausman 검정은 결과에 포함")
    return None


EXTENSION = {
    "catalog": [
        {"id": "panelGLS", "group": "Linear Models",
         "title": "Panel FGLS / PCSE / AR(1) (xtgls, xtpcse, xtregar)", "module": "openstat",
         "icon": "analysis-linearmodels.svg"},
        {"id": "hausmanTaylor", "group": "Linear Models",
         "title": "Hausman–Taylor / Amemiya–MaCurdy (xthtaylor)", "module": "openstat",
         "icon": "analysis-linearmodels.svg"},
    ],
    "aliases": {"xtgls": "panelGLS", "xtpcse": "panelGLS", "xtregar": "panelGLS", "panelgls": "panelGLS",
                "panel_gls": "panelGLS", "xthtaylor": "hausmanTaylor", "hausman_taylor": "hausmanTaylor",
                "hausmantaylor": "hausmanTaylor"},
    "dispatch": dispatch,
    "uses_alias": True,          # registry passes the typed name as params["_alias"] (alias -> method preset)
    "stata": stata_code,
    "r": r_code,
    "transpile": transpile,
}
