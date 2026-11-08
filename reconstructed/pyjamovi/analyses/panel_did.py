"""
OpenStat Studio — difference-in-differences and event studies for panel data (Stata xtdidregress / xthdidregress).
License: MIT

One analysis id ``panelDiD`` with three methods (option ``method``):

method / estimator      quantity                              computation (library call or published method)                  Stata command
twfe                    ATET of the two-way FE DiD            linearmodels PanelOLS(entity_effects=True) + time dummies,      xtdidregress (y x) (d), group() time()
                                                              cluster on group() with (N-1)/(N-K)·G/(G-1) (xtreg, fe)
  estat trendplots      observed means by group over time     pandas groupby                                                 estat trendplots, omeans
  estat ptrends         linear pre-trend test                 within regression + w·1(pre)·t, w·1(post)·t; Wald F             estat ptrends
                                                              (Stata [CAUSAL] didregress postestimation Methods and formulas)
  estat granger         anticipation test                     within regression + leads 1(t>=j)·w, j = 2..J-1; Wald F         estat granger
  estat grangerplot     lead/lag coefficients                 within regression with binned endpoint leads/lags (same M&F)    estat grangerplot
  wildbootstrap         restricted wild cluster bootstrap     Roodman, MacKinnon, Nielsen & Webb (2019) cluster-level form;   xtdidregress ..., wildbootstrap()
                                                              p-value and CI inversion as [CAUSAL] didregress M&F
event                   event study, reference period -1      linearmodels PanelOLS with relative-time dummies + time dummies, xtreg y F* L* x i.t, fe vce(cluster id)
                                                              cluster on group(); optional binned endpoints
hdid ra|ipw|aipw        ATT(g,t), Callaway & Sant'Anna (2021) 2x2 DiD per (g,t) on paired observations; statsmodels OLS       xthdidregress ra|ipw|aipw (y x) (d z),
                                                              (outcome model) and statsmodels Logit (propensity); influence    group() controlgroup() basetime()
                                                              functions of Sant'Anna & Zhao (2020) (reg, std. IPW, DR);
                                                              cluster sums of the influence functions
  estat aggregation     simple / cohort / time / dynamic      weighted ATT(g,t) with weight influence functions,              estat aggregation[, cohort time dynamic]
                                                              Callaway & Sant'Anna (2021) section 4.2; Stata weights(timecohort)
  estat ptrends         all pre-period ATT(g,t) = 0           Wald chi2 with the influence-function VCE                        estat ptrends
hdid twfe               ATT(g,t), Wooldridge (2025) extended  statsmodels OLS on within-transformed data (cohort x time       xthdidregress twfe (y x) (d), group()
                        TWFE (Mundlak)                        cells and covariate interactions); ATT and SE as margins vce(uncond.):
                                                              cluster scores + first-stage cohort-mean scores, factor
                                                              (N-1)/(N-K)·G/(G-1) with K = rank of the two-way Mundlak design
  estat aggregation     margins over treated observations     same score form (Stata [CAUSAL] hdidregress postestimation)     estat aggregation
  estat ptrends         pre-period cohort x time cells = 0    augmented regression, Wald F                                    estat ptrends

Conventions that reproduce Stata 19.5 (checked on the balanced sample and on an unbalanced variant):
  * cohorts = first treated period of each group() in the estimation sample (gaps can create later pseudo-cohorts);
  * not-yet-treated controls for ATT(g,t): never treated or first treated after max(t, g);
  * base period: g-1 (the cohort's last observed period before g) for t >= g; t-1 (adaptive) or g-1 (common) before;
    covariates taken at the earlier period of the 2x2 pair; only units observed in both periods (paired);
  * aggregation weights (weights(timecohort)): number of cohort units in the 2x2 sample of each ATT(g,t);
  * all units eventually treated: with notyet the last cohort is the control group and periods from its adoption on
    are not estimated (ra/ipw/aipw cells omitted, twfe cells not reported);
  * extended TWFE: a cell whose covariate interactions are not identified (for example a single-unit cohort) is omitted.

Library evaluation for Callaway–Sant'Anna (2026-11): csdid 0.4.5 (MIT, port of R did by its authors) reproduces Stata's
ra / ipw / aipw ATT(g,t), SEs and aggregations on the sample data when the outcome and treatment covariates coincide,
but it takes one covariate formula for both models (Stata aipw / ipw allow different sets), and it pins drdid==1.1.7
and requires plotnine and twine at run time. differences 0.3.0 is GPL-3 (not usable in this MIT package). pyfixest 0.60
offers did2s / Sun–Abraham / local projections, not ATT(g,t) with ra / ipw / aipw. The ATT(g,t) estimators therefore follow
the published formulas (Callaway & Sant'Anna 2021; Sant'Anna & Zhao 2020; Stata [CAUSAL] xthdidregress Methods and
formulas) with statsmodels for the nuisance models; csdid served as a cross-check.

Verified against Stata 19.5 (StataNow MP) on sample_data/Panel_DiD_Policy.csv; reference values are hard-coded in
tests/test_panel_did.py (the Stata do-files and logs are kept outside the repository).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseAnalysis, ResultsTable, resolve_choice
from .panel_module import (Fit, PanelData, _as_list, _as_single, _err, _flag, _fmt_val, _opt, _split_opts,
                           fit_within, independent_columns, xtset_lines)

METHODS = ("twfe", "event", "hdid")
METHOD_ALIASES = {"xtdidregress": "twfe", "did": "twfe", "didregress": "twfe", "classic": "twfe",
                  "eventstudy": "event", "event_study": "event", "es": "event",
                  "xthdidregress": "hdid", "heterogeneous": "hdid", "cs": "hdid", "csdid": "hdid",
                  "callaway": "hdid", "staggered": "hdid"}
ESTIMATORS = ("ra", "ipw", "aipw", "twfe")
EST_ALIASES = {"reg": "ra", "or": "ra", "regression": "ra", "dr": "aipw", "doubly": "aipw", "drdid": "aipw",
               "etwfe": "twfe", "wooldridge": "twfe", "mundlak": "twfe"}
CONTROL_GROUPS = ("never", "notyet")
CG_ALIASES = {"nevertreated": "never", "never_treated": "never", "notyettreated": "notyet", "not_yet": "notyet",
              "notyet_treated": "notyet", "nyt": "notyet"}
BASE_TIMES = ("adaptive", "common")
BT_ALIASES = {"varying": "adaptive", "universal": "common"}
WB_WEIGHTS = ("rademacher", "mammen", "webb", "normal", "gamma")
EST_LABEL = {"ra": "회귀 조정 (RA)", "ipw": "역확률 가중 (IPW)", "aipw": "이중 강건 (AIPW)",
             "twfe": "확장 이원 고정효과 (Wooldridge TWFE)"}
ALIAS_PRESETS = {"xtdidregress": "twfe", "did": "twfe", "eventstudy": "event", "xthdidregress": "hdid"}

DEFAULTS: Dict[str, Any] = {
    "group": "", "method": "twfe", "estimator": "ra", "tcovs": [], "controlGroup": "never", "baseTime": "adaptive",
    "cluster": "", "trendPlots": True, "ptrends": True, "granger": True, "grangerPlot": True,
    "nLeads": None, "nLags": None, "baseline": None, "refPeriod": -1, "bin": True,
    "aggOverall": True, "aggCohort": True, "aggTime": False, "aggDynamic": True, "atetPlot": False,
    "wildBootstrap": False, "wbReps": 1000, "wbWeight": "rademacher", "wbSeed": 12345,
    "showControls": False, "level": 95.0, "delta": 1.0,
}


# ----------------------------------------------------------------------------- small helpers
def _int_or_none(v: Any, label: str) -> Optional[int]:
    if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{label}: 정수 필요 (입력값 {v!r})")
    if not f.is_integer():
        raise ValueError(f"{label}: 정수 필요 (입력값 {v!r})")
    return int(f)


def _num_or_none(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    return float(v)


def _pstr(p: float) -> str:
    return "< .001" if p < 0.001 else f"= {p:.3f}"


def _within_matrix(a: np.ndarray, g: np.ndarray, n: int) -> np.ndarray:
    """x - xbar_i (entity demeaning)."""
    a2 = a if a.ndim == 2 else a[:, None]
    s = np.zeros((n, a2.shape[1]))
    np.add.at(s, g, a2)
    cnt = np.bincount(g, minlength=n).astype(float)
    out = a2 - (s / cnt[:, None])[g]
    return out if a.ndim == 2 else out[:, 0]


def _cluster_sum(a: np.ndarray, cl: np.ndarray, n_cl: int) -> np.ndarray:
    a2 = a if a.ndim == 2 else a[:, None]
    s = np.zeros((n_cl, a2.shape[1]))
    np.add.at(s, cl, a2)
    return s


def _independent_qr(M: np.ndarray, protect: Sequence[int] = (), tol: float = 1e-9) -> List[int]:
    """Columns kept (original order) after dropping those collinear with columns entering earlier — the rule of
    panel_module.independent_columns (Stata keeps the earlier variable), computed with one LAPACK Householder QR
    (scipy.linalg.qr): |R_jj| is the norm of column j after projecting out the earlier columns."""
    from scipy.linalg import qr
    if M.shape[1] == 0:
        return []
    order = list(protect) + [j for j in range(M.shape[1]) if j not in set(protect)]
    A = M[:, order]
    nrm = np.linalg.norm(A, axis=0)
    R = qr(A, mode="r", check_finite=False)[0]
    d = np.abs(np.diag(R)) if R.shape[0] >= R.shape[1] else np.r_[np.abs(np.diag(R)), np.zeros(R.shape[1] - R.shape[0])]
    keep = [order[j] for j in range(len(order)) if nrm[j] > 0 and np.isfinite(nrm[j]) and d[j] > tol * nrm[j]]
    # a column dependent on earlier ones leaves later diagonal entries unaffected only if it is removed; refine
    # once without the dropped columns so that the decision for later columns uses the kept set only
    if len(keep) < len(order):
        A2 = M[:, [j for j in order if j in set(keep)]]
        R2 = qr(A2, mode="r", check_finite=False)[0]
        d2 = np.abs(np.diag(R2))
        n2 = np.linalg.norm(A2, axis=0)
        kk = [j for j in order if j in set(keep)]
        keep = [kk[i] for i in range(len(kk)) if d2[i] > tol * n2[i]]
    return sorted(keep)


def _coef_columns(t: ResultsTable, stat: str, first: Sequence[Tuple[str, str, str]]):
    for name, title, typ in first:
        t.add_column(name, title, col_type=typ)
    t.add_column("b", "추정치", col_type="number")
    t.add_column("se", "표준오차", col_type="number")
    t.add_column("stat", stat, col_type="number")
    t.add_column("p", "p", col_type="number", fmt="pvalue")
    t.add_column("cil", "CI 하한", col_type="number")
    t.add_column("ciu", "CI 상한", col_type="number")


def _row(b: float, se: float, df_r: Optional[float], level: float) -> Dict[str, Any]:
    if se is None or not np.isfinite(se) or se <= 0:
        return {"b": b, "se": None, "stat": None, "p": None, "cil": None, "ciu": None}
    st = b / se
    if df_r is None:
        p = 2 * stats.norm.sf(abs(st))
        c = stats.norm.ppf(0.5 + level / 200)
    else:
        p = 2 * stats.t.sf(abs(st), df_r)
        c = stats.t.ppf(0.5 + level / 200, df_r)
    return {"b": float(b), "se": float(se), "stat": float(st), "p": float(p), "cil": float(b - c * se),
            "ciu": float(b + c * se)}


def _wald(b: np.ndarray, V: np.ndarray, df_r: Optional[float]) -> Optional[Dict[str, Any]]:
    """Joint test b = 0 (F with df_r, chi2 without); constraints with zero variance are dropped (Stata test)."""
    if len(b) == 0:
        return None
    Vs = (V + V.T) / 2
    keep = independent_columns(Vs, tol=1e-10) if len(b) > 1 else ([0] if Vs[0, 0] > 0 else [])
    if not keep:
        return None
    bk, Vk = b[keep], Vs[np.ix_(keep, keep)]
    W = float(bk @ np.linalg.solve(Vk, bk))
    q = len(keep)
    out = {"q": q, "dropped": len(b) - q, "chi2": W}
    if df_r is None:
        out.update({"stat": W, "p": float(stats.chi2.sf(W, q)), "df_r": None})
    else:
        out.update({"stat": W / q, "p": float(stats.f.sf(W / q, q, df_r)), "df_r": float(df_r)})
    return out


# ----------------------------------------------------------------------------- data
class DiDData:
    """Listwise-complete panel (panel_module.PanelData) with treatment, group and cohort information."""

    def __init__(self, df: pd.DataFrame, entity: str, time: str, dep: str, covs: List[str], tcovs: List[str],
                 treat: str, group: Optional[str], cluster: Optional[str], delta: float = 1.0):
        if not treat:
            raise ValueError(f"이중차분: 처치 변수(treat) 지정 필요. 사용 가능한 변수: {list(df.columns)}")
        if treat in covs or treat in tcovs:
            raise ValueError(f"이중차분: 처치 변수 '{treat}'가 공변량에도 포함됨")
        if treat == dep:
            raise ValueError("이중차분: 처치 변수와 결과 변수가 같음")
        group = group or entity
        for nm, v in (("group", group), ("cluster", cluster)):
            if v and v not in df.columns:
                raise ValueError(f"이중차분: 데이터에 없는 변수 — {v} ({nm}). 사용 가능한 변수: {list(df.columns)}")
        allc = list(dict.fromkeys(list(covs) + list(tcovs) + [treat]))
        extra = [c for c in (group, cluster) if c and c not in (entity, time)]
        P = PanelData(df, entity, time, dep, allc, extra=extra, delta=delta, label="이중차분")
        self.P = P
        self.entity, self.time, self.dep, self.treat, self.group = entity, time, dep, treat, group
        self.covs, self.tcovs = list(covs), list(tcovs)
        self.cluster_var = cluster
        d = P.df[treat].to_numpy(float)
        bad = ~np.isin(d, (0.0, 1.0))
        if bad.any():
            raise ValueError(f"이중차분: 처치 변수 '{treat}'는 0/1 값 필요 (다른 값 {np.unique(d[bad])[:5].tolist()})")
        self.d = d
        self.N, self.n, self.g, self.tpos = P.N, P.n, P.g, P.tpos
        self.y = P.y
        # group codes per row; panels must be nested within groups
        if group == entity:
            self.grp = P.g.copy()
            self.grp_labels = np.asarray(P.entity_labels)
        else:
            codes, labs = pd.factorize(P.df[group], sort=True)
            self.grp, self.grp_labels = codes.astype(np.int64), np.asarray(labs)
            chk = pd.DataFrame({"e": P.g, "c": self.grp}).groupby("e")["c"].nunique()
            if (chk > 1).any():
                raise ValueError(f"이중차분: 패널(개체 '{entity}')이 집단 변수 '{group}' 안에 포함되지 않음 "
                                 "(한 개체가 여러 집단에 속함) — Stata도 군집 오류 처리")
        self.n_grp = int(self.grp.max()) + 1
        # treatment constant within group x time
        chk = pd.DataFrame({"c": self.grp, "t": self.tpos, "d": d}).groupby(["c", "t"])["d"].nunique()
        if (chk > 1).any():
            raise ValueError(f"이중차분: 같은 집단·시점 안에서 처치 값이 다름 (집단 변수 '{group}' 수준에서 처치가 정해져야 함)")
        self.tlist = np.unique(self.tpos)
        lab = {}
        for tp, tv in zip(self.tpos, P.time_values):
            lab.setdefault(int(tp), tv)
        self.tlab = lab
        # cohort (first treated period) per group and absorbing check
        gd = pd.DataFrame({"c": self.grp, "t": self.tpos, "d": d}).drop_duplicates(["c", "t"]).sort_values(["c", "t"])
        first = gd[gd.d == 1].groupby("c")["t"].min()
        self.cohort_grp = np.full(self.n_grp, -1, dtype=np.int64)
        self.cohort_grp[first.index.to_numpy()] = first.to_numpy()
        after = gd.merge(first.rename("f"), left_on="c", right_index=True)
        self.absorbing = bool(((after.t < after.f) | (after.d == 1)).all())
        self.cohort_row = self.cohort_grp[self.grp]
        self.w_row = (self.cohort_row >= 0).astype(float)      # ever-treated group indicator
        unit_grp = np.zeros(self.n, dtype=np.int64)
        unit_grp[self.g] = self.grp
        self.unit_grp = unit_grp
        self.cohort_unit = self.cohort_grp[unit_grp]

    def label(self, tp: int) -> str:
        return _fmt_val(self.tlab.get(int(tp), tp))

    def time_dummies(self) -> Tuple[np.ndarray, List[str]]:
        cols = [(self.tpos == u).astype(float) for u in self.tlist[1:]]
        names = [f"{self.label(u)}.{self.time}" for u in self.tlist[1:]]
        return (np.column_stack(cols) if cols else np.zeros((self.N, 0))), names

    def cluster_codes(self, cluster: Optional[str]) -> Tuple[Optional[np.ndarray], str, int]:
        """Cluster array for fit_within (None = panel variable) and its name and count."""
        cl = cluster or self.group
        if cl == self.entity:
            return None, cl, self.n
        codes = pd.factorize(self.P.df[cl])[0]
        return codes, cl, int(codes.max()) + 1


# ----------------------------------------------------------------------------- wild cluster bootstrap
def _wb_weights(rng: np.random.Generator, kind: str, G: int, B: int) -> np.ndarray:
    if kind == "rademacher":
        return rng.choice(np.array([-1.0, 1.0]), size=(G, B))
    if kind == "mammen":
        phi = (1 + np.sqrt(5)) / 2
        return np.where(rng.random((G, B)) < phi / np.sqrt(5), 1 - phi, phi)
    if kind == "webb":
        v = np.sqrt(np.array([1.5, 1.0, 0.5]))
        return rng.choice(np.concatenate([-v, v]), size=(G, B))
    if kind == "normal":
        return rng.standard_normal((G, B))
    return rng.gamma(4.0, 0.5, size=(G, B)) - 2.0


def wild_bootstrap(y_res: np.ndarray, d_res: np.ndarray, Zt: np.ndarray, cl: np.ndarray, level: float = 95.0,
                   reps: int = 1000, weight: str = "rademacher", seed: Optional[int] = 12345,
                   W: Optional[np.ndarray] = None, return_draws: bool = False) -> Dict[str, Any]:
    """Restricted wild cluster bootstrap for one coefficient (Stata didregress wildbootstrap()).

    y_res, d_res: outcome and treatment after partialling out the panel effects and the other regressors (Zt, within
    transformed). Cluster-level algebra of Roodman, MacKinnon, Nielsen & Webb (2019): each replication needs only
    cluster sums. Equal-tail p-value 2·min(P_l, P_u) and CI bounds by bisection ([CAUSAL] didregress M&F).
    W: optional G x (B-1) weight matrix (used for full enumeration in checks); return_draws adds the bootstrap t
    statistics of H0: ATET = 0 (without the small-sample factor, which cancels in every comparison)."""
    G = int(cl.max()) + 1
    alpha = 1 - level / 100.0
    reps_adj = int(np.ceil(round(reps * alpha / 2, 9)) / (alpha / 2) + 1e-9)
    DD = float(d_res @ d_res)
    Dg = _cluster_sum(d_res ** 2, cl, G)[:, 0]
    if Zt.shape[1]:
        A = _cluster_sum(d_res[:, None] * Zt, cl, G)                 # G x K
        ZtZi = np.linalg.pinv(Zt.T @ Zt)
    else:
        A = np.zeros((G, 0))
        ZtZi = np.zeros((0, 0))
    S0 = _cluster_sum(d_res * y_res, cl, G)[:, 0]
    C0 = _cluster_sum(Zt * y_res[:, None], cl, G) if Zt.shape[1] else np.zeros((G, 0))
    b = float(d_res @ y_res) / DD
    e = y_res - b * d_res
    sc = _cluster_sum(d_res * e, cl, G)[:, 0]
    se1 = np.sqrt(float(sc @ sc)) / DD                                 # factor-free cluster SE (cancels in t ratios)
    if W is None:
        rng = np.random.default_rng(seed)
        W = _wb_weights(rng, weight, G, reps_adj - 1)
    M_AC_cache = A @ ZtZi if A.shape[1] else None

    def tstats(c: float):
        S = S0 - c * Dg
        C = C0 - c * A
        bstar = (S @ W) / DD                                           # b* - c
        score = S[:, None] * W - Dg[:, None] * bstar[None, :]
        if M_AC_cache is not None:
            score -= (M_AC_cache @ C.T) @ W
        sestar = np.sqrt((score ** 2).sum(0)) / DD
        tstar = bstar / sestar
        t_obs = (b - c) / se1
        return t_obs, tstar

    t_obs, tstar = tstats(0.0)
    Bn = tstar.size + 1
    p_l = float(np.sum(np.abs(t_obs) < np.abs(tstar))) / Bn
    p_u = float(np.sum(np.abs(t_obs) > np.abs(tstar))) / Bn
    pval = min(1.0, 2 * min(p_l, p_u))

    def upper_share(c):                       # share of t*(c) above t(c): rises with c
        t0, ts = tstats(c)
        return float(np.sum(ts >= t0)) / Bn

    def lower_share(c):                       # share of t*(c) below t(c): falls with c
        t0, ts = tstats(c)
        return float(np.sum(ts <= t0)) / Bn

    def bisect(fun, lo, hi, target, increasing):
        f_lo, f_hi = fun(lo), fun(hi)
        k = 0
        while (f_lo > target if increasing else f_lo < target) and k < 60:
            lo -= (hi - lo)
            f_lo = fun(lo)
            k += 1
        k = 0
        while (f_hi < target if increasing else f_hi > target) and k < 60:
            hi += (hi - lo)
            f_hi = fun(hi)
            k += 1
        for _ in range(100):
            mid = (lo + hi) / 2
            f = fun(mid)
            if (f > target) if increasing else (f < target):
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-10 * max(1.0, abs(mid)):
                break
        return (lo + hi) / 2

    se_naive = se1 * np.sqrt(G / (G - 1.0))
    width = 10 * se_naive if se_naive > 0 else 1.0
    lb = bisect(upper_share, b - width, b, alpha / 2, True)
    ub = bisect(lower_share, b, b + width, alpha / 2, False)
    out = {"b": b, "t": float(t_obs), "p": pval, "lb": lb, "ub": ub, "reps": int(reps_adj), "weight": weight,
           "G": G, "reps_note": reps_adj != reps}
    if return_draws:
        out["tstar"] = tstar
    return out


# ----------------------------------------------------------------------------- Callaway & Sant'Anna ATT(g,t)
def _ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """statsmodels OLS coefficients (collinear columns get 0, as Stata drops them)."""
    import statsmodels.api as sm
    keep = independent_columns(X, protect=[0])
    beta = np.zeros(X.shape[1])
    if keep:
        beta[keep] = sm.OLS(y, X[:, keep]).fit().params
    return beta, keep


def _logit_fit(Z: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """statsmodels Logit (Newton) fitted probabilities of the propensity model."""
    import statsmodels.api as sm
    import warnings
    keep = independent_columns(Z, protect=[0])
    Zk = Z[:, keep]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.Logit(D, Zk).fit(method="newton", disp=0, maxiter=200, tol=1e-12)
    if not res.mle_retvals.get("converged", True):
        raise ValueError("성향점수 logit 수렴 실패")
    ps = np.asarray(res.predict(Zk), float)
    return ps, keep


def att_2x2(dy: np.ndarray, D: np.ndarray, Xo: np.ndarray, Zp: np.ndarray, est: str) -> Tuple[float, np.ndarray]:
    """2x2 panel DiD ATT and its influence function (Sant'Anna & Zhao 2020; DRDID reg_did_panel, std_ipw_did_panel,
    drdid_panel). Xo / Zp: outcome and propensity covariates with the intercept in column 0."""
    n = len(dy)
    if est == "ra":
        beta, kx = _ols_fit(Xo[D == 0], dy[D == 0])
        X = Xo[:, kx]
        out = Xo @ beta
        w_t = D
        w_c = D
        att_t, att_c = w_t * dy, w_c * out
        eta_t, eta_c = att_t.mean() / w_t.mean(), att_c.mean() / w_c.mean()
        wo = 1 - D
        XpX = (wo[:, None] * X).T @ X / n
        asy_ols = ((wo * (dy - out))[:, None] * X) @ np.linalg.inv(XpX)
        inf_t = (att_t - w_t * eta_t) / w_t.mean()
        M1 = (w_c[:, None] * X).mean(0)
        inf_c = (att_c - w_c * eta_c + asy_ols @ M1) / w_c.mean()
        return float(eta_t - eta_c), inf_t - inf_c
    ps, kz = _logit_fit(Zp, D)
    Z = Zp[:, kz]
    ps = np.minimum(ps, 1 - 1e-6)
    w_t = D
    w_c = ps * (1 - D) / (1 - ps)
    Hs = np.linalg.inv((Z * (ps * (1 - ps))[:, None]).T @ Z) * n          # vcov(logit) * n
    asy_ps = ((D - ps)[:, None] * Z) @ Hs
    if est == "ipw":
        att_t, att_c = w_t * dy, w_c * dy
        eta_t, eta_c = att_t.mean() / w_t.mean(), att_c.mean() / w_c.mean()
        inf_t = (att_t - w_t * eta_t) / w_t.mean()
        M2 = ((w_c * (dy - eta_c))[:, None] * Z).mean(0)
        inf_c = (att_c - w_c * eta_c + asy_ps @ M2) / w_c.mean()
        return float(eta_t - eta_c), inf_t - inf_c
    beta, kx = _ols_fit(Xo[D == 0], dy[D == 0])
    X = Xo[:, kx]
    out = Xo @ beta
    r = dy - out
    att_t, att_c = w_t * r, w_c * r
    eta_t, eta_c = att_t.mean() / w_t.mean(), att_c.mean() / w_c.mean()
    wo = 1 - D
    XpX = (wo[:, None] * X).T @ X / n
    asy_ols = ((wo * r)[:, None] * X) @ np.linalg.inv(XpX)
    M1 = (w_t[:, None] * X).mean(0)
    inf_t = (att_t - w_t * eta_t - asy_ols @ M1) / w_t.mean()
    M2 = ((w_c * (r - eta_c))[:, None] * Z).mean(0)
    M3 = (w_c[:, None] * X).mean(0)
    inf_c = (att_c - w_c * eta_c + asy_ps @ M2 - asy_ols @ M3) / w_c.mean()
    return float(eta_t - eta_c), inf_t - inf_c


class ATTgtResult:
    """ATT(g,t) estimates, unit-level influence functions (n units x K) and cluster structure."""

    def __init__(self):
        self.cells: List[Dict[str, Any]] = []     # {"g", "t", "base", "att", "post", "omitted", "n1"}
        self.psi = np.zeros((0, 0))
        self.n = 0
        self.cl_unit: np.ndarray = np.zeros(0, dtype=np.int64)
        self.n_cl = 0
        self.notes: List[str] = []

    def vcov(self, Psi: np.ndarray) -> np.ndarray:
        Sc = _cluster_sum(Psi, self.cl_unit, self.n_cl)
        return Sc.T @ Sc / self.n ** 2


def callaway_santanna(D_: DiDData, est: str, control: str = "never", basetime: str = "adaptive") -> ATTgtResult:
    """ATT(g,t) for every cohort g and period t (Callaway & Sant'Anna 2021; Stata xthdidregress ra/ipw/aipw).

    2x2 sample: units in cohort g and control units (never treated, or not yet treated by period max(t, g) — the rule
    that reproduces Stata 19.5 for pre-treatment periods), observed at both t and the base period; covariates taken at
    the earlier of the two periods (the base period except for common-base pre-periods); influence functions scaled
    to all n units."""
    n = D_.n
    tl = list(D_.tlist)
    T = len(tl)
    # wide arrays (units x periods)
    pos = {tp: k for k, tp in enumerate(tl)}
    kcol = np.array([pos[int(t)] for t in D_.tpos])
    Y = np.full((n, T), np.nan)
    Y[D_.g, kcol] = D_.y
    obs = ~np.isnan(Y)
    Xw = np.full((n, T, len(D_.covs)), np.nan)
    Zw = np.full((n, T, len(D_.tcovs)), np.nan)
    if D_.covs:
        Xw[D_.g, kcol, :] = D_.P.df[D_.covs].to_numpy(float)
    if D_.tcovs:
        Zw[D_.g, kcol, :] = D_.P.df[D_.tcovs].to_numpy(float)
    coh = D_.cohort_unit.copy()                   # tpos of the first treated period, -1 never
    res = ATTgtResult()
    res.n = n
    first = tl[0]
    always = np.unique(D_.unit_grp[(coh == first)])
    if (coh == first).any():
        res.notes.append(f"첫 시점({D_.label(first)})부터 처치된 집단 {len(always)}개 제외 (처치 전 기준 시점 없음)")
        coh = np.where(coh == first, -2, coh)     # -2: excluded from cohorts and controls
    cohorts = sorted(int(c) for c in np.unique(coh) if c >= 0)
    never = coh == -1
    if not cohorts:
        raise ValueError("이중차분: 처치 집단 없음 (처치 변수가 1인 집단 없음)")
    t_last = None
    if control == "never" and not never.any():
        raise ValueError("이중차분: 처치되지 않은 집단(never treated) 없음 — 모든 집단이 결국 처치됨. "
                         "controlGroup='notyet' 사용 (Stata: consider using controlgroup(notyet))")
    est_cohorts = cohorts
    if control == "notyet" and not never.any():
        t_last = cohorts[-1]
        est_cohorts = cohorts[:-1]
        res.notes.append(f"처치되지 않은 집단 없음 — 마지막 코호트({D_.label(t_last)})를 통제집단으로 사용, "
                         f"{D_.label(t_last)} 이후 시점은 추정 불가 (Stata·R did와 동일)")
    # cluster of each unit
    if D_.group == D_.entity:
        res.cl_unit, res.n_cl = np.arange(n), n
    else:
        res.cl_unit, res.n_cl = D_.unit_grp.copy(), D_.n_grp
    cols = []
    def base_index(g: int) -> int:
        """Last period before g at which units of cohort g are observed (g-1 in a balanced panel; with gaps the latest
        earlier observed period, as Stata 19.5 does for units whose period g-1 is missing)."""
        kg = pos[g]
        seen = np.where(obs[coh == g][:, :kg].any(0))[0]
        return int(seen[-1]) if len(seen) else kg - 1

    for g in cohorts:
        kg = pos[g]
        kg1 = base_index(g)
        for kt in range(0 if basetime == "common" else 1, T):
            t = tl[kt]
            if basetime == "common":
                kb = kg1
                if kt == kb:
                    continue
            else:
                kb = kg1 if t >= g else kt - 1
            b = tl[kb]
            cell = {"g": g, "t": t, "base": b, "post": t >= g, "omitted": False, "att": 0.0, "n1": 0}
            if g not in est_cohorts or (t_last is not None and t >= t_last):
                cell["omitted"] = True
                res.cells.append(cell)
                cols.append(np.zeros(n))
                continue
            paired = obs[:, kt] & obs[:, kb]
            treated = (coh == g) & paired
            if control == "never":
                ctrl = never & paired
            else:
                ctrl = ((coh == -1) | (coh > max(t, g))) & paired      # not yet treated by max(t, g) (Stata)
            sub = treated | ctrl
            if treated.sum() == 0 or ctrl.sum() == 0:
                cell["omitted"] = True
                res.cells.append(cell)
                cols.append(np.zeros(n))
                continue
            idx = np.where(sub)[0]
            Dv = treated[idx].astype(float)
            dy = Y[idx, kt] - Y[idx, kb]
            kx = min(kt, kb)
            Xo = np.column_stack([np.ones(len(idx)), Xw[idx, kx, :]]) if D_.covs else np.ones((len(idx), 1))
            Zp = np.column_stack([np.ones(len(idx)), Zw[idx, kx, :]]) if D_.tcovs else np.ones((len(idx), 1))
            if np.isnan(Xo).any() or np.isnan(Zp).any():
                raise ValueError("이중차분: 기준 시점 공변량 결측")
            att, inf = att_2x2(dy, Dv, Xo, Zp, est)
            psi = np.zeros(n)
            psi[idx] = inf * (n / len(idx))
            cell.update({"att": att, "n1": int(len(idx)), "ind": treated.astype(float)})
            res.cells.append(cell)
            cols.append(psi)
    res.psi = np.column_stack(cols) if cols else np.zeros((n, 0))
    # aggregation weight indicators: cohort-g units observed at g-1 (Stata weights(timecohort), panel data)
    res.cohorts = est_cohorts
    return res


def aggregate_cs(R: ATTgtResult, kind: str) -> List[Dict[str, Any]]:
    """estat aggregation after ra / ipw / aipw: weighted ATT(g,t) with the influence function of the estimated
    weights (Callaway & Sant'Anna 2021 section 4.2; R did compute.aggte). Weights of Stata weights(timecohort): the
    share of cohort-g units in the 2x2 sample of each ATT(g,t) — the cohort share in a balanced panel."""
    att = np.array([c["att"] for c in R.cells])
    gs = np.array([c["g"] for c in R.cells])
    ts = np.array([c["t"] for c in R.cells])
    ok = np.array([not c["omitted"] for c in R.cells])
    out = []

    def combine(keep: np.ndarray):
        keep = np.where(keep & ok)[0]
        if len(keep) == 0:
            return None
        ind = np.column_stack([R.cells[k]["ind"] for k in keep])
        pw = ind.mean(0)
        if pw.sum() <= 0:
            return None
        w = pw / pw.sum()
        psi = R.psi[:, keep] @ w
        # influence of the estimated weights (R did wif)
        if1 = (ind - pw[None, :]) / pw.sum()
        if2 = (ind - pw[None, :]).sum(1)[:, None] * (pw / pw.sum() ** 2)[None, :]
        psi = psi + (if1 - if2) @ att[keep]
        V = R.vcov(psi[:, None])
        return float(w @ att[keep]), float(np.sqrt(V[0, 0]))

    if kind == "overall":
        r = combine(ts >= gs)
        if r:
            out.append({"key": "ATET", "att": r[0], "se": r[1]})
    elif kind == "cohort":
        for g in sorted(set(gs[ok])):
            r = combine((gs == g) & (ts >= gs))
            if r:
                out.append({"key": g, "att": r[0], "se": r[1]})
    elif kind == "time":
        for t in sorted(set(ts[ok & (ts >= gs)])):
            r = combine((ts == t) & (ts >= gs))
            if r:
                out.append({"key": t, "att": r[0], "se": r[1]})
    else:
        ev = ts - gs
        for e in sorted(set(ev[ok])):
            r = combine(ev == e)
            if r:
                out.append({"key": int(e), "att": r[0], "se": r[1]})
    return out


# ----------------------------------------------------------------------------- Wooldridge (2025) extended TWFE
class TWFEResult:
    def __init__(self):
        self.cells: List[Dict[str, Any]] = []
        self.notes: List[str] = []


def wooldridge_twfe(D_: DiDData, control: str = "never", ptrends: bool = False) -> TWFEResult:
    """xthdidregress twfe: extended TWFE regression with cohort x time treatment cells and covariate interactions
    (over(cohort): effects evaluated at the cohort covariate means), estimated by statsmodels OLS on the within-transformed
    data (the xtreg, fe estimates, equal to the two-way Mundlak POLS of Wooldridge 2025). Standard errors as margins vce(unconditional): cluster scores of the regression plus the
    scores of the cohort covariate means, small-sample factor (N-1)/(N-K)·G/(G-1) with K = rank of the Mundlak POLS
    design (regressors and their panel means)."""
    import statsmodels.api as sm
    R = TWFEResult()
    coh = D_.cohort_row.copy()
    first = int(D_.tlist[0])
    keep_rows = coh != first
    if (~keep_rows).any():
        R.notes.append(f"첫 시점({D_.label(first)})부터 처치된 집단 {len(np.unique(D_.grp[~keep_rows]))}개 제외")
    rows = np.where(keep_rows)[0]
    coh = coh[rows]
    tpos = D_.tpos[rows]
    y = D_.y[rows]
    gcodes, gl = pd.factorize(D_.g[rows], sort=True)
    n_u = len(gl)
    N = len(rows)
    covs = D_.covs
    Xc = D_.P.df[covs].to_numpy(float)[rows] if covs else np.zeros((N, 0))
    cohorts = sorted(int(c) for c in np.unique(coh) if c >= 0)
    never = (coh == -1).any()
    if not cohorts:
        raise ValueError("이중차분: 처치 집단 없음")
    if control == "never" and not never:
        raise ValueError("이중차분: 처치되지 않은 집단(never treated) 없음 — 모든 집단이 결국 처치됨. "
                         "controlGroup='notyet' 사용 (Stata: consider using controlgroup(notyet))")
    g_last = None
    mod = cohorts
    if not never:
        g_last = cohorts[-1]
        mod = cohorts[:-1]
        R.notes.append(f"마지막 코호트({D_.label(g_last)})를 통제집단으로 사용 (Wooldridge 2025; Stata controlgroup(notyet)); "
                       f"{D_.label(g_last)} 이후 시점의 ATET는 보고하지 않음")
    elif control == "notyet":
        R.notes.append("처치되지 않은 집단이 있어 통제집단은 never treated (Stata xthdidregress twfe와 동일)")
    R.control_used = "notyet" if g_last is not None else "never"
    tl = [int(t) for t in np.unique(tpos)]
    # regressors
    cells = [(g, s) for g in mod for s in tl if s >= g]
    pre_cells = [(g, s) for g in mod for s in tl[1:] if s < g]

    def design(with_pre: bool):
        cols: List[np.ndarray] = []
        names: List[Tuple] = []
        for s in tl[1:]:
            cols.append((tpos == s).astype(float)); names.append(("f", s))
        for g, s in cells:
            cols.append(((coh == g) & (tpos == s)).astype(float)); names.append(("D", g, s))
        if with_pre:
            for g, s in pre_cells:
                cols.append(((coh == g) & (tpos == s)).astype(float)); names.append(("W", g, s))
        for j in range(len(covs)):
            x = Xc[:, j]
            for g in mod:
                cols.append((coh == g) * x); names.append(("Gx", g, j))
            for s in tl[1:]:
                cols.append((tpos == s) * x); names.append(("fx", s, j))
        for j in range(len(covs)):
            x = Xc[:, j]
            for g, s in cells:
                cols.append(((coh == g) & (tpos == s)) * x); names.append(("Dx", g, s, j))
            if with_pre:
                for g, s in pre_cells:
                    cols.append(((coh == g) & (tpos == s)) * x); names.append(("Wx", g, s, j))
        for j in range(len(covs)):
            cols.append(Xc[:, j]); names.append(("x", j))
        return np.column_stack(cols), names

    if D_.group == D_.entity:
        cl, n_cl = gcodes, n_u
    else:
        cl = pd.factorize(D_.grp[rows])[0]
        n_cl = int(cl.max()) + 1
    yt = _within_matrix(y, gcodes, n_u)

    def fit(with_pre: bool):
        X, names = design(with_pre)
        Xt = _within_matrix(X, gcodes, n_u)
        xn = np.linalg.norm(X, axis=0)
        nz = [k for k in range(X.shape[1])
              if np.linalg.norm(Xt[:, k]) > 1e-9 * max(xn[k], 1e-300)]          # absorbed by the panel effects
        prot = [k for k in nz if names[k][0] in ("D", "W")]
        sub = _independent_qr(Xt[:, nz], protect=[nz.index(k) for k in prot])
        kept = sorted(nz[j] for j in sub)
        Xk, nk = Xt[:, kept], [names[k] for k in kept]
        b = np.asarray(sm.OLS(yt, Xk).fit(method="qr").params, float)          # within (xtreg, fe) estimates
        e = yt - Xk @ b
        # Mundlak POLS rank: rank[W, Wbar] = rank(within part) + rank(panel means of [1, cohort dummies, X]),
        # the two parts being orthogonal
        Gd = [(coh == g).astype(float) for g in mod]
        Wm = np.column_stack([np.ones(N)] + Gd + [X])
        sums = np.zeros((n_u, Wm.shape[1]))
        np.add.at(sums, gcodes, Wm)
        means = sums / np.bincount(gcodes, minlength=n_u)[:, None]
        Kp = len(kept) + len(_independent_qr(means, protect=[0]))
        fac = (N - 1.0) / (N - Kp) * n_cl / (n_cl - 1.0)
        XtXi = np.linalg.inv(Xk.T @ Xk)
        Sx = _cluster_sum(Xk * e[:, None], cl, n_cl)
        return b, nk, Kp, fac, XtXi, Sx

    b, nk, Kp, fac, XtXi, Sx = fit(False)
    pos = {nm: k for k, nm in enumerate(nk)}
    R.N, R.K, R.G, R.df_r, R.n_units = N, Kp, n_cl, n_cl - 1.0, n_u
    R.b, R.names, R.fac = b, nk, fac
    d_obs = D_.d[rows]

    def coef(nm):
        k = pos.get(nm)
        return 0.0 if k is None else float(b[k])

    def tau_rows(sel: np.ndarray):
        """Per-row treatment effect tau_gs + x'tau_x,gs and the derivative rows (len(sel) x k)."""
        Ld = np.zeros((len(sel), len(nk)))
        tau = np.zeros(len(sel))
        for r_i, r in enumerate(sel):
            g, s = int(coh[r]), int(tpos[r])
            k = pos.get(("D", g, s))
            if k is not None:
                Ld[r_i, k] = 1.0
                tau[r_i] += b[k]
            for j in range(len(covs)):
                kj = pos.get(("Dx", g, s, j))
                if kj is not None:
                    Ld[r_i, kj] = Xc[r, j]
                    tau[r_i] += b[kj] * Xc[r, j]
        return tau, Ld

    def se_of(L: np.ndarray, h: np.ndarray) -> float:
        a = Sx @ (XtXi @ L)
        return float(np.sqrt(fac * np.sum((a + h) ** 2)))

    def at_cohort_means(kind: str, g: int, s: int, b_, pos_):
        """Effect of cell (g,s) at the cohort covariate means (margins dydx(d) at(t=s) over(cohort)): L vector and
        the cluster scores of the cohort means (first-stage regressions of the covariates on cohort indicators)."""
        L = np.zeros(len(b_))
        k = pos_.get((kind, g, s))
        if k is not None:
            L[k] = 1.0
        inG = coh == g
        h = np.zeros(n_cl)
        if covs:
            xbar = Xc[inG].mean(0)
            tx = np.zeros(len(covs))
            for j in range(len(covs)):
                kj = pos_.get((kind + "x", g, s, j))
                if kj is not None:
                    L[kj] = xbar[j]
                    tx[j] = b_[kj]
            np.add.at(h, cl[inG], ((Xc[inG] - xbar) @ tx) / inG.sum())
        return L, h, k is None

    # ATT(g,s)
    R.unident = []
    for g, s in cells:
        if g_last is not None and s >= g_last:
            continue
        L, h, omitted = at_cohort_means("D", g, s, b, pos)
        if not omitted and any(("Dx", g, s, j) not in pos for j in range(len(covs))):
            omitted = True                  # covariate interactions of the cell not identified (Stata omits the cell)
            R.unident.append((g, s))
        R.cells.append({"g": g, "t": s, "att": float(L @ b) if not omitted else 0.0,
                        "se": se_of(L, h) if not omitted else None, "omitted": omitted, "post": True})
    # subpopulation of treated observations for estat aggregation (margins, subpop(if treat==1))
    S_rows = np.array([r for r in range(N) if coh[r] in mod and tpos[r] >= coh[r] and d_obs[r] == 1
                       and (g_last is None or tpos[r] < g_last)], dtype=np.int64)
    tau_S, Ld_S = tau_rows(S_rows)
    R.S_rows, R.tau_S, R.Ld_S = S_rows, tau_S, Ld_S
    R.coh_rows, R.t_rows, R.cl_rows = coh, tpos, cl

    def margin(mask: np.ndarray) -> Tuple[float, float]:
        sel = np.where(mask)[0]
        ame = float(tau_S[sel].mean())
        L = Ld_S[sel].mean(0)
        h = np.zeros(n_cl)
        np.add.at(h, cl[S_rows[sel]], (tau_S[sel] - ame) / len(sel))
        return ame, se_of(L, h)

    R.margin = margin
    R.ptrends = None
    if ptrends and pre_cells:
        # augmented model: pre-treatment cohort x time cells (and their covariate interactions); joint test of the
        # pre-treatment effects at the cohort covariate means ([CAUSAL] hdidregress postestimation M&F)
        b2, nk2, _, fac2, XtXi2, Sx2 = fit(True)
        p2 = {nm: k for k, nm in enumerate(nk2)}
        est, scores = [], []
        for g, s in pre_cells:
            L, h, om = at_cohort_means("W", g, s, b2, p2)
            if om or any(("Wx", g, s, j) not in p2 for j in range(len(covs))):
                continue                    # empty cell or covariate interactions not identified (omitted)
            est.append(float(L @ b2))
            scores.append(Sx2 @ (XtXi2 @ L) + h)
        if est:
            Sc = np.column_stack(scores)
            R.ptrends = _wald(np.array(est), fac2 * Sc.T @ Sc, R.df_r)
    return R


def aggregate_twfe(R: TWFEResult, kind: str) -> List[Dict[str, Any]]:
    """estat aggregation after the extended TWFE: margins over the treated observations, vce(unconditional)."""
    coh = R.coh_rows[R.S_rows]
    tt = R.t_rows[R.S_rows]
    out = []
    if len(R.S_rows) == 0:
        return out
    if kind == "overall":
        a, s = R.margin(np.ones(len(R.S_rows), bool))
        out.append({"key": "ATET", "att": a, "se": s})
    elif kind == "cohort":
        for g in sorted(set(coh)):
            a, s = R.margin(coh == g)
            out.append({"key": int(g), "att": a, "se": s})
    elif kind == "time":
        for t in sorted(set(tt)):
            a, s = R.margin(tt == t)
            out.append({"key": int(t), "att": a, "se": s})
    else:
        ev = tt - coh
        for e in sorted(set(ev)):
            a, s = R.margin(ev == e)
            out.append({"key": int(e), "att": a, "se": s})
    return out


# ----------------------------------------------------------------------------- classic TWFE pieces (xtdidregress)
def _rel_dummies(rel: np.ndarray, treated: np.ndarray, lo: int, hi: int, ref: int, bin_ends: bool
                 ) -> Tuple[List[np.ndarray], List[int]]:
    """Relative-time indicators k = lo..hi (k != ref) for treated rows; endpoints bin rel <= lo and rel >= hi."""
    cols, ks = [], []
    for k in range(lo, hi + 1):
        if k == ref:
            continue
        if bin_ends and k == lo:
            c = treated & (rel <= lo)
        elif bin_ends and k == hi:
            c = treated & (rel >= hi)
        else:
            c = treated & (rel == k)
        cols.append(c.astype(float))
        ks.append(k)
    return cols, ks


def _lead_lag_name(k: int) -> str:
    return f"_lead{-k}" if k < 0 else f"_lag{k}"


class PanelDiDAnalysis(BaseAnalysis):
    """Difference in differences for panel data (Stata xtdidregress / xthdidregress / event-study regressions)."""

    def __init__(self, data: pd.DataFrame, **o):
        self.o = o
        m = o["method"]
        title = {"twfe": "패널 이중차분 — 이원 고정효과 ATET (xtdidregress)",
                 "event": "패널 사건연구 — 선행·후행 처치 더미 (event study)",
                 "hdid": f"이질적 처치효과 이중차분 — {EST_LABEL[o['estimator']]} (xthdidregress {o['estimator']})"}[m]
        super().__init__(name="panelDiD", title=title, dataset=data, options=dict(o))
        self.syntax_python = python_call(o)
        self.syntax_r = "\n".join(r_code("panelDiD", o) or [])
        self.references = list(self.references) + [
            "Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods. "
            "Journal of Econometrics, 225, 200–230.",
            "Sant'Anna, P. H. C., & Zhao, J. (2020). Doubly robust difference-in-differences estimators. "
            "Journal of Econometrics, 219, 101–122.",
            "Wooldridge, J. M. (2025). Two-way fixed effects, the two-way Mundlak regression, and "
            "difference-in-differences estimators. Empirical Economics, 69, 2545–2587.",
            "StataCorp (2025). Stata 19 Causal Inference and Treatment-Effects Estimation Reference Manual "
            "(didregress, xthdidregress, hdidregress postestimation).",
        ]
        self.run()

    # ------------------------------------------------------------------ run
    def run(self):
        o = self.o
        D_ = DiDData(self.dataset, o["entity"], o["time"], o["dep"], o["covs"],
                     o["tcovs"] if (o["method"] == "hdid" and o["estimator"] in ("ipw", "aipw")) else [],
                     o["treat"], o["group"] or None, o["cluster"] or None, o["delta"])
        self.D = D_
        if o["method"] in ("event", "hdid") and not D_.absorbing:
            raise ValueError("이중차분: 처치가 한 번 시작된 뒤 0으로 돌아가는 집단이 있음 — 사건연구·이질적 처치효과 "
                             "추정은 처치가 지속되는 자료 필요 (Stata xthdidregress 가정; 처치 전환 자료는 xtswitchdid)")
        self.model_stats = {"N": int(D_.N), "Groups": int(D_.n), "Periods": int(len(D_.tlist))}
        if o["method"] == "twfe":
            self._run_twfe(D_)
        elif o["method"] == "event":
            self._run_event(D_)
        else:
            self._run_hdid(D_)
        if o.get("ignored") and self.tables:
            self.tables[0].add_note("이 추정량에 해당하지 않아 적용하지 않은 입력: " + "; ".join(o["ignored"]))

    def _safe(self, fn, label: str):
        try:
            fn()
        except Exception as e:      # a failing diagnostic must not hide the estimates
            t = ResultsTable(name=f"diag_error_{len(self.tables)}", title=f"{label} — 계산 불가")
            t.add_column("msg", "사유", col_type="text")
            t.add_row({"msg": str(e)})
            self.add_table(t)

    # ------------------------------------------------------------------ shared tables
    def _info_table(self, D_: DiDData):
        t = ResultsTable(name="treat_info", title="처치·시점 정보 (Treatment and time information)")
        t.add_column("item", "항목", col_type="text")
        t.add_column("ctrl", "통제", col_type="text")
        t.add_column("trt", "처치", col_type="text")
        treated = D_.cohort_grp >= 0
        first = D_.cohort_grp[treated]
        t0 = D_.label(D_.tlist[0])
        t.add_row({"item": f"집단 수 ({D_.group})", "ctrl": str(int((~treated).sum())), "trt": str(int(treated.sum()))})
        if treated.any():
            t.add_row({"item": "첫 처치 시점 최소", "ctrl": t0, "trt": D_.label(first.min())})
            t.add_row({"item": "첫 처치 시점 최대", "ctrl": t0, "trt": D_.label(first.max())})
        t.add_note(f"시점 변수 {D_.time}; 통제: {D_.treat} = 0, 처치: {D_.treat} = 1")
        if D_.P.n_dropped:
            t.add_note(f"결측으로 제외된 행: {D_.P.n_dropped}")
        if D_.P.time_note:
            t.add_note(D_.P.time_note)
        self.add_table(t)

    def _coef_table(self, name: str, title: str, rows: List[Dict[str, Any]], stat: str,
                    first: Sequence[Tuple[str, str, str]], notes: Sequence[str] = (), label="var"):
        t = ResultsTable(name=name, title=title)
        _coef_columns(t, stat, first)
        for r in rows:
            t.add_row(r)
        for nt in notes:
            t.add_note(nt)
        t.mark_coefficients(estimate="b", se="se", p="p", stat="stat", ci=("cil", "ciu"), label=label)
        self.add_table(t)
        return t

    def _vce_label(self, cl_name: str, G: int) -> str:
        return f"군집-강건 표준오차 ({cl_name} 군집 {G}개)"

    # ------------------------------------------------------------------ twfe (xtdidregress)
    def _twfe_design(self, D_: DiDData):
        Td, tnames = D_.time_dummies()
        X = np.column_stack([D_.d] + ([D_.P.df[D_.covs].to_numpy(float)] if D_.covs else []) + ([Td] if Td.shape[1] else []))
        names = [D_.treat] + list(D_.covs) + tnames
        return X, names

    def _fit(self, D_: DiDData, X: np.ndarray, names: List[str]) -> Fit:
        cl_arr, cl_name, G = D_.cluster_codes(self.o["cluster"] or None)
        fit = fit_within(D_.P, X, names, "clustered", cl_arr)
        fit.cl_name, fit.G = cl_name, G
        return fit

    def _run_twfe(self, D_: DiDData):
        o = self.o
        self._info_table(D_)
        X, names = self._twfe_design(D_)
        fit = self._fit(D_, X, names)
        if D_.treat not in fit.names:
            raise ValueError(f"이중차분: 처치 변수 '{D_.treat}'가 패널·시점 효과와 공선 — ATET 추정 불가")
        self.fit = fit
        k = fit.names.index(D_.treat)
        lev = o["level"]
        r = _row(fit.b[k], fit.se[k], fit.df_r, lev)
        r["var"] = f"{D_.treat} (1 vs 0)"
        notes = [f"표준오차: {self._vce_label(fit.cl_name, fit.G)}, 소표본 보정 (N-1)/(N-K)·G/(G-1), t 자유도 {int(fit.df_r)} "
                 "(Stata xtdidregress = xtreg, fe vce(cluster))",
                 "ATET: 패널 효과·시점 효과" + (" ·공변량" if D_.covs else "") + " 조정 (Stata note: ATET estimate adjusted for "
                 + ("covariates, " if D_.covs else "") + "panel effects, and time effects)",
                 "라이브러리: linearmodels PanelOLS (entity effects) + 시점 더미"]
        coh = np.unique(D_.cohort_grp[D_.cohort_grp >= 0])
        self.staggered = len(coh) > 1
        if self.staggered:
            notes.append("처치 시점이 집단마다 다름 (Treatment occurs at different times) — 이질적 처치효과는 method='hdid' 검토")
        if fit.omitted:
            notes.append("공선성으로 제외 (omitted): " + ", ".join(fit.omitted))
        self._coef_table("atet", f"ATET — 평균처치효과 (처치집단, 수준 {lev:g}%)", [r],
                         "t", [("var", "처치", "text")], notes)
        self.model_stats.update({"Clusters": int(fit.G), "ATET": float(fit.b[k])})
        if o["showControls"]:
            rows = []
            for j, nm in enumerate(fit.names):
                if nm in (D_.treat,):
                    continue
                rr = _row(fit.b[j], fit.se[j], fit.df_r, lev)
                rr["var"] = "_cons" if nm == "Intercept" else nm
                rows.append(rr)
            self._coef_table("controls", "통제변수·시점효과 계수 (Stata aequations)", rows, "t", [("var", "변수", "text")])
        if o["wildBootstrap"]:
            self._safe(lambda: self._wild_table(D_, fit, X, names), "와일드 군집 부트스트랩")
        if o["trendPlots"]:
            self._safe(lambda: self._trend_table(D_), "estat trendplots")
        if o["ptrends"]:
            self._safe(lambda: self._ptrends_twfe(D_, X, names), "estat ptrends")
        if o["granger"]:
            self._safe(lambda: self._granger(D_, X, names), "estat granger")
        if o["grangerPlot"]:
            self._safe(lambda: self._grangerplot(D_), "estat grangerplot")
        st = r
        self.apa_narrative = (f"이원 고정효과 이중차분 ATET = {st['b']:.3f} (SE = {st['se']:.3f}), t({int(fit.df_r)}) = "
                              f"{st['stat']:.2f}, p {_pstr(st['p'])}, {lev:g}% CI [{st['cil']:.3f}, {st['ciu']:.3f}] "
                              f"(관측치 {D_.N}개, 패널 {D_.n}개, 군집 {fit.G}개).")

    def _single_timing(self, D_: DiDData, what: str) -> int:
        coh = np.unique(D_.cohort_grp[D_.cohort_grp >= 0])
        if len(coh) == 0:
            raise ValueError("처치 집단 없음")
        # first observed treated period of every treated panel (Stata checks the timing panel by panel, so a panel
        # whose adoption period is missing makes the timing vary)
        first_unit = pd.Series(D_.tpos[D_.d == 1]).groupby(D_.g[D_.d == 1]).min()
        if len(coh) > 1 or first_unit.nunique() > 1:
            raise ValueError(f"처치 시점이 집단마다 다름 — Stata도 {what} 허용하지 않음 (treatment assignment times vary). "
                             "사건연구는 method='event', 이질적 처치효과는 method='hdid'")
        J = int(coh[0])
        if J == int(D_.tlist[0]):
            raise ValueError("첫 시점부터 처치됨 — 처치 전 시점 없음")
        return J

    def _wild_table(self, D_: DiDData, fit: Fit, X: np.ndarray, names: List[str]):
        o = self.o
        kept = [names.index(nm) for nm in fit.names if nm != "Intercept"]
        k_t = names.index(D_.treat)
        Zcols = [j for j in kept if j != k_t]
        Zt = _within_matrix(X[:, Zcols], D_.g, D_.n) if Zcols else np.zeros((D_.N, 0))
        dt = _within_matrix(D_.d, D_.g, D_.n)
        yt = _within_matrix(D_.y, D_.g, D_.n)
        if Zt.shape[1]:
            coef = np.linalg.lstsq(Zt, np.column_stack([yt, dt]), rcond=None)[0]
            yr, dr = yt - Zt @ coef[:, 0], dt - Zt @ coef[:, 1]
        else:
            yr, dr = yt, dt
        cl_arr, cl_name, G = D_.cluster_codes(o["cluster"] or None)
        cl = D_.g if cl_arr is None else cl_arr
        wb = wild_bootstrap(yr, dr, Zt, cl, o["level"], int(o["wbReps"]), o["wbWeight"], o["wbSeed"])
        t = ResultsTable(name="wildboot", title="와일드 군집 부트스트랩 추론 (DID with wild-cluster bootstrap inference)")
        t.add_column("var", "처치", col_type="text")
        t.add_column("b", "추정치", col_type="number")
        t.add_column("t", "t", col_type="number")
        t.add_column("p", "p", col_type="number", fmt="pvalue")
        t.add_column("lb", "CI 하한", col_type="number")
        t.add_column("ub", "CI 상한", col_type="number")
        kt = fit.names.index(D_.treat)
        t.add_row({"var": f"{D_.treat} (1 vs 0)", "b": wb["b"], "t": float(fit.b[kt] / fit.se[kt]), "p": wb["p"],
                   "lb": wb["lb"], "ub": wb["ub"]})
        self.wild = wb
        wt = {"rademacher": "Rademacher", "mammen": "Mammen", "webb": "Webb", "normal": "정규", "gamma": "감마"}[wb["weight"]]
        t.add_note(f"제약(ATET = 0) 와일드 군집 부트스트랩: 오차 가중 {wt}, 반복 {wb['reps']}회, 군집 {wb['G']}개 ({cl_name}), "
                   f"난수 시드 {o['wbSeed']}")
        if wb["reps_note"]:
            t.add_note(f"반복 수를 {wb['reps']}회로 조정 — 등꼬리 {o['level']:g}% 구간에 맞춤 (Stata note: setting repetitions)")
        t.add_note("p값: 등꼬리 2·min(P_l, P_u), 신뢰구간: 귀무값 이동 후 이분법 탐색 — Stata [CAUSAL] didregress Methods and "
                   "formulas; 군집 수준 계산식 Roodman, MacKinnon, Nielsen & Webb (2019)")
        t.add_note("부트스트랩 결과는 난수에 따라 달라짐 (Stata와 난수 생성기가 달라 값이 같지 않음)")
        self.add_table(t)

    def _trend_table(self, D_: DiDData):
        J = self._single_timing(D_, "estat trendplots")
        df = pd.DataFrame({"t": D_.tpos, "w": D_.w_row, "y": D_.y})
        m = df.groupby(["t", "w"])["y"].mean().unstack()
        t = ResultsTable(name="trend_means", title=f"관측 평균 추이 (estat trendplots, omeans) — {D_.dep}")
        t.add_column("time", D_.time, col_type="text")
        t.add_column("ctrl", "통제", col_type="number")
        t.add_column("trt", "처치", col_type="number")
        for tp in D_.tlist:
            t.add_row({"time": D_.label(tp), "ctrl": m.loc[tp].get(0.0, np.nan), "trt": m.loc[tp].get(1.0, np.nan)})
        k = list(D_.tlist).index(J)
        t.add_note(f"처치집단: 한 번이라도 처치되는 집단 ({D_.group}); 세로선: 처치 직전 시점 {D_.label(D_.tlist[k - 1])}")
        self.add_table(t)
        try:
            self.add_plot("평행추세 진단 — 관측 평균 (estat trendplots)",
                          _plot_trends(D_, m, D_.tlist[k - 1]))
        except Exception:
            pass

    def _ptrends_twfe(self, D_: DiDData, X: np.ndarray, names: List[str]):
        J = self._single_timing(D_, "estat ptrends")
        w = D_.w_row
        tt = D_.tpos.astype(float)
        a0 = w * (D_.tpos < J) * tt
        a1 = w * (D_.tpos >= J) * tt
        fit = self._fit(D_, np.column_stack([X, a0, a1]), names + ["_pre_trend", "_post_trend"])
        if "_pre_trend" not in fit.names:
            raise ValueError("처치 전 추세 항이 공선으로 제외됨 (처치 전 시점 부족)")
        k = fit.names.index("_pre_trend")
        F, q, df2, p = fit.wald([k])
        t = ResultsTable(name="ptrends", title="평행추세 검정 (estat ptrends) — H0: 처치 전 선형 추세가 평행")
        for c, ttl, typ in (("F", "F", "number"), ("df1", "df1", "integer"), ("df2", "df2", "integer"),
                            ("p", "p", "number")):
            t.add_column(c, ttl, col_type=typ, fmt="pvalue" if c == "p" else "")
        t.add_row({"F": F, "df1": q, "df2": int(df2), "p": p})
        t.add_note("증강 모형: w·1(처치 전)·t, w·1(처치 후)·t 추가 후 처치 전 기울기 차이 Wald 검정 — Stata [CAUSAL] didregress "
                   "postestimation Methods and formulas (w: 처치집단 지시변수)")
        self.add_table(t)

    def _granger(self, D_: DiDData, X: np.ndarray, names: List[str]):
        J = self._single_timing(D_, "estat granger")
        w = D_.w_row
        tl = list(D_.tlist)
        kJ = tl.index(J)
        leads = [w * (D_.tpos >= tl[j]) for j in range(1, kJ)]
        if not leads:
            raise ValueError("처치 전 시점이 2개 이상 필요")
        ln = [f"_glead{j}" for j in range(1, kJ)]
        fit = self._fit(D_, np.column_stack([X] + leads), names + ln)
        idx = [fit.names.index(n_) for n_ in ln if n_ in fit.names]
        F, q, df2, p = fit.wald(idx)
        t = ResultsTable(name="granger", title="Granger 인과 검정 (estat granger) — H0: 처치 이전 예상 효과 없음")
        for c, ttl, typ in (("F", "F", "number"), ("df1", "df1", "integer"), ("df2", "df2", "integer"),
                            ("p", "p", "number")):
            t.add_column(c, ttl, col_type=typ, fmt="pvalue" if c == "p" else "")
        t.add_row({"F": F, "df1": q, "df2": int(df2) if df2 is not None else None, "p": p})
        t.add_note("증강 모형: 가상 처치 시점 선행항 1(t ≥ j)·w, j = 2,…,J−1 추가 후 결합 Wald 검정 — Stata [CAUSAL] "
                   "didregress postestimation Methods and formulas")
        self.add_table(t)

    def _grangerplot(self, D_: DiDData):
        o = self.o
        J = self._single_timing(D_, "estat grangerplot")
        tl = list(D_.tlist)
        rel = D_.tpos - J
        treated = D_.w_row > 0
        lo_av, hi_av = int(rel[treated].min()), int(rel[treated].max())
        ref = -1
        if o["baseline"] is not None:
            bt = [tp for tp in tl if _fmt_val(D_.tlab[int(tp)]) == _fmt_val(o["baseline"])]
            if not bt:
                raise ValueError(f"baseline({o['baseline']}) — 시점 변수 {D_.time}의 값이 아님 "
                                 "(Stata: specified baseline not a level in time)")
            ref = int(bt[0]) - J
        nle, nla = o["nLeads"], o["nLags"]
        if nle is not None and nle <= 0:
            raise ValueError("nleads(#)는 0보다 커야 함")
        if nla is not None and nla < 0:
            raise ValueError("nlags(#)는 0 이상")
        lo = -min(nle, -lo_av) if nle is not None else lo_av
        hi = min(nla, hi_av) if nla is not None else hi_av
        cols, ks = _rel_dummies(rel, treated, lo, hi, ref, True)
        X0, names0 = self._twfe_design(D_)
        X0, names0 = X0[:, 1:], names0[1:]
        nm = [_lead_lag_name(k) for k in ks]
        fit = self._fit(D_, np.column_stack([X0] + cols), names0 + nm)
        rows = []
        for k, n_ in zip(ks, nm):
            if n_ not in fit.names:
                continue
            j = fit.names.index(n_)
            r = _row(fit.b[j], fit.se[j], fit.df_r, o["level"])
            r.update({"term": n_, "rel": k, "time": D_.label(J + k) if (J + k) in D_.tlab else ""})
            rows.append(r)
        notes = [f"기준 시기: 상대 시점 {ref} (시점 {D_.label(J + ref)}) 계수 0으로 고정",
                 f"끝점 묶음: 상대 시점 ≤ {lo}, ≥ {hi} (Stata nleads()/nlags() 정의)",
                 f"표준오차: {self._vce_label(fit.cl_name, fit.G)} (xtreg, fe vce(cluster)); 모형에 패널·시점 효과"
                 + (" ·공변량" if D_.covs else "") + " 포함",
                 "Stata [CAUSAL] didregress postestimation Methods and formulas (estat grangerplot)"]
        self._coef_table("grangerplot", "시점별 처치효과 — 선행·후행항 (estat grangerplot)", rows, "t",
                         [("term", "항", "text"), ("rel", "상대 시점", "integer"), ("time", D_.time, "text")], notes,
                         label="term")
        try:
            self.add_plot("시점별 처치효과 (estat grangerplot)", _plot_event(rows, ref, o["level"], "처치 시작 대비 상대 시점"))
        except Exception:
            pass

    # ------------------------------------------------------------------ event study
    def _run_event(self, D_: DiDData):
        o = self.o
        self._info_table(D_)
        treated = D_.cohort_row >= 0
        if not treated.any():
            raise ValueError("이중차분: 처치 집단 없음")
        rel = np.where(treated, D_.tpos - D_.cohort_row, 0)
        lo_av, hi_av = int(rel[treated].min()), int(rel[treated].max())
        ref = int(o["refPeriod"])
        nle, nla = o["nLeads"], o["nLags"]
        lo = -min(int(nle), -lo_av) if nle is not None else lo_av
        hi = min(int(nla), hi_av) if nla is not None else hi_av
        if not (lo <= ref <= hi):
            raise ValueError(f"기준 시기 {ref}가 추정 구간 [{lo}, {hi}] 밖에 있음")
        keep = np.ones(D_.N, bool)
        notes = []
        if not o["bin"] and (lo > lo_av or hi < hi_av):
            keep = ~treated | ((rel >= lo) & (rel <= hi))
            notes.append(f"구간 밖 처치집단 관측치 {int((~keep).sum())}개 제외 (끝점 묶음 사용 안 함)")
        Dk = _subset(D_, keep, o["delta"]) if not keep.all() else D_
        trk = Dk.cohort_row >= 0
        relk = np.where(trk, Dk.tpos - Dk.cohort_row, 0)
        cols, ks = _rel_dummies(relk, trk, lo, hi, ref, bool(o["bin"]))
        # order: lags, then leads from nearest to farthest (the farthest lead is dropped first when all units are
        # eventually treated and two normalizations are needed)
        order = sorted(range(len(ks)), key=lambda i: (ks[i] < 0, abs(ks[i])))
        cols, ks = [cols[i] for i in order], [ks[i] for i in order]
        Td, tnames = Dk.time_dummies()
        Xc = [Dk.P.df[Dk.covs].to_numpy(float)] if Dk.covs else []
        X = np.column_stack(Xc + ([Td] if Td.shape[1] else []) + cols)
        nm = [_lead_lag_name(k) for k in ks]
        names = list(Dk.covs) + tnames + nm
        fit = self._fit(Dk, X, names)
        self.fit = fit
        rows = []
        for k, n_ in sorted(zip(ks, nm)):
            if n_ not in fit.names:
                continue
            j = fit.names.index(n_)
            r = _row(fit.b[j], fit.se[j], fit.df_r, o["level"])
            lab = (f"≤ {k}" if (o["bin"] and k == lo and lo > lo_av) else
                   f"≥ {k}" if (o["bin"] and k == hi and hi < hi_av) else str(k))
            r.update({"term": n_, "rel": k, "rel_lab": lab})
            rows.append(r)
        omitted_ev = [n_ for n_ in nm if n_ not in fit.names]
        never = (D_.cohort_grp < 0).any()
        notes += [f"기준 시기: 상대 시점 {ref} (계수 0)",
                  f"표준오차: {self._vce_label(fit.cl_name, fit.G)}, 소표본 보정 (N-1)/(N-K)·G/(G-1), t 자유도 {int(fit.df_r)}",
                  "추정: 패널 고정효과 + 시점 더미" + (" + 공변량" if Dk.covs else "") + " — linearmodels PanelOLS "
                  "(Stata: xtreg y 선행·후행 더미 i.시점, fe vce(cluster))",
                  "통제집단: " + ("처치되지 않은 집단과 아직 처치되지 않은 관측치" if never else
                              "아직 처치되지 않은 관측치 (처치되지 않은 집단 없음)")]
        if o["bin"] and (lo > lo_av or hi < hi_av):
            notes.append(f"끝점 묶음: 상대 시점 ≤ {lo}, ≥ {hi}")
        if omitted_ev:
            notes.append("공선성으로 제외 (추가 정규화): " + ", ".join(omitted_ev))
        if len(np.unique(D_.cohort_grp[D_.cohort_grp >= 0])) > 1:
            notes.append("처치 시점이 다른 코호트를 합친 TWFE 사건연구 계수는 효과 이질성이 있으면 다른 코호트·시점 효과가 섞일 수 "
                         "있음 (Sun & Abraham 2021) — method='hdid' 결과와 비교 권장")
        self._coef_table("event", "사건연구 계수 (상대 시점별 처치효과)", rows, "t",
                         [("term", "항", "text"), ("rel_lab", "상대 시점", "text")], notes, label="term")
        pre = [fit.names.index(n_) for k, n_ in zip(ks, nm) if k < 0 and n_ in fit.names]
        post = [fit.names.index(n_) for k, n_ in zip(ks, nm) if k >= 0 and n_ in fit.names]
        t = ResultsTable(name="event_tests", title="결합 검정 (사건연구)")
        t.add_column("test", "검정", col_type="text")
        t.add_column("F", "F", col_type="number")
        t.add_column("df1", "df1", col_type="integer")
        t.add_column("df2", "df2", col_type="integer")
        t.add_column("p", "p", col_type="number", fmt="pvalue")
        for lab, idx in (("처치 전 계수 = 0 (평행추세·예상효과)", pre), ("처치 후 계수 = 0", post)):
            if idx:
                F, q, df2, p = fit.wald(idx)
                t.add_row({"test": lab, "F": F, "df1": q, "df2": int(df2) if df2 is not None else None, "p": p})
        t.add_note("Stata: test 선행항 / test 후행항 (xtreg 군집 VCE, F 자유도 G−1)")
        self.add_table(t)
        self.model_stats.update({"N": int(Dk.N), "Clusters": int(fit.G)})
        try:
            self.add_plot("사건연구 계수 (event study)", _plot_event(rows, ref, o["level"], "처치 시작 대비 상대 시점"))
        except Exception:
            pass
        if pre:
            F, q, df2, p = fit.wald(pre)
            self.apa_narrative = (f"TWFE 사건연구 (관측치 {Dk.N}개, 군집 {fit.G}개): 처치 전 계수 결합 검정 F({q}, {int(df2)}) = "
                                  f"{F:.2f}, p {_pstr(p)}.")

    # ------------------------------------------------------------------ hdid (xthdidregress)
    def _cohort_table(self, D_: DiDData, used_never: bool, ctrl_label: str):
        t = ResultsTable(name="cohorts", title="처치 코호트 정보 (cohort = 첫 처치 시점)")
        t.add_column("cohort", "코호트", col_type="text")
        t.add_column("groups", f"집단 수 ({D_.group})", col_type="integer")
        t.add_column("obs", "관측치 수", col_type="integer")
        ch = D_.cohort_row
        for c in [-1] + sorted(int(x) for x in np.unique(ch) if x >= 0):
            mask = ch == c
            if not mask.any():
                continue
            t.add_row({"cohort": "처치 안 됨 (Never treated)" if c < 0 else D_.label(c),
                       "groups": int(len(np.unique(D_.grp[mask]))), "obs": int(mask.sum())})
        t.add_note(f"코호트 수 {len(t.rows)} (처치 안 됨 포함); 통제집단: {ctrl_label}")
        self.add_table(t)

    def _run_hdid(self, D_: DiDData):
        o = self.o
        est = o["estimator"]
        lev = o["level"]
        cg = o["controlGroup"]
        if est == "twfe":
            R = wooldridge_twfe(D_, cg, ptrends=o["ptrends"])
            df_r, stat = R.df_r, "t"
            ctrl = "아직 처치되지 않은 집단 (마지막 코호트)" if R.control_used == "notyet" else "처치되지 않은 집단 (Never treated)"
            cells = R.cells
            vce = (f"군집-강건 표준오차 ({D_.group} 군집 {R.G}개), margins vce(unconditional)과 같은 형태: 회귀 점수 + 코호트 공변량 "
                   f"평균 점수, 보정 (N-1)/(N-K)·G/(G-1) (K = {R.K}, Mundlak 회귀 계수 수), t 자유도 {int(R.df_r)}")
            n_cl = R.G
        else:
            R = callaway_santanna(D_, est, cg, o["baseTime"])
            V = R.vcov(R.psi)
            for k, c in enumerate(R.cells):
                c["se"] = float(np.sqrt(V[k, k])) if not c["omitted"] else None
            df_r, stat = None, "z"
            cells = R.cells
            ctrl = "처치되지 않은 집단 (Never treated)" if cg == "never" else "아직 처치되지 않은 집단 (Not yet treated)"
            vce = (f"군집-강건 표준오차 ({D_.group} 군집 {R.n_cl}개): 영향함수 군집합 (Callaway & Sant'Anna 2021 정리 2; "
                   "Stata xthdidregress Methods and formulas)")
            n_cl = R.n_cl
        self.hd = R
        self._cohort_table(D_, True, ctrl)
        rows = []
        for c in cells:
            if c["omitted"]:
                rows.append({"cohort": D_.label(c["g"]), "time": D_.label(c["t"]), "b": None, "se": None, "stat": None,
                             "p": None, "cil": None, "ciu": None, "term": f"{D_.label(c['g'])}:{D_.label(c['t'])}"})
                continue
            r = _row(c["att"], c["se"], df_r, lev)
            r.update({"cohort": D_.label(c["g"]), "time": D_.label(c["t"]),
                      "term": f"{D_.label(c['g'])}:{D_.label(c['t'])}"})
            rows.append(r)
        notes = [f"추정량: {EST_LABEL[est]} — " + {
            "ra": "통제집단의 ΔY 선형회귀 (statsmodels OLS), Sant'Anna & Zhao (2020) 회귀 DiD 영향함수",
            "ipw": "처치 성향 logit (statsmodels Logit), 표준화 역확률 가중 DiD (Sant'Anna & Zhao 2020)",
            "aipw": "성향 logit + 결과 선형회귀, 이중 강건 DiD (Sant'Anna & Zhao 2020 식 3.1)",
            "twfe": "코호트×시점 처치 셀 + 공변량 상호작용, 패널 고정효과 (집단 내 변환 후 statsmodels OLS; Wooldridge 2025)"}[est],
            f"통제집단: {ctrl}", "표준오차: " + vce]
        if est != "twfe":
            notes.append("기준 시점: " + ("처치 후 g−1, 처치 전 직전 시점 t−1 (basetime(adaptive))" if o["baseTime"] == "adaptive"
                                       else "모든 시점 g−1 (basetime(common))")
                         + "; g−1은 코호트가 관측된 g 이전 마지막 시점")
            notes.append("2×2 표본: 두 시점에 모두 관측된 개체 (paired observations); 공변량은 두 시점 중 앞 시점 값"
                         + ("; 아직 처치되지 않은 통제: 처치 안 됨 또는 max(t, g) 이후 처치" if cg == "notyet" else ""))
        if D_.covs or (est in ("ipw", "aipw") and D_.tcovs):
            notes.append("ATET 공변량 조정 (ATET computed using covariates)")
        if any(c["omitted"] for c in cells):
            notes.append("추정 불가 셀 (Stata omitted): 처치·통제 관측치 부족" +
                         (" 또는 공변량 상호작용 식별 불가 (1개 집단 코호트 등)" if est == "twfe" else ""))
        notes += R.notes
        self._coef_table("attgt", f"코호트·시점별 ATET — ATT(g,t) (수준 {lev:g}%)", rows, stat,
                         [("cohort", "코호트", "text"), ("time", D_.time, "text")], notes, label="term")
        self.model_stats.update({"Clusters": int(n_cl)})
        agg_fun = (lambda kind: aggregate_twfe(R, kind)) if est == "twfe" else (lambda kind: aggregate_cs(R, kind))
        w_note = ("가중치: 처치 관측치 평균 (margins, subpop(if treat==1) vce(unconditional))" if est == "twfe" else
                  "가중치: 각 ATT(g,t) 2×2 표본의 코호트 개체 수 (Stata weights(timecohort)), 가중치 추정의 영향함수 포함 "
                  "(Callaway & Sant'Anna 2021 4.2절)")
        overall = None
        for kind, flag, title, col in (("overall", "aggOverall", "전체 ATET (estat aggregation)", "구분"),
                                       ("cohort", "aggCohort", "코호트별 ATET (estat aggregation, cohort)", "코호트"),
                                       ("time", "aggTime", "시점별 ATET (estat aggregation, time)", D_.time),
                                       ("dynamic", "aggDynamic", "노출 기간별 ATET (estat aggregation, dynamic)", "노출 기간")):
            if not o[flag]:
                continue
            res = agg_fun(kind)
            arows = []
            for a in res:
                r = _row(a["att"], a["se"], df_r, lev)
                key = a["key"]
                if kind == "overall":
                    lab = f"{D_.treat} (1 vs 0)"
                elif kind in ("cohort", "time"):
                    lab = D_.label(key)
                else:
                    lab = str(int(round(key)))
                r["key"] = lab
                arows.append(r)
            nts = [w_note, "표준오차: " + ("영향함수 군집합" if est != "twfe" else "회귀·가중치 점수 군집합, t 자유도 G−1")]
            if kind == "dynamic":
                nts.append("노출 기간: 첫 처치 시점 이후 경과 기간 (음수는 처치 전)")
            self._coef_table(f"agg_{kind}", title, arows, stat, [("key", col, "text")], nts, label="key")
            if kind == "overall" and arows:
                overall = arows[0]
            if kind == "dynamic" and arows:
                try:
                    pts = [dict(r, rel=int(r["key"])) for r in arows]
                    self.add_plot("노출 기간별 ATET (estat aggregation, dynamic)",
                                  _plot_event(pts, None, lev, "노출 기간 (첫 처치 시점 대비)"))
                except Exception:
                    pass
        if o["ptrends"]:
            self._safe(lambda: self._ptrends_hdid(D_, R, est, cells), "estat ptrends")
        if o["atetPlot"]:
            try:
                self.add_plot("코호트별 ATT(g,t) (estat atetplot)", _plot_atet(D_, cells, df_r, lev))
            except Exception:
                pass
        if overall is not None and overall.get("se"):
            self.model_stats["ATET"] = float(overall["b"])
            self.apa_narrative = (f"{EST_LABEL[est]} 이질적 처치효과 이중차분의 전체 ATET = {overall['b']:.3f} "
                                  f"(SE = {overall['se']:.3f}), {stat} = {overall['stat']:.2f}, p {_pstr(overall['p'])}, "
                                  f"{lev:g}% CI [{overall['cil']:.3f}, {overall['ciu']:.3f}] (패널 {D_.n}개, 군집 {n_cl}개).")

    def _ptrends_hdid(self, D_: DiDData, R, est: str, cells):
        if est == "twfe":
            w = R.ptrends
            if w is None:
                raise ValueError("처치 전 코호트·시점 셀 없음")
        else:
            pre = [k for k, c in enumerate(cells) if not c["post"] and not c["omitted"]]
            if not pre:
                raise ValueError("처치 전 ATT(g,t) 없음")
            V = R.vcov(R.psi[:, pre])
            w = _wald(np.array([cells[k]["att"] for k in pre]), V, None)
        t = ResultsTable(name="ptrends", title="평행추세 검정 (estat ptrends) — H0: 모든 처치 전 ATET = 0")
        if w["df_r"] is None:
            t.add_column("chi2", "χ²", col_type="number")
            t.add_column("df", "df", col_type="integer")
            t.add_column("p", "p", col_type="number", fmt="pvalue")
            t.add_row({"chi2": w["stat"], "df": w["q"], "p": w["p"]})
        else:
            t.add_column("F", "F", col_type="number")
            t.add_column("df1", "df1", col_type="integer")
            t.add_column("df2", "df2", col_type="integer")
            t.add_column("p", "p", col_type="number", fmt="pvalue")
            t.add_row({"F": w["stat"], "df1": w["q"], "df2": int(w["df_r"]), "p": w["p"]})
        if est == "twfe":
            t.add_note("증강 모형: 처치 전 코호트×시점 셀(공변량 상호작용 포함) 추가, 코호트 공변량 평균에서의 처치 전 효과 결합 "
                       "Wald 검정 — Stata [CAUSAL] hdidregress postestimation Methods and formulas")
        else:
            t.add_note("처치 전 ATT(g,t) 전체에 대한 Wald 검정 (영향함수 VCE) — Stata [CAUSAL] hdidregress postestimation")
        if w["dropped"]:
            t.add_note(f"분산행렬이 특이 — 선형종속 제약 {w['dropped']}개 제외 (앞 순서 제약 유지; 어느 제약을 빼는지에 따라 "
                       "통계량이 달라질 수 있음)")
        self.add_table(t)


def _subset(D_: DiDData, keep: np.ndarray, delta: float) -> DiDData:
    """DiDData on a subset of its rows (event study without endpoint binning)."""
    df = D_.P.df.loc[keep].drop(columns=["_tpos"])
    return DiDData(df, D_.entity, D_.time, D_.dep, D_.covs, D_.tcovs, D_.treat,
                   None if D_.group == D_.entity else D_.group, D_.cluster_var, delta)


# ----------------------------------------------------------------------------- plots
def _plot_trends(D_: DiDData, m: pd.DataFrame, tline) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plots import _clean_ax, _col, _hdr_col, fig_to_base64
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    xs = [D_.tlab[int(t)] for t in m.index]
    xv = np.arange(len(xs))
    if 0.0 in m.columns:
        ax.plot(xv, m[0.0].to_numpy(), color=_col(0), marker="o", linewidth=1.8, label="통제")
    if 1.0 in m.columns:
        ax.plot(xv, m[1.0].to_numpy(), color=_col(3), marker="o", linewidth=1.8, label="처치")
    k = list(m.index).index(tline)
    ax.axvline(k, color="#475569", linestyle="--", linewidth=1.0)
    ax.set_xticks(xv)
    ax.set_xticklabels([_fmt_val(x) for x in xs], rotation=45 if len(xs) > 8 else 0)
    ax.set_xlabel(D_.time)
    ax.set_ylabel(D_.dep)
    ax.set_title("관측 평균 추이 (통제 vs 처치)", color=_hdr_col())
    ax.legend(frameon=False)
    _clean_ax(ax)
    return fig_to_base64(fig)


def _plot_event(rows: List[Dict[str, Any]], ref: Optional[int], level: float, xlabel: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plots import _clean_ax, _col, _hdr_col, fig_to_base64
    pts = [(int(r["rel"]), r["b"], r.get("cil"), r.get("ciu")) for r in rows if r.get("b") is not None]
    if ref is not None:
        pts.append((ref, 0.0, None, None))
    pts.sort()
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    x = np.array([p[0] for p in pts], float)
    b = np.array([p[1] for p in pts], float)
    lo = np.array([np.nan if p[2] is None else p[2] for p in pts], float)
    hi = np.array([np.nan if p[3] is None else p[3] for p in pts], float)
    pre = x < 0
    ax.axhline(0, color="#94A3B8", linewidth=0.9)
    ax.axvline(-0.5, color="#475569", linestyle="--", linewidth=1.0)
    for mask, col, lab in ((pre, _col(0), "처치 전"), (~pre, _col(3), "처치 후")):
        if mask.any():
            ax.errorbar(x[mask], b[mask], yerr=[b[mask] - lo[mask], hi[mask] - b[mask]], fmt="o", color=col,
                        capsize=3, linewidth=1.2, label=lab)
    if ref is not None:
        ax.plot([ref], [0.0], marker="o", mfc="white", color=_col(0))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("추정치")
    ax.set_title(f"처치효과와 {level:g}% 신뢰구간", color=_hdr_col())
    ax.legend(frameon=False)
    _clean_ax(ax)
    return fig_to_base64(fig)


def _plot_atet(D_: DiDData, cells, df_r, level) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plots import _clean_ax, _col, _hdr_col, fig_to_base64
    gs = sorted({c["g"] for c in cells if not c["omitted"]})
    fig, axes = plt.subplots(len(gs), 1, figsize=(6.4, 2.4 * len(gs)), sharex=True, squeeze=False)
    for ax, g in zip(axes[:, 0], gs):
        cc = [c for c in cells if c["g"] == g and not c["omitted"]]
        x = np.array([c["t"] for c in cc], float)
        rr = [_row(c["att"], c["se"], df_r, level) for c in cc]
        b = np.array([r["b"] for r in rr])
        lo = np.array([r["cil"] for r in rr], float)
        hi = np.array([r["ciu"] for r in rr], float)
        post = np.array([c["post"] for c in cc])
        for mask, col in ((~post, _col(0)), (post, _col(3))):
            if mask.any():
                ax.errorbar(x[mask], b[mask], yerr=[b[mask] - lo[mask], hi[mask] - b[mask]], fmt="o", color=col, capsize=3)
        ax.axhline(0, color="#94A3B8", linewidth=0.9)
        ax.set_title(f"코호트 {D_.label(g)}", color=_hdr_col(), fontsize=10)
        _clean_ax(ax)
    ticks = list(D_.tlist)
    axes[-1, 0].set_xticks(ticks)
    axes[-1, 0].set_xticklabels([D_.label(t) for t in ticks], rotation=45 if len(ticks) > 8 else 0)
    axes[-1, 0].set_xlabel(D_.time)
    fig.tight_layout()
    return fig_to_base64(fig)


# ----------------------------------------------------------------------------- options, registry glue
def normalize_options(p: Dict[str, Any], preset: Optional[str] = None) -> Dict[str, Any]:
    p = p or {}
    method_raw = _opt(p, "method", "did_method", "type")
    if method_raw is None:
        method_raw = preset or "twfe"
    method = resolve_choice("방법(method)", method_raw, METHODS, METHOD_ALIASES, default="twfe")
    est = resolve_choice("추정량(estimator)", _opt(p, "estimator", "est", "estMethod", "est_method"), ESTIMATORS,
                         EST_ALIASES, default="ra")
    cg = resolve_choice("통제집단(controlGroup)", _opt(p, "controlGroup", "control_group", "controlgroup"),
                        CONTROL_GROUPS, CG_ALIASES, default="never")
    bt = resolve_choice("기준 시점(baseTime)", _opt(p, "baseTime", "base_time", "basetime", "basePeriod"), BASE_TIMES,
                        BT_ALIASES, default="adaptive")
    wbw = resolve_choice("부트스트랩 오차 가중(wbWeight)", _opt(p, "wbWeight", "errorweight", "wb_weight"), WB_WEIGHTS,
                         {"rad": "rademacher", "mam": "mammen"}, default="rademacher")
    vce = str(_opt(p, "covType", "vce", "cov_type", default="cluster")).strip().lower()
    if vce not in ("cluster", "clustered", "robust", ""):
        raise ValueError(f"표준오차(covType): '{vce}' 미지원 — 군집-강건(cluster, 기본 group 변수)만 지원 "
                         "(Stata xtdidregress vce(hc2/hc3/bootstrap) 미지원)")
    level = float(_opt(p, "level", "ciLevel", "ciWidth", default=95.0))
    if level <= 1:
        level *= 100
    if not 50 <= level < 100:
        raise ValueError("신뢰수준(level)은 50 이상 100 미만")
    o = {
        "entity": _as_single(_opt(p, "entity", "entity_col", "idVar", "panelVar")),
        "time": _as_single(_opt(p, "time", "time_col", "timeVar")),
        "dep": _as_single(_opt(p, "dep", "depvar", "outcome", "y")),
        "covs": _as_list(_opt(p, "covs", "indeps", "controls", "xvars", default=[])),
        "treat": _as_single(_opt(p, "treat", "treatment", "tvar", "d")),
        "group": _as_single(_opt(p, "group", "groupVar", "group_var")) or "",
        "method": method, "estimator": est,
        "tcovs": _as_list(_opt(p, "tcovs", "treatCovs", "treat_covs", "pscoreCovs", default=[])),
        "controlGroup": cg, "baseTime": bt,
        "cluster": _as_single(_opt(p, "cluster", "clusterVar", "cluster_var")) or "",
        "trendPlots": _flag(p, "trendPlots", "trendplots", default=True),
        "ptrends": _flag(p, "ptrends", "ptrend", default=True),
        "granger": _flag(p, "granger", default=True),
        "grangerPlot": _flag(p, "grangerPlot", "grangerplot", default=True),
        "nLeads": _int_or_none(_opt(p, "nLeads", "nleads", "leads"), "nLeads"),
        "nLags": _int_or_none(_opt(p, "nLags", "nlags", "lags"), "nLags"),
        "baseline": _num_or_none(_opt(p, "baseline")),
        "refPeriod": _int_or_none(_opt(p, "refPeriod", "ref_period", "ref", default=-1), "refPeriod"),
        "bin": _flag(p, "bin", "binEnds", "bin_ends", default=True),
        "aggOverall": _flag(p, "aggOverall", "agg_overall", default=True),
        "aggCohort": _flag(p, "aggCohort", "agg_cohort", default=True),
        "aggTime": _flag(p, "aggTime", "agg_time", default=False),
        "aggDynamic": _flag(p, "aggDynamic", "agg_dynamic", default=True),
        "atetPlot": _flag(p, "atetPlot", "atetplot", default=False),
        "wildBootstrap": _flag(p, "wildBootstrap", "wildbootstrap", "wild", default=False),
        "wbReps": _int_or_none(_opt(p, "wbReps", "reps", default=1000), "wbReps"),
        "wbWeight": wbw,
        "wbSeed": _int_or_none(_opt(p, "wbSeed", "rseed", "seed", default=12345), "wbSeed"),
        "showControls": _flag(p, "showControls", "aequations", default=False),
        "level": level, "delta": float(_opt(p, "delta", default=1.0)),
    }
    if o["wbReps"] is None or o["wbReps"] < 2:
        raise ValueError("부트스트랩 반복 수(wbReps)는 2 이상")
    # The web form keeps both covariate lists while the estimator is switched: a list the chosen estimator has no
    # equation for is not applied and the first table says so (the Stata command input cannot produce this case).
    o["ignored"] = []
    if o["method"] == "hdid" and o["estimator"] in ("ra", "twfe") and o["tcovs"]:
        o["ignored"].append(f"처치 모형 공변량 {', '.join(o['tcovs'])} (xthdidregress {o['estimator']}: 처치 모형 없음 — "
                            "ipw·aipw에서 사용)")
        o["tcovs"] = []
    if o["method"] == "hdid" and o["estimator"] == "ipw" and o["covs"]:
        o["ignored"].append(f"결과 모형 공변량 {', '.join(o['covs'])} (xthdidregress ipw: 결과 모형 없음 — 공변량은 처치 "
                            "모형 공변량(tcovs)에 지정, Stata ipw (y) (d z))")
        o["covs"] = []
    if o["method"] == "hdid" and o["estimator"] == "twfe" and o["baseTime"] == "common":
        raise ValueError("basetime(common)은 ra·ipw·aipw에서만 지정 가능 (Stata 동일)")
    if o["method"] == "hdid" and o["cluster"] and o["cluster"] != (o["group"] or o["entity"]):
        raise ValueError("xthdidregress: 군집 변수는 group() 변수만 지원 (vce(cluster group))")
    return o


def python_call(o: Dict[str, Any]) -> str:
    parts = [f"entity={o['entity']!r}", f"time={o['time']!r}", f"dep={o['dep']!r}", f"treat={o['treat']!r}"]
    if o["covs"]:
        parts.append(f"covs={o['covs']!r}")
    parts.append(f"method={o['method']!r}")
    for k, dv in DEFAULTS.items():
        if k in ("method",):
            continue
        if k == "estimator" and o["method"] != "hdid":
            continue
        v = o.get(k)
        if v != dv and not (isinstance(dv, list) and not v):
            parts.append(f"{k}={v!r}")
    return f"res = sess.run('panelDiD', {', '.join(parts)})"


def _preset(params: Dict[str, Any]) -> Optional[str]:
    name = params.get("_alias") or params.get("_analysis")
    if not name:
        return None
    return ALIAS_PRESETS.get(str(name).strip().lower())


def dispatch(t: str, df: pd.DataFrame, params: Dict[str, Any]):
    if t != "panelDiD":
        return None
    p = dict(params or {})
    preset = _preset(p)
    for k in ("_alias", "_analysis"):
        p.pop(k, None)
    o = normalize_options(p, preset)
    return PanelDiDAnalysis(df, **o)


# ----------------------------------------------------------------------------- code export
def _fmt_num(v: Any) -> str:
    return _fmt_val(v) if v is not None else ""


def stata_code(t: str, p: Dict[str, Any]) -> Optional[List[str]]:
    if t != "panelDiD":
        return None
    o = normalize_options(dict(p), _preset(p))
    e, tm, y, tr = o["entity"], o["time"], o["dep"], o["treat"]
    g = o["group"] or e
    x = " ".join(o["covs"])
    out = xtset_lines(e, tm, o["delta"])
    lev = "" if o["level"] == 95 else f" level({o['level']:g})"
    if o["method"] == "twfe":
        opts = f"group({g}) time({tm})"
        if o["cluster"] and o["cluster"] != g:
            opts += f" vce(cluster {o['cluster']})"
        if o["wildBootstrap"]:
            opts += (f" wildbootstrap(rseed({o['wbSeed']}) reps({o['wbReps']})"
                     + ("" if o["wbWeight"] == "rademacher" else f" errorweight({o['wbWeight']})") + ")")
        out.append(f"xtdidregress ({y}{' ' + x if x else ''}) ({tr}), {opts}{lev}")
        if o["trendPlots"]:
            out.append("estat trendplots")
        if o["ptrends"]:
            out.append("estat ptrends")
        if o["granger"]:
            out.append("estat granger")
        if o["grangerPlot"]:
            gp = []
            if o["nLeads"] is not None:
                gp.append(f"nleads({o['nLeads']})")
            if o["nLags"] is not None:
                gp.append(f"nlags({o['nLags']})")
            if o["baseline"] is not None:
                gp.append(f"baseline({_fmt_num(o['baseline'])})")
            out.append("estat grangerplot, " + " ".join(gp + ["verbose"]))
        return out
    if o["method"] == "event":
        cl = o["cluster"] or g
        ref = o["refPeriod"]
        out += ["* 사건연구: 첫 처치 시점(코호트)과 상대 시점 변수",
                f"bysort {g}: egen _first = min(cond({tr} == 1, {tm}, .))",
                f"generate _rel = ({tm} - _first)" + (f" / {o['delta']:g}" if o["delta"] != 1 else ""),
                "quietly summarize _rel",
                "local lo = r(min)", "local hi = r(max)"]
        if o["nLeads"] is not None:
            out.append(f"local lo = max(`lo', -{o['nLeads']})")
        if o["nLags"] is not None:
            out.append(f"local hi = min(`hi', {o['nLags']})")
        if not o["bin"] and (o["nLeads"] is not None or o["nLags"] is not None):
            out.append("drop if _rel < `lo' | (_rel > `hi' & _rel < .)   // 끝점 묶음 없이 구간 밖 처치 관측치 제외")
        out += ["local evars",
                "forvalues k = `lo'/`hi' {",
                f"    if `k' == {ref} continue",
                "    local nm = cond(`k' < 0, \"F\" + string(-`k'), \"L\" + string(`k'))"]
        if o["bin"]:
            out += ["    if `k' == `lo' generate byte `nm' = (_rel <= `k')",
                    "    else if `k' == `hi' generate byte `nm' = (_rel >= `k' & _rel < .)",
                    "    else generate byte `nm' = (_rel == `k')"]
        else:
            out.append("    generate byte `nm' = (_rel == `k')")
        out += ["    local evars `evars' `nm'", "}",
                f"xtreg {y} `evars'{' ' + x if x else ''} i.{tm}, fe vce(cluster {cl}){lev}",
                "* 처치 전 계수 결합 검정", "local pre", "foreach v of local evars {",
                "    if substr(\"`v'\", 1, 1) == \"F\" local pre `pre' `v'", "}", "test `pre'"]
        return out
    est = o["estimator"]
    tz = " ".join(o["tcovs"])
    om = f"({y}{' ' + x if x and est != 'ipw' else ''})"
    tmod = f"({tr}{' ' + tz if tz and est in ('ipw', 'aipw') else ''})"
    opts = f"group({g})"
    if o["controlGroup"] != "never":
        opts += " controlgroup(notyet)"
    if est != "twfe" and o["baseTime"] != "adaptive":
        opts += " basetime(common)"
    out.append(f"xthdidregress {est} {om} {tmod}, {opts}{lev}")
    if o["aggOverall"]:
        out.append("estat aggregation")
    for flag, kw in (("aggCohort", "cohort"), ("aggTime", "time"), ("aggDynamic", "dynamic")):
        if o[flag]:
            out.append(f"estat aggregation, {kw}" + (" graph" if kw == "dynamic" else ""))
    if o["ptrends"]:
        out.append("estat ptrends")
    if o["atetPlot"]:
        out.append("estat atetplot")
    return out


def r_code(t: str, p: Dict[str, Any]) -> Optional[List[str]]:
    if t != "panelDiD":
        return None
    o = normalize_options(dict(p), _preset(p))
    e, tm, y, tr = o["entity"], o["time"], o["dep"], o["treat"]
    g = o["group"] or e
    cl = o["cluster"] or g
    x = " + ".join(o["covs"])
    q = lambda s: '"' + str(s) + '"'
    if o["method"] == "twfe":
        out = ["# install.packages(\"fixest\")",
               f"fit <- fixest::feols({y} ~ {tr}{' + ' + x if x else ''} | {e} + {tm}, data = data, cluster = ~{cl})",
               "summary(fit)  # 소표본 보정: fixest 기본 ssc는 Stata xtreg, fe vce(cluster)와 다를 수 있음",
               "# Stata xtreg와 같은 보정: fixest::ssc(adj = TRUE, fixef.K = \"none\", cluster.adj = TRUE)",
               "# estat ptrends / granger / grangerplot에 해당하는 R 함수 없음 — 아래 사건연구 회귀로 확인",
               f"data$`_first` <- ave(ifelse(data${tr} == 1, data${tm}, NA), data${g}, FUN = function(v) suppressWarnings(min(v, na.rm = TRUE)))",
               f"data$`_rel` <- ifelse(is.finite(data$`_first`), data${tm} - data$`_first`, -1000)",
               f"es <- fixest::feols({y} ~ i(`_rel`, ref = c(-1, -1000)){' + ' + x if x else ''} | {e} + {tm}, data = data, cluster = ~{cl})",
               "fixest::iplot(es)"]
        if o["wildBootstrap"]:
            out += ["# 와일드 군집 부트스트랩: fwildclusterboot 패키지 (CRAN 보관됨) 또는 Stata boottest와 같은 방식",
                    f"# fwildclusterboot::boottest(fit, param = {q(tr)}, clustid = {q(cl)}, B = {o['wbReps']}, type = {q(o['wbWeight'])})"]
        return out
    if o["method"] == "event":
        ref = o["refPeriod"]
        out = ["# install.packages(\"fixest\")",
               f"data$`_first` <- ave(ifelse(data${tr} == 1, data${tm}, NA), data${g}, FUN = function(v) suppressWarnings(min(v, na.rm = TRUE)))",
               f"data$`_rel` <- ifelse(is.finite(data$`_first`), data${tm} - data$`_first`, -1000)  # -1000: 처치 안 됨"]
        if o["nLeads"] is not None or o["nLags"] is not None:
            lo = f"-{o['nLeads']}" if o["nLeads"] is not None else "-Inf"
            hi = f"{o['nLags']}" if o["nLags"] is not None else "Inf"
            if o["bin"]:
                out.append(f"data$`_rel` <- ifelse(data$`_rel` == -1000, -1000, pmin(pmax(data$`_rel`, {lo}), {hi}))  # 끝점 묶음")
            else:
                out.append(f"data <- subset(data, `_rel` == -1000 | (`_rel` >= {lo} & `_rel` <= {hi}))")
        out += [f"es <- fixest::feols({y} ~ i(`_rel`, ref = c({ref}, -1000)){' + ' + x if x else ''} | {e} + {tm}, "
                f"data = data, cluster = ~{cl})",
                "summary(es)", "fixest::iplot(es)",
                "# 처치 시점이 다른 코호트가 있을 때 Sun & Abraham (2021) 추정: "
                f"fixest::feols({y} ~ sunab(`_first`, {tm}){' + ' + x if x else ''} | {e} + {tm}, data = data, cluster = ~{cl})",
                "# (sunab은 코호트별 상대 시점 효과를 따로 추정한 뒤 합침 — TWFE 사건연구 계수와 다름)"]
        return out
    est = o["estimator"]
    if est == "twfe":
        return ["# install.packages(\"etwfe\")  # Wooldridge (2025) 확장 TWFE",
                f"data$`_first` <- ave(ifelse(data${tr} == 1, data${tm}, NA), data${g}, FUN = function(v) suppressWarnings(min(v, na.rm = TRUE)))",
                "data$`_first`[!is.finite(data$`_first`)] <- 0",
                f"fit <- etwfe::etwfe(fml = {y} ~ {x if x else '0'}, tvar = {tm}, gvar = `_first`, data = data, "
                f"vcov = ~{cl}{', cgroup = \"notyet\"' if o['controlGroup'] == 'notyet' else ''})",
                "etwfe::emfx(fit, type = \"simple\")    # estat aggregation",
                "etwfe::emfx(fit, type = \"group\")     # estat aggregation, cohort",
                "etwfe::emfx(fit, type = \"calendar\")  # estat aggregation, time",
                "etwfe::emfx(fit, type = \"event\")     # estat aggregation, dynamic",
                "# 공변량 처리(코호트 평균 중심화)와 소표본 보정은 Stata xthdidregress twfe와 다를 수 있음"]
    xf = " + ".join(o["covs"] if est != "ipw" else o["tcovs"])
    out = ["# install.packages(\"did\")",
           f"data$`_first` <- ave(ifelse(data${tr} == 1, data${tm}, NA), data${g}, FUN = function(v) suppressWarnings(min(v, na.rm = TRUE)))",
           "data$`_first`[!is.finite(data$`_first`)] <- 0   # 0 = 처치 안 됨",
           f"att <- did::att_gt(yname = {q(y)}, tname = {q(tm)}, idname = {q(e)}, gname = \"_first\", "
           f"xformla = {('~ ' + xf) if xf else '~ 1'}, data = data, est_method = {q({'ra': 'reg', 'ipw': 'ipw', 'aipw': 'dr'}[est])}, "
           f"control_group = {q('nevertreated' if o['controlGroup'] == 'never' else 'notyettreated')}, "
           f"base_period = {q('varying' if o['baseTime'] == 'adaptive' else 'universal')}"
           + (f", clustervars = {q(g)}" if g != e else "") + ", bstrap = FALSE)",
           "summary(att)",
           "did::aggte(att, type = \"simple\", bstrap = FALSE)    # estat aggregation",
           "did::aggte(att, type = \"group\", bstrap = FALSE)     # estat aggregation, cohort",
           "did::aggte(att, type = \"calendar\", bstrap = FALSE)  # estat aggregation, time",
           "did::ggdid(did::aggte(att, type = \"dynamic\", bstrap = FALSE))  # estat aggregation, dynamic graph"]
    if est == "aipw" and o["tcovs"] and set(o["tcovs"]) != set(o["covs"]):
        out.append("# did::att_gt는 결과·처치 모형에 같은 xformla 사용 — Stata aipw의 서로 다른 공변량 집합과 다름")
    if o["controlGroup"] == "notyet":
        out.append("# not-yet-treated 통제집단: 처치 전 시점에서 Stata는 g 이후 처치 코호트만 사용, R did는 t 이후 처치 코호트 사용")
    if o["baseTime"] == "common":
        out.append("# base_period = \"universal\": R did는 기준 시점 g-1의 ATT를 0으로 함께 보고")
    if g != e:
        out.append("# R did의 clustervars 표준오차는 곱셈 부트스트랩(bstrap = TRUE)에서만 반영")
    return out


# ----------------------------------------------------------------------------- Stata command input
_LAST_DID = "next(a for a in reversed(sess.history) if getattr(a, 'name', '') == 'panelDiD')"
_PAREN = re.compile(r"\(([^()]*)\)")


def _parse_vce_did(v: str, line: str):
    toks = v.split()
    if not toks:
        return None, None
    k = toks[0].lower()
    if "robust".startswith(k):
        return None, None
    if "cluster".startswith(k) and len(k) >= 2 and len(toks) == 2:
        return toks[1], None
    return None, _err(f"vce({v}) 미지원 — 군집-강건(vce(cluster 변수))만 지원 (hc2·hc3·bootstrap·jackknife 미지원): " + line)


def transpile(line: str) -> Optional[str]:
    s = line.strip()
    m = re.match(r"^(xtdidregress|xthdidregress)\s+(.*)$", s, re.IGNORECASE)
    if m:
        cmd = m.group(1).lower()
        rest = m.group(2).strip()
        est = None
        if cmd == "xthdidregress":
            mm = re.match(r"^(ra|ipw|aipw|twfe)\b\s*(.*)$", rest, re.IGNORECASE)
            if not mm:
                return _err("xthdidregress: 추정량 ra·ipw·aipw·twfe 중 하나 필요 — " + s)
            est, rest = mm.group(1).lower(), mm.group(2)
        if "," in rest:
            # split at the first comma outside parentheses
            depth, cut = 0, None
            for i, ch in enumerate(rest):
                depth += {"(": 1, ")": -1}.get(ch, 0)
                if ch == "," and depth == 0:
                    cut = i
                    break
            spec, opt_txt = (rest[:cut], rest[cut + 1:]) if cut is not None else (rest, "")
        else:
            spec, opt_txt = rest, ""
        groups = _PAREN.findall(spec)
        if len(groups) != 2 or _PAREN.sub("", spec).strip():
            return _err(f"{cmd}: (결과변수 공변량) (처치변수 [처치 공변량]) 형식 필요 — " + s)
        om, tmod = groups[0].split(), groups[1].replace(",", " , ").split()
        if not om or not tmod:
            return _err(f"{cmd}: 결과변수·처치변수 필요 — " + s)
        if "," in tmod:
            return _err(f"{cmd}: 처치변수 옵션(continuous 등) 미지원 — " + s)
        if any(re.search(r"[.#]", v) for v in om + tmod):
            return _err("요인변수·시계열 연산자(i.·c.·L.·#) 미지원: 변수를 만든 뒤 지정 — " + s)
        opts = _split_opts(opt_txt)
        kw = [f"dep={om[0]!r}", f"treat={tmod[0]!r}"]
        if om[1:]:
            kw.append(f"covs={om[1:]!r}")
        grp = opts.pop("group", None)
        if not grp:
            return _err(f"{cmd}: group() 필요 — " + s)
        gl = grp.split()
        if len(gl) != 1:
            return _err(f"{cmd}: group() 변수 1개만 지원 (삼중차분 미지원) — " + s)
        kw.append(f"group={gl[0]!r}")
        time_arg = "time=getattr(sess, 'panel_spec', {}).get('timeVar')"
        if "time" in opts:
            time_arg = f"time={opts.pop('time')!r}"
        if cmd == "xtdidregress":
            kw.append("method='twfe'")
            if len(tmod) > 1:
                return _err("xtdidregress: 처치 모형 공변량 미지원 — " + s)
            for bad in ("nointeract", "nogteffects", "aggregate"):
                if bad in opts:
                    return _err(f"xtdidregress {bad} 미지원 — " + s)
            if "wildbootstrap" in opts:
                wo = _split_opts(opts.pop("wildbootstrap"))
                kw.append("wildBootstrap=True")
                if "reps" in wo:
                    kw.append(f"wbReps={int(wo.pop('reps'))}")
                if "rseed" in wo:
                    kw.append(f"wbSeed={int(wo.pop('rseed'))}")
                if "errorweight" in wo:
                    kw.append(f"wbWeight={wo.pop('errorweight').strip().lower()!r}")
                wo.pop("blocksize", None)
                if wo:
                    return _err(f"wildbootstrap 옵션 미지원: {', '.join(wo)} — " + s)
        else:
            kw += ["method='hdid'", f"estimator={est!r}"]
            if len(tmod) > 1:
                if est in ("ra", "twfe"):
                    return _err(f"xthdidregress {est}: 처치 모형 공변량 지정 불가 — " + s)
                kw.append(f"tcovs={tmod[1:]!r}")
            if est == "ipw" and om[1:]:
                return _err("xthdidregress ipw: 결과 모형 공변량 지정 불가 — " + s)
            if "controlgroup" in opts:
                v = opts.pop("controlgroup").strip().lower()
                if v not in ("never", "notyet"):
                    return _err(f"controlgroup({v}) 미지원 — " + s)
                kw.append(f"controlGroup={v!r}")
            if "basetime" in opts:
                v = opts.pop("basetime").strip().lower()
                if v not in ("adaptive", "common"):
                    return _err(f"basetime({v}) 미지원 — " + s)
                kw.append(f"baseTime={v!r}")
            for bad in ("hettype", "over", "noxinteract", "usercohort", "cohortvar"):
                if bad in opts:
                    v = opts.pop(bad)
                    if bad == "hettype" and v.strip().lower() == "timecohort":
                        continue
                    if bad == "over" and v.strip().lower() == "cohort":
                        continue
                    if bad == "cohortvar":
                        continue
                    return _err(f"xthdidregress {bad}() 미지원 — " + s)
        if "vce" in opts:
            cl, err = _parse_vce_did(opts.pop("vce"), s)
            if err:
                return err
            if cl:
                kw.append(f"cluster={cl!r}")
        if "level" in opts:
            kw.append(f"level={float(opts.pop('level'))}")
        for k_ in ("nolog", "log", "nodots", "aequations", "coeflegend", "noci", "nopvalues", "vsquish"):
            opts.pop(k_, None)
        if opts:
            return _err(f"{cmd} 옵션 미지원: {', '.join(opts)} — " + s)
        ent = "entity=getattr(sess, 'panel_spec', {}).get('idVar')"
        return f"res = sess.run('panelDiD', {ent}, {time_arg}, {', '.join(kw)})"
    m = re.match(r"^estat\s+(\w+)\s*(?:,\s*(.*))?$", s, re.IGNORECASE)
    if not m:
        return None
    sub = m.group(1).lower()
    known = {"trendplots", "ptrends", "ptrend", "granger", "grangerplot", "aggregation", "atetplot", "bdecomp", "sci",
             "moderation"}
    if sub not in known:
        return None
    opts = _split_opts(m.group(2) or "")
    if sub in ("bdecomp", "sci", "moderation"):
        return _err(f"estat {sub} 미지원 (Bacon 분해·동시 신뢰구간·조절효과) — " + s)
    upd: Dict[str, Any] = {}
    if sub == "trendplots":
        upd["trendPlots"] = True
        for k_ in ("omeans", "ltrends", "notitle", "noxline", "nocommonlegend"):
            opts.pop(k_, None)
    elif sub in ("ptrends", "ptrend"):
        upd["ptrends"] = True
    elif sub == "granger":
        upd["granger"] = True
    elif sub == "grangerplot":
        upd["grangerPlot"] = True
        if "nleads" in opts:
            upd["nLeads"] = int(opts.pop("nleads"))
        if "nlags" in opts:
            upd["nLags"] = int(opts.pop("nlags"))
        if "baseline" in opts:
            upd["baseline"] = float(opts.pop("baseline"))
        for k_ in ("verbose", "post", "noci", "noyline", "lagview"):
            opts.pop(k_, None)
    elif sub == "atetplot":
        upd["atetPlot"] = True
        opts.pop("noci", None)
    else:
        kinds = [k_ for k_ in ("overall", "cohort", "time", "dynamic") if k_ in opts]
        for k_ in kinds:
            v = opts.pop(k_)
            if v:
                return _err(f"estat aggregation, {k_}(목록) 미지원 — 전체 목록만 계산 — " + s)
        kinds = kinds or ["overall"]
        if len(kinds) > 1:
            return _err("estat aggregation: overall·cohort·time·dynamic 중 하나만 지정 (Stata 동일) — " + s)
        upd[{"overall": "aggOverall", "cohort": "aggCohort", "time": "aggTime", "dynamic": "aggDynamic"}[kinds[0]]] = True
        if "weights" in opts and opts["weights"].strip().lower() != "timecohort":
            return _err("estat aggregation, weights(cohort) 미지원 — weights(timecohort)만 계산 — " + s)
        opts.pop("weights", None)
        opts.pop("graph", None)
        opts.pop("nograph", None)
        if "sci" in opts:
            return _err("estat aggregation, sci 미지원 (동시 신뢰구간) — " + s)
    opts.pop("level", None)
    if opts:
        return _err(f"estat {sub} 옵션 미지원: {', '.join(opts)} — " + s)
    upd_txt = ", ".join(f"{k}={v!r}" for k, v in upd.items())
    return (f"res = sess.run('panelDiD', **dict({_LAST_DID}.options, {upd_txt}))"
            f"  # [Stata] {s} → 직전 이중차분 결과에 추가")


EXTENSION = {
    "catalog": [{"id": "panelDiD", "group": "Linear Models",
                 "title": "Panel Difference-in-Differences (xtdidregress, event study, xthdidregress ra/ipw/aipw/twfe)",
                 "module": "openstat", "icon": "analysis-linearmodels.svg"}],
    "aliases": {"xtdidregress": "panelDiD", "xthdidregress": "panelDiD", "did": "panelDiD", "eventstudy": "panelDiD",
                "panel_did": "panelDiD", "paneldid": "panelDiD"},
    "dispatch": dispatch,
    "uses_alias": True,          # the typed name selects the method (xtdidregress -> twfe, eventstudy -> event, ...)
    "stata": stata_code,
    "r": r_code,
    "transpile": transpile,
}
