"""
OpenStat Studio — instrumental-variable regression (Stata `ivregress`) and panel IV (Stata `xtivreg`).
License: MIT

ivReg    ivregress 2sls | liml | gmm  y x1 (x2 = z1 z2), vce(unadjusted | robust | cluster c) small
         + estat firststage / endogenous / overid
panelIV  xtivreg y x1 (x2 = z1), fe | fd | be | re [ec2sls nosa]

Computation
- Estimators, variance matrices, first-stage diagnostics (R², partial R², Shea's partial R², partial F), Durbin,
  Wu–Hausman, Wooldridge score / regression tests, Sargan, Basmann, Basmann F, Hansen's J and the GMM C statistic:
  linearmodels (IV2SLS, IVLIML, IVGMM; `debiased=True` is Stata's `small`).
- Own code (Stata conventions the library does not provide): cluster `small` degrees of freedom G−1, GMM
  `vce(unadjusted)` (Ŝ = W⁻¹), Anderson–Rubin N(κ−1), adjusted R², Shea's adjusted partial R², the minimum-eigenvalue
  (Cragg–Donald) statistic with Stock–Yogo critical values, and for xtivreg the data transformations, variance
  components (Swamy–Arora / Baltagi–Chang), the u_i F test and R² definitions. No Python library provides G2SLS /
  EC2SLS (searched: linearmodels.panel, statsmodels, pydynpd); the 2SLS step on the transformed data is linearmodels.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import linalg, stats

from .base import BaseAnalysis, ResultsTable, resolve_choice


# ---------------------------------------------------------------- option helpers (copied from registry style)
def _opt(p: Dict[str, Any], *keys, default: Any = None) -> Any:
    for k in keys:
        if k in p and p[k] is not None and p[k] != "":
            return p[k]
    return default


def _flag(p: Dict[str, Any], *keys, default: bool = False) -> bool:
    v = _opt(p, *keys, default=default)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


def _as_list(val: Any) -> List[str]:
    if val is None or val == "":
        return []
    if isinstance(val, str):
        return [v.strip() for v in val.replace(",", " ").split() if v.strip()]
    return [str(v) for v in val if v is not None and str(v) != ""]


def _as_single(val: Any) -> Optional[str]:
    if isinstance(val, (list, tuple)):
        return str(val[0]) if len(val) > 0 else None
    if val is None or val == "":
        return None
    return str(val)


def _check_columns(df: pd.DataFrame, label: str, roles: Dict[str, Any]) -> None:
    missing = []
    for role, cols in roles.items():
        for c in (cols if isinstance(cols, (list, tuple)) else [cols]):
            if c and c not in df.columns:
                missing.append(f"{role} '{c}'")
    if missing:
        raise ValueError(f"{label}: 데이터에 없는 변수 — {', '.join(missing)}. 사용 가능한 변수: {', '.join(map(str, df.columns))}")


def _numeric(df: pd.DataFrame, cols: Sequence[str], label: str) -> None:
    bad = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise ValueError(f"{label}: 숫자형이 아닌 변수 — {', '.join(bad)} (범주형은 더미변수로 변환 후 지정)")


# ---------------------------------------------------------------- linear algebra glue
def _basis(A: np.ndarray, what: str, names: Optional[Sequence[str]] = None) -> np.ndarray:
    """Orthonormal basis of the column space of A; rank deficiency is an error (Stata would omit a variable)."""
    if A.shape[1] == 0:
        return np.zeros((A.shape[0], 0))
    q, r, piv = linalg.qr(A, mode="economic", pivoting=True)
    d = np.abs(np.diag(r))
    tol = d[0] * max(A.shape) * np.finfo(float).eps * 10 if d.size else 0.0
    rank = int((d > tol).sum())
    if rank < A.shape[1]:
        drop = [names[i] for i in piv[rank:]] if names is not None else []
        raise ValueError(f"{what}: 완전 공선성 — {', '.join(drop) if drop else '일부 변수'} 제외 후 다시 지정")
    return q


def _proj(v: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return Q @ (Q.T @ v) if Q.shape[1] else np.zeros_like(v)


def _annih(v: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return v - _proj(v, Q)


def _independent(base: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Columns of cand that add rank to [base, columns kept so far] (drops redundant panel instruments)."""
    kept: List[np.ndarray] = []
    cur = base
    for j in range(cand.shape[1]):
        trial = np.column_stack([cur, cand[:, j]]) if cur.shape[1] else cand[:, [j]]
        if np.linalg.matrix_rank(trial) > (np.linalg.matrix_rank(cur) if cur.shape[1] else 0):
            kept.append(cand[:, j])
            cur = trial
    return np.column_stack(kept) if kept else np.zeros((cand.shape[0], 0))


def _cluster_codes(c: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], int]:
    if c is None:
        return None, 0
    codes, uniq = pd.factorize(pd.Series(c), sort=True)
    return codes, len(uniq)


def _wald(b: np.ndarray, V: np.ndarray) -> float:
    if b.size == 0:
        return float("nan")
    return float(b @ np.linalg.solve(V, b))


VCE_TYPES = ("unadjusted", "robust", "cluster")
_VCE_ALIASES = {"ols": "unadjusted", "conventional": "unadjusted", "homoskedastic": "unadjusted", "iid": "unadjusted",
                "nonrobust": "unadjusted", "classical": "unadjusted", "r": "robust", "hc0": "robust", "hc1": "robust",
                "heteroskedastic": "robust", "white": "robust", "huber": "robust", "clustered": "cluster",
                "cl": "cluster", "clust": "cluster"}
ESTIMATORS = ("2sls", "liml", "gmm")
_EST_ALIASES = {"tsls": "2sls", "iv": "2sls", "2SLS": "2sls", "iv2sls": "2sls", "LIML": "liml", "ivliml": "liml",
                "GMM": "gmm", "ivgmm": "gmm", "twostep": "gmm", "2step": "gmm"}
EST_LABEL = {"2sls": "2단계 최소제곱 (2SLS)", "liml": "제한정보 최우추정 (LIML)", "gmm": "2단계 효율적 GMM"}
VCE_LABEL = {"unadjusted": "통상(동분산 가정)", "robust": "이분산 강건(robust)", "cluster": "군집 강건(cluster)"}
_LM_COV = {"unadjusted": "unadjusted", "robust": "robust", "cluster": "clustered"}


def _stat(t) -> Optional[float]:
    """Statistic of a linearmodels WaldTestStatistic (None for InvalidTestStatistic)."""
    v = getattr(t, "stat", None)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


# ---------------------------------------------------------------- estimator wrapper (linearmodels core)
class IVFit:
    """linearmodels IV2SLS / IVLIML / IVGMM with Stata ivregress conventions.

    X = [endogenous, included exogenous], Z = [included exogenous, excluded instruments]; parameter order follows Stata
    (endogenous first, constant last).
    """

    def __init__(self, y: np.ndarray, X2: np.ndarray, X1: np.ndarray, Z2: np.ndarray,
                 names2: Sequence[str], names1: Sequence[str], znames: Sequence[str],
                 estimator: str = "2sls", vce: str = "unadjusted", clusters: Optional[np.ndarray] = None,
                 small: bool = False, wmatrix: str = "robust", center: bool = False, has_const: bool = True):
        from linearmodels.iv import IV2SLS, IVGMM, IVLIML
        self.y = np.asarray(y, float)
        n = len(self.y)
        self.X2, self.X1, self.Z2 = (np.asarray(a, float).reshape(n, -1) for a in (X2, X1, Z2))
        self.names2, self.names1, self.znames = list(names2), list(names1), list(znames)
        self.estimator, self.vce, self.small, self.wmatrix, self.center = estimator, vce, small, wmatrix, center
        self.has_const = has_const
        self.n = n
        self.m, self.k1, self.l2 = self.X2.shape[1], self.X1.shape[1], self.Z2.shape[1]
        self.X = np.column_stack([self.X2, self.X1])
        self.Z = np.column_stack([self.X1, self.Z2])
        self.names = self.names2 + self.names1
        self.k, self.L = self.X.shape[1], self.Z.shape[1]
        self.codes, self.G = _cluster_codes(clusters)
        if self.m < 1:
            raise ValueError("도구변수 회귀: 내생변수 1개 이상 필요")
        if self.l2 < self.m:
            raise ValueError(f"도구변수 회귀: 식별 불가 — 제외 도구변수 {self.l2}개 < 내생변수 {self.m}개 (도구변수 수 ≥ 내생변수 수 필요)")
        if self.n <= self.L:
            raise ValueError(f"도구변수 회귀: 관측치 {self.n}개 — 도구변수 수({self.L})보다 많아야 함")
        if vce == "cluster" or (estimator == "gmm" and wmatrix == "cluster"):
            if self.codes is None:
                raise ValueError("군집 강건 표준오차: 군집 변수(clusterVar) 지정 필요")
            if self.G < 2:
                raise ValueError("군집 강건 표준오차: 군집 2개 이상 필요")
        self.QX = _basis(self.X, "회귀변수", self.names)
        self.QZ = _basis(self.Z, "도구변수", self.names1 + self.znames)
        self.QX1 = _basis(self.X1, "외생변수", self.names1) if self.k1 else np.zeros((n, 0))
        # linearmodels data (generic column names; exog first in linearmodels' parameter vector)
        self._dep = pd.Series(self.y, name="y")
        self._ex = pd.DataFrame(self.X1, columns=[f"x1_{i}" for i in range(self.k1)]) if self.k1 else None
        self._en = pd.DataFrame(self.X2, columns=[f"x2_{i}" for i in range(self.m)])
        self._in = pd.DataFrame(self.Z2, columns=[f"z2_{i}" for i in range(self.l2)])
        self._order = [f"x2_{i}" for i in range(self.m)] + [f"x1_{i}" for i in range(self.k1)]
        fit_kw: Dict[str, Any] = {"cov_type": _LM_COV[vce], "debiased": small}
        if vce == "cluster":
            fit_kw["clusters"] = self.codes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if estimator == "2sls":
                self.res = IV2SLS(self._dep, self._ex, self._en, self._in).fit(**fit_kw)
            elif estimator == "liml":
                self.res = IVLIML(self._dep, self._ex, self._en, self._in).fit(**fit_kw)
            else:
                wkw: Dict[str, Any] = {"weight_type": _LM_COV[wmatrix]}
                if wmatrix in ("robust", "cluster"):
                    wkw["center"] = center
                if wmatrix == "cluster":
                    wkw["clusters"] = self.codes
                gkw = dict(fit_kw)
                if vce == "unadjusted":
                    gkw["cov_type"] = "unadjusted"
                self.res = IVGMM(self._dep, self._ex, self._en, self._in, **wkw).fit(iter_limit=2, **gkw)
        self.b = self.res.params[self._order].to_numpy(float)
        V = self.res.cov.loc[self._order, self._order].to_numpy(float)
        if estimator == "gmm" and vce == "unadjusted":
            # Stata: vce(unadjusted) after GMM sets Ŝ = W⁻¹, giving the "optimal" variance N{X'ZWZ'X}⁻¹
            Wl = self.res.weight_matrix.to_numpy(float)
            lm_z = list(self.res.weight_matrix.index)
            Zl = pd.concat([d for d in (self._ex, self._in) if d is not None], axis=1)[lm_z].to_numpy(float)
            Xl = np.column_stack([self.X2, self.X1])
            Q = Zl.T @ Xl / n
            V = np.linalg.inv(Q.T @ Wl @ Q) / n * (n / (n - self.k) if small else 1.0)
        self.V = (V + V.T) / 2
        self.se = np.sqrt(np.clip(np.diag(self.V), 0, None))
        self.u = np.asarray(self.res.resids, float)
        self.rss = float(self.res.resid_ss)
        self.kappa = float(self.res.kappa) if estimator == "liml" else float("nan")
        self.df_r = (self.G - 1) if vce == "cluster" else (n - self.k)
        self._res_fs = None

    # ---- inference glue (Stata: small → t/F with N−k, or G−1 with clusters)
    def stat_dist(self) -> str:
        return "t" if self.small else "z"

    def pvalues(self) -> np.ndarray:
        z = self.b / self.se
        if self.small:
            return 2 * stats.t.sf(np.abs(z), self.df_r)
        return 2 * stats.norm.sf(np.abs(z))

    def ci(self, level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        q = stats.t.ppf(0.5 + level / 2, self.df_r) if self.small else stats.norm.ppf(0.5 + level / 2)
        return self.b - q * self.se, self.b + q * self.se

    def slope_index(self) -> List[int]:
        return [i for i, nm in enumerate(self.names) if nm != "Intercept"]

    def model_test(self) -> Dict[str, Any]:
        idx = self.slope_index()
        w = _wald(self.b[idx], self.V[np.ix_(idx, idx)])
        q = len(idx)
        if self.small:
            F = w / q
            return {"name": f"F({q}, {self.df_r})", "stat": F, "df1": q, "df2": self.df_r,
                    "p": float(stats.f.sf(F, q, self.df_r)), "kind": "F"}
        return {"name": f"Wald χ²({q})", "stat": w, "df1": q, "df2": None, "p": float(stats.chi2.sf(w, q)), "kind": "chi2"}

    def r2(self) -> Tuple[float, float, float]:
        n, k = self.n, self.k
        r2 = float(self.res.rsquared)              # linearmodels: centred with a constant, uncentred without (= Stata)
        adj = 1 - (1 - r2) * ((n - 1) if self.has_const else n) / (n - k)
        rmse = math.sqrt(self.rss / ((n - k) if self.small else n))
        return r2, adj, rmse

    # ---- first stage: linearmodels IV2SLS with the same VCE type, debiased (regress-style F as in estat firststage)
    def res_fs(self):
        if self._res_fs is None:
            from linearmodels.iv import IV2SLS
            kw: Dict[str, Any] = {"cov_type": _LM_COV[self.vce], "debiased": True}
            if self.vce == "cluster":
                kw["clusters"] = self.codes
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._res_fs = IV2SLS(self._dep, self._ex, self._en, self._in).fit(**kw)
        return self._res_fs

    def first_stage(self) -> Dict[str, Any]:
        n, L, l2 = self.n, self.L, self.l2
        diag = self.res_fs().first_stage.diagnostics
        df2 = (self.G - 1) if self.vce == "cluster" else (n - L)
        rows = []
        for j, nm in enumerate(self.names2):
            dr = diag.loc[f"x2_{j}"]
            F = float(dr["f.stat"])
            if not str(dr["f.dist"]).startswith("F"):
                F = F / l2                          # robust / cluster: Wald χ² → F = W/q (Stata)
            r2 = float(dr["rsquared"])
            rows.append({"var": nm, "r2": r2, "adj": 1 - (1 - r2) * ((n - 1) if self.has_const else n) / (n - L),
                         "pr2": float(dr["partial.rsquared"]), "F": F, "df1": l2, "df2": df2,
                         "p": float(stats.f.sf(F, l2, df2)), "shea_lm": float(dr["shea.rsquared"])})
        out: Dict[str, Any] = {"rows": rows}
        if self.m > 1:
            dn = (n - L + 1) if self.has_const else (n - L)
            for r in rows:
                r["shea"] = r["shea_lm"]
                r["shea_adj"] = 1 - (1 - r["shea"]) * (n - 1) / dn          # Stata estat firststage, all
        # minimum eigenvalue (Cragg–Donald) statistic — not in linearmodels (own code, Stata formula)
        Yh = _annih(self.X2, self.QX1) if self.k1 else self.X2
        Zh = _annih(self.Z2, self.QX1) if self.k1 else self.Z2
        QZh = _basis(Zh, "제외 도구변수")
        A = Yh.T @ _proj(Yh, QZh)
        V2 = _annih(self.X2, self.QZ)
        Svv = V2.T @ V2 / (n - L)
        ev = linalg.eigh(A, Svv, eigvals_only=True)
        out["mineig"] = float(np.min(ev) / l2)
        return out

    # ---- estat endogenous
    def endogeneity(self, tested: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        n = self.n
        tested = list(tested) if tested else list(self.names2)
        lm_vars = [f"x2_{self.names2.index(t)}" for t in tested]
        sub = lm_vars if len(tested) < self.m else None
        nt = len(tested)
        if self.estimator == "liml":
            return []                                     # Stata: estat endogenous not available after LIML
        if self.estimator == "gmm":
            C = _stat(self.res.c_stat(sub))
            return [{"test": "GMM C (차분 Hansen) χ²", "stat": C, "df": f"{nt}", "p": float(stats.chi2.sf(C, nt)) if C is not None else float("nan")}]
        if self.vce == "unadjusted":
            D, WH = self.res.durbin(sub), self.res.wu_hausman(sub)
            df2 = int(WH.df_denom)
            return [{"test": "Durbin (score) χ²", "stat": float(D.stat), "df": f"{nt}", "p": float(D.pval)},
                    {"test": "Wu–Hausman F", "stat": float(WH.stat), "df": f"{nt}, {df2}", "p": float(WH.pval)}]
        if sub is not None:
            raise ValueError("내생성 검정: 강건·군집 표준오차에서는 내생변수 전체만 검정 가능 (일부 지정은 통상 표준오차에서 Durbin·Wu–Hausman)")
        out: List[Dict[str, Any]] = []
        rf = self.res_fs()
        if self.vce == "robust":
            S = rf.wooldridge_score
            out.append({"test": "강건 점수 χ² (Robust score)", "stat": float(S.stat), "df": f"{nt}", "p": float(S.pval)})
        Wr = rf.wooldridge_regression                     # debiased robust / cluster Wald χ² of the first-stage residuals
        F = float(Wr.stat) / nt
        df2 = (self.G - 1) if self.vce == "cluster" else (n - self.k - nt)
        out.append({"test": "강건 회귀 F (Robust regression)", "stat": F, "df": f"{nt}, {df2}", "p": float(stats.f.sf(F, nt, df2))})
        return out

    # ---- estat overid
    def overid(self) -> List[Dict[str, Any]]:
        n, L, k = self.n, self.L, self.k
        q = L - k
        if q <= 0:
            return []
        if self.estimator == "gmm":
            J = self.res.j_stat
            return [{"test": "Hansen J χ²", "stat": float(J.stat), "df": f"{q}", "p": float(J.pval)}]
        if self.estimator == "liml":
            ar = n * (self.kappa - 1)                     # Stata: AR = N(κ − 1); linearmodels reports N·ln κ
            bf = self.res.basmann_f
            return [{"test": "Anderson–Rubin χ²", "stat": ar, "df": f"{q}", "p": float(stats.chi2.sf(ar, q))},
                    {"test": "Basmann F", "stat": float(bf.stat), "df": f"{q}, {n - L}", "p": float(bf.pval)}]
        if self.vce == "unadjusted":
            sg, bm = self.res.sargan, self.res.basmann
            return [{"test": "Sargan (score) χ²", "stat": float(sg.stat), "df": f"{q}", "p": float(sg.pval)},
                    {"test": "Basmann χ²", "stat": float(bm.stat), "df": f"{q}", "p": float(bm.pval)}]
        if self.vce == "cluster":
            return []                                     # Stata: not available after 2SLS with cluster–robust VCE
        W = self.res.wooldridge_overid
        return [{"test": "강건 점수 χ² (Score)", "stat": float(W.stat), "df": f"{q}", "p": float(W.pval)}]

    # ---- first-stage regressions (Stata `first`)
    def first_coefs(self) -> List[Tuple[str, List[str], np.ndarray, np.ndarray, np.ndarray, int]]:
        ind = self.res_fs().first_stage.individual
        zn = {f"x1_{i}": nm for i, nm in enumerate(self.names1)} | {f"z2_{i}": nm for i, nm in enumerate(self.znames)}
        out = []
        df_r = (self.G - 1) if self.vce == "cluster" else (self.n - self.L)
        for j, nm in enumerate(self.names2):
            r = ind[f"x2_{j}"]
            idx = [c for c in r.params.index if c != "x1_const"]
            order = [c for c in idx if zn.get(c) != "Intercept"] + [c for c in idx if zn.get(c) == "Intercept"]
            out.append((nm, [zn[c] for c in order], r.params[order].to_numpy(float), r.std_errors[order].to_numpy(float),
                        r.tstats[order].to_numpy(float), df_r))
        return out


# Stock & Yogo (2005) critical values for the minimum eigenvalue statistic (Tables 5.1, 5.2, 5.4), as used by Stata's
# estat firststage: p endogenous regressors, k2 excluded instruments |
#   2SLS relative bias 5% 10% 20% 30% | 2SLS size of nominal 5% Wald test 10% 15% 20% 25% | LIML size 10% 15% 20% 25%
_STOCK_YOGO_TXT = """
1  1 | . . . . | 16.38 8.96 6.66 5.53 | 16.38 8.96 6.66 5.53
1  2 | . . . . | 19.93 11.59 8.75 7.25 | 8.68 5.33 4.42 3.92
1  3 | 13.91 9.08 6.46 5.39 | 22.30 12.83 9.54 7.80 | 6.46 4.36 3.69 3.32
1  4 | 16.85 10.27 6.71 5.34 | 24.58 13.96 10.26 8.31 | 5.44 3.87 3.30 2.98
1  5 | 18.37 10.83 6.77 5.25 | 26.87 15.09 10.98 8.84 | 4.84 3.56 3.05 2.77
1  6 | 19.28 11.12 6.76 5.15 | 29.18 16.23 11.72 9.38 | 4.45 3.34 2.87 2.61
1  7 | 19.86 11.29 6.73 5.07 | 31.50 17.38 12.48 9.93 | 4.18 3.18 2.73 2.49
1  8 | 20.25 11.39 6.69 4.99 | 33.84 18.54 13.24 10.50 | 3.97 3.04 2.63 2.39
1  9 | 20.53 11.46 6.65 4.92 | 36.19 19.71 14.01 11.07 | 3.81 2.93 2.54 2.32
1 10 | 20.74 11.49 6.61 4.86 | 38.54 20.88 14.78 11.65 | 3.68 2.84 2.46 2.25
1 11 | 20.90 11.51 6.56 4.80 | 40.90 22.06 15.56 12.23 | 3.58 2.76 2.40 2.19
1 12 | 21.01 11.52 6.53 4.75 | 43.27 23.24 16.35 12.82 | 3.50 2.69 2.34 2.14
1 13 | 21.10 11.52 6.49 4.71 | 45.64 24.42 17.14 13.41 | 3.42 2.63 2.29 2.10
1 14 | 21.18 11.52 6.45 4.67 | 48.01 25.61 17.93 14.00 | 3.36 2.57 2.25 2.06
1 15 | 21.23 11.51 6.42 4.63 | 50.39 26.80 18.72 14.60 | 3.31 2.52 2.21 2.03
1 16 | 21.28 11.50 6.39 4.59 | 52.77 27.99 19.51 15.19 | 3.27 2.48 2.18 2.00
1 17 | 21.31 11.49 6.36 4.56 | 55.15 29.19 20.31 15.79 | 3.24 2.44 2.14 1.97
1 18 | 21.34 11.48 6.33 4.53 | 57.53 30.38 21.10 16.39 | 3.20 2.41 2.11 1.94
1 19 | 21.36 11.46 6.31 4.51 | 59.92 31.58 21.90 16.99 | 3.18 2.37 2.09 1.92
1 20 | 21.38 11.45 6.28 4.48 | 62.30 32.77 22.70 17.60 | 3.21 2.34 2.06 1.90
1 21 | 21.39 11.44 6.26 4.46 | 64.69 33.97 23.50 18.20 | 3.39 2.32 2.04 1.88
1 22 | 21.40 11.42 6.24 4.43 | 67.07 35.17 24.30 18.80 | 3.57 2.29 2.02 1.86
1 23 | 21.41 11.41 6.22 4.41 | 69.46 36.37 25.10 19.41 | 3.68 2.27 2.00 1.84
1 24 | 21.41 11.40 6.20 4.39 | 71.85 37.57 25.90 20.01 | 3.75 2.25 1.98 1.83
1 25 | 21.42 11.38 6.18 4.37 | 74.24 38.77 26.71 20.61 | 3.79 2.24 1.96 1.81
1 26 | 21.42 11.37 6.16 4.35 | 76.62 39.97 27.51 21.22 | 3.82 2.22 1.95 1.80
1 27 | 21.42 11.36 6.14 4.34 | 79.01 41.17 28.31 21.83 | 3.85 2.21 1.93 1.78
1 28 | 21.42 11.34 6.13 4.32 | 81.40 42.37 29.12 22.43 | 3.86 2.20 1.92 1.77
1 29 | 21.42 11.33 6.11 4.31 | 83.79 43.57 29.92 23.04 | 3.87 2.19 1.90 1.76
1 30 | 21.42 11.32 6.09 4.29 | 86.17 44.78 30.72 23.65 | 3.88 2.18 1.89 1.75
2  2 | . . . . | 7.03 4.58 3.95 3.63 | 7.03 4.58 3.95 3.63
2  3 | . . . . | 13.43 8.18 6.40 5.45 | 5.44 3.81 3.32 3.09
2  4 | 11.04 7.56 5.57 4.73 | 16.87 9.93 7.54 6.28 | 4.72 3.39 2.99 2.79
2  5 | 13.97 8.78 5.91 4.79 | 19.45 11.22 8.38 6.89 | 4.32 3.13 2.78 2.60
2  6 | 15.72 9.48 6.08 4.78 | 21.68 12.33 9.10 7.42 | 4.06 2.95 2.63 2.46
2  7 | 16.88 9.92 6.16 4.76 | 23.72 13.34 9.77 7.91 | 3.90 2.83 2.52 2.35
2  8 | 17.70 10.22 6.20 4.73 | 25.64 14.31 10.41 8.39 | 3.78 2.73 2.43 2.27
2  9 | 18.30 10.43 6.22 4.69 | 27.51 15.24 11.03 8.85 | 3.70 2.66 2.36 2.20
2 10 | 18.76 10.58 6.23 4.66 | 29.32 16.16 11.65 9.31 | 3.64 2.60 2.30 2.14
2 11 | 19.12 10.69 6.23 4.62 | 31.11 17.06 12.25 9.77 | 3.60 2.55 2.25 2.09
2 12 | 19.40 10.78 6.22 4.59 | 32.88 17.95 12.86 10.22 | 3.58 2.52 2.21 2.05
2 13 | 19.64 10.84 6.21 4.56 | 34.62 18.84 13.45 10.68 | 3.56 2.48 2.17 2.02
2 14 | 19.83 10.89 6.20 4.53 | 36.36 19.72 14.05 11.13 | 3.55 2.46 2.14 1.99
2 15 | 19.98 10.93 6.19 4.50 | 38.08 20.60 14.65 11.58 | 3.54 2.44 2.11 1.96
2 16 | 20.12 10.96 6.17 4.48 | 39.80 21.48 15.24 12.03 | 3.55 2.42 2.09 1.93
2 17 | 20.23 10.99 6.16 4.45 | 41.51 22.35 15.83 12.49 | 3.55 2.41 2.07 1.91
2 18 | 20.33 11.00 6.14 4.43 | 43.22 23.22 16.42 12.94 | 3.56 2.40 2.05 1.89
2 19 | 20.41 11.02 6.13 4.41 | 44.92 24.09 17.02 13.39 | 3.57 2.39 2.03 1.87
2 20 | 20.48 11.03 6.11 4.39 | 46.62 24.96 17.61 13.84 | 3.58 2.38 2.02 1.86
2 21 | 20.54 11.04 6.10 4.37 | 48.31 25.82 18.20 14.29 | 3.59 2.38 2.01 1.84
2 22 | 20.60 11.05 6.08 4.35 | 50.01 26.69 18.79 14.74 | 3.60 2.37 1.99 1.83
2 23 | 20.65 11.05 6.07 4.33 | 51.70 27.56 19.38 15.19 | 3.62 2.37 1.98 1.81
2 24 | 20.69 11.05 6.06 4.32 | 53.39 28.42 19.97 15.64 | 3.64 2.37 1.98 1.80
2 25 | 20.73 11.06 6.05 4.30 | 55.07 29.29 20.56 16.10 | 3.65 2.37 1.97 1.79
2 26 | 20.76 11.06 6.03 4.29 | 56.76 30.15 21.15 16.55 | 3.67 2.38 1.96 1.78
2 27 | 20.79 11.06 6.02 4.27 | 58.45 31.02 21.74 17.00 | 3.74 2.38 1.96 1.77
2 28 | 20.82 11.05 6.01 4.26 | 60.13 31.88 22.33 17.45 | 3.87 2.38 1.95 1.77
2 29 | 20.84 11.05 6.00 4.24 | 61.82 32.74 22.92 17.90 | 4.02 2.39 1.95 1.76
2 30 | 20.86 11.05 5.99 4.23 | 63.51 33.61 23.51 18.35 | 4.12 2.39 1.95 1.75
3  3 | . . . . | . . . . | . . . .
3  4 | . . . . | . . . . | . . . .
3  5 | 9.53 6.61 4.99 4.30 | . . . . | . . . .
3  6 | 12.20 7.77 5.35 4.40 | . . . . | . . . .
3  7 | 13.95 8.50 5.56 4.44 | . . . . | . . . .
3  8 | 15.18 9.01 5.69 4.46 | . . . . | . . . .
3  9 | 16.10 9.37 5.78 4.46 | . . . . | . . . .
3 10 | 16.80 9.64 5.83 4.45 | . . . . | . . . .
3 11 | 17.35 9.85 5.87 4.44 | . . . . | . . . .
3 12 | 17.80 10.01 5.90 4.42 | . . . . | . . . .
3 13 | 18.17 10.14 5.92 4.41 | . . . . | . . . .
3 14 | 18.47 10.25 5.93 4.39 | . . . . | . . . .
3 15 | 18.73 10.33 5.94 4.37 | . . . . | . . . .
3 16 | 18.94 10.41 5.94 4.36 | . . . . | . . . .
3 17 | 19.13 10.47 5.94 4.34 | . . . . | . . . .
3 18 | 19.29 10.52 5.94 4.32 | . . . . | . . . .
3 19 | 19.44 10.56 5.94 4.31 | . . . . | . . . .
3 20 | 19.56 10.60 5.93 4.29 | . . . . | . . . .
3 21 | 19.67 10.63 5.93 4.28 | . . . . | . . . .
3 22 | 19.77 10.65 5.92 4.27 | . . . . | . . . .
3 23 | 19.86 10.68 5.92 4.25 | . . . . | . . . .
3 24 | 19.94 10.70 5.91 4.24 | . . . . | . . . .
3 25 | 20.01 10.71 5.90 4.23 | . . . . | . . . .
3 26 | 20.07 10.73 5.90 4.21 | . . . . | . . . .
3 27 | 20.13 10.74 5.89 4.20 | . . . . | . . . .
3 28 | 20.18 10.75 5.88 4.19 | . . . . | . . . .
3 29 | 20.23 10.76 5.88 4.18 | . . . . | . . . .
3 30 | 20.27 10.77 5.87 4.17 | . . . . | . . . .
"""


def _parse_stock_yogo() -> Dict[Tuple[int, int], List[List[Optional[float]]]]:
    out: Dict[Tuple[int, int], List[List[Optional[float]]]] = {}
    for line in _STOCK_YOGO_TXT.strip().splitlines():
        head, *groups = line.split("|")
        p_, k_ = (int(x) for x in head.split())
        out[(p_, k_)] = [[None if v == "." else float(v) for v in g.split()] for g in groups]
    return out


STOCK_YOGO = _parse_stock_yogo()


def _fmt_p(p: float) -> str:
    return "< .001" if p < 0.001 else f"= {p:.3f}"


def _coef_table(name: str, title: str, names: Sequence[str], b, se, p, lo, hi, dist: str, level: float) -> ResultsTable:
    t = ResultsTable(name=name, title=title)
    t.add_column("term", "변수", "text")
    t.add_column("b", "계수", "number")
    t.add_column("se", "표준오차", "number")
    t.add_column("stat", dist, "number")
    t.add_column("p", "p", "number", "pvalue")
    t.add_column("lo", f"{level * 100:g}% 하한", "number", super_title="신뢰구간")
    t.add_column("hi", f"{level * 100:g}% 상한", "number", super_title="신뢰구간")
    for i, nm in enumerate(names):
        t.add_row({"term": nm, "b": float(b[i]), "se": float(se[i]), "stat": float(b[i] / se[i]) if se[i] > 0 else float("nan"),
                   "p": float(p[i]), "lo": float(lo[i]), "hi": float(hi[i])})
    t.mark_coefficients(estimate="b", se="se", p="p", stat="stat", ci=("lo", "hi"), label="term")
    return t


def _test_table(name: str, title: str, rows: List[Dict[str, Any]]) -> ResultsTable:
    t = ResultsTable(name=name, title=title)
    t.add_column("test", "검정", "text")
    t.add_column("stat", "통계량", "number")
    t.add_column("df", "자유도", "text")
    t.add_column("p", "p", "number", "pvalue")
    for r in rows:
        t.add_row({k: r[k] for k in ("test", "stat", "df", "p")})
    return t


def _prepare(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[pd.DataFrame, int]:
    sub = df[list(dict.fromkeys(cols))].copy()
    n0 = len(sub)
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    return sub, n0 - len(sub)


# ---------------------------------------------------------------- ivReg
class IVRegression(BaseAnalysis):
    """Stata ivregress 2sls | liml | gmm with estat firststage / endogenous / overid."""

    def __init__(self, df: pd.DataFrame, params: Dict[str, Any]):
        p = dict(params or {})
        dep = _as_single(_opt(p, "dep", "y", "depVar"))
        exog = _as_list(_opt(p, "exog", "covs", "exogenous", "indeps", default=[]))
        endog = _as_list(_opt(p, "endog", "endogenous", "instrumented", default=[]))
        inst = _as_list(_opt(p, "instruments", "inst", "excluded", "iv", default=[]))
        est = resolve_choice("추정법(estimator)", _opt(p, "estimator", "method", default="2sls"), ESTIMATORS, _EST_ALIASES)
        wm = resolve_choice("가중행렬(wmatrix)", _opt(p, "wmatrix", "wMatrix", "weight_matrix", default="robust"),
                            VCE_TYPES, _VCE_ALIASES)
        vraw = _opt(p, "covType", "cov_type", "vce", default="default")
        if str(vraw).strip().lower() in ("default", "기본"):
            vce = wm if est == "gmm" else "unadjusted"
        else:
            vce = resolve_choice("표준오차(covType)", vraw, VCE_TYPES, _VCE_ALIASES)
        clus = _as_single(_opt(p, "clusterVar", "cluster_var", "cluster", default=None))
        small = _flag(p, "small", default=False)
        const = _flag(p, "constant", "intercept", default=True) and not _flag(p, "noconstant", "noConstant", default=False)
        center = _flag(p, "center", default=False)
        level = float(_opt(p, "ciWidth", "ci_width", "level", default=95))
        level = level / 100 if level > 1 else level
        do_first = _flag(p, "firstStage", "first_stage", default=True)
        first_coef = _flag(p, "first", "firstCoefs", "first_coefs", default=False)
        do_endog = _flag(p, "endogTest", "endog_test", "estatEndogenous", default=True)
        do_overid = _flag(p, "overidTest", "overid_test", "estatOverid", default=True)
        endog_vars = _as_list(_opt(p, "endogTestVars", "endog_test_vars", default=[]))
        if not dep:
            raise ValueError("도구변수 회귀: 종속변수(dep) 지정 필요")
        if not endog:
            raise ValueError("도구변수 회귀: 내생변수(endog) 1개 이상 지정 필요")
        if not inst:
            raise ValueError("도구변수 회귀: 제외 도구변수(instruments) 1개 이상 지정 필요")
        need_cluster = vce == "cluster" or (est == "gmm" and wm == "cluster")
        if need_cluster and not clus:
            raise ValueError("군집 강건 표준오차: 군집 변수(clusterVar) 지정 필요")
        _check_columns(df, "도구변수 회귀", {"종속변수": dep, "외생변수": exog, "내생변수": endog, "도구변수": inst,
                                         "군집 변수": clus if need_cluster else None})
        overlap = (set(endog) & set(exog)) | (set(inst) & (set(exog) | set(endog))) | ({dep} & (set(exog) | set(endog) | set(inst)))
        if overlap:
            raise ValueError(f"도구변수 회귀: 변수 역할 중복 — {', '.join(sorted(overlap))}")
        bad_t = [v for v in endog_vars if v not in endog]
        if bad_t:
            raise ValueError(f"내생성 검정 대상: 내생변수가 아닌 변수 — {', '.join(bad_t)}")
        cols = [dep] + endog + exog + inst + ([clus] if need_cluster else [])
        _numeric(df, [dep] + endog + exog + inst, "도구변수 회귀")
        data, dropped = _prepare(df, cols)
        n = len(data)
        X1 = data[exog].to_numpy(float) if exog else np.zeros((n, 0))
        names1 = list(exog)
        if const:
            X1 = np.column_stack([X1, np.ones(n)])
            names1.append("Intercept")
        fit = IVFit(data[dep].to_numpy(float), data[endog].to_numpy(float), X1, data[inst].to_numpy(float),
                    endog, names1, inst, estimator=est, vce=vce,
                    clusters=data[clus].to_numpy() if need_cluster else None,
                    small=small, wmatrix=wm, center=center, has_const=const)
        options = {"dep": dep, "exog": exog, "endog": endog, "instruments": inst, "estimator": est, "covType": vce,
                   "clusterVar": clus if need_cluster else None, "wmatrix": wm if est == "gmm" else None,
                   "small": small, "constant": const, "center": center if est == "gmm" else False,
                   "ciWidth": level * 100, "firstStage": do_first, "first": first_coef, "endogTest": do_endog,
                   "overidTest": do_overid, "endogTestVars": endog_vars}
        super().__init__(name="ivReg", title=f"도구변수 회귀 — {EST_LABEL[est]} (ivregress {est})", dataset=df, options=options)
        self.fit = fit
        self._build(fit, data, dropped, level)

    def _build(self, f: IVFit, data: pd.DataFrame, dropped: int, level: float) -> None:
        o = self.options
        est, vce = o["estimator"], o["covType"]
        r2, adj, rmse = f.r2()
        mt = f.model_test()
        # model summary
        info = ResultsTable(name="ivModelFit", title="모형 요약 (Model summary)")
        info.add_column("stat", "항목", "text")
        info.add_column("value", "값", "number")
        info.add_column("df", "자유도", "text")
        info.add_column("p", "p", "number", "pvalue")
        info.add_row({"stat": "관측치 수 (N)", "value": f.n, "df": "", "p": "", "_rowfmt": "integer"})
        if vce == "cluster":
            info.add_row({"stat": f"군집 수 ({o['clusterVar']})", "value": f.G, "df": "", "p": "", "_rowfmt": "integer"})
        info.add_row({"stat": mt["name"].split("(")[0].strip(), "value": mt["stat"],
                      "df": f"{mt['df1']}" + (f", {mt['df2']}" if mt["df2"] is not None else ""), "p": mt["p"]})
        info.add_row({"stat": "R²" if f.has_const else "R² (비중심)", "value": r2 if r2 >= 0 else float("nan"), "df": "", "p": ""})
        if o["small"]:
            info.add_row({"stat": "수정 R²", "value": adj if r2 >= 0 else float("nan"), "df": "", "p": ""})
        info.add_row({"stat": "Root MSE", "value": rmse, "df": "", "p": ""})
        if est == "liml":
            info.add_row({"stat": "LIML κ", "value": f.kappa, "df": "", "p": ""})
        info.add_note(f"추정법: {EST_LABEL[est]}; 표준오차: {VCE_LABEL[vce]}" + (f" — 군집 변수 {o['clusterVar']}" if vce == "cluster" else "")
                      + ("; 소표본 보정(small): t·F 분포, 자유도 N−k" if o["small"] else "; 대표본 기준: z·χ² 분포 (Stata 기본)") + ".")
        if est == "gmm":
            info.add_note(f"GMM 가중행렬: {VCE_LABEL[o['wmatrix']]}" + (" (적률 중심화)" if o["center"] else "")
                          + ", 1단계 2SLS 잔차로 추정. 분산: " + ("Ŝ = W⁻¹인 최적 GMM 분산 N{X'ZWZ'X}⁻¹ (Stata vce(unadjusted))." if vce == "unadjusted"
                          else "최종 잔차로 다시 계산한 Ŝ의 샌드위치 공식 (Stata ivregress gmm 방식)."))
        info.add_note("계산: linearmodels " + {"2sls": "IV2SLS", "liml": "IVLIML", "gmm": "IVGMM (2단계)"}[est]
                      + (" (debiased = Stata small)" if o["small"] else "") + ".")
        if vce == "cluster":
            info.add_note("군집 표준오차 배율: " + ("(N−1)/(N−k) × G/(G−1) (small)" if o["small"]
                          else "없음 (Stata ivregress 기본: small 미지정 시 보정 없음)") + (", t·F 자유도 G−1" if o["small"] else "") + ".")
        elif vce == "robust" and o["small"]:
            info.add_note("강건 표준오차 배율: N/(N−k).")
        if r2 < 0:
            info.add_note("2SLS 계열 R²가 음수(잔차제곱합 > 총제곱합)여서 표시하지 않음 (Stata와 같은 처리). 도구변수 추정에서 R²는 적합도 지표로서 의미가 제한적.")
        if dropped:
            info.add_note(f"결측·무한값으로 제외한 관측치 {dropped}개.")
        self.add_table(info)
        # coefficients
        lo, hi = f.ci(level)
        ct = _coef_table("ivCoef", "계수 추정치 (Coefficients)", f.names, f.b, f.se, f.pvalues(), lo, hi, f.stat_dist(), level)
        ct.add_note(f"내생변수(Instrumented): {', '.join(f.names2)}.")
        ct.add_note(f"도구변수(Instruments): {', '.join([n for n in f.names1 if n != 'Intercept'] + f.znames)}"
                    + (" (상수항 포함)" if "Intercept" in f.names1 else "") + ".")
        self.add_table(ct)
        self.model_stats = {"N": f.n, "R2": r2 if r2 >= 0 else None}
        if o["small"]:
            self.model_stats["R2_adj"] = adj if r2 >= 0 else None
            self.model_stats["F"] = (mt["stat"], mt["p"])
        else:
            self.model_stats["Chi2"] = (mt["stat"], mt["p"])
        if vce == "cluster":
            self.model_stats["Groups"] = f.G
        # first stage
        fs = f.first_stage() if (o["firstStage"] or o["first"]) else None
        if fs and o["firstStage"]:
            t = ResultsTable(name="ivFirstStage", title="1단계 회귀 요약 (estat firststage)")
            t.add_column("var", "내생변수", "text")
            t.add_column("r2", "R²", "number", "zto")
            t.add_column("adj", "수정 R²", "number", "zto")
            t.add_column("pr2", "부분 R²", "number", "zto")
            if f.m > 1:
                t.add_column("shea", "Shea 부분 R²", "number", "zto")
                t.add_column("shea_adj", "Shea 수정 부분 R²", "number", "zto")
            t.add_column("F", "F", "number")
            t.add_column("dfs", "자유도", "text")
            t.add_column("p", "p", "number", "pvalue")
            for r in fs["rows"]:
                row = {"var": r["var"], "r2": r["r2"], "adj": r["adj"], "pr2": r["pr2"], "F": r["F"],
                       "dfs": f"{r['df1']}, {r['df2']}", "p": r["p"]}
                if f.m > 1:
                    row["shea"], row["shea_adj"] = r["shea"], r["shea_adj"]
                t.add_row(row)
            t.add_note("F: 제외 도구변수 계수가 모두 0이라는 가설의 검정" + ("" if vce == "unadjusted" else
                       f" ({'군집' if vce == 'cluster' else '이분산'} 강건 Wald 검정)") + ". 부분 R²: 포함 외생변수 효과를 제거한 뒤 제외 도구변수의 설명력.")
            if f.m > 1:
                t.add_note("Shea 부분 R²: 내생변수가 여럿일 때 다른 내생변수와의 상관을 고려한 도구변수 설명력 (Shea 1997).")
            self.add_table(t)
            w = ResultsTable(name="ivWeakId", title="약한 도구변수 검정 (Minimum eigenvalue statistic)")
            w.add_column("stat", "항목", "text")
            w.add_column("value", "값", "number")
            w.add_row({"stat": "최소 고유값 통계량 (Cragg–Donald)", "value": fs["mineig"]})
            sy = STOCK_YOGO.get((f.m, f.l2))
            labels = (("2SLS 상대 편의", ("5%", "10%", "20%", "30%")), ("2SLS 명목 5% Wald 검정 크기", ("10%", "15%", "20%", "25%")),
                      ("LIML 명목 5% Wald 검정 크기", ("10%", "15%", "20%", "25%")))
            shown = 0
            if sy:
                for (lab, lv), vals in zip(labels, sy):
                    for l_, v_ in zip(lv, vals):
                        if v_ is not None:
                            w.add_row({"stat": f"Stock–Yogo 임계값: {lab} {l_}", "value": v_})
                            shown += 1
            if shown:
                w.add_note(f"임계값: Stock & Yogo (2005) 표 5.1·5.2·5.4 (내생변수 {f.m}개, 제외 도구변수 {f.l2}개; Stata estat firststage와 같은 표). "
                           "귀무가설: 도구변수가 약함. 통계량이 임계값보다 크면 해당 수준(상대 편의 또는 Wald 검정 실제 크기)을 넘는 약한 도구변수 가설 기각."
                           + (" 상대 편의 값은 제외 도구변수가 내생변수보다 2개 이상 많을 때만 정의." if sy and all(v is None for v in sy[0]) else ""))
            else:
                w.add_note(f"Stock & Yogo (2005) 표에 내생변수 {f.m}개, 제외 도구변수 {f.l2}개의 임계값 없음 (표 범위: 내생변수 1–3개, 도구변수 30개 이하).")
            if vce != "unadjusted":
                w.add_note("최소 고유값 통계량과 Stock–Yogo 임계값은 동분산 오차 가정에 근거 (Stata는 강건 표준오차일 때 forcenonrobust 옵션에서만 표시). 강건 판단은 1단계 강건 F 참조.")
            if f.m == 1:
                w.add_note("내생변수 1개: 최소 고유값 통계량 = 동분산 1단계 F.")
            self.add_table(w)
        if fs is not None and o["first"]:
            self._first_coef_tables(f, level)
        # endogeneity
        if o["endogTest"]:
            try:
                rows, endog_err = f.endogeneity(o["endogTestVars"] or None), ""
            except ValueError as ex:
                rows, endog_err = [], str(ex)
            tested = o["endogTestVars"] or f.names2
            t = _test_table("ivEndog", "내생성 검정 (estat endogenous)", rows)
            if endog_err:
                t.add_note(endog_err)
            t.add_note(f"귀무가설: {', '.join(tested)} 외생적 (OLS 일치).")
            if endog_err:
                pass
            elif est == "liml":
                t.add_note("LIML 추정 후에는 내생성 검정 미제공 (Stata estat endogenous와 같음) — 2SLS 또는 GMM으로 추정하여 검정.")
            elif est == "gmm":
                t.add_note("C 통계량: 검정 대상을 외생변수로 둔 모형과 원 모형의 Hansen J 차이. 두 J 모두 외생 가정 모형의 적률 공분산 S 사용 (Hayashi 2000).")
            elif vce == "cluster":
                t.add_note("군집 강건 표준오차: 강건 회귀 F만 제공 (점수 검정은 군집 구조에 적용 불가, Stata와 같음). 1단계 잔차를 추가한 OLS에서 잔차 계수의 군집 강건 Wald 검정, 자유도 G−1.")
            elif vce != "unadjusted":
                t.add_note("강건 점수 검정: Wooldridge (1995). 강건 회귀 F: 1단계 잔차를 추가한 OLS에서 잔차 계수의 강건 Wald 검정.")
            else:
                t.add_note("Durbin·Wu–Hausman: 동분산 가정. 원 모형과 검정 대상을 외생으로 둔 모형의 잔차 사영 차이로 계산 (Stata estat endogenous 공식).")
            self.add_table(t)
        if o["overidTest"]:
            rows = f.overid()
            t = _test_table("ivOverid", "과대식별 검정 (estat overid)", rows)
            if not rows and f.L - f.k <= 0:
                t.add_note(f"정확식별(제외 도구변수 {f.l2}개 = 내생변수 {f.m}개): 과대식별 검정 불가.")
            elif not rows:
                t.add_note("2SLS·군집 강건 표준오차: 과대식별 강건 검정 미제공 (Stata와 같음) — GMM(가중행렬 cluster)의 Hansen J 사용.")
            else:
                t.add_note("귀무가설: 모든 도구변수가 오차항과 무상관 (과대식별 제약 성립).")
                if est == "liml":
                    t.add_note("Anderson–Rubin χ² = N(κ − 1), Basmann F = (κ − 1)(N − k_Z)/m (Stata 공식).")
                if est == "liml" and vce != "unadjusted":
                    t.add_note("Anderson–Rubin·Basmann F는 동분산 오차 가정에 근거 (Stata는 강건 표준오차일 때 forcenonrobust 지정 시에만 표시).")
                if est == "2sls" and vce != "unadjusted":
                    t.add_note("Wooldridge (1995) 강건 점수 검정.")
            self.add_table(t)
        # narrative and code
        b0, s0 = f.b[0], f.se[0]
        pz = f.pvalues()[0]
        self.apa_narrative = (f"{EST_LABEL[est]}으로 {o['dep']}을(를) 추정한 결과(N = {f.n}), 내생변수 {f.names2[0]}의 계수는 "
                              f"{b0:.3f} ({f.stat_dist()} = {b0 / s0:.2f}, p {_fmt_p(pz)})로 나타났다.")
        if fs and o["firstStage"]:
            r = fs["rows"][0]
            self.apa_narrative += f" 1단계 F({r['df1']}, {r['df2']}) = {r['F']:.2f}로 제외 도구변수의 설명력을 확인하였다."
        args = {k: v for k, v in o.items() if v not in (None, "", [], False) or k in ("constant",)}
        if args.get("constant") is True:
            args.pop("constant")
        self.syntax_python = "sess.run('ivReg', " + ", ".join(f"{k}={v!r}" for k, v in args.items()) + ")"
        self.syntax_r = "\n".join(r_code("ivReg", o) or [])

    def _first_coef_tables(self, f: IVFit, level: float) -> None:
        for j, (nm, names, g, se, tv, df_r) in enumerate(f.first_coefs()):
            pv = 2 * stats.t.sf(np.abs(tv), df_r)
            qv = stats.t.ppf(0.5 + level / 2, df_r)
            t = _coef_table(f"ivFirst_{j + 1}", f"1단계 회귀: {nm} (first)", names, g, se, pv, g - qv * se, g + qv * se, "t", level)
            t.add_note(f"OLS (linearmodels first_stage), 표준오차 {VCE_LABEL[f.vce]} (Stata regress 방식 소표본 보정, 자유도 {df_r}).")
            self.add_table(t)


# ---------------------------------------------------------------- panelIV (xtivreg)
PANEL_MODELS = ("fe", "re", "fd", "be")
_PANEL_ALIASES = {"within": "fe", "fixed": "fe", "random": "re", "g2sls": "re", "ec2sls": "re", "between": "be",
                  "firstdiff": "fd", "first_difference": "fd", "difference": "fd"}
PANEL_LABEL = {"fe": "고정효과 (within 2SLS)", "re": "확률효과 (G2SLS)", "fd": "1차 차분 2SLS", "be": "집단 간 (between 2SLS)"}


def _panel_time_codes(t: pd.Series) -> Tuple[np.ndarray, bool]:
    """Integer time index; returns (codes, regular) where regular means equally spaced numeric values."""
    if pd.api.types.is_numeric_dtype(t):
        v = t.to_numpy(float)
        u = np.unique(v)
        if len(u) > 1:
            step = np.min(np.diff(u))
            codes = np.round((v - u[0]) / step).astype(int)
            return codes, bool(np.allclose((v - u[0]) / step, codes))
        return np.zeros(len(v), int), True
    codes, _ = pd.factorize(t, sort=True)
    return codes.astype(int), False


def _corr2(a: np.ndarray, d: np.ndarray) -> float:
    return float(np.corrcoef(a, d)[0, 1] ** 2) if np.std(a) > 0 and np.std(d) > 0 else float("nan")


class PanelIV(BaseAnalysis):
    """Stata xtivreg: fe (within), fd (first difference), be (between), re (G2SLS / EC2SLS, Swamy–Arora or Baltagi–Chang)."""

    def __init__(self, df: pd.DataFrame, params: Dict[str, Any]):
        p = dict(params or {})
        dep = _as_single(_opt(p, "dep", "y"))
        exog = _as_list(_opt(p, "exog", "covs", "exogenous", default=[]))
        endog = _as_list(_opt(p, "endog", "endogenous", default=[]))
        inst = _as_list(_opt(p, "instruments", "inst", "excluded", default=[]))
        ent = _as_single(_opt(p, "entity", "idVar", "panelVar", "id", "entity_col"))
        tim = _as_single(_opt(p, "time", "timeVar", "time_col", "period"))
        model = resolve_choice("모형(model)", _opt(p, "model", "modelType", "model_type", default="fe"), PANEL_MODELS, _PANEL_ALIASES)
        raw_model = str(_opt(p, "model", "modelType", "model_type", default="fe")).strip().lower()
        re_method = resolve_choice("확률효과 추정법(reMethod)",
                                   _opt(p, "reMethod", "re_method", "instMethod", default="ec2sls" if raw_model == "ec2sls" else "g2sls"),
                                   ("g2sls", "ec2sls"), {"bvk": "g2sls", "baltagi": "ec2sls"})
        re_var = resolve_choice("분산 성분(reVar)", _opt(p, "reVar", "re_var", "varComponents", default="sa"), ("sa", "bc"),
                                {"swar": "sa", "swamy-arora": "sa", "nosa": "bc", "baltagi-chang": "bc"})
        if _flag(p, "nosa", default=False):
            re_var = "bc"
        small = _flag(p, "small", default=False)
        vce = resolve_choice("표준오차(covType)", _opt(p, "covType", "cov_type", "vce", default="conventional"),
                             ("conventional", "cluster"), {"unadjusted": "conventional", "robust": "cluster",
                                                            "clustered": "cluster", "ols": "conventional"})
        level = float(_opt(p, "ciWidth", "ci_width", "level", default=95))
        level = level / 100 if level > 1 else level
        do_first = _flag(p, "firstStage", "first_stage", "first", default=True)
        if not dep or not endog or not inst:
            raise ValueError("패널 도구변수 회귀: 종속변수(dep)·내생변수(endog)·제외 도구변수(instruments) 지정 필요")
        if not ent:
            raise ValueError("패널 도구변수 회귀: 패널 식별 변수(entity) 지정 필요 (Stata xtset의 panelvar)")
        if model == "fd" and not tim:
            raise ValueError("패널 도구변수 회귀(fd): 시간 변수(time) 지정 필요")
        if vce == "cluster" and model in ("re", "be"):
            raise ValueError("패널 도구변수 회귀: 패널 군집 표준오차는 fe·fd 모형에서만 제공 (Stata xtivreg와 같음)")
        _check_columns(df, "패널 도구변수 회귀", {"종속변수": dep, "외생변수": exog, "내생변수": endog, "도구변수": inst,
                                           "패널 변수": ent, "시간 변수": tim})
        overlap = (set(endog) & set(exog)) | (set(inst) & (set(exog) | set(endog)))
        if overlap:
            raise ValueError(f"패널 도구변수 회귀: 변수 역할 중복 — {', '.join(sorted(overlap))}")
        _numeric(df, [dep] + endog + exog + inst, "패널 도구변수 회귀")
        cols = [ent] + ([tim] if tim else []) + [dep] + endog + exog + inst
        data, dropped = _prepare(df, cols)
        if tim and data.duplicated([ent, tim]).any():
            raise ValueError("패널 도구변수 회귀: 같은 패널·시점에 관측치 중복 (Stata xtset 오류와 같은 조건)")
        data = data.sort_values([ent, tim] if tim else [ent]).reset_index(drop=True)
        options = {"dep": dep, "exog": exog, "endog": endog, "instruments": inst, "entity": ent, "time": tim,
                   "model": model, "reMethod": re_method if model == "re" else None, "reVar": re_var if model == "re" else None,
                   "covType": vce, "small": small, "ciWidth": level * 100, "firstStage": do_first}
        lab = PANEL_LABEL[model] if not (model == "re" and re_method == "ec2sls") else "확률효과 (EC2SLS)"
        opt_txt = model + (", ec2sls" if model == "re" and re_method == "ec2sls" else "") + (", nosa" if model == "re" and re_var == "bc" else "")
        super().__init__(name="panelIV", title=f"패널 도구변수 회귀 — {lab} (xtivreg, {opt_txt})", dataset=df, options=options)
        self._run(data, dropped, level)

    @staticmethod
    def _means(A: np.ndarray, codes: np.ndarray, G: int) -> np.ndarray:
        s = np.zeros((G, A.shape[1]))
        np.add.at(s, codes, A)
        cnt = np.bincount(codes, minlength=G).astype(float)
        return s / cnt[:, None]

    def _run(self, data: pd.DataFrame, dropped: int, level: float) -> None:
        o = self.options
        dep, exog, endog, inst, model = o["dep"], o["exog"], o["endog"], o["instruments"], o["model"]
        gcodes, guniq = pd.factorize(data[o["entity"]], sort=True)
        n_g = len(guniq)
        Ti = np.bincount(gcodes, minlength=n_g)
        y = data[dep].to_numpy(float)
        X2 = data[endog].to_numpy(float)
        X1 = data[exog].to_numpy(float) if exog else np.zeros((len(y), 0))
        Z2 = data[inst].to_numpy(float)
        N = len(y)
        gm = lambda A: self._means(A.reshape(N, -1), gcodes, n_g)                 # panel means (n_g rows)
        rep = lambda A: gm(A)[gcodes]                                              # panel means repeated
        notes: List[str] = []
        if dropped:
            notes.append(f"결측·무한값으로 제외한 관측치 {dropped}개.")
        C = "Intercept"
        extra: Dict[str, Any] = {}
        r2s: Dict[str, float] = {}
        clusters = data[o["entity"]].to_numpy() if o["covType"] == "cluster" else None
        Xreg = np.column_stack([X2, X1])                                           # regressors without the constant
        stata_wald = None

        if model == "fe":
            dm = lambda A: A.reshape(N, -1) - rep(A) + A.reshape(N, -1).mean(axis=0)   # Stata: w_it − w̄_i + w̄
            fit = IVFit(dm(y)[:, 0], dm(X2), np.column_stack([dm(X1), np.ones(N)]), dm(Z2), endog, exog + [C], inst,
                        vce="cluster" if clusters is not None else "unadjusted", clusters=clusters, small=True)
            K = fit.k                                                              # including the constant
            df_r = N - n_g - K + 1
            V = fit.V * ((N - K) / df_r if clusters is None else 1.0)              # Stata: (N−K)/(N−n−K+1)·V_IV
            b = fit.b
            sigma_e = math.sqrt(fit.rss / df_r)
            ui = gm(y)[:, 0] - gm(Xreg) @ b[:-1] - b[-1]
            extra.update(sigma_u=float(np.std(ui, ddof=1)) if n_g > 1 else float("nan"), sigma_e=sigma_e)
            xb = Xreg @ b[:-1]
            extra["corr"] = float(np.corrcoef(ui[gcodes], xb)[0, 1])
            r2s = {"within": float(fit.res.rsquared), "between": _corr2(gm(xb[:, None])[:, 0], gm(y)[:, 0]), "overall": _corr2(xb, y)}
            # F test that all u_i = 0 (Wooldridge 1990): second-stage regressions, instruments incl. panel dummies
            Qw = _basis(np.column_stack([X1 - rep(X1), Z2 - rep(Z2)]), "도구변수")
            X2h = rep(X2) + _proj(X2 - rep(X2), Qw)
            def _rss(Xm):
                bb = np.linalg.lstsq(Xm, y, rcond=None)[0]
                e = y - Xm @ bb
                return float(e @ e)
            ssr_r = _rss(np.column_stack([X2h, X1, np.ones(N)]))
            Xd = np.column_stack([X2h - rep(X2h), X1 - rep(X1)])
            yd = y - rep(y)[:, 0]
            bd = np.linalg.lstsq(Xd, yd, rcond=None)[0]
            ssr_u = float(((yd - Xd @ bd) ** 2).sum())
            Ff = ((ssr_r - ssr_u) / (n_g - 1)) / (fit.rss / df_r)
            extra["F_f"] = (Ff, n_g - 1, df_r, float(stats.f.sf(Ff, n_g - 1, df_r)))
            if clusters is None:
                stata_wald = _wald(b, V)                                           # Stata's displayed Wald: all K coefficients
            n_used, dfr_small = N, df_r if clusters is None else (len(np.unique(clusters)) - 1)
            names, first_src = fit.names, fit
            notes.append(f"집단 내 변환 w_it − w̄_i + w̄ 후 2SLS (linearmodels IV2SLS); 분산 = (N−K)/(N−n−K+1)·V_IV, σ_e² = RSS/{df_r} (Stata xtivreg fe)."
                         if clusters is None else "집단 내 변환 후 2SLS (linearmodels IV2SLS), 패널 군집 강건 분산 (N−1)/(N−K)·G/(G−1) (Stata xtivreg fe vce(robust)).")
            notes.append("상수항: 패널 효과 평균. σ_u: û_i = ȳ_i − x̄_i b − 상수의 패널 간 표준편차. R² 집단 내: 평균 제거 회귀의 R²; 집단 간·전체: 적합값과의 상관 제곱.")
        elif model == "fd":
            tcodes, regular = _panel_time_codes(data[o["time"]])
            prev_ok = np.zeros(N, bool)
            prev_ok[1:] = (gcodes[1:] == gcodes[:-1]) & (tcodes[1:] - tcodes[:-1] == 1)
            if not regular:
                notes.append("시간 변수가 숫자형이 아니어서 정렬 순서로 시점 간격을 판단.")
            idx = np.where(prev_ok)[0]
            d = lambda A: A.reshape(N, -1)[idx] - A.reshape(N, -1)[idx - 1]
            nd = len(idx)
            clus_d = clusters[idx] if clusters is not None else None
            fit = IVFit(d(y)[:, 0], d(X2), np.column_stack([d(X1), np.ones(nd)]), d(Z2), ["D." + v for v in endog],
                        ["D." + v for v in exog] + [C], ["D." + v for v in inst],
                        vce="cluster" if clus_d is not None else "unadjusted", clusters=clus_d, small=True)
            V, b = fit.V, fit.b
            df_r = nd - fit.k
            extra.update(sigma_e=math.sqrt(fit.rss / df_r))
            gaps = int(((gcodes[1:] == gcodes[:-1]) & (tcodes[1:] - tcodes[:-1] > 1)).sum())
            if gaps:
                notes.append(f"시점 간격(gap) {gaps}곳: 해당 차분은 결측 처리 (Stata D. 연산자와 같음).")
            # R² within / overall on the level data of the observations used (Stata xtivreg fd); between R² not reproduced
            xb_u, y_u, g_u = (Xreg @ b[:-1])[idx], y[idx], gcodes[idx]
            rep_u = lambda v: pd.Series(v).groupby(g_u).transform("mean").to_numpy()
            r2s = {"within": _corr2(xb_u - rep_u(xb_u), y_u - rep_u(y_u)), "overall": _corr2(xb_u, y_u)}
            notes.append("R² (집단 내·전체): 차분에 쓰인 관측치의 수준 자료에서 적합값과의 상관 제곱 (Stata 정의). Stata가 표시하는 fd 집단 간 R²는 계산 방식이 문서와 달라 표시하지 않음.")
            n_used, dfr_small = nd, df_r if clus_d is None else (len(np.unique(clus_d)) - 1)
            names, first_src = fit.names, fit
            notes.append(f"1차 차분 자료(관측치 {nd}개)에 상수항 포함 2SLS (linearmodels IV2SLS). σ_e = √(RSS/(N_D − k))"
                         + (" (Stata의 vce(robust) 결과 e(sigma_e)는 군집 자유도 G−1로 나눈 값이어서 다름)." if clus_d is not None else "."))
            Ti = np.bincount(gcodes[idx], minlength=n_g)
        elif model == "be":
            fit = IVFit(gm(y)[:, 0], gm(X2), np.column_stack([gm(X1), np.ones(n_g)]), gm(Z2), endog, exog + [C], inst, small=True)
            V, b = fit.V, fit.b
            df_r = n_g - fit.k
            xb = Xreg @ b[:-1]
            r2s = {"within": _corr2(xb - rep(xb)[:, 0], y - rep(y)[:, 0]), "between": float(fit.res.rsquared), "overall": _corr2(xb, y)}
            extra.update(rmse_b=math.sqrt(fit.rss / df_r))
            n_used, dfr_small = N, df_r
            names, first_src = fit.names, fit
            notes.append(f"패널 평균 {n_g}개에 대한 2SLS (linearmodels IV2SLS, 가중치 없음), σ² = RSS/(n − k). R² 집단 간: 평균 회귀의 R².")
        else:
            res = self._re(y, X2, X1, Z2, gcodes, n_g, Ti, endog, exog, inst, o["reMethod"], o["reVar"])
            fit, V, b = res["fit"], res["fit"].V, res["fit"].b
            n_used, dfr_small = N, N - fit.k
            extra.update(sigma_u=res["sigma_u"], sigma_e=res["sigma_e"], theta=res["theta"])
            xb = Xreg @ b[:-1]
            r2s = {"within": _corr2(xb - rep(xb)[:, 0], y - rep(y)[:, 0]), "between": _corr2(gm(xb[:, None])[:, 0], gm(y)[:, 0]),
                   "overall": _corr2(xb, y)}
            names, first_src = fit.names, (None if o["reMethod"] == "ec2sls" else fit)
            notes.extend(res["notes"])
        se = np.sqrt(np.clip(np.diag(V), 0, None))
        z = b / se
        if o["small"]:
            pv = 2 * stats.t.sf(np.abs(z), dfr_small)
            qv = stats.t.ppf(0.5 + level / 2, dfr_small)
        else:
            pv = 2 * stats.norm.sf(np.abs(z))
            qv = stats.norm.ppf(0.5 + level / 2)
        idx_s = [i for i, nm in enumerate(names) if nm != C]
        w = _wald(b[idx_s], V[np.ix_(idx_s, idx_s)])
        q = len(idx_s)
        info = ResultsTable(name="xtivModelFit", title="모형 요약 (Model summary)")
        info.add_column("stat", "항목", "text")
        info.add_column("value", "값", "number")
        info.add_column("df", "자유도", "text")
        info.add_column("p", "p", "number", "pvalue")
        info.add_row({"stat": "관측치 수 (N)", "value": n_used, "df": "", "p": "", "_rowfmt": "integer"})
        info.add_row({"stat": "패널 수 (n)", "value": n_g, "df": "", "p": "", "_rowfmt": "integer"})
        info.add_row({"stat": "패널당 관측치 (최소·평균·최대)", "value": float(Ti.mean()),
                      "df": f"{int(Ti.min())} · {Ti.mean():.1f} · {int(Ti.max())}", "p": ""})
        if o["small"]:
            F = w / q
            info.add_row({"stat": "F", "value": F, "df": f"{q}, {dfr_small}", "p": float(stats.f.sf(F, q, dfr_small))})
            self.model_stats = {"F": (F, float(stats.f.sf(F, q, dfr_small)))}
        else:
            info.add_row({"stat": "Wald χ²", "value": w, "df": f"{q}", "p": float(stats.chi2.sf(w, q))})
            self.model_stats = {"Chi2": (w, float(stats.chi2.sf(w, q)))}
        if stata_wald is not None and not o["small"]:
            info.add_row({"stat": "Wald χ² (Stata 표시값: 상수 포함)", "value": stata_wald, "df": f"{q}", "p": ""})
        for key, lab in (("within", "R² (집단 내)"), ("between", "R² (집단 간)"), ("overall", "R² (전체)")):
            if key in r2s:
                info.add_row({"stat": lab, "value": r2s[key], "df": "", "p": ""})
        for key, lab in (("sigma_u", "σ_u"), ("sigma_e", "σ_e"), ("rmse_b", "Root MSE (집단 간 회귀)")):
            if key in extra and extra[key] is not None:
                info.add_row({"stat": lab, "value": extra[key], "df": "", "p": ""})
        if "sigma_u" in extra and "sigma_e" in extra:
            su, se_ = extra["sigma_u"], extra["sigma_e"]
            rho = su ** 2 / (su ** 2 + se_ ** 2) if (su ** 2 + se_ ** 2) > 0 else float("nan")
            info.add_row({"stat": "ρ (패널 효과 분산 비율)", "value": rho, "df": "", "p": ""})
            extra["rho"] = rho
        if "corr" in extra:
            info.add_row({"stat": "corr(u_i, Xb)", "value": extra["corr"], "df": "", "p": ""})
        if "F_f" in extra:
            Ff, d1, d2, pf = extra["F_f"]
            info.add_row({"stat": "F 검정: 모든 u_i = 0", "value": Ff, "df": f"{d1}, {d2}", "p": pf})
        if "theta" in extra:
            th = extra["theta"]
            info.add_row({"stat": "θ (최소·중앙·최대)" if np.ptp(th) > 1e-12 else "θ", "value": float(np.median(th)),
                          "df": (f"{th.min():.4f} · {np.median(th):.4f} · {th.max():.4f}" if np.ptp(th) > 1e-12 else ""), "p": ""})
        info.add_note(("표준오차: 패널 군집 강건" if o["covType"] == "cluster" else "표준오차: 통상")
                      + ("; small: t·F 분포." if o["small"] else "; z·χ² 분포 (Stata xtivreg 기본)."))
        if stata_wald is not None and not o["small"]:
            info.add_note("Stata xtivreg fe가 표시하는 Wald χ²는 상수항까지 포함한 결합검정값(자유도 표기는 기울기 수)이어서 별도 행으로 표시. 모형 검정에는 기울기 Wald χ² 사용.")
        if "F_f" in extra:
            info.add_note("u_i = 0 F 검정: Wooldridge (1990) 2SLS F — 패널 더미를 포함한 도구변수로 만든 2단계 회귀의 잔차제곱합 차이 (Stata xtivreg fe).")
        for n_ in notes:
            info.add_note(n_)
        self.add_table(info)
        ct = _coef_table("xtivCoef", "계수 추정치 (Coefficients)", names, b, se, pv, b - qv * se, b + qv * se,
                         "t" if o["small"] else "z", level)
        ct.add_note(f"내생변수(Instrumented): {', '.join(endog)}. 도구변수(Instruments): {', '.join(exog + inst)}.")
        self.add_table(ct)
        self.model_stats.update({"N": n_used, "Groups": n_g})
        for key, ms in (("within", "R2_within"), ("between", "R2_between"), ("overall", "R2_overall")):
            if key in r2s:
                self.model_stats[ms] = r2s[key]
        if "rho" in extra:
            self.model_stats["rho"] = extra["rho"]
        if o["firstStage"] and first_src is None:
            info.add_note("EC2SLS: 도구변수 집합이 모든 회귀변수에 공통이어서 1단계 요약 표는 생략.")
        elif o["firstStage"]:
            fs = first_src.first_stage()
            t = ResultsTable(name="xtivFirstStage", title="1단계 회귀 요약 (변환 자료 기준)")
            t.add_column("var", "내생변수", "text")
            t.add_column("pr2", "부분 R²", "number", "zto")
            t.add_column("F", "F", "number")
            t.add_column("dfs", "자유도", "text")
            t.add_column("p", "p", "number", "pvalue")
            for r in fs["rows"]:
                t.add_row({"var": r["var"], "pr2": r["pr2"], "F": r["F"], "dfs": f"{r['df1']}, {r['df2']}", "p": r["p"]})
            t.add_note({"fe": "집단 평균 제거 자료", "fd": "1차 차분 자료", "be": "패널 평균 자료",
                        "re": "θ 변환(준-평균 제거) 자료"}[model] + "에서 내생변수를 전체 도구변수에 회귀 (linearmodels first_stage). "
                       "자유도는 변환 자료 기준 (fe의 패널 효과 자유도 미반영). Stata xtivreg first의 1단계 회귀표와 계수는 같고 요약 통계는 별도 제공 아님.")
            self.add_table(t)
        self.apa_narrative = (f"패널 도구변수 회귀({PANEL_LABEL[model]}, 패널 {n_g}개, 관측치 {n_used}개)에서 내생변수 {names[0]}의 계수는 "
                              f"{b[0]:.3f} ({'t' if o['small'] else 'z'} = {z[0]:.2f}, p {_fmt_p(pv[0])})로 추정되었다.")
        args = {k: v for k, v in o.items() if v not in (None, "", [], False)}
        self.syntax_python = "sess.run('panelIV', " + ", ".join(f"{k}={v!r}" for k, v in args.items()) + ")"
        self.syntax_r = "\n".join(r_code("panelIV", o) or [])

    def _re(self, y, X2, X1, Z2, gcodes, n_g, Ti, endog, exog, inst, method, var_method) -> Dict[str, Any]:
        """Variance components (Stata xtivreg re; own code) and the GLS-transformed 2SLS (linearmodels IV2SLS)."""
        N = len(y)
        gm = lambda A: self._means(A.reshape(N, -1), gcodes, n_g)
        rep = lambda A: gm(A)[gcodes]
        dm = lambda A: A.reshape(N, -1) - rep(A)
        K = X2.shape[1] + X1.shape[1] + 1                                         # Stata K includes the constant
        fw = IVFit(dm(y)[:, 0], dm(X2), dm(X1), dm(Z2), endog, exog, inst, small=True, has_const=False)
        fb = IVFit(gm(y)[:, 0], gm(X2), np.column_stack([gm(X1), np.ones(n_g)]), gm(Z2), endog, exog + ["Intercept"], inst, small=True)
        Zb = np.column_stack([gm(X2), gm(X1), np.ones(n_g)])
        ub = gm(y)[:, 0] - Zb @ fb.b
        sTu = float((Ti * ub ** 2).sum())
        if var_method == "sa":
            s2e = fw.rss / (N - n_g - K + 1)
            A_ = (Zb * Ti[:, None]).T @ Zb
            B_ = (Zb * (Ti.astype(float) ** 2)[:, None]).T @ Zb
            r = float(np.trace(np.linalg.solve(A_, B_)))
            s2u_raw = (sTu - (n_g - K) * s2e) / (N - r)
            vtxt = ("분산 성분 (Swamy–Arora, Stata 기본): σ_e² = 집단 내 2SLS 잔차제곱합/(N − n − K + 1); "
                    "σ_u² = {Σ T_i ū_i² − (n − K)σ_e²}/(N − r), r = tr{(Z̄'Z̄)⁻¹Z̄'Z_μZ_μ'Z̄} (불균형 패널 보정).")
        else:
            s2e = fw.rss / (N - n_g)
            s2u_raw = (sTu - n_g * s2e) / N
            vtxt = "분산 성분 (Baltagi–Chang 2000, Stata nosa): σ_e² = 집단 내 2SLS 잔차제곱합/(N − n); σ_u² = (Σ T_i ū_i² − nσ_e²)/N."
        s2u = max(0.0, s2u_raw)
        theta_g = 1 - np.sqrt(s2e / (Ti * s2u + s2e))
        th = theta_g[gcodes][:, None]
        qd = lambda A: A.reshape(N, -1) - th * rep(A)
        X1c = np.column_stack([X1, np.ones(N)])
        if method == "g2sls":
            fit = IVFit(qd(y)[:, 0], qd(X2), qd(X1c), qd(Z2), endog, exog + ["Intercept"], inst, small=True)
            mtxt = "G2SLS (Balestra & Varadharajan-Krishnakumar 1987, Stata 기본): y·X·도구변수를 모두 θ 변환 후 2SLS (linearmodels IV2SLS)."
        else:
            # Stata ec2sls instrument set (reproduced to 1e-11): panel means X̄_i, the transformed constant (1 − θ_i) and the
            # within transform with the grand mean added, X − X̄_i + X̿, for X = [included exogenous, excluded instruments];
            # the transformed regressors do not instrument themselves
            Xa = np.column_stack([X1, Z2])
            Xw = Xa - rep(Xa) + Xa.mean(axis=0)
            one_s = qd(np.ones(N))
            cand = _independent(one_s, np.column_stack([rep(Xa), Xw]))
            fit = IVFit(qd(y)[:, 0], np.column_stack([qd(X2), qd(X1)]), one_s, cand, endog + exog, ["Intercept"],
                        [f"ec{i}" for i in range(cand.shape[1])], small=True)
            mtxt = ("EC2SLS (Baltagi 1981, Stata ec2sls): θ 변환한 y·X를 도구변수 [패널 평균 X̄_i, (1 − θ_i), X − X̄_i + X̿]로 2SLS "
                    "(linearmodels IV2SLS; 도구변수 구성은 Stata 결과로 확인).")
        notes = [vtxt, mtxt, "분산: 변환 자료 2SLS의 통상 분산, s² = 변환 잔차제곱합/(N − K) (Stata xtivreg re와 같음)."]
        if s2u_raw <= 0:
            notes.append("σ_u² 추정치가 0 이하 → 0으로 두어 θ = 0 (통합 2SLS와 같음).")
        return {"fit": fit, "sigma_u": math.sqrt(s2u), "sigma_e": math.sqrt(s2e), "theta": theta_g, "notes": notes}



# ---------------------------------------------------------------- Stata / R code export
def _vce_stata(p: Dict[str, Any]) -> str:
    v = p.get("covType") or "unadjusted"
    if v == "cluster":
        return f"vce(cluster {p.get('clusterVar')})"
    return f"vce({v})"


def stata_code(t: str, p: Dict[str, Any]) -> Optional[List[str]]:
    if t == "ivReg":
        est = p.get("estimator", "2sls")
        exog = " ".join(_as_list(p.get("exog")))
        rhs = f"({' '.join(_as_list(p.get('endog')))} = {' '.join(_as_list(p.get('instruments')))})"
        opts = [_vce_stata(p)]
        if est == "gmm" and p.get("wmatrix"):
            wm = p["wmatrix"]
            opts.append(f"wmatrix({'cluster ' + str(p.get('clusterVar')) if wm == 'cluster' else wm})")
            if p.get("center"):
                opts.append("center")
        if p.get("small"):
            opts.append("small")
        if p.get("constant") is False:
            opts.append("noconstant")
        if p.get("first"):
            opts.append("first")
        lv = p.get("ciWidth")
        if lv and float(lv) != 95:
            opts.append(f"level({float(lv):g})")
        lines = [f"ivregress {est} {_as_single(p.get('dep'))} {exog + ' ' if exog else ''}{rhs}, {' '.join(opts)}"]
        if p.get("firstStage", True):
            lines.append("estat firststage")
        if p.get("endogTest", True):
            ev = " ".join(_as_list(p.get("endogTestVars")))
            lines.append("estat endogenous" + (f" {ev}" if ev else ""))
        if p.get("overidTest", True) and len(_as_list(p.get("instruments"))) > len(_as_list(p.get("endog"))):
            lines.append("estat overid")
        return lines
    if t == "panelIV":
        exog = " ".join(_as_list(p.get("exog")))
        rhs = f"({' '.join(_as_list(p.get('endog')))} = {' '.join(_as_list(p.get('instruments')))})"
        model = p.get("model", "fe")
        opts = [model] + (["ec2sls"] if model == "re" and p.get("reMethod") == "ec2sls" else []) \
            + (["nosa"] if model == "re" and p.get("reVar") == "bc" else [])
        if p.get("covType") == "cluster":
            opts.append("vce(robust)")          # xtivreg fe/fd: robust = clustered on the panel variable
        if p.get("small"):
            opts.append("small")
        if p.get("firstStage", True):
            opts.append("first")
        lines = [f"xtset {p.get('entity')}" + (f" {p.get('time')}" if p.get("time") else ""),
                 f"xtivreg {_as_single(p.get('dep'))} {exog + ' ' if exog else ''}{rhs}, {' '.join(opts)}"]
        return lines
    return None


def _rq(s: str) -> str:
    return '"' + str(s).replace('"', '\\"') + '"'


def r_code(t: str, p: Dict[str, Any]) -> Optional[List[str]]:
    if t == "ivReg":
        dep = _as_single(p.get("dep"))
        exog, endog, inst = _as_list(p.get("exog")), _as_list(p.get("endog")), _as_list(p.get("instruments"))
        c0 = "" if p.get("constant", True) is not False else " - 1"
        rhs1 = " + ".join(endog + exog) + c0
        rhs2 = " + ".join(exog + inst) + c0
        est = p.get("estimator", "2sls")
        lines = ["library(AER); library(sandwich); library(lmtest)"]
        if est != "2sls":
            lines.append(f"# {est.upper()}: AER::ivreg는 2SLS만 제공 — 아래는 2SLS 비교용 (LIML·GMM은 Stata ivregress {est} 또는 Python linearmodels 사용)")
        lines.append(f"fit <- ivreg({dep} ~ {rhs1} | {rhs2}, data = data)")
        v = p.get("covType") or "unadjusted"
        small = bool(p.get("small"))
        if v == "robust":
            vc = f'vcovHC(fit, type = "{"HC1" if small else "HC0"}")'
        elif v == "cluster":
            cl = p.get("clusterVar")
            vc = f'vcovCL(fit, cluster = ~{cl}, type = "{"HC1" if small else "HC0"}", cadjust = TRUE)'
        else:
            vc = "vcov(fit)" if small else "vcov(fit) * df.residual(fit) / nobs(fit)"
        lines.append(f"V <- {vc}")
        lines.append("coeftest(fit, vcov. = V" + ("" if small else ", df = Inf") + ")   # Stata: small 없으면 z 검정")
        if p.get("firstStage", True) or p.get("endogTest", True) or p.get("overidTest", True):
            lines.append("summary(fit, diagnostics = TRUE)   # 약한 도구변수 F, Wu-Hausman, Sargan (동분산 기준)")
        return lines
    if t == "panelIV":
        dep = _as_single(p.get("dep"))
        exog, endog, inst = _as_list(p.get("exog")), _as_list(p.get("endog")), _as_list(p.get("instruments"))
        model = {"fe": "within", "re": "random", "fd": "fd", "be": "between"}[p.get("model", "fe")]
        idx = f"c({_rq(p.get('entity'))}" + (f", {_rq(p.get('time'))})" if p.get("time") else ")")
        extra = ""
        if model == "random":
            extra = f', inst.method = "{"baltagi" if p.get("reMethod") == "ec2sls" else "bvk"}"'
        lines = ["library(plm)",
                 f"fit <- plm({dep} ~ {' + '.join(endog + exog)} | {' + '.join(exog + inst)}, data = data, index = {idx}, model = \"{model}\"{extra})",
                 "summary(fit)   # plm: t 검정·자유도 보정이 Stata xtivreg(z 검정)와 다를 수 있음"]
        if p.get("covType") == "cluster":
            lines.append('lmtest::coeftest(fit, vcov. = vcovHC(fit, cluster = "group", type = "HC1"))')
        return lines
    return None


# ---------------------------------------------------------------- dispatch, Stata input rules, extension entry
def dispatch(t: str, df: pd.DataFrame, params: Dict[str, Any]):
    if t == "ivReg":
        return IVRegression(df, params)
    if t == "panelIV":
        return PanelIV(df, params)
    return None


def _split_iv_varlist(body: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """'y x1 (x2 x3 = z1 z2) x4' -> (['y'], exog, endog, inst)."""
    import re
    m = re.search(r"\(([^()]*)\)", body)
    if not m or "=" not in m.group(1):
        raise ValueError("내생변수 = 도구변수 괄호 필요: (x2 = z1 z2)")
    left, right = m.group(1).split("=", 1)
    outside = (body[:m.start()] + " " + body[m.end():]).split()
    if not outside:
        raise ValueError("종속변수 필요")
    return outside[:1], outside[1:], left.split(), right.split()


def _tr_opts(opt_txt: str) -> Dict[str, str]:
    """Stata option string -> {name: value}; nested parentheses kept."""
    out: Dict[str, str] = {}
    i, s = 0, opt_txt or ""
    while i < len(s):
        while i < len(s) and s[i] in " \t,":
            i += 1
        j = i
        while j < len(s) and s[j] not in " \t,(":
            j += 1
        name = s[i:j].lower()
        if not name:
            break
        val = ""
        if j < len(s) and s[j] == "(":
            depth, k = 0, j
            while k < len(s):
                depth += {"(": 1, ")": -1}.get(s[k], 0)
                if depth == 0:
                    break
                k += 1
            val = s[j + 1:k].strip()
            j = k + 1
        out[name] = val
        i = j
    return out


def _vce_arg(v: str) -> Tuple[str, Optional[str]]:
    toks = v.split()
    if not toks:
        raise ValueError("vce() 비어 있음")
    head = toks[0].lower()
    if "unadjusted".startswith(head) and len(head) >= 2 or head in ("ols", "conventional"):
        return "unadjusted", None
    if "robust".startswith(head):
        return "robust", None
    if "cluster".startswith(head) and len(head) >= 2:
        if len(toks) != 2:
            raise ValueError("vce(cluster 변수) 형식 필요 (군집 변수 1개)")
        return "cluster", toks[1]
    raise ValueError(f"vce({v}) 미지원 (unadjusted·robust·cluster 변수 가능)")


def _ni(msg: str) -> str:
    return f"raise NotImplementedError({msg!r})"


def _call(aid: str, kw: Dict[str, Any]) -> str:
    return f"sess.run({aid!r}, " + ", ".join(f"{k}={v!r}" for k, v in kw.items()) + ")"


def transpile(line: str) -> Optional[str]:
    import re
    s = line.strip()
    m = re.match(r"^ivreg(?:ress)?\s+(2sls|liml|gmm)\s+([^,]+?)\s*(?:,\s*(.*))?$", s, re.IGNORECASE)
    if m:
        est, body, opt_txt = m.group(1).lower(), m.group(2), m.group(3) or ""
        if re.search(r"(^|\s|\()(i|c|l|f|d|s|ib\d*|o)\.\w|#", body, re.IGNORECASE):
            return _ni("요인변수·시계열 연산자 표기(i. c. L. D. #) 미지원: 변수를 만든 뒤 지정 — " + s)
        try:
            dep, exog, endog, inst = _split_iv_varlist(body)
        except ValueError as e:
            return _ni(f"ivregress: {e} — {s}")
        kw: Dict[str, Any] = {"dep": dep[0], "exog": exog, "endog": endog, "instruments": inst, "estimator": est}
        opts = _tr_opts(opt_txt)
        bad = []
        for k, v in opts.items():
            if k == "vce":
                try:
                    vt, cl = _vce_arg(v)
                except ValueError as e:
                    return _ni(f"ivregress: {e} — {s}")
                kw["covType"] = vt
                if cl:
                    kw["clusterVar"] = cl
            elif k in ("r", "rob", "robust"):
                kw["covType"] = "robust"
            elif k in ("cl", "clu", "clus", "clust", "cluster"):
                kw["covType"], kw["clusterVar"] = "cluster", v
            elif k == "small":
                kw["small"] = True
            elif k == "first":
                kw["first"] = True
            elif k in ("nocons", "nocon", "noconstant", "noconst"):
                kw["constant"] = False
            elif k in ("wmatrix", "wmat", "wm") and est == "gmm":
                try:
                    wt, cl = _vce_arg(v)
                except ValueError as e:
                    return _ni(f"ivregress wmatrix: {e} — {s}")
                kw["wmatrix"] = wt
                if cl:
                    kw.setdefault("clusterVar", cl)
            elif k == "center" and est == "gmm":
                kw["center"] = True
            elif k in ("level", "l"):
                kw["ciWidth"] = float(v)
            else:
                bad.append(k)
        if bad:
            return _ni("ivregress 옵션 미지원: " + ", ".join(bad) + " — " + s)
        return _call("ivReg", kw)
    m = re.match(r"^estat\s+(first(?:stage)?|endog(?:enous)?|overid)\b\s*([^,]*)(?:,\s*(.*))?$", s, re.IGNORECASE)
    if m:
        sub = m.group(1).lower()
        if m.group(3):
            return _ni("estat 옵션 미지원: " + m.group(3) + " — " + s)
        if sub.startswith("endog") and m.group(2).strip():
            vars_ = m.group(2).split()
            return ("sess.run('ivReg', **{**sess.history[-1].options, 'endogTestVars': " + repr(vars_) + "})"
                    "  # estat endogenous 변수 지정: 직전 ivReg를 검정 대상만 바꿔 다시 실행")
        tbl = {"first": "ivFirstStage", "endog": "ivEndog", "overid": "ivOverid"}[sub[:5] if sub.startswith("first") else sub[:5] if sub.startswith("endog") else "overid"]
        return (f"_t = sess.history[-1].get_table({tbl!r}); print(_t.to_string() if _t else "
                f"'estat {sub}: 직전 분석이 도구변수 회귀(ivReg)가 아님')")
    m = re.match(r"^xtivreg\s+([^,]+?)\s*(?:,\s*(.*))?$", s, re.IGNORECASE)
    if m:
        body, opt_txt = m.group(1), m.group(2) or ""
        if re.search(r"(^|\s|\()(i|c|l|f|d|s|ib\d*|o)\.\w|#", body, re.IGNORECASE):
            return _ni("요인변수·시계열 연산자 표기(i. c. L. D. #) 미지원: 변수를 만든 뒤 지정 — " + s)
        try:
            dep, exog, endog, inst = _split_iv_varlist(body)
        except ValueError as e:
            return _ni(f"xtivreg: {e} — {s}")
        kw = {"dep": dep[0], "exog": exog, "endog": endog, "instruments": inst}
        opts = _tr_opts(opt_txt)
        model, bad = None, []
        for k, v in opts.items():
            if k in ("fe", "re", "fd", "be"):
                if model and model != k:
                    return _ni("xtivreg: fe·re·fd·be 중 하나만 지정 — " + s)
                model = k
            elif k == "ec2sls":
                kw["reMethod"] = "ec2sls"
            elif k == "g2sls":
                kw["reMethod"] = "g2sls"
            elif k == "nosa":
                kw["reVar"] = "bc"
            elif k == "small":
                kw["small"] = True
            elif k == "first":
                kw["firstStage"] = True
            elif k == "vce":
                head = v.split()[0].lower() if v.split() else ""
                if head in ("conventional", "conv", "ols", "unadjusted"):
                    kw["covType"] = "conventional"
                elif head.startswith("r") or head.startswith("cl"):
                    kw["covType"] = "cluster"
                else:
                    bad.append(f"vce({v})")
            elif k in ("level", "l"):
                kw["ciWidth"] = float(v)
            else:
                bad.append(k)
        if bad:
            return _ni("xtivreg 옵션 미지원: " + ", ".join(bad) + " — " + s)
        kw["model"] = model or "re"
        if (kw.get("reMethod") or kw.get("reVar")) and kw["model"] != "re":
            return _ni("xtivreg: ec2sls·g2sls·nosa는 re 모형에서만 지정 — " + s)
        call = _call("panelIV", kw)[:-1]
        return (call + ", entity=getattr(sess, 'panel_spec', {}).get('idVar'), "
                "time=getattr(sess, 'panel_spec', {}).get('timeVar'))  # 패널·시간 변수: 앞의 xtset")
    return None


EXTENSION = {
    "catalog": [
        {"id": "ivReg", "group": "Regression", "title": "Instrumental Variables Regression (2SLS / LIML / GMM)",
         "module": "openstat", "icon": "analysis-regression.svg"},
        {"id": "panelIV", "group": "Linear Models", "title": "Panel IV Regression (xtivreg FE / RE / FD / BE)",
         "module": "openstat", "icon": "analysis-linearmodels.svg"},
    ],
    "aliases": {"ivregress": "ivReg", "ivreg": "ivReg", "iv2sls": "ivReg", "2sls": "ivReg", "tsls": "ivReg",
                "instrumentalVariables": "ivReg", "xtivreg": "panelIV", "panelIv": "panelIV"},
    "dispatch": dispatch,
    "stata": stata_code,
    "r": r_code,
    "transpile": transpile,
}
