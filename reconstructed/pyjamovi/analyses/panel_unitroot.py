"""
OpenStat Studio — panel unit-root tests (Stata xtunitroot) and panel cointegration tests (Stata xtcointtest).
License: MIT

No Python library computes these panel statistics (searched statsmodels 0.15, linearmodels 7.1, arch 8.0). Each statistic
follows the original paper and Stata [XT] xtunitroot / xtcointtest Methods and formulas; per-panel regressions are OLS
(numpy least squares); the per-panel unit-root tests of the Fisher-type test and all MacKinnon p-values come from arch.

Statistic (option test=)            computation: library call or published method                         Stata
panelUnitRoot
  llc   t_delta, t*_delta           Levin, Lin & Chu (2002) eqs (2)-(7), table 2 adjustments (linear        xtunitroot llc
                                    interpolation in T-tilde); ADF lags fixed or AIC/BIC/HQIC on the
                                    p_max sample; HAC long-run variance, lags LLC int(3.21 T^1/3) or
                                    Newey & West (1994)
  ht    rho, z                      Harris & Tzavalis (1999) theorem 1 moments (altt: T − 1)                xtunitroot ht
  breitung lambda, lambda robust    Breitung (2000); Breitung & Das (2005) robust form; prewhitening and    xtunitroot breitung
                                    Helmert-type detrending as in [XT] Methods and formulas
  ips   t-bar, t-tilde-bar,         Im, Pesaran & Shin (2003) table 1 (E, Var of t, t-tilde), table 2       xtunitroot ips
        Z-t-tilde-bar, W-t-bar      (fixed-N critical values), table 3 (E, Var of ADF t by T and lags)
  fisher P, Z, L*, Pm               Choi (2001), Maddala & Wu (1999); per panel arch.unitroot.ADF /         xtunitroot fisher
                                    PhillipsPerron, p-values arch mackinnonp (MacKinnon 1994); dfuller
                                    drift: t distribution; series with gaps: same regressions on the
                                    observations whose lags exist
  hadri z                           Hadri (2000) LM; robust (panel variances) and kernel() variants         xtunitroot hadri
panelCoint
  kao   modified DF t, DF t, ADF t, Kao (1999) eqs (9)-(12); long-run covariance Newey & West (1987),     xtcointtest kao
        unadjusted variants         bandwidth Newey & West (1994)
  pedroni panel v/rho/PP t/ADF t,   Pedroni (1999) table 1 statistics and table 2 adjustment terms         xtcointtest pedroni
        group rho/PP t/ADF t        (2-7 regressors), Pedroni (2004) terms for one regressor
  westerlund variance ratio         Westerlund (2005) VR_G (somepanels), VR_P (allpanels)                  xtcointtest westerlund

Conventions taken from Stata 19.5 where they decide the numbers (each one is stated in a result note):
- LLC table 2 value sigma*(T = 40, trend) = 0.971 as tabulated by Stata (the paper prints 0.871); with noconstant the
  information-criterion lag search uses regressions with a constant.
- IPS: t-tilde moments from the intercept column of table 1 also with trend; critical values are the table entries of
  the first N and T headings that are not smaller (no interpolation).
- Fisher L*: k = 3(5N+3)/(pi^2 N(5N+2)) (Choi 2001 prints 5N+4; that value is reported in a note); panels with p = 1 are
  left out of Z and L* as in Stata.
- Kao: u and eps are residuals of pooled AR(1) regressions without constant; the long-run covariance is the average of
  per-panel kernel estimates (demeaned, divisor T); the cointegrating regression uses t >= 2.
- Pedroni automatic lags: the Newey–West bandwidth of the first panel fixes the lag of all panels (option nwPerPanel
  selects it panel by panel); statistics scale by the longest panel's T; p-values 1 − Phi(|z|).
- Westerlund: adjustment terms recovered to 8 decimals from Stata output (journal table not available here).

Verified against Stata 19.5 (StataNow MP); reference values are hard-coded in tests/test_panel_unitroot.py.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseAnalysis, ResultsTable, resolve_choice
from .panel_module import _as_list, _as_single, _err, _flag, _opt, _split_opts, time_positions, xtset_lines

UR_TESTS = ("llc", "ht", "breitung", "ips", "fisher", "hadri")
UR_ALIASES = {"levinlin": "llc", "levin": "llc", "levin-lin-chu": "llc", "harris": "ht", "harris-tzavalis": "ht",
              "im-pesaran-shin": "ips", "ipshin": "ips", "madwu": "fisher", "choi": "fisher", "kpss": "hadri"}
CO_TESTS = ("kao", "pedroni", "westerlund")
KERNELS = ("bartlett", "parzen", "quadraticspectral")
KERNEL_ALIASES = {"ba": "bartlett", "bar": "bartlett", "nw": "bartlett", "nwest": "bartlett", "pa": "parzen",
                  "gallant": "parzen", "qs": "quadraticspectral", "quad": "quadraticspectral",
                  "quadratic": "quadraticspectral", "andrews": "quadraticspectral"}
KERNEL_LABEL = {"bartlett": "Bartlett", "parzen": "Parzen", "quadraticspectral": "Quadratic spectral"}

# ============================================================================ data preparation


class URPanel:
    """Panel arrays for one variable (or a dependent variable with regressors): listwise complete, sorted."""

    def __init__(self, df: pd.DataFrame, entity: str, time: str, cols: Sequence[str], label: str):
        if not entity or not time:
            miss = [n for n, v in (("개체 변수(entity)", entity), ("시점 변수(time)", time)) if not v]
            raise ValueError(f"{label}: 지정 필요 — {', '.join(miss)} (Stata 명령 입력 시 xtset 개체변수 시점변수 먼저 실행). "
                             f"사용 가능한 변수: {list(df.columns)}")
        need = [entity, time] + list(cols)
        absent = [c for c in dict.fromkeys(need) if c not in df.columns]
        if absent:
            raise ValueError(f"{label}: 데이터에 없는 변수 — {', '.join(absent)}. 사용 가능한 변수: {list(df.columns)}")
        for c in cols:
            if c in (entity, time):
                raise ValueError(f"{label}: 개체·시점 변수 '{c}'는 검정 변수로 지정할 수 없음")
        sub = df[list(dict.fromkeys(need))].copy()
        for c in cols:
            if not pd.api.types.is_numeric_dtype(sub[c]):
                conv = pd.to_numeric(sub[c], errors="coerce")
                if conv.notna().sum() < sub[c].notna().sum():
                    raise ValueError(f"{label}: '{c}'는 숫자형 변수 필요 (문자 값 포함)")
                sub[c] = conv
        sub = sub.dropna()
        if sub.empty:
            raise ValueError(f"{label}: 결측 제외 후 관측치 없음")
        tpos, self.time_note = time_positions(sub[time])
        sub["_tpos"] = tpos
        if sub.duplicated([entity, "_tpos"]).any():
            raise ValueError(f"{label}: 같은 개체·시점 관측치 중복 (Stata: repeated time values within panel)")
        sub = sub.sort_values([entity, "_tpos"], kind="mergesort").reset_index(drop=True)
        self.df = sub
        self.entity, self.time, self.cols = entity, time, list(cols)
        g, self.labels = pd.factorize(sub[entity], sort=True)
        self.g = g
        self.tpos = sub["_tpos"].to_numpy(np.int64)
        self.n = int(g.max()) + 1
        self.Ti = np.bincount(g, minlength=self.n)
        starts = np.r_[0, np.cumsum(self.Ti)[:-1]]
        self.slices = [slice(int(s), int(s + k)) for s, k in zip(starts, self.Ti)]
        self.gaps = any(np.any(np.diff(self.tpos[s]) != 1) for s in self.slices)
        first = np.array([self.tpos[s][0] for s in self.slices])
        self.balanced = bool(np.all(self.Ti == self.Ti[0]) and np.all(first == first[0]) and not self.gaps)
        self.N = int(len(sub))

    def values(self, col: str, demean: bool = False) -> np.ndarray:
        v = self.df[col].to_numpy(float)
        if demean:   # cross-sectional mean of each period removed (Levin, Lin & Chu 2002, sec. 5)
            m = pd.Series(v).groupby(self.tpos).transform("mean").to_numpy()
            v = v - m
        return v

    def series(self, col: str, demean: bool = False) -> List[np.ndarray]:
        v = self.values(col, demean)
        return [v[s] for s in self.slices]

    def require_balanced(self, name: str):
        if not self.balanced:
            raise ValueError(f"{name} 검정: 강한 균형 패널 필요 — 모든 개체의 관측 시점이 같아야 함 "
                             f"(Stata: {name} test requires strongly balanced data). 불균형 패널은 ips·fisher 사용")

    def require_no_gaps(self, name: str):
        if self.gaps:
            raise ValueError(f"{name} 검정: 시점 간격(gap)이 있는 패널 불가 (Stata: {name} test cannot have gaps in data)")


# ============================================================================ numerical helpers


def _ols(y: np.ndarray, X: np.ndarray) -> Dict[str, Any]:
    """OLS (numpy least squares): coefficients, residuals, rss, conventional standard errors, Gaussian log likelihood."""
    n, k = X.shape
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    rss = float(e @ e)
    xtx_inv = np.linalg.pinv(X.T @ X)
    s2 = rss / (n - k) if n > k else np.nan
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * s2, 0))
    ll = -0.5 * n * (1 + math.log(2 * math.pi) + math.log(rss / n)) if rss > 0 else np.inf
    return {"b": b, "e": e, "rss": rss, "se": se, "n": n, "k": k, "ll": ll, "s2": s2}


def _resid(y: np.ndarray, X: Optional[np.ndarray]) -> np.ndarray:
    if X is None or X.shape[1] == 0:
        return y.copy()
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b


def _lagmat(dy: np.ndarray, p: int) -> np.ndarray:
    """Columns L1 ... Lp of dy aligned with dy[p:] (first p rows dropped)."""
    n = len(dy)
    return np.column_stack([dy[p - j:n - j] for j in range(1, p + 1)]) if p > 0 else np.zeros((n - p, 0))


def _ic(fit: Dict[str, Any], crit: str) -> float:
    """Information criterion as Stata xtunitroot computes it: (-2 lnL + penalty * k) / n, k = parameters."""
    n, k, ll = fit["n"], fit["k"], fit["ll"]
    pen = {"aic": 2.0, "bic": math.log(n), "hqic": 2 * math.log(math.log(n))}[crit]
    return (-2 * ll + pen * k) / n


def kernel_weights(m: float, kernel: str, nlags: int) -> np.ndarray:
    """Weights K(j, m), j = 1..nlags, with z = j/(m+1) ([XT] xtunitroot Methods and formulas)."""
    j = np.arange(1, nlags + 1, dtype=float)
    z = j / (m + 1.0)
    if kernel == "bartlett":
        return np.where(z <= 1, 1 - z, 0.0)
    if kernel == "parzen":
        return np.where(z <= 0.5, 1 - 6 * z ** 2 + 6 * z ** 3, np.where(z <= 1, 2 * (1 - z) ** 3, 0.0))
    th = 6 * math.pi * z / 5
    return 3 * (np.sin(th) / th - np.cos(th)) / th ** 2


def _autocov_sums(w: np.ndarray, nlags: int) -> np.ndarray:
    """sum_t w_t w_{t-j}, j = 1..nlags (0 for j >= len(w))."""
    n = len(w)
    return np.array([float(w[j:] @ w[:n - j]) if j < n else 0.0 for j in range(1, nlags + 1)])


def nw_bandwidth_lags(w: np.ndarray, kernel: str) -> float:
    """Newey & West (1994) automatic lag selection, as in [R] ivregress Methods and formulas with z = h = 1;
    returns the number of lags m (bandwidth − 1)."""
    T = len(w)
    if kernel == "bartlett":
        q, cy, integer, n = 1, 1.1447, True, int(4 * (T / 100) ** (2 / 9))
    elif kernel == "parzen":
        q, cy, integer, n = 2, 2.6614, True, int(4 * (T / 100) ** (4 / 25))
    else:
        q, cy, integer, n = 2, 1.3221, False, int(4 * (T / 100) ** (2 / 25))
    sig = _autocov_sums(w, n)
    s0 = float(w @ w) + 2 * sig.sum()
    sq = 2 * float(np.sum(np.arange(1, n + 1, dtype=float) ** q * sig))
    p = 1 / (2 * q + 1)
    gamma = cy * ((sq / s0) ** 2) ** p
    if integer:
        hb = int(gamma * T ** p)
        return float(hb - 1 if hb < T else T - 2)
    hb = gamma * T ** p
    return float(hb - 1 if hb <= T else T - 1)


def lr_variance(w: np.ndarray, m: float, kernel: str, all_lags_qs: bool = True) -> float:
    """(1/n)(sum w² + 2 sum_j K(j, m) sum_t w_t w_{t-j}); Bartlett and Parzen use j = 1..int(m), the quadratic
    spectral kernel all lags (LLC long-run variance, [XT] xtunitroot Methods and formulas)."""
    n = len(w)
    nl = n if (kernel == "quadraticspectral" and all_lags_qs) else int(m)
    if nl <= 0:
        return float(w @ w) / n
    if kernel == "quadraticspectral" and all_lags_qs:
        z = np.arange(1, nl + 1) / (m + 1.0)
        x2 = 6 * math.pi * z / 5
        kw = (25 / (12 * (math.pi * z) ** 2)) * (np.sin(x2) / x2 - np.cos(x2))
    else:
        kw = kernel_weights(m, kernel, nl)
    return (float(w @ w) + 2 * float(kw @ _autocov_sums(w, nl))) / n


def _interp_rows(t: float, heads: Sequence[float], *cols: Sequence[float]) -> List[float]:
    """Linear interpolation in a table whose rows are indexed by T (below the first head: first row)."""
    out = []
    for c in cols:
        if t <= heads[0]:
            out.append(float(c[0]))
        elif t >= heads[-1]:
            out.append(float(c[-1]))
        else:
            out.append(float(np.interp(t, heads, c)))
    return out


# ============================================================================ Levin–Lin–Chu (2002)

# LLC (2002, table 2) mean and standard-deviation adjustments mu*, sigma* by T-tilde for models 1 (no deterministic
# terms), 2 (panel means) and 3 (panel means and trends), as tabulated in Stata xtunitroot. Stata's model-3 sigma* at
# T = 40 is 0.971; the published table prints 0.871 (the column is otherwise decreasing). Stata's value is used so
# that results match xtunitroot; the note of the result table states this.
_LLC_T = (25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 250)
_LLC_ADJ = {
    1: ((.004, .003, .002, .002, .001, .001, .001, .000, .000, .000, .000, .000, .000),
        (1.049, 1.035, 1.027, 1.021, 1.017, 1.014, 1.011, 1.008, 1.007, 1.006, 1.005, 1.001, 1.000)),
    2: ((-.554, -.546, -.541, -.537, -.533, -.531, -.527, -.524, -.521, -.520, -.518, -.509, -.500),
        (.919, .889, .867, .850, .837, .826, .810, .798, .789, .782, .776, .742, .707)),
    3: ((-.703, -.674, -.653, -.637, -.624, -.614, -.598, -.587, -.578, -.571, -.566, -.533, -.500),
        (1.003, .949, .906, .971, .842, .818, .780, .751, .728, .710, .695, .603, .500)),
}


def llc_adjustment(tt: float, model: int) -> Tuple[float, float]:
    mu, sd = _LLC_ADJ[model]
    if tt <= 25:
        return mu[0], sd[0]
    if tt > 250:
        return mu[-1], sd[-1]
    k = next(i for i, h in enumerate(_LLC_T) if tt <= h)
    w = (tt - _LLC_T[k - 1]) / (_LLC_T[k] - _LLC_T[k - 1])
    return (1 - w) * mu[k - 1] + w * mu[k], (1 - w) * sd[k - 1] + w * sd[k]


def _det(n: int, model: int, t0: int = 2) -> Optional[np.ndarray]:
    """Deterministic regressors for n observations starting at period t0 (1 = first period of the panel)."""
    if model == 1:
        return None
    c = np.ones((n, 1))
    if model == 2:
        return c
    return np.column_stack([np.arange(t0, t0 + n, dtype=float), c])


def _adf_select(y: np.ndarray, pmax: int, crit: str, model: int) -> int:
    """Lag order minimising the criterion over p = pmax..0 on the sample of the p_max regression (ties: larger p)."""
    dy = np.diff(y)
    ly = y[:-1]
    best, best_p = np.inf, pmax
    n_use = len(dy) - pmax
    for p in range(pmax, -1, -1):
        rows = slice(pmax, len(dy))
        L = _lagmat(dy, pmax)[:, :p]
        X = np.column_stack([ly[rows], L])
        d = _det(len(dy), model)
        if d is not None:
            X = np.column_stack([X, d[rows]])
        fit = _ols(dy[rows], X)
        if fit["n"] != n_use:
            raise RuntimeError("lag selection sample")
        v = _ic(fit, crit)
        if v < best:
            best, best_p = v, p
    return best_p


def _lag_spec(spec: Any, default: Optional[int], label: str, allow_ic: bool = True) -> Tuple[Optional[int], Optional[str]]:
    """lags(#) or lags(aic|bic|hqic #) -> (lags, criterion)."""
    if spec is None or spec == "" or spec == []:
        return default, None
    if isinstance(spec, (int, np.integer)) or (isinstance(spec, float) and float(spec).is_integer()):
        v = int(spec)
        if v < 0:
            raise ValueError(f"{label}: 시차(lags)는 0 이상 정수")
        return v, None
    toks = str(spec).replace(",", " ").split()
    if len(toks) == 1 and re.fullmatch(r"\d+", toks[0]):
        return int(toks[0]), None
    crit = toks[0].lower()
    if crit not in ("aic", "bic", "hqic") or not allow_ic or len(toks) > 2:
        raise ValueError(f"{label}: lags({spec}) 형식 오류 — 정수 또는 aic|bic|hqic # (예: 'aic 4')"
                         + ("" if allow_ic else "; 이 검정은 정수 시차만 허용"))
    if len(toks) == 2:
        if not re.fullmatch(r"\d+", toks[1]) or int(toks[1]) < 1:
            raise ValueError(f"{label}: lags({spec}) — 최대 시차는 1 이상 정수")
        return int(toks[1]), crit
    return 1, crit


def _kernel_spec(spec: Any, label: str, allow_auto: Sequence[str] = ("nwest", "llc"), default_auto: Optional[str] = "llc",
                 integer_qs: bool = False) -> Tuple[str, Optional[float], Optional[str]]:
    """kernel(kernel [# | nwest | llc]) -> (kernel, lags or None, automatic rule or None)."""
    if spec is None or spec == "":
        return "bartlett", None, default_auto
    if isinstance(spec, (list, tuple)):
        toks = [str(s) for s in spec if str(s).strip()]
    else:
        toks = str(spec).replace(",", " ").split()
    k = resolve_choice(f"{label} kernel", toks[0], KERNELS, KERNEL_ALIASES)
    if len(toks) > 2:
        raise ValueError(f"{label}: kernel({spec}) 형식 오류 — kernel [# | nwest]")
    if len(toks) == 1:
        return k, None, default_auto
    b = toks[1].lower()
    try:
        m = float(b)
    except ValueError:
        if b in ("nw", "nwe", "nwes", "nwest"):
            b = "nwest"
        if b not in allow_auto:
            raise ValueError(f"{label}: kernel() 시차 선택 방식 '{toks[1]}' 미지원 (가능: 정수"
                             + (", " + ", ".join(allow_auto) if allow_auto else "") + ")")
        return k, None, b
    if m < 0 or (m != int(m) and (k != "quadraticspectral" or integer_qs)):
        raise ValueError(f"{label}: kernel() 시차는 0 이상 정수 (quadratic spectral만 실수 허용)")
    return k, m, None


def llc_test(P: URPanel, var: str, model: int, demean: bool, lags: Any, kernel: Any) -> Dict[str, Any]:
    """Levin, Lin & Chu (2002) bias-adjusted t statistic ([XT] xtunitroot Methods and formulas)."""
    P.require_balanced("Levin–Lin–Chu")
    p0, crit = _lag_spec(lags, 1, "LLC")
    kern, kl, auto = _kernel_spec(kernel, "LLC", ("nwest", "llc"), "llc")
    e_all, v_all, s_i, p_used, hac_used, ti = [], [], [], [], [], []
    for y in P.series(var, demean):
        T = len(y)
        dy, ly = np.diff(y), y[:-1]
        if crit:
            # Stata includes a constant (and trend) in the lag-selection regressions even under noconstant
            p = _adf_select(y, p0, crit, 2 if model == 1 else model)
        else:
            p = p0
        if T - 1 - p < 3:
            raise ValueError(f"LLC 검정: 시점 수 {T}에 비해 ADF 시차 {p}가 큼")
        rows = slice(p, len(dy))
        L = _lagmat(dy, p)
        d = _det(len(dy), model)
        Z = L if d is None else np.column_stack([L, d[rows]])
        e = _resid(dy[rows], Z)
        v = _resid(ly[rows], Z)
        delta_i = float(v @ e / (v @ v))
        sig2 = float(np.sum((e - delta_i * v) ** 2)) / (T - p - 1)
        sd = math.sqrt(sig2)
        e_all.append(e / sd)
        v_all.append(v / sd)
        ti.append(len(dy) - p)
        p_used.append(p)
        # long-run variance of Δy (panel mean or panel trend removed)
        if model == 1:
            w = dy
        elif model == 2:
            w = dy - dy.mean()
        else:
            w = _resid(dy, _det(len(dy), 3))
        if kl is not None:
            m = kl
        elif auto == "llc":
            m = float(int(3.21 * T ** (1 / 3)))
        else:
            m = nw_bandwidth_lags(w, kern)
        hac_used.append(m)
        s_i.append(math.sqrt(lr_variance(w, m, kern) / sig2))
    e_, v_ = np.concatenate(e_all), np.concatenate(v_all)
    NT = len(e_)
    delta = float(v_ @ e_ / (v_ @ v_))
    rss = float(np.sum((e_ - delta * v_) ** 2))
    sig_e2 = rss / NT
    se_delta = math.sqrt(sig_e2 / float(v_ @ v_))
    td = delta / se_delta
    tt = float(np.mean(ti))
    sbar = float(np.mean(s_i))
    mu_adj, sig_adj = llc_adjustment(tt, model)
    tds = (td - NT * sbar / sig_e2 * se_delta * mu_adj) / sig_adj
    return {"test": "llc", "td": td, "p_td": float(stats.norm.cdf(td)) if model == 1 else None, "tds": tds,
            "p_tds": float(stats.norm.cdf(tds)), "delta": delta, "se_delta": se_delta, "Var_ep": sig_e2, "sbar": sbar,
            "ttilde": tt, "mu_adj": mu_adj, "sig_adj": sig_adj, "N": NT, "N_g": P.n, "N_t": int(P.Ti[0]),
            "adf_lags": float(np.mean(p_used)), "adf_method": crit, "hac_lags": float(np.mean(hac_used)),
            "hac_kernel": kern, "hac_method": auto if kl is None else None, "model": model, "demean": demean}


# ============================================================================ Harris–Tzavalis (1999)


def ht_test(P: URPanel, var: str, model: int, demean: bool, altt: bool) -> Dict[str, Any]:
    """Harris & Tzavalis (1999) test on the OLS / within estimate of rho (T fixed, N -> infinity)."""
    P.require_balanced("Harris–Tzavalis")
    T = int(P.Ti[0])
    num = den = 0.0
    for y in P.series(var, demean):
        yy, ly = y[1:], y[:-1]
        if model == 2:
            yy, ly = yy - yy.mean(), ly - ly.mean()
        elif model == 3:
            D = _det(len(yy), 3)
            yy, ly = _resid(yy, D), _resid(ly, D)
        num += float(ly @ yy)
        den += float(ly @ ly)
    rho = num / den
    Tu = T - 1 if altt else T
    N = P.n
    if model == 1:
        mean, var_ = 1.0, 2 / (Tu * (Tu - 1))
    elif model == 2:
        mean, var_ = 1 - 3 / (Tu + 1), 3 * (17 * Tu ** 2 - 20 * Tu + 17) / (5 * (Tu - 1) * (Tu + 1) ** 3)
    else:
        mean = 1 - 15 / (2 * (Tu + 2))
        var_ = 15 * (193 * Tu ** 2 - 728 * Tu + 1147) / (112 * (Tu + 2) ** 3 * (Tu - 2))
    var_ /= N
    z = (rho - mean) / math.sqrt(var_)
    return {"test": "ht", "rho": rho, "mean_rho": mean, "Var_rho": var_, "z": z, "p": float(stats.norm.cdf(z)),
            "N": N * T, "N_g": N, "N_t": T, "model": model, "demean": demean, "altt": altt}


# ============================================================================ Breitung (2000), Breitung & Das (2005)


def breitung_test(P: URPanel, var: str, model: int, demean: bool, lags: int, robust: bool) -> Dict[str, Any]:
    P.require_balanced("Breitung")
    T = int(P.Ti[0])
    N = P.n
    p = int(lags or 0)
    if p < 0:
        raise ValueError("Breitung 검정: lags()는 0 이상 정수")
    if T - p - 2 < 1:
        raise ValueError(f"Breitung 검정: 시점 수 {T}에 비해 사전 백색화 시차 {p}가 큼")
    if robust and (T - p - 1) < N:
        raise ValueError(f"Breitung 강건 검정: 시차 제외 시점 수(T − p − 1 = {T - p - 1})가 패널 수({N}) 이상이어야 함 "
                         "(Stata: robust test statistic requires more time periods net of lags than panels)")
    num = den = 0.0
    DV, YL = [], []          # transformed series per panel, aligned on a common index for the robust form
    for y in P.series(var, demean):
        dy = np.diff(y)                    # Δy_t, t = 2..T  (index t-2)
        if model in (1, 2):
            ly = y[:-1].copy()             # y_{t-1}
            if model == 2:
                ly = ly - y[p]             # minus y_{p+1}
            dv, yl = dy[p:], ly[p:]        # t = p+2..T
            if p > 0:
                L = _lagmat(dy, p)
                dv, yl = _resid(dv, L), _resid(yl, L)
            sig2 = float(dv @ dv) / (T - p - 2)
        else:
            if p > 0:
                L = _lagmat(dy, p)
                fit = _ols(dy[p:], np.column_stack([L, np.ones(len(dy) - p)]))
                a = fit["b"][:p]
                du = dy[p:] - L @ a                                   # Δu_s, t = p+2..T
                ut = y.copy()
                for j in range(1, p + 1):
                    ut[j:] = ut[j:] - a[j - 1] * y[:-j]
                u = ut[p:-1]                                          # u_t = utmp_{t-1}, t = p+2..T
            else:
                du, u = dy, y[:-1]
            n_s = T - p - 1
            mdu = du.sum() / n_s
            sig2 = float(np.sum((du - mdu) * du)) / (T - p - 2)
            s = np.arange(1, n_s + 1)
            yl = u - u[0] - (s - 1) * mdu
            csum = np.cumsum(du)
            with np.errstate(invalid="ignore", divide="ignore"):
                fwd = (csum[-1] - csum) / (n_s - s)
                dv = (du - fwd) * np.sqrt((n_s - s) / (n_s - s + 1.0))
            dv, yl = dv[:-1], yl[:-1]                                   # Helmert transform defined for s < T-p-1
        num += float(yl @ dv) / sig2
        den += float(yl @ yl) / sig2
        DV.append(dv)
        YL.append(yl)
    lam = num / math.sqrt(den)
    out = {"test": "breitung", "lambda": lam, "p": float(stats.norm.cdf(lam)), "N": N * T, "N_g": N, "N_t": T,
           "lags": p, "model": model, "demean": demean, "robust": robust}
    if robust:
        phi = num / den
        U = np.vstack([dv - phi * yl for dv, yl in zip(DV, YL)])       # N x S
        Om = U @ U.T / (T - p - 2)
        dV, yL = np.vstack(DV), np.vstack(YL)
        a = float(np.sum(dV * yL))
        b = float(np.einsum("it,ij,jt->", yL, Om, yL))
        lr = a / math.sqrt(b)
        out.update({"lrobust": lr, "p_lrobust": float(stats.norm.cdf(lr))})
    return out


# ============================================================================ Im–Pesaran–Shin (2003)

# IPS (2003, table 1): E and Var of t-tilde (columns 2-3) and of t (columns 4-5), model with intercept, by T.
_IPS1_T = (6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 100, 500, 1000)
_IPS1 = np.array([
    (-1.125, .497, -1.520, 1.745), (-1.178, .506, -1.514, 1.414), (-1.214, .506, -1.501, 1.228),
    (-1.244, .527, -1.501, 1.132), (-1.274, .521, -1.504, 1.069), (-1.349, .565, -1.514, .923),
    (-1.395, .592, -1.522, .851), (-1.423, .609, -1.520, .809), (-1.439, .623, -1.526, .789),
    (-1.463, .639, -1.523, .770), (-1.477, .656, -1.527, .760), (-1.504, .683, -1.532, .735),
    (-1.526, .704, -1.531, .715), (-1.526, .702, -1.529, .707), (-1.533, .706, -1.533, .706)])


def ips_table1(T: int) -> np.ndarray:
    """Moments for t-tilde and t at T observations: no interpolation for T <= 10 or T > 1000 (Stata xtunitroot)."""
    row = next((i for i, h in enumerate(_IPS1_T) if T <= h), len(_IPS1_T))
    if T <= 10 or T > 1000:
        return _IPS1[row]
    lo, hi = _IPS1_T[row - 1], _IPS1_T[row]
    w = (T - lo) / (hi - lo)
    return (1 - w) * _IPS1[row - 1] + w * _IPS1[row]


# IPS (2003, table 2): fixed-N exact critical values of t-bar (rows N = 5, 7, 10, 15, 20, 25, 50, 100; columns
# T = 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 100); Stata takes the first row/column whose value is >= N/T.
_IPS2_N = (5, 7, 10, 15, 20, 25, 50)
_IPS2_T = (5, 10, 15, 20, 25, 30, 40, 50, 60, 70)
_IPS2 = {
    ("c", 1): "-3.79 -2.66 -2.54 -2.50 -2.46 -2.44 -2.43 -2.42 -2.42 -2.40 -2.40|-3.45 -2.47 -2.38 -2.33 -2.32 -2.31 -2.29 -2.28 -2.28 -2.28 -2.27|-3.06 -2.32 -2.24 -2.21 -2.19 -2.18 -2.16 -2.16 -2.16 -2.16 -2.15|-2.79 -2.14 -2.10 -2.08 -2.07 -2.05 -2.04 -2.05 -2.04 -2.04 -2.04|-2.61 -2.06 -2.02 -2.00 -1.99 -1.99 -1.98 -1.98 -1.98 -1.97 -1.97|-2.51 -2.01 -1.97 -1.95 -1.94 -1.94 -1.93 -1.93 -1.93 -1.93 -1.92|-2.20 -1.85 -1.83 -1.82 -1.82 -1.82 -1.81 -1.81 -1.81 -1.81 -1.81|-2.00 -1.75 -1.74 -1.73 -1.73 -1.73 -1.73 -1.73 -1.73 -1.73 -1.73",
    ("c", 5): "-2.76 -2.28 -2.21 -2.19 -2.18 -2.16 -2.16 -2.15 -2.16 -2.15 -2.15|-2.57 -2.17 -2.11 -2.09 -2.08 -2.07 -2.07 -2.06 -2.06 -2.06 -2.05|-2.42 -2.06 -2.02 -1.99 -1.99 -1.99 -1.98 -1.98 -1.97 -1.98 -1.97|-2.28 -1.95 -1.92 -1.91 -1.90 -1.90 -1.90 -1.89 -1.89 -1.89 -1.89|-2.18 -1.89 -1.87 -1.86 -1.85 -1.85 -1.85 -1.85 -1.84 -1.84 -1.84|-2.11 -1.85 -1.83 -1.82 -1.82 -1.82 -1.82 -1.81 -1.81 -1.81 -1.81|-1.95 -1.75 -1.74 -1.73 -1.73 -1.73 -1.73 -1.73 -1.73 -1.73 -1.73|-1.84 -1.68 -1.67 -1.67 -1.67 -1.67 -1.67 -1.67 -1.67 -1.67 -1.67",
    ("c", 10): "-2.38 -2.10 -2.06 -2.04 -2.04 -2.02 -2.02 -2.02 -2.02 -2.02 -2.01|-2.27 -2.01 -1.98 -1.96 -1.95 -1.95 -1.95 -1.95 -1.94 -1.95 -1.94|-2.17 -1.93 -1.90 -1.89 -1.88 -1.88 -1.88 -1.88 -1.88 -1.88 -1.88|-2.06 -1.85 -1.83 -1.82 -1.82 -1.82 -1.81 -1.81 -1.81 -1.81 -1.81|-2.00 -1.80 -1.79 -1.78 -1.78 -1.78 -1.78 -1.78 -1.78 -1.77 -1.77|-1.96 -1.77 -1.76 -1.75 -1.75 -1.75 -1.75 -1.75 -1.75 -1.75 -1.75|-1.85 -1.70 -1.69 -1.69 -1.69 -1.69 -1.68 -1.68 -1.68 -1.68 -1.69|-1.77 -1.64 -1.64 -1.64 -1.64 -1.64 -1.64 -1.64 -1.64 -1.64 -1.64",
    ("ct", 1): "-8.12 -3.42 -3.21 -3.13 -3.09 -3.05 -3.03 -3.02 -3.00 -3.00 -2.99|-7.36 -3.20 -3.03 -2.97 -2.94 -2.93 -2.90 -2.88 -2.88 -2.87 -2.86|-6.44 -3.03 -2.88 -2.84 -2.82 -2.79 -2.78 -2.77 -2.76 -2.75 -2.75|-5.72 -2.86 -2.74 -2.71 -2.69 -2.68 -2.67 -2.65 -2.66 -2.65 -2.64|-5.54 -2.75 -2.67 -2.63 -2.62 -2.61 -2.59 -2.60 -2.59 -2.58 -2.58|-5.16 -2.69 -2.61 -2.58 -2.58 -2.56 -2.55 -2.55 -2.55 -2.54 -2.54|-4.50 -2.53 -2.48 -2.46 -2.45 -2.45 -2.44 -2.44 -2.44 -2.44 -2.43|-4.00 -2.42 -2.39 -2.38 -2.37 -2.37 -2.36 -2.36 -2.36 -2.36 -2.36",
    ("ct", 5): "-4.66 -2.98 -2.87 -2.82 -2.80 -2.79 -2.77 -2.76 -2.75 -2.75 -2.75|-4.38 -2.85 -2.76 -2.72 -2.70 -2.69 -2.68 -2.67 -2.67 -2.66 -2.66|-4.11 -2.74 -2.66 -2.63 -2.62 -2.60 -2.60 -2.59 -2.59 -2.58 -2.58|-3.88 -2.63 -2.57 -2.55 -2.53 -2.53 -2.52 -2.52 -2.52 -2.51 -2.51|-3.73 -2.56 -2.52 -2.49 -2.48 -2.48 -2.48 -2.47 -2.47 -2.46 -2.46|-3.62 -2.52 -2.48 -2.46 -2.45 -2.45 -2.44 -2.44 -2.44 -2.44 -2.43|-3.35 -2.42 -2.38 -2.38 -2.37 -2.37 -2.36 -2.36 -2.36 -2.36 -2.36|-3.13 -2.34 -2.32 -2.32 -2.31 -2.31 -2.31 -2.31 -2.31 -2.31 -2.31",
    ("ct", 10): "-3.73 -2.77 -2.70 -2.67 -2.65 -2.64 -2.63 -2.62 -2.63 -2.62 -2.62|-3.60 -2.68 -2.62 -2.59 -2.58 -2.57 -2.57 -2.56 -2.56 -2.55 -2.55|-3.45 -2.59 -2.54 -2.52 -2.51 -2.51 -2.50 -2.50 -2.50 -2.49 -2.49|-3.33 -2.52 -2.47 -2.46 -2.45 -2.45 -2.44 -2.44 -2.44 -2.44 -2.44|-3.26 -2.47 -2.44 -2.42 -2.41 -2.41 -2.41 -2.40 -2.40 -2.40 -2.40|-3.18 -2.44 -2.40 -2.39 -2.39 -2.38 -2.38 -2.38 -2.38 -2.38 -2.38|-3.02 -2.36 -2.33 -2.33 -2.33 -2.32 -2.32 -2.32 -2.32 -2.32 -2.32|-2.90 -2.30 -2.29 -2.28 -2.28 -2.28 -2.28 -2.28 -2.28 -2.28 -2.28",
}


def ips_critical(N: int, T: float, trend: bool) -> List[float]:
    r = next((i for i, h in enumerate(_IPS2_N) if N <= h), len(_IPS2_N))
    c = next((i for i, h in enumerate(_IPS2_T) if T <= h), len(_IPS2_T))
    key = "ct" if trend else "c"
    return [float(_IPS2[(key, a)].split("|")[r].split()[c]) for a in (1, 5, 10)]


# IPS (2003, table 3): E and Var of the ADF t statistic t_iT(p), p = 0..8 (rows), T = 10, 15, 20, 25, 30, 40, 50,
# 60, 70, 100 (columns), intercept ("c") and intercept + trend ("ct"), as tabulated in Stata xtunitroot.
_IPS3_T = (10, 15, 20, 25, 30, 40, 50, 60, 70, 100)
_IPS3 = {
    ("c", "mean"): """-1.504 -1.514 -1.522 -1.520 -1.526 -1.523 -1.527 -1.519 -1.524 -1.532
-1.488 -1.503 -1.516 -1.514 -1.519 -1.520 -1.524 -1.519 -1.522 -1.530
-1.319 -1.387 -1.428 -1.443 -1.460 -1.476 -1.493 -1.490 -1.498 -1.514
-1.306 -1.366 -1.413 -1.433 -1.453 -1.471 -1.489 -1.486 -1.495 -1.512
-1.171 -1.260 -1.329 -1.363 -1.394 -1.428 -1.454 -1.458 -1.470 -1.495
nan nan -1.313 -1.351 -1.384 -1.421 -1.451 -1.454 -1.467 -1.494
nan nan nan -1.289 -1.331 -1.380 -1.418 -1.427 -1.444 -1.476
nan nan nan -1.273 -1.319 -1.371 -1.411 -1.423 -1.441 -1.474
nan nan nan -1.212 -1.266 -1.329 -1.377 -1.393 -1.415 -1.456""",
    ("c", "var"): """1.069 0.923 0.851 0.809 0.789 0.770 0.760 0.749 0.736 0.735
1.255 1.011 0.915 0.861 0.831 0.803 0.781 0.770 0.753 0.745
1.421 1.078 0.969 0.905 0.865 0.830 0.798 0.789 0.766 0.754
1.759 1.181 1.037 0.952 0.907 0.858 0.819 0.802 0.782 0.761
2.080 1.279 1.097 1.005 0.946 0.886 0.842 0.819 0.801 0.771
nan nan 1.171 1.055 0.980 0.912 0.863 0.839 0.814 0.781
nan nan nan 1.114 1.023 0.942 0.886 0.858 0.834 0.795
nan nan nan 1.164 1.062 0.968 0.910 0.875 0.851 0.806
nan nan nan 1.217 1.105 0.996 0.929 0.896 0.871 0.818""",
    # row p = 8, T = 60: Stata tabulates -2.204 (out of line with the neighbouring -1.987 and -2.046)
    ("ct", "mean"): """-2.166 -2.167 -2.168 -2.167 -2.172 -2.173 -2.176 -2.174 -2.174 -2.177
-2.173 -2.169 -2.172 -2.172 -2.173 -2.177 -2.180 -2.178 -2.176 -2.179
-1.914 -1.999 -2.047 -2.074 -2.095 -2.120 -2.137 -2.143 -2.146 -2.158
-1.922 -1.977 -2.032 -2.065 -2.091 -2.117 -2.137 -2.142 -2.146 -2.158
-1.750 -1.823 -1.911 -1.968 -2.009 -2.057 -2.091 -2.103 -2.114 -2.135
nan nan -1.888 -1.955 -1.998 -2.051 -2.087 -2.101 -2.111 -2.135
nan nan nan -1.868 -1.923 -1.995 -2.042 -2.065 -2.081 -2.113
nan nan nan -1.851 -1.912 -1.986 -2.036 -2.063 -2.079 -2.112
nan nan nan -1.761 -1.835 -1.925 -1.987 -2.204 -2.046 -2.088""",
    ("ct", "var"): """1.132 0.869 0.763 0.713 0.690 0.655 0.633 0.621 0.610 0.597
1.453 0.975 0.845 0.769 0.734 0.687 0.654 0.641 0.627 0.605
1.627 1.036 0.882 0.796 0.756 0.702 0.661 0.653 0.634 0.613
2.482 1.214 0.983 0.861 0.808 0.735 0.688 0.674 0.650 0.625
3.947 1.332 1.052 0.913 0.845 0.759 0.705 0.685 0.662 0.629
nan nan 1.165 0.991 0.899 0.792 0.730 0.705 0.673 0.638
nan nan nan 1.055 0.945 0.828 0.753 0.725 0.689 0.650
nan nan nan 1.145 1.009 0.872 0.786 0.747 0.713 0.661
nan nan nan 1.208 1.063 0.902 0.808 0.766 0.728 0.670""",
}
_IPS3_ARR = {k: np.array([[float(x) for x in r.split()] for r in v.splitlines()]) for k, v in _IPS3.items()}


def ips_table3(p: int, T: int, trend: bool) -> Tuple[float, float]:
    if p < 0 or p > 8:
        return np.nan, np.nan
    key = "ct" if trend else "c"
    mu, va = _IPS3_ARR[(key, "mean")][p], _IPS3_ARR[(key, "var")][p]
    c = next((i for i, h in enumerate(_IPS3_T) if T <= h), len(_IPS3_T) - 1)
    if c == 0 or T >= 100 or T in _IPS3_T:
        return float(mu[c]), float(va[c])
    w = (T - _IPS3_T[c - 1]) / (_IPS3_T[c] - _IPS3_T[c - 1])
    return float((1 - w) * mu[c - 1] + w * mu[c]), float((1 - w) * va[c - 1] + w * va[c])


def ips_test(P: URPanel, var: str, trend: bool, demean: bool, lags: Any) -> Dict[str, Any]:
    P.require_no_gaps("Im–Pesaran–Shin")
    p0, crit = _lag_spec(lags, None, "IPS")
    model = 3 if trend else 2
    N = P.n
    tbar = []
    out: Dict[str, Any] = {"test": "ips", "N_g": N, "N_t": float(np.mean(P.Ti)), "trend": trend, "demean": demean,
                           "balanced": P.balanced, "adf_method": crit}
    nobs = 0
    if p0 is None:
        tt, mu_t, var_t, mu_tt, var_tt = [], [], [], [], []
        for y in P.series(var, demean):
            T = len(y)
            dy, ly = np.diff(y), y[:-1]
            X = np.column_stack([ly, _det(len(dy), model)])
            f = _ols(dy, X)
            nobs += f["n"]
            b, se = f["b"][0], f["se"][0]
            tbar.append(b / se)
            sd = float(np.std(dy, ddof=1))
            tt.append(b / (se / math.sqrt(f["s2"]) * sd))
            m = ips_table1(T)
            mu_tt.append(m[0]); var_tt.append(m[1]); mu_t.append(m[2]); var_t.append(m[3])
        tb, ttb = float(np.mean(tbar)), float(np.mean(tt))
        zt = math.sqrt(N) * (tb - np.mean(mu_t)) / math.sqrt(np.mean(var_t))
        ztt = math.sqrt(N) * (ttb - np.mean(mu_tt)) / math.sqrt(np.mean(var_tt))
        minT = int(P.Ti.min())
        if P.balanced:
            cv = ips_critical(N, float(np.mean(P.Ti)), trend)
            ok = minT > (6 if trend else 5)
        else:
            cv = None
            ok = minT > 9
        out.update({"tbar": tb, "ttildebar": ttb, "zt": zt, "zttildebar": ztt,
                    "p_zttildebar": float(stats.norm.cdf(ztt)) if ok else None, "cv": cv, "lags": None})
    else:
        if p0 > 8:
            raise ValueError(f"IPS 검정: E[t]·Var[t] 표(IPS 2003 표 3)는 시차 8까지 — lags({p0}) 불가 "
                             f"(Stata: mean and variance of t-statistic not available for T = {int(P.Ti.max())} with {p0} lags)")
        mus, vas, ps = [], [], []
        for y in P.series(var, demean):
            T = len(y)
            p = _adf_select(y, p0, crit, model) if crit else p0
            dy, ly = np.diff(y), y[:-1]
            rows = slice(p, len(dy))
            X = np.column_stack([ly[rows], _lagmat(dy, p), _det(len(dy), model)[rows]])
            f = _ols(dy[rows], X)
            nobs += f["n"]
            tbar.append(f["b"][0] / f["se"][0])
            mu, va = ips_table3(p, T, trend)
            if not np.isfinite(mu):
                raise ValueError(f"IPS 검정: T = {T}, 시차 {p}의 E[t]·Var[t] 없음 (IPS 2003 표 3; Stata: mean and variance "
                                 f"of t-statistic not available for T = {T} with {p} lags)")
            mus.append(mu); vas.append(va); ps.append(p)
        tb = float(np.mean(tbar))
        w = math.sqrt(N) * (tb - np.mean(mus)) / math.sqrt(np.mean(vas))
        out.update({"tbar": tb, "wtbar": w, "p_wtbar": float(stats.norm.cdf(w)), "lags": float(np.mean(ps))})
    out["N"] = nobs
    return out


# ============================================================================ Fisher-type (Choi 2001)


def _gap_adf(y: np.ndarray, tp: np.ndarray, p: int, trend: str) -> Tuple[float, int, int]:
    """ADF t statistic with Stata time-series operators across gaps (an observation is used only when all its lags
    exist): returns (t, nobs, df_resid)."""
    pos = {int(t): i for i, t in enumerate(tp)}

    def val(t):
        i = pos.get(t)
        return None if i is None else y[i]

    rows, Y = [], []
    for t in tp:
        t = int(t)
        y0, y1 = val(t), val(t - 1)
        if y1 is None:
            continue
        r = [y1]
        ok = True
        for j in range(1, p + 1):
            a, b = val(t - j), val(t - j - 1)
            if a is None or b is None:
                ok = False
                break
            r.append(a - b)
        if not ok:
            continue
        if trend == "ct":
            r.append(float(t))
        if trend in ("c", "ct"):
            r.append(1.0)
        rows.append(r)
        Y.append(y0 - y1)
    X, Y = np.asarray(rows, float), np.asarray(Y, float)
    f = _ols(Y, X)
    return float(f["b"][0] / f["se"][0]), f["n"], f["n"] - f["k"]


def _gap_pp(y: np.ndarray, tp: np.ndarray, q: int, trend: str) -> Tuple[float, int]:
    """Phillips–Perron Z(t) with Stata time-series operators across gaps: regression of y on L.y (+ constant/trend),
    Newey–West (Bartlett, q lags) long-run variance of the residuals using only pairs q periods apart."""
    pos = {int(t): i for i, t in enumerate(tp)}
    rows, Y, ts = [], [], []
    for t in tp:
        t = int(t)
        if t - 1 not in pos:
            continue
        r = [y[pos[t - 1]]]
        if trend == "ct":
            r.append(float(t))
        if trend in ("c", "ct"):
            r.append(1.0)
        rows.append(r); Y.append(y[pos[t]]); ts.append(t)
    X, Y = np.asarray(rows, float), np.asarray(Y, float)
    f = _ols(Y, X)
    n, k = f["n"], f["k"]
    e = f["e"]
    epos = {t: e[i] for i, t in enumerate(ts)}
    g0 = float(e @ e) / n
    lam2 = g0
    for j in range(1, q + 1):
        gj = sum(epos[t] * epos[t - j] for t in ts if (t - j) in epos) / n
        lam2 += 2 * (1 - j / (q + 1)) * gj
    s2 = f["rss"] / (n - k)
    t_rho = (f["b"][0] - 1) / f["se"][0]
    se = f["se"][0]
    zt = math.sqrt(g0 / lam2) * t_rho - 0.5 * (lam2 - g0) / math.sqrt(lam2) * n * se / math.sqrt(s2)
    return float(zt), n


def fisher_test(P: URPanel, var: str, method: str, demean: bool, lags: Any, trend: bool, drift: bool,
                noconstant: bool = False) -> Dict[str, Any]:
    """Choi (2001) combinations of per-panel ADF (Stata dfuller) or Phillips–Perron (pperron) p-values."""
    from arch.unitroot import ADF, PhillipsPerron
    from arch.unitroot.unitroot import mackinnonp
    method = resolve_choice("Fisher 검정 방식(method)", method, ("dfuller", "pperron"),
                            {"adf": "dfuller", "df": "dfuller", "pp": "pperron", "phillips": "pperron"})
    if noconstant:
        raise ValueError("Fisher 검정: noconstant 불가 — 이 경우 p값 없음 (Stata: cannot specify noconstant because "
                         "p-values are not available in this case)")
    if lags is None or lags == "":
        raise ValueError("Fisher 검정: 시차 lags(#) 지정 필요 (Stata: must specify lags())")
    p, crit = _lag_spec(lags, None, "Fisher", allow_ic=False)
    if method == "pperron" and drift:
        raise ValueError("Fisher 검정: drift는 dfuller와 함께만 가능 (Stata: cannot specify drift with pperron)")
    if trend and drift:
        raise ValueError("Fisher 검정: trend와 drift 동시 지정 불가 (Stata dfuller)")
    reg = "ct" if trend else "c"
    pv, failed = [], []
    vals = P.values(var, demean)
    for gi, s in enumerate(P.slices):
        y, tp = vals[s], P.tpos[s]
        gaps = bool(np.any(np.diff(tp) != 1))
        try:
            if method == "dfuller":
                if not gaps:
                    r = ADF(y, lags=p, trend=reg)
                    stat, df_ = float(r.stat), int(r.regression.df_resid)
                else:
                    stat, _, df_ = _gap_adf(y, tp, p, reg)
                pv.append(float(stats.t.cdf(stat, df_)) if drift else float(mackinnonp(stat, regression=reg)))
            else:
                if not gaps:
                    stat = float(PhillipsPerron(y, lags=p, trend=reg, test_type="tau").stat)
                else:
                    stat, _ = _gap_pp(y, tp, p, reg)
                pv.append(float(mackinnonp(stat, regression=reg)))
        except Exception:
            # too few observations for the per-panel regression: Stata dfuller/pperron return Z(t) = . with
            # MacKinnon p = 1, and xtunitroot fisher keeps that p-value
            failed.append(str(P.labels[gi]))
            pv.append(1.0)
    if len(failed) == len(pv):
        raise ValueError("Fisher 검정: 모든 패널에서 패널별 검정 계산 불가 (관측치 수에 비해 시차가 큼)")
    pv = np.maximum(np.asarray(pv), np.finfo(float).eps)     # p = 0 replaced by machine epsilon (Stata)
    N = len(pv)
    P_ = -2 * float(np.sum(np.log(pv)))
    # a panel with p = 1 has no finite inverse-normal or logit value; Stata leaves it out of Z and L* (and their N)
    zi = stats.norm.ppf(pv)
    zi = zi[np.isfinite(zi)]
    Z = float(np.sum(zi)) / math.sqrt(len(zi))
    with np.errstate(divide="ignore"):
        li = np.log(pv / (1 - pv))
    li = li[np.isfinite(li)]
    NL = len(li)
    L = float(np.sum(li))
    k = 3 * (5 * NL + 3) / (math.pi ** 2 * NL * (5 * NL + 2))      # Stata 19.5 xtunitroot fisher
    k_choi = 3 * (5 * NL + 4) / (math.pi ** 2 * NL * (5 * NL + 2))  # Choi (2001, p. 256)
    Ls = math.sqrt(k) * L
    Pm = float(np.sum(-2 * np.log(pv) - 2)) / (2 * math.sqrt(N))
    capT = P.N / P.n
    return {"test": "fisher", "method": method, "P": P_, "df_P": 2 * N, "p_P": float(stats.chi2.sf(P_, 2 * N)),
            "Z": Z, "p_Z": float(stats.norm.cdf(Z)), "L": Ls, "df_L": 5 * NL + 4,
            "p_L": float(stats.t.cdf(Ls, 5 * NL + 4)), "L_choi": math.sqrt(k_choi) * L, "Pm": Pm,
            "n_dropped_zl": N - min(len(zi), NL), "failed": failed,
            "p_Pm": float(stats.norm.sf(Pm)), "N_g": P.n, "N_t": capT, "N": int(round(capT * P.n)), "lags": p,
            "trend": trend, "drift": drift, "demean": demean, "balanced": P.balanced, "pvalues": pv}


# ============================================================================ Hadri (2000)


def hadri_test(P: URPanel, var: str, trend: bool, demean: bool, robust: bool, kernel: Any) -> Dict[str, Any]:
    P.require_balanced("Hadri LM")
    kern = m = None
    if kernel not in (None, ""):
        kern, m, auto = _kernel_spec(kernel, "Hadri", (), None)
        if robust:
            raise ValueError("Hadri 검정: kernel()과 robust 동시 지정 불가 (Stata: cannot specify both kernel() and robust)")
        if m is None:
            m = 1.0
    T = int(P.Ti[0])
    N = P.n
    D = np.column_stack([np.arange(1, T + 1, dtype=float), np.ones(T)]) if trend else np.ones((T, 1))
    div = T - 1 - (1 if trend else 0)
    ssum, esq, lrv, ratio = 0.0, 0.0, [], 0.0
    for y in P.series(var, demean):
        e = _resid(y, D)
        S2 = float(np.sum(np.cumsum(e) ** 2))
        ssum += S2
        ee = float(e @ e)
        esq += ee
        if robust:
            ratio += S2 / (ee / div)
        if kern:
            nl = int(m)
            kw = kernel_weights(m, kern, nl) if nl > 0 else np.zeros(0)
            lrv.append(ee / T + 2 * float(kw @ _autocov_sums(e, nl)) / T)
    if kern:
        LM = ssum / float(np.mean(lrv)) / (N * T ** 2)
    elif robust:
        LM = ratio / (N * T ** 2)
    else:
        LM = ssum / (N * T ** 2) / (esq / (N * div))
    mu, var_ = (1 / 15, 11 / 6300) if trend else (1 / 6, 1 / 45)
    z = math.sqrt(N) * (LM - mu) / math.sqrt(var_)
    return {"test": "hadri", "z": z, "p": float(stats.norm.sf(z)), "mu": mu, "var": var_, "LM": LM, "N": N * T,
            "N_g": N, "N_t": T, "trend": trend, "demean": demean, "robust": robust, "kernel": kern,
            "lags": m if kern else None}


# ============================================================================ panel cointegration: shared parts


def _coint_lrcov(w: np.ndarray, kernel: str, lags: Optional[float]) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Contemporaneous and long-run covariance of one panel's w (T x q), Newey & West (1987) kernel estimator with
    the conventions of Stata xtcointtest: columns demeaned, divisor T; fixed lags m use weights K(j, m) with
    z = j/(m+1); automatic lags follow Newey & West (1994) with the trace of the autocovariance matrices,
    bandwidth b = int(gamma T^(1/(2q+1))) (quadratic spectral: not truncated) and weights z = j/b.
    Returns (Sigma, Omega, displayed lags, bandwidth)."""
    w = np.asarray(w, float)
    if w.ndim == 1:
        w = w[:, None]
    w = w - w.mean(axis=0)
    T = len(w)
    S = w.T @ w / T
    if lags is None:
        q, cy, integer, n = {"bartlett": (1, 1.1447, True, int(4 * (T / 100) ** (2 / 9))),
                             "parzen": (2, 2.6614, True, int(4 * (T / 100) ** (4 / 25))),
                             "quadraticspectral": (2, 1.3221, False, int(4 * (T / 100) ** (2 / 25)))}[kernel]
        sig = np.array([float(np.sum(w[j:] * w[:T - j])) for j in range(1, n + 1)])
        s0 = float(np.sum(w * w)) + 2 * sig.sum()
        sq = 2 * float(np.sum(np.arange(1, n + 1, dtype=float) ** q * sig))
        pw = 1 / (2 * q + 1)
        bw = cy * ((sq / s0) ** 2) ** pw * T ** pw
        if integer:
            bw = float(int(bw))
        m_w = bw - 1.0
        shown = min(bw, float(n))
    else:
        m_w, bw, shown = float(lags), float(lags) + 1.0, float(lags)
    nl = (T - 1) if kernel == "quadraticspectral" else max(int(math.floor(m_w + 1e-12)), 0)
    if kernel != "quadraticspectral":
        nl = min(nl, T - 2)          # Stata: Bartlett/Parzen sums stop at lag T − 2
    O = S.copy()
    if nl > 0 and m_w + 1 > 0:            # quadratic spectral: also a bandwidth below 1 (Stata)
        kw = kernel_weights(m_w, kernel, nl)
        for j in range(1, nl + 1):
            if kw[j - 1] == 0:
                continue
            G = w[j:].T @ w[:T - j] / T
            O = O + kw[j - 1] * (G + G.T)
    return S, O, shown, bw


def _cond_var(M: np.ndarray) -> float:
    """M_00 − M_0x M_xx^{-1} M_x0 (variance of the first component conditional on the others)."""
    if M.shape[0] == 1:
        return float(M[0, 0])
    return float(M[0, 0] - M[0, 1:] @ np.linalg.solve(M[1:, 1:], M[1:, 0]))


class _CointPanel:
    """Sorted per-panel arrays of y and X for the cointegration tests (no gaps allowed)."""

    def __init__(self, df: pd.DataFrame, entity: str, time: str, dep: str, covs: List[str], demean: bool, label: str):
        if not dep:
            raise ValueError(f"{label}: 종속변수(dep) 지정 필요. 사용 가능한 변수: {list(df.columns)}")
        if not covs:
            raise ValueError(f"{label}: 설명변수(covs) 1개 이상 지정 필요 (Stata: at least one regressor must be specified)")
        if dep in covs:
            raise ValueError(f"{label}: 종속변수 '{dep}'를 설명변수로 지정할 수 없음 (Stata: may not specify dependent "
                             "variable as one of the regressors)")
        self.P = URPanel(df, entity, time, [dep] + list(covs), label)
        self.P.require_no_gaps(label.split(" ")[0])
        self.y = self.P.series(dep, demean)
        cols = [self.P.series(c, demean) for c in covs]
        self.X = [np.column_stack([c[i] for c in cols]) for i in range(self.P.n)]
        self.N = self.P.n
        self.k = len(covs)


def _pooled_ic(ylist, Xlist, crit: str) -> float:
    """estat ic after a pooled regression without constant: AIC = −2 lnL + 2k, BIC = −2 lnL + k ln n,
    HQIC = −2 lnL + 2 k ln ln n (Stata xtcointtest)."""
    f = _ols(np.concatenate(ylist), np.vstack(Xlist))
    n, k, ll = f["n"], f["k"], f["ll"]
    return -2 * ll + {"aic": 2.0 * k, "bic": k * math.log(n), "hqic": 2 * math.log(math.log(n)) * k}[crit]


def _adf_design(e: np.ndarray, p: int, start: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rows t = start..T−1 (0-based) of the ADF regression e_t on e_{t−1}, Δe_{t−1..t−p} (start >= p+1)."""
    de = np.diff(e)                       # de[t-1] = e_t − e_{t−1}
    rows = np.arange(start, len(e))
    X = np.column_stack([e[rows - 1]] + [de[rows - 1 - j] for j in range(1, p + 1)])
    return e[rows], X


def _select_adf_lag(E: List[np.ndarray], pmax: int, crit: str) -> int:
    """ADF lag for the pooled residual regression chosen by the criterion over p = pmax..1 on the p_max sample."""
    best, best_p = np.inf, pmax
    for p in range(pmax, 0, -1):
        ys, Xs = zip(*[_adf_design(e, p, pmax + 1) for e in E])
        v = _pooled_ic(ys, Xs, crit)
        if v < best:
            best, best_p = v, p
    return best_p


def _coint_lags(lags: Any, label: str) -> Tuple[int, Optional[str]]:
    p, crit = _lag_spec(lags, 1, label)
    if p < 1:
        raise ValueError(f"{label}: ADF 시차는 1 이상 (lags(0)은 Stata xtcointtest에서 오류 또는 해석 불가 값)")
    return p, crit


def _pval_abs(z: float) -> float:
    """Stata xtcointtest p-value: 1 − Φ(|z|)."""
    return float(stats.norm.sf(abs(z)))


# ============================================================================ Kao (1999)


def kao_test(df: pd.DataFrame, entity: str, time: str, dep: str, covs: List[str], demean: bool, lags: Any,
             kernel: Any) -> Dict[str, Any]:
    C = _CointPanel(df, entity, time, dep, covs, demean, "Kao 검정")
    p0, crit = _coint_lags(lags, "Kao 검정")
    kern, kl, auto = _kernel_spec(kernel, "Kao 검정", ("nwest",), "nwest", integer_qs=True)
    N = C.N
    # u_it, eps_it: residuals of pooled AR(1) regressions without constant of y and of each x (t >= 2)
    series = [C.y] + [[x[:, j] for x in C.X] for j in range(C.k)]
    resid = []
    for s in series:
        ly = np.concatenate([v[:-1] for v in s])
        yy = np.concatenate([v[1:] for v in s])
        b = float(ly @ yy / (ly @ ly))
        resid.append([v[1:] - b * v[:-1] for v in s])
    S_l, O_l, shown, bws = [], [], [], []
    for i in range(N):
        w = np.column_stack([r[i] for r in resid])
        S, O, sh, bw = _coint_lrcov(w, kern, kl)
        S_l.append(S); O_l.append(O); shown.append(sh); bws.append(bw)
    sig_nu = _cond_var(np.mean(S_l, axis=0))
    sig0_nu = _cond_var(np.mean(O_l, axis=0))
    # fixed-effects cointegrating regression on the sample with t >= 2
    yv = [v[1:] for v in C.y]
    Xv = [x[1:] for x in C.X]
    ydm = np.concatenate([v - v.mean() for v in yv])
    Xdm = np.vstack([x - x.mean(axis=0) for x in Xv])
    beta = np.linalg.lstsq(Xdm, ydm, rcond=None)[0]
    E = []
    for v, x in zip(yv, Xv):
        r = v - x @ beta
        E.append(r - r.mean())
    le = np.concatenate([e[:-1] for e in E])
    ee = np.concatenate([e[1:] for e in E])
    rho = float(le @ ee / (le @ le))
    n = len(ee)
    T = n / N
    s2 = float(np.sum((ee - rho * le) ** 2)) / n
    trho = (rho - 1) / math.sqrt(s2 / float(le @ le))
    a, b = sig_nu, sig0_nu
    df_rho_star = (math.sqrt(N) * T * (rho - 1) + 3 * math.sqrt(N) * a / b) / math.sqrt(3 + 36 * a ** 2 / (5 * b ** 2))
    den_t = math.sqrt(b / (2 * a) + 3 * a / (10 * b))
    df_t_star = (trho + math.sqrt(6 * N * a) / (2 * math.sqrt(b))) / den_t
    df_rho = (math.sqrt(N) * T * (rho - 1) + 3 * math.sqrt(N)) / math.sqrt(51 / 5)
    df_t = math.sqrt(5 / 4) * trho + math.sqrt(15 * N / 8)
    p = _select_adf_lag(E, p0, crit) if crit else p0
    ys, Xs = zip(*[_adf_design(e, p, p + 1) for e in E])
    fa = _ols(np.concatenate(ys), np.vstack(Xs))
    rho_adf = float(fa["b"][0])
    qsum = 0.0
    for Xi in Xs:
        lv = _resid(Xi[:, 0], Xi[:, 1:])
        qsum += float(lv @ lv)
    tadf = (rho_adf - 1) * math.sqrt(qsum) / math.sqrt(fa["rss"] / (N * T))
    adf = (tadf + math.sqrt(6 * N * a) / (2 * math.sqrt(b))) / den_t
    st = [("mdf", "Modified Dickey–Fuller t", df_rho_star), ("df", "Dickey–Fuller t", df_t_star),
          ("adf", "Augmented Dickey–Fuller t", adf), ("umdf", "Unadjusted modified Dickey–Fuller t", df_rho),
          ("udf", "Unadjusted Dickey–Fuller t", df_t)]
    return {"test": "kao", "stats": [(k_, lab, v, _pval_abs(v)) for k_, lab, v in st], "N_g": N, "N_t": T, "N": n,
            "hac_lags": float(np.mean(shown)), "bw": float(np.mean(bws)), "hac_kernel": kern,
            "hac_method": auto if kl is None else None, "adf_lags": p, "adf_method": crit, "rho": rho,
            "sigmasq_nu": a, "sigmasq0_nu": b, "demean": demean, "balanced": C.P.balanced, "k": C.k}


# ============================================================================ Pedroni (1999, 2004)

# Adjustment terms (mean, variance) of the standardized statistics, (stat - mu sqrt(N)) / sqrt(v).
# Order: panel v, panel rho, panel t (PP and ADF), group rho, group t (PP and ADF).
# k = 2..7 regressors: Pedroni (1999, table 2; "standard case", "heterogeneous intercepts", "heterogeneous
# deterministic trends and intercepts"). k = 1: single-regressor values of Pedroni (2004) as used by Stata.
# Stata 19.5 uses +25.889 for the panel rho mean with 7 regressors and no deterministic terms (table 2: −25.889);
# Stata's value is kept so that results match xtcointtest and the result note states it.
_PED = {
    "noconstant": {1: (4.00, 27.81, -2.77, 24.91, -1.01, 1.50, -6.84, 26.78, -1.39, 0.78),
                   2: (6.982, 81.145, -6.388, 64.288, -1.662, 1.559, -9.889, 41.943, -1.992, 0.649),
                   3: (10.402, 140.804, -10.191, 89.962, -2.156, 1.286, -13.865, 57.801, -2.440, 0.600),
                   4: (14.254, 182.450, -14.136, 103.176, -2.571, 1.028, -17.834, 72.097, -2.819, 0.567),
                   5: (18.198, 217.784, -18.042, 120.787, -2.926, 0.928, -21.805, 88.611, -3.151, 0.559),
                   6: (22.169, 256.530, -21.985, 132.499, -3.244, 0.820, -25.750, 103.371, -3.450, 0.544),
                   7: (26.120, 277.429, 25.889, 143.561, -3.533, 0.750, -29.627, 117.059, -3.723, 0.530)},
    "constant": {1: (8.62, 60.75, -6.02, 31.27, -1.73, 0.93, -9.05, 35.98, -2.03, 0.66),
                 2: (11.754, 104.546, -9.495, 57.610, -2.177, 0.964, -12.938, 51.49, -2.453, 0.618),
                 3: (15.197, 151.094, -13.256, 81.772, -2.576, 0.923, -16.888, 67.123, -2.827, 0.585),
                 4: (18.910, 190.661, -17.163, 99.331, -2.930, 0.843, -20.841, 81.835, -3.157, 0.560),
                 5: (22.715, 231.864, -21.013, 119.546, -3.241, 0.800, -24.775, 98.278, -3.452, 0.553),
                 6: (26.603, 270.451, -24.944, 134.341, -3.531, 0.750, -28.720, 113.131, -3.726, 0.542),
                 7: (30.457, 293.431, -28.795, 144.615, -3.795, 0.685, -32.538, 126.059, -3.976, 0.525)},
    "trend": {1: (17.86, 101.68, -10.54, 39.52, -2.29, 0.66, -13.65, 50.91, -2.53, 0.56),
              2: (21.162, 160.249, -14.011, 64.219, -2.648, 0.690, -17.359, 66.387, -2.872, 0.555),
              3: (24.556, 198.167, -17.600, 83.815, -2.967, 0.686, -21.116, 81.832, -3.179, 0.548),
              4: (28.046, 239.425, -21.287, 103.905, -3.262, 0.688, -24.930, 97.362, -3.464, 0.543),
              5: (31.738, 276.997, -25.130, 124.613, -3.545, 0.686, -28.849, 113.145, -3.737, 0.538),
              6: (35.537, 310.982, -28.981, 138.227, -3.806, 0.654, -32.716, 127.989, -3.986, 0.530),
              7: (39.231, 348.217, -32.756, 154.378, -4.047, 0.638, -36.494, 140.756, -4.217, 0.518)},
}


def _det_cols(T: int, det: str) -> List[np.ndarray]:
    if det == "noconstant":
        return []
    c = [np.ones(T)]
    if det == "trend":
        c.append(np.arange(1, T + 1, dtype=float))
    return c


def pedroni_test(df: pd.DataFrame, entity: str, time: str, dep: str, covs: List[str], det: str, ar: str,
                 demean: bool, lags: Any, kernel: Any, per_panel_nw: bool = False) -> Dict[str, Any]:
    C = _CointPanel(df, entity, time, dep, covs, demean, "Pedroni 검정")
    if C.k > 7:
        raise ValueError("Pedroni 검정: 설명변수는 최대 7개 (Stata: number of regressors may not exceed 7)")
    p0, crit = _coint_lags(lags, "Pedroni 검정")
    kern, kl, auto = _kernel_spec(kernel, "Pedroni 검정", ("nwest",), "nwest", integer_qs=True)
    N = C.N
    E, L11, sig2, s2, shown = [], [], [], [], []
    # Automatic lags (Stata 19.5 xtcointtest pedroni): the Newey–West bandwidth is computed for the first panel;
    # its lag min(b, n) is then used for all later panels, separately for the long-run variance of (Δy, Δx)
    # (L11) and of the DF residuals. per_panel_nw=True selects the bandwidth panel by panel instead.
    lag_l11 = lag_u = kl
    for i, (y, X) in enumerate(zip(C.y, C.X)):
        Z = np.column_stack([X] + _det_cols(len(y), det))
        e = _resid(y, Z)
        E.append(e)
        _, O, sh1, _ = _coint_lrcov(np.column_stack([np.diff(y), np.diff(X, axis=0)]), kern, lag_l11)
        L11.append(_cond_var(O))
        le = e[:-1]
        rho_i = float(le @ e[1:] / (le @ le))
        u = e[1:] - rho_i * le
        S, O, sh, _ = _coint_lrcov(u, kern, lag_u)
        s2.append(float(S[0, 0])); sig2.append(float(O[0, 0])); shown.append(sh)
        if kl is None and not per_panel_nw and i == 0:
            lag_l11, lag_u = sh1, sh
    L11, sig2, s2 = np.array(L11), np.array(sig2), np.array(s2)
    lam = 0.5 * (sig2 - s2)
    Tn = np.array([len(e) - 1 for e in E], float)
    T = float(Tn.sum() / N)              # reported (average) number of periods
    Ts = float(Tn.max())                 # T in the statistics: longest panel (Stata, also with unbalanced panels)
    w = 1 / L11
    a = np.array([float(e[:-1] @ np.diff(e)) for e in E])          # sum e_{t-1} Δe_t
    b = np.array([float(e[:-1] @ e[:-1]) for e in E])              # sum e_{t-1}^2
    num = a - Tn * lam
    p = _select_adf_lag(E, p0, crit) if crit else p0
    sst, aa, bb = [], [], []
    for e in E:
        Y, X = _adf_design(e, p, p + 1)
        ua = _resid(Y, X)
        sst.append(float(np.var(ua)))
        le, de = e[p:-1], np.diff(e)[p:]
        aa.append(float(le @ de)); bb.append(float(le @ le))
    sst, aa, bb = np.array(sst), np.array(aa), np.array(bb)
    tab = _PED["noconstant" if det == "noconstant" else det][C.k]
    sq = math.sqrt(N)

    def std(x, mu, v):
        return (x - mu * sq) / math.sqrt(v)

    if ar == "same":
        sig_t = float(np.mean(w * sig2))
        s_star = float(np.mean(w * sst))
        pv = Ts ** 2 * N ** 1.5 / float(np.sum(w * b))
        prho = Ts * sq * float(np.sum(w * num)) / float(np.sum(w * b))
        ptpp = float(np.sum(w * num)) / math.sqrt(sig_t * float(np.sum(w * b)))
        padf = float(np.sum(w * aa)) / math.sqrt(s_star * float(np.sum(w * bb)))
        st = [("pv", "Modified variance ratio", std(pv, tab[0], tab[1])),
              ("prho", "Modified Phillips–Perron t", std(prho, tab[2], tab[3])),
              ("ppp", "Phillips–Perron t", std(ptpp, tab[4], tab[5])),
              ("padf", "Augmented Dickey–Fuller t", std(padf, tab[4], tab[5]))]
    else:
        grho = Ts / sq * float(np.sum(num / b))
        gtpp = float(np.sum(num / np.sqrt(sig2 * b))) / sq
        gadf = float(np.sum(aa / np.sqrt(sst * bb))) / sq
        st = [("grho", "Modified Phillips–Perron t", std(grho, tab[6], tab[7])),
              ("gpp", "Phillips–Perron t", std(gtpp, tab[8], tab[9])),
              ("gadf", "Augmented Dickey–Fuller t", std(gadf, tab[8], tab[9]))]
    return {"test": "pedroni", "stats": [(k_, lab, v, _pval_abs(v)) for k_, lab, v in st], "N_g": N, "N_t": T,
            "N": int(Tn.sum()), "hac_lags": float(shown[-1]) if not per_panel_nw else float(np.mean(shown)),
            "hac_kernel": kern, "per_panel_nw": per_panel_nw,
            "hac_method": auto if kl is None else None, "adf_lags": p, "adf_method": crit, "det": det, "ar": ar,
            "demean": demean, "balanced": C.P.balanced, "k": C.k}


# ============================================================================ Westerlund (2005)

# Mean and variance (Theta, Psi) of the normalized variance-ratio statistics by number of regressors k = 1..7:
# (somepanels VR_G, allpanels VR_P) for "constant" and "trend". Westerlund (2005); the values are those used by
# Stata 19.5 xtcointtest, recovered to 8 decimals from Stata output on independent random walks (do-file
# moments.do in the verification folder) because the journal table was not available here.
_WEST = {
    "constant": {1: (0.03084814, 0.00052376, 0.04417266, 0.00161609), 2: (0.01946135, 0.00027834, 0.02952784, 0.00142304),
                 3: (0.01263815, 0.00013093, 0.01913412, 0.00104175), 4: (0.00885326, 0.00006222, 0.01290706, 0.00051325),
                 5: (0.00647340, 0.00003198, 0.00913085, 0.00027744), 6: (0.00488103, 0.00001718, 0.00670546, 0.00013565),
                 7: (0.00374555, 0.00000905, 0.00492034, 0.00006230)},
    "trend": {1: (0.00847774, 0.00002145, 0.01044646, 0.00004747), 2: (0.00655215, 0.00001480, 0.00817416, 0.00004034),
              3: (0.00515983, 0.00001031, 0.00643255, 0.00002876), 4: (0.00416246, 0.00000714, 0.00519797, 0.00002354),
              5: (0.00332133, 0.00000435, 0.00405747, 0.00001263), 6: (0.00272867, 0.00000297, 0.00331805, 0.00000895),
              7: (0.00227169, 0.00000193, 0.00273211, 0.00000561)},
}


def westerlund_test(df: pd.DataFrame, entity: str, time: str, dep: str, covs: List[str], trend: bool,
                    allpanels: bool, demean: bool) -> Dict[str, Any]:
    C = _CointPanel(df, entity, time, dep, covs, demean, "Westerlund 검정")
    if C.k > 7:
        raise ValueError("Westerlund 검정: 설명변수는 최대 7개 (Stata: number of regressors may not exceed 7)")
    N = C.N
    g = e2 = r = 0.0
    nobs = 0
    for y, X in zip(C.y, C.X):
        e = _resid(y, np.column_stack([X] + _det_cols(len(y), "trend" if trend else "constant")))
        Ecum = np.cumsum(e)
        g += float(Ecum @ Ecum) / float(e @ e)
        e2 += float(Ecum @ Ecum)
        r += float(e @ e)
        nobs += len(e)
    T = nobs / N                                   # reported (average) number of periods
    Tm = float(max(len(v) for v in C.y))           # T in the statistic: longest panel (Stata)
    th_g, ps_g, th_p, ps_p = _WEST["trend" if trend else "constant"][C.k]
    if allpanels:
        raw = e2 / r / Tm ** 2 * math.sqrt(N)
        z = (raw - th_p * math.sqrt(N)) / math.sqrt(ps_p)
    else:
        raw = g / Tm ** 2 / math.sqrt(N)
        z = (raw - th_g * math.sqrt(N)) / math.sqrt(ps_g)
    return {"test": "westerlund", "stats": [("vr", "Variance ratio", z, _pval_abs(z))], "N_g": N, "N_t": T, "N": nobs,
            "trend": trend, "allpanels": allpanels, "demean": demean, "balanced": C.P.balanced, "k": C.k}


# ============================================================================ result tables

TEST_LABEL = {"llc": "Levin–Lin–Chu", "ht": "Harris–Tzavalis", "breitung": "Breitung", "ips": "Im–Pesaran–Shin",
              "fisher": "Fisher형 (Choi 2001)", "hadri": "Hadri LM", "kao": "Kao", "pedroni": "Pedroni",
              "westerlund": "Westerlund"}
TEST_SOURCE = {
    "llc": "Levin, Lin & Chu (2002); Stata [XT] xtunitroot Methods and formulas",
    "ht": "Harris & Tzavalis (1999); Stata [XT] xtunitroot Methods and formulas",
    "breitung": "Breitung (2000), Breitung & Das (2005); Stata [XT] xtunitroot Methods and formulas",
    "ips": "Im, Pesaran & Shin (2003) 표 1–3; Stata [XT] xtunitroot Methods and formulas",
    "fisher": "Choi (2001), Maddala & Wu (1999); 패널별 검정 arch.unitroot ADF·PhillipsPerron (MacKinnon 1994 p값)",
    "hadri": "Hadri (2000); Stata [XT] xtunitroot Methods and formulas",
    "kao": "Kao (1999); Newey & West (1987, 1994) 장기공분산; Stata [XT] xtcointtest Methods and formulas",
    "pedroni": "Pedroni (1999) 표 1·2, Pedroni (2004); Stata [XT] xtcointtest Methods and formulas",
    "westerlund": "Westerlund (2005); Stata [XT] xtcointtest Methods and formulas",
}


def _yn(b: bool) -> str:
    return "포함" if b else "제외"


def _fmt_lags(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}"


def _info_table(name: str, title: str, rows: List[Tuple[str, str]]) -> ResultsTable:
    t = ResultsTable(name=name, title=title)
    t.add_column("item", "항목", col_type="text")
    t.add_column("value", "내용", col_type="text")
    for a, b in rows:
        t.add_row({"item": a, "value": b})
    return t


def _ptxt(p: Optional[float]) -> str:
    if p is None or not np.isfinite(p):
        return "p 없음"
    return "p < .001" if p < 0.001 else f"p = {p:.3f}"


def _model_of(trend: bool, noconstant: bool) -> int:
    return 3 if trend else (1 if noconstant else 2)


class PanelUnitRootAnalysis(BaseAnalysis):
    """Panel unit-root tests with Stata xtunitroot conventions (one or more variables)."""

    DEFAULTS = {"test": "llc", "trend": False, "noconstant": False, "demean": False, "lags": None, "kernel": None,
                "robust": False, "altt": False, "method": "dfuller", "drift": False, "plotSeries": True}

    def __init__(self, data: pd.DataFrame, entity: str, time: str, vars_: List[str], **opt):
        o = dict(self.DEFAULTS)
        o.update({k: v for k, v in opt.items() if v is not None or k in ("lags", "kernel")})
        test = resolve_choice("검정(test)", o["test"], UR_TESTS, UR_ALIASES, default="llc")
        o["test"] = test
        if not vars_:
            raise ValueError(f"패널 단위근 검정: 검정 변수(var) 지정 필요. 사용 가능한 변수: {list(data.columns)}")
        if o["trend"] and o["noconstant"]:
            raise ValueError("패널 단위근 검정: trend와 noconstant 동시 지정 불가 (Stata: cannot specify both noconstant and trend)")
        allowed = {"llc": {"trend", "noconstant", "demean", "lags", "kernel"},
                   "ht": {"trend", "noconstant", "demean", "altt"},
                   "breitung": {"trend", "noconstant", "demean", "robust", "lags"},
                   "ips": {"trend", "demean", "lags"},
                   "fisher": {"trend", "noconstant", "demean", "lags", "method", "drift"},
                   "hadri": {"trend", "demean", "robust", "kernel"}}[test]
        bad = [k for k in ("noconstant", "robust", "altt", "drift") if o.get(k) and k not in allowed]
        bad += [k for k in ("lags", "kernel") if o.get(k) not in (None, "") and k not in allowed]
        if bad:
            raise ValueError(f"xtunitroot {test}: 옵션 {', '.join(bad)} 사용 불가 (가능: {', '.join(sorted(allowed))})")
        if test == "fisher":
            o["method"] = resolve_choice("Fisher 검정 방식(method)", o.get("method") or "dfuller", ("dfuller", "pperron"),
                                         {"adf": "dfuller", "df": "dfuller", "pp": "pperron"})
        o.update({"entity": entity, "time": time, "vars": list(vars_)})
        super().__init__(name="panelUnitRoot",
                         title=f"패널 단위근 검정 — {TEST_LABEL[test]} (xtunitroot {test})", dataset=data, options=o)
        self.r: Dict[str, Dict[str, Any]] = {}
        self.syntax_python = self._py_syntax()
        self.syntax_r = "\n".join(r_code("panelUnitRoot", o) or [])
        self.run()

    def _py_syntax(self) -> str:
        o = self.options
        parts = [f"entity={o['entity']!r}", f"time={o['time']!r}",
                 f"var={o['vars'][0]!r}" if len(o["vars"]) == 1 else f"vars={o['vars']!r}", f"test={o['test']!r}"]
        for k, dv in self.DEFAULTS.items():
            if k == "test" or (k == "method" and o["test"] != "fisher"):
                continue
            if o.get(k) != dv and o.get(k) not in (None, ""):
                parts.append(f"{k}={o[k]!r}")
        return f"res = sess.run('panelUnitRoot', {', '.join(parts)})"

    def run(self):
        o = self.options
        multi = len(o["vars"]) > 1
        notes_txt = []
        for vi, var in enumerate(o["vars"]):
            sfx = f"_{vi + 1}" if multi else ""
            P = URPanel(self.dataset, o["entity"], o["time"], [var], "패널 단위근 검정")
            r = self._compute(P, var)
            self.r[var] = r
            self._tables(P, var, r, sfx, multi)
            notes_txt.append(self._narr(var, r))
            if o.get("plotSeries") and vi == 0:
                try:
                    self.add_plot(f"패널별 시계열 — {var}", _plot_series(P, var, o["demean"]))
                except Exception:
                    pass
        first = self.r[o["vars"][0]]
        self.model_stats = {"N": first.get("N"), "Groups": first.get("N_g")}
        self.apa_narrative = " ".join(notes_txt)

    def _compute(self, P: URPanel, var: str) -> Dict[str, Any]:
        o = self.options
        t = o["test"]
        model = _model_of(o["trend"], o["noconstant"])
        if t == "llc":
            return llc_test(P, var, model, o["demean"], o["lags"], o["kernel"])
        if t == "ht":
            return ht_test(P, var, model, o["demean"], o["altt"])
        if t == "breitung":
            lg = o["lags"]
            if lg not in (None, ""):
                lg, crit = _lag_spec(lg, 0, "Breitung", allow_ic=False)
            return breitung_test(P, var, model, o["demean"], int(lg or 0), o["robust"])
        if t == "ips":
            return ips_test(P, var, o["trend"], o["demean"], o["lags"])
        if t == "fisher":
            return fisher_test(P, var, o["method"], o["demean"], o["lags"], o["trend"], o["drift"], o["noconstant"])
        return hadri_test(P, var, o["trend"], o["demean"], o["robust"], o["kernel"])

    # ------------------------------------------------------------------ tables
    def _tables(self, P: URPanel, var: str, r: Dict[str, Any], sfx: str, multi: bool):
        o = self.options
        t = o["test"]
        vt = f" — {var}" if multi else f" — {var}"
        model = _model_of(o["trend"], o["noconstant"])
        h0 = "모든 패널이 정상" if t == "hadri" else ("모든 패널에 단위근" if t in ("ips", "fisher") else "패널에 단위근")
        ha = {"llc": "패널이 정상", "ht": "패널이 정상", "breitung": "패널이 정상", "ips": "일부 패널이 정상",
              "fisher": "적어도 한 패널이 정상", "hadri": "일부 패널에 단위근"}[t]
        nt = r["N_t"]
        per = (f"{nt:.0f}" if float(nt).is_integer() else f"{nt:.2f} (평균, 불균형 패널)")
        rows = [("H0", h0), ("Ha", ha), ("패널 수", str(r["N_g"])), ("기간 수", per)]
        if t != "hadri":
            rows.append(("AR 모수", "공통" if t in ("llc", "ht", "breitung") else "패널별"))
            rows.append(("패널 평균", _yn(model != 1)))
        rows.append(("시간 추세", _yn(model == 3)))
        if t == "fisher" and o["method"] == "dfuller":
            rows.append(("표류항(drift)", _yn(o["drift"])))
        if o["demean"]:
            rows.append(("횡단면 평균 제거", "예 (시점별 패널 평균 차감)"))
        asym = {"llc": "√N/T → 0" if model == 1 else "N/T → 0", "ht": "N → ∞, T 고정", "breitung": "T, N → ∞ (순차)",
                "ips": "T, N → ∞ (순차)", "fisher": "T → ∞", "hadri": "T, N → ∞ (순차)"}[t]
        rows.append(("점근 가정", asym))
        if t == "llc":
            rows.append(("ADF 회귀 시차", _fmt_lags(r["adf_lags"]) if not r["adf_method"]
                         else f"평균 {r['adf_lags']:.2f} (패널별 {r['adf_method'].upper()} 최소)"))
            lr = f"{KERNEL_LABEL[r['hac_kernel']]} 커널, 시차 " + (
                _fmt_lags(r["hac_lags"]) if not r["hac_method"] else
                f"평균 {r['hac_lags']:.2f} ({'LLC 규칙 int(3.21 T^1/3)' if r['hac_method'] == 'llc' else 'Newey–West 자동 선택'})")
            rows.append(("장기분산", lr))
        if t == "ips":
            if r.get("lags") is None:
                rows.append(("ADF 회귀 시차", "없음"))
            else:
                rows.append(("ADF 회귀 시차", _fmt_lags(r["lags"]) if not r["adf_method"]
                             else f"평균 {r['lags']:.2f} (패널별 {r['adf_method'].upper()} 최소)"))
        if t == "breitung":
            rows.append(("사전 백색화", f"시차 {r['lags']}" if r["lags"] else "없음"))
        if t == "fisher":
            rows.append(("패널별 검정", ("ADF (dfuller)" if o["method"] == "dfuller" else "Phillips–Perron (pperron)")
                         + f", 시차 {r['lags']}"))
        if t == "hadri":
            rows.append(("이분산", "강건" if (o["robust"] or r["kernel"]) else "비강건"))
            rows.append(("장기분산", f"{KERNEL_LABEL[r['kernel']]} 커널, 시차 {_fmt_lags(r['lags'])}" if r["kernel"]
                         else "미사용"))
        if t == "ht" and o["altt"]:
            rows.append(("소표본 조정", "T 대신 T − 1 사용 (altt)"))
        info = _info_table(f"info{sfx}", f"검정 개요{vt}", rows)
        info.add_note(f"계산 방식: {TEST_SOURCE[t]}")
        if P.time_note:
            info.add_note(P.time_note)
        self.add_table(info)

        st = ResultsTable(name=f"stats{sfx}", title=f"{TEST_LABEL[t]} 검정 통계량{vt}")
        st.add_column("stat", "통계량", col_type="text")
        st.add_column("value", "값", col_type="number")
        if t == "ht":
            st.add_column("z", "z", col_type="number")
        st.add_column("p", "p", col_type="number", fmt="pvalue")
        ips_cv = t == "ips" and r.get("lags") is None
        if ips_cv:
            for c, lab in (("cv1", "1%"), ("cv5", "5%"), ("cv10", "10%")):
                st.add_column(c, lab, col_type="number", super_title="고정 N 정확 임계값")
        nan = float("nan")
        if t == "llc":
            st.add_row({"stat": "Unadjusted t", "value": r["td"], "p": r["p_td"] if r["p_td"] is not None else nan})
            st.add_row({"stat": "Adjusted t*", "value": r["tds"], "p": r["p_tds"]})
            st.add_note("H0: δ = 0 (단위근). 조정 t*: 표준정규, 작은 값이 H0 기각 근거. 비조정 t의 p값은 패널 평균·추세가 없을 때만 산출 (Stata 동일)")
            st.add_note(f"부가 결과: δ = {r['delta']:.4f}, se(δ) = {r['se_delta']:.4f}, 평균 T̃ = {r['ttilde']:.2f}, "
                        f"S_N = {r['sbar']:.4f}, μ* = {r['mu_adj']:.4f}, σ* = {r['sig_adj']:.4f}")
            if model == 3 and 35 < r["ttilde"] < 45:
                st.add_note("조정값 σ*: Stata xtunitroot 표의 T̃ = 40, 추세 모형 값 0.971 사용 (LLC 2002 표 2 인쇄값 0.871)")
            if model == 1 and r["adf_method"]:
                st.add_note("정보기준 시차 선택 회귀에 상수항 포함 (Stata xtunitroot 방식); 최종 회귀는 상수항 제외")
        elif t == "ht":
            st.add_row({"stat": "rho", "value": r["rho"], "z": r["z"], "p": r["p"]})
            st.add_note(f"H0: ρ = 1. H0 아래 E(ρ̂) = {r['mean_rho']:.4f}, Var(ρ̂) = {r['Var_rho']:.6f} "
                        "(Harris & Tzavalis 1999 정리 1, N → ∞, T 고정); 작은 z가 H0 기각 근거")
        elif t == "breitung":
            if o["robust"]:
                st.add_row({"stat": "lambda*", "value": r["lrobust"], "p": r["p_lrobust"]})
                st.add_note("* 횡단면 상관에 강건한 lambda (Breitung & Das 2005)")
            else:
                st.add_row({"stat": "lambda", "value": r["lambda"], "p": r["p"]})
            st.add_note("lambda: 표준정규 (T → ∞ 다음 N → ∞), 작은 값이 H0 기각 근거")
        elif t == "ips":
            if r.get("lags") is None:
                cv = r["cv"] or [nan, nan, nan]
                st.add_row({"stat": "t-bar", "value": r["tbar"], "p": nan, "cv1": cv[0], "cv5": cv[1], "cv10": cv[2]})
                st.add_row({"stat": "t-tilde-bar", "value": r["ttildebar"], "p": nan, "cv1": nan, "cv5": nan, "cv10": nan})
                pz = r["p_zttildebar"]
                st.add_row({"stat": "Z-t-tilde-bar", "value": r["zttildebar"], "p": pz if pz is not None else nan,
                            "cv1": nan, "cv5": nan, "cv10": nan})
                st.add_note(f"Z-t-bar = {r['zt']:.4f} (점근 정규분포가 아니어서 p값 미제시, Stata r(zt))")
                if r["cv"] is None:
                    st.add_note("불균형 패널: 고정 N 정확 임계값(IPS 2003 표 2) 없음")
                else:
                    st.add_note("임계값: IPS (2003) 표 2에서 N, T 이상인 첫 행·열 값 (Stata xtunitroot와 같음)")
                if pz is None:
                    need = 10 if not P.balanced else (7 if o["trend"] else 6)
                    st.add_note(f"Z-t-tilde-bar 정규성: 패널당 관측치 {need}개 이상 필요 — p값 미제시")
                if o["trend"]:
                    st.add_note("추세 포함 시에도 t-tilde 모멘트는 IPS (2003) 표 1의 상수항 모형 값 사용 (Stata xtunitroot와 같음)")
            else:
                st.add_row({"stat": "W-t-bar", "value": r["wtbar"], "p": r["p_wtbar"]})
                st.add_note("W-t-bar: E[t]·Var[t]는 IPS (2003) 표 3에서 패널별 T·시차로 선형 보간 (T → ∞ 다음 N → ∞)")
        elif t == "fisher":
            st.add_row({"stat": f"Inverse chi-squared({r['df_P']}) P", "value": r["P"], "p": r["p_P"]})
            st.add_row({"stat": "Inverse normal Z", "value": r["Z"], "p": r["p_Z"]})
            st.add_row({"stat": f"Inverse logit t({r['df_L']}) L*", "value": r["L"], "p": r["p_L"]})
            st.add_row({"stat": "Modified inv. chi-squared Pm", "value": r["Pm"], "p": r["p_Pm"]})
            st.add_note("P 통계량: 유한 패널 수 가정. 나머지: 유한·무한 패널 수 모두 적용")
            st.add_note(f"L* 척도 k = 3(5N+3)/(π²N(5N+2)) (Stata 19.5 xtunitroot); Choi (2001)의 k = 3(5N+4)/(π²N(5N+2)) "
                        f"적용 시 L* = {r['L_choi']:.4f}")
            if r.get("failed"):
                st.add_note(f"관측치 부족으로 패널별 통계량 계산 불가 {len(r['failed'])}개 ({', '.join(r['failed'][:10])}): "
                            "Stata dfuller·pperron과 같이 p = 1로 처리")
            if r.get("n_dropped_zl"):
                st.add_note(f"p = 1인 패널 {r['n_dropped_zl']}개: Z·L* 계산에서 제외 (Stata와 같음)")
            if o["method"] == "dfuller" and o["drift"]:
                st.add_note("drift: 패널별 p값은 t 분포 단측 p값 (Stata dfuller, drift)")
            else:
                st.add_note("패널별 p값: MacKinnon (1994) 근사 (arch mackinnonp; Stata dfuller·pperron과 같음)")
        else:
            st.add_row({"stat": "z", "value": r["z"], "p": r["p"]})
            st.add_note(f"LM = {r['LM']:.4f}; H0 아래 평균 {r['mu']:.4f}, 분산 {r['var']:.6f}; 큰 z가 정상성 귀무가설 기각 근거")
        self.add_table(st)

    def _narr(self, var: str, r: Dict[str, Any]) -> str:
        t = self.options["test"]
        lead = f"{TEST_LABEL[t]} 패널 단위근 검정({var}, 패널 {r['N_g']}개)"
        if t == "llc":
            return f"{lead}: 조정 t* = {r['tds']:.2f}, {_ptxt(r['p_tds'])}."
        if t == "ht":
            return f"{lead}: ρ̂ = {r['rho']:.3f}, z = {r['z']:.2f}, {_ptxt(r['p'])}."
        if t == "breitung":
            v, p = (r["lrobust"], r["p_lrobust"]) if self.options["robust"] else (r["lambda"], r["p"])
            return f"{lead}: λ = {v:.2f}, {_ptxt(p)}."
        if t == "ips":
            if r.get("lags") is None:
                return f"{lead}: Z-t-tilde-bar = {r['zttildebar']:.2f}, {_ptxt(r['p_zttildebar'])}."
            return f"{lead}: W-t-bar = {r['wtbar']:.2f}, {_ptxt(r['p_wtbar'])}."
        if t == "fisher":
            return f"{lead}: 역정규 Z = {r['Z']:.2f}, {_ptxt(r['p_Z'])}; P = {r['P']:.2f}, {_ptxt(r['p_P'])}."
        return f"{lead}: z = {r['z']:.2f}, {_ptxt(r['p'])} (귀무가설: 모든 패널이 정상)."


def _plot_series(P: URPanel, var: str, demean: bool) -> str:
    import matplotlib.pyplot as plt
    from .plots import _clean_ax, _col, _hdr_col, fig_to_base64
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    vals = P.values(var, demean)
    tv = P.df[P.time].to_numpy()
    for i, s in enumerate(P.slices):
        ax.plot(tv[s], vals[s], "-", linewidth=0.9, alpha=0.45, color=_col(i % 8))
    m = pd.Series(vals).groupby(tv).mean()
    ax.plot(m.index, m.to_numpy(), "-", linewidth=2.4, color=_hdr_col(), label="시점별 패널 평균")
    ax.set_xlabel(P.time)
    ax.set_ylabel(var + (" (횡단면 평균 제거)" if demean else ""))
    ax.set_title(f"패널별 시계열 — {var} (패널 {P.n}개)", color=_hdr_col())
    ax.legend(frameon=False, fontsize=8, loc="best")
    _clean_ax(ax)
    return fig_to_base64(fig)


class PanelCointAnalysis(BaseAnalysis):
    """Panel cointegration tests with Stata xtcointtest conventions."""

    DEFAULTS = {"test": "kao", "trend": False, "noconstant": False, "demean": False, "lags": None, "kernel": None,
                "ar": "panelspecific", "allpanels": False, "nwPerPanel": False}

    def __init__(self, data: pd.DataFrame, entity: str, time: str, dep: str, covs: List[str], **opt):
        o = dict(self.DEFAULTS)
        o.update({k: v for k, v in opt.items() if v is not None or k in ("lags", "kernel")})
        test = resolve_choice("검정(test)", o["test"], CO_TESTS, {"kao1999": "kao", "pedroni1999": "pedroni",
                                                                   "westerlund2005": "westerlund"}, default="kao")
        o["test"] = test
        o["ar"] = resolve_choice("AR 모수(ar)", o.get("ar") or "panelspecific", ("panelspecific", "same"),
                                 {"panels": "panelspecific", "panel": "panelspecific", "panel specific": "panelspecific",
                                  "group": "panelspecific", "common": "same", "within": "same", "between": "panelspecific"})
        if o["trend"] and o["noconstant"]:
            raise ValueError("패널 공적분 검정: trend와 noconstant 동시 지정 불가 (Stata: cannot specify both noconstant and trend)")
        allowed = {"kao": {"demean", "lags", "kernel"},
                   "pedroni": {"trend", "noconstant", "demean", "lags", "kernel", "ar", "nwPerPanel"},
                   "westerlund": {"trend", "demean", "allpanels"}}[test]
        bad = [k for k in ("trend", "noconstant", "allpanels", "nwPerPanel") if o.get(k) and k not in allowed]
        bad += [k for k in ("lags", "kernel") if o.get(k) not in (None, "") and k not in allowed]
        if o["ar"] != "panelspecific" and "ar" not in allowed:
            bad.append("ar")
        if bad:
            raise ValueError(f"xtcointtest {test}: 옵션 {', '.join(bad)} 사용 불가 (가능: {', '.join(sorted(allowed))})")
        o.update({"entity": entity, "time": time, "dep": dep, "covs": list(covs)})
        super().__init__(name="panelCoint", title=f"패널 공적분 검정 — {TEST_LABEL[test]} (xtcointtest {test})",
                         dataset=data, options=o)
        self.r: Dict[str, Any] = {}
        self.syntax_python = self._py_syntax()
        self.syntax_r = "\n".join(r_code("panelCoint", o) or [])
        self.run()

    def _py_syntax(self) -> str:
        o = self.options
        parts = [f"entity={o['entity']!r}", f"time={o['time']!r}", f"dep={o['dep']!r}", f"covs={o['covs']!r}",
                 f"test={o['test']!r}"]
        for k, dv in self.DEFAULTS.items():
            if k != "test" and o.get(k) != dv and o.get(k) not in (None, ""):
                parts.append(f"{k}={o[k]!r}")
        return f"res = sess.run('panelCoint', {', '.join(parts)})"

    def run(self):
        o = self.options
        t = o["test"]
        if t == "kao":
            r = kao_test(self.dataset, o["entity"], o["time"], o["dep"], o["covs"], o["demean"], o["lags"], o["kernel"])
        elif t == "pedroni":
            det = "trend" if o["trend"] else ("noconstant" if o["noconstant"] else "constant")
            r = pedroni_test(self.dataset, o["entity"], o["time"], o["dep"], o["covs"], det, o["ar"], o["demean"],
                             o["lags"], o["kernel"], per_panel_nw=bool(o["nwPerPanel"]))
        else:
            r = westerlund_test(self.dataset, o["entity"], o["time"], o["dep"], o["covs"], o["trend"], o["allpanels"],
                                o["demean"])
        self.r = r
        self._tables(r)
        self.model_stats = {"N": r["N"], "Groups": r["N_g"]}
        lead = f"{TEST_LABEL[t]} 패널 공적분 검정({o['dep']} ~ {' + '.join(o['covs'])}, 패널 {r['N_g']}개)"
        self.apa_narrative = lead + ": " + "; ".join(f"{lab} = {v:.2f} ({_ptxt(p)})" for _, lab, v, p in r["stats"]) + "."

    def _tables(self, r: Dict[str, Any]):
        o = self.options
        t = o["test"]
        nt = r["N_t"]
        per = f"{nt:.0f}" if float(nt).is_integer() else f"{nt:.2f} (평균, 불균형 패널)"
        ha = ("일부 패널에서 공적분" if (t == "westerlund" and not o["allpanels"]) else "모든 패널에서 공적분")
        rows = [("H0", "공적분 없음"), ("Ha", ha), ("패널 수", str(r["N_g"])), ("기간 수", per),
                ("공적분 벡터", "공통" if t == "kao" else "패널별"),
                ("패널 평균", _yn(not o["noconstant"])), ("시간 추세", _yn(o["trend"]))]
        if t == "kao":
            ar_lab = "공통"
        elif t == "pedroni":
            ar_lab = "공통 (panel 통계량)" if o["ar"] == "same" else "패널별 (group-mean 통계량)"
        else:
            ar_lab = "공통 (allpanels)" if o["allpanels"] else "패널별 (somepanels)"
        rows.append(("AR 모수", ar_lab))
        if t in ("kao", "pedroni"):
            how = ("Newey–West 자동 선택" if r["hac_method"] else "지정")
            rows.append(("장기분산", f"{KERNEL_LABEL[r['hac_kernel']]} 커널, 시차 {_fmt_lags(r['hac_lags'])} ({how})"))
            rows.append(("ADF 추가 시차", f"{r['adf_lags']}" + (f" ({r['adf_method'].upper()} 최소)" if r["adf_method"] else "")))
        if o["demean"]:
            rows.append(("횡단면 평균 제거", "예 (시점별 패널 평균 차감)"))
        info = _info_table("info", "검정 개요", rows)
        info.add_note(f"계산 방식: {TEST_SOURCE[t]}")
        self.add_table(info)
        st = ResultsTable(name="stats", title=f"{TEST_LABEL[t]} 공적분 검정 통계량")
        st.add_column("stat", "통계량", col_type="text")
        st.add_column("value", "값", col_type="number")
        st.add_column("p", "p", col_type="number", fmt="pvalue")
        for _, lab, v, p in r["stats"]:
            st.add_row({"stat": lab, "value": v, "p": p})
        st.add_note("모든 통계량: 표준화 후 점근 표준정규. p값 1 − Φ(|z|) (Stata xtcointtest 방식)")
        wrong = [lab for key, lab, v, _ in r["stats"] if (v < 0 if key == "pv" else v > 0)]
        if wrong:
            st.add_note("기각 방향(분산비 +, 그 밖의 통계량 −)과 부호가 반대인 통계량: " + ", ".join(wrong)
                        + " — 이 경우 p값은 공적분 근거로 해석 불가")
        if t == "kao":
            st.add_note(f"부가 결과: ρ̂ = {r['rho']:.4f}, σ²_ν = {r['sigmasq_nu']:.4f}, σ²_0ν = {r['sigmasq0_nu']:.4f}; "
                        "u·ε는 y·x의 통합 AR(1) 회귀(상수항 없음) 잔차, 장기공분산은 패널별 추정의 평균")
        if t == "pedroni":
            st.add_note("조정 평균·분산: 설명변수 2–7개 Pedroni (1999) 표 2, 1개 Pedroni (2004)")
            if r["hac_method"] and not o["nwPerPanel"]:
                st.add_note("자동 시차: 첫 패널의 Newey–West 대역폭에서 정한 시차를 모든 패널의 장기분산(L11, 잔차)에 적용 "
                            "(Stata 19.5 xtcointtest 방식). nwPerPanel=True: 패널별 선택")
            if o["noconstant"] and r["k"] == 7 and o["ar"] == "same":
                st.add_note("설명변수 7개·상수항 없음의 panel rho 평균: Stata 값 +25.889 사용 (Pedroni 1999 표 2: −25.889)")
        if t == "westerlund":
            st.add_note("조정 평균·분산: Westerlund (2005); 수치는 Stata 19.5 xtcointtest와 같은 값(소수 8자리)")
        self.add_table(st)


# ============================================================================ registry glue


def _det_flags(p: Dict[str, Any]) -> Tuple[bool, bool]:
    """trend / noconstant flags, also from a single select `deterministics` = constant | trend | none."""
    d = _opt(p, "deterministics", "det")
    if d is not None:
        d = resolve_choice("결정항(deterministics)", d, ("constant", "trend", "none"),
                           {"noconstant": "none", "nocons": "none", "c": "constant", "ct": "trend", "n": "none"})
        return d == "trend", d == "none"
    return _flag(p, "trend"), _flag(p, "noconstant", "nocons")


def _kernel_param(p: Dict[str, Any]) -> Any:
    """kernel() text, or kernelType + kernelLags from the form."""
    k = _opt(p, "kernel")
    if k is None and _opt(p, "kernelType") is not None:
        lag = _opt(p, "kernelLags")
        k = str(_opt(p, "kernelType")) + (f" {lag}" if lag not in (None, "") else "")
    return k


# Dialog keys (web form): each is read only for the test it belongs to, so values left over from another test do not
# raise errors; the Stata-style keys (trend, lags, kernel, robust, ...) are checked strictly.
_UR_GUI = {"detCommon": ("deterministics", ("llc", "ht", "breitung")),
           "detSpecific": ("deterministics", ("ips", "fisher", "hadri")),
           "adfLags": ("lags", ("llc", "ips")), "breitungLags": ("lags", ("breitung",)), "fisherLags": ("lags", ("fisher",)),
           "llcKernel": ("kernelType", ("llc",)), "llcKernelLags": ("kernelLags", ("llc",)),
           "hadriKernel": ("kernelType", ("hadri",)), "hadriKernelLags": ("kernelLags", ("hadri",)),
           "breitungRobust": ("robust", ("breitung",)), "hadriRobust": ("robust", ("hadri",)), "htAltt": ("altt", ("ht",)),
           "fisherMethod": ("method", ("fisher",)), "fisherDrift": ("drift", ("fisher",))}
_CO_GUI = {"detPedroni": ("deterministics", ("pedroni",)), "detWesterlund": ("deterministics", ("westerlund",)),
           "coLags": ("lags", ("kao", "pedroni")), "coKernel": ("kernelType", ("kao", "pedroni")),
           "coKernelLags": ("kernelLags", ("kao", "pedroni")), "pedroniAr": ("ar", ("pedroni",)),
           "westerlundAll": ("allpanels", ("westerlund",)), "pedroniNw": ("nwPerPanel", ("pedroni",))}


def _apply_gui(p: Dict[str, Any], table: Dict[str, Tuple[str, Tuple[str, ...]]], test: str) -> Dict[str, Any]:
    q = {k: v for k, v in p.items() if k not in table}
    for k, (canon, tests) in table.items():
        if k in p and test in tests and p[k] not in (None, "") and q.get(canon) in (None, "", False):
            q[canon] = p[k]
    return q


def _ur_kwargs(p: Dict[str, Any]) -> Dict[str, Any]:
    test0 = resolve_choice("검정(test)", _opt(p, "test", default="llc"), UR_TESTS, UR_ALIASES, default="llc")
    p = _apply_gui(p, _UR_GUI, test0)
    vars_ = _as_list(_opt(p, "vars", "var", "variable", "dep", "varname", default=[]))
    trend, nocons = _det_flags(p)
    return dict(entity=_as_single(_opt(p, "entity", "entity_col", "idVar", "panelVar")),
                time=_as_single(_opt(p, "time", "time_col", "timeVar")), vars_=vars_,
                test=str(_opt(p, "test", "method_test", default="llc")),
                trend=trend, noconstant=nocons, demean=_flag(p, "demean"),
                lags=_opt(p, "lags", "lag"), kernel=_kernel_param(p), robust=_flag(p, "robust"), altt=_flag(p, "altt"),
                method=str(_opt(p, "method", "urtest", default="dfuller")), drift=_flag(p, "drift"),
                plotSeries=_flag(p, "plotSeries", "plot", default=True))


def _co_kwargs(p: Dict[str, Any]) -> Dict[str, Any]:
    test0 = resolve_choice("검정(test)", _opt(p, "test", default="kao"), CO_TESTS, default="kao")
    p = _apply_gui(p, _CO_GUI, test0)
    return dict(entity=_as_single(_opt(p, "entity", "entity_col", "idVar", "panelVar")),
                time=_as_single(_opt(p, "time", "time_col", "timeVar")), dep=_as_single(_opt(p, "dep", "depvar")),
                covs=_as_list(_opt(p, "covs", "indeps", "indepvars", default=[])),
                test=str(_opt(p, "test", default="kao")), trend=_det_flags(p)[0],
                noconstant=_det_flags(p)[1], demean=_flag(p, "demean"), lags=_opt(p, "lags", "lag"),
                kernel=_kernel_param(p), ar=str(_opt(p, "ar", default="panelspecific")),
                allpanels=_flag(p, "allpanels"), nwPerPanel=_flag(p, "nwPerPanel", "nw_per_panel"))


_IDS = {"panelUnitRoot": "panelUnitRoot", "xtunitroot": "panelUnitRoot", "panel_unit_root": "panelUnitRoot",
        "panelCoint": "panelCoint", "xtcointtest": "panelCoint", "panel_coint": "panelCoint", "xtcoint": "panelCoint"}


def dispatch(t: str, df: pd.DataFrame, params: Dict[str, Any]):
    t = _IDS.get(t)
    if t == "panelUnitRoot":
        kw = _ur_kwargs(params or {})
        return PanelUnitRootAnalysis(df, **kw)
    if t == "panelCoint":
        kw = _co_kwargs(params or {})
        return PanelCointAnalysis(df, **kw)
    return None


# ============================================================================ code export


def _q(s: Any) -> str:
    return '"' + str(s) + '"'


def stata_code(t: str, p: Dict[str, Any]) -> Optional[List[str]]:
    t = _IDS.get(t)
    if t is None:
        return None
    if t == "panelUnitRoot":
        kw = _ur_kwargs(p)
        test = resolve_choice("test", kw["test"], UR_TESTS, UR_ALIASES, default="llc")
        out = xtset_lines(kw["entity"], kw["time"])
        opts = []
        if test == "fisher":
            opts.append(resolve_choice("method", kw["method"], ("dfuller", "pperron"),
                                       {"adf": "dfuller", "df": "dfuller", "pp": "pperron"}))
        for k in ("trend", "noconstant", "drift", "demean", "robust", "altt"):
            if kw[k]:
                opts.append(k)
        if kw["lags"] not in (None, ""):
            opts.append(f"lags({kw['lags']})")
        if kw["kernel"] not in (None, ""):
            kv = kw["kernel"] if isinstance(kw["kernel"], str) else " ".join(map(str, kw["kernel"]))
            opts.append(f"kernel({kv})")
        for v in kw["vars_"]:
            out.append(f"xtunitroot {test} {v}" + (", " + " ".join(opts) if opts else ""))
        return out
    kw = _co_kwargs(p)
    test = resolve_choice("test", kw["test"], CO_TESTS, default="kao")
    out = xtset_lines(kw["entity"], kw["time"])
    opts = [k for k in ("trend", "noconstant", "demean", "allpanels") if kw[k]]
    if kw["lags"] not in (None, ""):
        opts.append(f"lags({kw['lags']})")
    if kw["kernel"] not in (None, ""):
        kv = kw["kernel"] if isinstance(kw["kernel"], str) else " ".join(map(str, kw["kernel"]))
        opts.append(f"kernel({kv})")
    if test == "pedroni" and str(kw["ar"]).lower().startswith(("same", "common", "within")):
        opts.append("ar(same)")
    if test == "pedroni" and kw["nwPerPanel"]:
        out.append("* nwPerPanel(패널별 Newey–West 대역폭)에 해당하는 Stata 옵션 없음 — Stata는 첫 패널의 대역폭을 모든 패널에 적용")
    out.append(f"xtcointtest {test} {kw['dep']} {' '.join(kw['covs'])}" + (", " + " ".join(opts) if opts else ""))
    return out


def r_code(t: str, p: Dict[str, Any]) -> Optional[List[str]]:
    t = _IDS.get(t)
    if t is None:
        return None
    if t == "panelCoint":
        kw = _co_kwargs(p)
        test = resolve_choice("test", kw["test"], CO_TESTS, default="kao")
        if test == "pedroni":
            return ["# xtcointtest pedroni: R에 같은 옵션 구조의 함수 없음",
                    "# 참고: pco 패키지 pco::pedroni99m() — Pedroni (1999) panel/group 통계량 7개, 입력은 시점 × 개체 × 변수 배열"
                    " (첫 변수가 종속변수); 장기분산 커널·시차 규칙, 추세 처리, Stata의 첫 패널 대역폭 규칙과 차이"]
        name = {"kao": "Kao (1999)", "westerlund": "Westerlund (2005) 분산비"}[test]
        return [f"# xtcointtest {test}: {name} 패널 공적분 검정 — R에 동일 함수 없음"]
    kw = _ur_kwargs(p)
    test = resolve_choice("test", kw["test"], UR_TESTS, UR_ALIASES, default="llc")
    e, tm = kw["entity"], kw["time"]
    out: List[str] = []
    if test in ("ht", "breitung"):
        return [f"# xtunitroot {test}: {TEST_LABEL[test]} 검정 — R plm::purtest에 없음 (R에 동일 함수 없음)"]
    vs = kw["vars_"]
    if kw["demean"]:
        for v in vs:
            out.append(f"data${v}_dm <- data${v} - ave(data${v}, data${tm})   # 시점별 횡단면 평균 제거 (demean)")
        vs = [f"{v}_dm" for v in vs]
    out.append(f"pdata <- plm::pdata.frame(data, index = c({_q(e)}, {_q(tm)}))")
    exo = "trend" if kw["trend"] else ("none" if kw["noconstant"] else "intercept")
    lag_txt = ""
    if kw["lags"] not in (None, ""):
        p0, crit = _lag_spec(kw["lags"], 1, "lags")
        if crit:
            lag_txt = f", lags = {_q({'aic': 'AIC', 'bic': 'SIC', 'hqic': 'Hall'}[crit])}, pmax = {p0}"
            if crit == "hqic":
                out.append("# HQIC 선택에 해당하는 plm 옵션 없음 — 'Hall'(일반→특수 t 검정) 사용, 선택 시차가 다를 수 있음")
        else:
            lag_txt = f", lags = {p0}"
    for v in vs:
        if test == "llc":
            if kw["kernel"] not in (None, "") and not str(kw["kernel"]).lower().startswith("ba"):
                out.append("# plm levinlin의 장기분산은 Bartlett 커널만 지원")
            out.append(f"plm::purtest(pdata${v}, test = \"levinlin\", exo = {_q(exo)}{lag_txt or ', lags = 1'})"
                       "  # 장기분산 시차 규칙·T̃ 조정표 보간이 Stata와 다를 수 있음")
        elif test == "ips":
            stat = "Wtbar" if kw["lags"] not in (None, "") else "Ztbar"
            out.append(f"plm::purtest(pdata${v}, test = \"ips\", exo = {_q(exo)}{lag_txt or ', lags = 0'}, ips.stat = {_q(stat)})"
                       "  # 표 보간·불균형 패널 처리 차이 가능")
        elif test == "fisher":
            if kw["method"].lower().startswith("pp"):
                out.append("# plm Fisher형 검정은 ADF 기반만 지원 — pperron에 해당하는 함수 없음 (아래는 ADF 기준)")
            if kw["drift"]:
                out.append("# dfuller drift(t 분포 p값)에 해당하는 plm 옵션 없음 — exo = \"intercept\" (MacKinnon p값)")
            for tt, lab in (("madwu", "P (역카이제곱)"), ("invnormal", "Z (역정규)"), ("logit", "L* (역로짓)"), ("Pm", "Pm (수정 역카이제곱)")):
                out.append(f"plm::purtest(pdata${v}, test = {_q(tt)}, exo = {_q(exo)}{lag_txt})  # {lab}; p값 근사 방식 차이 가능")
        else:
            if kw["kernel"] not in (None, ""):
                out.append("# plm hadri는 커널 장기분산(kernel()) 미지원")
            out.append(f"plm::purtest(pdata${v}, test = \"hadri\", exo = {_q('trend' if kw['trend'] else 'intercept')}, "
                       f"Hcons = {'TRUE' if kw['robust'] else 'FALSE'})  # Hcons = TRUE: 이분산 강건 (Stata robust)")
    return out


# ============================================================================ Stata command input

_UR_OPTS = {"trend": 1, "noconstant": 3, "demean": 6, "lags": 1, "kernel": 3, "robust": 1, "altt": 4,
            "dfuller": 3, "pperron": 2, "drift": 2, "regress": 3}
_CO_OPTS = {"trend": 1, "noconstant": 6, "demean": 6, "lags": 1, "kernel": 3, "ar": 2, "somepanels": 4, "allpanels": 3}


def _canon(name: str, table: Dict[str, int]) -> Optional[str]:
    hits = [k for k, m in table.items() if k.startswith(name) and len(name) >= m]
    return hits[0] if len(hits) == 1 else None


def transpile(line: str) -> Optional[str]:
    s = line.strip()
    m = re.match(r"^(xtunitroot|xtcointtest)\s+(\w+)\s*([^,]*?)\s*(?:,\s*(.*))?$", s, re.IGNORECASE)
    if not m:
        return None
    cmd, test, body, opt_txt = m.group(1).lower(), m.group(2).lower(), m.group(3).strip(), m.group(4) or ""
    if re.search(r"\b(if|in)\b", body):
        return _err(f"{cmd}: if·in 조건 미지원 — keep if 등으로 자료를 먼저 선택한 뒤 실행: {s}")
    toks = body.split()
    if any(re.search(r"[.#]", v) for v in toks):
        return _err(f"{cmd}: 시계열 연산자(D.·L. 등) 미지원 — 변수를 만든 뒤 지정 (예: gen dy = D.y): {s}")
    opts = _split_opts(opt_txt)
    kw: List[str] = []
    if cmd == "xtunitroot":
        if test not in UR_TESTS:
            return _err(f"xtunitroot: 검정 '{test}' 미지원 (가능: {' '.join(UR_TESTS)}) — {s}")
        if len(toks) != 1:
            return _err(f"xtunitroot: 변수 1개 지정 (Stata: varname required) — {s}")
        kw += [f"var={toks[0]!r}", f"test={test!r}"]
        for raw, val in opts.items():
            name = _canon(raw, _UR_OPTS)
            if name is None:
                return _err(f"xtunitroot {test}: 옵션 '{raw}' 미지원 — {s}")
            if name in ("lags", "kernel"):
                v = val.strip()
                if name == "lags" and re.fullmatch(r"\d+", v):
                    kw.append(f"lags={int(v)}")
                else:
                    kw.append(f"{name}={v!r}")
            elif name in ("dfuller", "pperron"):
                kw.append(f"method={name!r}")
            elif name == "regress":
                continue                       # no effect inside xtunitroot fisher (Stata)
            else:
                kw.append(f"{name}=True")
        return f"res = sess.run('panelUnitRoot', {_xt_index()}, {', '.join(kw)})"
    if test not in CO_TESTS:
        return _err(f"xtcointtest: 검정 '{test}' 미지원 (가능: kao pedroni westerlund) — {s}")
    if len(toks) < 2:
        return _err(f"xtcointtest {test}: 종속변수와 설명변수 1개 이상 필요 — {s}")
    kw += [f"dep={toks[0]!r}", f"covs={toks[1:]!r}", f"test={test!r}"]
    for raw, val in opts.items():
        name = _canon(raw, _CO_OPTS)
        if name is None:
            return _err(f"xtcointtest {test}: 옵션 '{raw}' 미지원 — {s}")
        if name == "lags":
            v = val.strip()
            kw.append(f"lags={int(v)}" if re.fullmatch(r"\d+", v) else f"lags={v!r}")
        elif name == "kernel":
            kw.append(f"kernel={val.strip()!r}")
        elif name == "ar":
            v = val.strip().lower()
            if v == "same":
                kw.append("ar='same'")
            elif v and not "panelspecific".startswith(v[:6]):
                return _err(f"xtcointtest pedroni: ar({val}) 미지원 (panelspecific 또는 same) — {s}")
        elif name == "somepanels":
            continue
        else:
            kw.append(f"{name}=True")
    return f"res = sess.run('panelCoint', {_xt_index()}, {', '.join(kw)})"


def _xt_index() -> str:
    return ("entity=getattr(sess, 'panel_spec', {}).get('idVar'), "
            "time=getattr(sess, 'panel_spec', {}).get('timeVar')")


EXTENSION = {
    "catalog": [
        {"id": "panelUnitRoot", "group": "Time Series",
         "title": "Panel Unit-Root Tests (xtunitroot llc/ht/breitung/ips/fisher/hadri)", "module": "openstat",
         "icon": "analysis-regression.svg"},
        {"id": "panelCoint", "group": "Time Series",
         "title": "Panel Cointegration Tests (xtcointtest kao/pedroni/westerlund)", "module": "openstat",
         "icon": "analysis-regression.svg"},
    ],
    "aliases": {"xtunitroot": "panelUnitRoot", "panel_unit_root": "panelUnitRoot", "panelunitroot": "panelUnitRoot",
                "xtcointtest": "panelCoint", "panel_coint": "panelCoint", "xtcoint": "panelCoint",
                "panelcoint": "panelCoint"},
    "dispatch": dispatch,
    "stata": stata_code,
    "r": r_code,
    "transpile": transpile,
}
