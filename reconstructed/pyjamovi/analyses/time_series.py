"""
OpenStat Studio — time-series analysis following Stata conventions.
License: MIT

arima     Stata `arima y [x], arima(p,d,q) sarima(P,D,Q,s) [noconstant] vce(opg|oim|robust)`:
          regression with (seasonal) ARMA disturbances, exact Gaussian ML through the Kalman filter.
          With d, D > 0 the dependent variable and the regressors are differenced (as Stata's arima(p,d,q) option)
          and the constant is the mean of the differenced series (Stata), not the ARMA intercept.
unitRoot  Stata `dfuller` (ADF), `pperron` (Phillips–Perron Z(ρ), Z(t)) and KPSS.
corrgram  Stata `corrgram` / `ac` / `pac` / `wntestq`.

Computation
- statsmodels: SARIMAX (likelihood, Kalman filter, OPG / OIM / robust covariance, forecasts), acf, pacf (ols = Stata
  regression PAC, ywm = Yule–Walker), acorr_ljungbox.
- arch: ADF, PhillipsPerron, KPSS (statistics, MacKinnon p-values, automatic lag selection, KPSS critical values).
- Own code (Stata conventions): time index (tsset) checks, differencing of y and x and the regression constant
  (ARMAX parameterization), level forecasts from the integrated model, /sigma and its delta-method SE, Fuller (1976)
  critical values interpolated in N as Stata reports them, the dfuller drift rule (t distribution), Stata output layout.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseAnalysis, ResultsTable, resolve_choice


# ---------------------------------------------------------------- option helpers
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


def _int(p: Dict[str, Any], *keys, default: Optional[int] = None, lo: int = 0, label: str = "") -> Optional[int]:
    v = _opt(p, *keys, default=default)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{label or keys[0]}: 정수 필요 (입력 '{v}')")
    if f != int(f) or f < lo:
        raise ValueError(f"{label or keys[0]}: {lo} 이상의 정수 필요 (입력 '{v}')")
    return int(f)


def _lag_list(v: Any, label: str) -> Optional[List[int]]:
    """'1 4', '1/3', [1, 4] -> sorted lag list; None when not given."""
    if v is None or v == "" or v == []:
        return None
    toks = v if isinstance(v, (list, tuple)) else str(v).replace(",", " ").split()
    out: List[int] = []
    for t in toks:
        t = str(t).strip()
        if "/" in t:
            a, b = t.split("/", 1)
            out.extend(range(int(a), int(b) + 1))
        elif t:
            out.append(int(float(t)))
    if any(x < 1 for x in out):
        raise ValueError(f"{label}: 1 이상의 시차 필요")
    return sorted(set(out))


def _check_columns(df: pd.DataFrame, label: str, roles: Dict[str, Any]) -> None:
    missing = []
    for role, cols in roles.items():
        for c in (cols if isinstance(cols, (list, tuple)) else [cols]):
            if c and c not in df.columns:
                missing.append(f"{role} '{c}'")
    if missing:
        raise ValueError(f"{label}: 데이터에 없는 변수 — {', '.join(missing)}. 사용 가능한 변수: {', '.join(map(str, df.columns))}")


def _fmt_p(p: float) -> str:
    return "< .001" if p < 0.001 else f"= {p:.3f}"


# ---------------------------------------------------------------- time index (tsset)
_FREQ_SEASON = {"M": 12, "Q": 4, "W": 52, "D": 7}
_FREQ_LABEL = {"D": "일", "W": "주", "M": "월", "Q": "분기", "Y": "연"}


class TimeIndex:
    """Sorted, regular time index of a data frame (Stata tsset): duplicates are errors, gaps are reported."""

    def __init__(self, df: pd.DataFrame, time: Optional[str], label: str):
        self.time = time
        self.freq: Optional[str] = None
        self.notes: List[str] = []
        if not time:
            self.df = df.reset_index(drop=True)
            self.pos = np.arange(len(self.df))
            self.labels = [str(i + 1) for i in range(len(self.df))]
            self.notes.append("시간 변수 미지정: 데이터 행 순서를 시간 순서로 사용 (Stata는 tsset 필요).")
            return
        t = df[time]
        if t.isna().any():
            raise ValueError(f"{label}: 시간 변수 '{time}'에 결측 {int(t.isna().sum())}개")
        if t.duplicated().any():
            dup = t[t.duplicated()].astype(str).unique()[:5]
            raise ValueError(f"{label}: 시간 변수 '{time}' 값 중복 ({', '.join(dup)}) — 시계열은 시점당 1행 필요 (Stata tsset 오류와 같은 조건)")
        if pd.api.types.is_numeric_dtype(t):
            v = t.to_numpy(float)
            order = np.argsort(v, kind="mergesort")
            vs = v[order]
            dif = np.diff(vs)
            step = float(dif.min()) if len(dif) else 1.0
            pos = np.round((vs - vs[0]) / step).astype(int) if step > 0 else np.zeros(len(vs), int)
            if not np.allclose((vs - vs[0]) / step, pos):
                raise ValueError(f"{label}: 시간 변수 '{time}' 간격이 일정하지 않음 (최소 간격 {step:g}의 배수가 아님)")
            labs = [f"{x:g}" for x in vs]
            self.step = step
        else:
            dt = pd.to_datetime(t.astype(str), errors="coerce")
            if dt.isna().any():
                bad = t[dt.isna()].astype(str).unique()[:3]
                raise ValueError(f"{label}: 시간 변수 '{time}'을(를) 날짜로 해석할 수 없음 ({', '.join(bad)}) — 숫자형 시간 변수 또는 YYYY-MM 형식 사용")
            order = np.argsort(dt.to_numpy(), kind="mergesort")
            dts = pd.DatetimeIndex(dt.to_numpy()[order])
            chosen = None
            for f in ("D", "W", "M", "Q", "Y"):
                per = dts.to_period(f)
                ords = per.asi8 if hasattr(per, "asi8") else np.array([x.ordinal for x in per])
                if len(np.unique(ords)) < len(ords):
                    continue
                d = np.diff(ords)
                if len(d) == 0 or d.min() == 1:
                    chosen = (f, per, ords)
                    break
            if chosen is None:
                raise ValueError(f"{label}: 시간 변수 '{time}'의 주기(일·주·월·분기·연)를 판단할 수 없음")
            f, per, ords = chosen
            self.freq = f
            pos = (ords - ords[0]).astype(int)
            labs = [str(x) for x in per]
            self.period = per
            self.notes.append(f"시간 변수 '{time}': {_FREQ_LABEL[f]} 단위로 해석.")
        self.df = df.iloc[order].reset_index(drop=True)
        self.pos = pos
        self.labels = labs
        gaps = np.where(np.diff(pos) > 1)[0]
        self.gaps = [(labs[i], labs[i + 1]) for i in gaps]

    def default_season(self) -> Optional[int]:
        return _FREQ_SEASON.get(self.freq or "", None)

    def future_labels(self, h: int) -> List[str]:
        if h <= 0:
            return []
        if self.freq:
            last = self.period[-1]
            return [str(last + i) for i in range(1, h + 1)]
        if self.time and hasattr(self, "step"):
            last = float(self.labels[-1])
            return [f"{last + i * self.step:g}" for i in range(1, h + 1)]
        return [str(len(self.labels) + i) for i in range(1, h + 1)]


def _require_no_gaps(ti: TimeIndex, rows: np.ndarray, label: str) -> None:
    pos = ti.pos[rows]
    g = np.where(np.diff(pos) > 1)[0]
    if len(g):
        pairs = ", ".join(f"{ti.labels[rows[i]]}→{ti.labels[rows[i + 1]]}" for i in g[:5])
        raise ValueError(f"{label}: 추정 구간에 시점 간격(gap) {len(g)}곳 ({pairs}) — 결측 시점을 채우거나 구간을 나누어 분석")


# ---------------------------------------------------------------- basic statistics (statsmodels; Stata corrgram / wntestq)
def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Sample autocorrelations r_1..r_nlags (statsmodels acf; same definition as Stata corrgram / ac)."""
    from statsmodels.tsa.stattools import acf as sm_acf
    return np.asarray(sm_acf(np.asarray(x, float), nlags=nlags, fft=False), float)[1:]


def pacf_regression(x: np.ndarray, nlags: int) -> np.ndarray:
    """Stata pac default: OLS of y_t on 1, y_{t−1}..y_{t−k} (statsmodels pacf method='ols')."""
    from statsmodels.tsa.stattools import pacf
    return np.asarray(pacf(np.asarray(x, float), nlags=nlags, method="ols"), float)[1:]


def pacf_yw(x: np.ndarray, nlags: int) -> np.ndarray:
    """Yule–Walker partial autocorrelations (statsmodels pacf method='ywm' = Stata pac, yw = R pacf)."""
    from statsmodels.tsa.stattools import pacf
    return np.asarray(pacf(np.asarray(x, float), nlags=nlags, method="ywm"), float)[1:]


def ljung_box(x: np.ndarray, lags: int, fitdf: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative Ljung–Box Q(1..lags) (statsmodels acorr_ljungbox): p from χ²(k) (Stata wntestq) and χ²(k − fitdf)."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    x = np.asarray(x, float)
    lb = acorr_ljungbox(x, lags=lags, return_df=True)
    q = lb["lb_stat"].to_numpy(float)
    p = lb["lb_pvalue"].to_numpy(float)
    pa = np.full(lags, np.nan)
    if fitdf > 0 and lags > fitdf:
        lba = acorr_ljungbox(x, lags=list(range(fitdf + 1, lags + 1)), model_df=fitdf, return_df=True)
        pa[fitdf:] = lba["lb_pvalue"].to_numpy(float)
    elif fitdf == 0:
        pa = p.copy()
    return q, p, pa


def default_q_lags(n: int) -> int:
    """Stata corrgram / wntestq default: min(floor(n/2) − 2, 40)."""
    return max(1, min(n // 2 - 2, 40))


# ---------------------------------------------------------------- plots
def _plot_acf_pacf(r: np.ndarray, pr: np.ndarray, n: int, title: str) -> str:
    import matplotlib.pyplot as plt
    from .plots import _col, _hdr_col, _clean_ax, fig_to_base64
    lags = np.arange(1, len(r) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    bart = 1.96 * np.sqrt((1 + 2 * np.concatenate([[0.0], np.cumsum(r[:-1] ** 2)])) / n)
    for ax, vals, band, name in ((axes[0], r, bart, "자기상관 (AC)"), (axes[1], pr, np.full(len(pr), 1.96 / np.sqrt(n)), "편자기상관 (PAC)")):
        ax.fill_between(lags, -band, band, color="#CBD5E1", alpha=0.55, step="mid", label="95% 신뢰대")
        ax.vlines(lags, 0, vals, color=_col(0), linewidth=2)
        ax.plot(lags, vals, "o", color=_col(0), markersize=3.5)
        ax.axhline(0, color="#64748B", linewidth=0.8)
        ax.set_xlabel("시차 (lag)")
        ax.set_title(name, color=_hdr_col())
        ax.set_ylim(-1.05, 1.05)
        _clean_ax(ax)
    axes[0].text(0.99, 0.02, "AC 신뢰대: Bartlett 공식", transform=axes[0].transAxes, ha="right", fontsize=7.5, color="#64748B")
    axes[1].text(0.99, 0.02, "PAC 신뢰대: ±1.96/√n", transform=axes[1].transAxes, ha="right", fontsize=7.5, color="#64748B")
    fig.suptitle(title, color=_hdr_col(), fontweight="bold")
    fig.tight_layout()
    return fig_to_base64(fig)


def _plot_forecast(x_lab: List[str], y: np.ndarray, fitted: Optional[np.ndarray], f_lab: List[str], fc: Optional[np.ndarray],
                   lo: Optional[np.ndarray], hi: Optional[np.ndarray], dep: str, level: float) -> str:
    import matplotlib.pyplot as plt
    from .plots import _col, _hdr_col, _clean_ax, fig_to_base64
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    n = len(y)
    xs = np.arange(n)
    ax.plot(xs, y, color=_col(0), linewidth=1.3, label="관측값")
    if fitted is not None:
        ax.plot(xs, fitted, color=_col(1), linewidth=1.0, alpha=0.85, linestyle="--", label="1단계 예측값 (적합)")
    if fc is not None and len(fc):
        xf = np.arange(n, n + len(fc))
        ax.fill_between(xf, lo, hi, color=_col(2), alpha=0.22, label=f"{level * 100:g}% 예측구간")
        ax.plot(xf, fc, color=_col(2), linewidth=1.6, label="예측")
    labs = list(x_lab) + list(f_lab)
    step = max(1, len(labs) // 8)
    ticks = list(range(0, len(labs), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([labs[i] for i in ticks], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(dep)
    ax.set_title(f"{dep}: 관측값·적합값·예측", color=_hdr_col())
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _clean_ax(ax)
    fig.tight_layout()
    return fig_to_base64(fig)


def _plot_resid(x_lab: List[str], e: np.ndarray, title: str) -> str:
    import matplotlib.pyplot as plt
    from .plots import _col, _hdr_col, _clean_ax, fig_to_base64
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    ax.plot(np.arange(len(e)), e, color=_col(0), linewidth=1.0)
    ax.axhline(0, color="#64748B", linewidth=0.8)
    step = max(1, len(x_lab) // 6)
    ticks = list(range(0, len(x_lab), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([x_lab[i] for i in ticks], rotation=30, ha="right", fontsize=8)
    ax.set_title("잔차 (1단계 예측오차)", color=_hdr_col())
    _clean_ax(ax)
    ax2 = axes[1]
    ax2.hist(e, bins=min(30, max(8, len(e) // 8)), color=_col(1), alpha=0.8, density=True)
    s = np.std(e, ddof=1)
    g = np.linspace(e.min(), e.max(), 100)
    ax2.plot(g, stats.norm.pdf(g, e.mean(), s), color="#334155", linewidth=1.2)
    ax2.set_title("잔차 분포", color=_hdr_col())
    _clean_ax(ax2)
    fig.suptitle(title, color=_hdr_col(), fontweight="bold")
    fig.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------- differencing helpers
def _difference(a: np.ndarray, d: int, D: int, s: int) -> np.ndarray:
    a = np.asarray(a, float)
    for _ in range(D):
        a = a[s:] - a[:-s]
    for _ in range(d):
        a = a[1:] - a[:-1]
    return a


def _integrated_one(n: int, d: int, D: int, s: int) -> np.ndarray:
    """A regressor c_t with (1−L)^d (1−L^s)^D c_t = 1, so that the level model carries the Stata constant."""
    c = np.ones(n)
    for _ in range(d):
        c = np.cumsum(c)
    for _ in range(D):
        out = c.copy()
        for t in range(s, n):
            out[t] = c[t] + out[t - s]
        c = out
    return c


def _dname(v: str, d: int, D: int, s: int) -> str:
    pre = ""
    if D:
        pre += f"S{s}." if D == 1 else f"S{s}{D}."
    if d:
        pre += "D." if d == 1 else f"D{d}."
    return pre + v


VCE_TS = ("opg", "oim", "robust")
_VCE_TS_ALIASES = {"outer": "opg", "hessian": "oim", "observed": "oim", "approx": "oim", "r": "robust", "qml": "robust",
                   "huber": "robust", "sandwich": "robust"}
VCE_TS_LABEL = {"opg": "OPG (Stata arima 기본)", "oim": "관측 정보행렬 (OIM)", "robust": "준최우 강건 (robust)"}
_SM_COV = {"opg": "opg", "oim": "approx", "robust": "robust_approx"}


# ---------------------------------------------------------------- arima
class ArimaAnalysis(BaseAnalysis):
    """Stata arima: regression with (seasonal) ARMA disturbances, exact ML (Kalman filter)."""

    def __init__(self, df: pd.DataFrame, params: Dict[str, Any]):
        p = dict(params or {})
        dep = _as_single(_opt(p, "dep", "y", "var", "series"))
        time = _as_single(_opt(p, "time", "timeVar", "time_var", "t"))
        exog = _as_list(_opt(p, "exog", "covs", "indeps", default=[]))
        if not dep:
            raise ValueError("ARIMA: 시계열 변수(dep) 지정 필요")
        _check_columns(df, "ARIMA", {"시계열 변수": dep, "시간 변수": time, "외생변수": exog})
        P_ = _int(p, "p", "ar_order", default=1, label="AR 차수 p")
        d = _int(p, "d", "diff", default=0, label="차분 차수 d")
        q = _int(p, "q", "ma_order", default=0, label="MA 차수 q")
        Ps = _int(p, "P", "sar", "seasonalP", default=0, label="계절 AR 차수 P")
        Ds = _int(p, "D", "sdiff", "seasonalD", default=0, label="계절 차분 D")
        Qs = _int(p, "Q", "sma", "seasonalQ", default=0, label="계절 MA 차수 Q")
        ar_l = _lag_list(_opt(p, "arLags", "ar_lags", "ar"), "AR 시차(arLags)")
        ma_l = _lag_list(_opt(p, "maLags", "ma_lags", "ma"), "MA 시차(maLags)")
        const = _flag(p, "constant", "intercept", default=True) and not _flag(p, "noconstant", "noConstant", default=False)
        diff_exog = _flag(p, "diffExog", "diff_exog", default=True)
        vce = resolve_choice("표준오차(covType)", _opt(p, "covType", "cov_type", "vce", default="opg"), VCE_TS, _VCE_TS_ALIASES)
        level = float(_opt(p, "ciWidth", "ci_width", "level", default=95))
        level = level / 100 if level > 1 else level
        h = _int(p, "forecast", "h", "horizon", "steps", default=12, label="예측 기간(forecast)")
        lb = _int(p, "lbLags", "lb_lags", "qLags", "lags", default=None, lo=0, label="Ljung–Box 시차") or None      # 0 = Stata default
        acf_lags = _int(p, "acfLags", "acf_lags", default=None, lo=0, label="잔차 ACF 시차") or None
        plots = _flag(p, "plots", "plot", default=True)
        maxiter = _int(p, "maxiter", "iterate", default=500, lo=1, label="최대 반복")
        rtype = resolve_choice("잔차 종류(residType)", _opt(p, "residType", "resid_type", default="raw"),
                               ("raw", "standardized"), {"std": "standardized", "innovation": "raw", "stata": "raw", "r": "standardized"})
        ti = TimeIndex(df, time, "ARIMA")
        s = _int(p, "s", "season", "period", "seasonalPeriod", default=None, lo=0, label="계절 주기 s") or None   # 0 = from tsset frequency
        if s == 1:
            raise ValueError("계절 주기 s: 2 이상 필요 (0 = 시간 변수 주기에서 판단)")
        if (Ps or Ds or Qs) and not s:
            s = ti.default_season()
            if not s:
                raise ValueError("ARIMA: 계절 항(P·D·Q) 사용 시 계절 주기 s 지정 필요 (월 자료 12, 분기 자료 4)")
        s = s or 0
        ar_lags = ar_l if ar_l is not None else list(range(1, P_ + 1))
        ma_lags = ma_l if ma_l is not None else list(range(1, q + 1))
        options = {"dep": dep, "time": time, "exog": exog, "p": len(ar_lags) if ar_l is None else None,
                   "arLags": ar_l, "d": d, "q": len(ma_lags) if ma_l is None else None, "maLags": ma_l,
                   "P": Ps, "D": Ds, "Q": Qs, "s": s if (Ps or Ds or Qs) else None, "constant": const,
                   "diffExog": diff_exog if (d or Ds) and exog else None, "covType": vce, "ciWidth": level * 100,
                   "forecast": h, "lbLags": lb, "acfLags": acf_lags, "residType": rtype, "plots": plots}
        if ti.freq:
            options["timeFreq"] = ti.freq          # informational (Stata export converts string dates); ignored on input
        order_txt = f"ARIMA({_lagtxt(ar_lags)},{d},{_lagtxt(ma_lags)})" + (f"({Ps},{Ds},{Qs})[{s}]" if (Ps or Ds or Qs) else "")
        super().__init__(name="arima", title=f"ARIMA 모형 — {order_txt}" + (" 회귀(ARMAX)" if exog else "") + " (arima)",
                         dataset=df, options=options)
        self._run(ti, dep, exog, ar_lags, d, ma_lags, Ps, Ds, Qs, s, const, diff_exog, vce, level, h, lb, acf_lags, plots, maxiter)

    def _run(self, ti, dep, exog, ar_lags, d, ma_lags, Ps, Ds, Qs, s, const, diff_exog, vce, level, h, lb, acf_lags, plots, maxiter):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        data = ti.df
        for c in [dep] + exog:
            if not pd.api.types.is_numeric_dtype(data[c]):
                raise ValueError(f"ARIMA: 숫자형이 아닌 변수 — {c}")
        y_all = data[dep].to_numpy(float)
        X_all = data[exog].to_numpy(float) if exog else np.zeros((len(data), 0))
        ok = ~np.isnan(y_all) & ~np.isnan(X_all).any(axis=1)
        if ok.sum() == 0:
            raise ValueError("ARIMA: 유효 관측치 없음")
        first, last = int(np.argmax(ok)), int(len(ok) - 1 - np.argmax(ok[::-1]))
        rows = np.arange(first, last + 1)
        if (~ok[rows]).any():
            bad = [ti.labels[i] for i in rows[~ok[rows]]][:5]
            raise ValueError(f"ARIMA: 추정 구간 중간에 결측 ({', '.join(bad)} 등) — 결측을 처리한 뒤 분석 (Stata arima도 연속 표본 필요)")
        _require_no_gaps(ti, rows, "ARIMA")
        y = y_all[rows]
        X = X_all[rows]
        notes = list(ti.notes)
        lost = d + s * Ds
        yw = _difference(y, d, Ds, s)
        Xw = np.column_stack([_difference(X[:, j], d, Ds, s) if diff_exog else X[lost:, j] for j in range(X.shape[1])]) \
            if exog else np.zeros((len(yw), 0))
        names_x = [(_dname(c, d, Ds, s) if diff_exog else c) for c in exog]
        if const:
            Xw = np.column_stack([Xw, np.ones(len(yw))])
            names_x.append("_cons")
        n = len(yw)
        n_par = len(names_x) + len(ar_lags) + len(ma_lags) + Ps + Qs + 1
        if n <= n_par + 2:
            raise ValueError(f"ARIMA: 관측치 {n}개 — 모수 {n_par}개를 추정하기에 부족")
        sorder = (Ps, 0, Qs, s) if (Ps or Qs) else (0, 0, 0, 0)
        mod = SARIMAX(yw, exog=Xw if Xw.shape[1] else None, order=(ar_lags, 0, ma_lags), seasonal_order=sorder,
                      trend="n", enforce_stationarity=True, enforce_invertibility=True)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            r1 = mod.fit(disp=False, maxiter=maxiter, cov_type="none")
            # L-BFGS often stops short of the optimum near the stationarity boundary: polish with Nelder–Mead
            r2 = mod.fit(start_params=r1.params, method="nm", maxiter=max(2000, 400 * len(r1.params)), disp=False, cov_type="none")
            best = r2 if r2.llf > r1.llf else r1
            res = mod.filter(best.params, cov_type=_SM_COV[vce])
        rv = [r.mle_retvals for r in (r1, r2) if isinstance(r.mle_retvals, dict)]
        conv = any(bool(v.get("converged", True)) for v in rv) if rv else True
        if not conv:
            notes.append("최우추정 반복이 수렴 기준에 도달하지 못함 — 결과 해석에 주의 (차수 축소·차분 검토).")
        warn_txt = [str(w.message) for w in wlist if "invert" in str(w.message).lower() or "stationar" in str(w.message).lower()]
        if warn_txt:
            notes.append("추정 경고: " + warn_txt[0][:160])
        prm = np.asarray(res.params, float)
        cov = np.asarray(res.cov_params(), float)
        pnames = list(res.model.param_names)
        # table rows in Stata order: regression equation, ARMA, seasonal ARMA, /sigma
        rows_out = []
        zq = stats.norm.ppf(0.5 + level / 2)
        nx = len(names_x)
        for i, nm in enumerate(pnames):
            if nm == "sigma2":
                continue
            if i < nx:
                eq, term = dep, names_x[i]                       # Stata e(b): equation named after the dependent variable
            elif nm.startswith("ar.S."):
                eq, term = f"ARMA{s}", f"ar L{int(nm.split('L')[-1]) // s}."
            elif nm.startswith("ma.S."):
                eq, term = f"ARMA{s}", f"ma L{int(nm.split('L')[-1]) // s}."
            elif nm.startswith("ar."):
                eq, term = "ARMA", f"ar L{nm.split('L')[-1]}."
            else:
                eq, term = "ARMA", f"ma L{nm.split('L')[-1]}."
            b, se = prm[i], math.sqrt(max(cov[i, i], 0))
            rows_out.append({"eq": eq, "term": term, "b": b, "se": se})
        i_s = pnames.index("sigma2")
        s2, v2 = prm[i_s], cov[i_s, i_s]
        sigma = math.sqrt(s2)
        se_sig = math.sqrt(max(v2, 0)) / (2 * sigma)
        rows_out.append({"eq": "", "term": "/sigma", "b": sigma, "se": se_sig})
        ct = ResultsTable(name="arimaCoef", title="계수 추정치 (Coefficients)")
        ct.add_column("eq", "식", "text")
        ct.add_column("term", "변수", "text")
        ct.add_column("b", "계수", "number")
        ct.add_column("se", f"표준오차 ({vce.upper()})", "number")
        ct.add_column("z", "z", "number")
        ct.add_column("p", "p", "number", "pvalue")
        ct.add_column("lo", f"{level * 100:g}% 하한", "number", super_title="신뢰구간")
        ct.add_column("hi", f"{level * 100:g}% 상한", "number", super_title="신뢰구간")
        for r in rows_out:
            z = r["b"] / r["se"] if r["se"] > 0 else float("nan")
            r.update(z=z, p=float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan"),
                     lo=r["b"] - zq * r["se"], hi=r["b"] + zq * r["se"])
            if r["term"] == "/sigma":
                r["lo"] = max(0.0, r["lo"])
            ct.add_row(r)
        ct.mark_coefficients(estimate="b", se="se", p="p", stat="z", ci=("lo", "hi"), label=["eq", "term"])
        ct.add_note("부호 규약: μ_t = ρ₁μ_{t−1} + … + θ₁ε_{t−1} + ε_t (Stata·R과 같음). " +
                    ("회귀식 상수(_cons)는 " + ("차분 계열의 평균" if (d or Ds) else "계열의 평균") + " (Stata 규약; R arima의 intercept·drift에 해당)." if const else "상수항 없음 (noconstant)."))
        ct.add_note("/sigma: 백색잡음 표준편차. 분산 0 검정은 단측이며 신뢰구간 하한은 0에서 절단.")
        if (d or Ds) and exog:
            ct.add_note("외생변수: " + ("종속변수와 같은 차수로 차분 (Stata arima(p,d,q) 옵션 규약)." if diff_exog else "차분하지 않은 수준값 사용 (Stata arima D.y x 형식)."))
        # fit statistics
        ll = float(res.llf)
        k_all = len(prm)
        aic = -2 * ll + 2 * k_all
        bic = -2 * ll + math.log(n) * k_all
        idx_w = [i for i, nm in enumerate(pnames) if nm != "sigma2" and not (const and i == nx - 1)]
        wald = float(prm[idx_w] @ np.linalg.solve(cov[np.ix_(idx_w, idx_w)], prm[idx_w])) if idx_w else float("nan")
        e_raw = np.asarray(res.resid, float)
        e_std = np.asarray(res.standardized_forecasts_error, float).reshape(-1) * math.sqrt(float(res.params[-1]))
        e = e_std if self.options["residType"] == "standardized" else e_raw
        fit = ResultsTable(name="arimaFit", title="모형 적합도 (Model fit)")
        fit.add_column("stat", "항목", "text")
        fit.add_column("value", "값", "number")
        fit.add_column("df", "자유도", "text")
        fit.add_column("p", "p", "number", "pvalue")
        fit.add_row({"stat": "관측치 수 (N)", "value": n, "df": "", "p": "", "_rowfmt": "integer"})
        fit.add_row({"stat": "표본 구간", "value": "", "df": f"{ti.labels[rows[lost]]} – {ti.labels[rows[-1]]}", "p": ""})
        if idx_w:
            fit.add_row({"stat": "Wald χ²", "value": wald, "df": f"{len(idx_w)}", "p": float(stats.chi2.sf(wald, len(idx_w)))})
        fit.add_row({"stat": "로그우도 (LL)", "value": ll, "df": "", "p": ""})
        fit.add_row({"stat": "AIC", "value": aic, "df": f"모수 {k_all}개", "p": ""})
        fit.add_row({"stat": "BIC", "value": bic, "df": "", "p": ""})
        fit.add_row({"stat": "σ² (잡음 분산)", "value": s2, "df": "", "p": ""})
        fit.add_note("정확 최우추정 (Kalman 필터, 정상 초기분포). 표준오차: " + VCE_TS_LABEL[vce] + ". AIC·BIC는 σ 포함 모수 수 기준 (Stata estat ic 방식).")
        if lost:
            fit.add_note(f"차분으로 첫 {lost}개 시점 제외 (Stata와 같은 표본).")
        if self.options["diffExog"] is False:
            pass
        for n_ in notes:
            fit.add_note(n_)
        self.add_table(fit)
        self.add_table(ct)
        # Ljung–Box on residuals (wntestq)
        lbl = lb or default_q_lags(n)
        lbl = min(lbl, n - 1)
        fitdf = len(ar_lags) + len(ma_lags) + Ps + Qs
        qv, pv, pa = ljung_box(e, lbl)[0], ljung_box(e, lbl)[1], ljung_box(e, lbl, fitdf)[2]
        lbt = ResultsTable(name="arimaLjungBox", title="잔차 백색잡음 검정 (Ljung–Box Q, wntestq)")
        lbt.add_column("lags", "시차", "integer")
        lbt.add_column("Q", "Q", "number")
        lbt.add_column("df", "자유도", "integer")
        lbt.add_column("p", "p (Stata)", "number", "pvalue")
        lbt.add_column("dfa", "자유도 − ARMA 모수", "integer")
        lbt.add_column("pa", "p (자유도 조정)", "number", "pvalue")
        marks = sorted(set([m for m in (6, 12, 24, 36) if m < lbl] + [lbl]))
        for m in marks:
            lbt.add_row({"lags": m, "Q": float(qv[m - 1]), "df": m, "p": float(pv[m - 1]),
                         "dfa": m - fitdf if m > fitdf else float("nan"), "pa": float(pa[m - 1])})
        lbt.add_note(f"귀무가설: 잔차가 백색잡음. Stata wntestq는 자유도 = 시차 (ARMA 모수 미차감); R Box.test(fitdf = {fitdf})는 자유도 조정 열에 해당.")
        lbt.add_note("잔차: " + ("1단계 예측오차 원값 (Stata predict, residuals → wntestq와 같음). 정확 최우추정의 정상 초기분포 때문에 처음 시점의 예측오차 분산이 커서 "
                                 "Q가 작게 나올 수 있음 — residType='standardized'(예측오차/예측 표준편차×σ, R arima 잔차)로 비교 가능."
                                 if self.options["residType"] == "raw" else
                                 "1단계 예측오차를 예측오차 표준편차로 나눈 뒤 σ를 곱한 표준화 잔차 (R arima 잔차와 같음; Stata predict, residuals와는 처음 시점이 다름)."))
        self.add_table(lbt)
        # residual ACF / PACF
        al = min(acf_lags or default_q_lags(n), n - 2)
        r_ = acf(e, al)
        pr_ = pacf_regression(e, al)
        # forecasts
        fc_tab = None
        fc = lo = hi = None
        f_lab: List[str] = []
        if h and h > 0:
            fut_rows = np.arange(last + 1, len(data))
            X_f = None
            if exog:
                Xa = X_all[fut_rows]
                okf = ~np.isnan(Xa).any(axis=1)
                nf = int(np.argmin(okf)) if (~okf).any() else len(okf)
                if nf == 0:
                    notes_fc = "외생변수가 있는 모형: 예측 구간의 외생변수 값(종속변수 결측 행)이 없어 예측 생략."
                    h_eff = 0
                else:
                    h_eff = min(h, nf)
                    X_f = Xa[:h_eff]
                    notes_fc = f"외생변수 값이 있는 미래 {h_eff}개 시점 예측." + (f" (요청 {h}개 중 외생변수 값 있는 {h_eff}개)" if h_eff < h else "")
            else:
                h_eff = h
                notes_fc = ""
            if h_eff > 0:
                try:
                    fc, lo, hi, f_lab = self._forecast(ti, rows, fut_rows, y, X, X_f, exog, ar_lags, d, ma_lags, Ps, Ds, Qs, s, const,
                                                       diff_exog, prm, h_eff, level)
                except Exception as ex:   # forecasting must not hide the estimates
                    notes_fc = f"예측 계산 실패: {type(ex).__name__}: {ex}"
                    fc = None
            fc_tab = ResultsTable(name="arimaForecast", title=f"예측 (h = {h_eff}단계, {level * 100:g}% 예측구간)")
            fc_tab.add_column("time", "시점", "text")
            fc_tab.add_column("fc", f"예측값 ({dep})", "number")
            fc_tab.add_column("se", "예측 표준오차", "number")
            fc_tab.add_column("lo", "하한", "number")
            fc_tab.add_column("hi", "상한", "number")
            if fc is not None:
                zz = stats.norm.ppf(0.5 + level / 2)
                for i in range(len(fc)):
                    fc_tab.add_row({"time": f_lab[i], "fc": fc[i], "se": (hi[i] - fc[i]) / zz, "lo": lo[i], "hi": hi[i]})
            fc_tab.add_note(("동적 다단계 예측 (Stata predict, dynamic(.)). " if fc is not None else "") +
                            ("차분 모형: 수준값으로 환산한 예측과 예측구간." if (d or Ds) else "") + (" " + notes_fc if notes_fc else ""))
            fc_tab.add_note("예측구간은 모수 추정 불확실성을 포함하지 않음 (Stata predict, mse·R predict.Arima와 같은 정의; 잡음 분산은 최우추정값 σ²).")
            if fc is not None:
                self.add_table(fc_tab)
            else:
                fit.add_note("예측: " + notes_fc.strip())
        if plots:
            y_lab = [ti.labels[i] for i in rows]
            fitted = np.full(len(y), np.nan)
            fitted[lost:] = y[lost:] - e
            self.add_plot("관측값·적합값·예측", _plot_forecast(y_lab, y, fitted, f_lab, fc, lo, hi, dep, level))
            self.add_plot("잔차 진단", _plot_resid(y_lab[lost:], e, "ARIMA 잔차"))
            self.add_plot("잔차 자기상관·편자기상관", _plot_acf_pacf(r_, pr_, n, "잔차 ACF·PACF"))
        self.model_stats = {"N": n, "LL": ll, "AIC": aic, "BIC": bic}
        if idx_w:
            self.model_stats["Chi2"] = (wald, float(stats.chi2.sf(wald, len(idx_w))))
        self.fit_result = res
        lbq = float(qv[min(lbl, 12 if lbl >= 12 else lbl) - 1])
        lbp = float(pv[min(lbl, 12 if lbl >= 12 else lbl) - 1])
        mlag = min(lbl, 12)
        self.apa_narrative = (f"{dep}에 대해 {self.title.split('— ')[1].split(' (')[0]} 모형을 정확 최우추정한 결과(N = {n}), 로그우도 = {ll:.2f}, "
                              f"AIC = {aic:.2f}, BIC = {bic:.2f}였다. 잔차의 Ljung–Box Q({mlag}) = {lbq:.2f} (p {_fmt_p(lbp)}).")
        o = self.options
        args = {k: v for k, v in o.items() if v not in (None, "", []) and k != "timeFreq"}
        self.syntax_python = "sess.run('arima', " + ", ".join(f"{k}={v!r}" for k, v in args.items()) + ")"
        self.syntax_r = "\n".join(r_code("arima", o) or [])

    def _forecast(self, ti, rows, fut_rows, y, X, X_f, exog, ar_lags, d, ma_lags, Ps, Ds, Qs, s, const, diff_exog, prm, h, level):
        """Level-model forecasts with the estimated parameters held fixed (integrates differenced forecasts with exact variances)."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        n_lv = len(y)
        sorder = (Ps, Ds, Qs, s) if (Ps or Ds or Qs) else (0, 0, 0, 0)
        cols, cols_f = [], []
        if exog:
            if diff_exog or not (d or Ds):
                cols.append(X)
                cols_f.append(X_f)
            else:
                # regressors entered in levels in the differenced equation: integrate them
                Xi = np.column_stack([_integrate_series(X[:, j], d, Ds, s, n_lv + h, X_f[:, j]) for j in range(X.shape[1])])
                cols.append(Xi[:n_lv])
                cols_f.append(Xi[n_lv:])
        if const:
            c = _integrated_one(n_lv + h, d, Ds, s)
            cols.append(c[:n_lv, None])
            cols_f.append(c[n_lv:, None])
        ex = np.column_stack(cols) if cols else None
        exf = np.column_stack(cols_f) if cols_f else None
        # stationary initial distribution for the ARMA states, exact diffuse for the integrated states
        lm = SARIMAX(y, exog=ex, order=(ar_lags, d, ma_lags), seasonal_order=sorder, trend="n",
                     enforce_stationarity=True, enforce_invertibility=False, use_exact_diffuse=True)
        rf = lm.filter(prm)
        fo = rf.get_forecast(steps=h, exog=exf)
        mean = np.asarray(fo.predicted_mean, float)
        ci = np.asarray(fo.conf_int(alpha=1 - level), float)
        labels = []
        for i in range(h):
            j = fut_rows[i] if i < len(fut_rows) else None
            labels.append(ti.labels[j] if j is not None and j < len(ti.labels) else None)
        fut = ti.future_labels(h)
        labels = [lab if lab is not None else fut[i] for i, lab in enumerate(labels)]
        return mean, ci[:, 0], ci[:, 1], labels


def _integrate_series(x: np.ndarray, d: int, D: int, s: int, n_tot: int, x_f: np.ndarray) -> np.ndarray:
    """Series z with Δ^d Δ_s^D z_t = x_t, used when level regressors enter a differenced equation (arima D.y x)."""
    full = np.concatenate([x, x_f])
    z = full.copy()
    for _ in range(d):
        z = np.cumsum(z)
    for _ in range(D):
        out = z.copy()
        for t in range(s, len(z)):
            out[t] = z[t] + out[t - s]
        z = out
    return z


def _lagtxt(l: List[int]) -> str:
    if not l:
        return "0"
    if l == list(range(1, len(l) + 1)):
        return str(len(l))
    return "[" + " ".join(map(str, l)) + "]"


# ---------------------------------------------------------------- unit-root tests
UR_TESTS = ("adf", "pp", "kpss", "all")
_UR_ALIASES = {"dfuller": "adf", "df": "adf", "pperron": "pp", "phillips": "pp", "kpss": "kpss", "전체": "all"}
UR_TRENDS = ("constant", "trend", "drift", "none")
_UR_TREND_ALIASES = {"c": "constant", "const": "constant", "ct": "trend", "n": "none", "nc": "none",
                     "noconstant": "none", "level": "constant"}


# Fuller (1976) critical values as used by Stata dfuller / pperron ("interpolated Dickey–Fuller"): sample sizes
# 25, 50, 100, 250, 500, ∞; rows 1%, 5%, 10%. Stata interpolates linearly in N up to 500 and uses the ∞ column above
# 500 (both checked against Stata 19 output for N = 25 … 799).
_FULLER_N = (25, 50, 100, 250, 500)
_FULLER_TAU = {"none": ((-2.66, -2.62, -2.60, -2.58, -2.58, -2.58), (-1.95, -1.95, -1.95, -1.95, -1.95, -1.95), (-1.60, -1.61, -1.61, -1.62, -1.62, -1.62)),
               "constant": ((-3.75, -3.58, -3.51, -3.46, -3.44, -3.43), (-3.00, -2.93, -2.89, -2.88, -2.87, -2.86), (-2.63, -2.60, -2.58, -2.57, -2.57, -2.57)),
               "trend": ((-4.38, -4.15, -4.04, -3.99, -3.98, -3.96), (-3.60, -3.50, -3.45, -3.43, -3.42, -3.41), (-3.24, -3.18, -3.15, -3.13, -3.13, -3.12))}
_FULLER_RHO = {"none": ((-11.9, -12.9, -13.3, -13.6, -13.7, -13.8), (-7.3, -7.7, -7.9, -8.0, -8.0, -8.1), (-5.3, -5.5, -5.6, -5.7, -5.7, -5.7)),
               "constant": ((-17.2, -18.9, -19.8, -20.3, -20.5, -20.7), (-12.5, -13.3, -13.7, -14.0, -14.0, -14.1), (-10.2, -10.7, -11.0, -11.2, -11.2, -11.3)),
               "trend": ((-22.5, -25.7, -27.4, -28.4, -28.9, -29.5), (-17.9, -19.8, -20.7, -21.3, -21.5, -21.8), (-15.6, -16.8, -17.5, -18.0, -18.1, -18.3))}


def fuller_cv(table: Dict[str, Tuple], trend: str, n: int) -> List[float]:
    out = []
    for row in table[trend]:
        if n > 500:
            out.append(row[5])
        elif n <= 25:
            out.append(row[0])
        else:
            out.append(float(np.interp(n, _FULLER_N, row[:5])))
    return out


_ARCH_TREND = {"none": "n", "constant": "c", "drift": "c", "trend": "ct"}


def adf_test(x: np.ndarray, lags: Optional[int], trend: str, method: Optional[str] = None, max_lags: Optional[int] = None) -> Dict[str, Any]:
    """arch ADF (statistic, MacKinnon 1994 p-value, regression) with Stata dfuller critical values and the drift rule."""
    from arch.unitroot import ADF
    t = ADF(np.asarray(x, float), lags=lags, trend=_ARCH_TREND[trend], method=method or "aic", max_lags=max_lags)
    stat, n = float(t.stat), int(t.nobs)
    reg = t.regression
    if trend == "drift":
        df_ = int(reg.df_resid)
        crit = [float(stats.t.ppf(a, df_)) for a in (0.01, 0.05, 0.10)]
        p = float(stats.t.cdf(stat, df_))
        pnote = f"drift: Z(t)는 t({df_}) 분포, 단측 p값 (Stata dfuller, drift)"
    else:
        crit = fuller_cv(_FULLER_TAU, trend, n)
        p = float(t.pvalue)
        pnote = "MacKinnon (1994) 근사 p값 (arch; Stata dfuller와 같음)"
    names = ["L.y"] + [f"L{j}D.y" for j in range(1, int(t.lags) + 1)]
    if trend == "trend":
        names.append("_trend")
    if trend in ("constant", "trend", "drift"):
        names.append("_cons")
    b = np.asarray(reg.params, float)
    se = np.asarray(reg.bse, float)
    # arch orders the regression as [L.y, deterministic terms, lagged differences]; reorder to Stata's display order
    k_det = {"none": 0, "constant": 1, "drift": 1, "trend": 2}[trend]
    order = [0] + list(range(1 + k_det, len(b))) + ([2, 1] if k_det == 2 else list(range(1, 1 + k_det)))
    return {"stat": stat, "n": n, "crit": crit, "p": p, "pnote": pnote, "b": b[order], "se": se[order], "names": names,
            "lags": int(t.lags), "df_resid": int(reg.df_resid)}


def pp_test(x: np.ndarray, lags: Optional[int], trend: str) -> Dict[str, Any]:
    """arch PhillipsPerron (Z(t), Z(ρ), MacKinnon p-value) with Stata's default lag rule and Fuller critical values."""
    from arch.unitroot import PhillipsPerron
    x = np.asarray(x, float)
    n = len(x) - 1
    q = lags if lags is not None else int(4 * (n / 100) ** (2 / 9))          # Stata pperron default lags
    tau = PhillipsPerron(x, lags=q, trend=_ARCH_TREND[trend], test_type="tau")
    rho = PhillipsPerron(x, lags=q, trend=_ARCH_TREND[trend], test_type="rho")
    return {"z_rho": float(rho.stat), "z_t": float(tau.stat), "n": int(tau.nobs), "lags": q, "p": float(tau.pvalue),
            "crit": fuller_cv(_FULLER_TAU, trend, int(tau.nobs)), "crit_rho": fuller_cv(_FULLER_RHO, trend, int(tau.nobs))}


def kpss_test(x: np.ndarray, lags: Optional[int], trend: str) -> Dict[str, Any]:
    """arch KPSS (Bartlett long-run variance; automatic bandwidth when lags is None)."""
    from arch.unitroot import KPSS
    t = KPSS(np.asarray(x, float), lags=lags, trend="ct" if trend == "trend" else "c")      # lags=None: Hobijn et al. (1998)
    cv = t.critical_values
    return {"stat": float(t.stat), "p": float(t.pvalue), "lags": int(t.lags), "crit": [cv["1%"], cv["5%"], cv["10%"]], "reg": trend}


class UnitRootAnalysis(BaseAnalysis):
    """Unit-root and stationarity tests: Stata dfuller, pperron, and KPSS."""

    def __init__(self, df: pd.DataFrame, params: Dict[str, Any]):
        p = dict(params or {})
        dep = _as_single(_opt(p, "dep", "var", "y", "series"))
        time = _as_single(_opt(p, "time", "timeVar", "time_var"))
        if not dep:
            raise ValueError("단위근 검정: 시계열 변수(dep) 지정 필요")
        _check_columns(df, "단위근 검정", {"시계열 변수": dep, "시간 변수": time})
        test = resolve_choice("검정(test)", _opt(p, "test", "tests", default="adf"), UR_TESTS, _UR_ALIASES)
        trend = resolve_choice("결정항(trend)", _opt(p, "trend", "deterministic", "regression", default="constant"),
                               UR_TRENDS, _UR_TREND_ALIASES)
        lags = _int(p, "lags", "adfLags", "adf_lags", default=0, label="ADF 시차(lags)")
        lag_sel = resolve_choice("ADF 시차 선택(lagSelect)", _opt(p, "lagSelect", "lag_select", "autolag", default="none"),
                                 ("none", "aic", "bic"), {"fixed": "none", "AIC": "aic", "BIC": "bic"})
        max_lag = _int(p, "maxLag", "max_lag", "maxlag", default=None, label="최대 시차") or None              # 0 = default rule
        pp_lags = _int(p, "ppLags", "pp_lags", default=None, label="PP 시차")
        kpss_lags = _int(p, "kpssLags", "kpss_lags", default=None, label="KPSS 시차")
        show_reg = _flag(p, "regress", "showRegression", default=False)
        if test in ("pp", "kpss") and trend == "drift":
            raise ValueError("단위근 검정: drift 옵션은 ADF(dfuller)에서만 가능 (pperron·KPSS: constant·trend" + ("·none" if test == "pp" else "") + ")")
        if test == "kpss" and trend == "none":
            raise ValueError("KPSS: 결정항 none 미지원 (constant·trend)")
        options = {"dep": dep, "time": time, "test": test, "trend": trend, "lags": lags, "lagSelect": lag_sel,
                   "maxLag": max_lag, "ppLags": pp_lags, "kpssLags": kpss_lags, "regress": show_reg}
        names = {"adf": "Augmented Dickey–Fuller (dfuller)", "pp": "Phillips–Perron (pperron)", "kpss": "KPSS", "all": "ADF·PP·KPSS"}
        super().__init__(name="unitRoot", title=f"단위근 검정 — {names[test]}", dataset=df, options=options)
        ti = TimeIndex(df, time, "단위근 검정")
        if ti.freq:
            self.options["timeFreq"] = ti.freq
        x_all = pd.to_numeric(ti.df[dep], errors="coerce").to_numpy(float)
        ok = ~np.isnan(x_all)
        if ok.sum() < 10:
            raise ValueError("단위근 검정: 유효 관측치 10개 이상 필요")
        first, last = int(np.argmax(ok)), int(len(ok) - 1 - np.argmax(ok[::-1]))
        rows = np.arange(first, last + 1)
        if (~ok[rows]).any():
            raise ValueError("단위근 검정: 계열 중간 결측 — 결측을 처리한 뒤 분석 (Stata는 결측 시점에서 표본이 끊김)")
        _require_no_gaps(ti, rows, "단위근 검정")
        x = x_all[rows]
        T = len(x)
        tlab = {"constant": "상수항", "trend": "상수항 + 추세", "drift": "상수항 (drift)", "none": "결정항 없음"}[trend]
        t = ResultsTable(name="unitRootTests", title=f"단위근·정상성 검정 ({dep}, {tlab})")
        t.add_column("test", "검정", "text")
        t.add_column("h0", "귀무가설", "text")
        t.add_column("stat", "통계량", "number")
        t.add_column("c1", "1% 임계값", "number")
        t.add_column("c5", "5% 임계값", "number")
        t.add_column("c10", "10% 임계값", "number")
        t.add_column("p", "p", "number", "pvalue")
        t.add_column("lags", "시차", "integer")
        t.add_column("n", "N", "integer")
        for n_ in ti.notes:
            t.add_note(n_)
        narrative = []
        if test in ("adf", "all"):
            if lag_sel != "none":
                mx = max_lag if max_lag is not None else int(12 * (T / 100) ** 0.25)
                r = adf_test(x, None, trend, method=lag_sel, max_lags=mx)
                t.add_note(f"ADF 시차: {lag_sel.upper()} 최소 기준으로 0–{mx} 중 {r['lags']} 선택 (arch ADF 자동 선택, 최대 시차 기준 공통 표본). Stata dfuller는 시차를 직접 지정.")
            else:
                r = adf_test(x, lags, trend)
            t.add_row({"test": "ADF Z(t)", "h0": "단위근 존재", "stat": r["stat"], "c1": r["crit"][0], "c5": r["crit"][1],
                       "c10": r["crit"][2], "p": r["p"], "lags": r["lags"], "n": r["n"]})
            t.add_note(f"ADF (arch ADF): Δy_t = " + {"constant": "α + ", "trend": "α + δt + ", "drift": "α + ", "none": ""}[trend]
                       + f"βy_(t−1) + Σζ_jΔy_(t−j) + e_t, Z(t) = β의 t 통계량. {r['pnote']}. "
                       + ("임계값: Fuller (1976) 표를 N으로 선형 보간 (Stata dfuller와 같은 값; N > 500은 점근값)." if trend != "drift"
                          else "임계값: t 분포."))
            if show_reg:
                rt = ResultsTable(name="adfRegression", title="ADF 회귀식 (dfuller, regress)")
                rt.add_column("term", "변수", "text")
                rt.add_column("b", "계수", "number")
                rt.add_column("se", "표준오차", "number")
                rt.add_column("t", "t", "number")
                rt.add_column("p", "p", "number", "pvalue")
                dfr = r["df_resid"]
                for nm, b_, s_ in zip(r["names"], r["b"], r["se"]):
                    rt.add_row({"term": nm.replace(".y", "." + dep), "b": b_, "se": s_, "t": b_ / s_,
                                "p": float(2 * stats.t.sf(abs(b_ / s_), dfr))})
                rt.mark_coefficients(estimate="b", se="se", p="p", stat="t", label="term")
                rt.add_note(f"종속변수: D.{dep} (arch ADF 회귀식, OLS). L.{dep} 계수의 p값은 일반 t 분포 기준이며 단위근 검정에는 위의 Z(t)·임계값 사용.")
            narrative.append(f"ADF Z(t) = {r['stat']:.3f} (p {_fmt_p(r['p'])})")
        if test in ("pp", "all"):
            if trend == "drift":
                pass
            else:
                r = pp_test(x, pp_lags, trend)
                t.add_row({"test": "PP Z(ρ)", "h0": "단위근 존재", "stat": r["z_rho"], "c1": r["crit_rho"][0], "c5": r["crit_rho"][1],
                           "c10": r["crit_rho"][2], "p": float("nan"), "lags": r["lags"], "n": r["n"]})
                t.add_row({"test": "PP Z(t)", "h0": "단위근 존재", "stat": r["z_t"], "c1": r["crit"][0], "c5": r["crit"][1],
                           "c10": r["crit"][2], "p": r["p"], "lags": r["lags"], "n": r["n"]})
                t.add_note(f"PP (arch PhillipsPerron): y_t = " + {"constant": "α + ", "trend": "α + δt + ", "none": ""}[trend]
                           + "ρy_(t−1) + u_t, Newey–West 장기분산(Bartlett 가중, 시차 "
                           + f"{r['lags']}{' = int(4(T/100)^(2/9)), Stata 기본값' if pp_lags is None else ''}). Z(t) p값: MacKinnon (1994). "
                           "Z(ρ)·Z(t) 임계값: Fuller (1976) 표 선형 보간 (Stata pperron과 같음).")
                narrative.append(f"PP Z(t) = {r['z_t']:.3f} (p {_fmt_p(r['p'])})")
        if test in ("kpss", "all") and trend != "none":
            r = kpss_test(x, kpss_lags, "trend" if trend == "trend" else "constant")
            t.add_row({"test": "KPSS η", "h0": "(추세) 정상", "stat": r["stat"], "c1": r["crit"][0], "c5": r["crit"][1],
                       "c10": r["crit"][2], "p": r["p"], "lags": r["lags"], "n": T})
            t.add_note(f"KPSS (arch KPSS): 귀무가설이 정상성 (ADF·PP와 반대). Bartlett 가중 장기분산, 시차 {r['lags']}"
                       + (" (Hobijn 외 1998 자동 선택)" if kpss_lags is None else "") + ". 임계값·p값: arch 반응곡면 (Kwiatkowski 외 1992 표와 소수 셋째 자리 차이 가능). "
                       "Stata에는 내장 KPSS가 없음 (ssc kpss).")
            narrative.append(f"KPSS η = {r['stat']:.3f} (p {_fmt_p(r['p'])})")
        self.add_table(t)
        if show_reg and test in ("adf", "all"):
            self.add_table(rt)
        self.apa_narrative = f"{dep}의 단위근 검정({tlab}) 결과, " + ", ".join(narrative) + "."
        args = {k: v for k, v in options.items() if v not in (None, "", [], False) and not (k == "lagSelect" and v == "none") and k != "timeFreq"}
        self.syntax_python = "sess.run('unitRoot', " + ", ".join(f"{k}={v!r}" for k, v in args.items()) + ")"
        self.syntax_r = "\n".join(r_code("unitRoot", options) or [])


# ---------------------------------------------------------------- corrgram
class CorrgramAnalysis(BaseAnalysis):
    """Stata corrgram (AC, PAC, Ljung–Box Q) with ac/pac plots and wntestq."""

    def __init__(self, df: pd.DataFrame, params: Dict[str, Any]):
        p = dict(params or {})
        dep = _as_single(_opt(p, "dep", "var", "y", "series"))
        time = _as_single(_opt(p, "time", "timeVar", "time_var"))
        if not dep:
            raise ValueError("자기상관 분석: 시계열 변수(dep) 지정 필요")
        _check_columns(df, "자기상관 분석", {"시계열 변수": dep, "시간 변수": time})
        lags = _int(p, "lags", "nlags", default=None, lo=0, label="시차(lags)") or None                       # 0 = Stata default
        method = resolve_choice("편자기상관 계산(pacMethod)", _opt(p, "pacMethod", "pac_method", default="regression"),
                                ("regression", "yw"), {"ols": "regression", "yule-walker": "yw", "yulewalker": "yw"})
        wq = _int(p, "wntestqLags", "wntestq_lags", "qLags", default=None, lo=0, label="wntestq 시차") or None
        plots = _flag(p, "plots", "plot", default=True)
        options = {"dep": dep, "time": time, "lags": lags, "pacMethod": method, "wntestqLags": wq, "plots": plots}
        super().__init__(name="corrgram", title=f"자기상관·편자기상관 (corrgram) — {dep}", dataset=df, options=options)
        ti = TimeIndex(df, time, "자기상관 분석")
        if ti.freq:
            self.options["timeFreq"] = ti.freq
        x_all = pd.to_numeric(ti.df[dep], errors="coerce").to_numpy(float)
        ok = ~np.isnan(x_all)
        first, last = int(np.argmax(ok)), int(len(ok) - 1 - np.argmax(ok[::-1]))
        rows = np.arange(first, last + 1)
        if (~ok[rows]).any() or ok.sum() < 4:
            raise ValueError("자기상관 분석: 계열 중간 결측 또는 관측치 부족")
        _require_no_gaps(ti, rows, "자기상관 분석")
        x = x_all[rows]
        n = len(x)
        L = min(lags or default_q_lags(n), n - 2)
        r = acf(x, L)
        pr = pacf_regression(x, L) if method == "regression" else pacf_yw(x, L)
        q, pq, _ = ljung_box(x, L)
        t = ResultsTable(name="corrgram", title=f"자기상관표 ({dep}, N = {n})")
        t.add_column("lag", "시차", "integer")
        t.add_column("ac", "AC", "number")
        t.add_column("pac", "PAC", "number")
        t.add_column("q", "Q", "number")
        t.add_column("p", "Prob > Q", "number", "pvalue")
        t.add_column("bse", "AC 표준오차 (Bartlett)", "number")
        bse = np.sqrt((1 + 2 * np.concatenate([[0.0], np.cumsum(r[:-1] ** 2)])) / n)
        for k in range(L):
            t.add_row({"lag": k + 1, "ac": r[k], "pac": pr[k], "q": q[k], "p": pq[k], "bse": bse[k]})
        t.add_note("AC: 전체 평균 기준 표본 자기상관. PAC: " + ("y_t를 상수와 y_(t−1)…y_(t−k)에 OLS 회귀한 y_(t−k) 계수 (Stata pac 기본)."
                   if method == "regression" else "Yule–Walker (Durbin–Levinson) 방식 (Stata pac, yw·R pacf와 같음).")
                   + " Q: Ljung–Box 누적 통계량, 자유도 = 시차.")
        for n_ in ti.notes:
            t.add_note(n_)
        self.add_table(t)
        wl = min(wq or default_q_lags(n), n - 1)
        qq, pp_, _ = ljung_box(x, wl)
        w = ResultsTable(name="wntestq", title="백색잡음 포트맨토 검정 (wntestq)")
        w.add_column("stat", "항목", "text")
        w.add_column("value", "값", "number")
        w.add_row({"stat": "Portmanteau (Q) 통계량", "value": float(qq[-1])})
        w.add_row({"stat": "자유도 (시차)", "value": wl, "_rowfmt": "integer"})
        w.add_row({"stat": "Prob > χ²", "value": float(pp_[-1]), "_rowfmt": "pvalue"})
        w.add_note(f"귀무가설: {dep}가 백색잡음. 기본 시차 = min(⌊n/2⌋ − 2, 40) (Stata 기본).")
        self.add_table(w)
        if plots:
            self.add_plot("자기상관·편자기상관 그림 (ac·pac)", _plot_acf_pacf(r, pr, n, f"{dep}: ACF·PACF"))
        self.apa_narrative = (f"{dep}의 시차 1 자기상관은 {r[0]:.3f}, 편자기상관은 {pr[0]:.3f}였고, "
                              f"Ljung–Box Q({wl}) = {qq[-1]:.2f} (p {_fmt_p(float(pp_[-1]))})로 백색잡음 가설을 "
                              f"{'기각하였다' if pp_[-1] < 0.05 else '기각하지 못하였다'}.")
        args = {k: v for k, v in options.items() if v not in (None, "", [], False) and not (k == "pacMethod" and v == "regression") and k != "timeFreq"}
        self.syntax_python = "sess.run('corrgram', " + ", ".join(f"{k}={v!r}" for k, v in args.items()) + ")"
        self.syntax_r = "\n".join(r_code("corrgram", options) or [])


# ---------------------------------------------------------------- Stata / R code export
_STATA_DATE = {
    "M": ('gen _tm = monthly(substr({v}, 1, 7), "YM")', "%tm", "_tm"),
    "Q": ('gen _tq = quarterly({v}, "YQ")   // YYYY-MM-DD 형식이면 qofd(date({v}, "YMD"))', "%tq", "_tq"),
    "Y": ("gen _ty = real(substr({v}, 1, 4))", "%ty", "_ty"),
    "W": ('gen _tw = wofd(date({v}, "YMD"))', "%tw", "_tw"),
    "D": ('gen _td = date({v}, "YMD")', "%td", "_td"),
}


def _ts_time(p: Dict[str, Any]) -> str:
    t = p.get("time")
    if not t:
        return "_t"
    f = p.get("timeFreq")
    return _STATA_DATE[f][2] if f in _STATA_DATE else t


def _ts_head(p: Dict[str, Any]) -> List[str]:
    t = p.get("time")
    if not t:
        return ["gen _t = _n", "tsset _t"]
    f = p.get("timeFreq")
    if f in _STATA_DATE:
        gen, fmt, var = _STATA_DATE[f]
        return [gen.format(v=t), f"format {var} {fmt}", f"tsset {var}"]
    return [f"tsset {t}"]


def stata_code(t: str, p: Dict[str, Any]) -> Optional[List[str]]:
    if t == "arima":
        dep = _as_single(p.get("dep"))
        exog = _as_list(p.get("exog"))
        d, D = int(p.get("d") or 0), int(p.get("D") or 0)
        s = p.get("s")
        ar = p.get("arLags") or list(range(1, int(p.get("p") or 0) + 1))
        ma = p.get("maLags") or list(range(1, int(p.get("q") or 0) + 1))
        opts = []
        if p.get("diffExog") is False and (d or D):
            pre = (f"S{s}." if D == 1 else f"S{s}{D}." if D else "") + ("D." if d == 1 else f"D{d}." if d else "")
            lhs = pre + dep
        else:
            lhs = dep
            if d and not D and not p.get("arLags") and not p.get("maLags") and not (p.get("P") or p.get("Q")):
                opts.append(f"arima({len(ar)},{d},{len(ma)})")
                ar = ma = []
            elif d or D:
                pre = (f"S{s}." if D == 1 else f"S{s}{D}." if D else "") + ("D." if d == 1 else f"D{d}." if d else "")
                lhs = pre + dep
                exog = [pre + x for x in exog]
        if ar:
            opts.append(f"ar({' '.join(map(str, ar))})")
        if ma:
            opts.append(f"ma({' '.join(map(str, ma))})")
        if p.get("P"):
            opts.append(f"mar({' '.join(str(i) for i in range(1, int(p['P']) + 1))}, {s})")
        if p.get("Q"):
            opts.append(f"mma({' '.join(str(i) for i in range(1, int(p['Q']) + 1))}, {s})")
        if p.get("constant") is False:
            opts.append("noconstant")
        if p.get("covType") and p["covType"] != "opg":
            opts.append(f"vce({p['covType']})")
        lines = _ts_head(p) + [f"arima {lhs}{' ' + ' '.join(exog) if exog else ''}, {' '.join(opts)}".rstrip(", "),
                               "estat ic", "predict double _res, residuals",
                               f"wntestq _res{', lags(' + str(p['lbLags']) + ')' if p.get('lbLags') else ''}"]
        h = int(p.get("forecast") or 0)
        if h > 0:
            tv = _ts_time(p)
            yopt = "y " if (d or D) else ""
            if exog:
                lines += [f"* 예측: 미래 시점의 {' '.join(exog)} 값이 있는 행(종속변수 결측)이 데이터 끝에 필요",
                          "summarize " + tv + f" if !missing({dep}), meanonly",
                          f"predict double _fc, {yopt}dynamic(`=r(max)+1')",
                          f"predict double _fcmse, mse dynamic(`=r(max)+1')"]
            else:
                lines += [f"tsappend, add({h})",
                          f"predict double _fc, {yopt}dynamic(`={tv}[_N-{h}+1]')",
                          f"predict double _fcmse, mse dynamic(`={tv}[_N-{h}+1]')"]
            lines.append("gen double _fclo = _fc - invnormal(0.975)*sqrt(_fcmse)   // 95% 예측구간"
                         if float(p.get("ciWidth") or 95) == 95 else
                         f"gen double _fclo = _fc - invnormal({0.5 + float(p.get('ciWidth')) / 200:g})*sqrt(_fcmse)")
        return lines
    if t == "unitRoot":
        dep = _as_single(p.get("dep"))
        tr = p.get("trend", "constant")
        opt = {"constant": "", "trend": " trend", "drift": " drift", "none": " noconstant"}[tr]
        lines = _ts_head(p)
        test = p.get("test", "adf")
        if test in ("adf", "all"):
            lines.append(f"dfuller {dep},{opt} lags({int(p.get('lags') or 0)}){' regress' if p.get('regress') else ''}")
            if p.get("lagSelect") and p["lagSelect"] != "none":
                lines.append("* 시차 자동 선택(AIC·BIC)은 dfuller에 없음 — varsoc 등으로 시차를 정한 뒤 lags() 지정")
        if test in ("pp", "all") and tr != "drift":
            lines.append(f"pperron {dep},{opt}" + (f" lags({p['ppLags']})" if p.get("ppLags") is not None else ""))
        if test in ("kpss", "all") and tr != "none":
            lines.append(f"kpss {dep}{', notrend' if tr != 'trend' else ''}" + (f" maxlag({p['kpssLags']})" if p.get("kpssLags") is not None else "")
                         + "   // 사용자 작성 명령: ssc install kpss")
        return lines
    if t == "corrgram":
        dep = _as_single(p.get("dep"))
        lg = f", lags({p['lags']})" if p.get("lags") else ""
        yw = (" yw" if lg else ", yw") if p.get("pacMethod") == "yw" else ""
        return _ts_head(p) + [f"corrgram {dep}{lg}{yw}", f"ac {dep}{lg}", f"pac {dep}{lg}{yw}",
                              f"wntestq {dep}" + (f", lags({p['wntestqLags']})" if p.get("wntestqLags") else "")]
    return None


def r_code(t: str, p: Dict[str, Any]) -> Optional[List[str]]:
    if t == "arima":
        dep = _as_single(p.get("dep"))
        exog = _as_list(p.get("exog"))
        d, D = int(p.get("d") or 0), int(p.get("D") or 0)
        s = p.get("s")
        ar = p.get("arLags") or list(range(1, int(p.get("p") or 0) + 1))
        ma = p.get("maLags") or list(range(1, int(p.get("q") or 0) + 1))
        pa, qa = (max(ar) if ar else 0), (max(ma) if ma else 0)
        lines = ["library(forecast)"]
        if p.get("time"):
            lines.append(f'data <- data[order(data${p["time"]}), ]')
        freq = int(s) if s else 1
        lines.append(f"y <- ts(data${dep}, frequency = {freq})")
        fixed = None
        if ar != list(range(1, pa + 1)) or ma != list(range(1, qa + 1)):
            fx = ["0" if i not in ar else "NA" for i in range(1, pa + 1)] + ["0" if i not in ma else "NA" for i in range(1, qa + 1)]
            fixed = fx
        xreg = ""
        if exog:
            lines.append(f"X <- as.matrix(data[, c({', '.join(_q(x) for x in exog)})])")
            xreg = ", xreg = X"
        seas = f", seasonal = c({int(p.get('P') or 0)}, {D}, {int(p.get('Q') or 0)})" if (p.get("P") or D or p.get("Q")) else ""
        const = p.get("constant", True) is not False
        if d + D == 0:
            cst = f", include.mean = {'TRUE' if const else 'FALSE'}"
        elif d == 1 and D == 0:
            cst = f", include.drift = {'TRUE' if const else 'FALSE'}   # drift = Stata _cons (차분 계열 평균)"
        elif d == 0 and D == 1:
            cst = f", include.drift = {'TRUE' if const else 'FALSE'}   # Stata _cons = {s} × drift (계절 차분 계열 평균)"
        else:
            cst = "   # d + D ≥ 2: R은 상수 미지원 — Stata는 상수 포함 가능"
        if fixed is not None:
            nfx = len(fixed) + (len(exog)) + (1 if const and d + D <= 1 else 0)
            lines.append(f"fx <- c({', '.join(fixed)}{', NA' * (nfx - len(fixed))})   # 비연속 시차: 제외 시차 0 고정")
            cst += ", fixed = fx, transform.pars = FALSE"
        lines.append(f'fit <- Arima(y, order = c({pa}, {d}, {qa}){seas}{xreg}{cst.split("   #")[0]}, method = "ML")'
                     + ("   #" + cst.split("   #")[1] if "   #" in cst else ""))
        lines += ["summary(fit)   # 표준오차: 수치 헤시안(Stata vce(oim)에 해당; Stata 기본은 OPG)",
                  f"Box.test(residuals(fit), lag = {p.get('lbLags') or 'min(length(y) %/% 2 - 2, 40)'}, type = \"Ljung-Box\")   # Stata wntestq: fitdf = 0"]
        h = int(p.get("forecast") or 0)
        if h > 0:
            lines.append(f"predict(fit, n.ahead = {h}{', newxreg = Xf' if exog else ''})   # 예측값·표준오차 (최우추정 σ² 사용 = Stata predict, mse)"
                         + ("; Xf: 미래 외생변수 값 행렬" if exog else ""))
            lines.append("# forecast::forecast()의 구간은 자유도 보정 σ²를 써서 약간 넓음")
        return lines
    if t == "unitRoot":
        dep = _as_single(p.get("dep"))
        tr = p.get("trend", "constant")
        lines = ["library(urca)"]
        if p.get("time"):
            lines.append(f'data <- data[order(data${p["time"]}), ]')
        lines.append(f"x <- data${dep}")
        test = p.get("test", "adf")
        typ = {"constant": "drift", "drift": "drift", "trend": "trend", "none": "none"}[tr]
        if test in ("adf", "all"):
            lines.append(f'summary(ur.df(x, type = "{typ}", lags = {int(p.get("lags") or 0)}))   # urca "drift" = Stata 기본(상수항); 임계값은 Fuller 표')
        if test in ("pp", "all") and tr != "drift":
            lines.append(f'summary(ur.pp(x, type = "Z-tau", model = "{"trend" if tr == "trend" else "constant"}"'
                         + (f", use.lag = {p['ppLags']}" if p.get("ppLags") is not None else ', lags = "short"') + "))")
        if test in ("kpss", "all") and tr != "none":
            lines.append(f'summary(ur.kpss(x, type = "{"tau" if tr == "trend" else "mu"}"'
                         + (f", use.lag = {p['kpssLags']}" if p.get("kpssLags") is not None else "") + "))")
        return lines
    if t == "corrgram":
        dep = _as_single(p.get("dep"))
        L = p.get("lags") or f"min(length(data${dep}) %/% 2 - 2, 40)"
        return [f"x <- data${dep}" if not p.get("time") else f'x <- data[order(data${p["time"]}), ]${dep}',
                f"acf(x, lag.max = {L})",
                f"pacf(x, lag.max = {L})   # R pacf = Yule–Walker (Stata pac, yw); Stata 기본은 회귀 방식",
                f"Box.test(x, lag = {p.get('wntestqLags') or L}, type = \"Ljung-Box\")"]
    return None


def _q(s: str) -> str:
    return '"' + str(s) + '"'


# ---------------------------------------------------------------- dispatch + Stata input rules
def dispatch(t: str, df: pd.DataFrame, params: Dict[str, Any]):
    if t == "arima":
        return ArimaAnalysis(df, params)
    if t == "unitRoot":
        return UnitRootAnalysis(df, params)
    if t == "corrgram":
        return CorrgramAnalysis(df, params)
    return None


def _tr_opts(opt_txt: str) -> Dict[str, str]:
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


def _numlist(v: str) -> List[int]:
    out: List[int] = []
    for tok in v.replace(",", " ").split():
        if "/" in tok:
            a, b = tok.split("/", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def _ts_prefix(v: str) -> Tuple[int, int, int, str]:
    """'D.y' -> (1, 0, 0, 'y'); 'S12.y' -> (0, 1, 12, 'y'); 'DS12.y' / 'S12D.y' -> (1, 1, 12, 'y')."""
    import re
    m = re.match(r"^([A-Za-z0-9]*)\.(\w+)$", v)
    if not m:
        return 0, 0, 0, v
    pre, name = m.group(1).upper(), m.group(2)
    d = D = s = 0
    for mm in re.finditer(r"([DS])(\d*)", pre):
        op, num = mm.group(1), mm.group(2)
        if op == "D":
            d += int(num) if num else 1
        else:
            D += 1
            s = int(num) if num else 1
    if re.sub(r"([DS])(\d*)", "", pre):
        raise ValueError(f"시계열 연산자 '{pre}.' 미지원 (D.·S#.만 가능)")
    return d, D, s, name


_TIME_EXPR = "time=getattr(sess, 'ts_spec', {}).get('timeVar')"


def _ni(msg: str) -> str:
    return f"raise NotImplementedError({msg!r})"


def transpile(line: str) -> Optional[str]:
    import re
    s = line.strip()
    m = re.match(r"^tsset\s+([^,]+?)\s*(?:,\s*(.*))?$", s, re.IGNORECASE)
    if m:
        vars_ = m.group(1).split()
        if len(vars_) == 2:
            return f"sess.xtset({vars_[0]!r}, {vars_[1]!r})  # [Stata] tsset 패널변수 시간변수 = xtset"
        if len(vars_) != 1:
            return _ni("tsset: 시간 변수 1개(또는 패널변수 시간변수) 필요 — " + s)
        return (f"sess.tsset({vars_[0]!r})  # [Stata] tsset: 시계열 분석(arima·unitRoot·corrgram)의 기본 시간 변수"
                + (f" (옵션 {m.group(2)}: 주기는 시간 변수 값에서 판단)" if m.group(2) else ""))
    m = re.match(r"^arima\s+([^,]+?)\s*(?:,\s*(.*))?$", s, re.IGNORECASE)
    if m:
        toks = m.group(1).split()
        try:
            pres = [_ts_prefix(t) for t in toks]
        except ValueError as e:
            return _ni(f"arima: {e} — {s}")
        dd, DD, ss, dep = pres[0]
        exog = [x[3] for x in pres[1:]]
        ex_pre = {(x[0], x[1], x[2]) for x in pres[1:]}
        kw: Dict[str, Any] = {"dep": dep}
        if exog:
            kw["exog"] = exog
        if len(ex_pre) > 1:
            return _ni("arima: 외생변수마다 차분 연산자가 다른 형식 미지원 — " + s)
        if exog and (dd or DD):
            if ex_pre == {(dd, DD, ss)}:
                kw["diffExog"] = True
            elif ex_pre == {(0, 0, 0)}:
                kw["diffExog"] = False
            else:
                return _ni("arima: 종속변수와 외생변수의 차분 연산자 조합 미지원 — " + s)
        elif exog and ex_pre != {(0, 0, 0)}:
            return _ni("arima: 종속변수를 차분하지 않고 외생변수만 차분한 형식 미지원 (변수를 만든 뒤 지정) — " + s)
        opts = _tr_opts(m.group(2) or "")
        d, D, sP = dd, DD, ss
        p_ = q_ = 0
        bad = []
        for k, v in opts.items():
            try:
                if k == "arima":
                    a = [int(x) for x in v.replace(" ", "").split(",")]
                    if len(a) != 3:
                        raise ValueError("arima(p,d,q) 형식")
                    p_, d2, q_ = a
                    d += d2
                elif k == "sarima":
                    a = [int(x) for x in v.replace(" ", "").split(",")]
                    if len(a) != 4:
                        raise ValueError("sarima(P,D,Q,s) 형식")
                    kw["P"], D2, kw["Q"], s2 = a
                    D += D2
                    if sP and sP != s2:
                        raise ValueError("계절 차분 주기와 sarima 주기 불일치")
                    sP = s2
                elif k == "ar":
                    kw["arLags"] = _numlist(v)
                elif k == "ma":
                    kw["maLags"] = _numlist(v)
                elif k in ("mar", "mma"):
                    lst, sv = v.rsplit(",", 1)
                    lags_ = _numlist(lst)
                    if lags_ != list(range(1, len(lags_) + 1)):
                        raise ValueError(f"{k}(): 1부터 연속된 시차만 가능")
                    kw["P" if k == "mar" else "Q"] = len(lags_)
                    if sP and sP != int(sv):
                        raise ValueError("계절 주기 불일치")
                    sP = int(sv)
                elif k in ("noconstant", "nocons", "nocon", "noconst"):
                    kw["constant"] = False
                elif k == "vce":
                    vv = v.strip().lower()
                    kw["covType"] = {"opg": "opg", "oim": "oim", "robust": "robust", "r": "robust"}[vv]
                elif k in ("level", "l"):
                    kw["ciWidth"] = float(v)
                elif k in ("iterate", "iter"):
                    kw["maxiter"] = int(v)
                else:
                    bad.append(k)
            except (ValueError, KeyError) as e:
                return _ni(f"arima {k}({v}): {e} — {s}")
        if bad:
            return _ni("arima 옵션 미지원: " + ", ".join(bad) + " — " + s)
        if "arLags" in kw and p_:
            return _ni("arima: arima(p,d,q)와 ar()를 함께 지정할 수 없음 — " + s)
        if "maLags" in kw and q_:
            return _ni("arima: arima(p,d,q)와 ma()를 함께 지정할 수 없음 — " + s)
        kw["p"], kw["d"], kw["q"] = p_, d, q_
        if "arLags" in kw:
            kw.pop("p")
        if "maLags" in kw:
            kw.pop("q")
        if D:
            kw["D"] = D
        if D or kw.get("P") or kw.get("Q"):
            kw["s"] = sP
        call = "sess.run('arima', " + ", ".join(f"{k}={v!r}" for k, v in kw.items())
        return call + f", {_TIME_EXPR})"
    m = re.match(r"^(dfuller|pperron|kpss)\s+(\S+)\s*(?:,\s*(.*))?$", s, re.IGNORECASE)
    if m:
        cmd, var, ot = m.group(1).lower(), m.group(2), m.group(3) or ""
        if "." in var:
            return _ni(f"{cmd}: 시계열 연산자 표기 미지원 (변수를 만든 뒤 지정) — {s}")
        opts = _tr_opts(ot)
        kw = {"dep": var, "test": {"dfuller": "adf", "pperron": "pp", "kpss": "kpss"}[cmd]}
        bad = []
        trend = None
        for k, v in opts.items():
            if k in ("trend", "t") and cmd != "kpss":
                trend = "trend"
            elif k == "drift" and cmd == "dfuller":
                trend = "drift"
            elif k in ("noconstant", "nocons", "nocon") and cmd != "kpss":
                trend = "none"
            elif k == "notrend" and cmd == "kpss":
                trend = "constant"
            elif k in ("lags", "lag", "l") and cmd != "kpss":
                kw["lags" if cmd == "dfuller" else "ppLags"] = int(v)
            elif k == "maxlag" and cmd == "kpss":
                kw["kpssLags"] = int(v)
            elif k in ("regress", "reg") and cmd == "dfuller":
                kw["regress"] = True
            else:
                bad.append(k)
        if bad:
            return _ni(f"{cmd} 옵션 미지원: " + ", ".join(bad) + " — " + s)
        if cmd == "kpss":
            trend = trend or "trend"        # Stata kpss (ssc) default: trend stationarity
            if "kpssLags" in kw:
                kw["kpssLags"] = kw["kpssLags"]
        kw["trend"] = trend or "constant"
        note = "  # kpss maxlag(): 지정 시차 1개로 계산 (ssc kpss는 0…maxlag 전부 표시)" if cmd == "kpss" and "kpssLags" in kw else ""
        return "sess.run('unitRoot', " + ", ".join(f"{k}={v!r}" for k, v in kw.items()) + f", {_TIME_EXPR}){note}"
    m = re.match(r"^(corrgram|ac|pac|wntestq)\s+(\S+)\s*(?:,\s*(.*))?$", s, re.IGNORECASE)
    if m:
        cmd, var, ot = m.group(1).lower(), m.group(2), m.group(3) or ""
        if "." in var:
            return _ni(f"{cmd}: 시계열 연산자 표기 미지원 (변수를 만든 뒤 지정) — {s}")
        opts = _tr_opts(ot)
        kw = {"dep": var}
        bad = []
        for k, v in opts.items():
            if k in ("lags", "lag", "l"):
                kw["wntestqLags" if cmd == "wntestq" else "lags"] = int(v)
            elif k == "yw" and cmd in ("corrgram", "pac"):
                kw["pacMethod"] = "yw"
            elif k in ("noplot", "nograph") or (cmd in ("ac", "pac") and k in ("level", "generate", "gen")):
                if k in ("generate", "gen"):
                    bad.append(k)
            else:
                bad.append(k)
        if bad:
            return _ni(f"{cmd} 옵션 미지원: " + ", ".join(bad) + " — " + s)
        return "sess.run('corrgram', " + ", ".join(f"{k}={v!r}" for k, v in kw.items()) + f", {_TIME_EXPR})"
    return None


EXTENSION = {
    "catalog": [
        {"id": "arima", "group": "Time Series", "title": "ARIMA / ARMAX (arima)", "module": "openstat", "icon": "analysis-regression.svg"},
        {"id": "unitRoot", "group": "Time Series", "title": "Unit Root Tests (dfuller / pperron / KPSS)", "module": "openstat",
         "icon": "analysis-regression.svg"},
        {"id": "corrgram", "group": "Time Series", "title": "Autocorrelations (corrgram / ac / pac / wntestq)", "module": "openstat",
         "icon": "analysis-regression.svg"},
    ],
    "aliases": {"sarima": "arima", "armax": "arima", "ARIMA": "arima", "dfuller": "unitRoot", "adf": "unitRoot",
                "pperron": "unitRoot", "kpss": "unitRoot", "unitroot": "unitRoot", "ac": "corrgram", "pac": "corrgram",
                "wntestq": "corrgram", "acf": "corrgram"},
    "dispatch": dispatch,
    "stata": stata_code,
    "r": r_code,
    "transpile": transpile,
}
