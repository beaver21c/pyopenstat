"""
Latent class / latent profile analysis (LCA, LPA), latent transition analysis (LTA) and latent class growth
analysis (LCGA) with Mplus TYPE = MIXTURE defaults. Python keyword options; engines in mixture_models.py.

Mplus defaults applied
- Indicators are continuous unless listed as categorical (Mplus CATEGORICAL); categorical indicators may have
  up to 10 categories, coded by their sorted observed values.
- ESTIMATOR = MLR, STARTS = 20 4, STITERATIONS = 10, STSEED = 0, missing indicators by FIML (MAR).
- LPA: class-specific means, variances equal across classes, covariances zero.
- LTA: measurement parameters equal across occasions (Mplus User's Guide example 8.13), time-specific
  transition probabilities. LCGA: growth factor variances zero, residual variances time-specific and
  equal across classes.
- TECH11 (VLMR, adjusted LMR) and TECH14 (bootstrap LRT, LRTSTARTS = 0 0 40 8) on request.
- AUXILIARY: R3STEP covariates (reference = last class), BCH continuous distal outcomes, ML three-step
  categorical distal outcomes.
- Classes are reported in descending order of estimated size.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from .base import BaseAnalysis, ResultsTable, resolve_choice
from .plots import fig_to_base64, _pal
from .plot_style import get_active_style
from . import mixture_models as mm

MAX_CATEGORIES = 10
ESTIMATOR_LABELS = {
    "mlr": "MLR: 최우추정, 샌드위치(강건) 표준오차 (Mplus TYPE = MIXTURE 기본)",
    "ml": "ML: 최우추정, 관측 정보행렬 표준오차",
}
VAR_LABELS = {("equal", "zero"): "평균 계층별, 분산 계층 간 동일, 공분산 0 (Mplus 기본)",
              ("free", "zero"): "평균·분산 계층별, 공분산 0",
              ("equal", "equal"): "평균 계층별, 분산·공분산 계층 간 동일",
              ("free", "free"): "평균·분산·공분산 모두 계층별"}
RESVAR_LABELS = {"time_class_invariant": "시점별 추정, 계층 간 동일 (Mplus 기본)",
                 "class_specific": "시점별·계층별 추정", "equal": "모든 시점·계층 동일"}
GROWTH_LABELS = {"linear": "선형", "quadratic": "2차", "cubic": "3차"}
FACTOR_LABELS = {"I": "절편(I)", "S": "기울기(S)", "Q": "2차항(Q)", "C": "3차항(C)"}


# ============================================================ option helpers
def _p(p: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in p and p[k] is not None and not (isinstance(p[k], (str, list, tuple)) and len(p[k]) == 0):
            return p[k]
    return default


def _flag(v, default=False) -> bool:
    if v is None or v == "":
        return default
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


def _as_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.replace("\n", ",").split(",") if x.strip()]
    return [str(x) for x in v if str(x).strip()]


def _int(v, label: str, lo: int, hi: int, default: int) -> int:
    if v is None or v == "":
        return default
    try:
        x = int(float(v))
    except (TypeError, ValueError):
        raise ValueError(f"{label}: 정수 지정 필요 (입력값 '{v}')")
    if x < lo or x > hi:
        raise ValueError(f"{label}: {lo}~{hi} 범위로 지정 (입력값 {x})")
    return x


def _class_range(v) -> List[int]:
    if v is None or v == "" or (isinstance(v, (list, tuple)) and not v):
        return []
    if isinstance(v, (int, np.integer)):
        return list(range(1, int(v) + 1))
    if isinstance(v, str):
        s = v.replace(" ", "")
        for sep in ("-", ":", "~"):
            if sep in s:
                a, b = s.split(sep, 1)
                return list(range(int(a), int(b) + 1))
        return sorted({int(x) for x in s.split(",") if x})
    vals = [int(x) for x in v]
    if len(vals) == 2 and vals[1] > vals[0] + 1:
        return list(range(vals[0], vals[1] + 1))
    return sorted(set(vals))


def _common_options(p: Dict[str, Any]) -> Dict[str, Any]:
    o = {
        "estimator": resolve_choice("추정 방법", _p(p, "estimator", default="mlr"), ["mlr", "ml"], {"robust": "mlr"}, default="mlr"),
        "starts": _int(_p(p, "starts"), "무작위 시작값 수(starts)", 0, 5000, 20),
        "stIterations": _int(_p(p, "stIterations", "stiterations"), "초기 단계 반복 수(stIterations)", 1, 1000, 10),
        "seed": _int(_p(p, "seed", "stseed"), "시드(seed)", 0, 2 ** 31 - 1, 0),
        "maxIter": _int(_p(p, "maxIter", "miterations"), "최대 반복 수(maxIter)", 10, 100000, 5000),
        "tech11": _flag(_p(p, "tech11"), False),
        "tech14": _flag(_p(p, "tech14"), False),
        "blrtDraws": _int(_p(p, "blrtDraws", "lrtBootstrap"), "부트스트랩 표본 수(blrtDraws)", 2, 2000, 100),
    }
    o["finalStarts"] = _int(_p(p, "finalStarts"), "최종 단계 시작값 수(finalStarts)", 1, 5000, 4 if o["starts"] else 1)
    if o["starts"] and o["finalStarts"] > o["starts"]:
        raise ValueError("최종 단계 시작값 수(finalStarts)는 무작위 시작값 수(starts) 이하로 지정")
    lrt = _p(p, "lrtStarts", default=[0, 0, 40, 8])
    lrt = [int(x) for x in (_as_list(lrt) if isinstance(lrt, str) else lrt)]
    if len(lrt) != 4 or any(x < 0 for x in lrt) or lrt[3] < 1 or (lrt[2] and lrt[3] > lrt[2]):
        raise ValueError("lrtStarts: 음이 아닌 정수 4개 [k-1 계층 시작값, 최종, k 계층 시작값, 최종] 지정 (Mplus 기본 0 0 40 8)")
    o["lrtStarts"] = lrt
    o["vlmrMethod"] = resolve_choice("TECH11 계산 방식", _p(p, "vlmrMethod", default="mplus"), ["mplus", "vuong"], default="mplus")
    return o


def _require_columns(df: pd.DataFrame, cols: Sequence[str], what: str):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{what}: 없는 변수 {miss}. 사용 가능한 변수: {list(df.columns)}")


def _level_label(v) -> str:
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _code_categorical(s: pd.Series, name: str) -> Tuple[np.ndarray, List[Any]]:
    nonmiss = s[s.notna()]
    vals = list(pd.unique(nonmiss))
    try:
        levels = sorted(vals, key=lambda x: float(x))
    except (TypeError, ValueError):
        levels = sorted(vals, key=lambda x: str(x))
    if len(levels) < 2:
        raise ValueError(f"범주형 지표 '{name}': 관측 범주가 {len(levels)}개 — 2개 이상 필요")
    if len(levels) > MAX_CATEGORIES:
        raise ValueError(f"범주형 지표 '{name}': 범주 {len(levels)}개 — 최대 {MAX_CATEGORIES}개(Mplus 기준). 연속형 지표로 지정하거나 범주 통합 필요")
    pos = {v: i for i, v in enumerate(levels)}
    codes = np.array([pos[v] if pd.notna(v) else -1 for v in s.tolist()], dtype=int)
    return codes, levels


def _numeric(s: pd.Series, name: str) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce")
    bad = s.notna() & x.isna()
    if bad.any():
        raise ValueError(f"연속형 지표 '{name}': 숫자가 아닌 값 {int(bad.sum())}개 (예: {s[bad].iloc[0]!r})")
    return x.to_numpy(dtype=float)


def _indicator_lists(p: Dict[str, Any], df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat = _as_list(_p(p, "categorical", "catVars", "categoricalVars"))
    cont = _as_list(_p(p, "continuous", "contVars", "continuousVars"))
    allv = _as_list(_p(p, "vars", "indicators"))
    for v in allv:  # Mplus: USEVARIABLES are continuous unless listed in CATEGORICAL
        if v not in cat and v not in cont:
            cont.append(v)
    both = sorted(set(cat) & set(cont))
    if both:
        raise ValueError(f"범주형·연속형 지표 중복 지정: {both}")
    if not cat and not cont:
        raise ValueError("지표 지정 필요: categorical(범주형) 또는 continuous(연속형)")
    _require_columns(df, cat + cont, "지표")
    return cat, cont


def _var_structure(p: Dict[str, Any]) -> Tuple[str, str]:
    v = resolve_choice("연속형 지표 분산", _p(p, "variances", default="equal"), ["equal", "free"],
                       {"class_invariant": "equal", "invariant": "equal", "class_specific": "free", "varying": "free"}, default="equal")
    c = resolve_choice("연속형 지표 공분산", _p(p, "covariances", default="zero"), ["zero", "equal", "free"],
                       {"none": "zero", "independent": "zero", "class_invariant": "equal", "class_specific": "free"}, default="zero")
    if (v, c) in (("free", "equal"), ("equal", "free")):
        raise ValueError("분산·공분산 조합: 공분산 'equal'은 분산 'equal'과, 공분산 'free'는 분산 'free'와 함께 지정")
    return v, c


def _zp(est, se):
    est, se = np.asarray(est, float), np.asarray(se, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = est / se
    pv = 2 * norm.sf(np.abs(z))
    return z, pv


def _assumption_table(rows: List[Tuple[str, str]]) -> ResultsTable:
    t = ResultsTable(name="assumptions", title="분석 설정")
    t.add_column("item", "항목", col_type="text")
    t.add_column("value", "설정", col_type="text")
    for a, b in rows:
        t.add_row({"item": a, "value": b})
    return t


def _starts_rows(o: Dict[str, Any]) -> List[Tuple[str, str]]:
    if o["starts"] <= 0:
        return [("시작값", "기본 시작값 1개 (STARTS = 0)")]
    return [("무작위 시작값", f"STARTS = {o['starts']} {o['finalStarts']}, STITERATIONS = {o['stIterations']}, STSEED = {o['seed']} (Mplus 기본 20 4, 10, 0)")]


def _fit_options(o):
    return dict(starts=o["starts"], final=o["finalStarts"], stiter=o["stIterations"], seed=o["seed"], max_iter=o["maxIter"])


def _small_class_note(fit: mm.MixtureFit) -> Optional[str]:
    p = fit.par.get("p")
    if p is None and "pi" in fit.par:
        p = fit.par["pi"]
    if p is not None and len(p) > 1 and float(np.min(p)) < 0.01:
        return (f"최소 계층 비율 {100 * float(np.min(p)):.2f}% — 1% 미만 계층은 퇴화 해일 가능성. "
                "계층 수 축소 또는 분산 구조 제약 검토 필요.")
    return None


def _replication_note(fit: mm.MixtureFit) -> Optional[str]:
    if fit.est.replicated is False:
        return ("최우 로그우도 값이 다른 시작값에서 반복되지 않음 — 지역 최댓값일 가능성. "
                "무작위 시작값 수(starts·finalStarts) 확대 필요.")
    if not fit.est.converged:
        return "최종 단계에서 수렴 기준 미충족 — 최대 반복 수(maxIter) 확대 필요."
    return None


def _starts_table(fit: mm.MixtureFit) -> ResultsTable:
    t = ResultsTable(name="starts", title="최종 단계 로그우도 (시작값별)")
    t.add_column("rank", "순위", col_type="integer")
    t.add_column("ll", "로그우도", col_type="number")
    t.add_column("seed", "시드", col_type="integer")
    t.add_column("start", "초기 단계 번호", col_type="integer")
    t.add_column("iter", "반복 수", col_type="integer")
    t.add_column("conv", "수렴", col_type="text")
    for i, r in enumerate(fit.est.finals, 1):
        t.add_row({"rank": i, "ll": r.ll, "seed": r.seed, "start": r.start, "iter": r.iterations, "conv": "예" if r.converged else "아니오"})
    if fit.est.replicated is True:
        t.add_note("최우 로그우도 값이 2개 이상의 시작값에서 반복됨.")
    note = _replication_note(fit)
    if note:
        t.add_note(note)
    return t


def _fit_table(fit: mm.MixtureFit, entropy: float, extra: Optional[List[Tuple[str, float]]] = None) -> ResultsTable:
    t = ResultsTable(name="fit", title="모형 적합도")
    t.add_column("index", "지표", col_type="text")
    t.add_column("value", "값", col_type="number")
    rows = [("로그우도 (H0)", fit.ll), ("자유모수 수", fit.npar), ("AIC", fit.aic), ("BIC", fit.bic),
            ("표본크기 조정 BIC", fit.sabic), ("상대 엔트로피", entropy), ("분석 사례 수", fit.n)]
    for a, b in rows + (extra or []):
        t.add_row({"index": a, "value": b})
    t.add_note("표본크기 조정 BIC: n* = (n + 2) / 24. 상대 엔트로피 = 1 − Σ(−p·ln p) / (n·ln K).")
    return t


def _class_size_table(post: np.ndarray, p_est, p_se, lg_est, lg_se) -> ResultsTable:
    n, K = post.shape
    mlc = mm.most_likely(post)
    t = ResultsTable(name="classSizes", title="계층 비율")
    t.add_column("cls", "계층", col_type="text")
    t.add_column("n_model", "추정 사례 수", col_type="number", super_title="추정 모형 기준")
    t.add_column("pct_model", "%", col_type="number", fmt="pc", super_title="추정 모형 기준")
    t.add_column("se_model", "SE(%)", col_type="number", super_title="추정 모형 기준")
    t.add_column("n_mlc", "사례 수", col_type="integer", super_title="최우추정 계층 소속 기준")
    t.add_column("pct_mlc", "%", col_type="number", fmt="pc", super_title="최우추정 계층 소속 기준")
    t.add_column("logit", "로짓", col_type="number", super_title=f"계층 로짓 ({K}계층 기준)")
    t.add_column("logit_se", "SE", col_type="number", super_title=f"계층 로짓 ({K}계층 기준)")
    t.add_column("logit_p", "p", col_type="number", fmt="pvalue", super_title=f"계층 로짓 ({K}계층 기준)")
    zz, pp = _zp(lg_est, lg_se)
    for k in range(K):
        nm = int((mlc == k).sum())
        row = {"cls": f"{k + 1}계층", "n_model": float(post[:, k].sum()), "pct_model": 100 * float(p_est[k]),
               "se_model": 100 * float(p_se[k]) if np.isfinite(p_se[k]) else np.nan, "n_mlc": nm, "pct_mlc": 100.0 * nm / n}
        if k < K - 1:
            row.update({"logit": float(lg_est[k]), "logit_se": float(lg_se[k]), "logit_p": float(pp[k])})
        t.add_row(row)
    t.add_note("계층 번호는 추정 비율 내림차순으로 재배열. 계층 로짓은 마지막 계층 대비 log(p_k / p_K) (Mplus Means of C#k).")
    return t


def _classification_table(post: np.ndarray, title_suffix: str = "") -> ResultsTable:
    K = post.shape[1]
    mlc = mm.most_likely(post)
    M = mm.average_posterior(post, mlc)
    t = ResultsTable(name="classification", title="분류 정확도: 최우추정 계층(행)별 평균 사후확률" + title_suffix)
    t.add_column("cls", "최우추정 계층", col_type="text")
    t.add_column("n", "사례 수", col_type="integer")
    for k in range(K):
        t.add_column(f"c{k + 1}", f"{k + 1}계층", col_type="number")
    for s in range(K):
        row = {"cls": f"{s + 1}계층", "n": int((mlc == s).sum())}
        for k in range(K):
            row[f"c{k + 1}"] = float(M[s, k]) if np.isfinite(M[s, k]) else np.nan
        t.add_row(row)
    t.add_note("대각 원소가 1에 가까울수록 분류 정확. Mplus: Average Latent Class Probabilities for Most Likely Latent Class Membership (Row) by Latent Class (Column).")
    return t


def _compute_lrt(fits: Dict[int, mm.MixtureFit], K: int, o: Dict[str, Any], infs: Dict[int, mm.Inference],
                 make_model=None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"K": K}
    f1, f0 = fits[K], fits[K - 1]
    out["ll0"], out["lr"] = f0.ll, 2 * (f1.ll - f0.ll)
    out["dp"] = f1.npar - f0.npar
    if o["tech11"]:
        for k in (K, K - 1):
            if k not in infs:
                infs[k] = mm.inference(fits[k].model, fits[k].par, o["estimator"])
        r = mm.vlmr_test(infs[K], infs[K - 1], f1.ll, f0.ll, f1.n, o["vlmrMethod"])
        out.update({"vlmr_mean": r["mean"], "vlmr_sd": r["sd"], "vlmr_p": r["p"], "lmr": r["adj"], "lmr_p": r["p_adj"]})
    if o["tech14"]:
        b = mm.blrt(f0, K, out["lr"], o["blrtDraws"], tuple(o["lrtStarts"]), o["stIterations"], o["seed"], o["maxIter"],
                    make_model=make_model)
        out.update({"blrt_p": b["p"], "blrt_draws": b["draws"]})
    return out


def _lrt_table(lrts: List[Dict[str, Any]], o: Dict[str, Any]) -> ResultsTable:
    t = ResultsTable(name="lrtTests", title="계층 수 검정: k−1 계층 대 k 계층")
    t.add_column("cmp", "비교", col_type="text")
    t.add_column("ll0", "H0 로그우도", col_type="number")
    t.add_column("lr", "2×로그우도 차이", col_type="number")
    t.add_column("dp", "모수 수 차이", col_type="integer")
    if o["tech11"]:
        t.add_column("vlmr_mean", "평균", col_type="number", super_title="VLMR (TECH11)")
        t.add_column("vlmr_sd", "표준편차", col_type="number", super_title="VLMR (TECH11)")
        t.add_column("vlmr_p", "p", col_type="number", fmt="pvalue", super_title="VLMR (TECH11)")
        t.add_column("lmr", "값", col_type="number", super_title="조정 LMR")
        t.add_column("lmr_p", "p", col_type="number", fmt="pvalue", super_title="조정 LMR")
    if o["tech14"]:
        t.add_column("blrt_p", "p", col_type="number", fmt="pvalue", super_title="부트스트랩 LRT (TECH14)")
        t.add_column("blrt_draws", "표본 수", col_type="integer", super_title="부트스트랩 LRT (TECH14)")
    for r in lrts:
        row = dict(r)
        row["cmp"] = f"{r['K'] - 1} 대 {r['K']} 계층"
        t.add_row(row)
    if o["tech11"]:
        t.add_note("VLMR: 가중 카이제곱 합 분포(Lo, Mendell & Rubin, 2001; Vuong, 1989), p는 Imhof 방법. "
                   + ("Mplus 방식: 정보행렬 역행렬 대신 샌드위치 공분산 사용(Vermunt, 2024). " if o["vlmrMethod"] == "mplus" else "Vuong(1989) 원식. ")
                   + "조정 LMR = 2×로그우도 차이 / (1 + 1/(모수 수 차이 × ln n)). "
                   "k−1 계층 모형은 별도 추정한 최우해 사용(Mplus는 k 계층 모형의 첫 계층 제거).")
    if o["tech14"]:
        s = o["lrtStarts"]
        t.add_note(f"부트스트랩 LRT: k−1 계층 추정치로 생성한 자료에 두 모형 재추정, LRTSTARTS = {s[0]} {s[1]} {s[2]} {s[3]} "
                   "(Mplus 기본 0 0 40 8), p = 생성 자료의 2×로그우도 차이가 관측값 이상인 비율. 관측 자료의 결측 형태 유지.")
    return t


def _enumeration_table(fits: Dict[int, mm.MixtureFit], Ks: List[int], entropy_fn, lrt_by_k: Dict[int, Dict[str, Any]],
                       o: Dict[str, Any]) -> ResultsTable:
    t = ResultsTable(name="enumeration", title="계층 수 비교")
    t.add_column("K", "계층 수", col_type="integer")
    t.add_column("ll", "로그우도", col_type="number")
    t.add_column("npar", "모수 수", col_type="integer")
    t.add_column("aic", "AIC", col_type="number")
    t.add_column("bic", "BIC", col_type="number")
    t.add_column("sabic", "SABIC", col_type="number")
    t.add_column("entropy", "엔트로피", col_type="number")
    t.add_column("minpct", "최소 계층 %", col_type="number", fmt="pc")
    t.add_column("rep", "최우 LL 반복", col_type="text")
    if o["tech11"]:
        t.add_column("vlmr_p", "VLMR p", col_type="number", fmt="pvalue")
        t.add_column("lmr_p", "조정 LMR p", col_type="number", fmt="pvalue")
    if o["tech14"]:
        t.add_column("blrt_p", "BLRT p", col_type="number", fmt="pvalue")
    best = min(Ks, key=lambda k: fits[k].bic)
    for K in Ks:
        f = fits[K]
        row = {"K": K, "ll": f.ll, "npar": f.npar, "aic": f.aic, "bic": f.bic, "sabic": f.sabic,
               "entropy": entropy_fn(f), "minpct": 100 * float(np.min(f.par["p"])) if "p" in f.par else np.nan,
               "rep": {True: "예", False: "아니오", None: "—"}[f.est.replicated]}
        if K in lrt_by_k:
            for key in ("vlmr_p", "lmr_p", "blrt_p"):
                if key in lrt_by_k[K]:
                    row[key] = lrt_by_k[K][key]
        t.add_row(row)
    t.add_note(f"BIC 최소: {best}계층. 계층 수 결정은 정보기준·검정 결과와 함께 해석 가능성·최소 계층 비율 고려.")
    return t


def _ic_plot(fits: Dict[int, mm.MixtureFit], Ks: List[int], title: str) -> str:
    c = _pal(3)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for key, lab, col in (("aic", "AIC", c[0]), ("bic", "BIC", c[1]), ("sabic", "SABIC", c[2])):
        ax.plot(Ks, [getattr(fits[k], key) for k in Ks], marker="o", color=col, label=lab, linewidth=1.8)
    ax.set_xticks(Ks)
    ax.set_xlabel("계층 수")
    ax.set_ylabel("정보기준")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = fig_to_base64(fig)
    plt.close(fig)
    return out


def _posterior_frame(index: pd.Index, rows: np.ndarray, post: np.ndarray) -> pd.DataFrame:
    K = post.shape[1]
    out = pd.DataFrame(np.nan, index=index, columns=[f"P_{k + 1}" for k in range(K)] + ["MLC"])
    out.iloc[rows, :K] = post
    out.iloc[rows, K] = mm.most_likely(post) + 1
    return out


# ============================================================ measurement result tables (categorical / continuous)
def _cat_prob_table(model, inf: mm.Inference, par_path, names: List[str], levels: List[List[Any]], K: int,
                    p_classes: np.ndarray, name: str = "catProbs", title: str = "범주형 지표의 계층별 응답확률",
                    time_label: Optional[str] = None) -> ResultsTable:
    t = ResultsTable(name=name, title=title)
    if time_label:
        t.add_column("time", "시점", col_type="text")
    t.add_column("item", "지표", col_type="text")
    t.add_column("cat", "범주", col_type="text")
    for k in range(K):
        st = f"{k + 1}계층 ({100 * p_classes[k]:.1f}%)"
        t.add_column(f"p{k + 1}", "확률", col_type="number", super_title=st)
        t.add_column(f"se{k + 1}", "SE", col_type="number", super_title=st)
    for j, nm in enumerate(names):
        C = len(levels[j])
        est, se = mm.delta(lambda th, j=j: par_path(model.unpack(th))["P"][j].ravel(), inf)
        est, se = est.reshape(K, C), se.reshape(K, C)
        for c in range(C):
            row = {"item": nm, "cat": _level_label(levels[j][c])}
            if time_label:
                row["time"] = time_label
            for k in range(K):
                row[f"p{k + 1}"] = float(est[k, c])
                row[f"se{k + 1}"] = float(se[k, c])
            t.add_row(row)
    return t


def _cont_table(model, inf: mm.Inference, par_path, names: List[str], K: int, p_classes: np.ndarray,
                name: str = "contParams", title: str = "연속형 지표의 계층별 평균·분산", time_label: Optional[str] = None,
                tied: bool = True) -> ResultsTable:
    t = ResultsTable(name=name, title=title)
    if time_label:
        t.add_column("time", "시점", col_type="text")
    t.add_column("item", "지표", col_type="text")
    t.add_column("param", "모수", col_type="text")
    for k in range(K):
        st = f"{k + 1}계층 ({100 * p_classes[k]:.1f}%)"
        t.add_column(f"e{k + 1}", "추정치", col_type="number", super_title=st)
        t.add_column(f"se{k + 1}", "SE", col_type="number", super_title=st)
    J = len(names)
    mu, mu_se = mm.delta(lambda th: par_path(model.unpack(th))["mu"].ravel(), inf)
    var, var_se = mm.delta(lambda th: par_path(model.unpack(th))["var"].ravel(), inf)
    mu, mu_se, var, var_se = (a.reshape(K, J) for a in (mu, mu_se, var, var_se))
    for j, nm in enumerate(names):
        for lab, e, s in (("평균", mu, mu_se), ("분산", var, var_se)):
            row = {"item": nm, "param": lab}
            if time_label:
                row["time"] = time_label
            for k in range(K):
                row[f"e{k + 1}"] = float(e[k, j])
                row[f"se{k + 1}"] = float(s[k, j])
            t.add_row(row)
    if tied:
        t.add_note("분산은 계층 간 동일 제약(Mplus 기본) — 계층별 값 동일.")
    return t


def _profile_plot(par_meas: Dict[str, Any], cat: List[str], levels, cont: List[str], meas: mm.Measurement,
                  p_classes: np.ndarray, title: str) -> str:
    K = len(p_classes)
    cols = _pal(K)
    panels = (1 if cat else 0) + (1 if cont else 0)
    fig, axes = plt.subplots(1, panels, figsize=(max(5.6, 0.75 * (len(cat) + len(cont)) + 3.6) if panels == 1 else 11.5, 4.3))
    axes = np.atleast_1d(axes)
    ai = 0
    if cat:
        ax = axes[ai]
        ai += 1
        x = np.arange(len(cat))
        for k in range(K):
            vals = [par_meas["P"][j][k, -1] for j in range(len(cat))]
            ax.plot(x, vals, marker="o", color=cols[k], linewidth=1.8, label=f"{k + 1}계층 ({100 * p_classes[k]:.1f}%)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{nm}\n(={_level_label(levels[j][-1])})" for j, nm in enumerate(cat)], rotation=35, ha="right", fontsize=8.5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("응답확률 (최상위 범주)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8.5)
    if cont:
        ax = axes[ai]
        x = np.arange(len(cont))
        sd = np.sqrt(meas.yvar)
        for k in range(K):
            z = (par_meas["mu"][k] - meas.ymean) / sd
            ax.plot(x, z, marker="s", color=cols[k], linewidth=1.8, label=f"{k + 1}계층 ({100 * p_classes[k]:.1f}%)")
        ax.axhline(0, color="#94A3B8", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cont, rotation=35, ha="right", fontsize=8.5)
        ax.set_ylabel("계층 평균 (표준화)")
        ax.grid(alpha=0.3)
        if not cat:
            ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = fig_to_base64(fig)
    plt.close(fig)
    return out


# ============================================================ auxiliary (three-step) tables
def _aux_tables(an: BaseAnalysis, post: np.ndarray, frame: pd.DataFrame, covs: List[str], dcont: List[str],
                dcat: List[str], estimator: str) -> List[str]:
    K = post.shape[1]
    notes = []
    if covs:
        Z = np.column_stack([_numeric(frame[c], c) for c in covs])
        r = mm.r3step(post, Z, estimator)
        t = ResultsTable(name="r3step", title="공변량의 계층 소속 효과 (R3STEP, 3단계)")
        t.add_column("cmp", "비교", col_type="text")
        t.add_column("var", "변수", col_type="text")
        t.add_column("est", "추정치", col_type="number")
        t.add_column("se", "SE", col_type="number")
        t.add_column("z", "z", col_type="number")
        t.add_column("p", "p", col_type="number", fmt="pvalue")
        t.add_column("or", "승산비", col_type="number")
        z, pv = _zp(r["coef"], r["se"])
        for k in range(K - 1):
            for q, nm in enumerate(["절편"] + covs):
                t.add_row({"cmp": f"{k + 1}계층 대 {K}계층", "var": nm, "est": float(r["coef"][k, q]), "se": float(r["se"][k, q]),
                           "z": float(z[k, q]), "p": float(pv[k, q]), "or": float(np.exp(r["coef"][k, q])) if q else np.nan})
        t.add_note(f"다항 로지스틱 회귀, 기준 = {K}계층(Mplus 기본: 마지막 계층). 분류 오류는 최우추정 계층과 사후확률로 보정(Vermunt, 2010). "
                   f"분석 사례 {r['n']}명" + (f", 공변량 결측 {r['n_dropped']}명 제외" if r["n_dropped"] else "") + ".")
        if r["singular"] or not r["converged"]:
            t.add_note("3단계 추정에서 정보행렬 특이 또는 수렴 미달 — 표준오차 해석 주의.")
        t.mark_coefficients(estimate="est", se="se", p="p", stat="z", exp_estimate="or", label=["cmp", "var"])
        an.add_table(t)
    if dcont:
        t = ResultsTable(name="bchDistal", title="연속형 원격 결과의 계층별 평균 (BCH)")
        t.add_column("var", "변수", col_type="text")
        t.add_column("cls", "계층", col_type="text")
        t.add_column("mean", "평균", col_type="number")
        t.add_column("se", "SE", col_type="number")
        tt = ResultsTable(name="bchTests", title="계층 간 평균 동일성 검정 (BCH)")
        tt.add_column("var", "변수", col_type="text")
        tt.add_column("cmp", "비교", col_type="text")
        tt.add_column("chi2", "χ²", col_type="number")
        tt.add_column("df", "df", col_type="integer")
        tt.add_column("p", "p", col_type="number", fmt="pvalue")
        for v in dcont:
            r = mm.bch_continuous(post, _numeric(frame[v], v))
            for k in range(K):
                t.add_row({"var": v, "cls": f"{k + 1}계층", "mean": float(r["mean"][k]), "se": float(r["se"][k])})
            stat, df_, pv = r["tests"]["overall"]
            tt.add_row({"var": v, "cmp": "전체", "chi2": stat, "df": df_, "p": pv})
            for a, b, stat, df_, pv in r["tests"]["pairs"]:
                tt.add_row({"var": v, "cmp": f"{a + 1} 대 {b + 1}계층", "chi2": stat, "df": df_, "p": pv})
        t.add_note("BCH 가중치: 최우추정 계층 분류 오류 행렬의 역행렬(Bolck, Croon & Hagenaars, 2004; Vermunt, 2010; Mplus BCH). "
                   "표준오차는 가중치를 고정한 샌드위치 추정. 결과 변수 결측 사례 제외.")
        an.add_table(t)
        an.add_table(tt)
    if dcat:
        t = ResultsTable(name="dcatDistal", title="범주형 원격 결과의 계층별 확률 (ML 3단계)")
        t.add_column("var", "변수", col_type="text")
        t.add_column("cat", "범주", col_type="text")
        for k in range(K):
            t.add_column(f"p{k + 1}", "확률", col_type="number", super_title=f"{k + 1}계층")
            t.add_column(f"se{k + 1}", "SE", col_type="number", super_title=f"{k + 1}계층")
        tt = ResultsTable(name="dcatTests", title="계층 간 범주 분포 동일성 검정 (ML 3단계)")
        tt.add_column("var", "변수", col_type="text")
        tt.add_column("cmp", "비교", col_type="text")
        tt.add_column("chi2", "χ²", col_type="number")
        tt.add_column("df", "df", col_type="integer")
        tt.add_column("p", "p", col_type="number", fmt="pvalue")
        for v in dcat:
            codes, levels = _code_categorical(frame[v], v)
            r = mm.ml3step_categorical(post, codes, len(levels), estimator)
            for c, lev in enumerate(levels):
                row = {"var": v, "cat": _level_label(lev)}
                for k in range(K):
                    row[f"p{k + 1}"] = float(r["prob"][k, c])
                    row[f"se{k + 1}"] = float(r["se"][k, c])
                t.add_row(row)
            stat, df_, pv = r["tests"]["overall"]
            tt.add_row({"var": v, "cmp": "전체", "chi2": stat, "df": df_, "p": pv})
            for a, b, stat, df_, pv in r["tests"]["pairs"]:
                tt.add_row({"var": v, "cmp": f"{a + 1} 대 {b + 1}계층", "chi2": stat, "df": df_, "p": pv})
        t.add_note("ML 3단계: 분류 오류를 고정한 혼합모형으로 계층별 범주 확률 추정(Vermunt, 2010). "
                   "Mplus DCAT에 대응하는 기능이며 추정 방식은 다름. 결과 변수 결측 사례 제외.")
        an.add_table(t)
        an.add_table(tt)
    return notes


def _syntax(func: str, args: Dict[str, Any]) -> str:
    parts = [f"{k}={v!r}" for k, v in args.items() if v not in (None, [], "", False)]
    return f"sess.{func}(" + ", ".join(parts) + ")"


# ============================================================ 1. latent class / profile analysis
class LatentClassAnalysis(BaseAnalysis):
    """Latent class (categorical indicators) and latent profile (continuous indicators) analysis."""

    def __init__(self, data: pd.DataFrame, **params):
        super().__init__(name="lca", title="잠재계층·잠재프로파일 분석 (LCA·LPA)", dataset=data, options=dict(params))
        self.posteriors: Optional[pd.DataFrame] = None
        self.run()

    def run(self):
        p, df = self.options, self.dataset
        cat, cont = _indicator_lists(p, df)
        o = _common_options(p)
        variances, covariances = _var_structure(p)
        K = _int(_p(p, "classes", "nClasses", "k"), "계층 수(classes)", 1, 20, 2)
        Ks = _class_range(_p(p, "classRange"))
        if any(k < 1 or k > 20 for k in Ks):
            raise ValueError("classRange: 1~20 범위의 계층 수 지정")
        covs = _as_list(_p(p, "covariates", "r3step"))
        dcont = _as_list(_p(p, "distalContinuous", "bch"))
        dcat = _as_list(_p(p, "distalCategorical", "dcat"))
        _require_columns(df, covs + dcont + dcat, "보조 변수")
        if (covs or dcont or dcat) and K < 2:
            raise ValueError("보조 변수(3단계) 분석은 계층 수 2 이상에서 실행")
        U, levels = (np.zeros((len(df), 0), int), [])
        if cat:
            coded = [_code_categorical(df[c], c) for c in cat]
            U = np.column_stack([c for c, _ in coded])
            levels = [lv for _, lv in coded]
        Y = np.column_stack([_numeric(df[c], c) for c in cont]) if cont else np.zeros((len(df), 0))
        keep = ((U >= 0).any(1) if cat else np.zeros(len(df), bool)) | (np.isfinite(Y).any(1) if cont else np.zeros(len(df), bool))
        rows = np.where(keep)[0]
        dropped = len(df) - rows.size
        if rows.size < 10:
            raise ValueError(f"분석 가능 사례 {rows.size}명 — 10명 이상 필요")
        meas = mm.Measurement(U[rows] if cat else None, [len(l) for l in levels], Y[rows] if cont else None, variances, covariances)
        frame = df.iloc[rows]

        need = set(Ks) | {K}
        if o["tech11"] or o["tech14"]:
            need |= {k - 1 for k in need if k >= 2}
        fits: Dict[int, mm.MixtureFit] = {}
        for k in sorted(need):
            fits[k] = mm.fit_model(mm.LatentClassModel(meas, k), **_fit_options(o))
        fit = fits[K]
        model, par = fit.model, fit.par
        post = fit.post
        ent = mm.relative_entropy(post)
        is_lpa = not cat
        kind = "잠재프로파일분석(LPA)" if is_lpa else ("잠재계층분석(LCA)" if not cont else "잠재계층분석(범주형·연속형 혼합 지표)")

        rows_a = [("분석 유형", kind), ("계층 수", f"{K}"),
                  ("지표", (f"범주형 {len(cat)}개: " + ", ".join(cat) if cat else "") + (" / " if cat and cont else "")
                   + (f"연속형 {len(cont)}개: " + ", ".join(cont) if cont else "")),
                  ("추정 방법", ESTIMATOR_LABELS[o["estimator"]]),
                  ("결측 처리", "FIML: 관측된 지표만 우도에 반영 (MAR 가정)" + (f", 모든 지표 결측 {dropped}명 제외" if dropped else "")),
                  ("국소 독립", "계층 내 지표 간 독립" if covariances == "zero" else "계층 내 연속형 지표 공분산 추정")]
        if cont:
            rows_a.append(("연속형 지표 분산 구조", VAR_LABELS[(variances, covariances)]))
        if cat:
            rows_a.append(("범주형 지표", "계층별 범주 확률 (Mplus 임계값 = 누적확률 로짓)"))
        rows_a += _starts_rows(o) + [("계층 순서", "추정 비율 내림차순 재배열"), ("분석 사례 수", f"{rows.size}명")]
        self.add_table(_assumption_table(rows_a))

        infs: Dict[int, mm.Inference] = {}
        lrt_by_k: Dict[int, Dict[str, Any]] = {}
        if o["tech11"] or o["tech14"]:
            for k in sorted(set(Ks) | {K}):
                if k >= 2:
                    lrt_by_k[k] = _compute_lrt(fits, k, o, infs)
        if Ks:
            self.add_table(_enumeration_table(fits, Ks, lambda f: mm.relative_entropy(f.post), lrt_by_k, o))
        if lrt_by_k:
            self.add_table(_lrt_table([lrt_by_k[k] for k in sorted(lrt_by_k)], o))

        inf = infs.get(K) or mm.inference(model, par, o["estimator"])
        extra = []
        chi = mm.pattern_chi2(meas, par, fit.npar)
        if chi:
            extra = [("Pearson χ² (범주 빈도표)", chi["pearson"]), ("Pearson χ² p", chi["p_pearson"]),
                     ("우도비 χ² (범주 빈도표)", chi["lr"]), ("우도비 χ² p", chi["p_lr"]), ("빈도표 자유도", chi["df"])]
        tf = _fit_table(fit, ent, extra)
        if chi:
            tf.add_note(f"범주 빈도표 χ²: 응답 유형 {chi['cells']:,}칸 기준, 자유도 = 칸 수 − 1 − 자유모수 수. 희소한 표에서는 근사 부정확.")
        if inf.singular:
            tf.add_note("정보행렬이 양정치가 아님 — 표준오차 신뢰 불가. 모형 식별·계층 수 재검토 필요.")
        if inf.n_fixed:
            tf.add_note(f"극단값(±{mm.LOGIT_LIMIT:.0f})에 도달한 로짓·하한에 도달한 분산 {inf.n_fixed}개는 표준오차 계산에서 고정 (Mplus 방식).")
        for note in (_replication_note(fit), _small_class_note(fit)):
            if note:
                tf.add_note(note)
        self.add_table(tf)
        self.add_table(_starts_table(fit))

        K_ = model.K
        p_est, p_se = mm.delta(lambda th: model.unpack(th)["p"], inf)
        if K_ > 1:
            lg, lg_se = mm.delta(lambda th: np.log(model.unpack(th)["p"][:-1] / model.unpack(th)["p"][-1]), inf)
        else:
            lg, lg_se = np.zeros(0), np.zeros(0)
        self.add_table(_class_size_table(post, p_est, p_se, lg, lg_se))
        if K_ > 1:
            self.add_table(_classification_table(post))
        if cat:
            self.add_table(_cat_prob_table(model, inf, lambda pr: pr["meas"], cat, levels, K_, par["p"]))
        if cont:
            self.add_table(_cont_table(model, inf, lambda pr: pr["meas"], cont, K_, par["p"], tied=(variances == "equal")))
        if K_ > 1:
            _aux_tables(self, post, frame, covs, dcont, dcat, o["estimator"])

        if _flag(_p(p, "profilePlot"), True) and K_ > 1:
            self.add_plot("계층별 프로파일", _profile_plot(par["meas"], cat, levels, cont, meas, par["p"],
                                                    f"{kind} 계층별 프로파일 ({K_}계층)"))
        if Ks and len(Ks) > 1 and _flag(_p(p, "icPlot"), True):
            self.add_plot("계층 수별 정보기준", _ic_plot(fits, Ks, "계층 수별 정보기준"))

        self.posteriors = _posterior_frame(df.index, rows, post)
        shares = ", ".join(f"{k + 1}계층 {100 * par['p'][k]:.1f}%" for k in range(K_))
        self.apa_narrative = (f"{kind}: {K_}계층 모형, 로그우도 {fit.ll:.2f}, BIC {fit.bic:.2f}"
                              + (f", 상대 엔트로피 {ent:.3f}" if K_ > 1 else "") + f". 계층 비율: {shares}."
                              + (f" BIC 최소 계층 수: {min(Ks, key=lambda k: fits[k].bic)}." if Ks else ""))
        self.syntax_python = _syntax("lca", {"categorical": cat, "continuous": cont, "classes": K, "classRange": Ks or None,
                                             "variances": variances if variances != "equal" else None,
                                             "covariances": covariances if covariances != "zero" else None,
                                             "covariates": covs, "distalContinuous": dcont, "distalCategorical": dcat,
                                             "tech11": o["tech11"], "tech14": o["tech14"]})
        self.syntax_r = "tidyLPA::estimate_profiles() / poLCA::poLCA()"


# ============================================================ 2. latent transition analysis
class LatentTransitionAnalysis(BaseAnalysis):
    """Latent transition analysis: the same K classes at each occasion, Markov transitions between occasions."""

    def __init__(self, data: pd.DataFrame, **params):
        super().__init__(name="lta", title="잠재전이분석 (LTA)", dataset=data, options=dict(params))
        self.posteriors: Optional[pd.DataFrame] = None
        self.run()

    def _arrays(self, p, df):
        waves = _p(p, "waves")
        cat_decl = _as_list(_p(p, "categorical", "catVars"))
        if waves:
            waves = [_as_list(w) for w in waves]
            J = len(waves[0])
            if len(waves) < 2 or any(len(w) != J for w in waves) or J == 0:
                raise ValueError("waves: 2개 이상 시점, 시점마다 같은 수의 지표 지정")
            allv = [v for w in waves for v in w]
            _require_columns(df, allv, "지표")
            types = []
            for j in range(J):
                flags = {waves[t][j] in cat_decl for t in range(len(waves))}
                if len(flags) > 1:
                    raise ValueError(f"{j + 1}번째 지표의 범주형 지정이 시점마다 다름: {[w[j] for w in waves]}")
                types.append(flags.pop())
            cat_pos = [j for j in range(J) if types[j]]
            cont_pos = [j for j in range(J) if not types[j]]
            T, n = len(waves), len(df)
            levels = []
            U = np.full((T, n, len(cat_pos)), -1, dtype=int)
            for a, j in enumerate(cat_pos):
                stacked = pd.concat([df[waves[t][j]] for t in range(T)], ignore_index=True)
                codes, lev = _code_categorical(stacked, waves[0][j])
                U[:, :, a] = codes.reshape(T, n)
                levels.append(lev)
            Y = np.full((T, n, len(cont_pos)), np.nan)
            for a, j in enumerate(cont_pos):
                for t in range(T):
                    Y[t, :, a] = _numeric(df[waves[t][j]], waves[t][j])
            names_c = [waves[0][j] for j in cat_pos]
            names_y = [waves[0][j] for j in cont_pos]
            labels = [f"{t + 1}시점" for t in range(T)]
            return U, levels, Y, names_c, names_y, labels, None, n
        cat, cont = _indicator_lists(p, df)
        idv = str(_p(p, "idVar", "id", default=""))
        tv = str(_p(p, "timeVar", "time", default=""))
        if not idv or not tv:
            raise ValueError("긴 형식 자료: idVar(개인 식별)·timeVar(시점) 지정, 또는 넓은 형식 waves 지정")
        _require_columns(df, [idv, tv], "식별·시점 변수")
        mask = (df[idv].notna() & df[tv].notna()).to_numpy()
        d = df[mask]
        if d.duplicated([idv, tv]).any():
            raise ValueError(f"'{idv}'×'{tv}' 조합 중복 — 개인·시점당 1행 필요")
        ids = pd.unique(d[idv])
        tvals = list(pd.unique(d[tv]))
        try:
            times = sorted(tvals, key=lambda x: float(x))
        except (TypeError, ValueError):
            times = sorted(tvals, key=lambda x: str(x))
        if len(times) < 2:
            raise ValueError("시점이 2개 이상 필요")
        n, T = len(ids), len(times)
        ipos = {v: i for i, v in enumerate(ids)}
        tpos = {v: i for i, v in enumerate(times)}
        ii = d[idv].map(ipos).to_numpy()
        tt = d[tv].map(tpos).to_numpy()
        levels = []
        U = np.full((T, n, len(cat)), -1, dtype=int)
        for a, c in enumerate(cat):
            codes, lev = _code_categorical(d[c], c)
            U[tt, ii, a] = codes
            levels.append(lev)
        Y = np.full((T, n, len(cont)), np.nan)
        for a, c in enumerate(cont):
            Y[tt, ii, a] = _numeric(d[c], c)
        labels = [f"{tv} {_level_label(x)}" for x in times]
        rowmap = (np.where(mask)[0], ii, tt)
        return U, levels, Y, cat, cont, labels, rowmap, n

    def run(self):
        p, df = self.options, self.dataset
        o = _common_options(p)
        variances, covariances = _var_structure(p)
        K = _int(_p(p, "classes", "nClasses", "k"), "계층 수(classes)", 2, 12, 2)
        invariant = _flag(_p(p, "invariance", "measurementInvariance"), True)
        U, levels, Y, cat, cont, labels, rowmap, n_all = self._arrays(p, df)
        T = len(labels)
        has = ((U >= 0).any(2) if U.shape[2] else np.zeros((T, n_all), bool)) | (np.isfinite(Y).any(2) if Y.shape[2] else np.zeros((T, n_all), bool))
        subj = np.where(has.any(0))[0]
        n = subj.size
        if n < 10:
            raise ValueError(f"분석 가능 사례 {n}명 — 10명 이상 필요")
        Uc, Yc = U[:, subj], Y[:, subj]
        nc = [len(l) for l in levels]
        if invariant:
            meas = mm.Measurement(Uc.reshape(T * n, -1) if cat else None, nc, Yc.reshape(T * n, -1) if cont else None,
                                  variances, covariances)
        else:
            meas = [mm.Measurement(Uc[t] if cat else None, nc, Yc[t] if cont else None, variances, covariances) for t in range(T)]
        model = mm.LatentTransitionModel(meas, K, n, T, invariant)
        fit = mm.fit_model(model, **_fit_options(o))
        par = fit.par
        _, gamma, xi = model.forward_backward(par)
        ent = mm.lta_entropy(gamma, xi)
        inf = mm.inference(model, par, o["estimator"])
        missing_waves = int((~has[:, subj]).sum())

        rows_a = [("분석 유형", "잠재전이분석(LTA)"), ("계층 수", f"시점마다 {K}개"), ("시점", ", ".join(labels)),
                  ("지표", (f"범주형 {len(cat)}개: " + ", ".join(cat) if cat else "") + (" / " if cat and cont else "")
                   + (f"연속형 {len(cont)}개: " + ", ".join(cont) if cont else "")),
                  ("측정 동일성", "시점 간 측정모수 동일 (Mplus 사용자 안내서 예제 8.13 설정)" if invariant else "시점별 측정모수 추정"),
                  ("전이 구조", "1차 마르코프, 시점별 전이확률 (Mplus c2 ON c1)"),
                  ("추정 방법", ESTIMATOR_LABELS[o["estimator"]]),
                  ("결측 처리", "FIML: 관측된 지표·시점만 우도에 반영 (MAR 가정)" + (f", 전체 결측 시점 {missing_waves}건" if missing_waves else ""))]
        if cont:
            rows_a.append(("연속형 지표 분산 구조", VAR_LABELS[(variances, covariances)]))
        rows_a += _starts_rows(o) + [("계층 순서", "시점 평균 비율 내림차순 재배열" if invariant else "1시점 비율 내림차순 재배열"),
                                     ("분석 사례 수", f"{n}명")]
        self.add_table(_assumption_table(rows_a))

        tf = _fit_table(fit, ent)
        tf.add_note("상대 엔트로피는 시점 전체의 계층 유형(결합 사후확률) 기준.")
        for note in (_replication_note(fit), _small_class_note(fit)):
            if note:
                tf.add_note(note)
        if inf.singular:
            tf.add_note("정보행렬이 양정치가 아님 — 표준오차 신뢰 불가.")
        if inf.n_fixed:
            tf.add_note(f"극단값(±{mm.LOGIT_LIMIT:.0f})에 도달한 로짓 {inf.n_fixed}개는 표준오차 계산에서 고정 (Mplus 방식).")
        self.add_table(tf)
        self.add_table(_starts_table(fit))

        marg, marg_se = mm.delta(lambda th: model.marginals(model.unpack(th)).ravel(), inf)
        marg, marg_se = marg.reshape(T, K), marg_se.reshape(T, K)
        tp = ResultsTable(name="prevalence", title="시점별 계층 비율")
        tp.add_column("time", "시점", col_type="text")
        for k in range(K):
            tp.add_column(f"p{k + 1}", "%", col_type="number", fmt="pc", super_title=f"{k + 1}계층")
            tp.add_column(f"se{k + 1}", "SE(%)", col_type="number", super_title=f"{k + 1}계층")
            tp.add_column(f"m{k + 1}", "최우추정 %", col_type="number", fmt="pc", super_title=f"{k + 1}계층")
        for t in range(T):
            obs_t = has[t, subj]
            mlc = np.argmax(gamma[obs_t, t], axis=1)
            row = {"time": labels[t]}
            for k in range(K):
                row[f"p{k + 1}"] = 100 * marg[t, k]
                row[f"se{k + 1}"] = 100 * marg_se[t, k]
                row[f"m{k + 1}"] = 100 * float((mlc == k).mean()) if mlc.size else np.nan
            tp.add_row(row)
        tp.add_note("추정 모형 기준 비율과 시점별 최우추정 계층 비율(해당 시점 응답자 기준).")
        self.add_table(tp)

        tau, tau_se = mm.delta(lambda th: model.unpack(th)["tau"].ravel(), inf)
        tau, tau_se = tau.reshape(T - 1, K, K), tau_se.reshape(T - 1, K, K)
        tt = ResultsTable(name="transitions", title="전이확률 (행: 이전 시점 계층, 열: 다음 시점 계층)")
        tt.add_column("trans", "전이", col_type="text")
        tt.add_column("from", "이전 계층", col_type="text")
        for l in range(K):
            tt.add_column(f"p{l + 1}", "확률", col_type="number", super_title=f"→ {l + 1}계층")
            tt.add_column(f"se{l + 1}", "SE", col_type="number", super_title=f"→ {l + 1}계층")
        for t in range(T - 1):
            for k in range(K):
                row = {"trans": f"{labels[t]} → {labels[t + 1]}", "from": f"{k + 1}계층"}
                for l in range(K):
                    row[f"p{l + 1}"] = float(tau[t, k, l])
                    row[f"se{l + 1}"] = float(tau_se[t, k, l])
                tt.add_row(row)
        tt.add_note("행 합계 1. 대각 원소는 계층 유지 확률.")
        self.add_table(tt)

        p_mean = marg.mean(0)
        if invariant:
            if cat:
                self.add_table(_cat_prob_table(model, inf, lambda pr: pr["meas"], cat, levels, K, p_mean,
                                               title="범주형 지표의 계층별 응답확률 (시점 공통)"))
            if cont:
                self.add_table(_cont_table(model, inf, lambda pr: pr["meas"], cont, K, p_mean,
                                           title="연속형 지표의 계층별 평균·분산 (시점 공통)", tied=(variances == "equal")))
        else:
            for t in range(T):
                if cat:
                    self.add_table(_cat_prob_table(model, inf, lambda pr, t=t: pr["meas"][t], cat, levels, K, marg[t],
                                                   name=f"catProbs_t{t + 1}", title=f"범주형 지표의 계층별 응답확률 ({labels[t]})",
                                                   time_label=labels[t]))
                if cont:
                    self.add_table(_cont_table(model, inf, lambda pr, t=t: pr["meas"][t], cont, K, marg[t],
                                               name=f"contParams_t{t + 1}", title=f"연속형 지표의 계층별 평균·분산 ({labels[t]})",
                                               time_label=labels[t], tied=(variances == "equal")))

        if _flag(_p(p, "transitionPlot"), True):
            self.add_plot("전이확률", self._transition_plot(tau, labels, K))
        if _flag(_p(p, "prevalencePlot"), True):
            self.add_plot("시점별 계층 비율", self._prevalence_plot(marg, labels, K))
        if _flag(_p(p, "profilePlot"), True) and invariant:
            self.add_plot("계층별 프로파일", _profile_plot(par["meas"], cat, levels, cont, meas, p_mean, f"잠재전이분석 계층별 프로파일 ({K}계층)"))

        if rowmap is not None:
            loc, ii, ttt = rowmap
            inv = {s: a for a, s in enumerate(subj)}
            out = pd.DataFrame(np.nan, index=df.index, columns=[f"P_{k + 1}" for k in range(K)] + ["MLC"])
            for r, i, t in zip(loc, ii, ttt):
                if i in inv and has[t, i]:
                    g = gamma[inv[i], t]
                    out.iloc[r, :K] = g
                    out.iloc[r, K] = int(np.argmax(g)) + 1
            self.posteriors = out
        else:
            cols = [f"P_{k + 1}_T{t + 1}" for t in range(T) for k in range(K)] + [f"MLC_T{t + 1}" for t in range(T)]
            out = pd.DataFrame(np.nan, index=df.index, columns=cols)
            for a, i in enumerate(subj):
                for t in range(T):
                    if has[t, i]:
                        for k in range(K):
                            out.iat[i, t * K + k] = gamma[a, t, k]
                        out.iat[i, T * K + t] = int(np.argmax(gamma[a, t])) + 1
            self.posteriors = out

        stay = ", ".join(f"{k + 1}계층 {tau[:, k, k].mean():.2f}" for k in range(K))
        self.apa_narrative = (f"잠재전이분석: 시점 {T}개, 계층 {K}개, 로그우도 {fit.ll:.2f}, BIC {fit.bic:.2f}, 상대 엔트로피 {ent:.3f}. "
                              f"평균 계층 유지 확률: {stay}.")
        if rowmap is not None:
            args = {"categorical": cat, "continuous": cont, "idVar": _p(p, "idVar", "id"), "timeVar": _p(p, "timeVar", "time")}
        else:
            args = {"waves": [_as_list(w) for w in _p(p, "waves")], "categorical": _as_list(_p(p, "categorical", "catVars"))}
        args.update({"classes": K, "invariance": None if invariant else False})
        self.syntax_python = _syntax("lta", args)
        self.syntax_r = "LMest::lmest() / depmixS4::depmix()"

    @staticmethod
    def _transition_plot(tau, labels, K):
        Tm1 = tau.shape[0]
        fig, axes = plt.subplots(1, Tm1, figsize=(3.9 * Tm1 + 0.6, 3.7))
        axes = np.atleast_1d(axes)
        for t in range(Tm1):
            ax = axes[t]
            im = ax.imshow(tau[t], vmin=0, vmax=1, cmap="Blues")
            for a in range(K):
                for b in range(K):
                    ax.text(b, a, f"{tau[t, a, b]:.2f}", ha="center", va="center", fontsize=9,
                            color="white" if tau[t, a, b] > 0.6 else "#1E293B")
            ax.set_xticks(range(K))
            ax.set_yticks(range(K))
            ax.set_xticklabels([f"{k + 1}" for k in range(K)])
            ax.set_yticklabels([f"{k + 1}" for k in range(K)])
            ax.set_xlabel(f"{labels[t + 1]} 계층")
            ax.set_ylabel(f"{labels[t]} 계층")
            ax.set_title(f"{labels[t]} → {labels[t + 1]}", fontsize=10)
        fig.colorbar(im, ax=axes.tolist(), shrink=0.8)
        fig.suptitle("전이확률", fontsize=12, fontweight="bold")
        out = fig_to_base64(fig)
        plt.close(fig)
        return out

    @staticmethod
    def _prevalence_plot(marg, labels, K):
        cols = _pal(K)
        fig, ax = plt.subplots(figsize=(max(4.8, 1.3 * len(labels) + 2.5), 3.9))
        bottom = np.zeros(len(labels))
        for k in range(K):
            ax.bar(labels, 100 * marg[:, k], bottom=bottom, color=cols[k], label=f"{k + 1}계층", edgecolor="white")
            for i, v in enumerate(marg[:, k]):
                if v > 0.06:
                    ax.text(i, bottom[i] + 50 * v, f"{100 * v:.0f}%", ha="center", va="center", fontsize=8.5, color="white")
            bottom += 100 * marg[:, k]
        ax.set_ylabel("%")
        ax.set_ylim(0, 100)
        ax.set_title("시점별 계층 비율 (추정 모형)")
        ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()
        out = fig_to_base64(fig)
        plt.close(fig)
        return out


# ============================================================ 3. latent class growth analysis
class LatentClassGrowthAnalysis(BaseAnalysis):
    """Latent class growth analysis (group-based trajectories) for a repeated continuous outcome."""

    def __init__(self, data: pd.DataFrame, **params):
        super().__init__(name="lcga", title="잠재계층 성장분석 (LCGA)", dataset=data, options=dict(params))
        self.posteriors: Optional[pd.DataFrame] = None
        self.run()

    def run(self):
        p, df = self.options, self.dataset
        o = _common_options(p)
        K = _int(_p(p, "classes", "nClasses", "k"), "계층 수(classes)", 1, 12, 2)
        Ks = _class_range(_p(p, "classRange"))
        growth = resolve_choice("성장 형태", _p(p, "growth", default="linear"), list(GROWTH_LABELS),
                                {"line": "linear", "quad": "quadratic", "cube": "cubic"}, default="linear")
        resvar = resolve_choice("잔차분산 구조", _p(p, "residualVariances", default="time_class_invariant"), list(RESVAR_LABELS),
                                {"default": "time_class_invariant", "time_specific": "time_class_invariant", "free": "class_specific",
                                 "class_varying": "class_specific", "homogeneous": "equal"}, default="time_class_invariant")
        covs = _as_list(_p(p, "covariates", "r3step"))
        wide = _as_list(_p(p, "vars"))
        if wide:
            _require_columns(df, wide + covs, "변수")
            Y = np.column_stack([_numeric(df[v], v) for v in wide])
            labels = list(wide)
            default_ts = np.arange(len(wide), dtype=float)
            subj_rows = np.arange(len(df))
            cov_frame = df
            rowmap = None
        else:
            dep = str(_p(p, "dep", default=""))
            idv = str(_p(p, "idVar", "id", default=""))
            tv = str(_p(p, "timeVar", "time", default=""))
            if not (dep and idv and tv):
                raise ValueError("긴 형식 자료: dep(반복 측정 결과)·idVar·timeVar 지정, 또는 넓은 형식 vars 지정")
            _require_columns(df, [dep, idv, tv] + covs, "변수")
            mask = (df[idv].notna() & df[tv].notna()).to_numpy()
            d = df[mask]
            if d.duplicated([idv, tv]).any():
                raise ValueError(f"'{idv}'×'{tv}' 조합 중복 — 개인·시점당 1행 필요")
            tnum = _numeric(d[tv], tv)
            times = np.sort(np.unique(tnum))
            if times.size < 3:
                raise ValueError("시점이 3개 이상 필요")
            ids = pd.unique(d[idv])
            ipos = {v: i for i, v in enumerate(ids)}
            tpos = {v: i for i, v in enumerate(times)}
            ii = d[idv].map(ipos).to_numpy()
            tt = np.array([tpos[x] for x in tnum])
            Y = np.full((len(ids), times.size), np.nan)
            Y[ii, tt] = _numeric(d[dep], dep)
            labels = [_level_label(x) for x in times]
            default_ts = times - times[0]
            rowmap = (np.where(mask)[0], ii)
            if covs:
                cf = {}
                for c in covs:
                    g = d.groupby(idv, sort=False)[c].agg(lambda s: s.dropna().nunique())
                    if (g > 1).any():
                        raise ValueError(f"공변량 '{c}': 개인 내 값이 시점마다 다름 — 시간불변 공변량 필요")
                    cf[c] = d.groupby(idv, sort=False)[c].agg(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan).reindex(ids).to_numpy()
                cov_frame = pd.DataFrame(cf)
            else:
                cov_frame = pd.DataFrame(index=range(len(ids)))
            subj_rows = np.arange(len(ids))
        ts = _p(p, "timeScores")
        if ts is not None:
            ts = np.array([float(x) for x in (_as_list(ts) if isinstance(ts, str) else ts)])
            if ts.size != Y.shape[1]:
                raise ValueError(f"timeScores: 시점 수({Y.shape[1]})와 같은 개수 지정")
        else:
            ts = default_ts
        q = mm.GROWTH_ORDERS[growth]
        if Y.shape[1] < q:
            raise ValueError(f"{GROWTH_LABELS[growth]} 성장모형은 시점 {q}개 이상 필요")
        keep = np.isfinite(Y).any(1)
        rows = np.where(keep)[0]
        if rows.size < 10:
            raise ValueError(f"분석 가능 사례 {rows.size}명 — 10명 이상 필요")
        gm = mm.GrowthMeasurement(Y[rows], ts, q, resvar)
        need = set(Ks) | {K}
        if o["tech11"] or o["tech14"]:
            need |= {k - 1 for k in need if k >= 2}
        fits = {k: mm.fit_model(mm.LatentClassModel(gm, k), **_fit_options(o)) for k in sorted(need)}
        fit = fits[K]
        model, par, post = fit.model, fit.par, fit.post
        ent = mm.relative_entropy(post)
        obs_rate = float(np.isfinite(Y[rows]).mean())

        rows_a = [("분석 유형", "잠재계층 성장분석(LCGA)"), ("계층 수", f"{K}"),
                  ("반복 측정", ", ".join(labels) + f" (시점 {len(labels)}개)"),
                  ("시간 점수", ", ".join(_level_label(x) for x in ts)),
                  ("성장 형태", f"{GROWTH_LABELS[growth]} — 성장요인 {', '.join(mm.GROWTH_NAMES[:q])}"),
                  ("계층 내 성장요인 분산", "0으로 고정 (LCGA, Mplus i-s@0)"),
                  ("잔차분산", RESVAR_LABELS[resvar]),
                  ("추정 방법", ESTIMATOR_LABELS[o["estimator"]]),
                  ("결측 처리", f"FIML: 관측된 시점만 우도에 반영 (MAR 가정), 관측률 {100 * obs_rate:.1f}%")]
        rows_a += _starts_rows(o) + [("계층 순서", "추정 비율 내림차순 재배열"), ("분석 사례 수", f"{rows.size}명")]
        self.add_table(_assumption_table(rows_a))

        infs: Dict[int, mm.Inference] = {}
        lrt_by_k: Dict[int, Dict[str, Any]] = {}
        if o["tech11"] or o["tech14"]:
            for k in sorted(set(Ks) | {K}):
                if k >= 2:
                    lrt_by_k[k] = _compute_lrt(fits, k, o, infs)
        if Ks:
            self.add_table(_enumeration_table(fits, Ks, lambda f: mm.relative_entropy(f.post), lrt_by_k, o))
        if lrt_by_k:
            self.add_table(_lrt_table([lrt_by_k[k] for k in sorted(lrt_by_k)], o))
        inf = infs.get(K) or mm.inference(model, par, o["estimator"])
        tf = _fit_table(fit, ent)
        if inf.singular:
            tf.add_note("정보행렬이 양정치가 아님 — 표준오차 신뢰 불가.")
        for note in (_replication_note(fit), _small_class_note(fit)):
            if note:
                tf.add_note(note)
        self.add_table(tf)
        self.add_table(_starts_table(fit))
        p_est, p_se = mm.delta(lambda th: model.unpack(th)["p"], inf)
        if K > 1:
            lg, lg_se = mm.delta(lambda th: np.log(model.unpack(th)["p"][:-1] / model.unpack(th)["p"][-1]), inf)
            self.add_table(_class_size_table(post, p_est, p_se, lg, lg_se))
            self.add_table(_classification_table(post))
        else:
            self.add_table(_class_size_table(post, p_est, p_se, np.zeros(0), np.zeros(0)))

        beta, beta_se = mm.delta(lambda th: model.unpack(th)["meas"]["beta"].ravel(), inf)
        beta, beta_se = beta.reshape(K, q), beta_se.reshape(K, q)
        z, pv = _zp(beta, beta_se)
        tg = ResultsTable(name="growth", title="계층별 성장요인 평균")
        tg.add_column("cls", "계층", col_type="text")
        tg.add_column("factor", "성장요인", col_type="text")
        tg.add_column("est", "추정치", col_type="number")
        tg.add_column("se", "SE", col_type="number")
        tg.add_column("z", "z", col_type="number")
        tg.add_column("p", "p", col_type="number", fmt="pvalue")
        for k in range(K):
            for f_ in range(q):
                tg.add_row({"cls": f"{k + 1}계층 ({100 * par['p'][k]:.1f}%)", "factor": FACTOR_LABELS[mm.GROWTH_NAMES[f_]],
                            "est": float(beta[k, f_]), "se": float(beta_se[k, f_]), "z": float(z[k, f_]), "p": float(pv[k, f_])})
        t0 = [labels[i] for i in range(len(ts)) if ts[i] == 0]
        tg.add_note("절편은 시간 점수 0" + (f"({t0[0]})" if t0 else "") + " 시점의 기대값. 기울기 단위는 시간 점수 1. 성장요인 분산은 0으로 고정.")
        tg.mark_coefficients(estimate="est", se="se", p="p", stat="z", label=["cls", "factor"])
        self.add_table(tg)

        var, var_se = mm.delta(lambda th: model.unpack(th)["meas"]["var"].ravel(), inf)
        var, var_se = var.reshape(K, -1), var_se.reshape(K, -1)
        tr = ResultsTable(name="residuals", title="잔차분산")
        tr.add_column("time", "시점", col_type="text")
        if resvar == "class_specific":
            for k in range(K):
                tr.add_column(f"e{k + 1}", "추정치", col_type="number", super_title=f"{k + 1}계층")
                tr.add_column(f"se{k + 1}", "SE", col_type="number", super_title=f"{k + 1}계층")
        else:
            tr.add_column("e", "추정치", col_type="number")
            tr.add_column("se", "SE", col_type="number")
        for t in range(len(labels)):
            row = {"time": labels[t]}
            if resvar == "class_specific":
                for k in range(K):
                    row[f"e{k + 1}"] = float(var[k, t])
                    row[f"se{k + 1}"] = float(var_se[k, t])
            else:
                row["e"], row["se"] = float(var[0, t]), float(var_se[0, t])
            tr.add_row(row)
        tr.add_note(RESVAR_LABELS[resvar] + ".")
        self.add_table(tr)

        if covs and K > 1:
            _aux_tables(self, post, cov_frame.iloc[rows].reset_index(drop=True), covs, [], [], o["estimator"])

        if _flag(_p(p, "trajectoryPlot"), True):
            self.add_plot("계층별 성장 궤적", self._trajectory_plot(model.meas, par, post, Y[rows], ts, labels, K, growth))
        if Ks and len(Ks) > 1 and _flag(_p(p, "icPlot"), True):
            self.add_plot("계층 수별 정보기준", _ic_plot(fits, Ks, "계층 수별 정보기준"))

        K_ = K
        if rowmap is None:
            self.posteriors = _posterior_frame(df.index, rows, post)
        else:
            loc, ii = rowmap
            inv = {s: a for a, s in enumerate(rows)}
            out = pd.DataFrame(np.nan, index=df.index, columns=[f"P_{k + 1}" for k in range(K_)] + ["MLC"])
            mlc = mm.most_likely(post)
            for r, i in zip(loc, ii):
                if i in inv:
                    out.iloc[r, :K_] = post[inv[i]]
                    out.iloc[r, K_] = mlc[inv[i]] + 1
            self.posteriors = out
        desc = "; ".join(f"{k + 1}계층 {100 * par['p'][k]:.1f}% (절편 {beta[k, 0]:.2f}, 기울기 {beta[k, 1]:.2f})" for k in range(K))
        self.apa_narrative = (f"잠재계층 성장분석: {GROWTH_LABELS[growth]} 성장, {K}계층 모형, 로그우도 {fit.ll:.2f}, BIC {fit.bic:.2f}"
                              + (f", 상대 엔트로피 {ent:.3f}" if K > 1 else "") + f". {desc}."
                              + (f" BIC 최소 계층 수: {min(Ks, key=lambda k: fits[k].bic)}." if Ks else ""))
        args = {"vars": wide} if wide else {"dep": _p(p, "dep"), "idVar": _p(p, "idVar", "id"), "timeVar": _p(p, "timeVar", "time")}
        args.update({"classes": K, "classRange": Ks or None, "growth": growth if growth != "linear" else None,
                     "residualVariances": resvar if resvar != "time_class_invariant" else None, "covariates": covs,
                     "tech11": o["tech11"], "tech14": o["tech14"]})
        self.syntax_python = _syntax("lcga", args)
        self.syntax_r = "lcmm::hlme(random = ~ -1) / traj"

    @staticmethod
    def _trajectory_plot(gm: mm.GrowthMeasurement, par, post, Y, ts, labels, K, growth):
        cols = _pal(K)
        fig, ax = plt.subplots(figsize=(6.8, 4.3))
        grid = np.linspace(ts.min(), ts.max(), 60)
        Xg = grid[:, None] ** np.arange(gm.q)[None]
        mlc = mm.most_likely(post)
        for k in range(K):
            ax.plot(grid, Xg @ par["meas"]["beta"][k], color=cols[k], linewidth=2.2,
                    label=f"{k + 1}계층 ({100 * par['p'][k]:.1f}%)")
            sel = mlc == k
            if sel.any():
                m = np.nanmean(Y[sel], axis=0)
                ax.plot(ts, m, linestyle="none", marker="o", color=cols[k], alpha=0.75, markersize=5)
        ax.set_xticks(ts)
        ax.set_xticklabels(labels)
        ax.set_xlabel("시점")
        ax.set_ylabel("추정 평균")
        ax.set_title(f"계층별 성장 궤적 ({GROWTH_LABELS[growth]})")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8.5)
        fig.text(0.01, 0.01, "선: 추정 궤적, 점: 최우추정 계층별 관측 평균", fontsize=8, color="#64748B")
        fig.tight_layout()
        out = fig_to_base64(fig)
        plt.close(fig)
        return out
