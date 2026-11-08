"""
OpenStat Studio — nonlinear panel-data models (Stata xtlogit · xtprobit · xtcloglog · xtpoisson · xtnbreg · xttobit ·
xtintreg · xtologit · xtoprobit · clogit).
License: MIT

One analysis id ``panelNonlinear`` with ``family`` (logit | probit | cloglog | poisson | nbreg | tobit | intreg |
ologit | oprobit) and ``model`` (fe | re | pa).  The name typed (xtlogit, xtpoisson, clogit, ...) presets the family
(and fe for clogit), as countReg does.

Computation (estimator -> library call or published method -> Stata command)

    FE logit (conditional)     statsmodels ConditionalLogit (likelihood, score); groups with all-0 / all-1 outcomes
                               dropped as Stata does                                         xtlogit, fe / clogit
    FE Poisson (conditional)   statsmodels ConditionalPoisson (likelihood, score) + the multinomial constant
                               lnG(sum y + 1) - sum lnG(y + 1) of [XT] xtpoisson Methods and formulas; groups kept by
                               Stata's rule (single-observation and all-zero groups dropped) xtpoisson, fe
    FE negative binomial       Hausman, Hall & Griliches (1984, eq. 3.5 conditional form); [XT] xtnbreg Methods and
                               formulas (constant kept); scipy.special gammaln/digamma          xtnbreg, fe
    RE Poisson, gamma effects  Hausman, Hall & Griliches (1984); [XT] xtpoisson Methods and formulas (/lnalpha)
                                                                                                xtpoisson, re
    RE negative binomial       Hausman, Hall & Griliches (1984, eq. 3.5); [XT] xtnbreg Methods and formulas
                               (beta-distributed dispersion, /ln_r /ln_s)                       xtnbreg, re
    RE with a normal effect    Butler & Moffitt (1982) Gauss–Hermite quadrature; mean–variance adaptive quadrature
    (logit, probit, cloglog,   of Naylor & Smith (1982) as in [XT] xtlogit Methods and formulas (intmethod(mvaghermite),
    Poisson normal, tobit,     default 12 points) or nonadaptive Gauss–Hermite (intmethod(ghermite)); nodes and
    intreg, ologit, oprobit)   weights numpy.polynomial.hermite.hermgauss            xtlogit/xtprobit/xtcloglog, re;
                                                                                      xtpoisson, re normal; xttobit;
                                                                                      xtintreg; xtologit; xtoprobit
    PA (GEE)                   statsmodels GEE: Binomial (logit/probit/cloglog link), Poisson, NegativeBinomial(alpha=1)
                               families; scale 1; Exchangeable / Independence; corr(ar 1) and corr(unstructured) with
                               the moment estimators of [XT] xtgee Methods and formulas (Pearson residuals)
                                                                                      xtlogit/xtprobit/xtcloglog/
                                                                                      xtpoisson/xtnbreg, pa
    Pooled comparison models   statsmodels Logit / Probit / GLM(Binomial, CLogLog) / Poisson / NegativeBinomialP(p=1) /
                               OrderedModel; censored-normal (tobit/intreg) likelihood with scipy.special.log_ndtr
                                                                                      LR test vs pooled (chibar2(01))
    Maximization               Newton–Raphson with step halving on analytic scores (Stata ml default technique(nr));
                               observed information (OIM) = statsmodels.tools.numdiff central-difference Jacobian of
                               the analytic score
    vce(robust) / cluster      sandwich H^-1 (sum_c s_c s_c') H^-1 * M/(M-1) over panels or clusters ([P] robust);
                               xtpoisson fe / re gamma without the factor (Stata _robust ..., minus(0));
                               GEE: statsmodels robust covariance * M/(M-1)
    Hausman FE vs RE           (b_FE - b_RE)' (V_FE - V_RE)^-1 (b_FE - b_RE) on the common slopes, generalized inverse,
                               df = rank (Stata hausman fe re)
    sigma_u at the boundary 0  one-sided score check at the pooled estimates; the pooled fit is reported with
                               chibar2(01) = 0 (Stata xtlogit sets chibar2 = 0 once /lnsig2u <= -14)
    Average marginal effects   analytic derivatives of the mean with delta-method SEs (Stata margins, dydx(*)):
                               PA -> population-averaged mean; RE binary -> predict(pr), the probability integrated
                               over u with Gauss–Hermite points = intpoints (Stata 19 margins default after
                               xtlogit/xtprobit/xtcloglog, re)

Known differences from Stata 19.5 (see tests): xtpoisson, re vce(robust) — Stata sums the /lnalpha panel score once per
observation (T_i times) and puts _cons and /lnalpha in its Wald chi2, here the panel score enters once; xtintreg with
intmethod(mvaghermite) — Stata's default tolerances stop 1.7e-5 below the maximum and its /sigma_e SE is a third of
the value from intmethod(ghermite) and from a brute-force Hessian.

Verified against Stata 19.5 MP on sample_data/Panel_Nonlinear_Care.csv (do-files and logs in
Stata do-files kept outside the repository); reference values are hard-coded in tests/test_panel_nonlinear.py.
"""
from __future__ import annotations

import math
import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import special, stats

from .base import BaseAnalysis, ResultsTable, resolve_choice
from .panel_module import independent_columns, time_positions, xtset_lines

SQ2 = math.sqrt(2.0)
LOG2PI = math.log(2.0 * math.pi)

# --------------------------------------------------------------------------------------------- families and options
FAMILIES: Dict[str, Dict[str, Any]] = {
    "logit": {"cmd": "xtlogit", "ko": "로지스틱", "models": ("fe", "re", "pa"), "kind": "binary", "eform": "or",
              "rho_c": math.pi ** 2 / 3.0},
    "probit": {"cmd": "xtprobit", "ko": "프로빗", "models": ("re", "pa"), "kind": "binary", "eform": None, "rho_c": 1.0},
    "cloglog": {"cmd": "xtcloglog", "ko": "보완 로그-로그", "models": ("re", "pa"), "kind": "binary", "eform": "eform",
                "rho_c": math.pi ** 2 / 6.0},
    "poisson": {"cmd": "xtpoisson", "ko": "포아송", "models": ("fe", "re", "pa"), "kind": "count", "eform": "irr"},
    "nbreg": {"cmd": "xtnbreg", "ko": "음이항", "models": ("fe", "re", "pa"), "kind": "count", "eform": "irr"},
    "tobit": {"cmd": "xttobit", "ko": "토빗", "models": ("re",), "kind": "censored", "eform": None},
    "intreg": {"cmd": "xtintreg", "ko": "구간 회귀", "models": ("re",), "kind": "interval", "eform": None},
    "ologit": {"cmd": "xtologit", "ko": "순서형 로짓", "models": ("re",), "kind": "ordinal", "eform": "or"},
    "oprobit": {"cmd": "xtoprobit", "ko": "순서형 프로빗", "models": ("re",), "kind": "ordinal", "eform": None},
}
FAMILY_ALIASES = {"logistic": "logit", "clogit": "logit", "cll": "cloglog", "pois": "poisson", "negbin": "nbreg",
                  "nb": "nbreg", "nb2": "nbreg", "censored": "tobit", "interval": "intreg", "ordinal": "ologit",
                  "ordered_logit": "ologit", "ordered_probit": "oprobit"}
MODELS = ("fe", "re", "pa")
MODEL_ALIASES = {"fixed": "fe", "conditional": "fe", "random": "re", "gee": "pa", "population": "pa",
                 "population_averaged": "pa"}
MODEL_LABEL = {"fe": "조건부 고정효과", "re": "확률효과", "pa": "모집단 평균 (GEE)"}
ALIAS_PRESETS = {"xtlogit": ("logit", None), "xtprobit": ("probit", None), "xtcloglog": ("cloglog", None),
                 "xtpoisson": ("poisson", None), "xtnbreg": ("nbreg", None), "xttobit": ("tobit", None),
                 "xtintreg": ("intreg", None), "xtologit": ("ologit", None), "xtoprobit": ("oprobit", None),
                 "clogit": ("logit", "fe")}
INT_METHODS = ("mvaghermite", "ghermite")
INT_ALIASES = {"mvagh": "mvaghermite", "mvaghermit": "mvaghermite", "adaptive": "mvaghermite", "gh": "ghermite",
               "ghermit": "ghermite", "nonadaptive": "ghermite"}
VCE_TYPES = ("oim", "robust", "cluster")
VCE_ALIASES = {"conventional": "oim", "default": "oim", "unadjusted": "oim", "clustered": "cluster", "cl": "cluster",
               "sandwich": "robust"}
PA_CORRS = ("exchangeable", "independent", "ar1", "unstructured")
PA_ALIASES = {"exch": "exchangeable", "exc": "exchangeable", "ind": "independent", "indep": "independent",
              "independence": "independent", "ar": "ar1", "ar 1": "ar1", "ar(1)": "ar1", "unstr": "unstructured",
              "uns": "unstructured"}
RE_DISTS = ("gamma", "normal")

_DEFAULTS = {"model": "re", "reDist": "gamma", "intMethod": "mvaghermite", "intPoints": 12, "covType": "oim",
             "clusterVar": None, "paCorr": "exchangeable", "force": False, "exposure": None, "offset": None,
             "ll": None, "ul": None, "eform": False, "constant": True, "hausman": True, "margins": True,
             "ciLevel": 95.0, "depUpper": None, "time": None}


def _opt(p: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in p and p[k] is not None and not (isinstance(p[k], str) and p[k].strip() == "") and p[k] != []:
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
        return val[0] if len(val) else None
    if isinstance(val, str) and not val.strip():
        return None
    return val


def _limit(val: Any, label: str) -> Any:
    """ll()/ul(): a number, a variable name, True (Stata ll without argument: min/max of depvar) or None."""
    if val is None or val is False or (isinstance(val, str) and not val.strip()):
        return None
    if val is True:
        return True
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    s = str(val).strip()
    try:
        return float(s)
    except ValueError:
        return s


def normalize_options(params: Dict[str, Any], preset: Optional[Tuple[str, Optional[str]]] = None) -> Dict[str, Any]:
    """Parameters (camelCase / snake_case / Stata names) -> validated canonical options.

    Options that belong to another family or estimator (for example intPoints with model='fe', exposure with a
    binary outcome, eform with xttobit — values the web form keeps for hidden fields) are left out and listed in
    options['ignored'], which the summary table shows; the Stata command input (transpile) rejects them as Stata does."""
    p = params or {}
    ignored: List[str] = []
    fam_raw = _opt(p, "family", "dist", "distribution")
    pre_fam, pre_model = preset or (None, None)
    if _flag(p, "clogit", default=False) and preset is None:
        pre_fam, pre_model = "logit", "fe"
        preset = (pre_fam, pre_model)
    if fam_raw is None:
        fam = pre_fam or "logit"
    else:
        fam = resolve_choice("모형 계열(family)", fam_raw, tuple(FAMILIES), FAMILY_ALIASES)
        if pre_fam and fam != pre_fam:
            raise ValueError(f"분석 이름과 모형 계열 불일치: 분석 이름은 '{pre_fam}', family='{fam_raw}' — 둘 중 하나만 지정")
    spec = FAMILIES[fam]
    mod_raw = _opt(p, "model", "modelType", "model_type", "estimator")
    for k in ("fe", "re", "pa"):
        if _flag(p, k, default=False):
            mod_raw = k if mod_raw is None else mod_raw
    model = resolve_choice("추정 방법(model)", mod_raw, MODELS, MODEL_ALIASES, default=pre_model or "re")
    if pre_model and model != pre_model:
        raise ValueError(f"clogit은 조건부 고정효과(fe) 모형 — model='{model}' 지정 불가")
    if model not in spec["models"]:
        raise ValueError(f"{spec['cmd']}: {model} 모형 미지원 (Stata {spec['cmd']} 가능: {', '.join(spec['models'])})")
    re_dist = resolve_choice("확률효과 분포(reDist)", _opt(p, "reDist", "re_dist", "redist"), RE_DISTS,
                             {"gaussian": "normal", "norm": "normal"},
                             default="normal" if _flag(p, "normal", default=False) else "gamma")
    # reDist matters for xtpoisson, re only (the other random-effects families use a normal effect)
    int_method = resolve_choice("적분 방법(intMethod)", _opt(p, "intMethod", "int_method", "intmethod"), INT_METHODS,
                                INT_ALIASES, default="mvaghermite")
    try:
        int_points = int(_opt(p, "intPoints", "int_points", "intpoints", default=12))
    except (TypeError, ValueError):
        raise ValueError("적분점 수(intPoints): 정수 필요")
    if not 2 <= int_points <= 195:
        raise ValueError(f"적분점 수(intPoints): 2~195 (입력 {int_points})")
    quad = model == "re" and (fam in ("logit", "probit", "cloglog", "tobit", "intreg", "ologit", "oprobit")
                              or (fam == "poisson" and re_dist == "normal"))
    if not quad and (int_method != "mvaghermite" or int_points != 12):
        ignored.append("intMethod·intPoints (정규분포 확률효과 구적법 모형에만 해당)")
        int_method, int_points = "mvaghermite", 12
    vce = resolve_choice("표준오차(covType)", _opt(p, "covType", "cov_type", "vce", default="oim"), VCE_TYPES, VCE_ALIASES)
    cluster = _as_single(_opt(p, "clusterVar", "cluster_var", "cluster"))
    cluster_default = False
    if vce == "cluster" and not cluster:
        ent = _as_single(_opt(p, "entity", "idVar", "panelVar", "group"))
        if not ent:
            raise ValueError("군집 표준오차(cluster): 군집 변수(clusterVar) 지정 필요")
        cluster, cluster_default = ent, True     # as the linear panel form: no cluster variable = the panel variable
    alias = (preset is not None and pre_model == "fe")        # clogit
    allowed = _allowed_vce(fam, model, re_dist, alias)
    if vce not in allowed:
        raise ValueError(f"{_cmd_label(fam, model, re_dist, alias)}: vce({vce}) 미지원 — Stata 허용: "
                         + ", ".join(allowed) + " (bootstrap·jackknife 제외)")
    corr = resolve_choice("작업 상관 구조(paCorr)", _opt(p, "paCorr", "pa_corr", "corr", default="exchangeable"), PA_CORRS,
                          PA_ALIASES)
    exposure = _as_single(_opt(p, "exposure"))
    offset = _as_single(_opt(p, "offset"))
    if exposure and offset:
        raise ValueError("exposure와 offset은 함께 지정할 수 없음 (Stata 규칙과 동일)")
    if exposure and spec["kind"] != "count":
        ignored.append("exposure (xtpoisson·xtnbreg에만 해당)")
        exposure = None
    if offset and fam in ("ologit", "oprobit", "intreg"):
        ignored.append(f"offset ({spec['cmd']} 미지원)")
        offset = None
    ll, ul = _limit(_opt(p, "ll"), "ll"), _limit(_opt(p, "ul"), "ul")
    if fam != "tobit":
        ll = ul = None          # censoring limits of xttobit (the web form keeps ll = 0 for other families)
    if fam == "tobit" and ll is None and ul is None:
        raise ValueError("xttobit: ll() 또는 ul() 중 하나 이상 지정 필요 (Stata: 검열 한계 없으면 xtreg 사용)")
    dep_upper = _as_single(_opt(p, "depUpper", "dep_upper", "depvarUpper", "upper", "dep2"))
    dep_raw = _opt(p, "dep", "depvar", "depLower", "dep_lower")
    if isinstance(dep_raw, (list, tuple)) and len(dep_raw) == 2 and fam == "intreg" and not dep_upper:
        dep_raw, dep_upper = dep_raw[0], dep_raw[1]
    dep = _as_single(dep_raw)
    if fam == "intreg" and not dep_upper:
        raise ValueError("xtintreg: 하한 변수(dep)와 상한 변수(depUpper) 모두 지정 필요")
    if fam != "intreg" and dep_upper:
        ignored.append("depUpper (xtintreg에만 해당)")
        dep_upper = None
    eform = _flag(p, "eform", "or", "irr", "exp", default=False)
    if eform and not spec["eform"]:
        ignored.append(f"eform ({spec['cmd']}: 지수 변환 계수 없음)")
        eform = False
    force = _flag(p, "force", default=False)
    if model != "pa" and (corr != "exchangeable" or force):
        ignored.append("paCorr·force (모집단 평균 모형에만 해당)")
        corr, force = "exchangeable", False
    level = float(_opt(p, "ciLevel", "ci_level", "level", default=95.0))
    if level < 1:
        level *= 100.0
    if not 10.0 <= level <= 99.99:
        raise ValueError(f"신뢰수준(ciLevel): 10~99.99 사이 값 필요 (입력 {level})")
    constant = _flag(p, "constant", default=not _flag(p, "noconstant", "noConstant", default=False))
    if not constant and (model == "fe" and fam != "nbreg" or spec["kind"] == "ordinal"):
        constant = True        # no constant in these models anyway
    hausman_default = fam in ("logit", "poisson") and model in ("fe", "re")
    time_var = _as_single(_opt(p, "time", "time_col", "timeVar"))
    if model == "pa" and corr in ("ar1", "unstructured") and not time_var:
        raise ValueError(f"pa corr({'ar 1' if corr == 'ar1' else corr}): 시점 변수(time) 필요 "
                         "(Stata: xtset 개체변수 시점변수)")
    return {
        "entity": _as_single(_opt(p, "group", "groupVar", "entity", "entity_col", "idVar", "panelVar") if alias
                             else _opt(p, "entity", "entity_col", "idVar", "panelVar", "group", "groupVar")),
        "time": time_var,
        "dep": dep, "depUpper": dep_upper,
        "covs": _as_list(_opt(p, "covs", "indeps", "indepvars", default=[])),
        "family": fam, "model": model, "reDist": re_dist if (fam == "poisson" and model == "re") else "gamma",
        "intMethod": int_method, "intPoints": int_points, "covType": vce,
        "clusterVar": cluster if vce == "cluster" else None, "paCorr": corr, "force": force,
        "exposure": exposure, "offset": offset, "ll": ll, "ul": ul, "eform": eform, "constant": constant,
        "hausman": _flag(p, "hausman", default=hausman_default) and hausman_default,
        "margins": _flag(p, "margins", "ame", default=True), "ciLevel": level, "clogit": bool(alias),
        "ignored": ignored,
        "clusterDefault": cluster_default,
    }


def _allowed_vce(fam: str, model: str, re_dist: str, clogit: bool) -> Tuple[str, ...]:
    if model == "pa":
        return ("oim", "robust")
    if model == "fe":
        if fam == "logit":
            return ("oim", "robust", "cluster") if clogit else ("oim",)
        if fam == "poisson":
            return ("oim", "robust")
        return ("oim",)
    if fam in ("nbreg", "tobit", "intreg"):
        return ("oim",)
    return ("oim", "robust", "cluster")


def _cmd_label(fam: str, model: str, re_dist: str = "gamma", clogit: bool = False) -> str:
    if clogit:
        return "clogit"
    cmd = FAMILIES[fam]["cmd"]
    if fam in ("tobit", "intreg", "ologit", "oprobit"):
        return cmd
    extra = " normal" if (fam == "poisson" and model == "re" and re_dist == "normal") else ""
    return f"{cmd}, {model}{extra}"


# --------------------------------------------------------------------------------------------- data
class NLData:
    """Listwise-complete panel arrays sorted by entity and time (Stata xtset order)."""

    def __init__(self, df: pd.DataFrame, o: Dict[str, Any]):
        fam, kind = o["family"], FAMILIES[o["family"]]["kind"]
        label = FAMILIES[fam]["cmd"]
        ent, tm, dep, covs = o["entity"], o["time"], o["dep"], list(o["covs"])
        if not ent or not dep:
            miss = [n for n, v in (("개체 변수(entity)", ent), ("종속변수(dep)", dep)) if not v]
            raise ValueError(f"{label}: 지정 필요 — {', '.join(miss)} (Stata 명령 입력 시 xtset 개체변수 [시점변수] 먼저 실행). "
                             f"사용 가능한 변수: {list(df.columns)}")
        extra = [o.get("depUpper"), o.get("exposure"), o.get("offset"), o.get("clusterVar")]
        extra += [v for v in (o.get("ll"), o.get("ul")) if isinstance(v, str)]
        cols = [ent] + ([tm] if tm else []) + [dep] + covs + [c for c in extra if c]
        absent = [c for c in dict.fromkeys(cols) if c not in df.columns]
        if absent:
            raise ValueError(f"{label}: 데이터에 없는 변수 — {', '.join(absent)}. 사용 가능한 변수: {list(df.columns)}")
        if dep in covs or (o.get("depUpper") and o["depUpper"] in covs):
            raise ValueError(f"{label}: 종속변수가 설명변수에도 포함됨")
        if ent in covs or (tm and tm in covs):
            raise ValueError(f"{label}: 개체·시점 변수는 설명변수로 지정할 수 없음")
        if len(set(covs)) != len(covs):
            raise ValueError(f"{label}: 설명변수 중복 지정")
        sub = df[list(dict.fromkeys(cols))].copy()
        numeric = [dep] + covs + [c for c in extra if c and c != o.get("clusterVar")]
        numeric += [v for v in (o.get("ll"), o.get("ul")) if isinstance(v, str)]
        for c in dict.fromkeys(numeric):
            if not pd.api.types.is_numeric_dtype(sub[c]):
                conv = pd.to_numeric(sub[c], errors="coerce")
                if conv.notna().sum() < sub[c].notna().sum():
                    raise ValueError(f"{label}: '{c}'는 숫자형 변수 필요 (문자 값 포함)")
                sub[c] = conv
        n0 = len(sub)
        if kind == "interval":
            up = o["depUpper"]
            keep_cols = [c for c in sub.columns if c not in (dep, up)]
            sub = sub[sub[keep_cols].notna().all(axis=1) & (sub[dep].notna() | sub[up].notna())]
        else:
            sub = sub.dropna()
        if sub.empty:
            raise ValueError(f"{label}: 결측 제외 후 관측치 없음")
        self.n_dropped = n0 - len(sub)
        if tm:
            tpos, self.time_note = time_positions(sub[tm])
            sub["_tpos"] = tpos
            if sub.duplicated([ent, "_tpos"]).any():
                dup = sub[sub.duplicated([ent, "_tpos"], keep=False)].head(3)
                raise ValueError(f"{label}: 같은 개체·시점 관측치 중복 (Stata: repeated time values within panel) — 예: "
                                 f"{dup[[ent, tm]].to_dict('records')}")
            sub = sub.sort_values([ent, "_tpos"], kind="mergesort").reset_index(drop=True)
        else:
            self.time_note = ""
            sub = sub.sort_values([ent], kind="mergesort").reset_index(drop=True)
            sub["_tpos"] = sub.groupby(ent).cumcount().to_numpy(np.int64)
        self.df = sub
        self.entity, self.time, self.dep, self.covs = ent, tm, dep, covs
        self.g, self.entity_labels = pd.factorize(sub[ent], sort=True)
        self.n = int(self.g.max()) + 1
        self.Ti = np.bincount(self.g, minlength=self.n)
        self.tpos = sub["_tpos"].to_numpy(np.int64)
        self.N = len(sub)
        self.X = sub[covs].to_numpy(float) if covs else np.zeros((self.N, 0))
        self.off = np.zeros(self.N)
        self.off_label = ""
        if o.get("exposure"):
            e = sub[o["exposure"]].to_numpy(float)
            if (e <= 0).any():
                raise ValueError(f"{label}: exposure 변수 '{o['exposure']}'에 0 이하 값 — 양수 필요")
            self.off = np.log(e)
            self.off_label = f"ln({o['exposure']})"
        elif o.get("offset"):
            self.off = sub[o["offset"]].to_numpy(float)
            self.off_label = o["offset"]
        self.cluster = None
        if o.get("clusterVar"):
            self.cluster = pd.factorize(sub[o["clusterVar"]])[0]
            chk = pd.DataFrame({"e": self.g, "c": self.cluster}).groupby("e")["c"].nunique()
            if (chk > 1).any():
                raise ValueError(f"{label}: 군집 변수 '{o['clusterVar']}' 안에 패널이 포함되지 않음 "
                                 "(Stata: panels are not nested within clusters)")
        y = sub[dep].to_numpy(float)
        self.y = y
        self.levels: List[Any] = []
        if kind == "binary":
            self.y = (y != 0).astype(float)
            if self.y.min() == self.y.max():
                raise ValueError(f"{label}: 종속변수 '{dep}'가 한 값만 가짐 (Stata: outcome does not vary)")
        elif kind == "count":
            if (y < 0).any():
                raise ValueError(f"{label}: 종속변수 '{dep}'에 음수 — 빈도(0 이상) 필요")
        elif kind == "ordinal":
            lv = np.unique(y)
            if len(lv) < 2:
                raise ValueError(f"{label}: 종속변수 '{dep}'의 범주가 2개 미만")
            if len(lv) > 50:
                raise ValueError(f"{label}: 종속변수 '{dep}'의 범주가 너무 많음 ({len(lv)}개) — 순서형 범주 변수 필요")
            self.levels = [float(v) for v in lv]
            self.y = np.searchsorted(lv, y).astype(np.int64)
        elif kind == "interval":
            lo = sub[dep].to_numpy(float)
            hi = sub[o["depUpper"]].to_numpy(float)
            bad = np.isfinite(lo) & np.isfinite(hi) & (lo > hi)
            if bad.any():
                raise ValueError(f"{label}: 하한 > 상한인 관측치 {int(bad.sum())}개 ('{dep}' > '{o['depUpper']}')")
            self.lo = np.where(np.isfinite(lo), lo, -np.inf)
            self.hi = np.where(np.isfinite(hi), hi, np.inf)
        if kind == "censored":
            def lim(v, default):
                if v is None:
                    return np.full(self.N, default)
                if v is True:
                    return np.full(self.N, y.min() if default < 0 else y.max())
                if isinstance(v, float):
                    return np.full(self.N, v)
                return sub[v].to_numpy(float)
            llv, ulv = lim(o.get("ll"), -np.inf), lim(o.get("ul"), np.inf)
            # Stata xttobit: y <= ll left-censored at ll, y >= ul right-censored at ul
            self.lo = np.where(y >= ulv, ulv, np.where(y <= llv, -np.inf, y))
            self.hi = np.where(y <= llv, llv, np.where(y >= ulv, np.inf, y))
            self.n_left = int(np.sum(y <= llv))
            self.n_right = int(np.sum(y >= ulv))
            self.ll_val, self.ul_val = o.get("ll"), o.get("ul")
        if kind == "interval":
            pt = self.lo == self.hi
            self.n_point = int(pt.sum())
            self.n_left = int(np.sum(~np.isfinite(self.lo)))
            self.n_right = int(np.sum(~np.isfinite(self.hi)))
            self.n_interval = self.N - self.n_point - self.n_left - self.n_right

    def subset(self, rows: np.ndarray) -> "NLData":
        """Copy restricted to rows (kept in order) with group codes renumbered."""
        new = object.__new__(NLData)
        new.__dict__.update(self.__dict__)
        new.df = self.df.iloc[rows].reset_index(drop=True)
        g_old = self.g[rows]
        new.g, uniq = pd.factorize(g_old, sort=True)
        new.entity_labels = self.entity_labels[uniq]
        new.n = len(uniq)
        new.Ti = np.bincount(new.g, minlength=new.n)
        new.tpos = self.tpos[rows]
        new.N = len(rows)
        new.X = self.X[rows]
        new.off = self.off[rows]
        new.y = self.y[rows]
        new.cluster = None if self.cluster is None else self.cluster[rows]
        for a in ("lo", "hi"):
            if hasattr(self, a):
                setattr(new, a, getattr(self, a)[rows])
        return new


def _starts(Ti: np.ndarray) -> np.ndarray:
    return np.r_[0, np.cumsum(Ti)[:-1]].astype(np.int64)


def _gsum(a: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.add.reduceat(a, starts, axis=0)


# --------------------------------------------------------------------------------------------- distributions
def _log_diff_ndtr(zh: np.ndarray, zl: np.ndarray) -> np.ndarray:
    """log(Phi(zh) - Phi(zl)) for zh > zl, stable in both tails."""
    out = np.empty(np.broadcast(zh, zl).shape)
    zh, zl = np.broadcast_to(zh, out.shape), np.broadcast_to(zl, out.shape)
    neg = zl > 0       # upper tail: use symmetry Phi(-zl) - Phi(-zh)
    a = np.where(neg, -zl, zh)
    b = np.where(neg, -zh, zl)
    la, lb = special.log_ndtr(a), special.log_ndtr(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[...] = la + np.log1p(-np.exp(np.minimum(lb - la, 0.0)))
    return out


def _phi(z):
    return np.exp(-0.5 * z * z - 0.5 * LOG2PI)


class _Binary:
    """Bernoulli outcome with logit / probit / cloglog link: log f(y | eta) and d log f / d eta."""
    k_anc = 0

    def __init__(self, y: np.ndarray, link: str):
        self.y = y
        self.q = (2.0 * y - 1.0)[:, None]
        self.link = link

    def logf(self, eta, anc, grad=True):
        y1 = self.y[:, None] > 0.5
        if self.link == "logit":
            qe = self.q * eta
            lf = special.log_expit(qe)
            d = self.q * special.expit(-qe) if grad else None
        elif self.link == "probit":
            qe = self.q * eta
            lf = special.log_ndtr(qe)
            d = self.q * np.exp(-0.5 * qe * qe - 0.5 * LOG2PI - lf) if grad else None
        else:
            ee = np.exp(np.minimum(eta, 700.0))
            with np.errstate(divide="ignore", over="ignore"):
                l1 = np.log(-np.expm1(-ee))
                d = np.where(y1, ee / np.expm1(np.maximum(ee, 1e-300)), -ee) if grad else None
            lf = np.where(y1, l1, -ee)
        return lf, d, []

    def mean_derivs(self, eta):
        """dP/deta and d2P/deta2 of P(y = 1 | eta) (for marginal effects)."""
        if self.link == "logit":
            p = special.expit(eta)
            d1 = p * (1 - p)
            return d1, d1 * (1 - 2 * p)
        if self.link == "probit":
            d1 = _phi(eta)
            return d1, -eta * d1
        ee = np.exp(np.minimum(eta, 700.0))
        d1 = ee * np.exp(-ee)
        return d1, d1 * (1 - ee)


class _PoissonN:
    """Poisson outcome (log link) for the normal random-effects model."""
    k_anc = 0

    def __init__(self, y: np.ndarray):
        self.y = y[:, None]
        self.c = special.gammaln(y + 1.0)[:, None]

    def logf(self, eta, anc, grad=True):
        mu = np.exp(np.minimum(eta, 700.0))
        lf = self.y * eta - mu - self.c
        return lf, (self.y - mu) if grad else None, []


class _Interval:
    """Censored / interval normal outcome (xttobit, xtintreg): anc = [ln sigma_e]."""
    k_anc = 1

    def __init__(self, lo: np.ndarray, hi: np.ndarray):
        self.lo, self.hi = lo[:, None], hi[:, None]
        self.point = (lo == hi)[:, None]
        self.left = (~np.isfinite(lo) & np.isfinite(hi))[:, None]
        self.right = (np.isfinite(lo) & ~np.isfinite(hi))[:, None]
        self.inter = ~(self.point | self.left | self.right)

    def logf(self, eta, anc, grad=True):
        lns = anc[0]
        s = math.exp(lns)
        with np.errstate(invalid="ignore"):
            zp = (np.where(self.point, self.lo, 0.0) - eta) / s
            zh = (np.where(np.isfinite(self.hi), self.hi, 0.0) - eta) / s
            zl = (np.where(np.isfinite(self.lo), self.lo, 0.0) - eta) / s
        lf_p = -0.5 * zp * zp - 0.5 * LOG2PI - lns
        lf_l = special.log_ndtr(zh)
        lf_r = special.log_ndtr(-zl)
        lf_i = _log_diff_ndtr(zh, np.where(self.inter, zl, zh - 1.0))
        lf = np.where(self.point, lf_p, np.where(self.left, lf_l, np.where(self.right, lf_r, lf_i)))
        if not grad:
            return lf, None, []
        lam_h = np.exp(-0.5 * zh * zh - 0.5 * LOG2PI - lf_l)        # phi(zh)/Phi(zh)
        lam_l = np.exp(-0.5 * zl * zl - 0.5 * LOG2PI - lf_r)        # phi(zl)/Phi(-zl)
        fh, fl = _phi(zh), _phi(zl)
        with np.errstate(over="ignore", invalid="ignore"):
            inv_d = np.exp(-lf_i)
        de = np.where(self.point, zp / s,
                      np.where(self.left, -lam_h / s,
                               np.where(self.right, lam_l / s, (fl - fh) * inv_d / s)))
        ds = np.where(self.point, zp * zp - 1.0,
                      np.where(self.left, -lam_h * zh,
                               np.where(self.right, lam_l * zl, (fl * zl - fh * zh) * inv_d)))
        return lf, de, [ds]


class _Ordinal:
    """Ordered logit / probit outcome: anc = cut points kappa_1 < ... < kappa_{K-1}."""

    def __init__(self, y: np.ndarray, K: int, link: str):
        self.y = y
        self.K = K
        self.k_anc = K - 1
        self.link = link
        self.top = (y == K - 1)[:, None]
        self.bot = (y == 0)[:, None]

    def _cdf_pdf(self, z):
        if self.link == "logit":
            return special.log_expit(z), special.expit(z) * special.expit(-z)
        return special.log_ndtr(z), _phi(z)

    def logf(self, eta, anc, grad=True):
        kap = np.r_[-np.inf, np.asarray(anc, float), np.inf]
        if np.any(np.diff(kap[1:-1]) <= 0):
            lf = np.full(eta.shape, -np.inf)
            return lf, np.zeros(eta.shape), [np.zeros(eta.shape) for _ in range(self.k_anc)]
        up = kap[self.y + 1][:, None] - eta
        lo = kap[self.y][:, None] - eta
        upf = np.where(self.top, 0.0, up)
        lof = np.where(self.bot, 0.0, lo)
        # log(F(up) - F(lo)) using symmetry for stability
        flip = lof > 0
        a = np.where(flip, -lof, upf)
        b = np.where(flip, -upf, lof)
        la, _ = self._cdf_pdf(a)
        lb, _ = self._cdf_pdf(b)
        with np.errstate(divide="ignore", invalid="ignore"):
            lmid = la + np.log1p(-np.exp(np.minimum(lb - la, 0.0)))
        l_top, _ = self._cdf_pdf(-lof)
        l_bot, _ = self._cdf_pdf(upf)
        lf = np.where(self.top, l_top, np.where(self.bot, l_bot, lmid))
        if not grad:
            return lf, None, []
        _, fu = self._cdf_pdf(upf)
        _, fl = self._cdf_pdf(lof)
        fu = np.where(self.top, 0.0, fu)
        fl = np.where(self.bot, 0.0, fl)
        inv = np.exp(-lf)
        de = -(fu - fl) * inv
        da = []
        for j in range(self.k_anc):
            # kappa_j (1-based j+1) is the upper cut of category j and the lower cut of category j+1
            du = np.where((self.y == j)[:, None], fu, 0.0)
            dl = np.where((self.y == j + 1)[:, None], -fl, 0.0)
            da.append((du + dl) * inv)
        return lf, de, da


# --------------------------------------------------------------------------------------------- optimizer
def _num_hessian(grad, theta: np.ndarray) -> np.ndarray:
    """Observed information: statsmodels central-difference Jacobian of the analytic score."""
    from statsmodels.tools.numdiff import approx_fprime
    H = np.atleast_2d(approx_fprime(np.asarray(theta, float), grad, centered=True))
    return (H + H.T) / 2.0


def newton(loglik, grad, theta0: np.ndarray, hess=None, maxit: int = 300, tol: float = 1e-13, stop=None):
    """Newton–Raphson with step halving (Stata ml technique(nr)); a non-concave Hessian is replaced by its
    eigenvalue-flipped negative-definite version. Converged when g'(-H)^-1 g < tol."""
    hess = hess or (lambda t: _num_hessian(grad, t))
    theta = np.asarray(theta0, float).copy()
    ll = loglik(theta)
    if not np.isfinite(ll):
        raise ValueError("시작값에서 로그우도 계산 불가 — 결과변수 유형(family)과 자료가 맞는지 확인 "
                         "(빈도 모형은 0 이상 정수·과산포, 순서형은 3개 이상 범주 등)")
    converged = False
    it = 0
    flat = 0
    for it in range(1, maxit + 1):
        g = grad(theta)
        H = hess(theta)
        ev, evec = np.linalg.eigh(H)
        scale_ = max(1.0, float(np.max(np.abs(ev))))
        if np.any(ev > -1e-10 * scale_):
            ev = -np.maximum(np.abs(ev), 1e-6 * scale_)
            H = (evec * ev) @ evec.T
        step = -np.linalg.solve(H, g)
        crit = float(g @ step)
        if crit < tol:
            converged = True
            break
        s = 1.0
        while True:
            tn = theta + s * step
            lln = loglik(tn)
            if np.isfinite(lln) and lln >= ll - 1e-12 * max(1.0, abs(ll)):
                break
            s *= 0.5
            if s < 1e-12:
                tn, lln = theta, ll
                break
        if s < 1e-12:
            converged = crit < 1e-8
            break
        dl = abs(lln - ll)
        theta, ll = tn, lln
        if dl < 1e-15 * max(1.0, abs(ll)) and float(np.max(np.abs(s * step))) < 1e-12:
            converged = True
            break
        # flat likelihood (a variance parameter drifting to its boundary): stop as Stata's ltolerance() does, once the
        # scaled gradient also satisfies Stata's default nrtolerance(1e-5)
        flat = flat + 1 if dl < 1e-12 * max(1.0, abs(ll)) else 0
        if flat >= 2 and crit < 1e-5:
            converged = True
            break
        if stop is not None and stop(theta, ll):
            break
    return theta, ll, converged, it


class FitResult:
    """Estimates in Stata's reporting parameterization."""

    def __init__(self, **kw):
        self.names: List[str] = []           # main-equation names (covariates, _cons)
        self.anc: List[str] = []             # ancillary names as Stata e(b) (/lnsig2u, /sigma_u, /cut1, ...)
        self.b = np.zeros(0)
        self.V = np.zeros((0, 0))
        self.derived: List[Dict[str, Any]] = []   # sigma_u, rho, alpha, r, s (delta method)
        self.ll = float("nan")
        self.ll_c: Optional[float] = None
        self.lr_label = ""
        self.chi2 = None                     # (stat, df, p, type)
        self.N = 0
        self.Ti = np.zeros(0, int)
        self.notes: List[str] = []
        self.omitted: List[str] = []
        self.dropped: List[str] = []
        self.converged = True
        self.iterations = 0
        self.vce_label = "OIM"
        self.n_clusters = None
        self.lib = ""
        self.extra: Dict[str, Any] = {}
        self.__dict__.update(kw)

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.V), 0, None))

    @property
    def k_main(self) -> int:
        return len(self.names)


def _robust_V(V: np.ndarray, scores: np.ndarray, groups: np.ndarray, small: bool = True) -> Tuple[np.ndarray, int]:
    """Cluster sandwich V (sum_c s_c s_c') V * M/(M-1) ([P] robust, ml with panels or clusters); small=False drops
    the M/(M-1) factor (Stata xtpoisson: _robust ..., minus(0))."""
    codes = pd.factorize(groups)[0]
    M = int(codes.max()) + 1
    if M < 2:
        raise ValueError("강건 표준오차: 군집 2개 이상 필요")
    Sc = np.zeros((M, scores.shape[1]))
    np.add.at(Sc, codes, scores)
    meat = Sc.T @ Sc
    return V @ meat @ V * ((M / (M - 1.0)) if small else 1.0), M


def _wald(b: np.ndarray, V: np.ndarray, idx: Sequence[int]) -> Optional[Tuple[float, int, float]]:
    idx = list(idx)
    if not idx:
        return None
    bs, Vs = b[idx], V[np.ix_(idx, idx)]
    try:
        W = float(bs @ np.linalg.solve(Vs, bs))
    except np.linalg.LinAlgError:
        W = float(bs @ np.linalg.pinv(Vs) @ bs)
    return W, len(idx), float(stats.chi2.sf(W, len(idx)))


def _inv_neg(H: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(-H)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(-H)


# --------------------------------------------------------------------------------------------- RE by quadrature
class QuadRE:
    """Random-effects likelihood with a normal panel effect u_i ~ N(0, sigma_u^2) integrated by Gauss–Hermite
    quadrature (Butler & Moffitt 1982). theta = (beta, ancillary, ln sigma_u^2).

    mvaghermite: nodes tau_im = mu_i + sqrt(2) s_i a_m, weights sqrt(2) s_i w_m exp(a_m^2) phi(tau_im; 0, sigma_u^2),
    with mu_i, s_i the posterior mean and SD of u_i (Naylor & Smith 1982; [XT] xtlogit Methods and formulas).
    ghermite:    nodes sqrt(2) sigma_u a_m, weights w_m / sqrt(pi)."""

    def __init__(self, dist, X: np.ndarray, off: np.ndarray, Ti: np.ndarray, method: str = "mvaghermite",
                 points: int = 12):
        self.dist, self.X, self.off = dist, X, off
        self.Ti = Ti
        self.starts = _starts(Ti)
        self.n = len(Ti)
        self.g = np.repeat(np.arange(self.n), Ti)
        self.a, self.w = np.polynomial.hermite.hermgauss(points)
        self.k = X.shape[1]
        self.ka = dist.k_anc
        self.adaptive = method == "mvaghermite"
        self.mu = np.zeros(self.n)
        self.sd = np.ones(self.n)

    def split(self, theta):
        k, ka = self.k, self.ka
        return theta[:k], theta[k:k + ka], float(theta[k + ka])

    def _nodes(self, lns2):
        s2 = math.exp(lns2)
        if self.adaptive:
            tau = self.mu[:, None] + SQ2 * self.sd[:, None] * self.a[None, :]
            base = (np.log(SQ2 * self.sd)[:, None] + np.log(self.w)[None, :] + (self.a ** 2)[None, :]
                    - 0.5 * LOG2PI - 0.5 * lns2 - tau * tau / (2.0 * s2))
            return tau, base, tau[self.g]
        tau = (math.sqrt(s2) * SQ2 * self.a)[None, :]
        base = np.broadcast_to(np.log(self.w / math.sqrt(math.pi))[None, :], (self.n, len(self.a)))
        return tau, base, tau

    def core(self, theta, want_grad=True):
        beta, anc, lns2 = self.split(theta)
        tau, base, tau_obs = self._nodes(lns2)
        xb = self.X @ beta + self.off
        eta = xb[:, None] + tau_obs
        lf, de, da = self.dist.logf(eta, anc, grad=want_grad)
        lp = base + _gsum(lf, self.starts)
        lli = special.logsumexp(lp, axis=1)
        if not want_grad:
            return lli, None, None, tau
        p = np.exp(lp - lli[:, None])
        pg = p[self.g]
        a_obs = np.sum(pg * de, axis=1)
        cols = [_gsum(a_obs[:, None] * self.X, self.starts)] if self.k else [np.zeros((self.n, 0))]
        for d in da:
            cols.append(_gsum(np.sum(pg * d, axis=1), self.starts)[:, None])
        if self.adaptive:
            s2 = math.exp(lns2)
            cols.append(np.sum(p * (-0.5 + tau * tau / (2.0 * s2)), axis=1)[:, None])
        else:
            Sd = _gsum(de, self.starts)
            cols.append(np.sum(p * Sd * tau * 0.5, axis=1)[:, None])
        return lli, np.column_stack(cols), p, tau

    def loglik(self, theta):
        try:
            lli = self.core(theta, False)[0]
        except (FloatingPointError, OverflowError):
            return -np.inf
        v = float(np.sum(lli))
        return v if np.isfinite(v) else -np.inf

    def grad(self, theta):
        return self.core(theta, True)[1].sum(axis=0)

    def scores(self, theta):
        return self.core(theta, True)[1]

    def adapt(self, theta, tol: float = 1e-10, maxit: int = 1001):
        """Posterior mean and SD of u_i at theta (iterated until the nodes stop moving)."""
        if not self.adaptive:
            return
        # start inside the prior: the posterior SD of u_i never exceeds sigma_u (log-concave outcome densities), and a
        # start far outside the prior mass can lock the iteration onto a wrong fixed point when sigma_u is small
        su = math.exp(0.5 * float(theta[-1]))
        self.sd = np.minimum(self.sd, su)
        self.mu = np.where(np.abs(self.mu) > 6.0 * su, 0.0, self.mu)
        if not np.isfinite(self.loglik(theta)):
            self.mu[:] = 0.0
            self.sd[:] = su
        for _ in range(maxit):
            _, _, p, tau = self.core(theta, True)
            m = np.sum(p * tau, axis=1)
            v = np.sum(p * tau * tau, axis=1) - m * m
            sd = np.sqrt(np.maximum(v, 1e-12))
            delta = max(float(np.max(np.abs(m - self.mu))), float(np.max(np.abs(sd - self.sd))))
            self.mu, self.sd = m, sd
            if delta < tol:
                break

    def fit(self, theta0: np.ndarray, ll_pooled: Optional[float] = None):
        """Newton–Raphson on fixed nodes, re-adapting the nodes until they stop moving. With ll_pooled given, the fit
        stops early when sigma_u heads to the boundary 0 (ln sigma_u^2 < -8 with no gain over the pooled model);
        self.boundary is then True and the caller reports the boundary solution."""
        theta = np.asarray(theta0, float)
        conv, its, ll_prev = False, 0, None
        self.boundary = False

        def at_bound(th, ll_):
            return ll_pooled is not None and th[-1] < -8.0 and ll_ - ll_pooled < 1e-4

        for _ in range(40):
            self.adapt(theta)
            new, ll, conv, it = newton(self.loglik, self.grad, theta, maxit=100, stop=at_bound)
            if at_bound(new, ll):
                self.boundary = True
                return new, ll, True, its + it, None
            its += it
            moved = float(np.max(np.abs(new - theta)))
            theta = new
            if not self.adaptive or moved < 1e-9 or theta[-1] < -14.0:
                break
            if ll_prev is not None and abs(ll - ll_prev) < 1e-12 * max(1.0, abs(ll)):
                break
            ll_prev = ll
        self.adapt(theta)
        theta, ll, conv2, it = newton(self.loglik, self.grad, theta, maxit=100)
        H = _num_hessian(self.grad, theta)
        return theta, ll, conv and conv2, its + it, H


# ----------------------------------------------------------------------------------- closed-form count likelihoods
class PoissonGammaRE:
    """xtpoisson, re (gamma heterogeneity): Hausman, Hall & Griliches (1984); [XT] xtpoisson Methods and formulas.
    theta = (beta, ln alpha), theta_gamma = 1/alpha."""

    def __init__(self, y, X, off, Ti):
        self.y, self.X, self.off, self.Ti = y, X, off, Ti
        self.starts = _starts(Ti)
        self.Y = _gsum(y, self.starts)
        self.cy = _gsum(special.gammaln(y + 1.0), self.starts)
        self.k = X.shape[1]

    def _parts(self, theta):
        beta, lna = theta[:-1], float(theta[-1])
        eta = self.X @ beta + self.off
        lam = np.exp(np.minimum(eta, 700.0))
        S = _gsum(lam, self.starts)
        th = math.exp(-lna)
        return eta, lam, S, th

    def loglik_i(self, theta):
        eta, lam, S, th = self._parts(theta)
        Y = self.Y
        return (special.gammaln(th + Y) - special.gammaln(th) + th * math.log(th) - (th + Y) * np.log(th + S)
                + _gsum(self.y * eta, self.starts) - self.cy)

    def loglik(self, theta):
        v = float(np.sum(self.loglik_i(theta)))
        return v if np.isfinite(v) else -np.inf

    def scores(self, theta):
        eta, lam, S, th = self._parts(theta)
        Y = self.Y
        r = (th + Y) / (th + S)
        sb = _gsum((self.y - r[np.repeat(np.arange(len(S)), self.Ti)] * lam)[:, None] * self.X, self.starts)
        dth = special.digamma(th + Y) - special.digamma(th) + math.log(th) + 1.0 - np.log(th + S) - r
        return np.column_stack([sb, -th * dth])

    def grad(self, theta):
        return self.scores(theta).sum(0)


class NBRandom:
    """xtnbreg, re: beta-distributed dispersion, Hausman, Hall & Griliches (1984, eq. 3.5); [XT] xtnbreg Methods and
    formulas. theta = (beta, ln r, ln s)."""

    def __init__(self, y, X, off, Ti):
        self.y, self.X, self.off, self.Ti = y, X, off, Ti
        self.starts = _starts(Ti)
        self.Y = _gsum(y, self.starts)
        self.cy = _gsum(special.gammaln(y + 1.0), self.starts)
        self.g = np.repeat(np.arange(len(Ti)), Ti)
        self.k = X.shape[1]

    def loglik_i(self, theta):
        beta, lr, ls = theta[:-2], float(theta[-2]), float(theta[-1])
        r, s = math.exp(lr), math.exp(ls)
        lam = np.exp(np.minimum(self.X @ beta + self.off, 700.0))
        L = _gsum(lam, self.starts)
        Y = self.Y
        return (special.gammaln(r + s) + special.gammaln(r + L) + special.gammaln(s + Y) - special.gammaln(r)
                - special.gammaln(s) - special.gammaln(r + s + L + Y)
                + _gsum(special.gammaln(lam + self.y) - special.gammaln(lam), self.starts) - self.cy)

    def loglik(self, theta):
        v = float(np.sum(self.loglik_i(theta)))
        return v if np.isfinite(v) else -np.inf

    def scores(self, theta):
        beta, lr, ls = theta[:-2], float(theta[-2]), float(theta[-1])
        r, s = math.exp(lr), math.exp(ls)
        lam = np.exp(np.minimum(self.X @ beta + self.off, 700.0))
        L = _gsum(lam, self.starts)
        Y = self.Y
        big = special.digamma(r + s + L + Y)
        c = (special.digamma(r + L) - big)[self.g] + special.digamma(lam + self.y) - special.digamma(lam)
        sb = _gsum((lam * c)[:, None] * self.X, self.starts)
        sr = r * (special.digamma(r + s) + special.digamma(r + L) - special.digamma(r) - big)
        ss = s * (special.digamma(r + s) + special.digamma(s + Y) - special.digamma(s) - big)
        return np.column_stack([sb, sr, ss])

    def grad(self, theta):
        return self.scores(theta).sum(0)


class NBConditional:
    """xtnbreg, fe: conditional likelihood of Hausman, Hall & Griliches (1984); [XT] xtnbreg Methods and formulas
    (constant and time-invariant covariates stay identified)."""

    def __init__(self, y, X, off, Ti):
        self.y, self.X, self.off, self.Ti = y, X, off, Ti
        self.starts = _starts(Ti)
        self.Y = _gsum(y, self.starts)
        self.cy = _gsum(special.gammaln(y + 1.0), self.starts)
        self.g = np.repeat(np.arange(len(Ti)), Ti)

    def loglik_i(self, theta):
        lam = np.exp(np.minimum(self.X @ theta + self.off, 700.0))
        L = _gsum(lam, self.starts)
        Y = self.Y
        return (special.gammaln(L) + special.gammaln(Y + 1.0) - special.gammaln(L + Y)
                + _gsum(special.gammaln(lam + self.y) - special.gammaln(lam), self.starts) - self.cy)

    def loglik(self, theta):
        v = float(np.sum(self.loglik_i(theta)))
        return v if np.isfinite(v) else -np.inf

    def scores(self, theta):
        lam = np.exp(np.minimum(self.X @ theta + self.off, 700.0))
        L = _gsum(lam, self.starts)
        c = (special.digamma(L) - special.digamma(L + self.Y))[self.g] + special.digamma(lam + self.y) \
            - special.digamma(lam)
        return _gsum((lam * c)[:, None] * self.X, self.starts)

    def grad(self, theta):
        return self.scores(theta).sum(0)


def _stata_conditional_poisson(y, X, g, off):
    """statsmodels ConditionalPoisson whose groups follow Stata xtpoisson, fe (the caller drops single-observation
    and all-zero groups); the library's own rule also drops groups with a constant positive count, which do enter
    Stata's conditional likelihood, so the per-group lists are rebuilt for every remaining group."""
    from statsmodels.discrete.conditional_models import ConditionalPoisson
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = ConditionalPoisson(y, X, groups=g, offset=off)
    starts = _starts(np.bincount(g))
    ends = np.r_[starts[1:], len(y)]
    mod._endog_grp = [y[a:b] for a, b in zip(starts, ends)]
    mod._exog_grp = [X[a:b] for a, b in zip(starts, ends)]
    mod._offset_grp = [off[a:b] for a, b in zip(starts, ends)]
    mod._groupsize = [b - a for a, b in zip(starts, ends)]
    mod._sumy = [float(np.sum(y[a:b])) for a, b in zip(starts, ends)]
    mod._n_groups = len(starts)
    mod.nobs = len(y)
    return mod


# --------------------------------------------------------------------------------------------- GEE working correlations
def _stata_ar1_struct(tpos_li: List[np.ndarray]):
    """statsmodels CovStruct for xtgee corr(ar 1): alpha = [sum r_t r_t+1 / sum(n_i - 1)] / [sum r^2 / N] on Pearson
    residuals of the panels used (the moment estimator panel_module.fit_pa matched for the gaussian family), and
    R_ts = alpha^|t - s| on the positions within the panel."""
    from statsmodels.genmod.cov_struct import CovStruct

    class StataAR1(CovStruct):
        def __init__(self):
            super().__init__()
            self.dep_params = 0.0

        def update(self, params):
            num = den = npair = nobs = 0.0
            varf = self.model.family.variance
            for i in range(self.model.num_group):
                mu = self.model.cached_means[i][0]
                r = (self.model.endog_li[i] - mu) / np.sqrt(varf(mu))
                den += float(np.sum(r * r))
                nobs += len(r)
                if len(r) < 2:
                    continue
                num += float(np.sum(r[1:] * r[:-1]))
                npair += len(r) - 1
            self.dep_params = (num / npair) / (den / nobs) if npair and den > 0 else 0.0

        def covariance_matrix(self, expval, index):
            ix = tpos_li[index]
            return self.dep_params ** np.abs(ix[:, None] - ix[None, :]).astype(float), True

        def summary(self):
            return f"Stata AR(1) correlation: {self.dep_params:.6f}"

    return StataAR1()


def _stata_unstructured_struct(tpos_li: List[np.ndarray], q: int):
    """statsmodels CovStruct for xtgee corr(unstructured): alpha_pq = [sum_i r_ip r_iq / N_pq] / [sum r^2 / N],
    N_pq = number of panels observed at both times p and q ([XT] xtgee Methods and formulas)."""
    from statsmodels.genmod.cov_struct import CovStruct

    class StataUnstructured(CovStruct):
        def __init__(self):
            super().__init__()
            self.dep_params = np.eye(q)

        def update(self, params):
            varf = self.model.family.variance
            num = np.zeros((q, q))
            cnt = np.zeros((q, q))
            ssr = nobs = 0.0
            for i in range(self.model.num_group):
                mu = self.model.cached_means[i][0]
                r = (self.model.endog_li[i] - mu) / np.sqrt(varf(mu))
                ix = tpos_li[i]
                num[np.ix_(ix, ix)] += np.outer(r, r)
                cnt[np.ix_(ix, ix)] += 1.0
                ssr += float(np.sum(r * r))
                nobs += len(r)
            with np.errstate(invalid="ignore", divide="ignore"):
                R = np.where(cnt > 0, num / np.maximum(cnt, 1.0), 0.0) / (ssr / nobs)
            np.fill_diagonal(R, 1.0)
            self.dep_params = R

        def covariance_matrix(self, expval, index):
            ix = tpos_li[index]
            return self.dep_params[np.ix_(ix, ix)], True

        def summary(self):
            return "Stata unstructured correlation"

    return StataUnstructured()


# --------------------------------------------------------------------------------------------- design helpers
def _collinear(X: np.ndarray, names: List[str], const: bool) -> Tuple[List[int], List[str]]:
    Z = np.column_stack([X, np.ones(len(X))]) if const else X
    if Z.shape[1] == 0:
        return [], []
    keep = independent_columns(Z, protect=[Z.shape[1] - 1] if const else [])
    kept = [j for j in keep if j < X.shape[1]]
    return kept, [names[j] for j in range(len(names)) if j not in kept]


def _within_kept(X: np.ndarray, g: np.ndarray, Ti: np.ndarray, names: List[str]):
    """Covariates identified in a conditional (within-panel) likelihood: no within variation -> omitted, then
    collinear columns of the within-demeaned data dropped (Stata: omitted because of no within-group variance)."""
    if X.shape[1] == 0:
        return [], [], []
    s = np.zeros((len(Ti), X.shape[1]))
    np.add.at(s, g, X)
    Xw = X - (s / Ti[:, None])[g]
    scale_ = np.maximum(np.abs(X).max(axis=0), 1.0)
    novar = [j for j in range(X.shape[1]) if float(np.max(np.abs(Xw[:, j]))) <= 1e-10 * scale_[j]]
    cand = [j for j in range(X.shape[1]) if j not in novar]
    kept = []
    if cand:
        kk = independent_columns(Xw[:, cand])
        kept = [cand[j] for j in kk]
    coll = [j for j in cand if j not in kept]
    return kept, [names[j] for j in novar], [names[j] for j in coll]


def _drop_groups(D: NLData, mask_group: np.ndarray) -> NLData:
    rows = np.where(mask_group[D.g])[0]
    return D.subset(rows)


def _main_names(names: List[str], const: bool) -> List[str]:
    return list(names) + (["_cons"] if const else [])


# --------------------------------------------------------------------------------------------- pooled comparison models
def _pooled(fam: str, y, Z, off, dist=None, X_noconst=None):
    """Pooled (no panel effect) model: (params, loglik). Library models where they exist."""
    import statsmodels.api as sm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        has_const = Z is not None and Z.shape[1] > 0 and np.allclose(Z[:, -1], 1.0)
        if fam in ("logit", "probit", "poisson"):
            cls = {"logit": sm.Logit, "probit": sm.Probit, "poisson": sm.Poisson}[fam]
            m = cls(y, Z, offset=off)
            start = np.zeros(Z.shape[1])
            if has_const:
                yb = min(max(float(np.mean(y)), 1e-6), 1 - 1e-6) if fam != "poisson" else max(float(np.mean(y)), 1e-6)
                c0 = (math.log(yb / (1 - yb)) if fam == "logit" else float(stats.norm.ppf(yb)) if fam == "probit"
                      else math.log(yb))
                start[-1] = c0 - float(np.mean(off))
            th, ll, _, _ = newton(m.loglike, m.score, start, hess=m.hessian)
            return th, float(ll)
        if fam == "cloglog":
            r = sm.GLM(y, Z, family=sm.families.Binomial(sm.families.links.CLogLog()), offset=off).fit(tol=1e-13,
                                                                                                      maxiter=300)
            return np.asarray(r.params), float(r.llf)
        if fam == "nb1":
            from statsmodels.discrete.discrete_model import NegativeBinomialP
            m = NegativeBinomialP(y, Z, p=1, offset=off)
            r = m.fit(method="bfgs", disp=0, maxiter=2000, gtol=1e-10, skip_hessian=True)
            th, ll, _, _ = newton(m.loglike, m.score, np.asarray(r.params))
            return th, float(ll)
        if fam in ("ologit", "oprobit"):
            from statsmodels.miscmodels.ordinal_model import OrderedModel
            m = OrderedModel(y, X_noconst, distr="logit" if fam == "ologit" else "probit")
            r = m.fit(method="bfgs", disp=0, maxiter=2000, gtol=1e-10, skip_hessian=True)
            th, ll, _, _ = newton(m.loglike, m.score, np.asarray(r.params))
            k = X_noconst.shape[1]
            cuts = np.cumsum(np.r_[th[k], np.exp(th[k + 1:])])
            return np.r_[th[:k], cuts], float(ll)
    # censored / interval normal (tobit, intreg): no library estimator; same observation likelihood as the RE model
    mid = np.where(np.isfinite(dist.lo[:, 0]) & np.isfinite(dist.hi[:, 0]), (dist.lo[:, 0] + dist.hi[:, 0]) / 2,
                   np.where(np.isfinite(dist.lo[:, 0]), dist.lo[:, 0], dist.hi[:, 0]))
    b0 = np.linalg.lstsq(Z, mid - off, rcond=None)[0]
    s0 = float(np.std(mid - off - Z @ b0)) or 1.0

    def ll_(t):
        lf = dist.logf((Z @ t[:-1] + off)[:, None], [t[-1]], grad=False)[0]
        v = float(np.sum(lf))
        return v if np.isfinite(v) else -np.inf

    def gr_(t):
        _, de, da = dist.logf((Z @ t[:-1] + off)[:, None], [t[-1]], grad=True)
        return np.r_[Z.T @ de[:, 0], np.sum(da[0])]

    th, ll, _, _ = newton(ll_, gr_, np.r_[b0, math.log(s0)])
    return th, float(ll)


# --------------------------------------------------------------------------------------------- estimators
def _vce_groups(D: NLData, o: Dict[str, Any]) -> Tuple[np.ndarray, str]:
    """Cluster id per panel for the sandwich: panels (vce(robust)) or the cluster variable."""
    st = _starts(D.Ti)
    if o["covType"] == "cluster" and D.cluster is not None:
        return D.cluster[st], f"군집-강건 ({o['clusterVar']})"
    return np.arange(D.n), "강건 (패널 군집)"


def _derived(name: str, est: float, se: float, lo: float, hi: float) -> Dict[str, Any]:
    return {"name": name, "est": est, "se": se, "lo": lo, "hi": hi}


def fit_re_quad(D: NLData, o: Dict[str, Any]) -> FitResult:
    fam, kind = o["family"], FAMILIES[o["family"]]["kind"]
    const = o["constant"] and kind != "ordinal"
    kept, omitted = _collinear(D.X, D.covs, const)
    names = [D.covs[j] for j in kept]
    X = D.X[:, kept]
    Z = np.column_stack([X, np.ones(D.N)]) if const else X
    zcrit = stats.norm.ppf(0.5 + o["ciLevel"] / 200.0)
    if kind == "binary":
        dist = _Binary(D.y, fam)
        bp, llc = _pooled(fam, D.y, Z, D.off)
        anc0, c = np.zeros(0), FAMILIES[fam]["rho_c"]
    elif fam == "poisson":
        dist = _PoissonN(D.y)
        bp, llc = _pooled("poisson", D.y, Z, D.off)
        anc0, c = np.zeros(0), None
    elif kind in ("censored", "interval"):
        dist = _Interval(D.lo, D.hi)
        tp, llc = _pooled(fam, None, Z, D.off, dist=dist)
        bp, anc0, c = tp[:-1], tp[-1:], None
    else:
        K = int(D.y.max()) + 1
        dist = _Ordinal(D.y, K, "logit" if fam == "ologit" else "probit")
        tp, llc = _pooled(fam, D.y, None, None, X_noconst=X)
        bp, anc0, c = tp[:X.shape[1]], tp[X.shape[1]:], math.pi ** 2 / 3 if fam == "ologit" else 1.0
    model = QuadRE(dist, Z, D.off, D.Ti, o["intMethod"], o["intPoints"])
    # starting values: small grid over sigma_u around the pooled fit
    best, best_ll = None, -np.inf
    for sgm in (0.25, 0.5, 1.0, 1.5, 2.0):
        if kind in ("censored", "interval"):
            tot = math.exp(float(anc0[0]))
            for f in (0.2, 0.5, 0.8):
                th = np.r_[bp, math.log(tot * math.sqrt(1 - f)), math.log(f * tot * tot)]
                model.mu[:] = 0.0
                model.sd[:] = 1.0
                model.adapt(th)
                ll = model.loglik(th)
                if ll > best_ll:
                    best, best_ll = th, ll
            continue
        if fam == "poisson":
            bb = bp.copy()
            if const:
                bb[-1] -= sgm * sgm / 2.0
            th = np.r_[bb, math.log(sgm * sgm)]
        else:
            f = math.sqrt(1.0 + sgm * sgm / c)
            th = np.r_[bp * f, anc0 * f, math.log(sgm * sgm)]
        model.mu[:] = 0.0
        model.sd[:] = 1.0
        model.adapt(th)
        ll = model.loglik(th)
        if ll > best_ll:
            best, best_ll = th, ll
    model.mu[:] = 0.0
    model.sd[:] = 1.0
    # one-sided check at the pooled estimates: LL falls as sigma_u^2 leaves 0 and no grid point beats the pooled LL ->
    # the maximum is on the boundary (no Newton drift of ln sigma_u^2 towards -inf)
    th_b = np.r_[bp, anc0, -8.0]
    model.adapt(th_b)
    boundary = best_ll <= llc + 1e-4 and float(model.grad(th_b)[-1]) <= 0.0
    model.mu[:] = 0.0
    model.sd[:] = 1.0
    if boundary:
        theta, ll, conv, its, H = th_b, llc, True, 0, None
    else:
        theta, ll, conv, its, H = model.fit(best, ll_pooled=llc)
        boundary = model.boundary
    if boundary:
        # sigma_u -> 0: the maximum is the pooled model (Stata: chibar2(01) = 0); report the pooled estimates with
        # ln sigma_u^2 at -14 (Stata xtlogit's threshold for rho = 0)
        if kind in ("censored", "interval"):
            theta = np.r_[bp, anc0, -14.0]
        elif kind == "ordinal":
            theta = np.r_[bp, anc0, -14.0]
        else:
            theta = np.r_[bp, -14.0]
        model.adapt(theta)
        ll = llc                                   # the likelihood at sigma_u = 0 is the pooled likelihood
        H = _num_hessian(model.grad, theta)
    if boundary:          # variance parameter not identified at the boundary: invert the remaining block
        V = np.full(H.shape, np.nan)
        V[:-1, :-1] = _inv_neg(H[:-1, :-1])
    else:
        V = _inv_neg(H)
    fr = FitResult(N=D.N, Ti=D.Ti, omitted=omitted, converged=conv, iterations=its, ll=ll, ll_c=llc)
    fr.lib = "Gauss–Hermite 구적법 (numpy hermgauss), Newton–Raphson"
    if boundary:
        fr.notes.append("확률효과 분산이 경계(0)로 수렴 — 통합(pooled) 모형과 같은 추정치, ln(sigma_u²) = -14로 표시 "
                        "(Stata: LR chibar2(01) = 0; 확률효과 분산의 표준오차는 해석 불가)")
    if o["covType"] in ("robust", "cluster"):
        groups, lab = _vce_groups(D, o)
        if boundary:
            Vb, M = _robust_V(V[:-1, :-1], model.scores(theta)[:, :-1], groups)
            V[:-1, :-1] = Vb
        else:
            V, M = _robust_V(V, model.scores(theta), groups)
        fr.vce_label, fr.n_clusters = f"{lab} 군집 {M}개", M
    k = Z.shape[1]
    beta, anc, lns2 = theta[:k], theta[k:k + dist.k_anc], float(theta[-1])
    V_in = V
    V = np.nan_to_num(V_in, nan=0.0)       # transforms below; the variance row is reset to NaN at the boundary
    se_l = math.sqrt(max(V_in[-1, -1], 0.0)) if np.isfinite(V_in[-1, -1]) else float("nan")
    fr.names = _main_names(names, const)
    if kind == "binary" or fam == "poisson":
        fr.anc = ["/lnsig2u"]
        fr.b, fr.V = theta.copy(), V
        su = math.exp(0.5 * lns2)
        lo_l, hi_l = lns2 - zcrit * se_l, lns2 + zcrit * se_l
        fr.derived.append(_derived("sigma_u", su, 0.5 * su * se_l, math.exp(0.5 * lo_l), math.exp(0.5 * hi_l)))
        if kind == "binary":
            e = math.exp(lns2)
            rho = e / (c + e)
            fr.derived.append(_derived("rho", rho, c * e / (c + e) ** 2 * se_l, math.exp(lo_l) / (c + math.exp(lo_l)),
                                       math.exp(hi_l) / (c + math.exp(hi_l))))
        fr.lr_label = "rho=0" if kind == "binary" else "sigma_u=0"
        tiny = lns2 <= -14
    elif kind in ("censored", "interval"):
        se_, su = math.exp(float(anc[0])), math.exp(0.5 * lns2)
        J = np.eye(len(theta))
        J = J[:k + 2]
        J[k, :] = 0.0
        J[k, -1] = su / 2.0                       # sigma_u = exp(lns2 / 2)
        J[k + 1, :] = 0.0
        J[k + 1, k] = se_                         # sigma_e = exp(ln sigma_e)
        fr.b = np.r_[beta, su, se_]
        fr.V = J @ V @ J.T
        fr.anc = ["/sigma_u", "/sigma_e"]
        Vs = fr.V[k:k + 2, k:k + 2]
        den = su * su + se_ * se_
        rho = su * su / den
        grad_r = np.array([2 * su * se_ * se_ / den ** 2, -2 * se_ * su * su / den ** 2])
        se_r = math.sqrt(max(float(grad_r @ Vs @ grad_r), 0.0))
        zr = stats.norm.ppf(rho)
        sz = se_r / stats.norm.pdf(zr)           # Stata diparm ci(probit)
        fr.derived.append(_derived("rho", rho, se_r, stats.norm.cdf(zr - zcrit * sz), stats.norm.cdf(zr + zcrit * sz)))
        fr.lr_label = "sigma_u=0"
        tiny = su < 1e-5
    else:
        s2 = math.exp(lns2)
        J = np.eye(len(theta))
        J[-1, -1] = s2
        fr.b = np.r_[beta, anc, s2]
        fr.V = J @ V @ J.T
        fr.anc = [f"/cut{j + 1}" for j in range(dist.k_anc)] + ["/sigma2_u"]
        fr.extra["log_ci"] = {"/sigma2_u": (math.exp(lns2 - zcrit * se_l), math.exp(lns2 + zcrit * se_l))}
        fr.lr_label = "ologit" if fam == "ologit" else "oprobit"
        tiny = lns2 <= -14
    if boundary:
        j_var = k if kind in ("censored", "interval") else len(fr.b) - 1     # /sigma_u, /lnsig2u or /sigma2_u
        fr.V[j_var, :] = np.nan
        fr.V[:, j_var] = np.nan
        if kind in ("censored", "interval"):        # Stata (natural scale): /sigma_u = 0, rho = 0
            fr.b[k] = 0.0
            fr.derived = [_derived("rho", 0.0, float("nan"), float("nan"), float("nan"))]
    fr.chi2_lr = 0.0 if (tiny or boundary) else abs(2.0 * (ll - llc))
    slopes = list(range(len(names)))
    w = _wald(fr.b, fr.V, slopes)
    fr.chi2 = (w[0], w[1], w[2], "Wald") if w else None
    fr.extra.update({"model": model, "theta": theta, "dist": dist, "Z": Z, "const": const, "kind": kind,
                     "int": (o["intMethod"], o["intPoints"])})
    if kind in ("censored", "interval"):
        fr.extra["censor"] = {k_: getattr(D, k_, None) for k_ in ("n_left", "n_right", "n_point", "n_interval")}
    return fr


def fit_clogit(D: NLData, o: Dict[str, Any]) -> FitResult:
    """xtlogit, fe / clogit — statsmodels ConditionalLogit."""
    from statsmodels.discrete.conditional_models import ConditionalLogit
    s1 = np.bincount(D.g, weights=D.y, minlength=D.n)
    keep_g = (s1 > 0) & (s1 < D.Ti)
    n_drop_g, n_drop_o = int(np.sum(~keep_g)), int(np.sum(D.Ti[~keep_g]))
    if not keep_g.any():
        raise ValueError("조건부 로짓: 결과가 모두 0 또는 모두 1인 집단만 있어 추정 불가")
    E = _drop_groups(D, keep_g)
    kept, novar, coll = _within_kept(E.X, E.g, E.Ti, E.covs)
    if not kept:
        raise ValueError("조건부 로짓: 집단 내 변동이 있는 설명변수 없음")
    X = E.X[:, kept]
    fr = FitResult(N=E.N, Ti=E.Ti, omitted=novar + coll)
    fr.lib = "statsmodels ConditionalLogit"
    multi = bool(np.any(np.bincount(E.g, weights=E.y) > 1))
    if multi:
        fr.notes.append("집단 내 양(1)의 결과가 여러 개인 집단 있음 (Stata: multiple positive outcomes within groups "
                        "encountered)")
    if n_drop_g:
        fr.dropped.append(f"결과가 모두 0 또는 모두 1인 {n_drop_g}개 집단({n_drop_o}개 관측치) 제외 (Stata: omitted because "
                          "of all positive or all negative outcomes)")
    for v in novar:
        fr.notes.append(f"{v}: 집단 내 변동 없음으로 제외 (Stata: omitted because of no within-group variance)")
    for v in coll:
        fr.notes.append(f"{v}: 공선성으로 제외 (Stata omitted)")
    kw = {"offset": E.off} if o.get("offset") else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = ConditionalLogit(E.y, X, groups=E.g, **kw)
        r0 = mod.fit(method="bfgs", maxiter=2000, disp=False, skip_hessian=True, gtol=1e-9)
    th, ll, conv, it = newton(mod.loglike, mod.score, np.asarray(r0.params))
    H = _num_hessian(mod.score, th)
    V = _inv_neg(H)
    fr.converged, fr.iterations, fr.ll = conv, it, float(ll)
    fr.ll_0 = float(mod.loglike(np.zeros(len(th))))
    if o["covType"] in ("robust", "cluster"):
        sc = np.vstack([mod.score_grp(i, th) for i in range(mod._n_groups)])
        if o["covType"] == "cluster":
            cl = E.cluster[_starts(E.Ti)]
            lab = f"군집-강건 ({o['clusterVar']})"
        else:
            cl, lab = np.arange(E.n), "강건 (집단 군집)"
        V, M = _robust_V(V, sc, cl)
        fr.vce_label, fr.n_clusters = f"{lab} 군집 {M}개", M
    fr.names = [E.covs[j] for j in kept]
    fr.b, fr.V = th, V
    if o["covType"] == "oim":
        lr = 2.0 * (fr.ll - fr.ll_0)
        fr.chi2 = (lr, len(th), float(stats.chi2.sf(lr, len(th))), "LR")
    else:
        w = _wald(th, V, range(len(th)))
        fr.chi2 = (w[0], w[1], w[2], "Wald")
    fr.extra.update({"n_drop_g": n_drop_g, "n_drop_o": n_drop_o, "const": False, "kind": "binary"})
    return fr


def fit_cpoisson(D: NLData, o: Dict[str, Any]) -> FitResult:
    """xtpoisson, fe — statsmodels ConditionalPoisson likelihood and score (+ multinomial constant)."""
    Y = np.bincount(D.g, weights=D.y, minlength=D.n)
    single = D.Ti == 1
    zero = (Y == 0) & ~single
    E = _drop_groups(D, ~single & ~zero)
    if E.N == 0:
        raise ValueError("조건부 포아송: 추정에 쓸 집단 없음 (관측치 1개 집단·결과가 모두 0인 집단 제외 후)")
    fr = FitResult(N=E.N, Ti=E.Ti)
    fr.lib = "statsmodels ConditionalPoisson"
    if single.any():
        fr.dropped.append(f"관측치 1개인 {int(single.sum())}개 집단({int(D.Ti[single].sum())}개 관측치) 제외 (Stata: dropped "
                          "because of only one obs per group)")
    if zero.any():
        fr.dropped.append(f"결과가 모두 0인 {int(zero.sum())}개 집단({int(D.Ti[zero].sum())}개 관측치) 제외 (Stata: dropped "
                          "because of all zero outcomes)")
    kept, novar, coll = _within_kept(E.X, E.g, E.Ti, E.covs)
    if not kept:
        raise ValueError("조건부 포아송: 집단 내 변동이 있는 설명변수 없음")
    for v in novar:
        fr.notes.append(f"{v}: 집단 내 상수로 제외 (Stata: dropped because it is constant within group)")
    for v in coll:
        fr.notes.append(f"{v}: 공선성으로 제외 (Stata omitted)")
    fr.omitted = novar + coll
    X = E.X[:, kept]
    mod = _stata_conditional_poisson(E.y, X, E.g, E.off)
    th, ll, conv, it = newton(mod.loglike, mod.score, np.zeros(X.shape[1]))
    H = _num_hessian(mod.score, th)
    V = _inv_neg(H)
    st = _starts(E.Ti)
    Ysum = _gsum(E.y, st)
    const_ll = float(np.sum(special.gammaln(Ysum + 1.0)) - np.sum(special.gammaln(E.y + 1.0)))
    fr.converged, fr.iterations, fr.ll = conv, it, float(ll) + const_ll
    if o["covType"] == "robust":
        eta = X @ th + E.off
        lam = np.exp(eta - np.max(eta))
        sl = np.bincount(E.g, weights=lam)
        p = lam / sl[E.g]
        sc = _gsum((E.y - Ysum[E.g] * p)[:, None] * X, st)
        V, M = _robust_V(V, sc, np.arange(E.n), small=False)
        fr.vce_label, fr.n_clusters = f"강건 (Wooldridge 1999, 집단 군집 {M}개)", M
    fr.names = [E.covs[j] for j in kept]
    fr.b, fr.V = th, V
    w = _wald(th, V, range(len(th)))
    fr.chi2 = (w[0], w[1], w[2], "Wald")
    fr.extra.update({"const": False, "kind": "count"})
    return fr


def fit_nb_fe(D: NLData, o: Dict[str, Any]) -> FitResult:
    Y = np.bincount(D.g, weights=D.y, minlength=D.n)
    single = D.Ti == 1
    zero = (Y == 0) & ~single
    E = _drop_groups(D, ~single & ~zero)
    fr = FitResult(N=E.N, Ti=E.Ti)
    fr.lib = "Hausman, Hall & Griliches (1984) 조건부 우도 (scipy.special gammaln·digamma), Newton–Raphson"
    if single.any():
        fr.dropped.append(f"관측치 1개인 {int(single.sum())}개 집단({int(D.Ti[single].sum())}개 관측치) 제외 (Stata: dropped "
                          "because of only one obs per group)")
    if zero.any():
        fr.dropped.append(f"결과가 모두 0인 {int(zero.sum())}개 집단({int(D.Ti[zero].sum())}개 관측치) 제외 (Stata: dropped "
                          "because of all zero outcomes)")
    const = o["constant"]
    kept, omitted = _collinear(E.X, E.covs, const)
    fr.omitted = omitted
    X = E.X[:, kept]
    Z = np.column_stack([X, np.ones(E.N)]) if const else X
    bp, _ = _pooled("poisson", E.y, Z, E.off)
    mod = NBConditional(E.y, Z, E.off, E.Ti)
    th, ll, conv, it = newton(mod.loglik, mod.grad, bp)
    V = _inv_neg(_num_hessian(mod.grad, th))
    fr.converged, fr.iterations, fr.ll = conv, it, float(ll)
    fr.names = _main_names([E.covs[j] for j in kept], const)
    fr.b, fr.V = th, V
    w = _wald(th, V, range(len(kept)))
    fr.chi2 = (w[0], w[1], w[2], "Wald") if w else None
    fr.extra.update({"const": const, "kind": "count"})
    return fr


def fit_pois_gamma(D: NLData, o: Dict[str, Any]) -> FitResult:
    const = o["constant"]
    kept, omitted = _collinear(D.X, D.covs, const)
    X = D.X[:, kept]
    Z = np.column_stack([X, np.ones(D.N)]) if const else X
    bp, llc = _pooled("poisson", D.y, Z, D.off)
    mod = PoissonGammaRE(D.y, Z, D.off, D.Ti)
    best = max((np.r_[bp, la] for la in (-2.0, -1.0, 0.0, 1.0)), key=mod.loglik)
    th, ll, conv, it = newton(mod.loglik, mod.grad, best)
    V = _inv_neg(_num_hessian(mod.grad, th))
    fr = FitResult(N=D.N, Ti=D.Ti, omitted=omitted, converged=conv, iterations=it, ll=float(ll), ll_c=llc)
    fr.lib = "Hausman, Hall & Griliches (1984) 감마 확률효과 우도, Newton–Raphson"
    if o["covType"] in ("robust", "cluster"):
        groups, lab = _vce_groups(D, o)
        V, M = _robust_V(V, mod.scores(th), groups, small=False)
        fr.vce_label, fr.n_clusters = f"{lab} 군집 {M}개", M
        fr.notes.append("강건 분산: 패널 단위 점수의 샌드위치 (Stata xtpoisson과 같은 소표본 보정 없음). Stata 19.5 "
                        "xtpoisson, re vce(robust)는 /lnalpha 패널 점수를 관측치마다 반복해 합산(T_i배)하고 Wald χ²에 "
                        "_cons·/lnalpha를 포함하므로 표준오차·χ²가 다름")
    zc = stats.norm.ppf(0.5 + o["ciLevel"] / 200.0)
    fr.names = _main_names([D.covs[j] for j in kept], const)
    fr.anc = ["/lnalpha"]
    fr.b, fr.V = th, V
    la, sl = float(th[-1]), math.sqrt(max(V[-1, -1], 0.0))
    fr.derived.append(_derived("alpha", math.exp(la), math.exp(la) * sl, math.exp(la - zc * sl),
                               math.exp(la + zc * sl)))
    fr.lr_label = "alpha=0"
    fr.chi2_lr = abs(2.0 * (ll - llc))
    w = _wald(th, V, range(len(kept)))
    fr.chi2 = (w[0], w[1], w[2], "Wald") if w else None
    fr.extra.update({"const": const, "kind": "count"})
    return fr


def fit_nb_re(D: NLData, o: Dict[str, Any]) -> FitResult:
    const = o["constant"]
    kept, omitted = _collinear(D.X, D.covs, const)
    X = D.X[:, kept]
    Z = np.column_stack([X, np.ones(D.N)]) if const else X
    bp, _ = _pooled("poisson", D.y, Z, D.off)
    _, llc = _pooled("nb1", D.y, Z, D.off)
    mod = NBRandom(D.y, Z, D.off, D.Ti)
    cands = [np.r_[bp, lr, ls] for lr in (0.0, 1.0, 2.0, 3.0) for ls in (0.0, 1.0, 2.0)]
    best = max(cands, key=mod.loglik)
    th, ll, conv, it = newton(mod.loglik, mod.grad, best)
    V = _inv_neg(_num_hessian(mod.grad, th))
    fr = FitResult(N=D.N, Ti=D.Ti, omitted=omitted, converged=conv, iterations=it, ll=float(ll), ll_c=llc)
    fr.lib = "Hausman, Hall & Griliches (1984) 베타 분포 산포 확률효과 우도, Newton–Raphson"
    zc = stats.norm.ppf(0.5 + o["ciLevel"] / 200.0)
    fr.names = _main_names([D.covs[j] for j in kept], const)
    fr.anc = ["/ln_r", "/ln_s"]
    fr.b, fr.V = th, V
    for nm, j in (("r", -2), ("s", -1)):
        v, sv = float(th[j]), math.sqrt(max(V[j, j], 0.0))
        fr.derived.append(_derived(nm, math.exp(v), math.exp(v) * sv, math.exp(v - zc * sv), math.exp(v + zc * sv)))
    fr.lr_label = "pooled"
    fr.chi2_lr = abs(2.0 * (ll - llc))
    w = _wald(th, V, range(len(kept)))
    fr.chi2 = (w[0], w[1], w[2], "Wald") if w else None
    fr.extra.update({"const": const, "kind": "count"})
    return fr


def fit_gee(D: NLData, o: Dict[str, Any]) -> FitResult:
    """Population-averaged model — statsmodels GEE with Stata xtgee conventions."""
    import statsmodels.api as sm
    from statsmodels.genmod import cov_struct as cs
    fam = o["family"]
    corr = o["paCorr"]
    fr = FitResult()
    E = D
    if corr == "ar1":
        same = np.r_[False, E.g[1:] == E.g[:-1]]
        gap = same & (np.r_[1, np.diff(E.tpos)] != 1)
        single = E.Ti < 2
        if gap.any() and not o["force"]:
            bad = np.zeros(E.n, bool)
            bad[E.g[gap]] = True
            fr.dropped.append(f"시점 간격이 일정하지 않은 {int(bad.sum())}개 집단 제외 (Stata: observations not equally spaced; "
                              "force 옵션으로 포함 가능)")
            E = _drop_groups(E, ~bad)
            single = E.Ti < 2
        if single.any():
            fr.dropped.append(f"관측치 2개 미만인 {int(single.sum())}개 집단 제외 (Stata: not possible to estimate correlations "
                              "for those groups)")
            E = _drop_groups(E, ~single)
        if E.N == 0:
            raise ValueError("pa corr(ar 1): 추정에 쓸 집단 없음")
    if o["paCorr"] == "ar1" and o["force"]:
        st = _starts(E.Ti)
        pos = np.arange(E.N) - np.repeat(st, E.Ti)
    else:
        pos = E.tpos - E.tpos.min()
    const = o["constant"]
    kept, omitted = _collinear(E.X, E.covs, const)
    X = E.X[:, kept]
    Z = np.column_stack([X, np.ones(E.N)]) if const else X
    if fam in ("logit", "probit", "cloglog"):
        link = {"logit": sm.families.links.Logit(), "probit": sm.families.links.Probit(),
                "cloglog": sm.families.links.CLogLog()}[fam]
        family = sm.families.Binomial(link=link)
    elif fam == "poisson":
        family = sm.families.Poisson()
    else:
        family = sm.families.NegativeBinomial(alpha=1.0)
    st = _starts(E.Ti)
    ends = np.r_[st[1:], E.N]
    tpos_li = [pos[a:b] for a, b in zip(st, ends)]
    if corr == "exchangeable":
        struct = cs.Exchangeable()
    elif corr == "independent":
        struct = cs.Independence()
    elif corr == "ar1":
        struct = _stata_ar1_struct(tpos_li)
    else:
        struct = _stata_unstructured_struct(tpos_li, int(pos.max()) + 1)
    kw = {"offset": E.off} if np.any(E.off != 0) else {}
    mod = sm.GEE(E.y, Z, groups=E.g, family=family, cov_struct=struct, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(maxiter=500, ctol=1e-11, ddof_scale=0, scale=None,
                      cov_type="naive" if o["covType"] == "oim" else "robust")
    V = np.asarray(res.cov_params())
    if o["covType"] != "oim":
        V = V * (E.n / (E.n - 1.0))
        fr.vce_label = f"반강건 (패널 군집 {E.n}개)" if fam in ("probit", "cloglog", "nbreg") else f"강건 (패널 군집 {E.n}개)"
        fr.n_clusters = E.n
    else:
        fr.vce_label = "일반 (conventional)"
    fr.N, fr.Ti = E.N, E.Ti
    fr.names = _main_names([E.covs[j] for j in kept], const)
    fr.omitted = omitted
    fr.b, fr.V = np.asarray(res.params, float), V
    fr.converged = bool(getattr(res, "converged", True))
    fr.lib = "statsmodels GEE"
    w = _wald(fr.b, V, range(len(kept)))
    fr.chi2 = (w[0], w[1], w[2], "Wald") if w else None
    fr.extra.update({"const": const, "kind": FAMILIES[fam]["kind"], "corr": corr,
                     "dep_params": mod.cov_struct.dep_params,
                     "Z": Z, "off": E.off, "family": family, "scale": float(res.scale),
                     "T": int(pos.max()) + 1})
    return fr


# --------------------------------------------------------------------------------------------- post-estimation
def hausman(fe: FitResult, re_: FitResult) -> Dict[str, Any]:
    """Stata hausman fe re: common slopes, generalized inverse of V_FE - V_RE, df = rank."""
    common = [n for n in fe.names if n in re_.names and n != "_cons"]
    if not common:
        raise ValueError("Hausman 검정: 두 모형에 공통 계수 없음")
    i_fe = [fe.names.index(n) for n in common]
    i_re = [re_.names.index(n) for n in common]
    d = fe.b[i_fe] - re_.b[i_re]
    Vd = fe.V[np.ix_(i_fe, i_fe)] - re_.V[np.ix_(i_re, i_re)]
    Vd = (Vd + Vd.T) / 2.0
    ev = np.linalg.eigvalsh(Vd)
    tol = 1e-12 * max(1e-300, float(np.max(np.abs(ev))))
    rank = int(np.sum(np.abs(ev) > tol))
    chi2 = float(d @ np.linalg.pinv(Vd) @ d)
    return {"terms": common, "b_fe": fe.b[i_fe], "b_re": re_.b[i_re], "diff": d,
            "se": np.sqrt(np.clip(np.diag(Vd), 0, None)), "chi2": chi2, "df": rank,
            "p": float(stats.chi2.sf(chi2, rank)) if chi2 >= 0 else float("nan"), "pos_def": bool(np.all(ev > tol))}


def average_marginal_effects(fr: FitResult, o: Dict[str, Any], D: NLData) -> Dict[str, Any]:
    """margins, dydx(*) with delta-method SEs: PA -> population-averaged mean h(eta); RE binary -> predict(pr),
    P = sum_m (w_m / sqrt(pi)) F(eta + sqrt(2) sigma_u a_m) with intpoints Gauss–Hermite points."""
    fam, model = o["family"], o["model"]
    kind = FAMILIES[fam]["kind"]
    b, V = fr.b, fr.V
    nk = len([n for n in fr.names if n != "_cons"])
    Z = fr.extra["Z"]
    k = Z.shape[1]
    beta = b[:k]
    if model == "pa":
        off = fr.extra["off"]
        eta = Z @ beta + off
        if kind == "binary":
            h1, h2 = _Binary(np.zeros(1), fam).mean_derivs(eta)
        else:
            h1 = h2 = np.exp(np.minimum(eta, 700.0))
        dl = None
        expr = ("Pr(" + D.dep + ")" if kind == "binary" else "E(" + D.dep + ") (predict(mu), offset 포함)")
    else:
        lns2 = float(b[-1])
        s = math.exp(0.5 * lns2)
        a, w = np.polynomial.hermite.hermgauss(o["intPoints"])
        z, om = SQ2 * a, w / math.sqrt(math.pi)
        E = (Z @ beta + D.off)[:, None] + s * z[None, :]
        d1, d2 = _Binary(np.zeros(1), fam).mean_derivs(E)
        h1, h2 = d1 @ om, d2 @ om
        dl = (d2 * z[None, :] * s * 0.5) @ om
        expr = f"Pr({D.dep}=1), predict(pr): u_i에 대해 적분한 확률"
    m1 = float(np.mean(h1))
    ame = m1 * beta[:nk]
    J = np.zeros((nk, len(b)))
    mz = (h2[:, None] * Z).mean(axis=0)
    for j in range(nk):
        J[j, :k] = mz * beta[j]
        J[j, j] += m1
        if dl is not None:
            J[j, -1] = float(np.mean(dl)) * beta[j]
    Va = J @ V @ J.T
    return {"terms": fr.names[:nk], "ame": ame, "se": np.sqrt(np.clip(np.diag(Va), 0, None)), "V": Va, "expr": expr}


# --------------------------------------------------------------------------------------------- analysis
def _pstr(p: float) -> str:
    if p is None or p != p:
        return "= —"
    return "< .001" if p < 0.001 else f"= {p:.3f}".replace("0.", ".", 1)


_EFORM_LABEL = {"or": "OR (오즈비)", "irr": "IRR (발생률비)", "eform": "exp(b)"}


class PanelNonlinear(BaseAnalysis):
    """Nonlinear panel-data models with Stata xt conventions."""

    def __init__(self, data: pd.DataFrame, o: Dict[str, Any]):
        fam, model = o["family"], o["model"]
        spec = FAMILIES[fam]
        cmd = _cmd_label(fam, model, o["reDist"], o.get("clogit", False))
        lab = MODEL_LABEL[model] + (" (정규분포 확률효과)" if fam == "poisson" and model == "re" and o["reDist"] == "normal"
                                    else "")
        super().__init__(name="panelNonlinear", title=f"비선형 패널 — {spec['ko']} {lab} ({cmd})", dataset=data,
                         options=dict(o))
        self.cmd = cmd
        self.syntax_python = python_call(o)
        self.syntax_r = "\n".join(r_code("panelNonlinear", o) or [])
        self.run()

    # ------------------------------------------------------------------ helpers
    def _safe(self, fn, label: str):
        try:
            fn()
        except Exception as e:   # an optional diagnostic must not hide the estimates
            t = ResultsTable(name=f"diag_error_{len(self.tables)}", title=f"{label} — 계산 불가")
            t.add_column("msg", "사유", col_type="text")
            t.add_row({"msg": str(e)})
            self.add_table(t)

    def _estimate(self, D: NLData, o: Dict[str, Any]) -> FitResult:
        fam, model = o["family"], o["model"]
        if model == "pa":
            return fit_gee(D, o)
        if model == "fe":
            return {"logit": fit_clogit, "poisson": fit_cpoisson, "nbreg": fit_nb_fe}[fam](D, o)
        if fam == "poisson" and o["reDist"] == "gamma":
            return fit_pois_gamma(D, o)
        if fam == "nbreg":
            return fit_nb_re(D, o)
        return fit_re_quad(D, o)

    # ------------------------------------------------------------------ run
    def run(self):
        o = self.options
        D = NLData(self.dataset, o)
        self.data = D
        fit = self._estimate(D, o)
        self.fit = fit
        self._summary_table(D, fit)
        self._coef_table(D, fit)
        if fit.anc or fit.derived:
            self._anc_table(fit)
        if getattr(fit, "chi2_lr", None) is not None and fit.ll_c is not None:
            self._lr_table(fit)
        if o["model"] == "pa":
            self._pa_corr_table(fit)
        if o["hausman"]:
            self._safe(lambda: self._hausman_table(D, fit), "Hausman 검정")
        kind = FAMILIES[o["family"]]["kind"]
        if o["margins"] and (o["model"] == "pa" or (o["model"] == "re" and kind == "binary")):
            self._safe(lambda: self._ame_table(D, fit), "평균 한계효과")
        self._narrative(D, fit)

    # ------------------------------------------------------------------ tables
    def _summary_table(self, D: NLData, fit: FitResult):
        o = self.options
        t = ResultsTable(name="summary", title=f"모형 요약 ({self.cmd})")
        t.add_column("metric", "항목", col_type="text")
        t.add_column("val", "값", col_type="number")
        Ti = fit.Ti
        rows: List[Tuple[str, Any, str]] = [("관측치 수 (N)", fit.N, "integer"), ("집단 수", len(Ti), "integer"),
                                            ("집단당 관측치 최소", int(Ti.min()), "integer"),
                                            ("집단당 관측치 평균", float(Ti.mean()), ""),
                                            ("집단당 관측치 최대", int(Ti.max()), "integer")]
        cz = fit.extra.get("censor")
        if cz and o["family"] == "tobit":
            rows += [("비검열 관측치", fit.N - cz["n_left"] - cz["n_right"], "integer"),
                     ("좌측 검열 관측치", cz["n_left"], "integer"), ("우측 검열 관측치", cz["n_right"], "integer")]
        elif cz:
            rows += [("점 관측치 (비검열)", cz["n_point"], "integer"), ("좌측 검열 관측치", cz["n_left"], "integer"),
                     ("우측 검열 관측치", cz["n_right"], "integer"), ("구간 검열 관측치", cz["n_interval"], "integer")]
        if o["model"] != "pa":
            rows.append(("로그우도" + (" (의사우도)" if o["covType"] != "oim" else ""), fit.ll, ""))
        if fit.chi2 is not None:
            rows += [(f"{fit.chi2[3]} χ²({fit.chi2[1]})", fit.chi2[0], ""), ("Prob > χ²", fit.chi2[2], "pvalue")]
        quad = "int" in fit.extra
        if quad:
            rows.append(("적분점 수", o["intPoints"], "integer"))
        if o["model"] == "pa":
            rows.append(("척도 모수 (scale)", 1.0, ""))
        for lab, v, fmt in rows:
            r = {"metric": lab, "val": v}
            if fmt:
                r["_rowfmt"] = fmt
            t.add_row(r)
        t.add_note(f"추정: {FAMILIES[o['family']]['ko']} {MODEL_LABEL[o['model']]} — Stata {self.cmd}")
        t.add_note("계산 방식: " + _source_note(o))
        if quad:
            t.add_note("적분 방법: " + ("평균–분산 적응 Gauss–Hermite 구적법 (Stata intmethod(mvaghermite))"
                                    if o["intMethod"] == "mvaghermite"
                                    else "비적응 Gauss–Hermite 구적법 (Stata intmethod(ghermite))"))
        if o["model"] == "pa":
            t.add_note("Stata xtgee 기본값: 이산 분포의 척도 모수 1, 작업 상관은 Pearson 잔차의 적률 추정")
            if o["family"] == "nbreg":
                t.add_note("음이항 계열의 alpha는 1로 고정 (Stata xtnbreg, pa = xtgee family(nbinomial) link(log); "
                           "결과는 alpha 값에 조건부)")
        if o["family"] == "tobit":
            lo = "−∞" if o["ll"] is None else ("min" if o["ll"] is True else _stata_lim(o["ll"]))
            hi = "+∞" if o["ul"] is None else ("max" if o["ul"] is True else _stata_lim(o["ul"]))
            t.add_note(f"검열 한계: 하한 {lo}, 상한 {hi} (Stata: 하한 이하 좌측 검열, 상한 이상 우측 검열)")
        if o["family"] == "intreg":
            t.add_note(f"구간 자료: {o['dep']} 결측 = 좌측 검열, {o['depUpper']} 결측 = 우측 검열, 두 값이 같으면 점 관측")
        for d in fit.dropped:
            t.add_note("제외: " + d)
        if o.get("ignored"):
            t.add_note("이 모형에 해당하지 않아 적용하지 않은 옵션: " + "; ".join(o["ignored"]))
        if o.get("clusterDefault"):
            t.add_note(f"군집 변수 미지정: 패널 변수({o['clusterVar']})로 군집 (Stata vce(cluster {o['clusterVar']}))")
        if not fit.converged:
            t.add_note("최대화 수렴 기준 미충족 — 결과 해석 주의 (Stata: convergence not achieved)")
        if D.time_note:
            t.add_note(D.time_note)
        if D.n_dropped:
            t.add_note(f"결측으로 제외된 행: {D.n_dropped}")
        self.add_table(t)
        ms: Dict[str, Any] = {"N": int(fit.N), "Groups": int(len(Ti))}
        if o["model"] != "pa":
            ms["LL"] = float(fit.ll)
        if fit.chi2 is not None:
            ms["Chi2"] = (float(fit.chi2[0]), float(fit.chi2[2]))
        for dv in fit.derived:
            ms[dv["name"]] = float(dv["est"])
        self.model_stats = ms

    def _coef_table(self, D: NLData, fit: FitResult):
        o = self.options
        k = fit.k_main
        b, se = fit.b[:k], fit.se[:k]
        zc = stats.norm.ppf(0.5 + o["ciLevel"] / 200.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(se > 0, b / se, np.nan)
        p = 2 * stats.norm.sf(np.abs(z))
        lv = f"{o['ciLevel']:g}%"
        t = ResultsTable(name="coef", title="계수 추정치 (Coefficients)")
        t.add_column("var", "변수", col_type="text")
        t.add_column("b", "추정치", col_type="number")
        t.add_column("se", "표준오차", col_type="number")
        t.add_column("z", "z", col_type="number")
        t.add_column("p", "p", col_type="number", fmt="pvalue")
        t.add_column("cil", f"{lv} CI 하한", col_type="number")
        t.add_column("ciu", f"{lv} CI 상한", col_type="number")
        ef = FAMILIES[o["family"]]["eform"] if o["eform"] else None
        if ef:
            el = _EFORM_LABEL[ef]
            t.add_column("eb", el, col_type="number")
            t.add_column("ese", f"{el.split()[0]} 표준오차", col_type="number")
            t.add_column("ecil", f"{el.split()[0]} {lv} CI 하한", col_type="number")
            t.add_column("eciu", f"{el.split()[0]} {lv} CI 상한", col_type="number")
        for i, nm in enumerate(fit.names):
            r = {"var": nm, "b": float(b[i]), "se": float(se[i]), "z": float(z[i]), "p": float(p[i]),
                 "cil": float(b[i] - zc * se[i]), "ciu": float(b[i] + zc * se[i])}
            if ef:
                e = math.exp(b[i])
                r.update({"eb": e, "ese": e * float(se[i]), "ecil": math.exp(r["cil"]), "eciu": math.exp(r["ciu"])})
            t.add_row(r)
        t.add_note(f"표준오차: {fit.vce_label}" + ("" if o["covType"] == "oim" or o["model"] != "pa" else ""))
        if ef:
            t.add_note(f"{_EFORM_LABEL[ef]}: exp(b), 표준오차는 델타 방법 exp(b)·se(b), 신뢰구간은 b 구간의 지수 변환 "
                       f"(Stata {ef} 옵션)")
            if "_cons" in fit.names and o["model"] == "re":
                t.add_note("_cons: 확률효과 0 조건의 기준값 (Stata: conditional on zero random effects)")
        if D.off_label:
            t.add_note(f"{D.off_label}: 계수 1로 고정 ({'exposure' if o.get('exposure') else 'offset'})")
        for v in fit.omitted:
            if not any(v in n for n in fit.notes):
                t.add_note(f"{v}: 공선성으로 제외 (Stata omitted)")
        for n_ in fit.notes:
            t.add_note(n_)
        if o["model"] == "fe" and o["family"] == "logit" and o["margins"]:
            t.add_note("평균 한계효과 미보고: 조건부 고정효과 로짓의 확률은 추정하지 않는 u_i에 의존 "
                       "(Stata margins 기본값 predict(pu0)는 u_i = 0 가정) — 오즈비(or) 해석 권장")
        if o["model"] == "fe" and o["family"] == "nbreg":
            t.add_note("xtnbreg, fe: 개체별 산포 모수를 조건부로 제거하는 모형 — 상수항·시간 불변 변수 추정 가능 "
                       "(Hausman, Hall & Griliches 1984)")
        t.mark_coefficients(estimate="b", se="se", p="p", stat="z", ci=("cil", "ciu"),
                            exp_estimate="eb" if ef else None, label="var")
        self.add_table(t)

    def _anc_table(self, fit: FitResult):
        o = self.options
        k = fit.k_main
        zc = stats.norm.ppf(0.5 + o["ciLevel"] / 200.0)
        t = ResultsTable(name="anc", title="보조 모수 (Ancillary parameters)")
        t.add_column("param", "모수", col_type="text")
        t.add_column("est", "추정치", col_type="number")
        t.add_column("se", "표준오차", col_type="number")
        t.add_column("z", "z", col_type="number")
        t.add_column("p", "p", col_type="number", fmt="pvalue")
        t.add_column("cil", f"{o['ciLevel']:g}% CI 하한", col_type="number")
        t.add_column("ciu", f"{o['ciLevel']:g}% CI 상한", col_type="number")
        show_z = o["family"] in ("tobit", "intreg")       # Stata shows z, p for /sigma_u and /sigma_e only
        log_ci = fit.extra.get("log_ci", {})
        for j, nm in enumerate(fit.anc):
            est, se = float(fit.b[k + j]), float(fit.se[k + j])
            r = {"param": nm, "est": est, "se": se, "z": None, "p": None,
                 "cil": est - zc * se, "ciu": est + zc * se}
            if nm in log_ci:
                r["cil"], r["ciu"] = log_ci[nm]
            if show_z and se > 0:
                r["z"] = est / se
                r["p"] = float(2 * stats.norm.sf(abs(est / se)))
            t.add_row(r)
        for dv in fit.derived:
            t.add_row({"param": dv["name"], "est": dv["est"], "se": dv["se"], "z": None, "p": None,
                       "cil": dv["lo"], "ciu": dv["hi"]})
        fam = o["family"]
        if fam in ("logit", "probit", "cloglog"):
            c = {"logit": "π²/3", "probit": "1", "cloglog": "π²/6"}[fam]
            t.add_note(f"sigma_u = exp(lnsig2u/2), rho = sigma_u² / (sigma_u² + {c}); 신뢰구간은 /lnsig2u 구간의 변환")
        elif fam == "poisson" and o["model"] == "re" and o["reDist"] == "normal":
            t.add_note("sigma_u = exp(lnsig2u/2); 신뢰구간은 /lnsig2u 구간의 변환")
        elif fam == "poisson":
            t.add_note("alpha = exp(lnalpha): 감마 분포 확률효과의 분산 (Var(ν_i) = alpha)")
        elif fam == "nbreg":
            t.add_note("1/(1 + δ_i) ~ Beta(r, s): 패널별 산포의 확률효과 분포 (Stata xtnbreg, re)")
        elif fam in ("tobit", "intreg"):
            t.add_note("rho = sigma_u² / (sigma_u² + sigma_e²); rho 신뢰구간은 프로빗 척도 변환 (Stata diparm ci(probit))")
        else:
            t.add_note("/cut#: 잠재변수의 절단점 (상수항 없음), /sigma2_u: 확률효과 분산 (신뢰구간은 로그 척도 변환)")
        self.add_table(t)

    def _lr_table(self, fit: FitResult):
        o = self.options
        t = ResultsTable(name="lr_pooled", title="통합 모형 대비 LR 검정 (LR test vs. pooled)")
        t.add_column("test", "검정", col_type="text")
        t.add_column("chi2", "chibar2(01)", col_type="number")
        t.add_column("p", "Prob ≥ chibar2", col_type="number", fmt="pvalue")
        t.add_column("ll_c", "통합 모형 로그우도", col_type="number")
        lab = {"rho=0": "LR test of rho=0", "sigma_u=0": "LR test of sigma_u=0", "alpha=0": "LR test of alpha=0",
               "pooled": "LR test vs. pooled", "ologit": "LR test vs. ologit model",
               "oprobit": "LR test vs. oprobit model"}[fit.lr_label]
        x = float(fit.chi2_lr)
        pv = 1.0 if x <= 0 else 0.5 * float(stats.chi2.sf(x, 1))
        t.add_row({"test": lab, "chi2": x, "p": pv, "ll_c": float(fit.ll_c)})
        pooled = {"logit": "logit", "probit": "probit", "cloglog": "cloglog", "poisson": "poisson",
                  "nbreg": "nbreg, dispersion(constant)", "tobit": "tobit", "intreg": "intreg", "ologit": "ologit",
                  "oprobit": "oprobit"}[o["family"]]
        t.add_note(f"통합(pooled) 모형: Stata {pooled}에 해당; 귀무가설의 모수가 경계(0)에 있어 χ²(0)과 χ²(1)의 50:50 혼합 분포")
        if o["covType"] != "oim":
            t.add_note("강건 표준오차 지정 시 Stata는 이 검정을 출력하지 않음 (의사우도 기준 LR 검정은 근사)")
        self.add_table(t)
        self.model_stats["LR_pooled"] = (x, pv)

    def _pa_corr_table(self, fit: FitResult):
        o = self.options
        corr = fit.extra["corr"]
        a = fit.extra["dep_params"]
        lab = {"exchangeable": "exchangeable", "independent": "independent", "ar1": "ar 1",
               "unstructured": "unstructured"}
        t = ResultsTable(name="pa_corr", title=f"작업 상관 구조 (corr({lab[corr]}))")
        t.add_column("item", "항목", col_type="text")
        t.add_column("val", "값", col_type="number")
        if corr == "independent":
            t.add_row({"item": "상관 (독립 가정)", "val": 0.0})
        elif corr in ("exchangeable", "ar1"):
            t.add_row({"item": "α (동일 개체 내 상관)" if corr == "exchangeable" else "ρ (AR(1) 1차 상관)", "val": float(a)})
        else:
            m = a.shape[0]
            for i in range(m):
                for j in range(i + 1, m):
                    t.add_row({"item": f"R[{i + 1},{j + 1}]", "val": float(a[i, j])})
            t.add_note("행·열 번호: 시점 순서 (Stata e(R)과 같은 배치)")
        t.add_note("Pearson 잔차 적률 추정 ([XT] xtgee Methods and formulas), 척도 모수 1")
        self.add_table(t)

    def _hausman_table(self, D: NLData, fit: FitResult):
        o = self.options
        other = dict(o, model="re" if o["model"] == "fe" else "fe", covType="oim", clusterVar=None, hausman=False,
                     clogit=False)
        base = fit
        if o["covType"] != "oim":
            base = self._estimate(D, dict(o, covType="oim", clusterVar=None))
        alt = self._estimate(D, other)
        fe, re_ = (base, alt) if o["model"] == "fe" else (alt, base)
        h = hausman(fe, re_)
        cmd = FAMILIES[o["family"]]["cmd"]
        t = ResultsTable(name="hausman", title="Hausman 검정 (FE vs RE)")
        t.add_column("chi2", "χ²", col_type="number")
        t.add_column("df", "df", col_type="integer")
        t.add_column("p", "p", col_type="number", fmt="pvalue")
        t.add_column("decision", "판단", col_type="text")
        if h["chi2"] < 0:
            dec = "χ² < 0: 점근 가정 미충족 (Stata 안내: suest 등 대안 검정)"
        else:
            dec = "고정효과 선호 (p < .05)" if h["p"] < 0.05 else "확률효과 일관성 기각 안 됨 (p ≥ .05)"
        t.add_row({"chi2": h["chi2"], "df": h["df"], "p": h["p"], "decision": dec})
        re_lab = ("re normal" if (o["family"] == "poisson" and o["reDist"] == "normal") else "re")
        t.add_note(f"Stata: {cmd} ..., fe → estimates store fe; {cmd} ..., {re_lab} → estimates store re; "
                   "hausman fe re")
        t.add_note("(b − B)'(V_b − V_B)⁻¹(b − B), 일반화 역행렬, df = rank(V_b − V_B); b = 고정효과(일치), B = 확률효과(효율)")
        if not h["pos_def"]:
            t.add_note("V_b − V_B가 양정치 아님 (Stata: V_b-V_B is not positive definite)")
        if o["covType"] != "oim":
            t.add_note("두 모형 모두 OIM 분산으로 재추정 (Stata hausman은 강건 분산 미지원)")
        self.add_table(t)
        ct = ResultsTable(name="hausman_coef", title="Hausman 검정 계수 비교")
        ct.add_column("var", "변수", col_type="text")
        ct.add_column("b", "(b) 고정효과", col_type="number")
        ct.add_column("B", "(B) 확률효과", col_type="number")
        ct.add_column("d", "b − B", col_type="number")
        ct.add_column("se", "sqrt(diag(V_b − V_B))", col_type="number")
        for nm, bf, br, dd, sd in zip(h["terms"], h["b_fe"], h["b_re"], h["diff"], h["se"]):
            ct.add_row({"var": nm, "b": float(bf), "B": float(br), "d": float(dd), "se": float(sd)})
        self.add_table(ct)
        self.model_stats["Hausman"] = (h["chi2"], h["p"])

    def _ame_table(self, D: NLData, fit: FitResult):
        o = self.options
        r = average_marginal_effects(fit, o, D)
        zc = stats.norm.ppf(0.5 + o["ciLevel"] / 200.0)
        t = ResultsTable(name="ame", title="평균 한계효과 (Average marginal effects, dy/dx)")
        t.add_column("var", "변수", col_type="text")
        t.add_column("dydx", "dy/dx", col_type="number")
        t.add_column("se", "표준오차 (델타 방법)", col_type="number")
        t.add_column("z", "z", col_type="number")
        t.add_column("p", "p", col_type="number", fmt="pvalue")
        t.add_column("cil", f"{o['ciLevel']:g}% CI 하한", col_type="number")
        t.add_column("ciu", f"{o['ciLevel']:g}% CI 상한", col_type="number")
        for nm, a, s in zip(r["terms"], r["ame"], r["se"]):
            z = a / s if s > 0 else float("nan")
            t.add_row({"var": nm, "dydx": float(a), "se": float(s), "z": float(z),
                       "p": float(2 * stats.norm.sf(abs(z))), "cil": float(a - zc * s), "ciu": float(a + zc * s)})
        t.add_note(f"예측값: {r['expr']} — Stata margins, dydx(*)")
        t.add_note("모든 설명변수를 연속 변수로 미분 (요인 변수 i. 지정이 없는 Stata margins와 동일)")
        if o["model"] == "re":
            t.add_note(f"적분: Gauss–Hermite {o['intPoints']}점 (Stata 19 xtlogit·xtprobit·xtcloglog, re의 margins 기본값 "
                       "predict(pr); predict(pu0)는 u_i = 0 가정)")
        else:
            t.add_note("모집단 평균 모형의 평균 반응 h(xb) 기준 (Stata predict(mu))")
        t.add_note(f"표준오차: 델타 방법, 모형 분산 = {fit.vce_label}")
        t.mark_coefficients(estimate="dydx", se="se", p="p", stat="z", ci=("cil", "ciu"), label="var")
        self.add_table(t)

    def _narrative(self, D: NLData, fit: FitResult):
        o = self.options
        txt = (f"{FAMILIES[o['family']]['ko']} {MODEL_LABEL[o['model']]} 패널 모형 (개체 {len(fit.Ti)}개, "
               f"관측치 {fit.N:,}개)")
        if fit.chi2 is not None:
            txt += f", {fit.chi2[3]} χ²({fit.chi2[1]}) = {fit.chi2[0]:.2f}, p {_pstr(fit.chi2[2])}"
        for dv in fit.derived:
            if dv["name"] in ("rho", "alpha"):
                sym = "ρ" if dv["name"] == "rho" else "α"
                txt += f"; {sym} = {dv['est']:.3f}"
        lr = self.model_stats.get("LR_pooled")
        if lr:
            txt += f"; 통합 모형 대비 LR χ̄²(01) = {lr[0]:.2f}, p {_pstr(lr[1])}"
        h = self.model_stats.get("Hausman")
        if h:
            txt += f"; Hausman χ² = {h[0]:.2f}, p {_pstr(h[1])}"
        self.apa_narrative = txt + "."


def _source_note(o: Dict[str, Any]) -> str:
    fam, model = o["family"], o["model"]
    if model == "pa":
        corr = {"exchangeable": "Exchangeable", "independent": "Independence",
                "ar1": "AR(1) (Stata 적률 추정)", "unstructured": "비구조 (Stata 적률 추정)"}[o["paCorr"]]
        return f"라이브러리: statsmodels GEE ({corr}); Liang & Zeger (1986), Stata [XT] xtgee Methods and formulas"
    if model == "fe":
        return {"logit": "라이브러리: statsmodels ConditionalLogit; Stata [R] clogit",
                "poisson": "라이브러리: statsmodels ConditionalPoisson (+ 다항 상수항); "
                           "Stata [XT] xtpoisson Methods and formulas",
                "nbreg": "Hausman, Hall & Griliches (1984), Stata [XT] xtnbreg Methods and formulas"}[fam]
    if fam == "poisson" and o["reDist"] == "gamma":
        return "Hausman, Hall & Griliches (1984), Stata [XT] xtpoisson Methods and formulas (감마 확률효과)"
    if fam == "nbreg":
        return "Hausman, Hall & Griliches (1984), Stata [XT] xtnbreg Methods and formulas (베타 분포 산포)"
    cmd = FAMILIES[fam]["cmd"]
    return (f"Butler & Moffitt (1982) Gauss–Hermite 구적법, Naylor & Smith (1982) 적응 구적법; Stata [XT] {cmd} "
            "Methods and formulas; 최대화 Newton–Raphson, OIM = 해석적 점수의 수치 미분")


# --------------------------------------------------------------------------------------------- registry glue
def _preset_of(name: Optional[str]) -> Optional[Tuple[str, Optional[str]]]:
    if not name:
        return None
    return ALIAS_PRESETS.get(str(name).strip().lower())


def dispatch(t: str, df: pd.DataFrame, params: Dict[str, Any]):
    if t != "panelNonlinear":
        return None
    p = dict(params or {})
    preset = _preset_of(p.pop("_alias", None) or p.pop("_analysis", None))
    o = normalize_options(p, preset)
    return PanelNonlinear(df, o)


def _non_default(o: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, dv in _DEFAULTS.items():
        v = o.get(k, dv)
        if k == "hausman":
            hd = o["family"] in ("logit", "poisson") and o["model"] in ("fe", "re")
            if bool(v) != hd:
                out[k] = v
            continue
        if k == "reDist" and not (o["family"] == "poisson" and o["model"] == "re"):
            continue
        if v != dv and not (v in (None, "", []) and dv in (None, "", [])):
            out[k] = v
    return out


def python_call(o: Dict[str, Any]) -> str:
    kw = [f"entity={o['entity']!r}"]
    if o.get("time"):
        kw.append(f"time={o['time']!r}")
    kw.append(f"dep={o['dep']!r}")
    if o.get("depUpper"):
        kw.append(f"depUpper={o['depUpper']!r}")
    kw += [f"covs={list(o['covs'])!r}", f"family={o['family']!r}", f"model={o['model']!r}"]
    if o.get("clogit"):
        kw.append("clogit=True")
    for k, v in _non_default(o).items():
        if k in ("time", "depUpper", "model"):
            continue
        kw.append(f"{k}={v!r}")
    return f"res = sess.run('panelNonlinear', {', '.join(kw)})"


def _resolve_t(t: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    low = str(t).strip()
    if low != "panelNonlinear" and low.lower() not in ALIAS_PRESETS and low.lower() not in ("panelnonlinear",
                                                                                            "panel_nonlinear"):
        return None
    p = {k: v for k, v in (params or {}).items() if k not in ("_alias", "_analysis")}
    try:
        return normalize_options(p, _preset_of(low))
    except ValueError:
        return None


# --------------------------------------------------------------------------------------------- code export
def _is_quad(o: Dict[str, Any]) -> bool:
    return o["model"] == "re" and (o["family"] in ("logit", "probit", "cloglog", "tobit", "intreg", "ologit", "oprobit")
                                   or (o["family"] == "poisson" and o["reDist"] == "normal"))


def _stata_lim(v: Any) -> str:
    if v is True:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _stata_cmd(o: Dict[str, Any], model: Optional[str] = None, vce: bool = True) -> str:
    fam = o["family"]
    model = model or o["model"]
    cmd = "clogit" if (o.get("clogit") and model == "fe") else FAMILIES[fam]["cmd"]
    dep = o["dep"] + (f" {o['depUpper']}" if o.get("depUpper") else "")
    opts: List[str] = []
    if cmd == "clogit":
        opts.append(f"group({o['entity']})")
    elif fam not in ("tobit", "intreg", "ologit", "oprobit"):
        opts.append(model)
        if fam == "poisson" and model == "re" and o["reDist"] == "normal":
            opts.append("normal")
    if fam == "tobit":
        if o.get("ll") is not None:
            opts.append(f"ll({_stata_lim(o['ll'])})" if o["ll"] is not True else "ll")
        if o.get("ul") is not None:
            opts.append(f"ul({_stata_lim(o['ul'])})" if o["ul"] is not True else "ul")
    if o.get("exposure"):
        opts.append(f"exposure({o['exposure']})")
    if o.get("offset"):
        opts.append(f"offset({o['offset']})")
    if not o.get("constant", True) and model != "fe":
        opts.append("noconstant")
    q = dict(o, model=model)
    if _is_quad(q):
        if o["intMethod"] != "mvaghermite":
            opts.append(f"intmethod({o['intMethod']})")
        if o["intPoints"] != 12:
            opts.append(f"intpoints({o['intPoints']})")
    if model == "pa":
        if o["paCorr"] != "exchangeable":
            opts.append("corr(" + {"ar1": "ar 1"}.get(o["paCorr"], o["paCorr"]) + ")")
        if o.get("force"):
            opts.append("force")
    if vce and o["covType"] == "robust":
        opts.append("vce(robust)")
    elif vce and o["covType"] == "cluster":
        opts.append(f"vce(cluster {o['clusterVar']})")
    if o.get("eform"):
        opts.append({"or": "or", "irr": "irr", "eform": "eform"}[FAMILIES[fam]["eform"]])
    if o.get("ciLevel", 95.0) != 95.0:
        opts.append(f"level({o['ciLevel']:g})")
    return f"{cmd} {dep} {' '.join(o['covs'])}".rstrip() + (", " + " ".join(opts) if opts else "")


def stata_code(t: str, params: Dict[str, Any]) -> Optional[List[str]]:
    o = _resolve_t(t, params)
    if o is None:
        return None
    e, tm = o["entity"], o.get("time")
    if not e:
        out: List[str] = ["* xtset 개체변수 시점변수 먼저 실행"]
    elif o.get("clogit") and not tm and not o["hausman"]:
        out = []
    elif tm:
        out = xtset_lines(e, tm)
    else:
        out = [f"capture confirm numeric variable {e}", f"if _rc encode {e}, generate(_panel_id)",
               "if _rc xtset _panel_id", f"else xtset {e}"]
    out.append(_stata_cmd(o))
    if o["family"] == "poisson" and o["model"] == "re" and o["reDist"] == "gamma" and o["covType"] != "oim":
        out.append("* Stata xtpoisson, re vce(robust): /lnalpha 패널 점수를 관측치마다 합산(T_i배) — OpenStat은 패널 점수 1회 "
                   "(표준오차·Wald χ² 차이)")
    kind = FAMILIES[o["family"]]["kind"]
    if o["margins"] and (o["model"] == "pa" or (o["model"] == "re" and kind == "binary")):
        out.append("margins, dydx(*)")
    if o["hausman"]:
        q = dict(o, covType="oim", eform=False)
        out += ["quietly " + _stata_cmd(dict(q, clogit=False), "fe"), "estimates store fe",
                "quietly " + _stata_cmd(q, "re"), "estimates store re", "hausman fe re"]
    return out


def r_code(t: str, params: Dict[str, Any]) -> Optional[List[str]]:
    o = _resolve_t(t, params)
    if o is None:
        return None
    fam, model = o["family"], o["model"]
    e, tm, y = o["entity"], o.get("time"), o["dep"]
    xs = " + ".join(o["covs"]) if o["covs"] else "1"
    q = lambda s_: '"' + str(s_) + '"'
    off = ""
    if o.get("exposure"):
        off = f" + offset(log({o['exposure']}))"
    elif o.get("offset"):
        off = f" + offset({o['offset']})"
    order = f"data <- data[order(data${e}" + (f", data${tm}" if tm else "") + "), ]"
    out = [order]
    agq = o["intPoints"]
    if model == "fe" and fam == "logit":
        rob = {"robust": ", robust = TRUE", "cluster": f", cluster = {o['clusterVar']}"}.get(o["covType"], "")
        out += [f"fit <- survival::clogit({y} ~ {xs}{off} + strata({e}), data = data, method = \"exact\"{rob})",
                "summary(fit)  # 조건부 우도 = Stata clogit; robust = TRUE의 소표본 보정은 Stata와 다를 수 있음"]
    elif model == "fe" and fam == "poisson":
        vc = ', vcov = ~' + e if o["covType"] == "robust" else ', vcov = "iid"'
        ex = (f", offset = ~log({o['exposure']})" if o.get("exposure")
              else f", offset = ~{o['offset']}" if o.get("offset") else "")
        out += [f"fit <- fixest::fepois({y} ~ {xs} | {e}, data = data{ex}{vc})",
                "summary(fit)  # 비조건부 FE 포아송 ML: 계수·표준오차가 조건부 우도(Stata xtpoisson, fe)와 같음; "
                "로그우도는 상수 차이; 강건 군집 보정(G/(G-1))은 Stata(보정 없음)와 다름"]
    elif model == "fe" and fam == "nbreg":
        out += [f"pdata <- plm::pdata.frame(data, index = c({q(e)}{', ' + q(tm) if tm else ''}))",
                f"fit <- pglm::pglm({y} ~ {xs}{off}, data = pdata, family = negbin, model = \"within\")",
                "summary(fit)  # Hausman, Hall & Griliches (1984) 조건부 음이항 — pglm 구현의 상수항 처리 확인 필요"]
    elif model == "re" and fam == "poisson" and o["reDist"] == "gamma":
        out += [f"pdata <- plm::pdata.frame(data, index = c({q(e)}{', ' + q(tm) if tm else ''}))",
                f"fit <- pglm::pglm({y} ~ {xs}{off}, data = pdata, family = poisson, model = \"random\")",
                "summary(fit)  # 감마 확률효과 포아송(HHG 1984); pglm의 산포 모수화는 Stata /lnalpha와 다를 수 있음"]
    elif model == "re" and fam == "nbreg":
        out += [f"pdata <- plm::pdata.frame(data, index = c({q(e)}{', ' + q(tm) if tm else ''}))",
                f"fit <- pglm::pglm({y} ~ {xs}{off}, data = pdata, family = negbin, model = \"random\")",
                "summary(fit)  # 베타 분포 산포 확률효과(HHG 1984) — Stata /ln_r, /ln_s와 모수화 대조 필요"]
    elif model == "re" and fam in ("logit", "probit", "cloglog", "poisson"):
        famr = {"logit": 'binomial(link = "logit")', "probit": 'binomial(link = "probit")',
                "cloglog": 'binomial(link = "cloglog")', "poisson": 'poisson(link = "log")'}[fam]
        n_agq = 1 if o["intMethod"] == "ghermite" else agq
        out += [f"fit <- lme4::glmer({y} ~ {xs}{off} + (1 | {e}), data = data, family = {famr}, nAGQ = {n_agq})",
                "summary(fit)  # lme4 nAGQ > 1: 최빈값·곡률 기준 적응 구적법 (Stata mvaghermite는 사후 평균·분산 기준) — "
                "소수점 아래 미세 차이; Stata /lnsig2u = log(분산 성분)"]
        if o["intMethod"] == "ghermite":
            out.append("# lme4에는 비적응 Gauss–Hermite 구적법이 없음 (nAGQ = 1은 Laplace 근사)")
        if o["covType"] != "oim":
            out.append("# Stata vce(robust)·vce(cluster)에 해당하는 glmer 옵션 없음 (merDeriv::sandwich 등 별도 계산)")
        if fam != "poisson" and o["margins"]:
            out.append("# 평균 한계효과(u_i 적분 확률): marginaleffects::avg_slopes(fit, re.form = NA)는 u_i = 0 기준 — "
                       "Stata predict(pr)와 다름")
    elif fam == "tobit":
        left = "-Inf" if o.get("ll") is None else (f"min(data${y})" if o["ll"] is True else _stata_lim(o["ll"]))
        right = "Inf" if o.get("ul") is None else (f"max(data${y})" if o["ul"] is True else _stata_lim(o["ul"]))
        out += [f"pdata <- plm::pdata.frame(data, index = c({q(e)}{', ' + q(tm) if tm else ''}))",
                f"fit <- censReg::censReg({y} ~ {xs}, left = {left}, right = {right}, data = pdata, nGHQ = {agq})",
                "summary(fit)  # censReg 패널 토빗: 비적응 Gauss–Hermite 구적법, logSigmaMu·logSigmaNu 모수화"]
    elif fam == "intreg":
        out += [f"# 확률효과 구간 회귀에 해당하는 R 함수 없음 — 참고: survival::survreg(Surv({y}, {o['depUpper']}, type = \"interval2\")"
                f" ~ {xs} + frailty({e}, distribution = \"gaussian\"), dist = \"gaussian\", data = data)",
                "# survreg의 frailty는 벌점 우도로 Stata xtintreg(정규 확률효과 ML)와 다른 추정치"]
    elif fam in ("ologit", "oprobit"):
        link = "logit" if fam == "ologit" else "probit"
        out += [f"fit <- ordinal::clmm(factor({y}) ~ {xs} + (1 | {e}), data = data, link = {q(link)}, nAGQ = {agq})",
                "summary(fit)  # 절단점 부호·의미는 Stata /cut#과 같음; clmm은 확률효과 표준편차 보고 (Stata /sigma2_u = 분산); "
                "nAGQ > 1은 최빈값 기준 적응 구적법"]
    else:   # pa
        famr = {"logit": 'binomial(link = "logit")', "probit": 'binomial(link = "probit")',
                "cloglog": 'binomial(link = "cloglog")', "poisson": 'poisson(link = "log")',
                "nbreg": "MASS::negative.binomial(theta = 1)"}[fam]
        cs_ = {"exchangeable": "exchangeable", "independent": "independence", "ar1": "ar1",
               "unstructured": "unstructured"}[o["paCorr"]]
        wv = f", waves = {tm}" if tm and o["paCorr"] in ("ar1", "unstructured") else ""
        out += [f"fit <- geepack::geeglm({y} ~ {xs}{off}, id = {e}, data = data, family = {famr}, corstr = {q(cs_)}"
                f"{wv}, scale.fix = TRUE)",
                "summary(fit)  # geepack 표준오차는 강건(sandwich) 분산 — Stata vce(robust)는 추가로 M/(M-1) 보정; "
                "작업 상관 추정식(자유도 보정)이 Stata 적률 추정과 다를 수 있음"]
        if fam == "nbreg":
            out.append("# Stata xtnbreg, pa: family(nbinomial) alpha = 1 고정 (MASS theta = 1/alpha = 1)")
        if o["margins"]:
            out.append("marginaleffects::avg_slopes(fit)  # Stata margins, dydx(*)")
    if o["hausman"]:
        out += ["# Hausman FE vs RE: 공통 계수 d = b_FE - b_RE, chi2 = t(d) %*% MASS::ginv(V_FE - V_RE) %*% d "
                "(두 모형의 OIM 분산)"]
    return out


# --------------------------------------------------------------------------------------------- Stata command input
_CMD_RE = re.compile(r"^(xtlogit|xtprobit|xtcloglog|xtpoisson|xtnbreg|xttobit|xtintreg|xtologit|xtoprobit|clogit)"
                     r"\b(.*)$", re.IGNORECASE)
_HARMLESS = ("nolog", "log", "difficult", "iterate", "tolerance", "ltolerance", "nrtolerance", "nonrtolerance",
             "technique", "trace", "gradient", "showstep", "hessian", "showtolerance", "noheader", "nocnsreport",
             "nodisplay", "coeflegend", "noomitted", "vsquish", "noemptycells", "baselevels", "allbaselevels",
             "nofvlabel", "fvwrap", "fvwrapon", "cformat", "pformat", "sformat", "nolstretch", "noci", "nopvalues",
             "from", "norefine")
# (full name, minimum abbreviation)
_OPTS = [("re", "re"), ("fe", "fe"), ("pa", "pa"), ("normal", "norm"), ("exposure", "e"), ("offset", "off"),
         ("noconstant", "nocons"), ("vce", "vce"), ("robust", "r"), ("cluster", "cl"), ("or", "or"), ("irr", "irr"),
         ("eform", "eform"), ("intmethod", "intm"), ("intpoints", "intp"), ("quad", "quad"), ("corr", "c"),
         ("force", "force"), ("level", "l"), ("ll", "ll"), ("ul", "ul"), ("group", "gr"), ("strata", "strata"),
         ("lrmodel", "lrmodel"), ("constraints", "constr"), ("asis", "asis"), ("nmp", "nmp"), ("scale", "scale"),
         ("rgf", "rgf"), ("collinear", "coll"), ("dispersion", "disp"), ("difficult", "dif"), ("tobit", "tobit"),
         ("technique", "tech"), ("iterate", "iter"), ("tolerance", "tol"), ("ltolerance", "ltol"),
         ("nrtolerance", "nrtol")] + [(h, h) for h in _HARMLESS]


def _canon(name: str) -> Optional[str]:
    low = name.lower()
    for full, _ in _OPTS:
        if low == full:
            return full
    for full, ab in _OPTS:
        if len(low) >= len(ab) and full.startswith(low) and low.startswith(ab):
            return full
    return None


def _parse_opts(txt: str) -> List[Tuple[str, Optional[str], str, bool]]:
    """'vce(cluster id) or' -> [(raw, canonical or None, value, has_parens)]."""
    out, i, n = [], 0, len(txt)
    while i < n:
        while i < n and txt[i] in " \t,":
            i += 1
        j = i
        while j < n and txt[j] not in " \t(,":
            j += 1
        name = txt[i:j]
        if not name:
            break
        val, par = "", False
        if j < n and txt[j] == "(":
            depth, k = 0, j
            while k < n:
                depth += txt[k] == "("
                depth -= txt[k] == ")"
                if depth == 0:
                    break
                k += 1
            val, par = txt[j + 1:k].strip(), True
            j = k + 1
        out.append((name, _canon(name), val, par))
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


_VAR_RE = re.compile(r"[A-Za-z_À-￿][\wÀ-￿]*")


def transpile(line: str) -> Optional[str]:
    s = (line or "").strip()
    m = _CMD_RE.match(s)
    if not m:
        return None
    cmd = m.group(1).lower()

    def stop(msg: str) -> str:
        return f"raise NotImplementedError({(cmd + ': ' + msg)!r})"

    body, opt_txt = _split_comma(m.group(2))
    if re.search(r"\[|\]", body):
        return stop(f"가중치([iweight=…] 등) 미지원 — {s}")
    if re.search(r"\b(if|in)\b", body):
        return stop(f"if·in 조건 미지원 — keep if 등으로 자료를 먼저 선택한 뒤 실행: {s}")
    toks = body.split()
    for tk in toks:
        if not _VAR_RE.fullmatch(tk):
            return stop(f"변수 표기 미지원 '{tk}' (요인변수 i.·시계열 연산자 L.·D.·상호작용 # 미지원 — 변수를 만든 뒤 지정): {s}")
    fam, model_fixed = ALIAS_PRESETS[cmd]
    n_dep = 2 if fam == "intreg" else 1
    if len(toks) < n_dep:
        return stop(("하한·상한 종속변수 2개 필요" if n_dep == 2 else "종속변수 필요") + f" — {s}")
    deps, covs = toks[:n_dep], toks[n_dep:]
    kw: Dict[str, Any] = {}
    comments: List[str] = []
    bad: List[str] = []
    model = model_fixed
    group = None
    for raw, name, val, par in _parse_opts(opt_txt):
        shown = raw + (f"({val})" if par else "")
        if name is None:
            bad.append(shown)
        elif name == "tobit" and fam == "tobit":
            comments.append("tobit (통합 토빗 대비 LR 검정은 기본 출력에 포함)")
        elif name in _HARMLESS or name in ("difficult", "technique", "iterate", "tolerance", "ltolerance",
                                           "nrtolerance"):
            comments.append(shown)
        elif name in ("re", "fe", "pa") and cmd != "clogit" and fam not in ("tobit", "intreg", "ologit", "oprobit"):
            model = name
        elif name == "normal" and fam == "poisson":
            kw["reDist"] = "normal"
        elif name in ("exposure", "offset") and par:
            kw[name] = val
        elif name == "noconstant":
            kw["constant"] = False
        elif name == "robust" and not par:
            kw["covType"] = "robust"
        elif name == "cluster" and par:
            kw["covType"], kw["clusterVar"] = "cluster", val
        elif name == "vce":
            vt = val.split()
            v0 = vt[0].lower() if vt else ""
            if v0 in ("oim", "conventional"):
                kw["covType"] = "oim"
            elif v0 and "robust".startswith(v0):
                kw["covType"] = "robust"
            elif len(v0) >= 2 and "cluster".startswith(v0) and len(vt) == 2:
                kw["covType"], kw["clusterVar"] = "cluster", vt[1]
            else:
                bad.append(shown + " (bootstrap·jackknife·opg 미지원)")
        elif name in ("or", "irr", "eform") and not par:
            kw["eform"] = True
        elif name == "intmethod":
            v = val.lower()
            if v in ("mvaghermite", "mvagh", "mvaghermit"):
                kw["intMethod"] = "mvaghermite"
            elif v in ("ghermite", "gh", "ghermit"):
                kw["intMethod"] = "ghermite"
            else:
                bad.append(shown + " (mvaghermite·ghermite만 지원)")
        elif name in ("intpoints", "quad"):
            if re.fullmatch(r"\d+", val):
                kw["intPoints"] = int(val)
            else:
                bad.append(shown)
        elif name == "corr":
            cv = val.lower().replace(" ", "")
            cmap = [("exchangeable", "exchangeable"), ("exc", "exchangeable"), ("independent", "independent"),
                    ("ind", "independent"), ("ar1", "ar1"), ("unstructured", "unstructured"), ("uns", "unstructured")]
            cv2 = next((v for k_, v in cmap if cv.startswith(k_) or (len(cv) >= 3 and k_.startswith(cv))), None)
            if cv2 is None or (cv.startswith("ar") and cv != "ar1"):
                bad.append(shown + " (exchangeable·independent·ar 1·unstructured만 지원)")
            else:
                kw["paCorr"] = cv2
        elif name == "force":
            kw["force"] = True
        elif name == "level":
            try:
                kw["ciLevel"] = float(val)
            except ValueError:
                bad.append(shown)
        elif name in ("ll", "ul") and fam == "tobit":
            if not par:
                kw[name] = True
            else:
                try:
                    kw[name] = float(val)
                except ValueError:
                    if _VAR_RE.fullmatch(val):
                        kw[name] = val
                    else:
                        bad.append(shown)
        elif name in ("group", "strata") and cmd == "clogit" and par:
            group = val
        else:
            bad.append(shown)
    if cmd == "clogit" and not group:
        return stop(f"group(변수) 옵션 필요 — {s}")
    if bad:
        return stop(f"옵션 미지원: {', '.join(bad)} — {s}")
    model = model or "re"
    parts = []
    if cmd == "clogit":
        parts.append(f"entity={group!r}")
    else:
        parts += ["entity=getattr(sess, 'panel_spec', {}).get('idVar')",
                  "time=getattr(sess, 'panel_spec', {}).get('timeVar')"]
    parts.append(f"dep={deps[0]!r}")
    if n_dep == 2:
        parts.append(f"depUpper={deps[1]!r}")
    parts += [f"covs={covs!r}", f"family={fam!r}", f"model={model!r}"]
    if cmd == "clogit":
        parts.append("clogit=True")
    for k_, v in kw.items():
        parts.append(f"{k_}={v!r}")
    quad = model == "re" and (fam != "poisson" or kw.get("reDist") == "normal") and fam != "nbreg"
    strict = []
    if ("intMethod" in kw or "intPoints" in kw) and not quad:
        strict.append("intmethod()·intpoints() (정규분포 확률효과 모형에만 해당)")
    if "exposure" in kw and FAMILIES[fam]["kind"] != "count":
        strict.append("exposure()")
    if "offset" in kw and fam in ("ologit", "oprobit", "intreg"):
        strict.append("offset()")
    if kw.get("eform") and not FAMILIES[fam]["eform"]:
        strict.append("or·irr·eform")
    if ("paCorr" in kw or "force" in kw) and model != "pa":
        strict.append("corr()·force (pa 모형에만 해당)")
    if "reDist" in kw and model != "re":
        strict.append("normal (re 모형에만 해당)")
    if strict:
        return stop(f"옵션 미지원: {', '.join(strict)} — {s}")
    try:
        chk = {"entity": "_e", "time": "_t", "dep": deps[0], "covs": covs, "family": fam, "model": model,
               "depUpper": deps[1] if n_dep == 2 else None, "clogit": cmd == "clogit"}
        if fam == "tobit" and "ll" not in kw and "ul" not in kw:
            return stop(f"ll() 또는 ul() 지정 필요 (검열 한계 없으면 xtreg) — {s}")
        normalize_options(dict(chk, **kw))
    except ValueError as err:
        return stop(f"{err} — {s}")
    code = f"res = sess.run('panelNonlinear', {', '.join(parts)})"
    if comments:
        code += f"\n# [Stata] {', '.join(comments)}: 반복·출력 제어 옵션 — 추정 결과와 무관하여 생략"
    return code


EXTENSION = {
    "catalog": [{"id": "panelNonlinear", "group": "Linear Models",
                 "title": "Nonlinear Panel Models (xtlogit/xtprobit/xtcloglog/xtpoisson/xtnbreg/xttobit/xtintreg/"
                          "xtologit/xtoprobit, clogit; FE/RE/PA)",
                 "module": "openstat", "icon": "analysis-linearmodels.svg"}],
    "aliases": {**{k: "panelNonlinear" for k in ALIAS_PRESETS}, "panel_nonlinear": "panelNonlinear",
                "panelnonlinear": "panelNonlinear", "xtbinary": "panelNonlinear"},
    "dispatch": dispatch,
    "uses_alias": True,          # registry passes the typed name as params["_alias"] (alias -> family preset)
    "stata": stata_code,
    "r": r_code,
    "transpile": transpile,
}
