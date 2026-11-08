"""
pyjamovi Structural Equation Modeling (CFA / SEM / path analysis)

Own maximum likelihood engine (`sem_engine`) with Mplus default conventions; the model is a Python string with
lavaan-style operators (=~, ~, ~~, ~ 1, :=, ==) and options are Python keyword arguments.

    sess.sem(model='''
        Visual  =~ x1_vis + x2_vis + x3_vis
        Textual =~ x4_txt + x5_txt + x6_txt
        GPA ~ b*Textual + c*Visual
        Textual ~ a*Visual
        ind := a*b
    ''', estimator="mlr", indirect=["GPA <- Visual"], modIndices=True)

Categorical (binary / ordinal) variables as in Mplus CATEGORICAL ARE (estimator WLSMV by default; WLSM, WLS, ULSMV,
ULS; parameterization delta or theta; covariates conditioned on as probit regressions):

    sess.sem(model="S =~ s1 + s2 + s3 + s4\nuse ~ S + age", categorical=["s1", "s2", "s3", "s4", "use"])

Continuous data: ML (default, FIML), MLR (FIML, Yuan–Bentler), MLM (listwise, Satorra–Bentler).
"""

from typing import Any, Dict, List, Optional

import math
import numpy as np
import pandas as pd

from .base import BaseAnalysis, ResultsTable, resolve_choice
from .plots import plot_sem_diagram
from . import sem_engine as se

_ALIASES = {
    "model_syntax": "model", "std_est": "stdEst", "fit_measures": "fitMeasures", "path_diagram": "pathDiagram",
    "mod_indices": "modIndices", "modindices": "modIndices", "mi_min": "miMin", "ci_type": "ciType",
    "invariance_test": "invarianceTest", "group_order": "groupOrder", "ci_level": "ciLevel",
    "r_square": "rsquare", "rSquare": "rsquare", "conditional_x": "conditionalX", "sample_stats": "sampstat",
    "sampStat": "sampstat", "sampStats": "sampstat", "ordered": "categorical",
}
# keyword options read by SEM (registry/session can pass these through unchanged)
SEM_OPTION_KEYS = ("estimator", "missing", "information", "group", "invariance", "invarianceTest", "indirect",
                   "bootstrap", "ciType", "ciLevel", "modIndices", "miMin", "standardized", "rsquare", "seed",
                   "groupOrder", "categorical", "parameterization", "conditionalX", "sampstat")
_ESTIMATORS = ["ml", "mlr", "mlm", "wlsmv", "wlsm", "wls", "ulsmv", "uls"]
_WLS = ("wlsmv", "wlsm", "wls", "ulsmv", "uls")
_EST_KO = {"ml": "최대우도(ML)", "mlr": "로버스트 최대우도(MLR)", "mlm": "로버스트 최대우도(MLM, Satorra–Bentler)",
           "wlsmv": "WLSMV (평균·분산 보정 가중최소제곱)", "wlsm": "WLSM (평균 보정 가중최소제곱)",
           "wls": "WLS (가중최소제곱, 전체 가중행렬)", "ulsmv": "ULSMV (평균·분산 보정 비가중최소제곱)",
           "uls": "ULS (비가중최소제곱)"}
_KIND_KO = {"BY": "요인적재(BY)", "ON": "회귀(ON)", "WITH": "공분산(WITH)", "EQ": "동일성 제약 해제"}
_STD_KO = {"stdyx": "STDYX", "stdy": "STDY", "std": "STD", "none": "없음"}


def _flag(v, default):
    if v is None or v == "":
        return default
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


def _mplus_text(lhs: str, op: str, rhs: str) -> str:
    if op == "=~":
        return f"{lhs} BY {rhs}"
    if op == "~":
        return f"{lhs} ON {rhs}"
    if op == "~~":
        return lhs if lhs == rhs else f"{lhs} WITH {rhs}"
    if op == "~1":
        return f"[{lhs}]"
    if op == "|":
        return f"[{lhs}${rhs[1:]}]"
    if op == "~*~":
        return f"{{{lhs}}}"
    return f"{lhs} {op} {rhs}"


def _as_names(v) -> List[str]:
    """Variable names from a list or a comma / semicolon / newline separated string (names may contain blanks)."""
    if v is None or v == "" or v is False:
        return []
    if isinstance(v, str):
        import re as _re
        return [x.strip() for x in _re.split(r"[,;\n]+", v) if x.strip()]
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    return [str(v)]


def _as_list(v) -> List[Any]:
    if v is None or v == "" or v is False:
        return []
    if isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v) and isinstance(v, tuple):
        return [v]
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str):
        return [x.strip() for x in v.replace(";", "\n").splitlines() if x.strip()]
    return [v]


class SEM(BaseAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        model_syntax: str = "",
        std_est: bool = True,
        fit_measures: bool = True,
        variances: bool = False,
        path_diagram: bool = True,
        **options,
    ):
        opts: Dict[str, Any] = {}
        for k, v in options.items():
            opts[_ALIASES.get(k, k)] = v
        model = opts.pop("model", None) or model_syntax or ""
        opts.setdefault("stdEst", std_est)
        opts.setdefault("fitMeasures", fit_measures)
        opts.setdefault("variances", variances)
        opts.setdefault("pathDiagram", path_diagram)
        opts["model"] = model
        super().__init__(name="sem", title="구조방정식모형 (CFA / SEM)", dataset=data, options=opts)
        self.model_syntax = model
        self.result: Optional[se.SEMResult] = None
        self.run()

    # ------------------------------------------------------------------ option handling
    def _settings(self) -> Dict[str, Any]:
        o = self.options
        if not str(o.get("model") or "").strip():
            raise ValueError("모형식(model) 필요: 예) F =~ y1 + y2 + y3")
        cat = _as_names(o.get("categorical"))
        est = resolve_choice("추정법", o.get("estimator"), _ESTIMATORS + ["auto"],
                             {"robust": "mlr", "ml_robust": "mlr", "dwls": "wlsmv", "satorra": "mlm", "default": "auto",
                              "자동": "auto"}, "auto")
        if est == "auto":
            est = "wlsmv" if cat else "ml"
        wls = est in _WLS
        if cat and not wls:
            raise ValueError(f"범주형 변수(categorical)와 {est.upper()} 추정(수치 적분, 로짓·프로빗 링크)은 지원하지 않음 — "
                             "estimator='wlsmv'(Mplus 범주형 변수 기본 추정법) 사용")
        if wls and not cat:
            raise ValueError(f"{est.upper()} 추정은 범주형 변수(categorical) 지정 모형에서 사용 — 연속형 변수만인 모형은 "
                             "ML·MLR·MLM 사용")
        miss = resolve_choice("결측 처리", o.get("missing"), ["fiml", "listwise", "auto"],
                              {"ml": "fiml", "fml": "fiml", "available": "fiml", "listwise_deletion": "listwise",
                               "complete": "listwise", "default": "auto"}, "auto")
        if miss == "auto":
            miss = "listwise" if (wls or est == "mlm") else "fiml"
        if miss == "fiml" and (wls or est == "mlm"):
            raise ValueError(f"{est.upper()} 추정은 완전정보최대우도(FIML) 미지원 — 결측 처리는 목록별 삭제(listwise) "
                             "(결측 자료의 로버스트 추정은 MLR)")
        param = resolve_choice("모수화", o.get("parameterization"), ["delta", "theta"], {"델타": "delta", "세타": "theta"},
                               "delta")
        info = resolve_choice("정보행렬", o.get("information"), ["observed", "expected"], {"obs": "observed", "exp": "expected"},
                              "observed")
        inv = resolve_choice("측정동일성", o.get("invariance"), list(se.INVARIANCE_LABELS),
                             {"none": "configural", "weak": "metric", "loadings": "metric", "strong": "scalar",
                              "intercepts": "scalar", "residuals": "strict"}, "scalar")
        std_default = "stdyx" if _flag(o.get("stdEst"), True) else "none"
        stdk = resolve_choice("표준화", o.get("standardized"), list(_STD_KO), {"true": "stdyx", "false": "none", "all": "stdyx"},
                              std_default) if not isinstance(o.get("standardized"), bool) else \
            ("stdyx" if o.get("standardized") else "none")
        ci = resolve_choice("신뢰구간", o.get("ciType"), ["symmetric", "percentile", "bc"],
                            {"bootstrap": "percentile", "bcbootstrap": "bc", "bias_corrected": "bc"}, "symmetric")
        boot = int(float(o.get("bootstrap") or 0))
        if boot < 0:
            raise ValueError("붓스트랩 횟수는 0 이상")
        if boot and est == "mlr":
            raise ValueError("붓스트랩은 ML 추정에서만 사용 (Mplus 규칙: MLR 추정치는 ML과 같으므로 estimator='ml'로 지정)")
        if boot and est != "ml":
            raise ValueError(f"붓스트랩은 ML 추정에서만 사용 ({est.upper()}: 샌드위치 표준오차)")
        group = o.get("group") or None
        if isinstance(group, (list, tuple)):
            group = group[0] if group else None
        return {
            "estimator": est, "missing": miss, "information": info, "invariance": inv, "std": stdk, "ciType": ci,
            "bootstrap": boot, "group": group, "seed": int(float(o.get("seed") or 20260)),
            "miMin": float(o.get("miMin") if o.get("miMin") not in (None, "") else 10.0),
            "modIndices": _flag(o.get("modIndices"), False), "invarianceTest": _flag(o.get("invarianceTest"), False),
            "indirect": _as_list(o.get("indirect")), "rsquare": _flag(o.get("rsquare"), True),
            "fit": _flag(o.get("fitMeasures"), True), "diagram": _flag(o.get("pathDiagram"), True),
            "ciLevel": float(o.get("ciLevel") or 0.95), "groupOrder": o.get("groupOrder"),
            "categorical": cat, "parameterization": param, "conditionalX": _flag(o.get("conditionalX"), True),
            "sampstat": _flag(o.get("sampstat"), False), "wls": wls,
        }

    def _syntax(self, s: Dict[str, Any]) -> str:
        parts = [f'model="""\n{self.model_syntax.strip()}\n"""']
        if s["categorical"]:
            parts.append(f"categorical={s['categorical']!r}")
            if s["estimator"] != "wlsmv":
                parts.append(f"estimator={s['estimator']!r}")
            if s["parameterization"] != "delta":
                parts.append(f"parameterization={s['parameterization']!r}")
            if not s["conditionalX"]:
                parts.append("conditionalX=False")
            if s["sampstat"]:
                parts.append("sampstat=True")
        elif s["estimator"] != "ml":
            parts.append(f"estimator={s['estimator']!r}")
        if s["missing"] != ("listwise" if (s["wls"] or s["estimator"] == "mlm") else "fiml"):
            parts.append(f"missing={s['missing']!r}")
        if s["group"]:
            parts.append(f"group={s['group']!r}")
            if s["invariance"] != "scalar":
                parts.append(f"invariance={s['invariance']!r}")
            if s["invarianceTest"]:
                parts.append("invarianceTest=True")
        if s["indirect"]:
            parts.append(f"indirect={s['indirect']!r}")
        if s["bootstrap"]:
            parts.append(f"bootstrap={s['bootstrap']}")
            if s["ciType"] != "symmetric":
                parts.append(f"ciType={s['ciType']!r}")
        if s["modIndices"]:
            parts.append("modIndices=True" + (f", miMin={s['miMin']:g}" if s["miMin"] != 10 else ""))
        if s["std"] not in ("stdyx",):
            parts.append(f"standardized={s['std']!r}")
        if s["information"] != "observed" and s["estimator"] in ("ml", "mlr"):
            parts.append(f"information={s['information']!r}")
        return "sess.sem(" + ", ".join(parts) + ")"

    def _syntax_r(self, s: Dict[str, Any]) -> str:
        est = s["estimator"].upper()
        args = ["model", "data = data", f"estimator = '{est}'"]
        if s["categorical"]:
            args.append("ordered = c(" + ", ".join(f"'{c}'" for c in s["categorical"]) + ")")
            args.append(f"parameterization = '{s['parameterization']}'")
            if not s["conditionalX"]:
                args.append("conditional.x = FALSE, fixed.x = FALSE")
        else:
            args.append(f"missing = '{'fiml' if s['missing'] == 'fiml' else 'listwise'}'")
        if s["group"]:
            args.append(f"group = '{s['group']}'")
        args += ["meanstructure = TRUE", "mimic = 'Mplus'"]
        return "lavaan::sem(" + ", ".join(args) + ")"

    # ------------------------------------------------------------------ main
    def run(self):
        s = self._settings()
        self.syntax_python = self._syntax(s)
        self.syntax_r = self._syntax_r(s)
        try:
            res = se.estimate(self.dataset, self.model_syntax, group=s["group"], estimator=s["estimator"],
                              missing=s["missing"], invariance=s["invariance"], information=s["information"],
                              bootstrap=s["bootstrap"], seed=s["seed"], group_order=s["groupOrder"],
                              categorical=s["categorical"] or None, parameterization=s["parameterization"],
                              conditional_x=s["conditionalX"])
        except se.SEMError as e:
            raise ValueError(f"구조방정식모형: {e}") from e
        self.result = res
        spec = res.spec
        G = spec.G
        der = se.Derived(res)
        level = s["ciLevel"]
        boot = res.boot if (res.boot is not None and res.boot.shape[0] >= 2) else None
        glabel = lambda g: str(res.sem.group_levels[g]) if G > 1 else ""

        # ---------------------------------------------------------------- 1. estimates
        std_kind = s["std"]
        std_val = std_se = None
        if std_kind != "none":
            std_val, std_se = se.delta_se(lambda th: der.std_values(th, std_kind), res.theta, res.vcov)
            if boot is not None:
                bd = np.array([der.std_values(th, std_kind) for th in boot])
                std_se = bd.std(axis=0, ddof=1)
        has_label = any(r.label and not r.label.startswith(".") for r in spec.rows)
        t = ResultsTable(name="estimates", title="모수 추정치 (Model Results)")
        if G > 1:
            t.add_column("group", "집단", col_type="text")
        t.add_column("section", "구분", col_type="text")
        t.add_column("mplus", "Mplus 표기", col_type="text")
        t.add_column("lval", "좌변", col_type="text")
        t.add_column("op", "연산자", col_type="text")
        t.add_column("rval", "우변", col_type="text")
        if has_label:
            t.add_column("label", "라벨", col_type="text")
        t.add_column("est", "추정치", col_type="number")
        t.add_column("se", "SE", col_type="number")
        t.add_column("z", "Est./S.E.", col_type="number")
        t.add_column("p", "p", col_type="number", fmt="pvalue")
        if boot is not None:
            t.add_column("ci_lo", f"{level:.0%} CI 하한", col_type="number")
            t.add_column("ci_hi", f"{level:.0%} CI 상한", col_type="number")
        if std_kind != "none":
            t.add_column("std", f"표준화 ({_STD_KO[std_kind]})", col_type="number", fmt="zto")
            t.add_column("std_se", "표준화 SE", col_type="number")
        free_est = res.theta
        ci_lo = ci_hi = None
        if boot is not None:
            ci_lo, ci_hi = se.boot_ci(free_est, boot, res.se, s["ciType"], level)
        diagram_rows = []
        wls = s["wls"]
        show_scale = any(r.op == "~*~" and r.free for r in spec.rows)     # Mplus prints Scales only then
        for n, r in enumerate(spec.rows):
            val = res.theta[r.pid] if r.free else r.value
            if r.kind == "cat_resvar":
                continue                      # y* residual variance: not a parameter (shown in the R² table)
            if r.op == "~*~" and not show_scale:
                continue
            if not r.free and val == 0 and not r.user:
                continue
            if not r.free and r.op == "~1" and val == 0:
                continue
            sev = res.se[r.pid] if r.free else float("nan")
            z = val / sev if r.free and sev > 0 else float("nan")
            row = {"section": r.section, "mplus": _mplus_text(r.lhs, r.op, r.rhs), "lval": r.lhs, "op": r.op,
                   "rval": r.rhs, "est": float(val), "se": float(sev), "z": float(z), "p": se._pval(z)}
            if G > 1:
                row["group"] = glabel(r.group)
            if has_label:
                row["label"] = r.label if (r.label and not r.label.startswith(".")) else ""
            if boot is not None:
                row["ci_lo"] = float(ci_lo[r.pid]) if r.free else float("nan")
                row["ci_hi"] = float(ci_hi[r.pid]) if r.free else float("nan")
            if std_kind != "none":
                row["std"] = float(std_val[n])
                row["std_se"] = float(std_se[n])
            t.add_row(row)
            if r.group == 0 and r.op in ("=~", "~"):
                diagram_rows.append({"lval": r.lhs, "op": r.op, "rval": r.rhs,
                                     "Estimate": float(std_val[n]) if std_kind != "none" else float(val),
                                     "p-value": se._pval(z) if r.free else float("nan")})
        t.add_note("Est./S.E. = 추정치/표준오차, p는 양측 정규 근사. 고정 모수는 SE 미표시.")
        if wls:
            t.add_note("범주형 변수(" + ", ".join(spec.categorical) + "): 잠재반응변수 y*의 프로빗 모형 — 적재량·회귀계수는 "
                       "y* 척도, 문턱값 [y$k]는 범주 k와 k+1의 경계(Mplus 표기). "
                       + ("delta 모수화: Var(y*" + ("| x" if spec.qx else "") + ") = 1/Δ² (척도요인 Δ {y}, 기본 1 고정), "
                          "y* 잔차분산은 모수가 아니며 설명력 표에 표시."
                          if s["parameterization"] == "delta" else
                          "theta 모수화: y* 잔차분산 1로 고정(지정 시 자유 모수), 척도 요인은 설명력 표에 표시."))
        if std_kind != "none":
            t.add_note(f"표준화 {_STD_KO[std_kind]}: "
                       + {"stdyx": "잠재·관측 변수 모두의 분산으로 표준화",
                          "stdy": "관측 공변량(x)은 원척도 유지, 나머지 변수의 분산으로 표준화",
                          "std": "잠재변수의 분산으로만 표준화"}[std_kind] + ", SE는 델타법" + (" (붓스트랩 시 붓스트랩 SE)" if boot is not None else "") + ".")
        t.mark_coefficients(estimate="est", se="se", p="p", stat="z", std_estimate="std", ci=("ci_lo", "ci_hi"),
                            label=["group", "mplus"])
        self.add_table(t)

        # ---------------------------------------------------------------- 2. fit
        fs = res.fitstats
        self.model_stats = {"N": int(fs["N"]), "LL": fs["ll0"], "AIC": fs["aic"], "BIC": fs["bic"],
                            "Chi2": (fs["chi2_test"], fs["pvalue"]), "df": int(fs["df"]), "CFI": fs["cfi"],
                            "TLI": fs["tli"], "RMSEA": fs["rmsea"], "SRMR": fs["srmr"]}
        if wls:
            for k in ("LL", "AIC", "BIC"):
                self.model_stats.pop(k)
        mlr = s["estimator"] == "mlr"
        mlm = s["estimator"] == "mlm"
        if s["fit"] and wls:
            self._fit_table_wls(fs, s)
        elif s["fit"]:
            ft = ResultsTable(name="fit", title="모형 적합도 (Model Fit Indices)")
            ft.add_column("section", "구분", col_type="text")
            ft.add_column("index", "지수", col_type="text")
            ft.add_column("value", "값", col_type="number")
            rows = [("모수", "자유 모수 수", fs["npar"]), ("로그우도", "H0 로그우도", fs["ll0"])]
            if mlr and "c0" in fs:
                rows.append(("로그우도", "H0 척도 보정 계수 (MLR)", fs["c0"]))
            rows.append(("로그우도", "H1 로그우도", fs["ll1"]))
            if mlr and "c1" in fs:
                rows.append(("로그우도", "H1 척도 보정 계수 (MLR)", fs["c1"]))
            rows += [("정보기준", "AIC", fs["aic"]), ("정보기준", "BIC", fs["bic"]),
                     ("정보기준", "표본크기 보정 BIC (SABIC)", fs["sabic"]),
                     ("χ² 모형 검정", "χ² (모형 적합도 검정)" + (" — MLR 척도 보정" if mlr else ""), fs["chi2_test"]),
                     ("χ² 모형 검정", "자유도 (df)", fs["df"]), ("χ² 모형 검정", "p (χ²)", fs["pvalue"])]
            if mlr and "cd" in fs:
                rows.append(("χ² 모형 검정", "χ² 척도 보정 계수 (MLR)", fs["cd"]))
            if mlm and "scaling" in fs:
                rows[-3] = ("χ² 모형 검정", "χ² (모형 적합도 검정) — MLM Satorra–Bentler 보정", fs["chi2_test"])
                rows += [("χ² 모형 검정", "χ² 척도 보정 계수 (MLM)", fs["scaling"]),
                         ("χ² 모형 검정", "보정 전 χ² (ML)", fs["chi2"])]
            rows += [("RMSEA", "RMSEA", fs["rmsea"]), ("RMSEA", "RMSEA 90% CI 하한", fs["rmsea_lo"]),
                     ("RMSEA", "RMSEA 90% CI 상한", fs["rmsea_hi"]), ("RMSEA", "P(RMSEA ≤ .05)", fs["rmsea_p"]),
                     ("CFI/TLI", "Comparative Fit Index (CFI)", fs["cfi"]), ("CFI/TLI", "Tucker-Lewis Index (TLI)", fs["tli"]),
                     ("기저모형", "기저모형 χ²", fs["chi2_base"]), ("기저모형", "기저모형 df", fs["df_base"]),
                     ("기저모형", "기저모형 p", fs["p_base"]), ("SRMR", "SRMR", fs["srmr"]), ("사례", "사례 수 (N)", fs["N"])]
            if mlm and "scaling_base" in fs:
                rows.insert(-2, ("기저모형", "기저모형 척도 보정 계수 (MLM)", fs["scaling_base"]))
            for sec, idx, val in rows:
                ft.add_row({"section": sec, "index": idx, "value": float(val) if val is not None else float("nan")})
            if fs["df"] == 0:
                ft.add_note("포화모형(df = 0): χ² = 0, RMSEA = 0, CFI = TLI = 1로 표시.")
            ft.add_note("χ² = 2(H1 로그우도 − H0 로그우도), H1은 제약 없는 평균·공분산 모형"
                        + (" (FIML 결측 시 EM 추정)" if s["missing"] == "fiml" else "") + ". "
                        "RMSEA = √G·√(max(χ² − df, 0)/(df·N)), CFI·TLI 기저모형은 종속변수의 평균·분산만 자유(Mplus 기준), "
                        "TLI는 1을 넘으면 1로 표시(Mplus 출력 방식). "
                        "SRMR은 상관·표준화 평균·분산 잔차 포함(Asparouhov & Muthén, 2018)"
                        + (", 공변량 항 제외" if spec.qx else "") + ".")
            if mlr:
                ft.add_note("MLR: 표준오차는 Huber–White 샌드위치, χ²는 Yuan–Bentler 척도 보정 "
                            "(척도 보정 계수 = (p1·c1 − p0·c0)/(p1 − p0)). RMSEA·CFI·TLI는 보정 χ² 기준.")
            if mlm:
                ft.add_note("MLM(Mplus): 완전 자료 ML 추정치, 표준오차는 기대 정보행렬 기반 샌드위치(ADF Γ), "
                            "χ²는 Satorra–Bentler 평균 보정 T/c (c = tr(UΓ)/df). RMSEA·CFI·TLI는 보정 χ² 기준(기저모형도 보정). "
                            "보정 χ²의 차이는 그대로 χ² 차이검정에 사용 불가(척도 보정 차이검정 필요).")
            self.add_table(ft)

        # ---------------------------------------------------------------- 3. settings
        at = ResultsTable(name="assumptions", title="분석 설정")
        at.add_column("item", "항목", col_type="text")
        at.add_column("value", "설정", col_type="text")
        if wls:
            rows = self._settings_rows_wls(res, s)
        else:
            rows = self._settings_rows_ml(res, s, G, glabel)
        for item, val in rows:
            at.add_row({"item": item, "value": val})
        for nt in res.notes:
            at.add_note(nt)
        self.add_table(at)
        self._after_settings(res, der, s, boot, level, G, glabel, diagram_rows, std_kind)

    def _settings_rows_ml(self, res, s, G, glabel):
        spec = res.spec
        mlr = s["estimator"] == "mlr"
        mlm = s["estimator"] == "mlm"
        rows = [
            ("추정법", _EST_KO[s["estimator"]]),
            ("표준오차", ("붓스트랩 " + f"{s['bootstrap']}회" if s["bootstrap"] else
                         ("샌드위치(기대 정보행렬·ADF Γ, Mplus MLM)" if mlm else
                          ("샌드위치(관측 정보행렬 기준)" if mlr else "역정보행렬")))
             + ("" if mlm else f", 정보행렬: {'관측' if s['information'] == 'observed' else '기대'}")),
            ("결측 처리", "완전정보최대우도(FIML, MAR 가정)" if s["missing"] == "fiml" else "목록별 삭제"),
            ("공변량(x) 처리", ("공변량에 조건부 추정(x의 평균·분산·공분산은 모수 아님): " + ", ".join(spec.ox)) if spec.qx else
             ("공변량을 모형에 포함(모형식에서 분산·평균 지정)" if spec.x_joint else "관측 공변량 없음")),
            ("평균구조", "포함 (관측 종속·지표 절편 자유, 잠재변수 평균 0" + (", 2번째 이후 집단 자유" if G > 1 and s["invariance"] in ("scalar", "strict") else "") + ")"),
            ("요인 척도", "각 요인의 첫 지표 적재량 1로 고정 (지정자로 변경 가능)"),
            ("기본 공분산", "외생 잠재변수 간 공분산 자유, 최종 종속변수 잔차 공분산 자유, 지표 잔차 공분산 0"),
            ("사례 수", f"{res.sem.N:,}명 사용 / 입력 {res.sem.n_input:,}명"),
            ("결측 패턴 수", str(sum(len(gd.patterns) for gd in res.sem.data))),
            ("수렴", "수렴" if res.fit.converged else "수렴 기준 미충족"),
        ]
        if G > 1:
            rows.insert(4, ("다집단", f"집단 변수 {s['group']}: " + ", ".join(f"{glabel(g)}({n:,})" for g, n in enumerate(res.sem.group_n))
                            + f" — 첫 집단 기준, 동일성: {se.INVARIANCE_LABELS[s['invariance']]}"))
        return rows

    def _settings_rows_wls(self, res, s):
        sem, spec = res.sem, res.spec
        est = s["estimator"]
        cats = []
        for v, info in sem.cat_info.items():
            mp = ", ".join(f"{c}→{k}" for k, c in enumerate(info["categories"]))
            cats.append(f"{v}: {len(info['categories'])}범주 ({mp}; 빈도 {'/'.join(str(n) for n in info['counts'])})")
        kinds = {sem.ss.types[i] for i in range(len(sem.ss.types))}
        corr = ["다분상관(polychoric, 범주형–범주형)"]
        if "num" in kinds:
            corr += ["다계열상관(polyserial, 범주형–연속형)", "피어슨 공분산(연속형–연속형)"]
        rows = [
            ("추정법", _EST_KO[est] + (" — Mplus 범주형 변수 기본" if est == "wlsmv" else "")),
            ("범주형 변수 (CATEGORICAL)", " / ".join(cats)),
            ("범주 부호화", "분석 표본의 관측 범주를 0, 1, …, K−1로 재부호화 (숫자는 오름차순, 순서형 범주는 지정 순서)"),
            ("모수화", (f"delta (Var(y*{' | x' if spec.qx else ''}) = 1/Δ², 척도요인 Δ {{y}}는 1로 고정 — 'y ~*~ y'로 자유, "
                      "y* 잔차분산은 이로부터 계산; Mplus·lavaan 기본)")
             if s["parameterization"] == "delta" else
             "theta (y* 잔차분산 = 1로 고정 — 'y ~~ y'로 자유, 척도 요인은 이로부터 계산)"),
            ("표본 통계량", "3단계 제한정보 추정(Muthén, 1984): 문턱값·" + "·".join(corr)
             + (", 공변량에 대한 프로빗·선형 회귀계수와 잔차 상관" if spec.qx else "")
             + f" — 통계량 {len(sem.ss.s)}개, 점근 공분산 Γ는 사례별 영향함수"),
            ("가중행렬", {"wlsmv": "W = diag(Γ)⁻¹ (대각 가중)", "wlsm": "W = diag(Γ)⁻¹ (대각 가중)",
                       "wls": "W = Γ⁻¹ (전체 가중행렬)", "ulsmv": "W = I (비가중)", "uls": "W = I (비가중)"}[est]),
            ("표준오차", "모형 기반 (Δ'WΔ)⁻¹/N" if est == "wls" else
             "샌드위치 (Δ'WΔ)⁻¹Δ'WΓWΔ(Δ'WΔ)⁻¹/N (Mplus 8.8 출력 기준)"),
            ("χ² 검정", {"wlsmv": "평균·분산 보정(scaled-shifted, Asparouhov & Muthén, 2010)",
                       "ulsmv": "평균·분산 보정(scaled-shifted)", "wlsm": "평균 보정(Satorra–Bentler)",
                       "wls": "T = N·F", "uls": "제공하지 않음 (Mplus 규칙: ULSMV 사용)"}[est]),
            ("결측 처리", "목록별 삭제 (모형 변수 중 하나라도 결측인 사례 제외)"),
            ("공변량(x) 처리", ("공변량에 조건부 추정(Mplus·lavaan 기본): y*·연속형 종속변수를 x에 대해 프로빗·선형 회귀, "
                             "잔차 상관 사용 — " + ", ".join(spec.ox)) if spec.qx else
             ("공변량을 결합 상관구조에 포함(conditionalX = False, lavaan conditional.x = FALSE·fixed.x = FALSE; "
              "Mplus 기본과 다름)" if spec.x_joint else "관측 공변량 없음")),
            ("평균구조", "범주형 변수: 문턱값 자유, y* 절편 0 / 연속형 변수: 절편 자유 / 잠재변수 평균 0"),
            ("요인 척도", "각 요인의 첫 지표 적재량 1로 고정 (지정자로 변경 가능)"),
            ("사례 수", f"{sem.N:,}명 사용 / 입력 {sem.n_input:,}명"),
            ("수렴", "수렴" if res.fit.converged else "수렴 기준 미충족"),
        ]
        return rows

    def _after_settings(self, res, der, s, boot, level, G, glabel, diagram_rows, std_kind):
        spec = res.spec
        wls = s["wls"]
        mlr = s["estimator"] == "mlr"
        fs = res.fitstats

        # ---------------------------------------------------------------- 4. R²
        if s["rsquare"]:
            r2 = der.rsquare_values(res.theta)
            if r2:
                f2 = lambda th: np.array([v for _, _, v in der.rsquare_values(th)])
                val, sev = se.delta_se(f2, res.theta, res.vcov)
                if boot is not None:
                    sev = np.array([f2(th) for th in boot]).std(axis=0, ddof=1)
                rt = ResultsTable(name="rsquare", title="설명력 (R²)")
                if G > 1:
                    rt.add_column("group", "집단", col_type="text")
                rt.add_column("var", "변수", col_type="text")
                rt.add_column("type", "유형", col_type="text")
                rt.add_column("r2", "R²", col_type="number", fmt="zto")
                rt.add_column("se", "SE", col_type="number")
                rt.add_column("z", "Est./S.E.", col_type="number")
                rt.add_column("p", "p", col_type="number", fmt="pvalue")
                cat_set = set(spec.categorical)
                if cat_set:
                    rt.add_column("resvar", "y* 잔차분산", col_type="number")
                    rt.add_column("scale", "척도 요인", col_type="number")
                    A_, S_, M_ = res.sem.ram.matrices(res.theta, 0)
                    B_, Om_, BM_, _, _, _ = res.sem.ram.all_moments(res.theta, 0)
                    mu_c, Om_c, Pi_c, _ = res.sem.ram.conditional(res.theta, 0)
                for (g, v, _), x, e in zip(r2, val, sev):
                    z = x / e if e > 0 else float("nan")
                    typ = "잠재변수" if v in spec.latent else ("범주형 (y*)" if v in cat_set else "관측변수")
                    row = {"var": v, "type": typ, "r2": float(x), "se": float(e), "z": float(z), "p": se._pval(z)}
                    if cat_set:
                        if v in cat_set:
                            k = spec.idx[v]
                            row["resvar"] = float(S_[k, k])
                            row["scale"] = float(1.0 / math.sqrt(Om_c[k, k])) if Om_c[k, k] > 0 else float("nan")
                        else:
                            row["resvar"] = float("nan")
                            row["scale"] = float("nan")
                    if G > 1:
                        row["group"] = glabel(g)
                    rt.add_row(row)
                rt.add_note("R² = 1 − 잔차분산/모형 내재 분산, SE는 델타법" + (" (붓스트랩 SE)" if boot is not None else "") + "."
                            + (" 범주형 변수는 잠재반응변수 y*의 R²(Mplus R-SQUARE), 척도 요인 Δ = 1/√Var(y*"
                               + ("| x" if spec.qx else "") + ")." if cat_set else ""))
                self.add_table(rt)

        # ---------------------------------------------------------------- 5. defined parameters
        if spec.defined:
            val, sev = se.delta_se(der.defined_values, res.theta, res.vcov)
            bd = None
            if boot is not None:
                bd = np.array([der.defined_values(th) for th in boot])
                sev = bd.std(axis=0, ddof=1)
            lo, hi = se.boot_ci(val, bd, sev, s["ciType"] if bd is not None else "symmetric", level)
            dt = ResultsTable(name="defined", title="정의 모수 (Mplus MODEL CONSTRAINT NEW)")
            dt.add_column("name", "이름", col_type="text")
            dt.add_column("expr", "정의식", col_type="text")
            for c, ti in (("est", "추정치"), ("se", "SE"), ("z", "Est./S.E.")):
                dt.add_column(c, ti, col_type="number")
            dt.add_column("p", "p", col_type="number", fmt="pvalue")
            dt.add_column("ci_lo", f"{level:.0%} CI 하한", col_type="number")
            dt.add_column("ci_hi", f"{level:.0%} CI 상한", col_type="number")
            for (name, expr), x, e, a, b in zip(spec.defined, val, sev, lo, hi):
                z = x / e if e > 0 else float("nan")
                dt.add_row({"name": name, "expr": expr, "est": float(x), "se": float(e), "z": float(z), "p": se._pval(z),
                            "ci_lo": float(a), "ci_hi": float(b)})
            dt.add_note("SE는 델타법" + (f", 붓스트랩 {boot.shape[0]}회 SE·신뢰구간({s['ciType']})" if boot is not None else "")
                        + ". 신뢰구간 기본은 대칭(추정치 ± z·SE, Mplus CINTERVAL 기본).")
            dt.mark_coefficients(estimate="est", se="se", p="p", stat="z", ci=("ci_lo", "ci_hi"), label="name")
            self.add_table(dt)

        # ---------------------------------------------------------------- 6. indirect effects
        if s["indirect"]:
            self._indirect_table(res, der, s, boot, level)

        # ---------------------------------------------------------------- 7. modification indices
        if s["modIndices"]:
            if wls:
                from . import sem_wls
                mis = sem_wls.modification_indices_wls(res, s["miMin"])
            else:
                mis = se.modification_indices(res, s["miMin"])
            mt = ResultsTable(name="modindices", title="수정지수 (Model Modification Indices)")
            if G > 1:
                mt.add_column("group", "집단", col_type="text")
            mt.add_column("kind", "유형", col_type="text")
            mt.add_column("mplus", "Mplus 표기", col_type="text")
            mt.add_column("lval", "좌변", col_type="text")
            mt.add_column("op", "연산자", col_type="text")
            mt.add_column("rval", "우변", col_type="text")
            mt.add_column("mi", "M.I.", col_type="number")
            mt.add_column("epc", "E.P.C.", col_type="number")
            mt.add_column("std_epc", "StdYX E.P.C.", col_type="number")
            for m in mis:
                row = {"kind": _KIND_KO[m["kind"]], "mplus": _mplus_text(m["lhs"], m["op"], m["rhs"]), "lval": m["lhs"],
                       "op": m["op"], "rval": m["rhs"], "mi": m["mi"], "epc": m["epc"], "std_epc": m["std_epc"]}
                if G > 1:
                    row["group"] = glabel(m["group"])
                mt.add_row(row)
            if not mis:
                mt.add_note(f"M.I. ≥ {s['miMin']:g}인 모수 없음.")
            mt.add_note(f"M.I.: 해당 모수를 자유 추정할 때 기대되는 χ² 감소량(기대 정보행렬 기반 점수검정), 표시 기준 M.I. ≥ {s['miMin']:g}"
                        "(Mplus 기본 10). 후보: 교차 적재, 잠재변수 간 회귀, 잔차 공분산"
                        + (", 집단 간 동일성 제약 해제" if G > 1 else "") + ". 이론적 근거 없는 수정은 지양."
                        + (" WLS 계열: 보정 전 χ²(T = (N − 1)·F) 척도의 점수검정(Δ'WΔ 정보행렬, lavaan modindices와 같은 정의)."
                           if wls else ""))
            self.add_table(mt)

        # ---------------------------------------------------------------- 7b. sample statistics (Mplus SAMPSTAT)
        if wls and s["sampstat"]:
            self._sampstat_table(res)

        # ---------------------------------------------------------------- 8. invariance testing
        if s["invarianceTest"]:
            if not s["group"]:
                raise ValueError("측정동일성 검정(invarianceTest)에는 집단 변수(group) 필요")
            self._invariance_table(s)

        # ---------------------------------------------------------------- plot + narrative
        if s["diagram"] and diagram_rows:
            self.add_plot("경로도", plot_sem_diagram(pd.DataFrame(diagram_rows), title="구조방정식 경로도"))
        p_txt = "< .001" if (np.isfinite(fs["pvalue"]) and fs["pvalue"] < 0.001) else (f"= {fs['pvalue']:.3f}" if np.isfinite(fs["pvalue"]) else "— (포화모형)")
        est_txt = s["estimator"].upper()
        if wls:
            est_txt += f", 범주형 변수 {len(spec.categorical)}개, {s['parameterization']} 모수화"
        if s["estimator"] == "uls":
            self.apa_narrative = (f"구조방정식모형(ULS, 범주형 변수 {len(spec.categorical)}개, 목록별 삭제, N = {res.sem.N:,}): "
                                  f"χ² 검정 없음, SRMR = {fs['srmr']:.3f}.")
        else:
            self.apa_narrative = (
                f"구조방정식모형({est_txt}, {'FIML' if s['missing'] == 'fiml' else '목록별 삭제'}, N = {res.sem.N:,}) 적합도: "
                f"χ²({fs['df']}) = {fs['chi2_test']:.2f}, p {p_txt}, CFI = {fs['cfi']:.3f}, TLI = {fs['tli']:.3f}, "
                f"RMSEA = {fs['rmsea']:.3f} [90% CI {fs['rmsea_lo']:.3f}, {fs['rmsea_hi']:.3f}], SRMR = {fs['srmr']:.3f}."
            )

    # ------------------------------------------------------------------ WLS-family tables
    def _fit_table_wls(self, fs, s):
        est = s["estimator"]
        ft = ResultsTable(name="fit", title="모형 적합도 (Model Fit Indices)")
        ft.add_column("section", "구분", col_type="text")
        ft.add_column("index", "지수", col_type="text")
        ft.add_column("value", "값", col_type="number")
        lab = {"wlsmv": "WLSMV 평균·분산 보정", "ulsmv": "ULSMV 평균·분산 보정", "wlsm": "WLSM 평균 보정",
               "wls": "WLS", "uls": "ULS"}[est]
        rows = [("모수", "자유 모수 수", fs["npar"]), ("모수", "표본 통계량 수", fs["nstat"])]
        if est != "uls":
            rows += [("χ² 모형 검정", f"χ² (모형 적합도 검정) — {lab}", fs["chi2_test"]),
                     ("χ² 모형 검정", "자유도 (df)", fs["df"]), ("χ² 모형 검정", "p (χ²)", fs["pvalue"])]
            if est in ("wlsmv", "ulsmv"):
                rows += [("χ² 모형 검정", "척도 보정 계수 (1/a)", fs["scaling"]), ("χ² 모형 검정", "이동 모수 (b)", fs["shift"])]
            elif est == "wlsm":
                rows += [("χ² 모형 검정", "척도 보정 계수 (c)", fs["scaling"])]
            if est != "wls":
                rows += [("χ² 모형 검정", "보정 전 χ² (T = N·F)", fs["chi2"])]
            rows += [("RMSEA", "RMSEA", fs["rmsea"]), ("RMSEA", "RMSEA 90% CI 하한", fs["rmsea_lo"]),
                     ("RMSEA", "RMSEA 90% CI 상한", fs["rmsea_hi"]), ("RMSEA", "P(RMSEA ≤ .05)", fs["rmsea_p"]),
                     ("CFI/TLI", "Comparative Fit Index (CFI)", fs["cfi"]), ("CFI/TLI", "Tucker-Lewis Index (TLI)", fs["tli"]),
                     ("기저모형", "기저모형 χ²", fs["chi2_base"]), ("기저모형", "기저모형 df", fs["df_base"]),
                     ("기저모형", "기저모형 p", fs["p_base"])]
        else:
            rows += [("χ² 모형 검정", "자유도 (df)", fs["df"])]
        rows += [("SRMR", "SRMR", fs["srmr"])]
        if est in ("wlsmv", "wlsm", "wls"):
            rows += [("WRMR", "WRMR", fs["wrmr"])]
        rows += [("사례", "사례 수 (N)", fs["N"])]
        for sec, idx, val in rows:
            ft.add_row({"section": sec, "index": idx, "value": float(val) if val is not None else float("nan")})
        if fs["df"] == 0:
            ft.add_note("포화모형(df = 0): χ² = 0, RMSEA = 0, CFI = TLI = 1로 표시.")
        if est == "uls":
            ft.add_note("ULS: χ² 검정과 χ² 기반 적합도 지수 미제공(Mplus 규칙) — 검정이 필요하면 estimator='ulsmv'.")
        else:
            ft.add_note("F = (s − σ(θ))'W(s − σ(θ)), s는 문턱값·상관 등 표본 통계량, 보정 전 T = N·F "
                        "(Mplus 출력의 최적 함수값은 F/2). "
                        + {"wlsmv": "WLSMV·ULSMV χ² = a·T + b, a = √(df/tr((UΓ)²)), b = df − a·tr(UΓ) "
                                    "(Asparouhov & Muthén, 2010; df는 통상의 자유도).",
                           "ulsmv": "ULSMV χ² = a·T + b, a = √(df/tr((UΓ)²)), b = df − a·tr(UΓ).",
                           "wlsm": "WLSM χ² = T/c, c = tr(UΓ)/df.", "wls": "WLS χ² = T (W = Γ⁻¹)."}[est]
                        + " RMSEA = √(max(χ² − df, 0)/(df·N)), TLI는 1을 넘으면 1로 표시(Mplus 출력 방식). "
                        "CFI·TLI 기저모형은 문턱값·연속형 평균·분산만 자유인 독립모형(보정 방식 동일). "
                        "보정 χ²의 차이는 χ² 차이검정에 그대로 사용 불가(Mplus DIFFTEST 방식 필요).")
        ft.add_note("SRMR: Mplus 8.1 이후 정의(Asparouhov & Muthén, 2018) — (y*, y, x) 결합 상관 잔차, 연속형 종속변수의 "
                    "표준화 평균·분산 잔차, 범주형 변수의 범주 확률 잔차를 0이 아닌 잔차 수로 나눈 제곱평균제곱근."
                    + (" WRMR = √(T/표본 통계량 수): Mplus 8.1부터 출력하지 않는 지수(대표본에서 성능 저하, "
                       "DiStefano et al., 2018)로 참고용." if est in ("wlsmv", "wlsm", "wls") else ""))
        self.add_table(ft)

    def _sampstat_table(self, res):
        ss = res.sem.ss
        st = ResultsTable(name="sampstat", title="표본 통계량 (Mplus SAMPSTAT: 문턱값·다분·다계열 상관)")
        st.add_column("kind", "구분", col_type="text")
        st.add_column("var1", "변수", col_type="text")
        st.add_column("var2", "변수 2 / 문턱값", col_type="text")
        st.add_column("value", "추정치", col_type="number")
        st.add_column("se", "SE", col_type="number")
        kinds = {"th": "문턱값", "mean": "평균(절편)", "slope": "회귀계수 (y* on x)", "var": "분산(잔차분산)", "cov": "상관·공분산"}
        se_ = np.sqrt(np.clip(np.diag(ss.Gamma), 0, None) / ss.n)
        for k, (kind, lab) in enumerate(ss.labels):
            if kind == "th":
                v1, v2 = lab[0], f"{lab[0]}${lab[1] + 1}"
            elif kind == "cov":
                v1, v2 = lab[1], lab[0]
                i, j = ss.names.index(lab[0]), ss.names.index(lab[1])
                ki = ("범주형" if ss.types[i] == "ord" else "연속형") + "–" + ("범주형" if ss.types[j] == "ord" else "연속형")
                kind_txt = {"범주형–범주형": "다분상관 (polychoric)", "연속형–범주형": "다계열상관 (polyserial)",
                            "범주형–연속형": "다계열상관 (polyserial)", "연속형–연속형": "공분산 (Pearson)"}[ki]
                st.add_row({"kind": kind_txt, "var1": v1, "var2": v2, "value": float(ss.s[k]), "se": float(se_[k])})
                continue
            elif kind == "slope":
                v1, v2 = lab[0], lab[1]
            else:
                v1, v2 = lab, ""
            st.add_row({"kind": kinds[kind], "var1": v1, "var2": v2, "value": float(ss.s[k]), "se": float(se_[k])})
        st.add_note("다계열상관 행은 상관 × 연속형 변수의 표준편차(공분산 척도). SE = √(Γ의 대각/N)"
                    + (". 공변량이 있으면 문턱값·상관은 x에 대한 조건부(잔차) 통계량" if ss.xnames else "") + ".")
        self.add_table(st)

    # ------------------------------------------------------------------ helpers
    def _indirect_table(self, res, der, s, boot, level):
        spec = res.spec
        G = spec.G
        items = []  # (g, x, y, label, list of paths, kind)
        for itm in s["indirect"]:
            x, y, through = se.parse_indirect(itm)
            for nm in (x, y):
                if nm not in spec.idx:
                    raise ValueError(f"간접효과 변수 '{nm}' 없음 (모형 변수: {', '.join(spec.vars)})")
            for g in range(G):
                paths = der.paths(g, x, y, through)
                if not paths:
                    raise ValueError(f"간접효과: {x} → {y} 경로 없음" + (f" ({' → '.join(through)} 경유)" if through else ""))
                direct = [p for p in paths if len(p) == 1]
                ind = [p for p in paths if len(p) > 1]
                for p in ind:
                    items.append((g, x, y, "특정 간접효과", " → ".join(der.path_names(p)), [p]))
                if through is None:
                    if ind:
                        items.append((g, x, y, "총 간접효과", f"{x} → … → {y}", ind))
                    if direct:
                        items.append((g, x, y, "직접효과", f"{x} → {y}", direct))
                    items.append((g, x, y, "총효과", f"{x} ⇒ {y}", paths))

        def values(th):
            return np.array([sum(der.path_value(th, p) for p in pl) for (_, _, _, _, _, pl) in items])

        def std_values(th):
            out = []
            for (g, x, y, _, _, pl) in items:
                A, S, M, var, BM = der._group_stats(th, g)
                sx = math.sqrt(var[spec.idx[x]])
                sy = math.sqrt(var[spec.idx[y]])
                out.append(sum(der.path_value(th, p) for p in pl) * sx / sy)
            return np.array(out)

        val, sev = se.delta_se(values, res.theta, res.vcov)
        sval, ssev = se.delta_se(std_values, res.theta, res.vcov)
        bd = None
        if boot is not None:
            bd = np.array([values(th) for th in boot])
            sev = bd.std(axis=0, ddof=1)
            ssev = np.array([std_values(th) for th in boot]).std(axis=0, ddof=1)
        lo, hi = se.boot_ci(val, bd, sev, s["ciType"] if bd is not None else "symmetric", level)
        it = ResultsTable(name="indirect", title="간접효과 (Mplus MODEL INDIRECT)")
        if G > 1:
            it.add_column("group", "집단", col_type="text")
        it.add_column("type", "효과", col_type="text")
        it.add_column("path", "경로", col_type="text")
        for c, ti in (("est", "추정치"), ("se", "SE"), ("z", "Est./S.E.")):
            it.add_column(c, ti, col_type="number")
        it.add_column("p", "p", col_type="number", fmt="pvalue")
        it.add_column("ci_lo", f"{level:.0%} CI 하한", col_type="number")
        it.add_column("ci_hi", f"{level:.0%} CI 상한", col_type="number")
        it.add_column("std", "STDYX", col_type="number", fmt="zto")
        it.add_column("std_se", "STDYX SE", col_type="number")
        for k, (g, x, y, kind, ptxt, _) in enumerate(items):
            z = val[k] / sev[k] if sev[k] > 0 else float("nan")
            row = {"type": kind, "path": ptxt, "est": float(val[k]), "se": float(sev[k]), "z": float(z), "p": se._pval(z),
                   "ci_lo": float(lo[k]), "ci_hi": float(hi[k]), "std": float(sval[k]), "std_se": float(ssev[k])}
            if G > 1:
                row["group"] = str(res.sem.group_levels[g])
            it.add_row(row)
        it.add_note("특정 간접효과 = 경로 계수의 곱, 총효과 = 직접효과 + 총 간접효과. SE는 델타법"
                    + (f", 붓스트랩 {boot.shape[0]}회 SE·신뢰구간({s['ciType']})" if boot is not None else
                       (" (붓스트랩 신뢰구간은 bootstrap=1000, ciType='bc' 지정)" if s["estimator"] == "ml" else
                        (" (WLS 계열: 로버스트 공분산 기반 델타법)" if s["wls"] else
                         " (붓스트랩은 ML 추정에서만 사용: estimator='ml', bootstrap=1000)"))) + "."
                    + (" 범주형 매개·결과변수의 효과는 잠재반응변수 y* 척도(프로빗)의 곱(Mplus WLSMV 방식)." if s["wls"] else ""))
        it.mark_coefficients(estimate="est", se="se", p="p", stat="z", std_estimate="std", ci=("ci_lo", "ci_hi"),
                             label=["group", "type", "path"])
        self.add_table(it)

    def _invariance_table(self, s):
        models = []
        for inv in ("configural", "metric", "scalar"):
            r = se.estimate(self.dataset, self.model_syntax, group=s["group"], estimator=s["estimator"], missing=s["missing"],
                            invariance=inv, information=s["information"], group_order=s["groupOrder"])
            models.append((inv, r))
        tt = ResultsTable(name="invariance", title="측정동일성 비교 (configural → metric → scalar)")
        tt.add_column("model", "모형", col_type="text")
        for c, ti, ty in (("npar", "모수 수", "integer"), ("chi2", "χ²", "number"), ("df", "df", "integer"),
                          ("p", "p", "number"), ("cfi", "CFI", "number"), ("tli", "TLI", "number"),
                          ("rmsea", "RMSEA", "number"), ("srmr", "SRMR", "number"), ("dchi2", "Δχ²", "number"),
                          ("ddf", "Δdf", "integer"), ("dp", "Δp", "number"), ("dcfi", "ΔCFI", "number"),
                          ("drmsea", "ΔRMSEA", "number")):
            tt.add_column(c, ti, col_type=ty, fmt="pvalue" if c in ("p", "dp") else "")
        prev = None
        for inv, r in models:
            f = r.fitstats
            row = {"model": se.INVARIANCE_LABELS[inv], "npar": f["npar"], "chi2": f["chi2_test"], "df": f["df"], "p": f["pvalue"],
                   "cfi": f["cfi"], "tli": f["tli"], "rmsea": f["rmsea"], "srmr": f["srmr"]}
            if prev is not None:
                pf = prev.fitstats
                p0, p1 = f["npar"], pf["npar"]
                d = 2.0 * (pf["ll0"] - f["ll0"])
                if s["estimator"] == "mlr" and "c0" in f and "c0" in pf:
                    cd = (p0 * f["c0"] - p1 * pf["c0"]) / (p0 - p1)
                    d = d / cd if cd > 0 else float("nan")
                ddf = int(p1 - p0)
                row.update({"dchi2": max(d, 0.0), "ddf": ddf, "dp": float(se.stats.chi2.sf(max(d, 0.0), ddf)) if ddf > 0 else float("nan"),
                            "dcfi": f["cfi"] - pf["cfi"], "drmsea": f["rmsea"] - pf["rmsea"]})
            tt.add_row(row)
            prev = r
        tt.add_note("인접 모형 비교: Δχ² = 2(LL 덜 제약 − LL 더 제약)" + (", MLR은 척도 보정 차이검정 TRd = Δχ²/cd, "
                    "cd = (p0·c0 − p1·c1)/(p0 − p1)" if s["estimator"] == "mlr" else "") + ". ΔCFI ≤ −.01은 동일성 기각 경향(Cheung & Rensvold, 2002).")
        self.add_table(tt)
