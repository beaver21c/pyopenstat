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
    "r_square": "rsquare", "rSquare": "rsquare",
}
# keyword options read by SEM (registry/session can pass these through unchanged)
SEM_OPTION_KEYS = ("estimator", "missing", "information", "group", "invariance", "invarianceTest", "indirect",
                   "bootstrap", "ciType", "ciLevel", "modIndices", "miMin", "standardized", "rsquare", "seed",
                   "groupOrder")
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
    return f"{lhs} {op} {rhs}"


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
        est = resolve_choice("추정법", o.get("estimator"), ["ml", "mlr"], {"robust": "mlr", "ml_robust": "mlr"}, "ml")
        miss = resolve_choice("결측 처리", o.get("missing"), ["fiml", "listwise"],
                              {"ml": "fiml", "fml": "fiml", "available": "fiml", "listwise_deletion": "listwise",
                               "complete": "listwise"}, "fiml")
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
        }

    def _syntax(self, s: Dict[str, Any]) -> str:
        parts = [f'model="""\n{self.model_syntax.strip()}\n"""']
        if s["estimator"] != "ml":
            parts.append(f"estimator={s['estimator']!r}")
        if s["missing"] != "fiml":
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
        if s["information"] != "observed":
            parts.append(f"information={s['information']!r}")
        return "sess.sem(" + ", ".join(parts) + ")"

    # ------------------------------------------------------------------ main
    def run(self):
        s = self._settings()
        self.syntax_python = self._syntax(s)
        self.syntax_r = "lavaan::sem(model, data, estimator = 'ML', missing = 'fiml', meanstructure = TRUE)"
        try:
            res = se.estimate(self.dataset, self.model_syntax, group=s["group"], estimator=s["estimator"],
                              missing=s["missing"], invariance=s["invariance"], information=s["information"],
                              bootstrap=s["bootstrap"], seed=s["seed"], group_order=s["groupOrder"])
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
        for n, r in enumerate(spec.rows):
            val = res.theta[r.pid] if r.free else r.value
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
        mlr = s["estimator"] == "mlr"
        if s["fit"]:
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
            rows += [("RMSEA", "RMSEA", fs["rmsea"]), ("RMSEA", "RMSEA 90% CI 하한", fs["rmsea_lo"]),
                     ("RMSEA", "RMSEA 90% CI 상한", fs["rmsea_hi"]), ("RMSEA", "P(RMSEA ≤ .05)", fs["rmsea_p"]),
                     ("CFI/TLI", "Comparative Fit Index (CFI)", fs["cfi"]), ("CFI/TLI", "Tucker-Lewis Index (TLI)", fs["tli"]),
                     ("기저모형", "기저모형 χ²", fs["chi2_base"]), ("기저모형", "기저모형 df", fs["df_base"]),
                     ("기저모형", "기저모형 p", fs["p_base"]), ("SRMR", "SRMR", fs["srmr"]), ("사례", "사례 수 (N)", fs["N"])]
            for sec, idx, val in rows:
                ft.add_row({"section": sec, "index": idx, "value": float(val) if val is not None else float("nan")})
            if fs["df"] == 0:
                ft.add_note("포화모형(df = 0): χ² = 0, RMSEA = 0, CFI = TLI = 1로 표시.")
            ft.add_note("χ² = 2(H1 로그우도 − H0 로그우도), H1은 제약 없는 평균·공분산 모형"
                        + (" (FIML 결측 시 EM 추정)" if s["missing"] == "fiml" else "") + ". "
                        "RMSEA = √G·√(max(χ² − df, 0)/(df·N)), CFI·TLI 기저모형은 종속변수의 평균·분산만 자유(Mplus 기준). "
                        "SRMR은 상관·표준화 평균·분산 잔차 포함(Asparouhov & Muthén, 2018)"
                        + (", 공변량 항 제외" if spec.qx else "") + ".")
            if mlr:
                ft.add_note("MLR: 표준오차는 Huber–White 샌드위치, χ²는 Yuan–Bentler 척도 보정 "
                            "(척도 보정 계수 = (p1·c1 − p0·c0)/(p1 − p0)). RMSEA·CFI·TLI는 보정 χ² 기준.")
            self.add_table(ft)

        # ---------------------------------------------------------------- 3. settings
        at = ResultsTable(name="assumptions", title="분석 설정")
        at.add_column("item", "항목", col_type="text")
        at.add_column("value", "설정", col_type="text")
        rows = [
            ("추정법", "최대우도(ML)" if not mlr else "로버스트 최대우도(MLR)"),
            ("표준오차", ("붓스트랩 " + f"{s['bootstrap']}회" if s["bootstrap"] else
                         ("샌드위치(관측 정보행렬 기준)" if mlr else "역정보행렬")) + f", 정보행렬: {'관측' if s['information'] == 'observed' else '기대'}"),
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
        for item, val in rows:
            at.add_row({"item": item, "value": val})
        for nt in res.notes:
            at.add_note(nt)
        self.add_table(at)

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
                for (g, v, _), x, e in zip(r2, val, sev):
                    z = x / e if e > 0 else float("nan")
                    row = {"var": v, "type": "잠재변수" if v in spec.latent else "관측변수", "r2": float(x), "se": float(e),
                           "z": float(z), "p": se._pval(z)}
                    if G > 1:
                        row["group"] = glabel(g)
                    rt.add_row(row)
                rt.add_note("R² = 1 − 잔차분산/모형 내재 분산, SE는 델타법" + (" (붓스트랩 SE)" if boot is not None else "") + ".")
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
                        + (", 집단 간 동일성 제약 해제" if G > 1 else "") + ". 이론적 근거 없는 수정은 지양.")
            self.add_table(mt)

        # ---------------------------------------------------------------- 8. invariance testing
        if s["invarianceTest"]:
            if not s["group"]:
                raise ValueError("측정동일성 검정(invarianceTest)에는 집단 변수(group) 필요")
            self._invariance_table(s)

        # ---------------------------------------------------------------- plot + narrative
        if s["diagram"] and diagram_rows:
            self.add_plot("경로도", plot_sem_diagram(pd.DataFrame(diagram_rows), title="구조방정식 경로도"))
        p_txt = "< .001" if (np.isfinite(fs["pvalue"]) and fs["pvalue"] < 0.001) else (f"= {fs['pvalue']:.3f}" if np.isfinite(fs["pvalue"]) else "— (포화모형)")
        self.apa_narrative = (
            f"구조방정식모형({'MLR' if mlr else 'ML'}, {'FIML' if s['missing'] == 'fiml' else '목록별 삭제'}, N = {res.sem.N:,}) 적합도: "
            f"χ²({fs['df']}) = {fs['chi2_test']:.2f}, p {p_txt}, CFI = {fs['cfi']:.3f}, TLI = {fs['tli']:.3f}, "
            f"RMSEA = {fs['rmsea']:.3f} [90% CI {fs['rmsea_lo']:.3f}, {fs['rmsea_hi']:.3f}], SRMR = {fs['srmr']:.3f}."
        )

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
                        " (붓스트랩은 ML 추정에서만 사용: estimator='ml', bootstrap=1000)")) + ".")
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
