"""
OpenStat Studio — side-by-side model comparison table (Stata `esttab` / hierarchical regression layout).
License: MIT

Rows are the union of model terms (first-appearance order) followed by model statistics (N, R², AIC …);
each model contributes a column group (B · SE · p, optionally β). The table is marked as a coefficient
table, so the social-science export format turns each group into "B*** (SE)" cells or SE rows.

Sources: coefficient-table results (`BaseAnalysis.comparison_models`). A hierarchical linear regression
with k blocks contributes k models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base import BaseAnalysis, ResultsTable, resolve_choice

# key -> (Korean label, English label, row format)
STAT_DEFS: Dict[str, tuple] = {
    "N": ("N", "N", "integer"),
    "Groups": ("집단 수", "Groups", "integer"),
    "Periods": ("시점 수", "Periods", "integer"),
    "Events": ("사건 수", "Events", "integer"),
    "R2": ("R²", "R²", "zto"),
    "R2_adj": ("수정 R²", "Adj. R²", "zto"),
    "R2_change": ("ΔR²", "ΔR²", "zto"),
    "F": ("F", "F", ""),
    "F_change": ("ΔF", "ΔF", ""),
    "R2_within": ("R² (집단 내)", "R² (within)", "zto"),
    "R2_between": ("R² (집단 간)", "R² (between)", "zto"),
    "R2_overall": ("R² (전체)", "R² (overall)", "zto"),
    "R2_McF": ("McFadden R²", "McFadden R²", "zto"),
    "R2_N": ("Nagelkerke R²", "Nagelkerke R²", "zto"),
    "R2_pseudo": ("유사 R²", "Pseudo R²", "zto"),
    "Chi2": ("χ²", "χ²", ""),
    "df": ("df", "df", "integer"),
    "CFI": ("CFI", "CFI", "zto"),
    "TLI": ("TLI", "TLI", "zto"),
    "RMSEA": ("RMSEA", "RMSEA", "zto"),
    "SRMR": ("SRMR", "SRMR", "zto"),
    "LL": ("로그우도", "Log likelihood", ""),
    "Deviance": ("이탈도", "Deviance", ""),
    "AIC": ("AIC", "AIC", ""),
    "BIC": ("BIC", "BIC", ""),
    "ICC": ("ICC", "ICC", "zto"),
    "Var_between": ("집단 간 분산", "Between variance", ""),
    "Var_within": ("집단 내 분산", "Within variance", ""),
    "Concordance": ("C 지수", "Concordance", "zto"),
    "rho": ("ρ", "ρ", ""),
    "lambda": ("λ", "λ", ""),
}
AUTO_STATS = ("N", "Groups", "Events", "R2", "R2_adj", "R2_change", "F", "R2_within", "R2_McF", "R2_N", "R2_pseudo",
              "Chi2", "df", "CFI", "TLI", "RMSEA", "SRMR", "ICC", "Concordance", "LL", "AIC", "BIC")
ESTIMATES = ("b", "beta", "both", "exp")
_EST_ALIASES = {"coef": "b", "B": "b", "estimate": "b", "β": "beta", "std": "beta", "standardized": "beta",
                "b+beta": "both", "or": "exp", "hr": "exp", "odds": "exp", "exp(b)": "exp"}


def _num(v: Any) -> Optional[float]:
    if v is None or v == "" or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(f) or np.isinf(f)) else f


def _split(v: Any) -> tuple:
    """Stat entry -> (value, p)."""
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return v[0], v[1]
    return v, None


class ModelComparison(BaseAnalysis):
    """Side-by-side table of several models' coefficients and fit statistics."""

    def __init__(
        self,
        models: Sequence[Dict[str, Any]],
        names: Optional[Sequence[str]] = None,
        stats: Any = "auto",
        estimate: str = "b",
        labels: Optional[Dict[str, str]] = None,
        terms: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
        lang: str = "ko",
        dataset: Optional[Any] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        lang = resolve_choice("언어(lang)", lang, ("ko", "en"), default="ko")
        super().__init__(
            name="compareModels",
            title=title or ("모형 비교" if lang == "ko" else "Model Comparison"),
            dataset=dataset,
            options=dict(options or {}),
        )
        self.models = [dict(m) for m in models]
        if len(self.models) < 1:
            raise ValueError("모형 비교: 모형 1개 이상 필요")
        self.estimate = resolve_choice("계수 표시(estimate)", estimate, ESTIMATES, _EST_ALIASES, default="b")
        self.lang = lang
        k = len(self.models)
        if names is None or names == "" or names == []:
            names = [(f"모형 {i}" if lang == "ko" else f"Model {i}") for i in range(1, k + 1)]
        elif isinstance(names, str):
            names = [n.strip() for n in names.split(",")]
        names = [str(n) for n in names]
        if len(names) != k:
            raise ValueError(f"모형 비교: 모형 이름 {len(names)}개, 모형 {k}개 — 개수 일치 필요")
        if len(set(names)) != len(names):
            raise ValueError("모형 비교: 모형 이름 중복")
        self.names = names
        self.labels = dict(labels or {})
        self.terms = list(terms) if terms else None
        self.stats_keys = self._resolve_stats(stats)
        self.run()

    # -------------------------------------------------------------------------------------------
    def _resolve_stats(self, stats: Any) -> List[str]:
        present = []
        for m in self.models:
            for key in (m.get("stats") or {}):
                if key not in present:
                    present.append(key)
        if stats in (None, "auto", ""):
            return [k for k in AUTO_STATS if k in present]
        if stats in ("none", False):
            return []
        if isinstance(stats, str):
            stats = [s.strip() for s in stats.split(",") if s.strip()]
        out = []
        lookup = {k.lower(): k for k in STAT_DEFS}
        for s in stats:
            key = lookup.get(str(s).strip().lower())
            if key is None:
                raise ValueError(f"모형 비교: 알 수 없는 통계량 '{s}' (가능: {', '.join(STAT_DEFS)})")
            out.append(key)
        return out

    def run(self):
        ko = self.lang == "ko"
        k = len(self.models)
        est_title = {"b": "B", "beta": "β", "both": "B", "exp": "Exp(B)"}[self.estimate]

        t = ResultsTable(name="compare", title=self.title)
        t.add_column("term", "변수" if ko else "Variable", col_type="text")
        groups = []
        for i, name in enumerate(self.names, start=1):
            t.add_column(f"m{i}_est", est_title, col_type="number", super_title=name)
            g = {"estimate": f"m{i}_est", "p": f"m{i}_p"}
            if self.estimate in ("b", "both"):
                t.add_column(f"m{i}_se", "SE", col_type="number", super_title=name)
                g["se"] = f"m{i}_se"
            if self.estimate == "both":
                t.add_column(f"m{i}_std", "β", col_type="number", fmt="zto", super_title=name)
                g["std_estimate"] = f"m{i}_std"
            t.add_column(f"m{i}_p", "p", col_type="number", fmt="pvalue", super_title=name)
            groups.append(g)

        # term order: union in first-appearance order (or the requested subset/order)
        order: List[str] = []
        per_model: List[Dict[str, Dict[str, Any]]] = []
        for m in self.models:
            d = {}
            for c in m.get("coefs") or []:
                term = str(c.get("term", ""))
                d[term] = c
                if term not in order:
                    order.append(term)
            per_model.append(d)
        if self.terms:
            missing = [x for x in self.terms if x not in order]
            if missing:
                raise ValueError(f"모형 비교: 모형에 없는 변수 {', '.join(missing)} (가능: {', '.join(order)})")
            order = list(self.terms)

        source_key = {"b": "est", "beta": "std", "both": "est", "exp": "exp"}[self.estimate]
        for term in order:
            row: Dict[str, Any] = {"term": self.labels.get(term, term)}
            for i, d in enumerate(per_model, start=1):
                c = d.get(term)
                if c is None:
                    row.update({f"m{i}_est": "", f"m{i}_p": ""})
                    if self.estimate in ("b", "both"):
                        row[f"m{i}_se"] = ""
                    if self.estimate == "both":
                        row[f"m{i}_std"] = ""
                    continue
                v = _num(c.get(source_key))
                row[f"m{i}_est"] = v if v is not None else ""
                row[f"m{i}_p"] = _num(c.get("p")) if (v is not None and _num(c.get("p")) is not None) else ""
                if self.estimate in ("b", "both"):
                    se = _num(c.get("se"))
                    row[f"m{i}_se"] = se if se is not None else ""
                if self.estimate == "both":
                    sv = _num(c.get("std"))
                    row[f"m{i}_std"] = sv if sv is not None else ""
            t.add_row(row)

        def blank_row(term: str) -> Dict[str, Any]:
            row: Dict[str, Any] = {"term": term, "_section": "stats"}
            for i in range(1, k + 1):
                row.update({f"m{i}_est": "", f"m{i}_p": ""})
                if self.estimate in ("b", "both"):
                    row[f"m{i}_se"] = ""
                if self.estimate == "both":
                    row[f"m{i}_std"] = ""
            return row

        # dependent variables
        deps = [str(m.get("dep") or "") for m in self.models]
        if len(set(deps)) > 1 and any(deps):
            row = blank_row("종속변수" if ko else "Dependent variable")
            for i, dv in enumerate(deps, start=1):
                row[f"m{i}_est"] = dv
            t.add_row(row)
        for key in self.stats_keys:
            lab_ko, lab_en, rf = STAT_DEFS[key]
            row = blank_row(lab_ko if ko else lab_en)
            if rf:
                row["_rowfmt"] = rf
            for i, m in enumerate(self.models, start=1):
                v, p = _split((m.get("stats") or {}).get(key))
                fv = _num(v)
                row[f"m{i}_est"] = fv if fv is not None else ""
                row[f"m{i}_p"] = _num(p) if (fv is not None and _num(p) is not None) else ""
            t.add_row(row)

        t.kind = "coefficients"
        t.coef_groups = groups
        t.coef_label = ["term"]
        if len(set(deps)) == 1 and deps[0]:
            t.add_note(("종속변수: " if ko else "Dependent variable: ") + deps[0] + ".")
        srcs = []
        for name, m in zip(self.names, self.models):
            src = m.get("source") or ""
            if src:
                srcs.append(f"{name} = {src}" + (f" ({m['block']})" if m.get("block") else ""))
        if srcs:
            t.add_note(("모형 출처: " if ko else "Models: ") + "; ".join(srcs) + ".")
        if self.estimate == "beta":
            t.add_note("β = 표준화 계수, 유의확률은 비표준화 계수 검정 기준." if ko else "β = standardized coefficients; p from the unstandardized test.")
        if self.estimate == "exp":
            t.add_note("Exp(B) = 승산비·위험비 등 지수 변환 계수." if ko else "Exp(B) = exponentiated coefficients (odds or hazard ratios).")
        self.add_table(t)
        self.apa_narrative = ""


def expand_results(results: Sequence[BaseAnalysis], table: Optional[str] = None) -> List[Dict[str, Any]]:
    """Comparison model entries of several results (hierarchical regressions expand to one model per block)."""
    out: List[Dict[str, Any]] = []
    for r in results:
        ms = r.comparison_models(table)
        for j, m in enumerate(ms, start=1):
            m = dict(m)
            m.setdefault("source", r.title)
            if len(ms) > 1:
                m["block"] = f"블록 {j}"
            out.append(m)
    return out


def compare_results(results: Sequence[BaseAnalysis], names: Optional[Sequence[str]] = None, stats: Any = "auto",
                    estimate: str = "b", labels: Optional[Dict[str, str]] = None, terms: Optional[Sequence[str]] = None,
                    table: Optional[str] = None, title: Optional[str] = None, lang: str = "ko",
                    options: Optional[Dict[str, Any]] = None) -> ModelComparison:
    """Comparison table from already-estimated results (no re-estimation)."""
    if not results:
        raise ValueError("모형 비교: 비교할 분석 결과 1개 이상 필요")
    ds = getattr(results[0], "dataset", None)
    return ModelComparison(expand_results(results, table), names=names, stats=stats, estimate=estimate, labels=labels,
                           terms=terms, title=title, lang=lang, dataset=ds, options=options)


def normalize_model_spec(spec: Any) -> Dict[str, Any]:
    """{"analysis": type, **params} from either that form or {"analysis_type": ..., "params": {...}}."""
    if not isinstance(spec, dict):
        raise ValueError(f"모형 비교: 모형 지정은 딕셔너리여야 함 (입력값 {spec!r})")
    if "analysis_type" in spec:
        return {"analysis": spec["analysis_type"], **dict(spec.get("params") or {})}
    if "analysis" not in spec:
        raise ValueError("모형 비교: 각 모형에 'analysis'(분석 종류) 지정 필요")
    return dict(spec)
