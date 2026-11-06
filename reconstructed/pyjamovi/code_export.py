"""Convert the session command record (see :mod:`command_log`) to Stata (.do) or R (.R) code on request.

The conversion is best effort: commands with a direct counterpart are written as Stata / R commands; the
rest are kept as comments that show the OpenStat call, so the file always lists every step in order.
Results may differ between programs because of estimators, defaults and missing-data handling.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional


def _lst(v: Any) -> List[str]:
    if v is None or v == "":
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if str(x).strip()]
    return [x for x in re.split(r"[,\s]+", str(v)) if x]


def _q(s: Any) -> str:
    return '"' + str(s).replace('"', '\\"') + '"'


# ----------------------------------------------------------------------------- expressions
_FN_STATA = {"LOG": "log10", "log10": "log10", "LN": "ln", "log": "ln", "ln": "ln", "EXP": "exp", "exp": "exp",
             "SQRT": "sqrt", "sqrt": "sqrt", "ABS": "abs", "abs": "abs", "ROUND": "round", "round": "round"}
_FN_R = {"LOG": "log10", "log10": "log10", "LN": "log", "log": "log", "ln": "log", "EXP": "exp", "exp": "exp",
         "SQRT": "sqrt", "sqrt": "sqrt", "ABS": "abs", "abs": "abs", "ROUND": "round", "round": "round",
         "IFELSE": "ifelse"}


def _expr_stata(expr: str) -> Optional[str]:
    e = str(expr)
    if re.search(r"\b(Z|z|MEAN|SUM|MIN|MAX|IFELSE)\s*\(", e):
        return None                      # needs egen / cond(): written as a comment
    for k, v in _FN_STATA.items():
        e = re.sub(rf"\b{k}\s*\(", v + "(", e)
    e = e.replace("**", "^")
    e = re.sub(r"\band\b", "&", e)
    e = re.sub(r"\bor\b", "|", e)
    e = re.sub(r"\bnot\b", "!", e)
    e = re.sub(r"'([^']*)'", r'"\1"', e)
    m = re.match(r"^\s*(\w+)\s+in\s+\[(.+)\]\s*$", e)
    if m:
        e = f"inlist({m.group(1)}, {m.group(2)})"
    return e


def _expr_r(expr: str) -> str:
    e = str(expr)
    for k, v in _FN_R.items():
        e = re.sub(rf"\b{k}\s*\(", v + "(", e)
    e = re.sub(r"\b[Zz]\s*\(\s*(\w+)\s*\)", r"as.numeric(scale(\1))", e)
    for fn, rfn in (("MEAN", "rowMeans"), ("SUM", "rowSums")):
        e = re.sub(rf"\b{fn}\s*\(([^()]*)\)", rf"{rfn}(cbind(\1), na.rm = TRUE)", e)
    e = e.replace("**", "^")
    e = re.sub(r"\band\b", "&", e)
    e = re.sub(r"\bor\b", "|", e)
    e = re.sub(r"\bnot\b", "!", e)
    m = re.match(r"^\s*(\w+)\s+in\s+\[(.+)\]\s*$", e)
    if m:
        e = f"{m.group(1)} %in% c({m.group(2)})"
    return e


def _comment(prefix: str, code: str) -> List[str]:
    return [f"{prefix} {ln}" for ln in str(code).splitlines()]


# ----------------------------------------------------------------------------- data source
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _delim(path: str) -> str:
    """Delimiter of a text data file (first line), ',' when unknown."""
    full = path if os.path.isabs(path) else os.path.join(_ROOT, path)
    try:
        with open(full, encoding="utf-8-sig", errors="replace") as f:
            head = f.readline()
    except OSError:
        return ","
    counts = {d: head.count(d) for d in (",", ";", "\t")}
    best = max(counts, key=counts.get)
    return best if counts[best] else ","


def _load_stata(path: str) -> List[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv", ".txt"):
        d = _delim(path)
        dopt = "" if d == "," else (" delimiters(tab)" if d == "\t" else f' delimiters("{d}")')
        return [f"import delimited {_q(path)}, clear varnames(1) encoding(UTF-8){dopt}"]
    if ext in (".xlsx", ".xls"):
        return [f"import excel {_q(path)}, firstrow clear"]
    if ext == ".dta":
        return [f"use {_q(path)}, clear"]
    if ext in (".sav", ".zsav", ".por"):
        return [f"import spss using {_q(path)}, clear"]
    return [f"* {path}: Stata에서 직접 읽을 수 없는 형식 — CSV 등으로 저장한 뒤 import delimited 사용"]


def _load_r(path: str) -> List[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt"):
        d = _delim(path)
        sep = "" if d == "," else (', sep = "\\t"' if d == "\t" else f', sep = "{d}"')
        return [f"data <- read.csv({_q(path)}{sep}, check.names = FALSE, fileEncoding = \"UTF-8-BOM\")"]
    if ext == ".tsv":
        return [f"data <- read.delim({_q(path)}, check.names = FALSE)"]
    if ext in (".xlsx", ".xls"):
        return [f"data <- as.data.frame(readxl::read_excel({_q(path)}))"]
    if ext == ".dta":
        return [f"data <- as.data.frame(haven::read_dta({_q(path)}))"]
    if ext in (".sav", ".zsav", ".por"):
        return [f"data <- as.data.frame(haven::read_sav({_q(path)}))"]
    if ext == ".omv":
        return [f"data <- jmvReadWrite::read_omv({_q(path)})"]
    return [f"# {path}: R에서 읽는 방법을 지정해야 하는 형식"]


# ----------------------------------------------------------------------------- data steps
def _op_stata(op: Dict[str, Any]) -> Optional[List[str]]:
    name = op.get("op")
    if name == "compute":
        e = _expr_stata(op.get("formula", ""))
        if e is None:
            return None
        v = op.get("col_name")
        return [f"capture drop {v}", f"generate {v} = {e}"]
    if name == "filter_rows":
        e = _expr_stata(op.get("condition", ""))
        return [f"keep if {e}"] if e else None
    if name == "rename_var":
        return [f"rename {op.get('old_name')} {op.get('new_name')}"]
    if name == "drop_vars":
        return ["drop " + " ".join(_lst(op.get("cols")))]
    if name == "standardize" and str(op.get("method", "zscore")) == "zscore":
        return [f"egen z_{c} = std({c})" for c in _lst(op.get("columns"))]
    if name == "reshape":
        stubs = _lst(op.get("stubnames")) or _lst(op.get("value_vars"))
        i = " ".join(_lst(op.get("id_vars")))
        j = op.get("time_col") or "time"
        d = op.get("direction", "long")
        return [f"reshape {d} {' '.join(stubs)}, i({i}) j({j})"]
    if name == "merge":
        kind = op.get("kind") or "1:1"
        using = op.get("other_source", "")
        lines = [f"* using 자료는 .dta로 저장되어 있어야 함: {using}"]
        lines.append(f"merge {kind} {' '.join(_lst(op.get('on')))} using {_q(os.path.splitext(str(using))[0] + '.dta')}")
        return lines
    if name == "append":
        return [f"append using {_q(os.path.splitext(str(op.get('other_source', '')))[0] + '.dta')}"]
    if name == "convert_type":
        to = op.get("to")
        cols = " ".join(_lst(op.get("columns")))
        if to in ("numeric", "integer"):
            ign = f" ignore({_q(op.get('ignore'))})" if op.get("ignore") else ""
            return [f"destring {cols}, replace{ign}" + (" force" if op.get("force") else "")]
        if to == "text":
            return [f"tostring {cols}, replace"]
        return None
    if name == "encode":
        return [f"encode {op.get('source')}, generate({op.get('target')})"]
    if name == "decode":
        return [f"decode {op.get('source')}, generate({op.get('target')})"]
    if name == "drop_rows":
        rows = [int(r) + 1 for r in _lst(op.get("rows")) if str(r).lstrip("-").isdigit()]
        return [f"drop in {r}" for r in sorted(rows, reverse=True)]
    if name == "set_value":
        v = op.get("value")
        val = v if isinstance(v, (int, float)) and not isinstance(v, bool) else _q(v)
        return [f"replace {op.get('col_name')} = {val} in {int(op.get('row_idx', 0)) + 1}"]
    if name == "add_rows":
        return [f"set obs `=_N + {int(op.get('n', 1))}'"]
    return None


def _op_r(op: Dict[str, Any]) -> Optional[List[str]]:
    name = op.get("op")
    if name == "compute":
        return [f"data${op.get('col_name')} <- with(data, {_expr_r(op.get('formula', ''))})"]
    if name == "filter_rows":
        return [f"data <- subset(data, {_expr_r(op.get('condition', ''))})"]
    if name == "rename_var":
        return [f"names(data)[names(data) == {_q(op.get('old_name'))}] <- {_q(op.get('new_name'))}"]
    if name == "drop_vars":
        return [f"data <- data[, setdiff(names(data), c({', '.join(_q(c) for c in _lst(op.get('cols')))})), drop = FALSE]"]
    if name == "standardize" and str(op.get("method", "zscore")) == "zscore":
        return [f"data$z_{c} <- as.numeric(scale(data${c}))" for c in _lst(op.get("columns"))]
    if name == "reshape":
        stubs = _lst(op.get("stubnames")) or _lst(op.get("value_vars"))
        i = _lst(op.get("id_vars"))
        j = op.get("time_col") or "time"
        if op.get("direction", "long") == "long":
            pat = "|".join(stubs)
            return [f"data <- tidyr::pivot_longer(data, cols = tidyr::matches(\"^({pat})[0-9]+$\"),",
                    f"                            names_to = c(\".value\", {_q(j)}), names_pattern = \"^({pat})([0-9]+)$\")"]
        return [f"data <- tidyr::pivot_wider(data, id_cols = c({', '.join(_q(x) for x in i)}), names_from = {_q(j)},",
                f"                           values_from = c({', '.join(_q(x) for x in stubs)}), names_sep = \"\")"]
    if name == "merge":
        using = os.path.splitext(str(op.get("other_source", "")))[0]
        by = ", ".join(_q(x) for x in _lst(op.get("on")))
        return [f"using <- read.csv({_q(str(op.get('other_source', '')))}, check.names = FALSE)  # {using}",
                f"data <- merge(data, using, by = c({by}), all.x = TRUE)"]
    if name == "convert_type":
        to = op.get("to")
        conv = {"numeric": "as.numeric", "integer": "as.integer", "text": "as.character"}.get(to)
        if conv:
            return [f"data${c} <- {conv}(data${c})" for c in _lst(op.get("columns"))]
        return None
    if name == "encode":
        return [f"data${op.get('target')} <- as.integer(factor(data${op.get('source')}))"]
    if name == "decode":
        return [f"data${op.get('target')} <- as.character(data${op.get('source')})"]
    if name == "set_value":
        v = op.get("value")
        val = v if isinstance(v, (int, float)) and not isinstance(v, bool) else _q(v)
        return [f"data[{int(op.get('row_idx', 0)) + 1}, {_q(op.get('col_name'))}] <- {val}"]
    return None


# ----------------------------------------------------------------------------- analyses
def _p(params: Dict[str, Any], key: str, default: Any = None) -> Any:
    v = params.get(key, default)
    return default if v in (None, "", []) else v


def _an_stata(t: str, p: Dict[str, Any]) -> Optional[List[str]]:
    if t == "descriptives":
        v = " ".join(_lst(_p(p, "vars")))
        sb = _p(p, "splitBy")
        return [f"bysort {sb}: summarize {v}, detail" if sb else f"summarize {v}, detail"]
    if t == "ttestIS":
        g = _p(p, "group")
        return [f"ttest {d}, by({g})" + (" unequal welch" if p.get("welchs") else "") for d in _lst(_p(p, "vars"))] + \
               ([f"ranksum {d}, by({g})" for d in _lst(_p(p, "vars"))] if p.get("mann") else [])
    if t == "ttestPS":
        return [f"ttest {_p(p, 'var1')} == {_p(p, 'var2')}"] + ([f"signrank {_p(p, 'var1')} = {_p(p, 'var2')}"] if p.get("wilcoxon") else [])
    if t == "ttestOneS":
        mu = _p(p, "testValue", 0)
        return [f"ttest {v} == {mu}" for v in _lst(_p(p, "vars"))]
    if t == "anovaOneW":
        g = _p(p, "group")
        out = []
        for d in _lst(_p(p, "deps")):
            out.append(f"oneway {d} {g}, tabulate")
            ph = _p(p, "phMethod", "tukey")
            if ph in ("tukey", "scheffe", "bonferroni"):
                out.append(f"pwmean {d}, over({g}) mcompare({ph}) effects")
            elif ph == "gamesHowell":
                out.append("* Games-Howell: Stata 기본 명령 없음 (pwmean ..., mcompare(tukey)로 대신하거나 사용자 명령 pwmc 사용)")
        return out
    if t == "anovaFactorial":
        f = "##".join(f"i.{x}" for x in _lst(_p(p, "factors")))
        return [f"anova {_p(p, 'dep')} {f}"]
    if t == "ancova":
        f = " ".join(f"i.{x}" for x in _lst(_p(p, "factors")))
        c = " ".join(f"c.{x}" for x in _lst(_p(p, "covs")))
        return [f"anova {_p(p, 'dep')} {f} {c}"]
    if t == "kruskal":
        return [f"kwallis {d}, by({_p(p, 'group')})" for d in _lst(_p(p, "deps"))]
    if t == "corrMatrix":
        v = " ".join(_lst(_p(p, "vars")))
        out = [f"pwcorr {v}, sig obs"] if p.get("pearson", True) else []
        if p.get("spearman"):
            out.append(f"spearman {v}, stats(rho p)")
        if p.get("kendall"):
            out.append(f"ktau {v}, stats(taub p)")
        return out
    if t == "partCorr":
        v = _lst(_p(p, "vars"))
        return [f"pcorr {' '.join(v)} {' '.join(_lst(_p(p, 'controls')))}"]
    if t == "linReg":
        x = " ".join(_lst(_p(p, "covs")) + [f"i.{f}" for f in _lst(_p(p, "factors"))])
        out = [f"regress {_p(p, 'dep')} {x}" + (", beta" if p.get("stdEst") else "")]
        if p.get("collin"):
            out.append("estat vif")
        return out
    if t == "logRegBin":
        x = " ".join(_lst(_p(p, "covs")) + [f"i.{f}" for f in _lst(_p(p, "factors"))])
        out = [f"logit {_p(p, 'dep')} {x}" + (", or" if p.get("oddsRatio") else "")]
        if p.get("classTable"):
            out.append("estat classification")
        if p.get("rocPlot"):
            out.append("lroc")
        return out
    if t == "contTables":
        opts = ["chi2"] + (["exact"] if p.get("fisher") else []) + (["lrchi2"] if p.get("likeRat") else []) + \
               (["V"] if p.get("phiCra") else []) + (["expected"] if p.get("exp") else []) + (["row"] if p.get("pcRow") else []) + \
               (["column"] if p.get("pcCol") else [])
        return [f"tabulate {_p(p, 'rows')} {_p(p, 'cols')}, {' '.join(opts)}"]
    if t == "reliability":
        rev = _lst(_p(p, "revItems"))
        return [f"alpha {' '.join(_lst(_p(p, 'vars')))}, item" + (f" reverse({' '.join(rev)})" if rev else "")]
    if t == "pca":
        out = [f"pca {' '.join(_lst(_p(p, 'vars')))}"]
        rot = _p(p, "rotation", "varimax")
        if rot in ("varimax", "promax", "oblimin"):
            out.append(f"rotate, {rot}" if rot != "oblimin" else "rotate, oblimin oblique")
        return out
    if t == "efa":
        ext = {"ml": "ml", "pa": "pf", "minres": "ipf"}.get(_p(p, "extraction", "minres"), "ipf")
        out = [f"factor {' '.join(_lst(_p(p, 'vars')))}, {ext}"]
        rot = _p(p, "rotation", "oblimin")
        if rot in ("varimax", "promax", "oblimin"):
            out.append(f"rotate, {rot}" if rot != "oblimin" else "rotate, oblimin oblique")
        return out
    if t == "sem":
        model = str(_p(p, "model", ""))
        parts = []
        for line in re.split(r"[\n;]", model):
            line = re.sub(r"#.*$", "", line).strip()
            m = re.match(r"^(\S+)\s*=~\s*(.+)$", line)
            if m:
                inds = [re.sub(r"^[^*]*\*", "", x).strip() for x in m.group(2).split("+")]
                parts.append(f"({m.group(1)} -> {' '.join(inds)})")
                continue
            m = re.match(r"^(\S+)\s*~(?![~1])\s*(.+)$", line)
            if m:
                preds = [re.sub(r"^[^*]*\*", "", x).strip() for x in m.group(2).split("+")]
                parts.append(f"({m.group(1)} <- {' '.join(preds)})")
        if not parts:
            return None
        lat = sorted({re.match(r"\((\S+) ->", x).group(1) for x in parts if "->" in x})
        note = [f"* Stata sem은 잠재변수 이름이 대문자로 시작해야 함: {', '.join(lat)}"] if lat else []
        opts = ["method(mlmv)" if _p(p, "missing", "fiml") == "fiml" else "method(ml)"] + (["vce(robust)"] if _p(p, "estimator") == "mlr" else []) + ["standardized"]
        return note + [f"sem {' '.join(parts)}, {' '.join(opts)}", "estat gof, stats(all)"]
    if t == "mediation":
        x, m, y = _p(p, "pred"), _p(p, "med"), _p(p, "dep")
        return [f"sem ({m} <- {x}) ({y} <- {m} {x})", "estat teffects"]
    if t == "moderation":
        return [f"regress {_p(p, 'dep')} c.{_p(p, 'pred')}##c.{_p(p, 'mod')}",
                f"margins, dydx({_p(p, 'pred')}) at({_p(p, 'mod')}=(-1 0 1))  // 조절변수 값은 표준화 기준 예시"]
    if t == "panel":
        e, tm, y = _p(p, "entity"), _p(p, "time"), _p(p, "dep")
        x = " ".join(_lst(_p(p, "covs")))
        mt = _p(p, "modelType", "fixed")
        vce = {"robust": " vce(robust)", "clustered": f" vce(cluster {e})"}.get(_p(p, "covType"), "")
        out = [f"xtset {e} {tm}"]
        if mt == "fixed":
            out.append(f"xtreg {y} {x}, fe{vce}")
        elif mt == "twoway":
            out.append(f"xtreg {y} {x} i.{tm}, fe{vce}")
        elif mt == "random":
            out.append(f"xtreg {y} {x}, re{vce}")
        else:
            out.append(f"regress {y} {x}{vce.replace(' vce', ', vce') if vce else ''}")
        if p.get("hausman"):
            out += [f"quietly xtreg {y} {x}, fe", "estimates store fe", f"quietly xtreg {y} {x}, re", "estimates store re", "hausman fe re"]
        return out
    if t == "mixedModel":
        x = " ".join(_lst(_p(p, "covs")) + [f"i.{f}" for f in _lst(_p(p, "factors"))])
        return [f"mixed {_p(p, 'dep')} {x} || {_p(p, 'cluster')}:" + (", reml" if p.get("reml", True) else ", mle"), "estat icc"]
    if t == "survival":
        out = [f"stset {_p(p, 'time')}, failure({_p(p, 'event')})"]
        g = _p(p, "group")
        if g:
            out += [f"sts graph, by({g})", f"sts test {g}"]
        cov = _lst(_p(p, "covs"))
        if cov or g:
            out.append("stcox " + " ".join(([f"i.{g}"] if g else []) + cov))
        return out
    if t in ("spatial", "spatialPanel", "spatial_moran", "spatialWeights"):
        return None
    return None


def _an_r(t: str, p: Dict[str, Any], syntax_r: Optional[str]) -> Optional[List[str]]:
    vr = lambda key: "c(" + ", ".join(_q(x) for x in _lst(_p(p, key))) + ")"
    if t == "descriptives":
        sb = _p(p, "splitBy")
        return [f"jmv::descriptives(data, vars = {vr('vars')}" + (f", splitBy = {_q(sb)}" if sb else "") + ")"]
    if t == "ttestIS":
        return [f"jmv::ttestIS(data, vars = {vr('vars')}, group = {_q(_p(p, 'group'))}, welchs = {str(bool(p.get('welchs'))).upper()}, "
                f"mann = {str(bool(p.get('mann'))).upper()}, effectSize = TRUE)"]
    if t == "ttestPS":
        return [f"jmv::ttestPS(data, pairs = list(list(i1 = {_q(_p(p, 'var1'))}, i2 = {_q(_p(p, 'var2'))})), effectSize = TRUE)"]
    if t == "ttestOneS":
        return [f"jmv::ttestOneS(data, vars = {vr('vars')}, testValue = {_p(p, 'testValue', 0)})"]
    if t == "anovaOneW":
        ph = {"tukey": "tukey", "gamesHowell": "gamesHowell"}.get(_p(p, "phMethod", "tukey"), "tukey")
        return [f"jmv::anovaOneW(data, deps = {vr('deps')}, group = {_q(_p(p, 'group'))}, welchs = TRUE, fishers = TRUE, "
                f"phMethod = {_q(ph)}, desc = TRUE)"]
    if t == "anovaFactorial":
        return [f"jmv::ANOVA(data, dep = {_q(_p(p, 'dep'))}, factors = {vr('factors')}, effectSize = c(\"eta\", \"partEta\"), ss = \"{_p(p, 'ssType', 3)}\")"]
    if t == "ancova":
        return [f"jmv::ancova(data, dep = {_q(_p(p, 'dep'))}, factors = {vr('factors')}, covs = {vr('covs')})"]
    if t == "kruskal":
        return [f"jmv::anovaNP(data, deps = {vr('deps')}, group = {_q(_p(p, 'group'))}, pairs = TRUE)"]
    if t == "corrMatrix":
        return [f"jmv::corrMatrix(data, vars = {vr('vars')}, pearson = {str(bool(p.get('pearson', True))).upper()}, "
                f"spearman = {str(bool(p.get('spearman'))).upper()}, kendall = {str(bool(p.get('kendall'))).upper()}, flag = TRUE)"]
    if t == "partCorr":
        return [f"jmv::corrPart(data, vars = {vr('vars')}, controls = {vr('controls')})"]
    if t == "linReg":
        cov, fac = _lst(_p(p, "covs")), _lst(_p(p, "factors"))
        return [f"jmv::linReg(data, dep = {_q(_p(p, 'dep'))}, covs = {vr('covs')}, factors = {vr('factors')}, "
                f"blocks = list(c({', '.join(_q(x) for x in cov + fac)})), r2Adj = TRUE, stdEst = TRUE, collin = TRUE)"]
    if t == "logRegBin":
        cov, fac = _lst(_p(p, "covs")), _lst(_p(p, "factors"))
        return [f"jmv::logRegBin(data, dep = {_q(_p(p, 'dep'))}, covs = {vr('covs')}, factors = {vr('factors')}, "
                f"blocks = list(c({', '.join(_q(x) for x in cov + fac)})), OR = TRUE, class = TRUE, auc = TRUE)"]
    if t == "contTables":
        return [f"jmv::contTables(data, rows = {_q(_p(p, 'rows'))}, cols = {_q(_p(p, 'cols'))}, phiCra = TRUE, fisher = {str(bool(p.get('fisher'))).upper()})"]
    if t == "reliability":
        return [f"jmv::reliability(data, vars = {vr('vars')}, omegaScale = TRUE, revItems = {vr('revItems')})"]
    if t == "pca":
        return [f"jmv::pca(data, vars = {vr('vars')}, rotation = {_q(_p(p, 'rotation', 'varimax'))}, kmo = TRUE, bartlett = TRUE)"]
    if t == "efa":
        return [f"jmv::efa(data, vars = {vr('vars')}, extraction = {_q(_p(p, 'extraction', 'minres'))}, rotation = {_q(_p(p, 'rotation', 'oblimin'))})"]
    if t == "sem":
        model = str(_p(p, "model", "")).replace('"', "'")
        est = "MLR" if _p(p, "estimator") == "mlr" else "ML"
        return [f"model <- \"\n{model}\n\"",
                f"fit <- lavaan::sem(model, data = data, estimator = {_q(est)}, missing = {_q('fiml' if _p(p, 'missing', 'fiml') == 'fiml' else 'listwise')}, mimic = \"Mplus\")",
                "summary(fit, fit.measures = TRUE, standardized = TRUE)"]
    if t == "mediation":
        return [f"jamm::jammGLM(data, dep = {_q(_p(p, 'dep'))}, mediators = {_q(_p(p, 'med'))}, covs = {_q(_p(p, 'pred'))})  # 또는 medmod::med(...)"]
    if t == "moderation":
        return [f"medmod::mod(data, dep = {_q(_p(p, 'dep'))}, mod = {_q(_p(p, 'mod'))}, pred = {_q(_p(p, 'pred'))}, simpleSlopeEst = TRUE)"]
    if t == "panel":
        e, tm, y = _p(p, "entity"), _p(p, "time"), _p(p, "dep")
        f = f"{y} ~ " + " + ".join(_lst(_p(p, "covs")))
        model = {"fixed": "within", "twoway": "within", "random": "random", "pooled": "pooling"}.get(_p(p, "modelType", "fixed"), "within")
        eff = ", effect = \"twoways\"" if _p(p, "modelType") == "twoway" else ""
        return [f"pdata <- plm::pdata.frame(data, index = c({_q(e)}, {_q(tm)}))",
                f"fit <- plm::plm({f}, data = pdata, model = {_q(model)}{eff})", "summary(fit)"]
    if t == "mixedModel":
        f = f"{_p(p, 'dep')} ~ " + (" + ".join(_lst(_p(p, "covs")) + _lst(_p(p, "factors"))) or "1") + f" + (1 | {_p(p, 'cluster')})"
        return [f"fit <- lme4::lmer({f}, data = data, REML = {str(bool(p.get('reml', True))).upper()})", "summary(fit)"]
    if t == "survival":
        g = _p(p, "group")
        rhs = g if g else "1"
        out = [f"km <- survival::survfit(survival::Surv({_p(p, 'time')}, {_p(p, 'event')}) ~ {rhs}, data = data)", "summary(km)"]
        cov = _lst(_p(p, "covs")) + ([g] if g else [])
        if cov:
            out.append(f"summary(survival::coxph(survival::Surv({_p(p, 'time')}, {_p(p, 'event')}) ~ {' + '.join(cov)}, data = data))")
        return out
    if syntax_r and "::" in syntax_r and "data=data" in syntax_r.replace(" ", ""):
        return [syntax_r]
    return None


# ----------------------------------------------------------------------------- whole record
def convert(entries: List[Dict[str, Any]], fmt: str, started: str = "") -> str:
    fmt = fmt.lower()
    if fmt not in ("do", "r"):
        raise ValueError("fmt must be 'do' or 'r'")
    stata = fmt == "do"
    c = "*" if stata else "#"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = [f"{c} OpenStat Studio 명령 기록 → {'Stata do-file' if stata else 'R script'} 변환 ({now})",
           f"{c} 기록 시작: {started} · 명령 {len(entries)}건",
           f"{c} 대응 명령이 없는 단계는 OpenStat 명령을 주석으로 남김. 추정 방법·기본 설정 차이로 결과가 다를 수 있음",
           f"{c} 자료 경로는 OpenStat 저장소 폴더 기준 상대 경로임. 필요하면 작업 폴더를 저장소 폴더로 지정",
           ""]
    if stata:
        out += ["version 17", "set more off", ""]
    else:
        out += ["# 필요한 패키지: jmv, lavaan, plm, lme4, survival, readxl, haven, tidyr (사용하는 단계만)", ""]
    load = (_load_stata if stata else _load_r)
    conv_op = _op_stata if stata else _op_r
    source_lines: List[str] = []          # how the current data was loaded
    applied: List[Dict[str, Any]] = []    # data steps applied since then (for undo / filter removal)

    def replay(ops: List[Dict[str, Any]], why: str) -> List[str]:
        res = [f"{c} {why}: 자료를 다시 불러와 남은 처리 단계를 다시 적용"] + source_lines
        for op in ops:
            res += conv_op(op) or ([f"{c} (대응 명령 없음 — OpenStat 명령)"] + _comment(c, op.get("_code", "")))
        return res

    for i, e in enumerate(entries, 1):
        kind = e.get("kind")
        out.append(f"{c} [{i}] {e.get('time', '')} {e.get('title', '')}")
        lines: Optional[List[str]] = None
        if kind == "load":
            lines = load(str(e.get("path", "")))
            source_lines, applied = list(lines), []
        elif kind == "data":
            op = dict(e.get("op") or {}, _code=e.get("code", ""))
            applied.append(op)
            lines = conv_op(op)
        elif kind == "data_special":
            code = e.get("code", "")
            if "undo_data_op" in code and source_lines:
                applied = applied[:-1]
                lines = replay(applied, "실행 취소")
            elif "clear_filters" in code and source_lines:
                applied = [op for op in applied if op.get("op") != "filter_rows"]
                lines = replay(applied, "필터 해제")
        elif kind == "code" and stata and e.get("source") and e.get("source").strip() != e.get("code", "").strip():
            # typed as Stata commands in the program: written back as they were entered
            lines = [ln for ln in e["source"].splitlines()]
        elif kind == "analysis":
            if stata:
                lines = _an_stata(str(e.get("analysis_type")), e.get("params") or {})
            else:
                lines = _an_r(str(e.get("analysis_type")), e.get("params") or {}, e.get("syntax_r"))
        elif kind == "export":
            lines = [f"{c} 결과 내보내기(.{e.get('fmt')}): Stata는 putdocx·putexcel·esttab, R은 rmarkdown·officer 등으로 작성"]
        elif kind == "note":
            lines = [f"{c} {e.get('code', '').lstrip('# ').strip()}"]
        if lines is None:
            lines = [f"{c} (대응 명령 없음 — OpenStat 명령)"] + _comment(c, e.get("code", ""))
        out += lines + [""]
    return "\n".join(out) + "\n"
