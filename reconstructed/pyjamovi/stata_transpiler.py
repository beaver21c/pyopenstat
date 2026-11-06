"""
OpenStat / pyjamovi Stata Syntax Transpiler
Translates classic Stata command lines (.do syntax) into equivalent OpenStat / pyjamovi Python calls.
Allows users to type Stata commands in the Command Bar or Do-file Editor and execute them directly!
"""

import re
from typing import Optional, Tuple


def _stata_recode_rules(rule_text: str) -> Tuple[str, Optional[str]]:
    """
    Converts Stata recode rules into OpenStat `recode_variable` rules.

    `(1/3=1) (4 5=2) (.=9) (else=0)` -> (`"1..3=1; 4=2; 5=2; missing=9"`, `"0"`)
    Supports `#/#` ranges (with min/max), several values per rule, `missing`/`.`,
    `nonmissing`, and `else`/`*`. Value labels (`(1/3=1 "Low")`) are dropped.
    """
    bodies = re.findall(r"\(([^()]*)\)", rule_text)
    if not bodies:
        raise ValueError(f"No recode rules found in: {rule_text}")
    clauses = []
    else_value: Optional[str] = None
    for body in bodies:
        body = re.sub(r'"[^"]*"', "", body).strip()
        if "=" not in body:
            raise ValueError(f"Invalid Stata recode rule: ({body})")
        lhs, rhs = (x.strip() for x in body.split("=", 1))
        if not lhs or not rhs or len(rhs.split()) != 1:
            raise ValueError(f"Invalid Stata recode rule: ({body})")
        for tok in lhs.split():
            low = tok.lower()
            if low in ("else", "*"):
                else_value = rhs
            elif low in (".", "missing") or re.fullmatch(r"\.[a-z]", low):
                clauses.append(f"missing={rhs}")
            elif low == "nonmissing":
                clauses.append(f"min..max={rhs}")
            elif "/" in tok:
                lo, hi = tok.split("/", 1)
                clauses.append(f"{lo}..{hi}={rhs}")
            else:
                clauses.append(f"{tok}={rhs}")
    return "; ".join(clauses), else_value


# ---------------------------------------------------------------------------
# Helpers for option parsing (Stata allows abbreviations and nested parentheses)
# ---------------------------------------------------------------------------
_OPT_ABBREV = {
    # full name: minimum abbreviation
    "generate": "gen", "nogenerate": "nogen", "keepusing": "keepus", "keep": "keep", "update": "update",
    "replace": "replace", "assert": "assert", "unmatched": "unm", "force": "force", "ignore": "ignore",
    "percent": "percent", "format": "format", "label": "label", "string": "string", "method": "method",
    "vce": "vce", "group": "group", "ginvariant": "ginv", "standardized": "stand", "latent": "latent",
    "nocapslatent": "nocaps", "cov": "cov", "covariance": "cov", "var": "var", "variance": "var", "means": "means",
    "lclass": "lclass", "nolabel": "nol", "noreport": "norep", "sorted": "sorted", "_merge": "_merge", "by": "by",
    "reps": "reps",
}


def _canon_opt(name: str) -> str:
    low = name.lower()
    for full, ab in _OPT_ABBREV.items():
        if low == full or (len(low) >= len(ab) and full.startswith(low)):
            return full
    return low


def _parse_options(opt_str: str) -> dict:
    """'keep(match master) nogen update' -> {'keep': 'match master', 'nogenerate': '', 'update': ''}."""
    out: dict = {}
    i, n = 0, len(opt_str or "")
    txt = opt_str or ""
    while i < n:
        while i < n and txt[i] in " \t,":
            i += 1
        j = i
        while j < n and txt[j] not in " \t,(":
            j += 1
        name = txt[i:j]
        if not name:
            break
        val = ""
        if j < n and txt[j] == "(":
            depth, k = 0, j
            while k < n:
                if txt[k] == "(":
                    depth += 1
                elif txt[k] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            val = txt[j + 1:k]
            j = k + 1
        key = _canon_opt(name)
        out[key] = (out[key] + " " + val).strip() if key in out and key in ("cov", "var", "means") else val
        i = j
    return out


def _split_top(s: str):
    """Split 'sem (a -> b) (c <- d), opts' into (['a -> b', 'c <- d'], 'opts')."""
    groups, depth, start, i = [], 0, None, 0
    rest_at = len(s)
    while i < len(s):
        ch = s[i]
        if ch == "(":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start is not None:
                groups.append(s[start:i])
                start = None
        elif ch == "," and depth == 0:
            rest_at = i
            break
        i += 1
    return groups, s[rest_at + 1:].strip() if rest_at < len(s) else ""


def _names(txt: str):
    return [t for t in re.split(r"\s+", (txt or "").strip()) if t]


def _using_file(name: str) -> str:
    name = name.strip().strip('"').strip("'")
    if not re.search(r"\.[A-Za-z0-9]{2,8}$", name):
        name += ".dta"  # Stata default extension
    return name


def _stata_fmt_to_py(fmt: str):
    m = re.fullmatch(r"%-?0?\d*\.(\d+)f", (fmt or "").strip())
    return "{:.%sf}" % m.group(1) if m else None


def _translate_sem(body: str, cmd: str) -> str:
    groups, opt_str = _split_top(body)
    opts = _parse_options(opt_str)
    if not groups:
        raise ValueError("sem: 경로 지정 (A -> b) 없음")
    explicit_latent = set(_names(opts.get("latent", "")))
    caps = "nocapslatent" not in opts

    def is_latent(v: str) -> bool:
        return v in explicit_latent or (caps and bool(re.match(r"[A-Z]", v)))

    edges = []   # (source, target, constraint)
    for g in groups:
        g = g.split(",")[0]
        parts = re.split(r"(->|<-)", g)
        if len(parts) < 3:
            raise ValueError(f"sem: 경로 해석 불가 ({g.strip()})")
        segs = [_names(p) for p in parts[0::2]]
        arrows = parts[1::2]
        for k, arrow in enumerate(arrows):
            left, right = segs[k], segs[k + 1]
            srcs, tgts = (left, right) if arrow == "->" else (right, left)
            for t in tgts:
                t_name, t_c = (t.split("@", 1) + [None])[:2]
                for sname in srcs:
                    s_name, s_c = (sname.split("@", 1) + [None])[:2]
                    if s_name == "_cons":
                        continue
                    c = (t_c or s_c) if arrow == "->" else (s_c or t_c)
                    edges.append((s_name, t_name, c))
    # an observed outcome is a measurement indicator only when all of its predictors are latent
    obs_pred = {t for s_, t, _ in edges if not is_latent(s_) and not is_latent(t)}
    meas: dict = {}
    reg: dict = {}
    order_meas: list = []
    for s_name, t_name, c in edges:
        pre = f"{c}*" if c else ""
        if is_latent(s_name) and not is_latent(t_name) and t_name not in obs_pred:
            if s_name not in meas:
                meas[s_name] = []
                order_meas.append(s_name)
            meas[s_name].append(pre + t_name)
        else:
            reg.setdefault(t_name, []).append(pre + s_name)
    fixed_var = set()
    cov_lines = []
    for spec in _names(opts.get("cov", "")) + _names(opts.get("var", "")):
        spec = spec.replace("e.", "")
        val = None
        if "@" in spec:
            spec, val = spec.split("@", 1)
        if "*" in spec:
            a, b = spec.split("*", 1)
        else:
            a = b = spec
        if a == b and val is not None and is_latent(a):
            fixed_var.add(a)
        cov_lines.append(f"{a} ~~ {(val + '*') if val is not None else ''}{b}")
    lines = []
    for lv in order_meas:
        terms = meas[lv]
        if lv in fixed_var and terms and "*" not in terms[0]:
            terms = ["NA*" + terms[0]] + terms[1:]
        lines.append(f"{lv} =~ " + " + ".join(terms))
    for dv, terms in reg.items():
        lines.append(f"{dv} ~ " + " + ".join(terms))
    lines.extend(cov_lines)
    kw = []
    meth = opts.get("method", "").lower()
    if meth:
        if meth.startswith("mlmv"):
            kw.append("missing='fiml'")
        elif meth.startswith("ml"):
            kw.append("missing='listwise'")
        else:
            raise ValueError(f"sem method({meth}) 미지원 (ml, mlmv만 변환)")
    vce = opts.get("vce", "").lower()
    if vce:
        if vce.startswith(("robust", "sbentler", "r")) and not vce.startswith("boot"):
            kw.append("estimator='mlr'")
        elif vce.startswith("boot"):
            rm = re.search(r"reps\((\d+)\)", vce)
            kw.append(f"bootstrap={int(rm.group(1)) if rm else 1000}")
    if "group" in opts:
        kw.append(f"group={opts['group'].strip()!r}")
        gi = set(_names(opts.get("ginvariant", "mcoef mcons").lower()))
        inv = ("configural" if "none" in gi else "strict" if ("all" in gi or {"mcoef", "mcons", "merrvar"} <= gi)
               else "scalar" if {"mcoef", "mcons"} <= gi else "metric" if "mcoef" in gi else "configural")
        kw.append(f"invariance={inv!r}")
    if "standardized" in opts:
        kw.append("standardized='stdyx'")
    model = "\n".join(lines)
    return f"sess.sem(model={model!r}" + (", " + ", ".join(kw) if kw else "") + ")"


def _translate_gsem(body: str) -> str:
    groups, opt_str = _split_top(body)
    opts = _parse_options(opt_str)
    lc = _names(opts.get("lclass", ""))
    if len(lc) < 2:
        raise ValueError("gsem: 잠재계층 lclass(C k) 지정 시에만 변환 (그 밖의 gsem 모형 미지원)")
    k = int(lc[1])
    default_fam = "logit" if any(f in opts for f in ("logit", "ologit", "mlogit", "probit", "oprobit")) else "gaussian"
    cat, cont = [], []
    for g in groups:
        fam = default_fam
        if "," in g:
            g, gopt = g.split(",", 1)
            go = gopt.lower()
            if any(w in go for w in ("logit", "probit", "mlogit", "ologit", "bernoulli")):
                fam = "logit"
            elif any(w in go for w in ("gaussian", "regress")):
                fam = "gaussian"
        lhs = re.split(r"<-|->", g)[0] if "<-" in g else (re.split(r"->", g)[1] if "->" in g else g)
        vars_ = [v for v in _names(lhs) if v not in ("_cons",) and not re.match(r"^[A-Z]$", v)]
        (cat if fam == "logit" else cont).extend(vars_)
    parts = []
    if cat:
        parts.append(f"categorical={cat!r}")
    if cont:
        parts.append(f"continuous={cont!r}")
    return f"sess.lca({', '.join(parts)}, classes={k})"


def _transpile_extended(s: str):
    """Data management, SEM and latent class commands. Returns None when no rule applies."""
    low = s.lower()
    # harmless session commands
    if re.match(r"^(set\s|clear\b|version\b|log\s|capture\s+log|cd\s|cls\b|macro\s+drop|graph\s+close|more\b|discard\b)", low):
        return f"# [Stata] {s}  → 건너뜀(세션 설정 명령)"
    m = re.match(r"^use\s+(.+?)(?:\s*,\s*clear)?\s*$", s, re.IGNORECASE)
    if m and " using " not in f" {m.group(1)} ":
        return f"sess.load({_using_file(m.group(1))!r})"
    m = re.match(r"^save\s+(\S+)(?:\s*,.*)?$", s, re.IGNORECASE)
    if m:
        return f"sess.save_data({_using_file(m.group(1))!r})"

    # merge 1:1 k1 k2 using f [, opts]
    m = re.match(r"^merge\s+(?:(1:1|m:1|1:m|m:m)\s+)?(.+?)\s+using\s+(\"[^\"]+\"|\S+)\s*(?:,\s*(.*))?$", s, re.IGNORECASE)
    if m:
        kind, keys, fname, opt = m.group(1), _names(m.group(2)), _using_file(m.group(3)), _parse_options(m.group(4) or "")
        kw = [f"on={keys!r}"] + ([f"kind={kind.lower()!r}"] if kind else [])
        if "keep" in opt:
            kw.append(f"keep={_names(opt['keep'])!r}")
        if "keepusing" in opt:
            kw.append(f"keepusing={_names(opt['keepusing'])!r}")
        if "nogenerate" in opt:
            kw.append("indicator=False")
        elif opt.get("generate"):
            kw.append(f"indicator={opt['generate'].strip()!r}")
        if "update" in opt:
            kw.append("update=True")
        if "replace" in opt:
            kw.append("replace=True")
        if "assert" in opt:
            kw.append(f"assert_results={_names(opt['assert'])!r}")
        return f"sess.merge({fname!r}, {', '.join(kw)})"
    m = re.match(r"^append\s+using\s+(.+?)\s*(?:,\s*(.*))?$", s, re.IGNORECASE)
    if m:
        files = [_using_file(x) for x in re.findall(r"\"[^\"]+\"|\S+", m.group(1))]
        opt = _parse_options(m.group(2) or "")
        kw = []
        if opt.get("generate"):
            kw.append(f"generate={opt['generate'].strip()!r}")
        if "force" in opt:
            kw.append("force=True")
        src = files[0] if len(files) == 1 else files
        return f"sess.append({src!r}" + (", " + ", ".join(kw) if kw else "") + ")"
    m = re.match(r"^joinby\s+(?:(.+?)\s+)?using\s+(\"[^\"]+\"|\S+)\s*(?:,\s*(.*))?$", s, re.IGNORECASE)
    if m:
        keys = _names(m.group(1) or "")
        opt = _parse_options(m.group(3) or "")
        kw = [f"on={keys!r}" if keys else "on=None"]
        if opt.get("unmatched"):
            kw.append(f"unmatched={opt['unmatched'].strip().lower()!r}")
        if opt.get("_merge"):
            kw.append(f"indicator={opt['_merge'].strip()!r}")
        if "update" in opt:
            kw.append("update=True")
        if "replace" in opt:
            kw.append("replace=True")
        return f"sess.joinby({_using_file(m.group(2))!r}, {', '.join(kw)})"
    m = re.match(r"^reshape\s+(long|wide)\s+(.+?)\s*,\s*(.*)$", s, re.IGNORECASE)
    if m:
        direction, stubs, opt = m.group(1).lower(), _names(m.group(2)), _parse_options(m.group(3))
        if any("@" in x for x in stubs):
            return f"raise ValueError({('reshape의 @ 위치 지정 stub 미지원: ' + s)!r})"
        ids = _names(opt.get("i", ""))
        j = _names(opt.get("j", ""))
        if not ids or not j:
            return f"raise ValueError({('reshape: i()와 j() 지정 필요: ' + s)!r})"
        if direction == "long":
            return f"sess.reshape('long', id_vars={ids!r}, stubnames={stubs!r}, time_col={j[0]!r})"
        return f"sess.reshape('wide', id_vars={ids!r}, value_vars={stubs!r}, time_col={j[0]!r})"
    m = re.match(r"^collapse\s+(.+?)\s*(?:,\s*(.*))?$", s, re.IGNORECASE)
    if m:
        spec, opt = m.group(1), _parse_options(m.group(2) or "")
        stats, stat = {}, "mean"
        for tok in re.findall(r"\([^)]*\)|\S+=\S+|\S+", spec):
            if tok.startswith("("):
                stat = tok.strip("()").strip().lower()
            elif "=" in tok:
                new, old = tok.split("=", 1)
                stats[new] = (old, stat)
            else:
                stats[tok] = stat
        by = _names(opt.get("by", ""))
        return f"sess.collapse({stats!r}" + (f", by={by!r}" if by else "") + ")"
    m = re.match(r"^(?:(?:bysort|bys|by)\s+([^:]+):\s*)?egen\s+(?:\w+\s+)?(\S+)\s*=\s*(\w+)\(([^)]*)\)\s*(?:,\s*(.*))?$", s, re.IGNORECASE)
    if m:
        by = [x for x in _names((m.group(1) or "").replace("(", " ").replace(")", " "))]
        opt = _parse_options(m.group(5) or "")
        by = by or _names(opt.get("by", ""))
        src = _names(m.group(4))
        src_arg = src[0] if len(src) == 1 else src
        return (f"sess.egen({m.group(2)!r}, {m.group(3).lower()!r}, {src_arg!r}"
                + (f", by={by!r}" if by else "") + ")")
    m = re.match(r"^(destring|tostring)\s+(.+?)\s*,\s*(.*)$", s, re.IGNORECASE)
    if m:
        cmd, cols, opt = m.group(1).lower(), _names(m.group(2)), _parse_options(m.group(3))
        kw = [f"{cols[0] if len(cols) == 1 else cols!r}"]
        if opt.get("generate"):
            g = _names(opt["generate"])
            kw.append(f"generate={g[0] if len(g) == 1 else g!r}")
        elif "replace" not in opt:
            return f"raise ValueError({(cmd + ': generate() 또는 replace 지정 필요: ' + s)!r})"
        if cmd == "destring":
            if opt.get("ignore"):
                kw.append(f"ignore={opt['ignore'].strip().strip(chr(34))!r}")
            if "force" in opt:
                kw.append("force=True")
            if "percent" in opt:
                kw.append("percent=True")
        else:
            pf = _stata_fmt_to_py(opt.get("format", ""))
            if pf:
                kw.append(f"text_format={pf!r}")
        return f"sess.{cmd}({', '.join(kw)})"
    m = re.match(r"^(encode|decode)\s+(\S+)\s*,\s*(.*)$", s, re.IGNORECASE)
    if m:
        opt = _parse_options(m.group(3))
        if not opt.get("generate"):
            return f"raise ValueError({(m.group(1) + ': generate() 지정 필요: ' + s)!r})"
        return f"sess.{m.group(1).lower()}({m.group(2)!r}, {opt['generate'].strip()!r})"
    m = re.match(r"^la(?:bel|b)?\s+de(?:fine|f)?\s+(\w+)\s+(.+?)(?:\s*,\s*(?:add|modify|replace)\s*)*$", s, re.IGNORECASE)
    if m:
        pairs = re.findall(r"(-?\d+(?:\.\d+)?)\s+\"([^\"]*)\"", m.group(2))
        if not pairs:
            return f"raise ValueError({('label define 해석 불가: ' + s)!r})"
        lab = {int(float(k)) if float(k).is_integer() else float(k): v for k, v in pairs}
        return f"_lbl_{m.group(1)} = {lab!r}"
    m = re.match(r"^la(?:bel|b)?\s+val(?:ues)?\s+(.+?)\s+(\w+)\s*$", s, re.IGNORECASE)
    if m:
        cols = _names(m.group(1))
        return "; ".join(f"sess.set_value_labels({c!r}, _lbl_{m.group(2)})" for c in cols)
    m = re.match(r"^la(?:bel|b)?\s+var(?:iable)?\s+", s, re.IGNORECASE)
    if m:
        return f"# [Stata] {s}  → 건너뜀(변수 설명 라벨)"
    m = re.match(r"^sem\s+(.+)$", s, re.IGNORECASE)
    if m:
        try:
            return _translate_sem(m.group(1), "sem")
        except ValueError as e:
            return f"raise ValueError({str(e) + ': ' + s!r})"
    m = re.match(r"^gsem\s+(.+)$", s, re.IGNORECASE)
    if m:
        try:
            return _translate_gsem(m.group(1))
        except ValueError as e:
            return f"raise ValueError({str(e) + ': ' + s!r})"
    m = re.match(r"^estat\s+(\w+)", s, re.IGNORECASE)
    if m:
        hint = {"gof": "적합도는 sem 결과에 기본 포함", "eqgof": "R²는 sem 결과에 기본 포함",
                "mindices": "수정지수는 sess.sem(..., modIndices=True)", "teffects": "간접·총효과는 sess.sem(..., indirect=[...])",
                "ginvariant": "측정동일성 비교는 sess.sem(..., group=..., invarianceTest=True)",
                "lcprob": "계층 비율은 sess.lca 결과에 기본 포함", "lcmean": "계층별 응답확률·평균은 sess.lca 결과에 기본 포함",
                "lcgof": "정보기준은 sess.lca 결과에 기본 포함"}.get(m.group(1).lower())
        if hint:
            return f"# [Stata] {s}  → 건너뜀: {hint}"
    return None



# ---------------------------------------------------------------- regression + eststo / esttab
_REG_OPT_OK = {"beta"}
_ESTTAB_STATS = {"n": "N", "r2": "R2", "r2_a": "R2_adj", "ar2": "R2_adj", "r2_p": "R2_McF", "pr2": "R2_McF", "ll": "LL",
                 "aic": "AIC", "bic": "BIC", "f": "F", "chi2": "Chi2", "df_m": "df"}
_ESTTAB_EXT = {"docx": "docx", "rtf": "docx", "tex": "tex", "csv": "csv", "html": "html", "htm": "html", "md": "md",
               "xlsx": "xlsx", "hwpx": "hwpx", "pdf": "pdf", "txt": "md"}


def _translate_regression(s: str):
    """regress / logit / logistic with options checked (unsupported options stop with a message)."""
    m = re.match(r"^(regress|reg|logit|logistic)\s+([^,]+?)\s*(?:,\s*(.*))?$", s, re.IGNORECASE)
    if not m:
        return None
    cmd, varlist, opt_txt = m.group(1).lower(), _names(m.group(2)), m.group(3) or ""
    if len(varlist) < 2:
        return f"raise NotImplementedError({('Stata ' + cmd + ': 종속변수와 독립변수 필요 — ' + s)!r})"
    if any(re.search(r"[.#]", v) for v in varlist):
        return f"raise NotImplementedError({('요인변수 표기(i.·c.·#) 미지원: 더미변수를 만든 뒤 지정 — ' + s)!r})"
    opts = _parse_options(opt_txt)
    dep, covs = varlist[0], varlist[1:]
    if cmd in ("regress", "reg"):
        bad = [k for k in opts if k not in _REG_OPT_OK]
        if bad:
            return f"raise NotImplementedError({('regress 옵션 미지원: ' + ', '.join(bad) + ' — ' + s)!r})"
        extra = ", std_est=True" if "beta" in opts else ""
        return f"sess.regression(dep={dep!r}, covs={covs!r}{extra})"
    bad = [k for k in opts if k not in ("or",)]
    if bad:
        return f"raise NotImplementedError({('logit 옵션 미지원: ' + ', '.join(bad) + ' — ' + s)!r})"
    return f"sess.logistic(dep={dep!r}, covs={covs!r})"


def _translate_estimates(s: str, ctx: dict):
    """eststo / estimates store / esttab -> named results and sess.compare_models(...) (+ social-format export)."""
    stored = ctx.setdefault("stored", [])
    low = s.lower()
    if re.match(r"^(eststo|estimates|est)\s+clear\s*$", low) or re.match(r"^eststo\s*,\s*clear\s*$", low):
        stored.clear()
        return f"# [Stata] {s}  → 저장한 추정 결과 목록 초기화"
    m = re.match(r"^(?:estimates|est)\s+sto(?:re)?\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", s, re.IGNORECASE)
    if m:
        stored.append(m.group(1))
        return f"{m.group(1)} = sess.history[-1]"
    m = re.match(r"^eststo(?:\s+([A-Za-z_][A-Za-z0-9_]*))?\s*(?::\s*(.+))?$", s, re.IGNORECASE)
    if m and (m.group(1) or m.group(2)):
        name = m.group(1) or f"est{len(stored) + 1}"
        inner = (m.group(2) or "").strip()
        if inner:
            expr = transpile_stata_line(inner)
            if expr is None or expr.startswith("raise ") or expr.startswith("#"):
                return expr or f"raise NotImplementedError({('Stata 명령 변환 미지원: ' + inner)!r})"
            stored.append(name)
            return f"{name} = {expr}"
        stored.append(name)
        return f"{name} = sess.history[-1]"
    m = re.match(r"^esttab\b\s*(.*?)\s*(?:using\s+(\"[^\"]+\"|\S+))?\s*(?:,\s*(.*))?$", s, re.IGNORECASE)
    if not m:
        return None
    names = _names(m.group(1))
    if not names:
        names = list(stored)
    if not names:
        return f"raise NotImplementedError({('esttab: 저장된 추정 결과 없음(eststo로 저장 후 지정) — ' + s)!r})"
    opts = _parse_options(m.group(3) or "")
    comments, kw, rep = [], [], {}
    stats = ["N"]
    if "se" in opts:
        rep["se_position"] = "below"
    else:
        rep["se_position"] = "below"
        comments.append("esttab 기본(괄호 안 t 통계량)은 지원하지 않아 괄호 안에 표준오차를 표시")
    if "p" in opts:
        rep["keep_p"] = True
        comments.append("esttab p 옵션: 괄호 대신 p 열로 표시")
    if "nostar" in opts:
        rep["stars"] = "none"
    elif opts.get("star"):
        toks = _names(opts["star"])
        pairs = [[float(toks[i + 1]), toks[i]] for i in range(0, len(toks) - 1, 2)]
        if pairs != [[0.05, "*"], [0.01, "**"], [0.001, "***"]]:
            rep["stars"] = pairs
    fmt_val = opts.get("b") or ""
    dm = re.fullmatch(r"%?-?0?\d*\.(\d+)[fg]c?|(\d+)", fmt_val.strip()) if fmt_val else None
    if dm:
        rep["decimals"] = int(dm.group(1) or dm.group(2))
    for o in ("r2", "ar2", "pr2", "aic", "bic"):
        if o in opts:
            stats.append(_ESTTAB_STATS[o])
    sc = opts.get("scalars") or opts.get("stats") or ""
    if sc:
        stats = []
        for t in _names(sc.replace('"', " ")):
            key = _ESTTAB_STATS.get(t.lower())
            if key is None:
                return f"raise NotImplementedError({('esttab stats/scalars 미지원 항목: ' + t)!r})"
            stats.append(key)
    if "beta" in opts:
        kw.append("estimate='beta'")
    elif "eform" in opts:
        kw.append("estimate='exp'")
    if opts.get("mtitles"):
        kw.append(f"names={re.findall(r'\"([^\"]*)\"', opts['mtitles']) or _names(opts['mtitles'])!r}")
    if opts.get("title"):
        kw.append(f"title={opts['title'].strip().strip(chr(34))!r}")
    ignored = [k for k in opts if k not in ("se", "p", "nostar", "star", "b", "r2", "ar2", "pr2", "aic", "bic", "scalars", "stats",
                                            "beta", "eform", "mtitles", "title", "label", "replace", "nogaps", "compress",
                                            "nonumbers", "nomtitles", "noobslast", "obslast", "append", "booktabs", "plain")]
    if ignored:
        return f"raise NotImplementedError({('esttab 옵션 미지원: ' + ', '.join(ignored))!r})"
    kw.insert(0, f"stats={stats!r}")
    lines = [f"# [Stata] {c}" for c in comments]
    lines.append(f"_esttab = sess.compare_models({', '.join(names)}, {', '.join(kw)})")
    target = m.group(2)
    if target:
        path = target.strip().strip('"')
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext not in _ESTTAB_EXT:
            return f"raise NotImplementedError({('esttab using 파일 형식 미지원: ' + path)!r})"
        if _ESTTAB_EXT[ext] != ext:
            new_path = path[: -len(ext)] + _ESTTAB_EXT[ext]
            lines.insert(0, f"# [Stata] esttab .{ext} 출력은 .{_ESTTAB_EXT[ext]}로 저장")
            path = new_path
        lines.append(f"_esttab.export_results({path!r}, report_style='social', report_options={rep!r})")
    else:
        lines.append("print(_esttab.to_markdown())")
    return "\n".join(lines)

def transpile_stata_line(line: str, ctx: Optional[dict] = None) -> Optional[str]:
    """
    Translates a single Stata command line into an OpenStat Python session expression.
    If the line is already valid Python or comments, returns it unchanged or None.
    """
    s = line.strip()
    if not s or s.startswith("#") or s.startswith("*") or s.startswith("//"):
        return None

    # Already Python?
    if s.startswith("sess.") or s.startswith("ost.") or s.startswith("jmv.") or s.startswith("df[") or s.startswith("import "):
        return s

    est = _translate_estimates(s, ctx if ctx is not None else {})
    if est is not None:
        return est
    reg = _translate_regression(s)
    if reg is not None:
        return reg

    ext = _transpile_extended(s)
    if ext is not None:
        return ext

    # 1. ttest dep, by(group)
    m = re.match(r"^ttest\s+([A-Za-z0-9_]+)\s*,\s*by\s*\(\s*([A-Za-z0-9_]+)\s*\)", s, re.IGNORECASE)
    if m:
        dep, grp = m.group(1), m.group(2)
        return f"sess.ttest(dep={dep!r}, group={grp!r})"

    # ttest dep == val
    m = re.match(r"^ttest\s+([A-Za-z0-9_]+)\s*==\s*([0-9\.\-]+)", s, re.IGNORECASE)
    if m:
        dep, val = m.group(1), float(m.group(2))
        return f"sess.ttest_one_sample(dep={dep!r}, test_value={val})"

    # 2. regress / reg dep indep1 indep2 ...
    m = re.match(r"^(?:regress|reg)\s+([A-Za-z0-9_]+)\s+(.+)$", s, re.IGNORECASE)
    if m:
        dep = m.group(1)
        indeps = [x.strip() for x in m.group(2).split() if x.strip() and not x.startswith(",")]
        return f"sess.regression(dep={dep!r}, covs={indeps!r})"

    # 3. anova dep factor1 factor2 ...
    m = re.match(r"^anova\s+([A-Za-z0-9_]+)\s+(.+)$", s, re.IGNORECASE)
    if m:
        dep = m.group(1)
        factors = [x.strip() for x in m.group(2).split() if x.strip() and not x.startswith(",")]
        return f"sess.anova(dep={dep!r}, factors={factors!r})"

    # 4. summarize / sum var1 var2 ...
    m = re.match(r"^(?:summarize|sum|su)\b(?:\s+(.+))?$", s, re.IGNORECASE)
    if m:
        rest = m.group(1)
        vars_list = [x.strip() for x in rest.split() if x.strip() and not x.startswith(",")] if rest else []
        if vars_list:
            return f"sess.descriptives(vars={vars_list!r})"
        return "sess.descriptives()"

    # 5. tabulate / tab var1 [var2]
    m = re.match(r"^(?:tabulate|tab)\s+([A-Za-z0-9_]+)(?:\s+([A-Za-z0-9_]+))?", s, re.IGNORECASE)
    if m:
        r_var = m.group(1)
        c_var = m.group(2)
        if c_var:
            return f"sess.crosstab(rows={r_var!r}, cols={c_var!r})"
        return f"sess.descriptives(vars=[{r_var!r}], freq=True)"

    # 6. correlate / corr var1 var2 ...
    m = re.match(r"^(?:correlate|corr|pwcorr)\s+(.+)$", s, re.IGNORECASE)
    if m:
        vars_list = [x.strip() for x in m.group(1).split() if x.strip() and not x.startswith(",")]
        return f"sess.corr_matrix(vars={vars_list!r})"

    # 7. generate / gen newvar = formula
    m = re.match(r"^(?:generate|gen)\s+([A-Za-z0-9_]+)\s*=\s*(.+)$", s, re.IGNORECASE)
    if m:
        var = m.group(1)
        form = m.group(2).strip()
        return f"sess.compute({var!r}, {form!r})"

    # replace var = formula
    m = re.match(r"^replace\s+([A-Za-z0-9_]+)\s*=\s*(.+)$", s, re.IGNORECASE)
    if m:
        var = m.group(1)
        form = m.group(2).strip()
        return f"sess.compute({var!r}, {form!r})"

    # 8. keep if condition / drop if condition
    m = re.match(r"^keep\s+if\s+(.+)$", s, re.IGNORECASE)
    if m:
        cond = m.group(1).strip()
        return f"sess.filter_rows({cond!r})"
    m = re.match(r"^drop\s+if\s+(.+)$", s, re.IGNORECASE)
    if m:
        cond = m.group(1).strip()
        return f"sess.filter_rows('not (' + {cond!r} + ')')"

    # 9. drop var1 var2
    m = re.match(r"^drop\s+(.+)$", s, re.IGNORECASE)
    if m and not s.lower().startswith("drop if"):
        cols = [x.strip() for x in m.group(1).split() if x.strip()]
        return f"sess.drop_vars({cols!r})"

    # 10. rename old new
    m = re.match(r"^rename\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)", s, re.IGNORECASE)
    if m:
        return f"sess.rename_var({m.group(1)!r}, {m.group(2)!r})"

    # 10.5 recode var (rule) [(rule) ...] [, gen(newvar)]
    m = re.match(r"^recode\s+([A-Za-z0-9_]+)\s+(\([^,]*\))\s*(?:,\s*(.*))?$", s, re.IGNORECASE)
    if m:
        src = m.group(1)
        opts = m.group(3) or ""
        m_gen = re.search(r"\bgen(?:erate)?\s*\(\s*([A-Za-z0-9_]+)\s*\)", opts, re.IGNORECASE)
        tgt = m_gen.group(1) if m_gen else src
        try:
            rules, else_value = _stata_recode_rules(m.group(2))
        except ValueError as e:
            return f"raise ValueError({str(e)!r})"
        call = f"sess.recode(source_col={src!r}, rules={rules!r}, target_col={tgt!r}"
        if else_value is not None:
            call += f", else_value={else_value!r}"
        return call + ")"
    if re.match(r"^recode\s", s, re.IGNORECASE):
        return f"raise ValueError({('Unsupported Stata recode syntax (one variable per line): ' + s)!r})"

    # 11. xtreg dep indep1 ..., fe / re
    m = re.match(r"^xtreg\s+([A-Za-z0-9_]+)\s+([^,]+)(?:,\s*(fe|re))?", s, re.IGNORECASE)
    if m:
        dep = m.group(1)
        indeps = [x.strip() for x in m.group(2).split() if x.strip()]
        model_type = m.group(3).lower() if m.group(3) else "fe"
        return f"sess.panel(dep={dep!r}, indeps={indeps!r}, model={model_type!r})"

    # 12a. Sp setup commands: weights and effects are handled by the spatial analysis options
    m = re.match(r"^(spmatrix|spset|spshape2dta|spcompare|spdistance|estat\s+impact|estat\s+moran)\b(.*)$", s, re.IGNORECASE)
    if m:
        return (f"# [Stata] {s}  → 건너뜀: 공간가중행렬은 분석 옵션(wType·wNorm 등)과 연결된 지도 경계로 지정, "
                "직접·간접·총효과와 Moran 검정은 분석 결과에 기본 포함")

    # 12b. xtset panelvar [timevar]
    m = re.match(r"^xtset\s+(\S+)(?:\s+(\S+))?", s, re.IGNORECASE)
    if m:
        return f"sess.xtset({m.group(1)!r}, {m.group(2)!r})" if m.group(2) else f"sess.xtset({m.group(1)!r})"

    # 13. spregress / spxtregress / xsmle: dvarlag(W) errorlag(W) ivarlag(W: x1 x2) | model(sar|sdm|sem|sac) fe re type(both, leeyu)
    m = re.match(r"^(spregress|spxtregress|xsmle)\s+(\S+)\s+([^,]+)(?:,\s*(.*))?", s, re.IGNORECASE)
    if m:
        cmd = m.group(1).lower()
        dep = m.group(2)
        indeps = [x.strip() for x in m.group(3).split() if x.strip()]
        opt_str = (m.group(4) or "").lower()
        lag = "dvarlag(" in opt_str
        err = "errorlag(" in opt_str
        iv = re.search(r"ivarlag\(\s*\w+\s*:\s*([^)]*)\)", opt_str)
        durbin = [x for x in (iv.group(1).split() if iv else []) if x]
        mm = re.search(r"model\((\w+)\)", opt_str)
        if mm:
            m_type = {"gspre": "sem"}.get(mm.group(1), mm.group(1))
            dm = re.search(r"durbin\(([^)]*)\)", opt_str)
            durbin = dm.group(1).split() if dm else durbin
        else:
            m_type = ("sac" if lag and err else "sdm" if lag and durbin else "sdem" if err and durbin else
                      "slx" if durbin else "sem" if err else "sar")
        extra = f", durbinVars={durbin!r}" if durbin else ""
        note = "  # W: 분석 옵션의 가중행렬(기본: 면 경계 퀸 인접·행 표준화) 사용"
        if "gs2sls" in opt_str:
            note += ", gs2sls 대신 최우추정(ML)"
        if cmd == "spregress":
            return f"sess.spatial_regression(dep={dep!r}, covs={indeps!r}, model_type={m_type!r}{extra}){note}"
        eff = "re" if re.search(r"(^|[\s,])re($|[\s,])", opt_str) else "fe"
        tm = re.search(r"type\(([^)]*)\)", opt_str)
        type_parts = [x.strip() for x in tm.group(1).split(",")] if tm else []
        fe_type = {"ind": "ind", "time": "time", "both": "both"}.get(type_parts[0] if type_parts else "ind", "ind")
        # xsmle applies the Lee-Yu correction only with type(..., leeyu); spxtregress, fe always uses it
        leeyu = ("leeyu" in type_parts) if cmd == "xsmle" else True
        return (f"sess.spatial_panel(dep={dep!r}, covs={indeps!r}, model={m_type!r}, effectsType={eff!r}, "
                f"feType={fe_type!r}, leeyu={leeyu}{extra}){note}")

    # 14. moran var
    m = re.match(r"^moran\s+(\S+)", s, re.IGNORECASE)
    if m:
        return f"sess.spatial_regression(dep={m.group(1)!r}, model_type='moran_only')"

    # No rule matched: keep valid Python, stop with an explicit error for unsupported Stata commands
    import ast
    try:
        ast.parse(s)
        return s
    except SyntaxError:
        return f"raise NotImplementedError({('Stata 명령 변환 미지원: ' + s)!r})"


def _bracket_delta(line: str) -> int:
    """Net count of opened ( [ { on a Python line, ignoring string contents and # comments (approximate)."""
    depth, quote, i = 0, None, 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if line.startswith(quote, i):
                i += len(quote)
                quote = None
                continue
        elif ch == "#":
            break
        elif line.startswith('"""', i) or line.startswith("'''", i):
            quote = line[i:i + 3]
            i += 3
            continue
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        i += 1
    return depth


def transpile_stata_script(script_text: str) -> str:
    """Translates Stata command lines in a script into Python while keeping Python code as written.

    A script that is already valid Python is returned unchanged. Otherwise each top-level line (outside
    brackets, triple-quoted text and indented blocks) is converted when it is a Stata command; Python
    lines keep their indentation, so multi-line calls, function definitions and loops still run."""
    ctx: dict = {"stored": []}                                         # eststo names across lines
    text = re.sub(r"/\*.*?\*/", " ", script_text, flags=re.S)          # /* block comments */
    text = re.sub(r"[ \t]*///[^\n]*\n[ \t]*", " ", text)                 # /// line continuation
    try:
        import ast
        ast.parse(script_text)
        bare_stata = [ln for ln in script_text.splitlines()
                      if re.fullmatch(r"[A-Za-z_]\w*", ln.rstrip()) and transpile_stata_line(ln) not in (None, ln.strip())]
        if not re.search(r"(?m)^\s*(\*|//)", script_text) and not bare_stata:
            return script_text                                         # plain Python: nothing to translate
    except SyntaxError:
        pass
    py_lines = []
    depth = 0
    in_triple = False
    for raw in text.splitlines():
        line = re.sub(r"\s+//\s.*$", "", raw) if not in_triple else raw  # trailing // comments
        stripped = line.strip()
        triple_count = line.count('"""') + line.count("'''")
        top_level = depth <= 0 and not in_triple and not line[:1].isspace()
        if not top_level or not stripped:
            py_lines.append(line)
        else:
            transpiled = transpile_stata_line(line, ctx)
            if transpiled is None:
                py_lines.append(line if stripped.startswith("#") or not stripped else f"# {stripped}")
            elif transpiled.startswith("raise NotImplementedError") and (
                    _bracket_delta(line) > 0 or stripped.endswith((":", "\\", ",")) or triple_count % 2 == 1):
                py_lines.append(line)                                  # first line of a multi-line Python statement
            else:
                py_lines.append(transpiled)
        if in_triple:
            if triple_count % 2 == 1:
                in_triple = False
        else:
            if triple_count % 2 == 1:
                in_triple = True
            else:
                depth += _bracket_delta(line)
    return "\n".join(py_lines)