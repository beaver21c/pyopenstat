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


def transpile_stata_line(line: str) -> Optional[str]:
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
        return f"sess.compute({var + ' = ' + form!r})"

    # replace var = formula
    m = re.match(r"^replace\s+([A-Za-z0-9_]+)\s*=\s*(.+)$", s, re.IGNORECASE)
    if m:
        var = m.group(1)
        form = m.group(2).strip()
        return f"sess.compute({var + ' = ' + form!r})"

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

    # 12. sem (path specification)
    m = re.match(r"^sem\s*\((.+)\)", s, re.IGNORECASE)
    if m:
        spec = m.group(1)
        return f"sess.sem(formula={spec!r})"

    # 13. spregress dep indep1 ..., dvarlag(W) / errorlag(W)
    m = re.match(r"^spregress\s+([A-Za-z0-9_]+)\s+([^,]+)(?:,\s*(.*))?", s, re.IGNORECASE)
    if m:
        dep = m.group(1)
        indeps = [x.strip() for x in m.group(2).split() if x.strip()]
        opt_str = (m.group(3) or "").lower()
        m_type = "error" if "errorlag" in opt_str else "lag"
        return f"sess.spatial_regression(dep={dep!r}, covs={indeps!r}, model_type={m_type!r})"

    # 14. moran var
    m = re.match(r"^moran\s+([A-Za-z0-9_]+)", s, re.IGNORECASE)
    if m:
        return f"sess.spatial_regression(dep={m.group(1)!r}, model_type='moran_only')"

    # Return as-is if no rule matched
    return s


def transpile_stata_script(script_text: str) -> str:
    """Translates a full multi-line Stata .do script into Python syntax."""
    py_lines = []
    for line in script_text.splitlines():
        transpiled = transpile_stata_line(line)
        if transpiled is not None:
            py_lines.append(transpiled)
        else:
            py_lines.append(f"# {line}")
    return "\n".join(py_lines)
