"""
OpenStat Studio — Data Wrangling, Variable Transformation & Reshape/Merge Engine (`data_ops.py`)
License: MIT

Provides Stata `.do` / SPSS Syntax / jamovi Data-tab equivalent operations in pure Python:
1. `compute_variable(df, col_name, formula)`       : Formula-based derived variables (LOG, LN, EXP, SQRT, ABS, ROUND, Z, MEAN, SUM, IFELSE)
2. `recode_variable(df, source_col, target_col, rules, else_value='copy')` : Value/range/inequality recoding (`min..29='20s'; 30..49='30_40s'; 50..max='50+'`)
3. `filter_dataset(df, condition)`                 : Safe boolean expression row filtering (`age >= 20 and score > 0`)
4. `merge_datasets(left_df, right_df, on=None, how='left', suffixes=('', '_merged'))` : Horizontal relational join (1:1, m:1, left/inner/outer) & vertical `append`
5. `reshape_dataset(df, direction='long', ...)`    : Wide-to-Long (`melt` / multi-stub Stata `reshape long`) & Long-to-Wide (`pivot`)
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def load_tabular_source(source: Union[str, pd.DataFrame], root_dir: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[str]]:
    """Loads a DataFrame from a path (.csv, .tsv, .xlsx, .omv) or returns a copy of an in-memory DataFrame."""
    if isinstance(source, pd.DataFrame):
        return source.copy(), None
    if not isinstance(source, str):
        raise TypeError("Data source must be a file path string or pandas DataFrame.")

    resolved = source
    if not os.path.exists(resolved):
        base_root = root_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        cand1 = os.path.join(base_root, source)
        cand2 = os.path.join(base_root, "sample_data", os.path.basename(source))
        if os.path.exists(cand1):
            resolved = cand1
        elif os.path.exists(cand2):
            resolved = cand2

    ext = os.path.splitext(resolved)[1].lower()
    if ext == ".omv":
        from .omv.reader import read_omv
        df, _ = read_omv(resolved)
        return df, resolved
    elif ext in (".csv", ".tsv"):
        try:
            df = pd.read_csv(resolved, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(resolved)
        return df, resolved
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(resolved), resolved
    raise ValueError(f"Unsupported dataset format '{ext}'. Expected .csv, .tsv, .xlsx, or .omv.")


def compute_variable(df: pd.DataFrame, col_name: str, formula: str) -> pd.DataFrame:
    """
    Evaluates a vectorized formula to create or update `col_name` on a copy of `df`.
    Supported functions: LOG, LN, EXP, SQRT, ABS, ROUND, Z, MEAN, SUM, MIN, MAX, IFELSE.
    """
    target = str(col_name or "").strip()
    if not target:
        raise ValueError("Computed column name is required.")
    expr = str(formula or "").strip()
    if not expr:
        raise ValueError("Formula expression is required.")

    out = df.copy()

    def _zscore(s):
        num = pd.to_numeric(s, errors="coerce")
        sd = num.std(ddof=1)
        return (num - num.mean()) / sd if sd and not np.isnan(sd) else num * 0.0

    def _row_mean(*args):
        return pd.concat([pd.to_numeric(a, errors="coerce") for a in args], axis=1).mean(axis=1)

    def _row_sum(*args):
        return pd.concat([pd.to_numeric(a, errors="coerce") for a in args], axis=1).sum(axis=1)

    def _row_min(*args):
        return pd.concat([pd.to_numeric(a, errors="coerce") for a in args], axis=1).min(axis=1)

    def _row_max(*args):
        return pd.concat([pd.to_numeric(a, errors="coerce") for a in args], axis=1).max(axis=1)

    def _ifelse(cond, true_val, false_val):
        return np.where(cond, true_val, false_val)

    env: Dict[str, Any] = {
        "np": np,
        "pd": pd,
        "LOG": lambda s: np.log10(pd.to_numeric(s, errors="coerce")),
        "LN": lambda s: np.log(pd.to_numeric(s, errors="coerce")),
        "EXP": lambda s: np.exp(pd.to_numeric(s, errors="coerce")),
        "SQRT": lambda s: np.sqrt(pd.to_numeric(s, errors="coerce")),
        "ABS": lambda s: np.abs(pd.to_numeric(s, errors="coerce")),
        "ROUND": lambda s, d=2: np.round(pd.to_numeric(s, errors="coerce"), int(d)),
        "Z": _zscore,
        "MEAN": _row_mean,
        "SUM": _row_sum,
        "MIN": _row_min,
        "MAX": _row_max,
        "IFELSE": _ifelse,
        # Lowercase aliases for developer / interactive convenience
        "log": lambda s: np.log(pd.to_numeric(s, errors="coerce")),
        "log10": lambda s: np.log10(pd.to_numeric(s, errors="coerce")),
        "ln": lambda s: np.log(pd.to_numeric(s, errors="coerce")),
        "exp": lambda s: np.exp(pd.to_numeric(s, errors="coerce")),
        "sqrt": lambda s: np.sqrt(pd.to_numeric(s, errors="coerce")),
        "abs": lambda s: np.abs(pd.to_numeric(s, errors="coerce")),
        "round": lambda s, d=2: np.round(pd.to_numeric(s, errors="coerce"), int(d)),
        "z": _zscore,
        "mean": _row_mean,
        "sum": _row_sum,
        "min": _row_min,
        "max": _row_max,
        "ifelse": _ifelse,
    }
    for c in out.columns:
        env[str(c)] = out[c]

    result = eval(expr, {"__builtins__": {}}, env)
    out[target] = result
    return out


def _parse_scalar(token: str) -> Any:
    t = token.strip()
    if (len(t) >= 2) and ((t[0] == "'" and t[-1] == "'") or (t[0] == '"' and t[-1] == '"')):
        return t[1:-1]
    if t.lower() in ("nan", "na", "null", "missing", "."):
        return np.nan
    try:
        if "." in t:
            return float(t)
        return int(t)
    except ValueError:
        return t


def _parse_recode_rules(rules: Union[str, Dict[Any, Any], Sequence[Dict[str, Any]]]) -> List[Tuple[str, Any, Any]]:
    """
    Normalizes recode rules into a list of `(kind, lhs, rhs)` tuples where `kind` is one of:
    - `'missing'` : matches NaN
    - `'range'`   : `(low_or_none, high_or_none)` inclusive numeric interval
    - `'cmp'`     : `(op_str, threshold_float)` inequality (`<=`, `>=`, `<`, `>`, `==`, `!=`)
    - `'eq'`      : exact scalar or string match
    """
    parsed: List[Tuple[str, Any, Any]] = []

    if isinstance(rules, dict):
        for k, v in rules.items():
            parsed.extend(_parse_recode_rules(f"{k}={v!r}"))
        return parsed

    if isinstance(rules, (list, tuple)):
        for item in rules:
            if isinstance(item, dict):
                cond = str(item.get("from", item.get("cond", ""))).strip()
                to_val = item.get("to", item.get("value"))
                parsed.extend(_parse_recode_rules(f"{cond}={to_val!r}"))
        return parsed

    raw = str(rules or "").strip()
    if not raw:
        raise ValueError("Recode rules cannot be empty.")

    clauses = [c.strip() for c in re.split(r"[;\n]+", raw) if c.strip()]
    for clause in clauses:
        # Split on `->` or first `=` that is not part of `<=`, `>=`, `==`, `!=`
        if "->" in clause:
            lhs_str, rhs_str = clause.split("->", 1)
        else:
            m_eq = re.search(r"(?<![<>!=])=(?!=)", clause)
            if not m_eq:
                continue
            lhs_str = clause[:m_eq.start()]
            rhs_str = clause[m_eq.end():]

        lhs = lhs_str.strip()
        rhs = _parse_scalar(rhs_str)

        if lhs.lower() in ("missing", "na", "nan", "null"):
            parsed.append(("missing", None, rhs))
            continue

        # Range syntax: `min..29`, `30..49`, `50..max`, or `10:20`
        m_rng = re.match(
            r"^([-+]?(?:\d+\.?\d*|\.\d+)|min|lo|lowest|-inf)\s*(?:\.\.|:)\s*([-+]?(?:\d+\.?\d*|\.\d+)|max|hi|highest|inf|\+inf)$",
            lhs,
            re.IGNORECASE,
        )
        if m_rng:
            lo_raw, hi_raw = m_rng.group(1).strip().lower(), m_rng.group(2).strip().lower()
            lo = -np.inf if lo_raw in ("min", "lo", "lowest", "-inf") else float(lo_raw)
            hi = np.inf if hi_raw in ("max", "hi", "highest", "inf", "+inf") else float(hi_raw)
            parsed.append(("range", (lo, hi), rhs))
            continue

        # Inequality syntax: `<= 30`, `>= 60`, `< 0`, `> 100`
        m_cmp = re.match(r"^(<=|>=|!=|==|<|>)\s*([-+]?\d*\.?\d+)$", lhs)
        if m_cmp:
            parsed.append(("cmp", (m_cmp.group(1), float(m_cmp.group(2))), rhs))
            continue

        parsed.append(("eq", _parse_scalar(lhs), rhs))

    return parsed


def recode_variable(
    df: pd.DataFrame,
    source_col: str,
    target_col: Optional[str] = None,
    rules: Union[str, Dict[Any, Any], Sequence[Dict[str, Any]]] = "",
    else_value: Any = "copy",
) -> pd.DataFrame:
    """
    Recodes `source_col` into `target_col` (or overwrites `source_col` if `target_col` is None/same).
    Example `rules`:
      `"min..29='20대'; 30..49='30-40대'; 50..max='50대이상'"`
      `"1='Control'; 2='Treatment'"`
      `"<60='Fail'; >=60='Pass'"`
    """
    if source_col not in df.columns:
        raise ValueError(f"Source column '{source_col}' not found in dataset.")
    dst = (target_col or source_col).strip()
    parsed_rules = _parse_recode_rules(rules)

    out = df.copy()
    src = out[source_col]
    num_src = pd.to_numeric(src, errors="coerce")

    if str(else_value).lower() == "copy":
        result_vals: List[Any] = list(src.values)
    elif str(else_value).lower() in ("missing", "nan", "na", "null"):
        result_vals = [np.nan] * len(out)
    else:
        default_lit = _parse_scalar(str(else_value)) if isinstance(else_value, str) else else_value
        result_vals = [default_lit] * len(out)

    for idx in range(len(out)):
        orig_v = src.iloc[idx]
        num_v = num_src.iloc[idx]
        is_na = pd.isna(orig_v)

        for kind, cond_spec, target_v in parsed_rules:
            matched = False
            if kind == "missing":
                matched = bool(is_na)
            elif not is_na:
                if kind == "range" and not pd.isna(num_v):
                    lo, hi = cond_spec
                    matched = bool(lo <= float(num_v) <= hi)
                elif kind == "cmp" and not pd.isna(num_v):
                    op, thr = cond_spec
                    fv = float(num_v)
                    if op == "<=":
                        matched = fv <= thr
                    elif op == ">=":
                        matched = fv >= thr
                    elif op == "<":
                        matched = fv < thr
                    elif op == ">":
                        matched = fv > thr
                    elif op == "==":
                        matched = fv == thr
                    elif op == "!=":
                        matched = fv != thr
                elif kind == "eq":
                    if not pd.isna(num_v) and isinstance(cond_spec, (int, float)):
                        matched = bool(float(num_v) == float(cond_spec))
                    else:
                        matched = bool(str(orig_v).strip() == str(cond_spec).strip())
            if matched:
                result_vals[idx] = target_v
                break

    out[dst] = result_vals
    return out


def filter_dataset(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """
    Filters rows matching a boolean expression (Stata `keep if ...`).
    Supports both Pythonic (`and`/`or`/`&`/`|`) and `df.query` syntax.
    """
    cond = str(condition or "").strip()
    if not cond:
        return df.copy()
    try:
        filtered = df.query(cond, engine="python")
        return filtered.reset_index(drop=True).copy()
    except Exception:
        env: Dict[str, Any] = {"np": np, "pd": pd}
        for c in df.columns:
            env[str(c)] = df[c]
        mask = eval(cond, {"__builtins__": {}}, env)
        return df.loc[mask].reset_index(drop=True).copy()


def merge_datasets(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    how: str = "left",
    suffixes: Tuple[str, str] = ("", "_merged"),
) -> pd.DataFrame:
    """
    Relational merge (`left`, `inner`, `right`, `outer`) or vertical row stacking (`append`),
    matching Stata `merge 1:1 / m:1` and `append using`.
    """
    mode = str(how or "left").strip().lower()
    if mode in ("append", "concat", "rows"):
        return pd.concat([left_df, right_df], axis=0, ignore_index=True)

    if mode not in ("left", "inner", "right", "outer"):
        raise ValueError(f"Unsupported merge mode '{how}'. Choose left, inner, right, outer, or append.")

    join_keys: Optional[Union[str, List[str]]] = None
    if isinstance(on, str) and on.strip():
        join_keys = [k.strip() for k in on.split(",") if k.strip()]
        if len(join_keys) == 1:
            join_keys = join_keys[0]
    elif isinstance(on, (list, tuple)) and len(on) > 0:
        join_keys = [str(k).strip() for k in on if str(k).strip()]

    if not join_keys:
        common = [c for c in left_df.columns if c in right_df.columns]
        if not common:
            raise ValueError("No common join key columns found between datasets; specify `on=...`.")
        join_keys = common[0]

    merged = pd.merge(left_df, right_df, on=join_keys, how=mode, suffixes=suffixes)
    return merged.reset_index(drop=True)


def reshape_dataset(
    df: pd.DataFrame,
    direction: str = "long",
    id_vars: Optional[Union[str, List[str]]] = None,
    value_vars: Optional[Union[str, List[str]]] = None,
    stubnames: Optional[Union[str, List[str]]] = None,
    time_col: str = "time",
    value_col: str = "value",
    sep: str = "",
    strip_prefix: bool = True,
) -> pd.DataFrame:
    """
    Reshapes dataset between Wide and Long formats (Stata `reshape long` / `reshape wide` equivalent):
    - `direction='long'`:
        * Multi-stub mode (`stubnames=['score_', 'stress_']`, `id_vars=['id', 'group']`, `time_col='wave'`):
          reshapes multiple repeated series simultaneously into panel columns (`id, group, wave, score, stress`).
        * Explicit `value_vars` mode (`value_vars=['y_t1', 'y_t2', 'y_t3']`, `id_vars=['id', 'group']`,
          `time_col='wave', value_col='score'`): melts wide wave columns into Long format.
    - `direction='wide'`:
        * Pivots Long data back into Wide columns using `id_vars` as row index, `time_col` as column suffix,
          and `value_col` (or `value_vars`) as values.
    """
    mode = str(direction or "long").strip().lower()

    def _norm_cols(val: Optional[Union[str, List[str]]]) -> List[str]:
        if val is None:
            return []
        if isinstance(val, str):
            return [x.strip() for x in val.split(",") if x.strip()]
        return [str(x).strip() for x in val if str(x).strip()]

    ids = [c for c in _norm_cols(id_vars) if c in df.columns]
    v_vars = [c for c in _norm_cols(value_vars) if c in df.columns]
    stubs = _norm_cols(stubnames)

    if mode == "long":
        if stubs:
            # Stata-style multi-stub reshape long
            clean_df = df.copy()
            if not ids:
                matched_cols = {
                    c for c in clean_df.columns
                    if any(str(c).startswith(st) and str(c) != st for st in stubs)
                }
                ids = [c for c in clean_df.columns if c not in matched_cols]
            if not ids:
                clean_df["_row_id"] = np.arange(1, len(clean_df) + 1)
                ids = ["_row_id"]

            long_df = pd.wide_to_long(
                clean_df,
                stubnames=stubs,
                i=ids,
                j=time_col,
                sep=sep,
                suffix=r".+",
            ).reset_index()
            # Clean trailing underscore from stub column names if present (e.g. `score_` -> `score`)
            rename_map = {st: st.rstrip("_") for st in stubs if st.endswith("_") and st.rstrip("_") not in long_df.columns}
            if rename_map:
                long_df.rename(columns=rename_map, inplace=True)
            return long_df

        if not v_vars:
            v_vars = [c for c in df.columns if c not in ids and pd.api.types.is_numeric_dtype(df[c])]
        if not v_vars:
            raise ValueError("Specify `value_vars` or `stubnames` for Wide -> Long reshape.")
        if not ids:
            ids = [c for c in df.columns if c not in v_vars]

        melted = pd.melt(
            df,
            id_vars=ids,
            value_vars=v_vars,
            var_name=time_col,
            value_name=value_col,
        )
        if strip_prefix and len(v_vars) >= 2:
            common_pref = os.path.commonprefix([str(v) for v in v_vars])
            if common_pref and all(len(str(v)) > len(common_pref) for v in v_vars):
                melted[time_col] = melted[time_col].astype(str).str[len(common_pref):].str.lstrip("_.")
        # Sort by id_vars + time_col like Stata `reshape long`
        sort_cols = [c for c in ids if c in melted.columns] + [time_col]
        return melted.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    elif mode == "wide":
        if time_col not in df.columns:
            raise ValueError(f"Pivot/time column '{time_col}' not found in dataset.")
        val_targets = v_vars if v_vars else ([value_col] if value_col in df.columns else [])
        if not val_targets:
            val_targets = [c for c in df.columns if c not in ids and c != time_col]
        if not val_targets:
            raise ValueError("Specify `value_col` or `value_vars` for Long -> Wide reshape.")
        if not ids:
            ids = [c for c in df.columns if c not in val_targets and c != time_col]
        if not ids:
            raise ValueError("Specify at least one `id_vars` column for Long -> Wide reshape.")

        piv = df.pivot_table(
            index=ids,
            columns=time_col,
            values=val_targets if len(val_targets) > 1 else val_targets[0],
            aggfunc="first",
        )
        if isinstance(piv.columns, pd.MultiIndex):
            piv.columns = [f"{v}_{t}" for v, t in piv.columns]
        else:
            prefix = val_targets[0]
            piv.columns = [f"{prefix}_{t}" for t in piv.columns]
        return piv.reset_index()

    raise ValueError(f"Unsupported reshape direction '{direction}'. Use 'long' or 'wide'.")
