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


def read_csv_robust(path: str) -> pd.DataFrame:
    """
    Read a delimited text file. The delimiter is taken from the header line (',' ';' tab '|'),
    so long free-text cells (documents, interview transcripts) cannot confuse delimiter sniffing.
    Encodings tried: UTF-8 (with/without BOM), then CP949 (Korean Excel exports).
    """
    last_err: Optional[Exception] = None
    for enc in ("utf-8-sig", "cp949"):
        try:
            with open(path, "r", encoding=enc, newline="") as fh:
                header = fh.readline()
        except UnicodeDecodeError as e:
            last_err = e
            continue
        counts = {d: header.count(d) for d in (",", ";", "\t", "|")}
        sep = max(counts, key=counts.get) if max(counts.values()) > 0 else None
        if path.lower().endswith(".tsv"):
            sep = "\t"
        try:
            if sep is None:
                return pd.read_csv(path, sep=None, engine="python", encoding=enc)
            return pd.read_csv(path, sep=sep, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise ValueError(f"CSV 파일 읽기 실패 (UTF-8·CP949 모두 실패): {path}") from last_err


SUPPORTED_DATA_EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".omv", ".parquet", ".dta", ".sav", ".zsav", ".por")


def _resolve_data_path(source: str, root_dir: Optional[str] = None) -> str:
    if os.path.exists(source):
        return source
    base_root = root_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for cand in (os.path.join(base_root, source), os.path.join(base_root, "sample_data", os.path.basename(source))):
        if os.path.exists(cand):
            return cand
    return source


def read_dataset(path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Read a data file and its metadata.

    Returns (df, meta) with meta = {"value_labels": {col: {code: label}}, "variable_labels": {col: label}}.
    Stata .dta keeps numeric codes and stores value labels in meta (Stata `encode`/`label values` semantics);
    SPSS .sav/.zsav/.por need pyreadstat.
    """
    ext = os.path.splitext(path)[1].lower()
    meta: Dict[str, Any] = {"value_labels": {}, "variable_labels": {}}
    if ext == ".omv":
        from .omv.reader import read_omv
        df, _ = read_omv(path)
    elif ext in (".csv", ".tsv", ".txt"):
        df = read_csv_robust(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".dta":
        try:
            import pyreadstat
        except ImportError:
            pyreadstat = None
        if pyreadstat is not None:
            df, m = pyreadstat.read_dta(path, apply_value_formats=False)
            meta["value_labels"] = {c: {(int(k) if float(k).is_integer() else k): str(v) for k, v in labs.items()}
                                    for c, labs in (m.variable_value_labels or {}).items()}
            meta["variable_labels"] = {c: l for c, l in zip(m.column_names, m.column_labels or []) if l}
        else:
            with pd.io.stata.StataReader(path) as rdr:
                df = rdr.read(convert_categoricals=False, convert_missing=False)
                vlab = rdr.value_labels()
                lbl_names = list(getattr(rdr, "_lbllist", []) or [])
                var_labels = rdr.variable_labels()
            for col, lname in zip(df.columns, lbl_names):
                if lname and lname in vlab:
                    meta["value_labels"][col] = {(int(k) if float(k).is_integer() else k): str(v) for k, v in vlab[lname].items()}
            meta["variable_labels"] = {k: v for k, v in (var_labels or {}).items() if v}
    elif ext in (".sav", ".zsav", ".por"):
        try:
            import pyreadstat
        except ImportError as e:  # pragma: no cover
            raise ValueError("SPSS 파일 읽기에는 pyreadstat 필요: pip install pyreadstat") from e
        reader = pyreadstat.read_por if ext == ".por" else pyreadstat.read_sav
        df, m = reader(path, apply_value_formats=False)
        meta["value_labels"] = {c: {(int(k) if float(k).is_integer() else k): str(v) for k, v in labs.items()}
                                for c, labs in (m.variable_value_labels or {}).items()}
        meta["variable_labels"] = {c: l for c, l in zip(m.column_names, m.column_labels or []) if l}
    else:
        raise ValueError(f"지원하지 않는 자료 형식 '{ext}' (가능: {', '.join(SUPPORTED_DATA_EXTENSIONS)})")
    return df, meta


def load_tabular_source(source: Union[str, pd.DataFrame], root_dir: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[str]]:
    """Loads a DataFrame from a path (see SUPPORTED_DATA_EXTENSIONS) or returns a copy of an in-memory DataFrame."""
    if isinstance(source, pd.DataFrame):
        return source.copy(), None
    if not isinstance(source, str):
        raise TypeError("Data source must be a file path string or pandas DataFrame.")
    resolved = _resolve_data_path(source, root_dir)
    df, _ = read_dataset(resolved)
    return df, resolved


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


# ---------------------------------------------------------------------------
# Combining datasets (Stata merge / joinby / append)
# ---------------------------------------------------------------------------
MERGE_KINDS = ("1:1", "m:1", "1:m", "m:m")
MERGE_RESULT_LABELS = {1: "master only (1)", 2: "using only (2)", 3: "matched (3)",
                       4: "missing updated (4)", 5: "nonmissing conflict (5)"}
_KEEP_ALIASES = {"master": [1], "1": [1], "using": [2], "2": [2], "match": [3, 4, 5], "matched": [3, 4, 5], "3": [3],
                 "match_update": [4], "4": [4], "match_conflict": [5], "5": [5]}
_HOW_TO_KEEP = {"left": ["master", "match"], "inner": ["match"], "right": ["using", "match"],
                "outer": ["master", "using", "match"]}


def _as_name_list(val: Optional[Union[str, Sequence[str]]]) -> List[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [x.strip() for x in re.split(r"[,\s]+", val) if x.strip()]
    return [str(x).strip() for x in val if str(x).strip()]


def _merge_keys(on, left: pd.DataFrame, right: pd.DataFrame) -> List[str]:
    keys = [k.strip() for k in on.split(",") if k.strip()] if isinstance(on, str) else _as_name_list(on)
    if not keys:
        common = [c for c in left.columns if c in right.columns]
        raise ValueError(f"병합 키(on) 지정 필요 (Stata merge의 키 변수 목록). 두 자료의 공통 변수: {common}")
    for k in keys:
        if k not in left.columns:
            raise ValueError(f"키 변수 '{k}'가 현재 자료(master)에 없음")
        if k not in right.columns:
            raise ValueError(f"키 변수 '{k}'가 결합할 자료(using)에 없음")
    return keys


def _dup_message(df: pd.DataFrame, keys: List[str], side: str, kind: str) -> str:
    dup = df[df.duplicated(subset=keys, keep=False)]
    n_keys = len(dup.drop_duplicates(subset=keys))
    ex = dup[keys].drop_duplicates().head(3).to_dict("records")
    return (f"키 변수 {', '.join(keys)}가 {side} 자료에서 관측치를 고유하게 식별하지 않음 "
            f"(중복 키 {n_keys}개, 예: {ex}) — {kind} 병합 불가 (Stata r(459))")


def _resolve_keep(keep, how: Optional[str]) -> List[int]:
    if keep is None or keep == [] or keep == "":
        names = _HOW_TO_KEEP.get(how or "outer", _HOW_TO_KEEP["outer"])
    else:
        names = _as_name_list(keep) if isinstance(keep, str) else [str(k) for k in keep]
    codes: List[int] = []
    for n in names:
        c = _KEEP_ALIASES.get(str(n).strip().lower())
        if c is None:
            raise ValueError(f"keep 값 '{n}' 인식 불가 (가능: master, using, match, match_update, match_conflict 또는 1~5)")
        codes.extend(c)
    return sorted(set(codes))


def _combine_overlap(merged: pd.DataFrame, overlap: List[str], code: np.ndarray, update: bool, replace: bool) -> np.ndarray:
    """Stata rules for variables present in both files: master values kept; using fills using-only rows;
    `update` fills missing master values (code 4); `replace` also overwrites conflicting values (code 5)."""
    code = code.copy()
    matched = code == 3
    any_upd = np.zeros(len(merged), dtype=bool)
    any_conf = np.zeros(len(merged), dtype=bool)
    for v in overlap:
        mv, uv = merged[v], merged[v + "__using"]
        using_only = code == 2
        out = mv.where(~using_only, uv)
        if update:
            m_na, u_ok = mv.isna().values, uv.notna().values
            upd = matched & m_na & u_ok
            conf = matched & ~m_na & u_ok & (mv.astype(object).values != uv.astype(object).values)
            out = out.where(~upd, uv)
            if replace:
                out = out.where(~conf, uv)
            any_upd |= upd
            any_conf |= conf
        merged[v] = out
        merged.drop(columns=[v + "__using"], inplace=True)
    if update:
        code[matched & any_upd] = 4
        code[matched & any_conf] = 5
    return code


def _mm_sequential(left: pd.DataFrame, right: pd.DataFrame, keys: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stata m:m: within a key group the k-th master row is paired with the k-th using row; the last row of the
    shorter group is repeated for the remaining rows of the longer group."""
    both = pd.concat([left[keys], right[keys]], ignore_index=True)
    gid = both.groupby(keys, dropna=False, sort=False).ngroup().to_numpy()
    gl, gr = gid[: len(left)], gid[len(left):]
    L = left.assign(__gid=gl, __seq=left.groupby(gl, sort=False).cumcount().to_numpy())
    R = right.assign(__gid=gr, __seq=right.groupby(gr, sort=False).cumcount().to_numpy())
    nl, nr = pd.Series(gl).value_counts(), pd.Series(gr).value_counts()
    common = nl.index.intersection(nr.index)

    def extend(D: pd.DataFrame, own: pd.Series, other: pd.Series) -> pd.DataFrame:
        need = {g: int(other[g] - own[g]) for g in common if other[g] > own[g]}
        if not need:
            return D
        last = D.groupby("__gid", sort=False).tail(1).set_index("__gid")
        parts = [D]
        for g, k in need.items():
            rep = last.loc[[g] * k].reset_index()
            rep["__seq"] = np.arange(own[g], own[g] + k)
            parts.append(rep[D.columns])
        return pd.concat(parts, ignore_index=True)

    return extend(L, nl, nr), extend(R, nr, nl)


def merge_datasets(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    how: Optional[str] = None,
    suffixes: Tuple[str, str] = ("", "_merged"),
    kind: Optional[str] = None,
    keep: Optional[Union[str, List[str]]] = None,
    keepusing: Optional[Union[str, List[str]]] = None,
    indicator: Optional[Union[str, bool]] = None,
    update: bool = False,
    replace: bool = False,
    assert_results: Optional[Union[str, List[str]]] = None,
    return_report: bool = False,
):
    """Key-based merge following Stata `merge` semantics, or vertical `append` (`how='append'`).

    - `kind`: "1:1" | "m:1" | "1:m" | "m:m". When omitted it is inferred from key uniqueness; duplicated keys on
      both sides raise an error (no silent many-to-many expansion; use kind="m:m" or `joinby_datasets`).
    - `keep`: subset of master / using / match / match_update / match_conflict (Stata keep()); default all, or the
      rows implied by the legacy `how` (left / inner / right / outer).
    - Stata mode (default): variables present in both files keep master values (`update` fills missing master values,
      `replace` overwrites conflicts), `_merge` indicator 1–5 is added (`indicator=False` to skip; Stata nogenerate).
    - Legacy mode (`how` given without kind/update/replace): pandas-style suffixes for overlapping variables, no
      indicator unless requested, same cardinality checks.
    """
    mode = str(how or "").strip().lower()
    if mode in ("append", "concat", "rows"):
        out = append_datasets(left_df, right_df)
        return (out, {"appended": len(right_df)}) if return_report else out
    if mode and mode not in _HOW_TO_KEEP:
        raise ValueError(f"how='{how}' 인식 불가 (가능: left, inner, right, outer, append)")
    if replace and not update:
        raise ValueError("replace는 update와 함께 지정 (Stata: merge ..., update replace)")
    stata_mode = (not mode) or kind is not None or update
    keys = _merge_keys(on, left_df, right_df)
    right = right_df
    ku = _as_name_list(keepusing)
    if ku:
        miss = [c for c in ku if c not in right.columns]
        if miss:
            raise ValueError(f"keepusing 변수가 using 자료에 없음: {miss}")
        right = right[keys + [c for c in ku if c not in keys]]

    lu = not left_df.duplicated(subset=keys).any()
    ru = not right.duplicated(subset=keys).any()
    inferred = kind is None
    if kind is None:
        if lu and ru:
            kind = "1:1"
        elif ru:
            kind = "m:1"
        elif lu:
            kind = "1:m"
        else:
            raise ValueError(_dup_message(right, keys, "using", "m:m 확장 우려로")
                             + ". 두 자료 모두 키 중복 — 순차 짝짓기는 kind='m:m', 모든 조합은 joinby 사용")
    kind = str(kind).strip().lower()
    if kind not in MERGE_KINDS:
        raise ValueError(f"kind='{kind}' 인식 불가 (가능: {', '.join(MERGE_KINDS)})")
    if kind in ("1:1", "1:m") and not lu:
        raise ValueError(_dup_message(left_df, keys, "master(현재)", kind))
    if kind in ("1:1", "m:1") and not ru:
        raise ValueError(_dup_message(right, keys, "using(결합)", kind))

    keep_codes = _resolve_keep(keep, mode or None)
    ind_name = None
    if indicator is None:
        ind_name = "_merge" if stata_mode else None
    elif indicator is not False and indicator != "":
        ind_name = "_merge" if indicator is True else str(indicator)
    if ind_name and ind_name in left_df.columns:
        raise ValueError(f"변수 '{ind_name}'가 이미 있음 — 삭제하거나 indicator에 다른 이름 지정 (Stata: _merge already defined)")

    if kind == "m:m":
        L, R = _mm_sequential(left_df, right, keys)
        on_cols = keys + ["__seq"]
        L, R = L.drop(columns="__gid"), R.drop(columns="__gid")
    else:
        L, R, on_cols = left_df, right, keys

    overlap = [c for c in R.columns if c in L.columns and c not in on_cols]
    if stata_mode:
        merged = pd.merge(L, R, on=on_cols, how="outer", sort=False, suffixes=("", "__using"), indicator="__m")
    else:
        merged = pd.merge(L, R, on=on_cols, how="outer", sort=False, suffixes=suffixes, indicator="__m")
    code = merged["__m"].map({"left_only": 1, "right_only": 2, "both": 3}).to_numpy().astype(int)
    merged = merged.drop(columns="__m")
    if stata_mode and overlap:
        code = _combine_overlap(merged, overlap, code, update, replace)
    if "__seq" in merged.columns:
        merged = merged.drop(columns="__seq")

    counts = {k: int((code == k).sum()) for k in MERGE_RESULT_LABELS}
    if assert_results:
        allowed = set(_resolve_keep(assert_results, None))
        bad = {k: n for k, n in counts.items() if n and k not in allowed}
        if bad:
            raise ValueError(f"assert 위반: 허용되지 않은 병합 결과 {bad} (Stata r(9))")
    sel = np.isin(code, keep_codes)
    merged = merged.loc[sel]
    if ind_name:
        merged[ind_name] = code[sel].astype("int8")
    merged = merged.reset_index(drop=True)
    report = {"kind": kind, "inferred": inferred, "keys": keys, "counts": counts, "kept": int(sel.sum()),
              "overlap": overlap if stata_mode else [], "indicator": ind_name}
    return (merged, report) if return_report else merged


def joinby_datasets(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    unmatched: str = "none",
    indicator: Optional[str] = "_merge",
    update: bool = False,
    replace: bool = False,
    return_report: bool = False,
):
    """Stata `joinby`: all pairwise combinations within key groups. `unmatched`: none | both | master | using.
    Overlapping variables keep master values unless update/replace. The indicator is added when unmatched != none."""
    um = str(unmatched or "none").strip().lower()
    if um not in ("none", "both", "master", "using"):
        raise ValueError(f"unmatched='{unmatched}' 인식 불가 (가능: none, both, master, using)")
    if replace and not update:
        raise ValueError("replace는 update와 함께 지정")
    keys = _merge_keys(on, left_df, right_df)
    overlap = [c for c in right_df.columns if c in left_df.columns and c not in keys]
    merged = pd.merge(left_df, right_df, on=keys, how="outer", sort=False, suffixes=("", "__using"), indicator="__m")
    code = merged["__m"].map({"left_only": 1, "right_only": 2, "both": 3}).to_numpy().astype(int)
    merged = merged.drop(columns="__m")
    if overlap:
        code = _combine_overlap(merged, overlap, code, update, replace)
    keep_codes = {"none": [3, 4, 5], "both": [1, 2, 3, 4, 5], "master": [1, 3, 4, 5], "using": [2, 3, 4, 5]}[um]
    sel = np.isin(code, keep_codes)
    counts = {k: int((code == k).sum()) for k in MERGE_RESULT_LABELS}
    merged = merged.loc[sel]
    if um != "none" and indicator:
        if indicator in left_df.columns:
            raise ValueError(f"변수 '{indicator}'가 이미 있음 — 다른 이름 지정")
        merged[indicator] = code[sel].astype("int8")
    merged = merged.reset_index(drop=True)
    report = {"kind": "joinby", "keys": keys, "counts": counts, "kept": int(sel.sum())}
    return (merged, report) if return_report else merged


def append_datasets(master: pd.DataFrame, using: Union[pd.DataFrame, Sequence[pd.DataFrame]],
                    generate: Optional[str] = None, force: bool = False) -> pd.DataFrame:
    """Stata `append`: stack rows; variables are matched by name. A text/numeric mismatch raises unless `force`
    (then the using values of that variable become missing). `generate` marks the source: 0 = master, 1.. = using files."""
    usings = [using] if isinstance(using, pd.DataFrame) else list(using)
    parts = [master]
    for j, u in enumerate(usings, 1):
        u = u.copy()
        for c in u.columns:
            if c in master.columns:
                m_num = pd.api.types.is_numeric_dtype(master[c]) or master[c].isna().all()
                u_num = pd.api.types.is_numeric_dtype(u[c]) or u[c].isna().all()
                if m_num != u_num:
                    if not force:
                        raise ValueError(f"변수 '{c}'의 형식 불일치(현재 {'숫자' if m_num else '문자'}, 추가 자료 {'숫자' if u_num else '문자'}) "
                                         "— force=True로 추가 자료의 값을 결측 처리하거나 형식 변환 후 실행 (Stata r(106))")
                    u[c] = np.nan
        parts.append(u)
    if generate:
        if generate in master.columns:
            raise ValueError(f"변수 '{generate}'가 이미 있음")
        parts = [p.assign(**{generate: k}) for k, p in enumerate(parts)]
    return pd.concat(parts, axis=0, ignore_index=True, sort=False)


# ---------------------------------------------------------------------------
# Reshape (Stata reshape long / wide)
# ---------------------------------------------------------------------------
def _fmt_j(v: Any) -> str:
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


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
    Reshape between Wide and Long (Stata `reshape long` / `reshape wide`).
    - long, `stubnames` (Stata stubs, e.g. ['bp_w', 'qol_w']) with `id_vars` = i(): i must uniquely identify rows.
    - long, explicit `value_vars` (melt into `value_col`).
    - wide: `id_vars` = i() (required, no missing values), `time_col` = j(), `value_vars`/`stubnames` = stubs
      (default `value_col`). New names are stub + sep + j (Stata default sep ""). Errors as in Stata when (i, j) is not
      unique or a variable outside the stubs varies within i.
    """
    mode = str(direction or "long").strip().lower()

    def _norm_cols(val: Optional[Union[str, List[str]]]) -> List[str]:
        if val is None:
            return []
        if isinstance(val, str):
            return [x.strip() for x in val.split(",") if x.strip()]
        return [str(x).strip() for x in val if str(x).strip()]

    ids_in = _norm_cols(id_vars)
    missing_ids = [c for c in ids_in if c not in df.columns]
    if missing_ids:
        raise ValueError(f"식별 변수(i)가 자료에 없음: {missing_ids}")
    ids = ids_in
    v_vars = _norm_cols(value_vars)
    bad_v = [c for c in v_vars if c not in df.columns]
    if bad_v and mode != "wide":
        raise ValueError(f"value_vars 변수가 자료에 없음: {bad_v}")
    stubs = _norm_cols(stubnames)

    if mode == "long":
        if stubs:
            clean_df = df.copy()
            for st in stubs:
                if not any(str(c).startswith(st + sep) and str(c) != st for c in clean_df.columns):
                    raise ValueError(f"stub '{st}'로 시작하는 변수 없음")
            if not ids:
                matched_cols = {c for c in clean_df.columns if any(str(c).startswith(st) and str(c) != st for st in stubs)}
                ids = [c for c in clean_df.columns if c not in matched_cols]
            if not ids:
                clean_df["_row_id"] = np.arange(1, len(clean_df) + 1)
                ids = ["_row_id"]
            if clean_df.duplicated(subset=ids).any():
                n = int(clean_df.duplicated(subset=ids, keep=False).sum())
                raise ValueError(f"식별 변수 {ids}가 관측치를 고유하게 식별하지 않음 (중복 {n}행) — Stata: variable id does not uniquely identify the observations")
            long_df = pd.wide_to_long(clean_df, stubnames=stubs, i=ids, j=time_col, sep=sep, suffix=r".+").reset_index()
            rename_map = {st: st.rstrip("_") for st in stubs if st.endswith("_") and st.rstrip("_") not in long_df.columns}
            if rename_map:
                long_df.rename(columns=rename_map, inplace=True)
            return long_df.sort_values(ids + [time_col], kind="mergesort").reset_index(drop=True)

        if not v_vars:
            v_vars = [c for c in df.columns if c not in ids and pd.api.types.is_numeric_dtype(df[c])]
        if not v_vars:
            raise ValueError("Wide → Long 변환: stubnames 또는 value_vars 지정 필요")
        if not ids:
            ids = [c for c in df.columns if c not in v_vars]
        melted = pd.melt(df, id_vars=ids, value_vars=v_vars, var_name=time_col, value_name=value_col)
        if strip_prefix and len(v_vars) >= 2:
            common_pref = os.path.commonprefix([str(v) for v in v_vars])
            if common_pref and all(len(str(v)) > len(common_pref) for v in v_vars):
                melted[time_col] = melted[time_col].astype(str).str[len(common_pref):].str.lstrip("_.")
        sort_cols = [c for c in ids if c in melted.columns] + [time_col]
        return melted.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    if mode == "wide":
        if time_col not in df.columns:
            raise ValueError(f"시점 변수(j) '{time_col}'가 자료에 없음")
        targets = v_vars or stubs or ([value_col] if value_col in df.columns else [])
        bad_t = [c for c in targets if c not in df.columns]
        if bad_t:
            raise ValueError(f"wide로 펼칠 변수가 자료에 없음: {bad_t}")
        if not targets:
            raise ValueError("Long → Wide 변환: 펼칠 변수(value_vars) 지정 필요")
        if not ids:
            raise ValueError("Long → Wide 변환: 식별 변수(i) 지정 필요 (Stata reshape wide ..., i() j())")
        if df[ids].isna().any().any():
            n = int(df[ids].isna().any(axis=1).sum())
            raise ValueError(f"식별 변수(i)에 결측값 {n}행 — 결측 행 처리 후 실행")
        if df[time_col].isna().any():
            raise ValueError(f"시점 변수(j) '{time_col}'에 결측값 {int(df[time_col].isna().sum())}행 — 결측 행 처리 후 실행")
        dup = df.duplicated(subset=ids + [time_col], keep=False)
        if dup.any():
            ex = df.loc[dup, ids + [time_col]].drop_duplicates().head(3).to_dict("records")
            raise ValueError(f"(i, j) 조합 중복 {int(dup.sum())}행, 예: {ex} — Stata: values of variable j not unique within i")
        others = [c for c in df.columns if c not in ids + [time_col] + targets]
        if others:
            nun = df.groupby(ids, dropna=False, sort=False)[others].nunique(dropna=False)
            varying = [c for c in others if (nun[c] > 1).any()]
            if varying:
                raise ValueError(f"i 안에서 값이 변하는 변수 {varying[:6]} — value_vars에 포함하거나 삭제 후 실행 "
                                 "(Stata: variable not constant within i)")
        piv = df.pivot(index=ids, columns=time_col, values=targets)
        j_vals = list(piv.columns.get_level_values(1).unique())
        new_cols = []
        for t in targets:
            for j in j_vals:
                new_cols.append((t, j))
        piv = piv[new_cols]
        names = [f"{t}{sep}{_fmt_j(j)}" for t, j in new_cols]
        clash = [n for n in names if n in others or n in ids]
        if clash:
            raise ValueError(f"새 변수명이 기존 변수와 중복: {clash[:5]}")
        piv.columns = names
        out = piv.reset_index()
        if others:
            const = df.groupby(ids, dropna=False, sort=False)[others].first().reset_index()
            out = out.merge(const, on=ids, how="left")
        return out

    raise ValueError(f"reshape 방향 '{direction}' 인식 불가 (가능: long, wide)")


# ---------------------------------------------------------------------------
# Aggregation (Stata collapse / egen)
# ---------------------------------------------------------------------------
_STAT_FUNCS = {
    "mean": "mean", "median": "median", "sum": "sum", "count": "count", "min": "min", "max": "max",
    "sd": "std", "first": "first", "last": "last", "nunique": "nunique",
    "p10": lambda s: s.quantile(.10), "p25": lambda s: s.quantile(.25), "p75": lambda s: s.quantile(.75),
    "p90": lambda s: s.quantile(.90),
}
_STAT_ALIASES = {"total": "sum", "std": "sd", "n": "count", "p50": "median", "firstnm": "first", "lastnm": "last"}


def _stat_name(st: str) -> str:
    k = str(st).strip().lower()
    k = _STAT_ALIASES.get(k, k)
    if k not in _STAT_FUNCS:
        raise ValueError(f"통계 '{st}' 인식 불가 (가능: {', '.join(_STAT_FUNCS)})")
    return k


def collapse_dataset(df: pd.DataFrame, stats: Dict[str, Any], by: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
    """Stata `collapse (stat) varlist, by(groups)`.
    `stats`: {"var": "mean"} or {"mean": ["y", "z"], "sum": ["w"]} or {"new_name": ("var", "stat")}."""
    by_cols = _as_name_list(by)
    spec: Dict[str, Tuple[str, str]] = {}
    for k, v in (stats or {}).items():
        if isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], str) and str(v[1]).lower() in set(_STAT_FUNCS) | set(_STAT_ALIASES) and v[0] in df.columns:
            spec[k] = (v[0], _stat_name(v[1]))
        elif str(k).lower() in set(_STAT_FUNCS) | set(_STAT_ALIASES) and k not in df.columns:
            for col in _as_name_list(v):
                spec[col] = (col, _stat_name(k))
        else:
            spec[k] = (k, _stat_name(v))
    if not spec:
        raise ValueError("collapse: 집계할 변수와 통계 지정 필요")
    miss = sorted({c for c, _ in spec.values() if c not in df.columns} | {c for c in by_cols if c not in df.columns})
    if miss:
        raise ValueError(f"collapse: 자료에 없는 변수 {miss}")
    agg = {new: pd.NamedAgg(column=col, aggfunc=_STAT_FUNCS[st]) for new, (col, st) in spec.items()}
    if by_cols:
        return df.groupby(by_cols, dropna=False, sort=True).agg(**agg).reset_index()
    return df.assign(__all=0).groupby("__all").agg(**agg).reset_index(drop=True)


_EGEN_FUNCS = set(_STAT_FUNCS) | {"rank", "group", "tag", "rowmean", "rowtotal", "rowmax", "rowmin", "rowmiss"}


def egen_variable(df: pd.DataFrame, target: str, func: str, source: Optional[Union[str, List[str]]] = None,
                  by: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
    """Stata `egen target = func(source), by(groups)`. Group statistics are broadcast to every row;
    `group` numbers the distinct combinations of `source` (1..k), `tag` marks the first row of each `by` group,
    row* functions work across the listed variables."""
    fn = str(func).strip().lower()
    fn = {"total": "sum", "std": "sd", "n": "count"}.get(fn, fn)
    if fn not in _EGEN_FUNCS:
        raise ValueError(f"egen 함수 '{func}' 인식 불가 (가능: {', '.join(sorted(_EGEN_FUNCS))})")
    if target in df.columns:
        raise ValueError(f"변수 '{target}'가 이미 있음 (Stata r(110))")
    src = _as_name_list(source)
    by_cols = _as_name_list(by)
    miss = [c for c in src + by_cols if c not in df.columns]
    if miss:
        raise ValueError(f"egen: 자료에 없는 변수 {miss}")
    out = df.copy()
    if fn.startswith("row"):
        block = out[src].apply(pd.to_numeric, errors="coerce")
        out[target] = {"rowmean": block.mean(axis=1), "rowtotal": block.sum(axis=1, min_count=0), "rowmax": block.max(axis=1),
                       "rowmin": block.min(axis=1), "rowmiss": block.isna().sum(axis=1)}[fn]
        return out
    if fn == "group":
        keys = src or by_cols
        g = out.groupby(keys, dropna=True, sort=True).ngroup()
        out[target] = (g + 1).where(g >= 0, np.nan)
        return out
    if fn == "tag":
        if not by_cols and not src:
            raise ValueError("egen tag: 묶음 변수 지정 필요")
        out[target] = (~out.duplicated(subset=by_cols or src)).astype(int)
        return out
    if len(src) != 1:
        raise ValueError(f"egen {fn}: 대상 변수 1개 지정")
    col = src[0]
    if fn == "rank":
        vals = pd.to_numeric(out[col], errors="coerce")
        out[target] = vals.groupby([out[c] for c in by_cols]).rank(method="average") if by_cols else vals.rank(method="average")
        return out
    f_ = _STAT_FUNCS[fn]
    if by_cols:
        out[target] = out.groupby(by_cols, dropna=False)[col].transform(f_)
    else:
        s = out[col]
        out[target] = s.agg(f_) if isinstance(f_, str) else f_(s)
    return out


# ---------------------------------------------------------------------------
# Variable type conversion (Stata destring / tostring / encode / decode)
# ---------------------------------------------------------------------------
CONVERT_TARGETS = ("numeric", "integer", "text", "date")


def _to_numeric_series(s: pd.Series, ignore: str = "", force: bool = False, percent: bool = False) -> Tuple[pd.Series, int]:
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return (s.astype(float) if not pd.api.types.is_integer_dtype(s) else s), 0
    txt = s.astype("string").str.strip()
    if ignore:
        txt = txt.str.replace("[" + re.escape(ignore) + "]", "", regex=True)
    pct = txt.str.endswith("%").fillna(False) if percent else None
    if percent:
        txt = txt.str.rstrip("%")
    txt = txt.replace({"": pd.NA, ".": pd.NA})
    num = pd.to_numeric(txt, errors="coerce").astype("Float64").astype(float)
    bad_mask = txt.notna() & num.isna()
    bad = list(pd.unique(s[bad_mask.fillna(False).to_numpy()]))[:5]
    if bad and not force:
        raise ValueError(f"'{s.name}'에 숫자로 바꿀 수 없는 값 {int(bad_mask.sum())}건 (예: {bad}) — ignore로 제거할 문자 지정 "
                         "또는 force=True로 결측 처리 (Stata destring: contains nonnumeric characters)")
    if percent and pct is not None:
        num = num.where(~pct.to_numpy(dtype=bool), num / 100.0)
    if num.notna().any() and np.all(np.mod(num.dropna().to_numpy(), 1) == 0) and not num.isna().any():
        num = num.astype("int64")
    return num, int(bad_mask.fillna(False).sum())


def convert_variables(df: pd.DataFrame, columns: Union[str, List[str]], to: str, ignore: str = "", force: bool = False,
                      percent: bool = False, generate: Optional[Union[str, List[str]]] = None,
                      date_format: Optional[str] = None, text_format: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Change the storage type of variables.

    to = "numeric" (Stata destring; `ignore` characters removed, `percent` divides "12%" by 100, unconvertible values
    raise unless `force`), "integer" (numeric, then non-integers raise unless `force` rounds them), "text"
    (Stata tostring; whole numbers without decimals, `text_format` like "{:.2f}"), "date" (pandas to_datetime with
    optional `date_format`). `generate`: new variable names (one per column); default replaces in place.
    """
    cols = _as_name_list(columns) if isinstance(columns, str) else [str(c) for c in columns]
    target = str(to).strip().lower()
    target = {"number": "numeric", "float": "numeric", "decimal": "numeric", "destring": "numeric", "int": "integer",
              "string": "text", "str": "text", "tostring": "text", "datetime": "date"}.get(target, target)
    if target not in CONVERT_TARGETS:
        raise ValueError(f"변환 형식 '{to}' 인식 불가 (가능: {', '.join(CONVERT_TARGETS)})")
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"자료에 없는 변수: {miss}")
    gens = _as_name_list(generate) if isinstance(generate, str) else ([str(g) for g in generate] if generate else [])
    if gens and len(gens) != len(cols):
        raise ValueError("generate 이름 수와 변환할 변수 수가 다름")
    out = df.copy()
    report: Dict[str, Any] = {"to": target, "forced_missing": {}}
    for k, c in enumerate(cols):
        s = out[c]
        n_forced = 0
        if target in ("numeric", "integer"):
            new, n_forced = _to_numeric_series(s, ignore, force, percent)
            if target == "integer":
                fin = new.dropna()
                if len(fin) and not np.all(np.mod(fin.to_numpy(dtype=float), 1) == 0):
                    if not force:
                        raise ValueError(f"'{c}'에 정수가 아닌 값 {int((np.mod(fin.to_numpy(dtype=float), 1) != 0).sum())}건 — force=True로 반올림")
                    x = new.to_numpy(dtype=float)
                    new = pd.Series(np.sign(x) * np.floor(np.abs(x) + 0.5), index=new.index)  # half away from zero (Stata round)
                if not new.isna().any():
                    new = new.astype("int64")
        elif target == "text":
            def _fmt(v):
                if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
                    return None
                if text_format:
                    return text_format.format(v)
                if isinstance(v, (float, np.floating)) and float(v).is_integer():
                    return str(int(v))
                return str(v)
            new = pd.Series([_fmt(v) for v in s.tolist()], index=s.index, dtype=object)
        else:
            new = pd.to_datetime(s, format=date_format, errors="coerce")
            bad_mask = s.notna() & new.isna()
            n_forced = int(bad_mask.sum())
            if bad_mask.any() and not force:
                raise ValueError(f"'{c}'에 날짜로 바꿀 수 없는 값 {int(bad_mask.sum())}건 (예: {list(s[bad_mask].unique()[:3])}) "
                                 "— date_format 지정 또는 force=True")
        report["forced_missing"][c] = n_forced
        dst = gens[k] if gens else c
        if gens and dst in out.columns:
            raise ValueError(f"변수 '{dst}'가 이미 있음")
        out[dst] = new
    return out, report


def encode_variable(df: pd.DataFrame, source: str, target: str, order: Optional[List[Any]] = None,
                    labels: Optional[Dict[Any, str]] = None) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Stata `encode`: text categories -> integer codes 1..k (alphabetical unless `order`); returns value labels.
    Existing `labels` ({code: text}) are reused so repeated encodes keep the same codes."""
    if source not in df.columns:
        raise ValueError(f"자료에 없는 변수: {source}")
    if target in df.columns:
        raise ValueError(f"변수 '{target}'가 이미 있음 (Stata encode는 새 변수 생성)")
    s = df[source]
    vals = [v for v in pd.unique(s.dropna())]
    text_vals = [str(v) for v in vals]
    lab = {int(k): str(v) for k, v in (labels or {}).items()}
    inv = {v: k for k, v in lab.items()}
    if order:
        seq = [str(v) for v in order]
        extra = [v for v in text_vals if v not in seq]
        if extra:
            raise ValueError(f"order에 없는 값: {extra[:5]}")
    else:
        seq = sorted(text_vals)
    nxt = max(lab) + 1 if lab else 1
    for v in seq:
        if v not in inv:
            lab[nxt] = v
            inv[v] = nxt
            nxt += 1
    out = df.copy()
    codes = s.map(lambda v: inv.get(str(v)) if pd.notna(v) else np.nan)
    out[target] = codes.astype("Int64").astype(float) if codes.isna().any() else codes.astype("int64")
    return out, dict(sorted(lab.items()))


def decode_variable(df: pd.DataFrame, source: str, target: str, labels: Optional[Dict[Any, str]]) -> pd.DataFrame:
    """Stata `decode`: integer codes -> label text using value labels (codes without a label become text of the code)."""
    if source not in df.columns:
        raise ValueError(f"자료에 없는 변수: {source}")
    if target in df.columns:
        raise ValueError(f"변수 '{target}'가 이미 있음")
    if not labels:
        raise ValueError(f"'{source}'에 값 라벨 없음 — set_value_labels 또는 encode 후 실행 (Stata decode: not labeled)")
    lab = {}
    for k, v in labels.items():
        try:
            lab[float(k)] = str(v)
        except (TypeError, ValueError):
            lab[k] = str(v)
    out = df.copy()

    def _dec(v):
        if pd.isna(v):
            return None
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return str(v)
        return lab.get(fv, _fmt_j(v))
    out[target] = pd.Series([_dec(v) for v in df[source].tolist()], index=df.index, dtype=object)
    return out


# ---------------------------------------------------------------------------
# Writers keeping value / variable labels (Stata .dta, SPSS .sav)
# ---------------------------------------------------------------------------
def _int_labels(df: pd.DataFrame, value_labels: Dict[str, Dict[Any, str]]) -> Dict[str, Dict[int, str]]:
    out: Dict[str, Dict[int, str]] = {}
    for col, labs in (value_labels or {}).items():
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        clean = {}
        for k, v in labs.items():
            try:
                fk = float(k)
            except (TypeError, ValueError):
                continue
            if fk.is_integer():
                clean[int(fk)] = str(v)
        if clean:
            out[col] = clean
    return out


def write_stata(df: pd.DataFrame, path: str, value_labels: Optional[Dict[str, Dict[Any, str]]] = None,
                variable_labels: Optional[Dict[str, str]] = None) -> str:
    """Write a Stata 14+ .dta (UTF-8 names) with value labels on integer-coded numeric variables."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].where(out[c].notna(), "").astype(str)
    vl = _int_labels(out, value_labels or {})
    for c in vl:
        if out[c].notna().all():
            out[c] = out[c].astype("int32")
    out.to_stata(path, write_index=False, version=118, value_labels=vl or None,
                 variable_labels={k: str(v)[:80] for k, v in (variable_labels or {}).items() if k in out.columns} or None)
    return path


def write_spss(df: pd.DataFrame, path: str, value_labels: Optional[Dict[str, Dict[Any, str]]] = None,
               variable_labels: Optional[Dict[str, str]] = None) -> str:
    """Write an SPSS .sav / .zsav with value and variable labels (pyreadstat)."""
    try:
        import pyreadstat
    except ImportError as e:  # pragma: no cover
        raise ValueError("SPSS 파일 저장에는 pyreadstat 필요: pip install pyreadstat") from e
    import warnings
    rename: Dict[str, str] = {}
    used: set = set()
    for c in df.columns:
        n = re.sub(r"[^\w@#$.]", "_", str(c))
        if not n or not (n[0].isalpha() or n[0] in "@#$"):
            n = "v" + n
        n = n.rstrip(".")[:64] or "v"
        base, k = n, 1
        while n.lower() in used:
            k += 1
            n = f"{base}_{k}"
        used.add(n.lower())
        if n != c:
            rename[c] = n
    out = df.rename(columns=rename)
    vl = _int_labels(out, {rename.get(k, k): v for k, v in (value_labels or {}).items()})
    labels = [str((variable_labels or {}).get(c, "")) or None for c in df.columns]
    if rename:
        warnings.warn(f"SPSS 변수명 규칙에 맞게 이름 변경: {rename}")
    pyreadstat.write_sav(out, path, column_labels=labels, variable_value_labels=vl or None,
                         compress=path.lower().endswith(".zsav"))
    return path
