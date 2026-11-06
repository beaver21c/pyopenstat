"""
OpenStat Studio Agentic SDK (`openstat.Session` / `pyjamovi.JamoviSession` v3.0)
License: MIT
High-level stateful session and agent-orchestration interface designed for Claude Code, Codex, Antigravity, and Python scripts.
Supports:
- All 28 statistical modules (including Panel FE/Two-Way FE/RE/Hausman & CFA/SEM)
- Real-time `.py` script generation & live workspace sync (`workspace/live_session.py` + `live_session.json`)
- Multi-format analyzed table export (`.xlsx`, `.docx`, `.hwpx`, `.tex`, `.csv`, `.tsv`, `.md`, `.html`, `.json`, `.omv`)
- Customizable `PlotStyle` design & multi-format vector/raster export (`.svg`, `.pdf`, `.png`)
"""

from typing import List, Optional, Union, Dict, Any, Tuple
import os
import json
import re
import urllib.request
import numpy as np
import pandas as pd

from .omv.reader import read_omv
from .omv.writer import write_omv
from .omv.dataset import JamoviDataset
from .registry import dispatch_analysis
from .renderers.html_renderer import render_jamovi_html_report

DATA_OP_METHODS = (
    "compute", "recode", "filter_rows", "merge", "joinby", "append", "reshape", "collapse", "egen",
    "convert_type", "encode", "decode", "set_value_labels", "set_measure", "standardize",
    "rename_var", "drop_vars", "set_value", "add_rows", "drop_rows",
)
MEASURE_LEVELS = ("Continuous", "Ordinal", "Nominal", "ID")
MERGE_RESULT_LABELS_REF = {1: "master only (1)", 2: "using only (2)", 3: "matched (3)", 4: "missing updated (4)", 5: "nonmissing conflict (5)"}


class JamoviSession:
    """
    Stateful statistical analysis session for Python scripts and AI coding agents (Claude Code, Codex, Antigravity).
    Accessible as both `openstat.Session` (`ost.Session`) and `pyjamovi.JamoviSession`.
    """

    def __init__(
        self,
        data_source: Optional[Union[str, pd.DataFrame]] = None,
        auto_sync_workspace: bool = True
    ):
        self.df: Optional[pd.DataFrame] = None
        self.dataset: Optional[JamoviDataset] = None
        self.source_path: Optional[str] = None
        self.history: List[Any] = []
        self.specs: List[Dict[str, Any]] = []
        # Stata `.do`-style pipeline log: data wrangling steps & export steps (replayed by `.py` scripts)
        self.data_ops: List[Dict[str, Any]] = []
        self.export_ops: List[Dict[str, Any]] = []
        self._raw_df: Optional[pd.DataFrame] = None
        self.auto_sync_workspace: bool = auto_sync_workspace
        # Variable metadata (Stata value labels / variable labels, jamovi measure levels)
        self.value_labels: Dict[str, Dict[Any, str]] = {}
        self.variable_labels: Dict[str, str] = {}
        self.measure_levels: Dict[str, str] = {}
        self._raw_meta: Dict[str, Any] = {"value_labels": {}, "variable_labels": {}}
        self.last_merge_report: Optional[Dict[str, Any]] = None

        # Automatic startup self-check guard (cached per Python + package version fingerprint)
        from .diagnostics import startup_self_check_guard
        self.health_status = startup_self_check_guard(force=False, check_pypi=False)

        if data_source is not None:
            self.load(data_source)

    def load(self, data_source: Union[str, pd.DataFrame]) -> "JamoviSession":
        if isinstance(data_source, pd.DataFrame):
            self.df = data_source.copy()
            self.dataset = JamoviDataset(self.df)
            self.source_path = None
        elif isinstance(data_source, str):
            from .data_ops import _resolve_data_path, read_dataset
            resolved = _resolve_data_path(data_source)
            self.source_path = resolved
            self.df, meta = read_dataset(resolved)
            self.dataset = JamoviDataset(self.df)
            self._raw_meta = {"value_labels": {k: dict(v) for k, v in meta.get("value_labels", {}).items()},
                              "variable_labels": dict(meta.get("variable_labels", {}))}
        else:
            raise TypeError("data_source must be a file path string or pandas DataFrame")
        if isinstance(data_source, pd.DataFrame):
            self._raw_meta = {"value_labels": {}, "variable_labels": {}}
        self._raw_df = self.df.copy() if self.df is not None else None
        self.data_ops = []
        self.export_ops = []
        self._reset_meta()
        return self

    def _reset_meta(self):
        self.value_labels = {k: dict(v) for k, v in self._raw_meta.get("value_labels", {}).items()}
        self.variable_labels = dict(self._raw_meta.get("variable_labels", {}))
        self.measure_levels = {}
        self.last_merge_report = None

    def _prune_meta(self):
        cols = set(self.df.columns) if self.df is not None else set()
        for d in (self.value_labels, self.variable_labels, self.measure_levels):
            for k in [k for k in d if k not in cols]:
                d.pop(k, None)

    # ------------------------------------------------------------------
    # Data management (Stata `generate` / `recode` / `keep if` / `merge` / `reshape` / `rename`)
    # Every call mutates `self.df`, is appended to `self.data_ops`, and is replayed by `to_python_script()`.
    # ------------------------------------------------------------------
    def _apply_data_op(self, new_df: pd.DataFrame, op: Dict[str, Any]) -> "JamoviSession":
        self.df = new_df.reset_index(drop=True)
        self.dataset = JamoviDataset(self.df)
        self.data_ops.append(op)
        self._prune_meta()
        self._sync_workspace_if_enabled()
        return self

    def _require_df(self):
        if self.df is None:
            raise ValueError("No data loaded in session")

    def compute(self, col_name: str, formula: Optional[str] = None) -> "JamoviSession":
        """Create/overwrite a variable from a formula (Stata `generate`/`replace`).
        Also accepts one Stata-style string: ``sess.compute("loglen = log(len)")``."""
        from .data_ops import compute_variable
        if formula is None:
            if "=" not in str(col_name):
                raise ValueError("compute() needs a formula: compute('new', 'expr') or compute('new = expr')")
            col_name, formula = (x.strip() for x in str(col_name).split("=", 1))
        self._require_df()
        return self._apply_data_op(
            compute_variable(self.df, col_name, formula),
            {"op": "compute", "col_name": col_name, "formula": formula},
        )

    def recode(self, source_col: str, target_col: Optional[str] = None, rules: Any = "", else_value: Any = "copy") -> "JamoviSession":
        """Recode values / ranges into a new variable (Stata `recode ..., gen()`)."""
        from .data_ops import recode_variable
        self._require_df()
        return self._apply_data_op(
            recode_variable(self.df, source_col, target_col, rules, else_value),
            {"op": "recode", "source_col": source_col, "target_col": target_col, "rules": rules, "else_value": else_value},
        )

    def filter_rows(self, condition: str) -> "JamoviSession":
        """Keep rows satisfying a boolean condition (Stata `keep if`)."""
        from .data_ops import filter_dataset
        self._require_df()
        return self._apply_data_op(filter_dataset(self.df, condition), {"op": "filter_rows", "condition": condition})

    def filter(self, condition: str) -> "JamoviSession":
        """Alias for filter_rows (Stata `keep if`)."""
        return self.filter_rows(condition)

    def standardize(self, columns: Union[str, List[str]], method: str = "zscore") -> "JamoviSession":
        """Standardize numeric columns (z-score, min-max, or robust)."""
        self._require_df()
        cols = [columns] if isinstance(columns, str) else list(columns)
        new_df = self.df.copy()
        for c in cols:
            if c not in new_df.columns:
                raise ValueError(f"Column '{c}' not found")
            s = pd.to_numeric(new_df[c], errors="coerce")
            if method == "zscore":
                mean = s.mean()
                std = s.std(ddof=1)
                new_df[f"{c}_z"] = (s - mean) / std if std > 0 else 0.0
            elif method == "minmax":
                s_min, s_max = s.min(), s.max()
                diff = s_max - s_min
                new_df[f"{c}_norm"] = (s - s_min) / diff if diff > 0 else 0.0
            elif method == "robust":
                median = s.median()
                iqr = s.quantile(0.75) - s.quantile(0.25)
                new_df[f"{c}_rob"] = (s - median) / iqr if iqr > 0 else 0.0
            else:
                raise ValueError(f"Unknown standardization method '{method}'. Choose 'zscore', 'minmax', or 'robust'.")
        op = {"op": "standardize", "columns": cols, "method": method}
        return self._apply_data_op(new_df, op)

    @staticmethod
    def _op(name: str, **kw) -> Dict[str, Any]:
        """Data-op record without arguments left at their defaults (keeps the replay script short)."""
        op: Dict[str, Any] = {"op": name}
        for k, v in kw.items():
            if v is None or v is False or v == "" or v == [] or v == {}:
                continue
            op[k] = list(v) if isinstance(v, tuple) else v
        return op

    def _other_frames(self, other_source):
        from .data_ops import load_tabular_source
        if isinstance(other_source, (list, tuple)):
            frames, paths = [], []
            for src in other_source:
                d, p = load_tabular_source(src)
                frames.append(d)
                paths.append(p or "<in-memory DataFrame>")
            return frames, paths
        d, p = load_tabular_source(other_source)
        return d, (p or "<in-memory DataFrame>")

    def merge(
        self,
        other_source: Union[str, pd.DataFrame],
        on: Optional[Union[str, List[str]]] = None,
        kind: Optional[str] = None,
        keep: Optional[Union[str, List[str]]] = None,
        keepusing: Optional[Union[str, List[str]]] = None,
        indicator: Optional[Union[str, bool]] = None,
        update: bool = False,
        replace: bool = False,
        assert_results: Optional[Union[str, List[str]]] = None,
        how: Optional[str] = None,
        suffixes: Tuple[str, str] = ("", "_merged"),
    ) -> "JamoviSession":
        """Merge another dataset by key variables (Stata `merge 1:1 / m:1 / 1:m / m:m`).

        `kind` is inferred from key uniqueness when omitted; duplicated keys on both sides raise an error.
        Default keeps master, using and matched rows and adds `_merge` (1 master only, 2 using only, 3 matched,
        4 missing updated, 5 nonmissing conflict). Variables in both files keep master values unless `update`
        (fill missing) / `replace` (overwrite). `keep=["match", "master"]`, `keepusing=[...]`, `indicator=False`
        (Stata nogenerate), `assert_results=["match"]`. Legacy `how="left"|"inner"|"right"|"outer"` keeps the
        pandas-style suffix behaviour; `how="append"` stacks rows.
        """
        from .data_ops import merge_datasets
        self._require_df()
        right_df, right_path = self._other_frames(other_source)
        new_df, report = merge_datasets(self.df, right_df, on=on, how=how, suffixes=tuple(suffixes), kind=kind, keep=keep,
                                        keepusing=keepusing, indicator=indicator, update=update, replace=replace,
                                        assert_results=assert_results, return_report=True)
        op = self._op("merge", other_source=right_path, on=on, kind=kind, keep=keep, keepusing=keepusing,
                      indicator=indicator if indicator is not None else None, update=update, replace=replace,
                      assert_results=assert_results, how=how)
        if indicator is False:
            op["indicator"] = False
        if tuple(suffixes) != ("", "_merged"):
            op["suffixes"] = list(suffixes)
        self._apply_data_op(new_df, op)
        if report.get("indicator"):
            self.value_labels[report["indicator"]] = dict(MERGE_RESULT_LABELS_REF)
        self.last_merge_report = report
        return self

    def joinby(self, other_source: Union[str, pd.DataFrame], on: Optional[Union[str, List[str]]] = None,
               unmatched: str = "none", indicator: str = "_merge", update: bool = False, replace: bool = False) -> "JamoviSession":
        """All pairwise combinations within key groups (Stata `joinby`); `unmatched`: none | both | master | using."""
        from .data_ops import joinby_datasets
        self._require_df()
        right_df, right_path = self._other_frames(other_source)
        new_df, report = joinby_datasets(self.df, right_df, on=on, unmatched=unmatched, indicator=indicator,
                                         update=update, replace=replace, return_report=True)
        op = self._op("joinby", other_source=right_path, on=on, unmatched=None if unmatched == "none" else unmatched,
                      indicator=None if indicator == "_merge" else indicator, update=update, replace=replace)
        self._apply_data_op(new_df, op)
        self.last_merge_report = report
        return self

    def append(self, other_source: Union[str, pd.DataFrame, List[Union[str, pd.DataFrame]]], generate: Optional[str] = None,
               force: bool = False) -> "JamoviSession":
        """Stack the rows of one or more datasets (Stata `append using`); `generate` marks the source (0 = current)."""
        from .data_ops import append_datasets
        self._require_df()
        frames, paths = self._other_frames(other_source)
        new_df = append_datasets(self.df, frames, generate=generate, force=force)
        return self._apply_data_op(new_df, self._op("append", other_source=paths, generate=generate, force=force))

    def reshape(
        self,
        direction: str = "long",
        id_vars: Optional[Union[str, List[str]]] = None,
        value_vars: Optional[Union[str, List[str]]] = None,
        stubnames: Optional[Union[str, List[str]]] = None,
        time_col: str = "time",
        value_col: str = "value",
        sep: str = "",
        strip_prefix: bool = True,
    ) -> "JamoviSession":
        """Reshape Wide <-> Long (Stata `reshape long stubs, i() j()` / `reshape wide stubs, i() j()`)."""
        from .data_ops import reshape_dataset
        self._require_df()
        new_df = reshape_dataset(
            self.df, direction=direction, id_vars=id_vars, value_vars=value_vars, stubnames=stubnames,
            time_col=time_col, value_col=value_col, sep=sep, strip_prefix=strip_prefix,
        )
        op = {
            "op": "reshape", "direction": direction, "id_vars": id_vars, "value_vars": value_vars,
            "stubnames": stubnames, "time_col": time_col, "value_col": value_col,
        }
        if sep:
            op["sep"] = sep
        if not strip_prefix:
            op["strip_prefix"] = False
        return self._apply_data_op(new_df, op)

    def collapse(self, stats: Dict[str, Any], by: Optional[Union[str, List[str]]] = None) -> "JamoviSession":
        """Aggregate to one row per group (Stata `collapse (mean) y (sum) z, by(id)`).
        stats: {"mean": ["y"], "sum": ["z"]}, {"y": "mean"} or {"new": ("y", "max")}."""
        from .data_ops import collapse_dataset
        self._require_df()
        clean = {k: (list(v) if isinstance(v, tuple) else v) for k, v in stats.items()}
        return self._apply_data_op(collapse_dataset(self.df, stats, by), self._op("collapse", stats=clean, by=by))

    def egen(self, target: str, func: str, source: Optional[Union[str, List[str]]] = None,
             by: Optional[Union[str, List[str]]] = None) -> "JamoviSession":
        """Group / row statistics as a new variable (Stata `egen target = func(source), by()`)."""
        from .data_ops import egen_variable
        self._require_df()
        return self._apply_data_op(egen_variable(self.df, target, func, source, by),
                                   self._op("egen", target=target, func=func, source=source, by=by))

    # ---- variable types, labels and measure levels ------------------------------------------
    def convert_type(self, columns: Union[str, List[str]], to: str, ignore: str = "", force: bool = False,
                     percent: bool = False, generate: Optional[Union[str, List[str]]] = None,
                     date_format: Optional[str] = None, text_format: Optional[str] = None) -> "JamoviSession":
        """Change the storage type: to = "numeric" | "integer" | "text" | "date" (Stata destring / tostring).
        Values that cannot be converted raise an error unless `force=True` (then they become missing)."""
        from .data_ops import convert_variables
        self._require_df()
        new_df, report = convert_variables(self.df, columns, to, ignore=ignore, force=force, percent=percent,
                                           generate=generate, date_format=date_format, text_format=text_format)
        self.last_convert_report = report
        return self._apply_data_op(new_df, self._op("convert_type", columns=columns, to=to, ignore=ignore, force=force,
                                                    percent=percent, generate=generate, date_format=date_format,
                                                    text_format=text_format))

    def destring(self, columns: Union[str, List[str]], ignore: str = "", force: bool = False, percent: bool = False,
                 generate: Optional[Union[str, List[str]]] = None) -> "JamoviSession":
        """Text -> numeric (Stata `destring`); recorded as convert_type(to="numeric")."""
        return self.convert_type(columns, "numeric", ignore=ignore, force=force, percent=percent, generate=generate)

    def tostring(self, columns: Union[str, List[str]], generate: Optional[Union[str, List[str]]] = None,
                 text_format: Optional[str] = None) -> "JamoviSession":
        """Numeric -> text (Stata `tostring`); recorded as convert_type(to="text")."""
        return self.convert_type(columns, "text", generate=generate, text_format=text_format)

    def encode(self, source: str, target: str, order: Optional[List[Any]] = None) -> "JamoviSession":
        """Text categories -> integer codes with value labels (Stata `encode source, generate(target)`);
        codes follow alphabetical order unless `order` lists the categories."""
        from .data_ops import encode_variable
        self._require_df()
        new_df, labels = encode_variable(self.df, source, target, order=order)
        self._apply_data_op(new_df, self._op("encode", source=source, target=target, order=order))
        self.value_labels[target] = labels
        self.measure_levels.setdefault(target, "Ordinal" if order else "Nominal")
        return self

    def decode(self, source: str, target: str) -> "JamoviSession":
        """Integer codes -> label text using the variable's value labels (Stata `decode source, generate(target)`)."""
        from .data_ops import decode_variable
        self._require_df()
        return self._apply_data_op(decode_variable(self.df, source, target, self.value_labels.get(source)),
                                   self._op("decode", source=source, target=target))

    def set_value_labels(self, column: str, labels: Optional[Dict[Any, str]]) -> "JamoviSession":
        """Attach value labels {code: text} to a variable (Stata `label define` + `label values`); None removes them."""
        self._require_df()
        if column not in self.df.columns:
            raise ValueError(f"Variable '{column}' not found")
        clean: Dict[Any, str] = {}
        for k, v in (labels or {}).items():
            try:
                fk = float(k)
                clean[int(fk) if fk.is_integer() else fk] = str(v)
            except (TypeError, ValueError):
                clean[k] = str(v)
        op = self._op("set_value_labels", column=column, labels=clean if labels else None)
        if not clean:
            op["labels"] = None
        self._apply_data_op(self.df, op)
        if clean:
            self.value_labels[column] = clean
        else:
            self.value_labels.pop(column, None)
        return self

    def set_measure(self, columns: Union[str, List[str]], level: str) -> "JamoviSession":
        """Set the measure level used by the GUI and analyses: Continuous | Ordinal | Nominal | ID."""
        from .analyses.base import resolve_choice
        self._require_df()
        lvl = resolve_choice("측정 수준", level, list(MEASURE_LEVELS),
                             {"scale": "Continuous", "continuous": "Continuous", "interval": "Continuous", "ratio": "Continuous",
                              "ordinal": "Ordinal", "nominal": "Nominal", "categorical": "Nominal", "id": "ID"})
        cols = [columns] if isinstance(columns, str) else list(columns)
        miss = [c for c in cols if c not in self.df.columns]
        if miss:
            raise ValueError(f"Variable not found: {miss}")
        self._apply_data_op(self.df, self._op("set_measure", columns=columns, level=lvl))
        for c in cols:
            self.measure_levels[c] = lvl
        return self

    def rename_var(self, old_name: str, new_name: str) -> "JamoviSession":
        """Rename a variable (Stata `rename`)."""
        self._require_df()
        if old_name not in self.df.columns:
            raise ValueError(f"Variable '{old_name}' not found")
        if new_name in self.df.columns and new_name != old_name:
            raise ValueError(f"Variable '{new_name}' already exists")
        carried = [(d, d.pop(old_name)) for d in (self.value_labels, self.variable_labels, self.measure_levels) if old_name in d]
        self._apply_data_op(self.df.rename(columns={old_name: new_name}), {"op": "rename_var", "old_name": old_name, "new_name": new_name})
        for d, v in carried:
            d[new_name] = v
        return self

    def drop_vars(self, cols: Union[str, List[str]]) -> "JamoviSession":
        """Drop variables (Stata `drop varlist`)."""
        self._require_df()
        cols_l = [cols] if isinstance(cols, str) else list(cols)
        return self._apply_data_op(self.df.drop(columns=[c for c in cols_l if c in self.df.columns]), {"op": "drop_vars", "cols": cols_l})

    def set_value(self, row_idx: int, col_name: str, value: Any) -> "JamoviSession":
        """Edit one cell (Stata `replace var = value in row`). Rows beyond the end are appended."""
        self._require_df()
        if col_name not in self.df.columns:
            raise ValueError(f"Variable '{col_name}' not found")
        out = self.df.copy()
        while row_idx >= len(out):
            out.loc[len(out)] = [np.nan] * len(out.columns)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            out.at[row_idx, col_name] = np.nan
        else:
            num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if not pd.isna(num) and (pd.api.types.is_numeric_dtype(out[col_name]) or out[col_name].dropna().empty):
                out[col_name] = pd.to_numeric(out[col_name], errors="coerce")
                out.at[row_idx, col_name] = float(num)
            else:
                out[col_name] = out[col_name].astype(object)
                out.at[row_idx, col_name] = str(value)
        return self._apply_data_op(out, {"op": "set_value", "row_idx": int(row_idx), "col_name": col_name, "value": value})

    def add_rows(self, n: int = 1) -> "JamoviSession":
        """Append `n` empty rows (Stata `set obs`)."""
        self._require_df()
        blank = pd.DataFrame({c: [np.nan] * int(n) for c in self.df.columns})
        return self._apply_data_op(pd.concat([self.df, blank], ignore_index=True), {"op": "add_rows", "n": int(n)})

    def drop_rows(self, rows: Union[int, List[int]]) -> "JamoviSession":
        """Drop rows by 0-based position (Stata `drop in`)."""
        self._require_df()
        idx = [rows] if isinstance(rows, int) else [int(r) for r in rows]
        keep = [i for i in range(len(self.df)) if i not in set(idx)]
        return self._apply_data_op(self.df.iloc[keep], {"op": "drop_rows", "rows": idx})

    def apply_data_op(self, op: Dict[str, Any]) -> "JamoviSession":
        """Apply one recorded data-op dict (used by replay / undo / GUI)."""
        args = {k: v for k, v in op.items() if k != "op"}
        name = op.get("op")
        if name not in DATA_OP_METHODS:
            raise ValueError(f"Unknown data op '{name}'")
        if name == "merge" and "suffixes" in args:
            args["suffixes"] = tuple(args["suffixes"])
        return getattr(self, name)(**args)

    def undo_data_op(self) -> "JamoviSession":
        """Undo the last data operation by reloading raw data and replaying all previous steps."""
        if not self.data_ops:
            return self
        return self._replay_data_ops(self.data_ops[:-1])

    def clear_filters(self) -> "JamoviSession":
        """Remove every row filter (`keep if`) and rebuild the data from the raw snapshot with the other steps."""
        remaining = [op for op in self.data_ops if op.get("op") != "filter_rows"]
        if len(remaining) == len(self.data_ops):
            return self
        return self._replay_data_ops(remaining)

    def _replay_data_ops(self, remaining: List[Dict[str, Any]]) -> "JamoviSession":
        raw = self._raw_df.copy() if self._raw_df is not None else None
        if raw is None:
            raise ValueError("Raw data snapshot unavailable for undo")
        keep_path, keep_exports = self.source_path, list(self.export_ops)
        auto = self.auto_sync_workspace
        self.auto_sync_workspace = False
        self.df, self.dataset, self.data_ops = raw, JamoviDataset(raw), []
        self._reset_meta()
        for op in remaining:
            self.apply_data_op(op)
        self.source_path, self.export_ops = keep_path, keep_exports
        self.auto_sync_workspace = auto
        self._sync_workspace_if_enabled()
        return self

    def save_data(self, output_path: str) -> str:
        """Save the current (transformed) dataset to .csv / .xlsx / .omv / .dta / .sav / .parquet (Stata `save`)."""
        self._require_df()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".omv":
            write_omv(self.df, output_path)
        elif ext in (".xlsx", ".xls"):
            self.df.to_excel(output_path, index=False)
        elif ext == ".dta":
            from .data_ops import write_stata
            write_stata(self.df, output_path, self.value_labels, self.variable_labels)
        elif ext in (".sav", ".zsav"):
            from .data_ops import write_spss
            write_spss(self.df, output_path, self.value_labels, self.variable_labels)
        elif ext == ".parquet":
            self.df.to_parquet(output_path, index=False)
        else:
            self.df.to_csv(output_path, index=False, encoding="utf-8-sig")
        self._record_export({"op": "save_data", "output_path": output_path})
        return output_path

    def save_dataset(self, output_path: str) -> str:
        """Alias for save_data."""
        return self.save_data(output_path)

    def pipeline_log(self) -> List[Dict[str, Any]]:
        """Return the full ordered pipeline (data ops -> analyses -> exports)."""
        return (
            [dict(o, stage="data") for o in self.data_ops]
            + [{"stage": "analysis", "analysis_type": s["analysis_type"], "title": s.get("title")} for s in self.specs]
            + [dict(o, stage="export") for o in self.export_ops]
        )

    def _record_export(self, op: Dict[str, Any]):
        if op not in self.export_ops:
            self.export_ops.append(op)
        self._sync_workspace_if_enabled()

    def inspect_for_llm(self) -> str:
        if self.df is None:
            return "No dataset currently loaded in session."

        lines = [
            f"### Dataset Overview ({len(self.df)} rows, {len(self.df.columns)} columns)",
            f"Source: {self.source_path or 'In-Memory DataFrame'}\n"
        ]

        for col in self.df.columns:
            s = self.df[col]
            missing = int(s.isna().sum())
            n_unique = s.nunique()
            if pd.api.types.is_numeric_dtype(s) and n_unique > 5:
                summary = f"Continuous (Min: {s.min():.2f}, Mean: {s.mean():.2f}, SD: {s.std(ddof=1):.2f}, Max: {s.max():.2f})"
            elif pd.api.types.is_numeric_dtype(s):
                top_vals = sorted(list(s.dropna().unique()[:6]))
                summary = f"Ordinal/Discrete ({n_unique} levels: {top_vals})"
            else:
                top_vals = list(s.dropna().unique()[:5])
                summary = f"Nominal ({n_unique} levels: {top_vals})"
            lines.append(f"- **{col}**: {summary} | Missing: {missing}")

        lines.append("\n**Recommended OpenStat Studio Analyses (`sess.run(...)`):**")
        lines.append("- Group comparison (2 groups): `sess.ttest(dep=..., group=...)` / `sess.bayes_ttest(dep=..., group=...)`")
        lines.append("- Multi-group / Factorial / RM: `sess.anova(...)`, `sess.factorial_anova(...)`, `sess.rm_anova(...)`, `sess.kruskal(...)`")
        lines.append("- Regression & Logistic: `sess.regression(dep=..., covs=[...])`, `sess.logistic(dep=..., covs=[...])`")
        lines.append("- Psychometrics & SEM: `sess.reliability(vars=[...])`, `sess.efa(vars=[...])`, `sess.sem(model_syntax=...)`")
        lines.append("- Mediation / Moderation: `sess.mediation(dep=..., pred=..., med=...)`, `sess.moderation(dep=..., pred=..., mod=...)`")
        lines.append("- Panel / Multilevel / Survival: `sess.panel(...)`, `sess.mixed_model(...)`, `sess.survival(...)`")
        lines.append("- Standalone Graphs (15 types -> Results): `sess.graph(chart_type='raincloud', x=..., y=..., group=...)` (`bar`, `stacked_bar`, `line`, `area`, `pie`, `donut`, `histogram`, `density`, `box`, `violin`, `raincloud`, `scatter`, `pairplot`, `heatmap`, `errorbar`)")
        lines.append("- Send Custom Figure to Results: `sess.add_figure(fig, title='...', attach_to_last=False)`")
        lines.append("- Export Results (Tables + Embedded Graphs): `sess.export_results('out.pdf')` (`.pdf`, `.hwpx`, `.docx`, `.xlsx`, `.tex`, `.csv`, `.md`, `.html`, `.omv`)")
        lines.append("- Customize plots: pass `plotStyle={'palette': 'nature', 'theme': 'apa_classic', 'title': '...', 'dpi': 300}`")
        return "\n".join(lines)

    def _sync_workspace_if_enabled(self):
        if not self.auto_sync_workspace:
            return
        try:
            from .syntax_engine import save_workspace_sync
            ds_name = os.path.basename(self.source_path) if self.source_path else "Dataset.csv"
            save_workspace_sync(
                dataset_name=ds_name,
                specs=self.specs,
                analyses=self.history,
                py_code=self.to_python_script(),
                data_ops=self.data_ops,
                export_ops=self.export_ops,
            )
        except Exception:
            pass

    def run(self, analysis_type: str, **params) -> Any:
        if self.df is None:
            raise ValueError("No data loaded in session")
        res = dispatch_analysis(analysis_type, self.df, params)
        self.history.append(res)
        self.specs.append({
            "analysis_type": analysis_type,
            "title": res.title,
            "params": params
        })
        self._sync_workspace_if_enabled()
        return res

    # Fluent convenience methods
    def descriptives(self, vars: Union[str, List[str]], split_by: Optional[str] = None, **kwargs):
        return self.run("descriptives", vars=vars, splitBy=split_by, **kwargs)

    def scatr(self, x: str, y: Optional[str] = None, group: Optional[str] = None, **kwargs):
        return self.run("scatr", x=x, y=y, group=group, **kwargs)

    def graph(
        self,
        chart_type: Optional[str] = None,
        x: Optional[str] = None,
        y: Optional[str] = None,
        vars: Optional[List[str]] = None,
        group: Optional[str] = None,
        chartType: Optional[str] = None,
        **kwargs,
    ):
        """Draw any of 15 standalone chart types and append directly to the Analysis Results Sheet."""
        ctype = chart_type or chartType or kwargs.pop("chartType", "bar")
        return self.run("graph", chartType=ctype, x=x, y=y, vars=vars, group=group, **kwargs)

    def add_figure(
        self,
        fig_or_src: Any,
        title: str = "Custom Visualization",
        caption: str = "",
        summary_df: Optional[pd.DataFrame] = None,
        attach_to_last: bool = False,
        plot_style: Optional[Dict[str, Any]] = None,
    ):
        """Send a custom Matplotlib Figure (or PNG data URI) into the Analysis Results Sheet (as a card or attached to the latest analysis)."""
        from .analyses.graph_builder import CustomFigureResult
        if attach_to_last and self.history:
            target = self.history[-1]
            target.attach_figure(fig_or_src, title=title, plot_style=plot_style)
            self._sync_workspace_if_enabled()
            return target
        card = CustomFigureResult(
            fig_or_src=fig_or_src,
            title=title,
            caption=caption,
            summary_df=summary_df,
            dataset=self.df,
            plot_style=plot_style,
        )
        self.history.append(card)
        self.specs.append({
            "analysis_type": "customFigure",
            "title": card.title,
            "params": {"title": title, "caption": caption, "plotStyle": plot_style or {}},
        })
        self._sync_workspace_if_enabled()
        return card

    def ttest(self, dep: Optional[Union[str, List[str]]] = None, group: str = "", welchs: bool = False, vars: Optional[Union[str, List[str]]] = None, **kwargs):
        target = dep if dep is not None else (vars if vars is not None else kwargs.pop("vars", None))
        return self.run("ttestIS", vars=target, group=group, welchs=welchs, **kwargs)

    def paired_ttest(self, pairs: Optional[Any] = None, var1: Optional[str] = None, var2: Optional[str] = None, **kwargs):
        if pairs and isinstance(pairs, list):
            pairs = [tuple(p) if isinstance(p, (list, tuple)) else p for p in pairs]
        return self.run("ttestPS", pairs=pairs, var1=var1, var2=var2, **kwargs)

    def onesample_ttest(self, vars: Optional[Union[str, List[str]]] = None, test_value: Optional[float] = None, testValue: Optional[float] = None, **kwargs):
        tv = test_value if test_value is not None else (testValue if testValue is not None else kwargs.pop("testValue", 0.0))
        return self.run("ttestOneS", vars=vars, testValue=tv, **kwargs)

    def bayes_ttest(self, dep: Optional[Union[str, List[str]]] = None, group: str = "", vars: Optional[Union[str, List[str]]] = None, **kwargs):
        target = dep if dep is not None else (vars if vars is not None else kwargs.pop("vars", None))
        return self.run("bayesTTestIS", vars=target, group=group, **kwargs)

    def summary_ttest(self, m1: float, s1: float, n1: int, m2: float, s2: float, n2: int, **kwargs):
        """JASP-style summary statistics independent samples t-test without raw microdata."""
        return self.run("summaryTTest", m1=m1, s1=s1, n1=n1, m2=m2, s2=s2, n2=n2, **kwargs)

    def anova(self, dep: Optional[Union[str, List[str]]] = None, group: str = "", welchs: bool = True, deps: Optional[Union[str, List[str]]] = None, **kwargs):
        target = dep if dep is not None else (deps if deps is not None else kwargs.pop("deps", None))
        factors = kwargs.pop("factors", None)
        if factors and not group:
            factors = [factors] if isinstance(factors, str) else list(factors)
            if len(factors) > 1:
                # several factors: factorial ANOVA on a single dependent variable
                dv = target[0] if isinstance(target, (list, tuple)) else target
                return self.run("anovaFactorial", dep=dv, factors=factors, **kwargs)
            group = factors[0]
        return self.run("anovaOneW", deps=target, group=group, welchs=welchs, **kwargs)

    def anova_oneway(self, deps: Optional[Union[str, List[str]]] = None, dep: Optional[Union[str, List[str]]] = None, group: str = "", welchs: bool = True, **kwargs):
        target = deps if deps is not None else dep
        return self.run("anovaOneW", deps=target, group=group, welchs=welchs, **kwargs)

    def factorial_anova(self, dep: str, factors: Union[str, List[str]], **kwargs):
        return self.run("anovaFactorial", dep=dep, factors=factors, **kwargs)

    def ancova(self, dep: str, factors: Union[str, List[str]], covs: Union[str, List[str]], **kwargs):
        return self.run("ancova", dep=dep, factors=factors, covs=covs, **kwargs)

    def rm_anova(self, rm_vars: Optional[List[str]] = None, between: Optional[str] = None, rmVars: Optional[List[str]] = None, **kwargs):
        target = rm_vars if rm_vars is not None else (rmVars if rmVars is not None else kwargs.pop("rmVars", None))
        return self.run("anovaRM", rmVars=target, between=between, **kwargs)

    def kruskal(self, dep: Optional[Union[str, List[str]]] = None, group: str = "", deps: Optional[Union[str, List[str]]] = None, **kwargs):
        target = dep if dep is not None else (deps if deps is not None else kwargs.pop("deps", None))
        return self.run("kruskal", deps=target, group=group, **kwargs)

    def bayes_anova(self, dep: str, factors: Union[str, List[str]], **kwargs):
        return self.run("bayesANOVA", dep=dep, factors=factors, **kwargs)

    def corrmatrix(self, vars: List[str], **kwargs):
        return self.run("corrMatrix", vars=vars, **kwargs)

    correlation = corrmatrix
    corr_matrix = corrmatrix

    def partial_corr(self, vars: List[str], controls: Union[str, List[str]], **kwargs):
        return self.run("partCorr", vars=vars, controls=controls, **kwargs)

    def bayes_corr(self, vars: List[str], **kwargs):
        return self.run("bayesCorr", vars=vars, **kwargs)

    def regression(self, dep: str, covs: List[str], **kwargs):
        return self.run("linReg", dep=dep, covs=covs, **kwargs)

    linreg = regression
    lin_reg = regression

    def logistic(self, dep: str, covs: List[str], **kwargs):
        return self.run("logRegBin", dep=dep, covs=covs, **kwargs)

    logreg = logistic
    log_reg = logistic

    def crosstab(self, rows: str, cols: str, **kwargs):
        return self.run("contTables", rows=rows, cols=cols, **kwargs)

    conttables = crosstab
    cont_tables = crosstab

    def reliability(self, vars: List[str], **kwargs):
        return self.run("reliability", vars=vars, **kwargs)

    def pca(self, vars: List[str], **kwargs):
        return self.run("pca", vars=vars, **kwargs)

    def efa(self, vars: List[str], **kwargs):
        return self.run("efa", vars=vars, **kwargs)

    def network_analysis(self, vars: List[str], method: str = "ebicglasso", gamma: float = 0.25, **kwargs):
        """JASP-style Network Analysis (EBICglasso partial correlations & centrality measures)."""
        return self.run("networkAnalysis", vars=vars, method=method, gamma=gamma, **kwargs)

    network = network_analysis

    # --- Text mining & relational networks -------------------------------------------------
    def load_texts(self, source: Union[str, List[str]], recursive: bool = True) -> "JamoviSession":
        """Replace the dataset with text files (folder or file list; .txt .md .docx .pdf .hwpx), one row per document."""
        from .text_import import load_texts as _load_texts
        return self.load(_load_texts(source, recursive=recursive))

    @classmethod
    def from_texts(cls, source: Union[str, List[str]], recursive: bool = True, **kwargs) -> "JamoviSession":
        from .text_import import load_texts as _load_texts
        return cls(_load_texts(source, recursive=recursive), **kwargs)

    def text_keywords(self, text: str, group: Optional[str] = None, **kwargs):
        """Keyword frequency / TF-IDF tables, group keywords (c-TF-IDF) and word clouds (kiwipiepy tokenization)."""
        return self.run("textKeywords", text=text, group=group, **kwargs)

    def text_network(self, text: str, unit: str = "sentence", **kwargs):
        """Keyword co-occurrence network with centrality and Louvain communities."""
        return self.run("textNetwork", text=text, unit=unit, **kwargs)

    def text_cluster(self, text: str, embedding: str = "tfidf", method: str = "hdbscan", **kwargs):
        """Document embedding (TF-IDF+LSA or sentence-transformers) + UMAP + clustering + c-TF-IDF keywords."""
        return self.run("textCluster", text=text, embedding=embedding, method=method, **kwargs)

    def graph_network(self, source: str, target: str, weight: Optional[str] = None, directed: bool = False, **kwargs):
        """Relational network analysis from an edge list (centrality, communities, network map)."""
        return self.run("graphNetwork", source=source, target=target, weight=weight, directed=directed, **kwargs)

    def synthetic_data(self, n_samples: Optional[int] = None, cols: Optional[List[str]] = None, epsilon: Optional[float] = None, **kwargs) -> pd.DataFrame:
        """Generates privacy-preserving synthetic microdata preserving multivariate correlations and empirical marginals."""
        from .synthetic_data import generate_synthetic_dataset
        self._require_df()
        return generate_synthetic_dataset(self.df, n_samples=n_samples, cols=cols, epsilon=epsilon, **kwargs)

    def sem(self, model_syntax: Optional[str] = None, model: Optional[str] = None, **kwargs):
        ms = model_syntax or model or kwargs.pop("model", "")
        return self.run("sem", model=ms, **kwargs)

    def lca(self, categorical: Optional[List[str]] = None, continuous: Optional[List[str]] = None, classes: int = 2, **kwargs):
        """Latent class / latent profile analysis with Mplus defaults (MLR, STARTS 20 4, FIML).
        categorical = Mplus CATEGORICAL indicators, continuous = LPA indicators; classRange=[1, 5] compares K."""
        return self.run("lca", categorical=categorical, continuous=continuous, classes=classes, **kwargs)

    def lpa(self, continuous: Optional[List[str]] = None, classes: int = 2, **kwargs):
        """Latent profile analysis (continuous indicators; Mplus default: class-invariant variances, zero covariances)."""
        return self.run("lca", continuous=continuous, classes=classes, **kwargs)

    def lta(self, categorical: Optional[List[str]] = None, continuous: Optional[List[str]] = None, classes: int = 2, **kwargs):
        """Latent transition analysis (long data: idVar + timeVar, or wide waves=[[...], [...]])."""
        return self.run("lta", categorical=categorical, continuous=continuous, classes=classes, **kwargs)

    def lcga(self, dep: Optional[str] = None, classes: int = 2, **kwargs):
        """Latent class growth analysis (long data: dep + idVar + timeVar, or wide vars=[...])."""
        return self.run("lcga", dep=dep, classes=classes, **kwargs)

    def mediation(self, dep: str, pred: str, med: str, **kwargs):
        return self.run("mediation", dep=dep, pred=pred, med=med, **kwargs)

    def moderation(self, dep: str, pred: str, mod: str, **kwargs):
        return self.run("moderation", dep=dep, pred=pred, mod=mod, **kwargs)

    def panel(self, entity_col: Optional[str] = None, time_col: Optional[str] = None, dep: str = "", covs: Optional[List[str]] = None, model_type: Optional[str] = None, modelType: Optional[str] = None, entity: Optional[str] = None, time: Optional[str] = None, **kwargs):
        ent = entity_col or entity or kwargs.pop("entity", "")
        tm = time_col or time or kwargs.pop("time", "")
        mtype = model_type or modelType or kwargs.pop("modelType", "fixed")
        return self.run("panel", entity=ent, time=tm, dep=dep, covs=covs or [], modelType=mtype, **kwargs)

    def mixed_model(self, dep: str, cluster: str, covs: Optional[List[str]] = None, **kwargs):
        return self.run("mixedModel", dep=dep, cluster=cluster, covs=covs or [], **kwargs)

    def survival(self, time_col: Optional[str] = None, event_col: Optional[str] = None, time: Optional[str] = None, event: Optional[str] = None, group: Optional[str] = None, covs: Optional[List[str]] = None, **kwargs):
        t_col = time_col or time or kwargs.pop("time", "")
        e_col = event_col or event or kwargs.pop("event", "")
        return self.run("survival", time=t_col, event=e_col, group=group, covs=covs or [], **kwargs)

    # ---------------- GIS spatial analysis ----------------
    def attach_boundary(self, path: str, key: str, data_key: Optional[str] = None) -> Dict[str, Any]:
        """Link a boundary file (.shp/.zip/.geojson/.gpkg) to the data: `key` = boundary attribute, `data_key` = data column.
        Later spatial calls use it unless they pass `boundary=` explicitly."""
        from .spatial_io import match_keys, read_boundary
        gdf = read_boundary(path)
        data_key = data_key or key
        if key not in gdf.columns:
            raise ValueError(f"경계 파일에 '{key}' 속성 없음")
        if self.df is None or data_key not in self.df.columns:
            raise ValueError(f"데이터에 '{data_key}' 변수 없음")
        self.boundary = {"boundary": path, "boundaryKey": key, "idVar": data_key}
        return match_keys(self.df[data_key], gdf[key])

    def xtset(self, panel_id: str, time_var: Optional[str] = None):
        """Declare the panel structure (Stata xtset) used as default idVar/timeVar by spatial panel models."""
        self.panel_spec = {"idVar": panel_id, "timeVar": time_var}
        return None

    def _spatial(self, analysis_type: str, params: Dict[str, Any]):
        ps = getattr(self, "panel_spec", None)
        if ps and analysis_type == "spatialPanel":
            for k_, v_ in ps.items():
                if v_ and not params.get(k_):
                    params[k_] = v_
        b = getattr(self, "boundary", None)
        if b and not params.get("boundary") and not (params.get("coordX") or params.get("lon") or params.get("lat")):
            for k_, v_ in b.items():
                params.setdefault(k_, v_)
        return self.run(analysis_type, **params)

    def spatial_weights(self, **kwargs):
        """Spatial weights matrix W: contiguity / kNN / distance / inverse distance / file; summary, link map, export."""
        return self._spatial("spatialWeights", kwargs)

    def spatial_autocorrelation(self, vars: Union[str, List[str], None] = None, **kwargs):
        """Global Moran's I, LISA clusters and Getis-Ord Gi* hot spots."""
        return self._spatial("spatial_moran", {"vars": vars or kwargs.pop("dep", None), **kwargs})

    def moran(self, dep: Optional[str] = None, lat: Optional[str] = None, lon: Optional[str] = None, k: int = 4, **kwargs):
        """Global Moran's I & LISA spatial clustering."""
        if lat:
            kwargs.setdefault("coordY", lat)
        if lon:
            kwargs.setdefault("coordX", lon)
        return self._spatial("spatial_moran", {"vars": kwargs.pop("vars", None) or [dep], "k": k, **kwargs})

    def spatial_regression(self, dep: str = "", covs: Optional[List[str]] = None, lat: Optional[str] = None, lon: Optional[str] = None,
                           model_type: Optional[str] = None, modelType: Optional[str] = None, **kwargs):
        """Cross-sectional spatial regression (OLS diagnostics, SLX, SAR, SEM, SDM, SDEM, SAC; Stata spregress)."""
        model = kwargs.pop("model", None) or model_type or modelType or "sar"
        if lat:
            kwargs.setdefault("coordY", lat)
        if lon:
            kwargs.setdefault("coordX", lon)
        return self._spatial("spatial", {"dep": dep, "covs": covs or [], "model": model, **kwargs})

    spatial = spatial_regression

    def spatial_panel(self, dep: str = "", covs: Optional[List[str]] = None, idVar: Optional[str] = None, timeVar: Optional[str] = None, **kwargs):
        """Spatial panel regression: FE (ind/time/both, Lee-Yu) or RE; SAR/SEM/SDM/SDEM/SLX/SAC (Stata xsmle, spxtregress)."""
        params = {"dep": dep, "covs": covs or [], "timeVar": timeVar or kwargs.pop("time", None), **kwargs}
        if idVar:
            params["idVar"] = idVar
        return self._spatial("spatialPanel", params)

    xsmle = spatial_panel

    def gwr(self, dep: str = "", covs: Optional[List[str]] = None, **kwargs):
        """Geographically weighted regression (mgwr): bandwidth selection, local coefficient maps."""
        return self._spatial("gwr", {"dep": dep, "covs": covs or [], **kwargs})

    def spatial_map(self, vars: Union[str, List[str], None] = None, **kwargs):
        """Thematic (choropleth) map of one or more variables."""
        return self._spatial("spatialMap", {"vars": vars, **kwargs})

    def restyle_plots(self, plot_style: Dict[str, Any]) -> "JamoviSession":
        """Re-execute all analyses in session history using an updated `PlotStyle` dictionary."""
        if self.df is None or not self.specs:
            return self
        new_history = []
        new_specs = []
        for idx, sp in enumerate(self.specs):
            if sp.get("analysis_type") == "customFigure":
                if idx < len(self.history):
                    new_history.append(self.history[idx])
                    new_specs.append(sp)
                continue
            merged_params = dict(sp.get("params") or {})
            prev_style = dict(merged_params.get("plotStyle") or {})
            prev_style.update(plot_style or {})
            merged_params["plotStyle"] = prev_style
            res = dispatch_analysis(sp["analysis_type"], self.df, merged_params)
            new_history.append(res)
            new_specs.append({
                "analysis_type": sp["analysis_type"],
                "title": res.title,
                "params": merged_params
            })
        self.history = new_history
        self.specs = new_specs
        self._sync_workspace_if_enabled()
        return self

    def to_python_script(self, dataset_path: Optional[str] = None, include_exports: bool = True) -> str:
        """Generate a Stata `.do`-style runnable `.py` script: load -> data ops -> analyses -> exports."""
        from .syntax_engine import generate_python_script
        ds_path = dataset_path or self.source_path or "sample_data/ToothGrowth.csv"
        return generate_python_script(
            self.specs,
            dataset_path=ds_path,
            data_ops=self.data_ops,
            export_ops=self.export_ops if include_exports else None,
        )

    def save_py(self, output_path: str, dataset_path: Optional[str] = None) -> str:
        """Write reproducible standalone `.py` script to disk."""
        code = self.to_python_script(dataset_path=dataset_path)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        return output_path

    def exec_py(self, code_or_path: str) -> List[Any]:
        """Execute a `.py` file path or raw Python syntax string and append captured analyses to session."""
        from .syntax_engine import execute_python_syntax
        if os.path.isfile(code_or_path):
            with open(code_or_path, "r", encoding="utf-8") as f:
                code = f.read()
        else:
            code = code_or_path
        analyses, _, final_df, _, meta = execute_python_syntax(
            code,
            default_df=self.df,
            default_dataset_path=self.source_path,
            sync_workspace=self.auto_sync_workspace,
            return_meta=True,
        )
        if final_df is not None:
            self.df = final_df
            self.dataset = JamoviDataset(self.df)
        self.data_ops.extend(meta.get("data_ops") or [])
        for op in (meta.get("export_ops") or []):
            if op not in self.export_ops:
                self.export_ops.append(op)
        for a in analyses:
            self.history.append(a)
            self.specs.append({
                "analysis_type": a.name,
                "title": a.title,
                "params": a.options
            })
        return analyses

    def export_tables(self, output_path: str, fmt: Optional[str] = None, report_style: Optional[str] = None,
                      report_options: Optional[Dict[str, Any]] = None) -> str:
        """Export all analyzed tables & embedded graphs in session to `.xlsx`, `.docx`, `.hwpx`, `.pdf`, `.tex`, `.csv`, `.tsv`, `.md`, `.html`, `.json`, or `.omv`.

        report_style="social" applies the social-science table format to coefficient tables (significance stars,
        SE in parentheses, p column dropped) and numbered captions to all tables; see `report_options`
        (stars, star_target, se_position, keep_p, keep_stat, keep_ci, decimals, leading_zero, star_note,
        table_numbers, lang, raw_sheet).
        """
        from .exporters import export_analyses
        from .report_format import resolve_report_format
        rf = resolve_report_format(report_style, report_options)
        out = export_analyses(self.history, output_path=output_path, fmt=fmt, dataset=self.df, report_style=rf)
        op: Dict[str, Any] = {"op": "export_results", "output_path": output_path}
        if fmt:
            op["fmt"] = fmt
        if rf is not None:
            op["report_style"] = "social"
            if rf.options():
                op["report_options"] = rf.options()
        self._record_export(op)
        return out

    def export_results(self, output_path: str, fmt: Optional[str] = None, report_style: Optional[str] = None,
                       report_options: Optional[Dict[str, Any]] = None) -> str:
        """Export full Analysis Results (all statistical tables + standalone/analysis graphs + APA summaries) to 11 formats."""
        return self.export_tables(output_path=output_path, fmt=fmt, report_style=report_style, report_options=report_options)

    def compare_models(
        self,
        *models: Any,
        names: Optional[List[str]] = None,
        stats: Any = "auto",
        estimate: str = "b",
        labels: Optional[Dict[str, str]] = None,
        terms: Optional[List[str]] = None,
        table: Optional[str] = None,
        title: Optional[str] = None,
        lang: str = "ko",
    ):
        """Side-by-side model comparison table (Stata esttab layout) appended to the results.

        models: results of this session (objects or 1-based result numbers) or specs
        {"analysis": "linReg", "dep": ..., ...}. A hierarchical linear regression (blocks) contributes one
        model per block. stats: "auto", "none" or keys such as ["N", "R2", "R2_adj", "AIC"];
        estimate: "b" (B, SE, p), "beta" (β), "both" (B, SE, β) or "exp" (odds/hazard ratios).
        The specs are recorded, so the reproducible script re-estimates the models.
        """
        from .analyses.model_compare import compare_results, normalize_model_spec
        if len(models) == 1 and isinstance(models[0], (list, tuple)):
            models = tuple(models[0])
        if not models:
            raise ValueError("compare_models: 비교할 모형 1개 이상 필요")
        results, specs = [], []
        for m in models:
            if isinstance(m, dict):
                sp = normalize_model_spec(m)
                q = dict(sp)
                a_type = q.pop("analysis")
                results.append(dispatch_analysis(a_type, self.df, q))
                specs.append(sp)
                continue
            if isinstance(m, (int, np.integer)) and not isinstance(m, bool):
                if not (1 <= int(m) <= len(self.history)):
                    raise ValueError(f"compare_models: 결과 번호 {m} 없음 (1~{len(self.history)})")
                m = self.history[int(m) - 1]
            idx = next((i for i, h in enumerate(self.history) if h is m), None)
            if idx is None or idx >= len(self.specs):
                raise ValueError("compare_models: 이 세션에서 실행한 결과만 지정 가능(재현 스크립트에 모형 지정을 기록하기 위함)")
            sp = self.specs[idx]
            if sp.get("analysis_type") in ("customFigure", "compareModels"):
                raise ValueError(f"compare_models: '{sp.get('title')}'는 계수표가 있는 분석이 아님")
            results.append(m)
            specs.append({"analysis": sp["analysis_type"], **dict(sp.get("params") or {})})
        params: Dict[str, Any] = {"models": specs}
        for k, v, d in (("names", names, None), ("stats", stats, "auto"), ("estimate", estimate, "b"),
                        ("labels", labels, None), ("terms", terms, None), ("table", table, None),
                        ("title", title, None), ("lang", lang, "ko")):
            if v is not None and v != d:
                params[k] = list(v) if isinstance(v, tuple) else v
        res = compare_results(results, names=names, stats=stats, estimate=estimate, labels=labels, terms=terms,
                              table=table, title=title, lang=lang, options=params)
        self.history.append(res)
        self.specs.append({"analysis_type": "compareModels", "title": res.title, "params": params})
        self._sync_workspace_if_enabled()
        return res

    def save_plots(self, out_dir: str, fmt: str = "svg") -> List[str]:
        """Export all figures from session analyses as `.svg` (vector), `.pdf` (vector), or `.png`."""
        ext = fmt.lower().lstrip(".")
        os.makedirs(out_dir, exist_ok=True)
        saved: List[str] = []
        for a_idx, a in enumerate(self.history, start=1):
            for p_idx, p in enumerate(getattr(a, "plots", [])):
                safe = re.sub(r"[^A-Za-z0-9가-힣_-]+", "_", p.get("title", f"plot_{p_idx + 1}")).strip("_")[:48]
                fpath = os.path.join(out_dir, f"{a_idx:02d}_{p_idx + 1:02d}_{safe}.{ext}")
                a.save_plot(fpath, plot_index=p_idx)
                saved.append(fpath)
        self._record_export({"op": "save_plots", "out_dir": out_dir, "fmt": ext})
        return saved

    def push_to_ui(self, analysis_type: str, params: Dict[str, Any], server_url: str = "http://127.0.0.1:8500") -> Dict[str, Any]:
        """Push an analysis directly into the OpenStat Studio Web UI when the user has launched it on demand."""
        dataset_name = os.path.basename(self.source_path) if self.source_path else None
        payload = json.dumps({
            "dataset": dataset_name,
            "analysis_type": analysis_type,
            "params": params
        }).encode("utf-8")
        urls = [server_url.rstrip("/")]
        if "8500" in server_url:
            urls.append("http://127.0.0.1:8000")
        last_err = None
        for u in urls:
            try:
                req = urllib.request.Request(
                    f"{u}/api/ai/push",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                last_err = e
        raise RuntimeError(f"OpenStat Studio Web UI is not running ({last_err})")

    def plot_box(self, var: str, group: Optional[str] = None) -> str:
        from .analyses.plots import plot_descriptives_box
        return plot_descriptives_box(self.df, var=var, group=group)

    def plot_mean_ci(self, dep: str, group: str) -> str:
        from .analyses.plots import plot_mean_ci
        return plot_mean_ci(self.df, dep=dep, group=group)

    def to_markdown(self) -> str:
        if not self.history:
            return "No analyses executed yet."
        return "\n\n".join([a.to_markdown() for a in self.history])

    def to_apa(self) -> str:
        if not self.history:
            return "No analyses executed yet."
        narratives = [getattr(a, "apa_narrative", "") for a in self.history if getattr(a, "apa_narrative", "")]
        return "\n\n".join(narratives)

    def save_omv(self, output_path: str) -> str:
        if self.df is None or self.dataset is None:
            raise ValueError("No dataset to save")
        return write_omv(self.dataset, output_path, analyses=self.history)

    def save_html(self, output_path: str, title: str = "OpenStat Studio — Statistical Analysis Report") -> str:
        html_content = render_jamovi_html_report(self.history, title=title)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return output_path

    def __getattr__(self, name: str):
        from .registry import ANALYSIS_CATALOG
        clean = name.lower().replace("_", "")
        for item in ANALYSIS_CATALOG:
            cid = item.get("id", "")
            if cid == name or cid.lower().replace("_", "") == clean:
                return lambda **kwargs: self.run(cid, **kwargs)
        raise AttributeError(f"'JamoviSession' object has no attribute '{name}'")
