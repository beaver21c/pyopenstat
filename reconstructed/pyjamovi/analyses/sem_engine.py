"""
OpenStat SEM engine — maximum likelihood structural equation models with Mplus default conventions.

Model language (Python string, lavaan-style operators)
    F =~ y1 + y2 + y3        measurement (Mplus BY); first loading fixed at 1 unless a modifier is given
    y ~ F + x                regression (Mplus ON)
    y1 ~~ y2                 (residual) covariance (Mplus WITH); `y ~~ y` = (residual) variance
    y ~ 1 / F ~ 1            intercept or mean (Mplus [y])
    modifiers                a*x (label), 1*x / 0*x (fixed), NA*x (free), c(a, b)*x (one entry per group)
    ind := a*b               defined parameter (Mplus MODEL CONSTRAINT NEW)
    a == b, a == 0.5         equality / fixing constraints on labels

Mplus defaults implemented (User's Guide v8 chapters 16-18; Asparouhov & Muthén 2018 SRMR note)
    ESTIMATOR = ML, missing data handled by FIML (all available data, MAR); observed information matrix
    observed exogenous covariates conditioned on (y|x likelihood, x moments not parameters, cases missing x removed)
    mean structure: indicator / outcome intercepts free, factor means 0 (first group in multiple-group analysis)
    first factor loading fixed at 1; exogenous factor covariances free; residual covariances of final dependent
    variables free; residual covariances among factor indicators fixed at 0
    multiple groups: loadings and indicator intercepts equal across groups (scalar), factor means free beyond group 1
    MLR: sandwich standard errors and Yuan–Bentler scaled chi-square with H0/H1 scaling correction factors
    fit: χ², RMSEA (N, √G), CFI/TLI against the Mplus baseline model, SRMR with means, AIC, BIC, SABIC
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

LOG2PI = math.log(2.0 * math.pi)
SECTION_LABELS = {
    "=~": "측정모형",
    "~": "구조모형",
    "~~cov": "공분산",
    "~1": "절편·평균",
    "~~var": "분산·잔차분산",
}
INVARIANCE_LABELS = {
    "configural": "형태 동일성(제약 없음)",
    "metric": "측정단위 동일성(요인적재량 동일)",
    "scalar": "절편 동일성(요인적재량·절편 동일, Mplus 다집단 기본)",
    "strict": "엄격 동일성(요인적재량·절편·잔차분산 동일)",
}


class SEMError(ValueError):
    pass


# =============================================================================================== parser
@dataclass
class Term:
    name: str
    mods: List[Optional[Tuple[str, Any]]]  # one entry (applies to all groups) or one per group


@dataclass
class Statement:
    lhs: str
    op: str
    terms: List[Term] = field(default_factory=list)
    expr: str = ""
    raw: str = ""


_IDENT = re.compile(r"^[^\W\d][\w.]*$", re.UNICODE)


def _strip_name(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == "`" and s[-1] == "`":
        s = s[1:-1]
    return s.strip()


def _split_top(s: str, sep: str) -> List[str]:
    out, depth, cur, quote = [], 0, [], False
    for ch in s:
        if ch == "`":
            quote = not quote
        elif not quote and ch == "(":
            depth += 1
        elif not quote and ch == ")":
            depth -= 1
        if ch == sep and depth == 0 and not quote:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def _parse_mod_token(tok: str) -> Tuple[str, Any]:
    t = tok.strip()
    if not t:
        raise SEMError("빈 모수 지정자")
    if t.upper() == "NA":
        return ("free", None)
    try:
        return ("fixed", float(t))
    except ValueError:
        pass
    m = re.match(r"^start\((.+)\)$", t)
    if m:
        return ("start", float(m.group(1)))
    if not _IDENT.match(t):
        raise SEMError(f"모수 이름(라벨) 형식 오류: '{t}' — 문자로 시작하는 이름 사용")
    return ("label", t)


def _parse_term(raw: str) -> Term:
    raw = raw.strip()
    if not raw:
        raise SEMError("빈 항: '+' 앞뒤에 변수명 필요")
    parts = _split_top(raw, "*")
    if len(parts) == 1:
        return Term(_strip_name(parts[0]), [None])
    if len(parts) > 2:
        raise SEMError(f"항 형식 오류: '{raw}'")
    mod, name = parts[0].strip(), _strip_name(parts[1])
    m = re.match(r"^c\((.*)\)$", mod)
    if m:
        toks = [x for x in m.group(1).split(",")]
        return Term(name, [_parse_mod_token(x) for x in toks])
    return Term(name, [_parse_mod_token(mod)])


def _logical_lines(text: str) -> List[str]:
    lines = []
    for raw in str(text or "").replace(";", "\n").splitlines():
        line = re.split(r"[#!]", raw, maxsplit=1)[0].strip()
        if not line:
            continue
        if lines and (line.startswith("+") or re.search(r"(=~|~~|~|\+|\*|:=|==)\s*$", lines[-1])):
            lines[-1] = lines[-1] + " " + line
        else:
            lines.append(line)
    return lines


def parse_model(text: str) -> List[Statement]:
    stmts: List[Statement] = []
    for line in _logical_lines(text):
        for op in (":=", "==", "=~", "~~", "~"):
            if op in line:
                lhs, rhs = line.split(op, 1)
                break
        else:
            raise SEMError(f"연산자(=~, ~, ~~, :=, ==) 없는 줄: '{line}'")
        lhs, rhs = lhs.strip(), rhs.strip()
        if not lhs or not rhs:
            raise SEMError(f"좌변 또는 우변 누락: '{line}'")
        if op in (":=", "=="):
            stmts.append(Statement(lhs=lhs, op=op, expr=rhs, raw=line))
            continue
        lhs_names = [_strip_name(x) for x in _split_top(lhs, "+")]
        terms = [_parse_term(t) for t in _split_top(rhs, "+")]
        for ln in lhs_names:
            if not ln:
                raise SEMError(f"좌변 변수명 누락: '{line}'")
            if op == "~" and all(t.name == "1" for t in terms):
                stmts.append(Statement(lhs=ln, op="~1", terms=terms, raw=line))
                continue
            if op == "~" and any(t.name == "1" for t in terms):
                stmts.append(Statement(lhs=ln, op="~1", terms=[t for t in terms if t.name == "1"], raw=line))
                terms_r = [t for t in terms if t.name != "1"]
                stmts.append(Statement(lhs=ln, op="~", terms=terms_r, raw=line))
                continue
            stmts.append(Statement(lhs=ln, op=op, terms=terms, raw=line))
    if not any(s.op in ("=~", "~", "~~", "~1") for s in stmts):
        raise SEMError("모형식이 비어 있음: 측정(=~)·회귀(~)·공분산(~~) 문장 1개 이상 필요")
    return stmts


# ================================================================================ safe expression eval
_FUNCS = {"exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs}


def _expr_names(expr: str) -> List[str]:
    tree = ast.parse(expr, mode="eval")
    return sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id not in _FUNCS})


def _eval_expr(expr: str, env: Dict[str, float]) -> float:
    tree = ast.parse(expr, mode="eval")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name):
            if n.id not in env:
                raise SEMError(f"정의식의 알 수 없는 이름: {n.id}")
            return env[n.id]
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            ops = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                   ast.Div: lambda: a / b, ast.Pow: lambda: a ** b}
            for k, f in ops.items():
                if isinstance(n.op, k):
                    return f()
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and len(n.args) == 1:
            return float(_FUNCS[n.func.id](ev(n.args[0])))
        raise SEMError("정의식에 허용되지 않는 표현 포함 (사칙연산·거듭제곱·exp·log·sqrt·abs만 허용)")

    return float(ev(tree))


# ===================================================================================== parameter table
@dataclass
class ParRow:
    group: int
    lhs: str
    op: str
    rhs: str
    mat: str
    i: int
    j: int
    free: bool
    value: float
    label: str = ""
    pid: int = -1
    user: bool = False
    kind: str = ""
    start: Optional[float] = None

    @property
    def section(self) -> str:
        if self.op == "~~":
            return SECTION_LABELS["~~var"] if self.lhs == self.rhs else SECTION_LABELS["~~cov"]
        return SECTION_LABELS[self.op]


def _mod_for(term: Term, g: int, G: int) -> Optional[Tuple[str, Any]]:
    if len(term.mods) == 1:
        return term.mods[0]
    if len(term.mods) != G:
        raise SEMError(f"c(...) 지정 개수({len(term.mods)})와 집단 수({G}) 불일치: {term.name}")
    return term.mods[g]


class ModelSpec:
    """Variables, roles and the parameter table of a (multiple-group) model."""

    def __init__(self, statements: List[Statement], columns: Sequence[str], n_groups: int = 1,
                 invariance: str = "scalar"):
        self.statements = statements
        self.G = int(n_groups)
        self.invariance = invariance
        cols = [str(c) for c in columns]
        meas = [s for s in statements if s.op == "=~"]
        regs = [s for s in statements if s.op == "~"]
        covs = [s for s in statements if s.op == "~~"]
        ints = [s for s in statements if s.op == "~1"]
        self.defined = [(s.lhs.strip(), s.expr) for s in statements if s.op == ":="]
        self.constraints = [(s.lhs.strip(), s.expr.strip()) for s in statements if s.op == "=="]

        order: List[str] = []

        def note(n):
            if n not in order:
                order.append(n)

        for s in statements:
            if s.op in ("=~", "~", "~~", "~1"):
                note(s.lhs)
                for t in s.terms:
                    if t.name != "1":
                        note(t.name)
        self.latent = [n for n in order if any(s.lhs == n for s in meas)]
        observed = [n for n in order if n not in self.latent]
        missing_cols = [n for n in observed if n not in cols]
        if missing_cols:
            raise SEMError(f"데이터에 없는 변수: {', '.join(missing_cols)} (잠재변수는 '=~' 좌변에 정의). "
                           f"사용 가능한 변수: {', '.join(cols[:40])}{' …' if len(cols) > 40 else ''}")
        self.indicators = {n for s in meas for t in s.terms for n in [t.name]}
        self.dep_reg = [n for n in order if any(s.lhs == n for s in regs)]
        self.predictors = {t.name for s in regs for t in s.terms}
        dependents = set(self.dep_reg) | self.indicators
        mentioned = {s.lhs for s in covs + ints} | {t.name for s in covs for t in s.terms}
        x_cand = [n for n in observed if n in self.predictors and n not in dependents]
        self.x_joint = any(x in mentioned for x in x_cand)
        self.ox = [] if self.x_joint else x_cand          # conditioned covariates (Mplus x)
        self.x_names = list(x_cand)                        # observed exogenous predictors (STDY rule)
        self.ov = [n for n in observed if n not in self.ox]
        self.obs = self.ov + self.ox
        self.vars = self.obs + self.latent
        self.idx = {n: k for k, n in enumerate(self.vars)}
        self.p = len(self.obs)
        self.py = len(self.ov)
        self.qx = len(self.ox)
        self.m = len(self.vars)
        self.dependents = dependents
        self.exo_latent = [f for f in self.latent if f not in dependents]
        self.exo_obs = [n for n in self.ov if n not in dependents]
        self.final_dep = [n for n in self.dep_reg if n not in self.predictors and n not in self.indicators]

        rows: Dict[Tuple[int, str, int, int], ParRow] = {}
        first_ind: Dict[str, str] = {}
        for s in meas:
            if s.lhs not in first_ind and s.terms:
                first_ind[s.lhs] = s.terms[0].name

        def key(g, mat, i, j):
            if mat == "S":
                i, j = min(i, j), max(i, j)
            return (g, mat, i, j)

        def put(g, lhs, op, rhs, mat, i, j, mod, default_free, default_value, kind, user):
            free, value, label, start = default_free, default_value, "", None
            if mod is not None:
                typ, val = mod
                if typ == "fixed":
                    free, value = False, float(val)
                elif typ == "free":
                    free = True
                elif typ == "label":
                    free, label = True, str(val)
                elif typ == "start":
                    free, start = True, float(val)
            k = key(g, mat, i, j)
            if k in rows and not user:
                return
            if mat == "S" and i > j:
                lhs, rhs, i, j = rhs, lhs, j, i
            rows[k] = ParRow(g, lhs, op, rhs, mat, i, j, free, value, label, -1, user, kind, start)

        for g in range(self.G):
            for s in meas:
                f = self.idx[s.lhs]
                for t in s.terms:
                    if t.name == s.lhs:
                        raise SEMError(f"자기 자신을 지표로 지정: {s.lhs}")
                    mod = _mod_for(t, g, self.G)
                    first = first_ind.get(s.lhs) == t.name
                    put(g, s.lhs, "=~", t.name, "A", self.idx[t.name], f, mod, not first, 1.0, "loading", True)
            for s in regs:
                for t in s.terms:
                    if t.name == s.lhs:
                        raise SEMError(f"자기 회귀 지정: {s.lhs}")
                    put(g, s.lhs, "~", t.name, "A", self.idx[s.lhs], self.idx[t.name], _mod_for(t, g, self.G),
                        True, 0.0, "regression", True)
            for s in covs:
                for t in s.terms:
                    if t.name in self.ox or s.lhs in self.ox:
                        raise SEMError("조건부 공변량의 분산·공분산 지정")
                    kind = "variance" if t.name == s.lhs else "covariance"
                    put(g, s.lhs, "~~", t.name, "S", self.idx[s.lhs], self.idx[t.name], _mod_for(t, g, self.G),
                        True, 0.0, kind, True)
            for s in ints:
                if s.lhs in self.ox:
                    raise SEMError("조건부 공변량의 평균 지정")
                put(g, s.lhs, "~1", "", "M", self.idx[s.lhs], self.idx[s.lhs], _mod_for(s.terms[0], g, self.G),
                    True, 0.0, "intercept" if s.lhs not in self.latent else "mean", True)
            # Mplus defaults
            for v in self.ov + self.latent:
                k = self.idx[v]
                put(g, v, "~~", v, "S", k, k, None, True, 0.0, "variance", False)
            for a_i, a in enumerate(self.exo_latent):
                for b in self.exo_latent[a_i + 1:]:
                    put(g, a, "~~", b, "S", self.idx[a], self.idx[b], None, True, 0.0, "covariance", False)
            for a_i, a in enumerate(self.exo_obs):
                for b in self.exo_obs[a_i + 1:]:
                    put(g, a, "~~", b, "S", self.idx[a], self.idx[b], None, True, 0.0, "covariance", False)
            for a_i, a in enumerate(self.final_dep):
                for b in self.final_dep[a_i + 1:]:
                    put(g, a, "~~", b, "S", self.idx[a], self.idx[b], None, True, 0.0, "covariance", False)
            for v in self.ov:
                put(g, v, "~1", "", "M", self.idx[v], self.idx[v], None, True, 0.0, "intercept", False)
            if self.G > 1 and invariance in ("scalar", "strict") and g > 0:
                for f in self.latent:
                    put(g, f, "~1", "", "M", self.idx[f], self.idx[f], None, True, 0.0, "mean", False)

        # invariance labels (only where the user gave no label / fixed value)
        if self.G > 1:
            for r in rows.values():
                if not r.free or r.label:
                    continue
                if invariance in ("metric", "scalar", "strict") and r.op == "=~":
                    r.label = f".L.{r.lhs}.{r.rhs}"
                elif invariance in ("scalar", "strict") and r.op == "~1" and r.lhs in self.indicators \
                        and r.lhs not in self.latent:
                    r.label = f".I.{r.lhs}"
                elif invariance == "strict" and r.op == "~~" and r.lhs == r.rhs and r.lhs in self.indicators \
                        and r.lhs not in self.latent:
                    r.label = f".E.{r.lhs}"
        self.rows: List[ParRow] = sorted(rows.values(), key=lambda r: (r.group, ["=~", "~", "~~", "~1"].index(r.op)))
        self._apply_constraints()
        self._assign_ids()

    # ------------------------------------------------------------------ constraints and ids
    def _apply_constraints(self):
        labels = {r.label for r in self.rows if r.label}
        parent = {l: l for l in labels}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in self.constraints:
            if a not in labels:
                raise SEMError(f"제약식의 라벨 없음: {a}")
            try:
                val = float(b)
                for r in self.rows:
                    if r.label == a:
                        r.free, r.value = False, val
                continue
            except ValueError:
                pass
            if b not in labels:
                raise SEMError(f"지원하지 않는 제약식: {a} == {b} (라벨 == 라벨 또는 라벨 == 숫자만 지원)")
            parent[find(a)] = find(b)
        for r in self.rows:
            if r.label and r.label in parent:
                r.label_group = find(r.label)
            else:
                r.label_group = ""

    def _assign_ids(self):
        ids: Dict[str, int] = {}
        q = 0
        for n, r in enumerate(self.rows):
            if not r.free:
                r.pid = -1
                continue
            k = r.label_group if r.label_group else f"#row{n}"
            if k not in ids:
                ids[k] = q
                q += 1
            r.pid = ids[k]
        self.q = q
        self.label_pid = {}
        for r in self.rows:
            if r.label and r.free:
                self.label_pid[r.label] = r.pid
        self.free_pos: List[List[Tuple[int, str, int, int]]] = [[] for _ in range(self.G)]
        for r in self.rows:
            if r.free:
                self.free_pos[r.group].append((r.pid, r.mat, r.i, r.j))
        known = set(self.label_pid) | {r.label for r in self.rows if r.label} | {d[0] for d in self.defined}
        for name, expr in self.defined:
            for nm in _expr_names(expr):
                if nm not in known:
                    raise SEMError(f"정의식 '{name} := {expr}'의 라벨 '{nm}' 없음 (모수에 a*x 형식으로 라벨 지정)")


# =============================================================================================== data
@dataclass
class Pattern:
    idx: np.ndarray
    Z: np.ndarray
    n: int
    mean: np.ndarray
    C: np.ndarray


class GroupData:
    def __init__(self, X: np.ndarray):
        self.X = X
        self.n = X.shape[0]
        mask = ~np.isnan(X)
        keys = [tuple(r) for r in mask]
        pats: Dict[Tuple[bool, ...], List[int]] = {}
        for i, k in enumerate(keys):
            pats.setdefault(k, []).append(i)
        self.patterns: List[Pattern] = []
        self.case_pattern = np.empty(self.n, dtype=int)
        self.case_pos = np.empty(self.n, dtype=int)
        for pi, (k, rows_) in enumerate(pats.items()):
            idx = np.where(np.array(k))[0]
            Z = X[np.ix_(rows_, idx)]
            mean = Z.mean(axis=0)
            D = Z - mean
            self.patterns.append(Pattern(idx, Z, len(rows_), mean, D.T @ D / len(rows_)))
            self.case_pattern[rows_] = pi
            self.case_pos[rows_] = np.arange(len(rows_))
        self.rows_by_pattern = list(pats.values())
        # patterns stacked by number of observed variables for batched likelihood evaluation
        by_k: Dict[int, List[Pattern]] = {}
        for pat in self.patterns:
            by_k.setdefault(len(pat.idx), []).append(pat)
        self.batches = []
        for k, lst in by_k.items():
            if k == 0:
                continue
            O = np.stack([pt.idx for pt in lst])
            self.batches.append((O, np.stack([pt.mean for pt in lst]), np.stack([pt.C for pt in lst]),
                                 np.array([pt.n for pt in lst], dtype=float)))


# ===================================================================================== likelihood core
def _pattern_terms(mu, Sig, pat):
    o = pat.idx
    S_oo = Sig[np.ix_(o, o)]
    c, low = linalg.cho_factor(S_oo, lower=True, check_finite=False)
    logdet = 2.0 * np.sum(np.log(np.diag(c)))
    Sinv = linalg.cho_solve((c, low), np.eye(len(o)), check_finite=False)
    d = pat.mean - mu[o]
    return o, Sinv, logdet, d


def _accumulate(mu, Sig, gd: GroupData, grad: bool = True, hess: Optional[str] = None):
    """Pattern-wise FIML pieces in moment space (mu, vec Sigma).

    Returns ll, gmu (p), G (p, p) with dll = gmu·dmu + tr(G dSigma), and for hess in {"observed", "expected"}
    the moment-space Hessian blocks Hmm (p, p), HmS (p, p*p), HSS (p*p, p*p) (row-major vec of Sigma).
    """
    p = len(mu)
    ll = 0.0
    gmu = np.zeros(p)
    G = np.zeros((p, p))
    Hmm = HmS = HSS = None
    if hess:
        Hmm = np.zeros((p, p))
        HmS = np.zeros((p, p * p))
        HSS = np.zeros((p * p, p * p))
    for pat in gd.patterns:
        o, Sinv, logdet, d = _pattern_terms(mu, Sig, pat)
        k, n = len(o), pat.n
        T = pat.C + np.outer(d, d)
        ll += -0.5 * n * (k * LOG2PI + logdet + np.sum(Sinv * T))
        if grad or hess:
            u = Sinv @ d
            gmu[o] += n * u
            P = Sinv @ T @ Sinv
            G[np.ix_(o, o)] += 0.5 * n * (P - Sinv)
        if hess:
            iS = (o[:, None] * p + o[None, :]).ravel()
            Hmm[np.ix_(o, o)] += -n * Sinv
            if hess == "observed":
                HmS[np.ix_(o, iS)] += (-n * np.einsum("ab,c->abc", Sinv, u)).reshape(k, k * k)
                K = 0.5 * n * (-2.0 * np.einsum("ab,cd->bcda", Sinv, P) + np.einsum("ab,cd->bcda", Sinv, Sinv))
            else:
                K = -0.5 * n * np.einsum("ab,cd->bcda", Sinv, Sinv)
            HSS[np.ix_(iS, iS)] += K.reshape(k * k, k * k)
    return ll, gmu, G, Hmm, HmS, HSS


def _accumulate_batched(mu, Sig, gd: GroupData, grad: bool = True):
    """Same ll / gmu / G as _accumulate, with missing-data patterns of equal size evaluated together."""
    p = len(mu)
    ll = 0.0
    gmu = np.zeros(p)
    G = np.zeros((p, p))
    for O, means, Cs, ns in gd.batches:
        k = O.shape[1]
        S = Sig[O[:, :, None], O[:, None, :]]
        L = np.linalg.cholesky(S)
        logdet = 2.0 * np.sum(np.log(np.diagonal(L, axis1=1, axis2=2)), axis=1)
        Li = np.linalg.inv(L)
        Sinv = np.einsum("nji,njk->nik", Li, Li)
        d = means - mu[O]
        T = Cs + d[:, :, None] * d[:, None, :]
        ll += float(np.sum(-0.5 * ns * (k * LOG2PI + logdet + np.einsum("nij,nij->n", Sinv, T))))
        if grad:
            u = np.einsum("nij,nj->ni", Sinv, d)
            np.add.at(gmu, O, ns[:, None] * u)
            P = np.einsum("nij,njk,nkl->nil", Sinv, T, Sinv, optimize=True)
            np.add.at(G, (O[:, :, None], O[:, None, :]), 0.5 * ns[:, None, None] * (P - Sinv))
    return ll, gmu, G


def loglik(moments: List[Tuple], data: List[GroupData], grad: bool = False):
    """moments: per group (mu, Sigma, Jmu, JS). Returns (ll, gradient)."""
    ll = 0.0
    g = None
    for (mu, Sig, Jmu, JS), gd in zip(moments, data):
        if grad and g is None:
            g = np.zeros(Jmu.shape[1])
        try:
            l_g, gmu, G = _accumulate_batched(mu, Sig, gd, grad=grad)
        except (np.linalg.LinAlgError, linalg.LinAlgError, ValueError):
            return -np.inf, (np.zeros_like(g) if g is not None else None)
        if not np.isfinite(l_g):
            return -np.inf, (np.zeros_like(g) if g is not None else None)
        ll += l_g
        if grad:
            g += Jmu.T @ gmu + np.tensordot(G, JS, axes=([0, 1], [0, 1]))
    return ll, g


def _info_from_pieces(Jmu, JS, Hmm, HmS, HSS) -> np.ndarray:
    p, q = Jmu.shape
    JSv = JS.reshape(p * p, q)
    C = Jmu.T @ HmS @ JSv
    H = Jmu.T @ Hmm @ Jmu + C + C.T + JSv.T @ HSS @ JSv
    return -0.5 * (H + H.T)


def information(moments, data, kind: str = "expected") -> np.ndarray:
    """Expected information, or the exact observed information for models whose moments are linear in theta."""
    I = None
    for (mu, Sig, Jmu, JS), gd in zip(moments, data):
        _, _, _, Hmm, HmS, HSS = _accumulate(mu, Sig, gd, grad=False, hess=kind)
        Ig = _info_from_pieces(Jmu, JS, Hmm, HmS, HSS)
        I = Ig if I is None else I + Ig
    return I


def expected_info(moments, data) -> np.ndarray:
    return information(moments, data, "expected")


def linear_observed_info(moments, data) -> np.ndarray:
    return information(moments, data, "observed")


def casewise_scores(moments, data) -> np.ndarray:
    out = []
    for (mu, Sig, Jmu, JS), gd in zip(moments, data):
        p, q = Jmu.shape
        Sm = np.zeros((gd.n, p))
        SS = np.zeros((gd.n, p * p))
        for pat, rows_ in zip(gd.patterns, gd.rows_by_pattern):
            o, Sinv, logdet, d = _pattern_terms(mu, Sig, pat)
            k = len(o)
            R = (pat.Z - mu[o]) @ Sinv
            iS = (o[:, None] * p + o[None, :]).ravel()
            Sm[np.ix_(rows_, o)] = R
            SS[np.ix_(rows_, iS)] = 0.5 * (np.einsum("na,nb->nab", R, R) - Sinv[None, :, :]).reshape(len(rows_), k * k)
        out.append(Sm @ Jmu + SS @ JS.reshape(p * p, q))
    return np.vstack(out) if out else np.zeros((0, 0))


def numeric_hessian(fun_grad: Callable[[np.ndarray], np.ndarray], theta: np.ndarray, rel: float = 1e-5) -> np.ndarray:
    q = len(theta)
    H = np.zeros((q, q))
    for j in range(q):
        h = rel * max(1.0, abs(theta[j]))
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        H[:, j] = (fun_grad(tp) - fun_grad(tm)) / (2 * h)
    return 0.5 * (H + H.T)


# ===================================================================================== moment models
class RAMModel:
    def __init__(self, spec: ModelSpec, xmom: List[Tuple[np.ndarray, np.ndarray]]):
        self.spec = spec
        m = spec.m
        self.A0, self.S0, self.M0 = [], [], []
        for g in range(spec.G):
            A = np.zeros((m, m))
            S = np.zeros((m, m))
            M = np.zeros(m)
            for r in spec.rows:
                if r.group != g or r.free:
                    continue
                if r.mat == "A":
                    A[r.i, r.j] = r.value
                elif r.mat == "S":
                    S[r.i, r.j] = S[r.j, r.i] = r.value
                else:
                    M[r.i] = r.value
            if spec.qx:
                xs = slice(spec.py, spec.p)
                S[xs, xs] = xmom[g][1]
                M[xs] = xmom[g][0]
            self.A0.append(A)
            self.S0.append(S)
            self.M0.append(M)
        self.q = spec.q

    def matrices(self, theta, g):
        A, S, M = self.A0[g].copy(), self.S0[g].copy(), self.M0[g].copy()
        for pid, mat, i, j in self.spec.free_pos[g]:
            v = theta[pid]
            if mat == "A":
                A[i, j] = v
            elif mat == "S":
                S[i, j] = S[j, i] = v
            else:
                M[i] = v
        return A, S, M

    def all_moments(self, theta, g):
        A, S, M = self.matrices(theta, g)
        B = np.linalg.inv(np.eye(self.spec.m) - A)
        Om = B @ S @ B.T
        return B, Om, B @ M, A, S, M

    def moments(self, theta, jac=True):
        p, q = self.spec.p, self.q
        out = []
        for g in range(self.spec.G):
            B, Om, BM, A, S, M = self.all_moments(theta, g)
            mu, Sig = BM[:p], Om[:p, :p]
            Jmu = np.zeros((p, q))
            JS = np.zeros((p, p, q))
            if jac:
                for pid, mat, i, j in self.spec.free_pos[g]:
                    if mat == "A":
                        u = B[:p, i]
                        D = np.outer(u, Om[j, :p])
                        JS[:, :, pid] += D + D.T
                        Jmu[:, pid] += u * BM[j]
                    elif mat == "S":
                        u = B[:p, i]
                        if i == j:
                            JS[:, :, pid] += np.outer(u, u)
                        else:
                            D = np.outer(u, B[:p, j])
                            JS[:, :, pid] += D + D.T
                    else:
                        Jmu[:, pid] += B[:p, i]
            out.append((mu, Sig, Jmu, JS))
        return out

    def position_jacobian(self, theta, g, mat, i, j):
        """d(mu, Sigma)/d(value at a matrix position) for a fixed or released parameter."""
        p = self.spec.p
        B, Om, BM, A, S, M = self.all_moments(theta, g)
        jm = np.zeros(p)
        js = np.zeros((p, p))
        if mat == "A":
            u = B[:p, i]
            D = np.outer(u, Om[j, :p])
            js = D + D.T
            jm = u * BM[j]
        elif mat == "S":
            u = B[:p, i]
            js = np.outer(u, u) if i == j else np.outer(u, B[:p, j]) + np.outer(B[:p, j], u)
        else:
            jm = B[:p, i].copy()
        return jm, js


class SaturatedModel:
    """Unrestricted means and covariances (per group): parameters mu and vech(Sigma)."""

    def __init__(self, p: int, G: int):
        self.p, self.G = p, G
        self.tri = [(a, b) for a in range(p) for b in range(a, p)]
        self.qg = p + len(self.tri)
        self.q = self.qg * G

    def pack(self, mus, Sigs):
        th = []
        for mu, Sig in zip(mus, Sigs):
            th.extend(mu)
            th.extend(Sig[a, b] for a, b in self.tri)
        return np.array(th)

    def moments(self, theta, jac=True):
        p, out = self.p, []
        for g in range(self.G):
            t = theta[g * self.qg:(g + 1) * self.qg]
            mu = t[:p].copy()
            Sig = np.zeros((p, p))
            for k, (a, b) in enumerate(self.tri):
                Sig[a, b] = Sig[b, a] = t[p + k]
            Jmu = np.zeros((p, self.q))
            JS = np.zeros((p, p, self.q))
            off = g * self.qg
            for a in range(p):
                Jmu[a, off + a] = 1.0
            for k, (a, b) in enumerate(self.tri):
                JS[a, b, off + p + k] = 1.0
                JS[b, a, off + p + k] = 1.0
            out.append((mu, Sig, Jmu, JS))
        return out


def em_saturated(gd: GroupData, max_iter: int = 10000, tol: float = 1e-11):
    X = gd.X
    p = X.shape[1]
    mu = np.nanmean(X, axis=0)
    Sig = np.diag(np.nanvar(X, axis=0))
    if len(gd.patterns) == 1 and len(gd.patterns[0].idx) == p:
        return gd.patterns[0].mean.copy(), gd.patterns[0].C.copy(), 1
    N = gd.n
    it = 0
    for it in range(1, max_iter + 1):
        sx = np.zeros(p)
        sxx = np.zeros((p, p))
        for pat in gd.patterns:
            o = pat.idx
            mm = np.setdiff1d(np.arange(p), o)
            Z = pat.Z
            if len(mm) == 0:
                sx += Z.sum(0)
                sxx += Z.T @ Z
                continue
            Soo = Sig[np.ix_(o, o)]
            Bm = linalg.solve(Soo, Sig[np.ix_(o, mm)], assume_a="pos").T
            Y = np.empty((pat.n, p))
            Y[:, o] = Z
            Y[:, mm] = mu[mm] + (Z - mu[o]) @ Bm.T
            sx += Y.sum(0)
            sxx += Y.T @ Y
            sxx[np.ix_(mm, mm)] += pat.n * (Sig[np.ix_(mm, mm)] - Bm @ Sig[np.ix_(o, mm)])
        mu_n = sx / N
        Sig_n = sxx / N - np.outer(mu_n, mu_n)
        delta = max(np.max(np.abs(mu_n - mu)), np.max(np.abs(Sig_n - Sig)))
        mu, Sig = mu_n, 0.5 * (Sig_n + Sig_n.T)
        if delta < tol:
            break
    return mu, Sig, it


# ======================================================================================== results
@dataclass
class FitResult:
    spec: ModelSpec
    model: RAMModel
    data: List[GroupData]
    theta: np.ndarray
    ll_joint: float
    ll_x: float
    converged: bool
    iterations: int
    message: str = ""


class SEMModel:
    """Fit a model to a DataFrame with Mplus-default conventions."""

    def __init__(self, df: pd.DataFrame, model: str, group: Optional[str] = None, estimator: str = "ml",
                 missing: str = "fiml", invariance: str = "scalar", information: str = "observed",
                 group_order: Optional[Sequence[Any]] = None):
        self.model_text = model
        self.statements = parse_model(model)
        self.estimator = estimator
        self.missing = missing
        self.information = information
        self.group = group or None
        self.invariance = invariance
        self.notes: List[str] = []
        df = df.copy()
        if self.group:
            if self.group not in df.columns:
                raise SEMError(f"집단 변수 '{self.group}' 없음")
            df = df[df[self.group].notna()]
            levels = list(group_order) if group_order else sorted(df[self.group].unique(), key=lambda v: str(v))
            if len(levels) < 2:
                raise SEMError("집단 변수의 수준이 2개 미만")
            self.group_levels = levels
        else:
            self.group_levels = [None]
        G = len(self.group_levels)
        self.spec = ModelSpec(self.statements, df.columns, G, invariance if G > 1 else "configural")
        spec = self.spec
        bad = [c for c in spec.obs if not pd.api.types.is_numeric_dtype(df[c])]
        if bad:
            raise SEMError(f"숫자형이 아닌 변수: {', '.join(bad)} — 범주형 지표(WLSMV)는 지원하지 않음. 숫자 부호화 후 연속형으로 분석")
        n_raw = len(df)
        self.n_input = n_raw
        if missing == "listwise":
            keep = df[spec.obs].notna().all(axis=1)
            if (~keep).sum():
                self.notes.append(f"목록별 삭제: 결측이 있는 {int((~keep).sum())}개 사례 제외")
            df = df[keep]
        else:
            if spec.ox:
                miss_x = df[spec.ox].isna().any(axis=1)
                if miss_x.sum():
                    self.notes.append(f"공변량(x) 결측 {int(miss_x.sum())}개 사례 제외 (Mplus 규칙: 분석이 x에 조건부)")
                df = df[~miss_x]
            all_y_missing = df[spec.ov].isna().all(axis=1)
            if all_y_missing.sum():
                self.notes.append(f"종속·지표 변수가 모두 결측인 {int(all_y_missing.sum())}개 사례 제외")
            df = df[~all_y_missing]
        self.df_used = df
        self.data: List[GroupData] = []
        self.xmom: List[Tuple[np.ndarray, np.ndarray]] = []
        self.group_n: List[int] = []
        for lv in self.group_levels:
            sub = df if lv is None else df[df[self.group] == lv]
            X = sub[spec.obs].to_numpy(dtype=float)
            if X.shape[0] < 2:
                raise SEMError(f"집단 '{lv}'의 사례 수 부족")
            self.data.append(GroupData(X))
            self.group_n.append(X.shape[0])
            if spec.qx:
                Xx = X[:, spec.py:]
                mx = Xx.mean(0)
                Sx = np.atleast_2d(np.cov(Xx.T, bias=True))
                self.xmom.append((mx, Sx))
            else:
                self.xmom.append((np.zeros(0), np.zeros((0, 0))))
        self.N = int(sum(self.group_n))
        if spec.q == 0:
            raise SEMError("추정할 자유 모수 없음")
        self.ram = RAMModel(spec, self.xmom)

    # ------------------------------------------------------------------ helpers
    def ll_x(self) -> float:
        if not self.spec.qx:
            return 0.0
        tot = 0.0
        for (mx, Sx), n in zip(self.xmom, self.group_n):
            k = len(mx)
            sign, logdet = np.linalg.slogdet(Sx)
            if sign <= 0:
                raise SEMError("공변량(x) 공분산행렬이 양정치가 아님 (상수·완전 공선 공변량 확인)")
            tot += -0.5 * n * (k * LOG2PI + logdet + k)
        return tot

    def start_values(self) -> np.ndarray:
        spec = self.spec
        theta = np.zeros(spec.q)
        seen = set()
        X = np.vstack([gd.X for gd in self.data])
        cols = {n: k for k, n in enumerate(spec.obs)}
        var = np.nanvar(X, axis=0)
        mean = np.nanmean(X, axis=0)
        cov_df = pd.DataFrame(X).cov(min_periods=2).to_numpy()
        first_ind: Dict[str, str] = {}
        for s in self.statements:
            if s.op == "=~" and s.lhs not in first_ind:
                first_ind[s.lhs] = s.terms[0].name

        def ref_var(f, depth=0):
            r = first_ind.get(f)
            if r is None or depth > 5:
                return 1.0
            if r in cols:
                v = var[cols[r]]
                return 0.5 * v if np.isfinite(v) and v > 0 else 1.0
            return ref_var(r, depth + 1) * 0.8

        for r in spec.rows:
            if not r.free or r.pid in seen:
                continue
            seen.add(r.pid)
            if r.start is not None:
                v = r.start
            elif r.op == "=~":
                ref = first_ind.get(r.lhs)
                v = 1.0
                if ref in cols and r.rhs in cols:
                    c = cov_df[cols[r.rhs], cols[ref]]
                    phi = ref_var(r.lhs)
                    if np.isfinite(c) and phi > 0:
                        v = float(np.clip(c / phi, -10, 10))
                        if abs(v) < 0.05:
                            v = 0.5
            elif r.op == "~":
                v = 0.0
            elif r.op == "~~":
                if r.lhs == r.rhs:
                    if r.lhs in cols:
                        vv = var[cols[r.lhs]]
                        vv = vv if np.isfinite(vv) and vv > 0 else 1.0
                        v = vv if r.lhs in spec.exo_obs else 0.5 * vv
                    else:
                        v = ref_var(r.lhs) * (0.5 if r.lhs in spec.dependents else 1.0)
                else:
                    v = 0.0
            else:
                v = float(mean[cols[r.lhs]]) if r.lhs in cols and np.isfinite(mean[cols[r.lhs]]) else 0.0
            theta[r.pid] = v
        return theta

    def _negll(self, theta):
        ll, g = loglik(self.ram.moments(theta), self.data, grad=True)
        if not np.isfinite(ll):
            return 1e10, np.zeros_like(theta)
        return -ll / self.N, -g / self.N

    def grad_ll(self, theta):
        return loglik(self.ram.moments(theta), self.data, grad=True)[1]

    def fit(self, start: Optional[np.ndarray] = None, polish: bool = True, max_iter: int = 2000) -> FitResult:
        th0 = self.start_values() if start is None else np.asarray(start, float).copy()
        best = None
        for attempt in range(3):
            if attempt == 1:
                th0 = self.start_values() * 0.9 + 0.05
            elif attempt == 2:
                th0 = self._alt_start()
            try:
                res = optimize.minimize(self._negll, th0, jac=True, method="BFGS",
                                        options={"gtol": 1e-9, "maxiter": max_iter})
            except Exception:  # pragma: no cover
                continue
            if best is None or res.fun < best.fun:
                best = res
            if res.fun < 1e9 and np.max(np.abs(res.jac)) < 1e-5:
                break
        if best is None or best.fun >= 1e9:
            raise SEMError("추정 실패: 모형 내재 공분산행렬이 양정치가 되지 않음 (모형 식별·시작값 확인)")
        theta = best.x
        converged = bool(np.max(np.abs(best.jac)) < 1e-5)
        self.last_hessian = None
        if polish:
            for _ in range(4):
                g = self.grad_ll(theta)
                if np.max(np.abs(g)) / self.N < 1e-9 and self.last_hessian is not None:
                    break
                H = numeric_hessian(self.grad_ll, theta)
                self.last_hessian = (theta.copy(), H)
                if np.max(np.abs(g)) / self.N < 1e-9:
                    break
                try:
                    step = np.linalg.solve(H, g)
                except np.linalg.LinAlgError:
                    break
                cand = theta - step
                if loglik(self.ram.moments(cand, jac=False), self.data)[0] >= loglik(self.ram.moments(theta, jac=False), self.data)[0] - 1e-9:
                    theta = cand
                else:
                    break
            converged = converged or np.max(np.abs(self.grad_ll(theta))) / self.N < 1e-6
        ll = loglik(self.ram.moments(theta, jac=False), self.data)[0]
        return FitResult(self.spec, self.ram, self.data, theta, ll, self.ll_x(), converged, int(best.nit),
                         str(best.message))

    def _alt_start(self):
        th = self.start_values()
        for r in self.spec.rows:
            if r.free and r.op == "=~":
                th[r.pid] = 1.0
        return th


# ============================================================================== full analysis wrapper
@dataclass
class SEMResult:
    sem: SEMModel
    fit: FitResult
    vcov: np.ndarray
    se: np.ndarray
    fitstats: Dict[str, float]
    notes: List[str]
    boot: Optional[np.ndarray] = None
    identified: bool = True

    @property
    def spec(self) -> ModelSpec:
        return self.sem.spec

    @property
    def theta(self) -> np.ndarray:
        return self.fit.theta


def _pval(z):
    return float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")


def _trace_HinvB(H: np.ndarray, B: np.ndarray) -> float:
    try:
        return float(np.trace(np.linalg.solve(H, B)))
    except np.linalg.LinAlgError:
        return float("nan")


def h1_statistics(sem: SEMModel, want_mlr: bool) -> Dict[str, Any]:
    spec = sem.spec
    mus, Sigs = [], []
    for gd in sem.data:
        mu, Sig, _ = em_saturated(gd)
        mus.append(mu)
        Sigs.append(Sig)
    sat = SaturatedModel(spec.p, spec.G)
    th = sat.pack(mus, Sigs)
    mom = sat.moments(th)
    ll_joint = loglik(mom, sem.data)[0]
    out = {"mus": mus, "Sigs": Sigs, "ll_joint": ll_joint}
    if want_mlr:
        I = linear_observed_info(mom, sem.data) if sem.information == "observed" else expected_info(mom, sem.data)
        Bm = casewise_scores(mom, sem.data)
        tr_full = _trace_HinvB(I, Bm.T @ Bm)
        tr_x = 0.0
        if spec.qx:
            xs = np.arange(spec.py, spec.p)
            xd = [GroupData(gd.X[:, xs]) for gd in sem.data]
            satx = SaturatedModel(spec.qx, spec.G)
            thx = satx.pack([m for m, _ in sem.xmom], [S for _, S in sem.xmom])
            momx = satx.moments(thx)
            Ix = expected_info(momx, xd)
            Bx = casewise_scores(momx, xd)
            tr_x = _trace_HinvB(Ix, Bx.T @ Bx)
        out["trace"] = tr_full - tr_x
    return out


def baseline_statistics(sem: SEMModel, want_mlr: bool) -> Dict[str, float]:
    """Mplus baseline: free means and variances of the dependent variables, no covariances (x conditioned)."""
    spec = sem.spec
    ll, tr, npar = 0.0, 0.0, 0
    for gd in sem.data:
        for k in range(spec.py):
            y = gd.X[:, k]
            y = y[~np.isnan(y)]
            n = len(y)
            if n < 2:
                continue
            m, v = y.mean(), y.var()
            ll += -0.5 * n * (LOG2PI + math.log(v) + 1.0)
            npar += 2
            if want_mlr:
                s1 = (y - m) / v
                s2 = -0.5 / v + 0.5 * (y - m) ** 2 / v ** 2
                Bm = np.array([[s1 @ s1, s1 @ s2], [s1 @ s2, s2 @ s2]])
                Hm = np.diag([n / v, n / (2 * v * v)])
                tr += _trace_HinvB(Hm, Bm)
    return {"ll": ll, "npar": npar, "trace": tr}


def rmsea_ci(chi2: float, df: int, N: int, G: int, level: float = 0.90):
    if df <= 0 or not np.isfinite(chi2):
        return 0.0, 0.0, 0.0, 1.0
    est = math.sqrt(G) * math.sqrt(max(chi2 - df, 0.0) / (df * N))

    def solve(target):
        f = lambda lam: stats.ncx2.cdf(chi2, df, lam) - target if lam > 0 else stats.chi2.cdf(chi2, df) - target
        if f(1e-12) < 0:
            return 0.0
        hi = max(chi2, 1.0)
        while f(hi) > 0:
            hi *= 2
            if hi > 1e8:
                return hi
        return optimize.brentq(f, 1e-12, hi, xtol=1e-10)

    lo = solve(1 - (1 - level) / 2)
    up = solve((1 - level) / 2)
    lo_r = math.sqrt(G) * math.sqrt(lo / (df * N))
    up_r = math.sqrt(G) * math.sqrt(up / (df * N))
    nc0 = 0.05 ** 2 * df * N / G
    pclose = float(stats.ncx2.sf(chi2, df, nc0)) if nc0 > 0 else float("nan")
    return est, lo_r, up_r, pclose


def srmr(sem: SEMModel, theta: np.ndarray, h1: Dict[str, Any]) -> float:
    spec = sem.spec
    mom = sem.ram.moments(theta, jac=False)
    p, qx = spec.p, spec.qx
    vals = []
    for (mu0, S0, _, _), mu1, S1, n in zip(mom, h1["mus"], h1["Sigs"], sem.group_n):
        tot = 0.0
        for j in range(p):
            for k in range(j):
                tot += (S1[j, k] / math.sqrt(S1[j, j] * S1[k, k]) - S0[j, k] / math.sqrt(S0[j, j] * S0[k, k])) ** 2
            tot += (mu1[j] / math.sqrt(S1[j, j]) - mu0[j] / math.sqrt(S0[j, j])) ** 2
            tot += ((S1[j, j] - S0[j, j]) / S1[j, j]) ** 2
        denom = p * (p + 1) / 2 + p - qx * (qx + 1) / 2 - qx
        vals.append((math.sqrt(tot / denom) if denom > 0 else 0.0, n))
    N = sum(n for _, n in vals)
    return float(sum(v * n / N for v, n in vals))


def estimate(df: pd.DataFrame, model: str, group: Optional[str] = None, estimator: str = "ml",
             missing: str = "fiml", invariance: str = "scalar", information: str = "observed",
             bootstrap: int = 0, seed: int = 20260, group_order: Optional[Sequence[Any]] = None) -> SEMResult:
    sem = SEMModel(df, model, group=group, estimator=estimator, missing=missing, invariance=invariance,
                   information=information, group_order=group_order)
    fit = sem.fit()
    notes = list(sem.notes)
    if not fit.converged:
        notes.append("수렴 기준 미충족: 추정치 해석 주의 (시작값·모형 식별 확인)")
    theta = fit.theta
    mom = sem.ram.moments(theta)
    q = len(theta)
    if information == "expected":
        Info = expected_info(mom, sem.data)
    else:
        lh = getattr(sem, "last_hessian", None)
        if lh is not None and np.max(np.abs(lh[0] - theta) / (1.0 + np.abs(theta))) < 1e-6:
            Info = -lh[1]
        else:
            Info = -numeric_hessian(sem.grad_ll, theta)
    identified = True
    try:
        ev = np.linalg.eigvalsh(Info)
        if ev.min() <= 1e-8 * max(ev.max(), 1.0):
            identified = False
    except np.linalg.LinAlgError:
        identified = False
    if not identified:
        V = np.full((q, q), np.nan)
        w, U = np.linalg.eigh(Info)
        worst = int(np.argmax(np.abs(U[:, 0])))
        bad = [r for r in sem.spec.rows if r.pid == worst]
        nm = f"{bad[0].lhs} {bad[0].op} {bad[0].rhs}".strip() if bad else str(worst)
        notes.append(f"모형 식별 문제: 정보행렬이 비정칙 — 표준오차 계산 불가 (관련 모수: {nm})")
    else:
        Hinv = np.linalg.inv(Info)
        if estimator == "mlr":
            Sc = casewise_scores(mom, sem.data)
            Bm = Sc.T @ Sc
            V = Hinv @ Bm @ Hinv
        else:
            V = Hinv
    boot = None
    if bootstrap and bootstrap > 0:
        boot = bootstrap_draws(sem, theta, int(bootstrap), seed)
        if boot.shape[0] >= 2:
            V = np.cov(boot.T, ddof=1) if q > 1 else np.atleast_2d(np.var(boot[:, 0], ddof=1))
            notes.append(f"붓스트랩 표준오차: 유효 표본 {boot.shape[0]}/{bootstrap}회 (사례 재표집, 집단별)")
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    fs = fit_statistics(sem, fit, Info if identified else None)
    for r in sem.spec.rows:
        if r.op == "~~" and r.lhs == r.rhs and r.free and theta[r.pid] < 0:
            notes.append(f"음의 (잔차)분산 추정: {r.lhs} = {theta[r.pid]:.3f} (부적절 해, 모형 수정 검토)")
    return SEMResult(sem, fit, V, se, fs, notes, boot, identified)


def fit_statistics(sem: SEMModel, fit: FitResult, Info: Optional[np.ndarray]) -> Dict[str, float]:
    spec = sem.spec
    mlr = sem.estimator == "mlr"
    q = spec.q
    N, G = sem.N, spec.G
    ll0 = fit.ll_joint - fit.ll_x
    h1 = h1_statistics(sem, mlr)
    ll1 = h1["ll_joint"] - fit.ll_x
    p_h1 = G * (spec.py + spec.py * (spec.py + 1) // 2 + spec.py * spec.qx)
    df = int(p_h1 - q)
    chi2 = max(2.0 * (ll1 - ll0), 0.0)
    out: Dict[str, float] = {"npar": q, "ll0": ll0, "ll1": ll1, "N": N, "G": G, "df": df, "chi2": chi2,
                             "p_h1": p_h1}
    out["aic"] = -2 * ll0 + 2 * q
    out["bic"] = -2 * ll0 + q * math.log(N)
    out["sabic"] = -2 * ll0 + q * math.log((N + 2) / 24.0)
    base = baseline_statistics(sem, mlr)
    llb, pb = base["ll"], base["npar"]
    dfb = int(p_h1 - pb)
    chi2b = max(2.0 * (ll1 - llb), 0.0)
    T, TB = chi2, chi2b
    if mlr and Info is not None:
        mom = sem.ram.moments(fit.theta)
        Sc = casewise_scores(mom, sem.data)
        c0 = _trace_HinvB(Info, Sc.T @ Sc) / q
        c1 = h1["trace"] / p_h1
        cb = base["trace"] / pb if pb else float("nan")
        out["c0"], out["c1"] = c0, c1
        if df > 0:
            cd = (p_h1 * c1 - q * c0) / (p_h1 - q)
            out["cd"] = cd
            T = chi2 / cd if cd > 0 else float("nan")
        if dfb > 0:
            cdb = (p_h1 * c1 - pb * cb) / (p_h1 - pb)
            TB = chi2b / cdb if cdb > 0 else float("nan")
            out["cdb"] = cdb
    out["chi2_test"] = T
    out["pvalue"] = float(stats.chi2.sf(T, df)) if df > 0 and np.isfinite(T) else float("nan")
    out["chi2_base"], out["df_base"] = TB, dfb
    out["p_base"] = float(stats.chi2.sf(TB, dfb)) if dfb > 0 and np.isfinite(TB) else float("nan")
    if df > 0:
        est, lo, up, pc = rmsea_ci(T, df, N, G)
    else:
        est, lo, up, pc = 0.0, 0.0, 0.0, float("nan")
    out["rmsea"], out["rmsea_lo"], out["rmsea_hi"], out["rmsea_p"] = est, lo, up, pc
    num = max(T - df, 0.0)
    den = max(T - df, TB - dfb, 0.0)
    out["cfi"] = 1.0 - num / den if den > 0 else 1.0
    if dfb > 0 and df > 0 and TB / dfb - 1 != 0:
        out["tli"] = ((TB / dfb) - (T / df)) / ((TB / dfb) - 1.0)
    else:
        out["tli"] = 1.0
    out["srmr"] = srmr(sem, fit.theta, h1)
    out["h1"] = h1
    return out


def bootstrap_draws(sem: SEMModel, theta: np.ndarray, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = []
    df = sem.df_used
    for b in range(B):
        if sem.group:
            parts = []
            for lv in sem.group_levels:
                sub = df[df[sem.group] == lv]
                parts.append(sub.iloc[rng.integers(0, len(sub), len(sub))])
            dfb = pd.concat(parts)
        else:
            dfb = df.iloc[rng.integers(0, len(df), len(df))]
        try:
            semb = SEMModel(dfb, sem.model_text, group=sem.group, estimator="ml", missing=sem.missing,
                            invariance=sem.invariance, information=sem.information, group_order=sem.group_levels)
            fb = semb.fit(start=theta, polish=False)
            if fb.converged:
                draws.append(fb.theta)
        except Exception:
            continue
    return np.array(draws) if draws else np.zeros((0, len(theta)))


# ============================================================================ derived quantities
class Derived:
    """Standardized solutions, R², indirect effects and defined parameters as functions of theta."""

    def __init__(self, res: SEMResult):
        self.res = res
        self.spec = res.spec
        self.ram = res.sem.ram

    def _group_stats(self, theta, g):
        B, Om, BM, A, S, M = self.ram.all_moments(theta, g)
        var = np.diag(Om).copy()
        return A, S, M, var, BM

    def std_values(self, theta, kind: str) -> np.ndarray:
        spec = self.spec
        cache = [self._group_stats(theta, g) for g in range(spec.G)]
        out = np.full(len(spec.rows), np.nan)
        lat = set(spec.latent)
        xs = set(spec.x_names)
        for n, r in enumerate(spec.rows):
            A, S, M, var, BM = cache[r.group]
            val = theta[r.pid] if r.free else r.value
            sd = lambda name: math.sqrt(var[spec.idx[name]]) if var[spec.idx[name]] > 0 else float("nan")
            def scale(name, role):
                is_lat = name in lat
                if kind == "std":
                    return sd(name) if is_lat else 1.0
                if kind == "stdy" and role == "pred" and name in xs:
                    return 1.0
                return sd(name)
            if r.op == "=~":
                out[n] = val * scale(r.lhs, "pred") / scale(r.rhs, "dep")
            elif r.op == "~":
                out[n] = val * scale(r.rhs, "pred") / scale(r.lhs, "dep")
            elif r.op == "~~":
                i, j = spec.idx[r.lhs], spec.idx[r.rhs]
                if i == j:
                    out[n] = val / (scale(r.lhs, "dep") ** 2)
                else:
                    si, sj = S[i, i], S[j, j]
                    if kind == "std":
                        di = math.sqrt(si) if r.lhs in lat and si > 0 else 1.0
                        dj = math.sqrt(sj) if r.rhs in lat and sj > 0 else 1.0
                        out[n] = val / (di * dj)
                    else:
                        out[n] = val / math.sqrt(si * sj) if si > 0 and sj > 0 else float("nan")
            elif r.op == "~1":
                out[n] = val / scale(r.lhs, "dep")
        return out

    def rsquare_values(self, theta) -> List[Tuple[int, str, float]]:
        spec = self.spec
        out = []
        endo = [v for v in spec.ov + spec.latent if v in spec.dependents]
        for g in range(spec.G):
            A, S, M, var, BM = self._group_stats(theta, g)
            for v in endo:
                k = spec.idx[v]
                out.append((g, v, 1.0 - S[k, k] / var[k] if var[k] > 0 else float("nan")))
        return out

    def label_env(self, theta) -> Dict[str, float]:
        env = {lab: float(theta[pid]) for lab, pid in self.spec.label_pid.items()}
        for r in self.spec.rows:
            if r.label and not r.free:
                env[r.label] = r.value
        for name, expr in self.spec.defined:
            env[name] = _eval_expr(expr, env)
        return env

    def defined_values(self, theta) -> np.ndarray:
        env = self.label_env(theta)
        return np.array([env[name] for name, _ in self.spec.defined])

    # ---------------------------------------------------------------- indirect effects
    def _edges(self, g) -> Dict[str, List[Tuple[str, ParRow]]]:
        e: Dict[str, List[Tuple[str, ParRow]]] = {}
        for r in self.spec.rows:
            if r.group != g:
                continue
            if r.op == "~":
                e.setdefault(r.rhs, []).append((r.lhs, r))
            elif r.op == "=~":
                e.setdefault(r.lhs, []).append((r.rhs, r))
        return e

    def paths(self, g, x, y, through: Optional[Sequence[str]] = None) -> List[List[ParRow]]:
        e = self._edges(g)
        out: List[List[ParRow]] = []

        def dfs(node, visited, acc):
            if node == y and acc:
                out.append(list(acc))
                return
            if len(acc) > 12:
                return
            for nxt, row in e.get(node, []):
                if nxt in visited:
                    continue
                acc.append(row)
                dfs(nxt, visited | {nxt}, acc)
                acc.pop()

        dfs(x, {x}, [])
        if through is not None:
            seq = [x] + list(through) + [y]
            out = [p for p in out if self.path_names(p) == seq]
        return out

    @staticmethod
    def path_names(path: List[ParRow]) -> List[str]:
        first = path[0]
        names = [first.rhs if first.op == "~" else first.lhs]
        for r in path:
            names.append(r.lhs if r.op == "~" else r.rhs)
        return names

    def path_value(self, theta, path: List[ParRow]) -> float:
        v = 1.0
        for r in path:
            v *= theta[r.pid] if r.free else r.value
        return v


def parse_indirect(spec_item) -> Tuple[str, str, Optional[List[str]]]:
    """'y <- x' | 'x -> m -> y' | ('x', 'y') | ('x', 'm', 'y') -> (x, y, through)."""
    if isinstance(spec_item, (list, tuple)):
        items = [str(v).strip() for v in spec_item]
    else:
        s = str(spec_item).strip()
        if "<-" in s:
            items = [v.strip() for v in s.split("<-")][::-1]
        elif "->" in s:
            items = [v.strip() for v in s.split("->")]
        elif re.search(r"\s+IND\s+", s, re.I):
            items = [v.strip() for v in re.split(r"\s+IND\s+", s, flags=re.I)]
            items = items[1].split()[::-1] + [items[0]] if len(items) == 2 else items
        else:
            raise SEMError(f"간접효과 지정 형식 오류: '{s}' (예: 'y <- x', 'x -> m -> y')")
    if len(items) < 2:
        raise SEMError("간접효과 지정에는 시작 변수와 결과 변수가 필요")
    x, y = items[0], items[-1]
    through = items[1:-1] if len(items) > 2 else None
    return x, y, through


def numeric_jacobian(fun: Callable[[np.ndarray], np.ndarray], theta: np.ndarray, rel: float = 1e-6) -> np.ndarray:
    f0 = np.atleast_1d(fun(theta))
    J = np.zeros((len(f0), len(theta)))
    for j in range(len(theta)):
        h = rel * max(1.0, abs(theta[j]))
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        J[:, j] = (np.atleast_1d(fun(tp)) - np.atleast_1d(fun(tm))) / (2 * h)
    return J


def delta_se(fun, theta, V) -> Tuple[np.ndarray, np.ndarray]:
    val = np.atleast_1d(fun(theta))
    if not np.all(np.isfinite(V)):
        return val, np.full(len(val), np.nan)
    J = numeric_jacobian(fun, theta)
    cov = J @ V @ J.T
    return val, np.sqrt(np.clip(np.diag(cov), 0, None))


def boot_ci(est: np.ndarray, draws: np.ndarray, se: np.ndarray, ci_type: str, level: float = 0.95):
    z = stats.norm.ppf(0.5 + level / 2)
    if draws is None or draws.shape[0] < 2 or ci_type == "symmetric":
        return est - z * se, est + z * se
    a = (1 - level) / 2
    if ci_type == "percentile":
        return np.percentile(draws, 100 * a, axis=0), np.percentile(draws, 100 * (1 - a), axis=0)
    lo, hi = np.empty(len(est)), np.empty(len(est))
    for k in range(len(est)):
        d = draws[:, k]
        prop = np.mean(d < est[k])
        prop = min(max(prop, 1.0 / (len(d) + 1)), 1 - 1.0 / (len(d) + 1))
        z0 = stats.norm.ppf(prop)
        lo[k] = np.percentile(d, 100 * stats.norm.cdf(2 * z0 - z))
        hi[k] = np.percentile(d, 100 * stats.norm.cdf(2 * z0 + z))
    return lo, hi


# ============================================================================== modification indices
def modification_indices(res: SEMResult, min_mi: float = 10.0) -> List[Dict[str, Any]]:
    """Score tests (expected information) for fixed parameters and released cross-group equalities."""
    sem, spec = res.sem, res.spec
    theta = res.theta
    mom = sem.ram.moments(theta)
    lat = spec.latent
    existing = {(r.group, r.mat, min(r.i, r.j) if r.mat == "S" else r.i, max(r.i, r.j) if r.mat == "S" else r.j)
                for r in spec.rows}
    cands: List[Tuple[int, str, int, int, str, str, str, str]] = []
    obs_ind = [v for v in spec.ov if v in spec.indicators]
    struct_nodes = lat + [v for v in spec.ov if v in spec.dep_reg and v not in spec.indicators]
    reg_pairs = {(r.group, r.lhs, r.rhs) for r in spec.rows if r.op == "~"}
    for g in range(spec.G):
        for f in lat:
            for y in obs_ind:
                k = (g, "A", spec.idx[y], spec.idx[f])
                if k not in existing:
                    cands.append((g, "A", spec.idx[y], spec.idx[f], f, "=~", y, "BY"))
        for a in struct_nodes:
            for b in lat:
                if a == b or (g, a, b) in reg_pairs or (g, b, a) in reg_pairs:
                    continue
                k = (g, "A", spec.idx[a], spec.idx[b])
                if k not in existing:
                    cands.append((g, "A", spec.idx[a], spec.idx[b], a, "~", b, "ON"))
        ys = [v for v in spec.ov if v in spec.dependents]
        for gi, groupvars in enumerate((ys, lat)):
            for ai, a in enumerate(groupvars):
                for b in groupvars[ai + 1:]:
                    i, j = sorted((spec.idx[a], spec.idx[b]))
                    if (g, "S", i, j) not in existing:
                        cands.append((g, "S", i, j, a, "~~", b, "WITH"))
    release = []
    if spec.G > 1:
        counts: Dict[int, int] = {}
        for r in spec.rows:
            if r.free:
                counts[r.pid] = counts.get(r.pid, 0) + 1
        for r in spec.rows:
            if r.free and counts.get(r.pid, 0) > 1:
                release.append((r.group, r.mat, r.i, r.j, r.lhs, r.op, r.rhs, "EQ"))
    allc = cands + release
    if not allc:
        return []
    I_tt = expected_info(mom, sem.data)
    try:
        I_inv = np.linalg.inv(I_tt)
    except np.linalg.LinAlgError:
        return []
    pieces = []
    for (mu, Sig, Jmu, JS), gd in zip(mom, sem.data):
        _, gmu, G, Hmm, HmS, HSS = _accumulate(mu, Sig, gd, grad=True, hess="expected")
        p = len(mu)
        pieces.append((gmu, G, -Hmm, -HSS, Jmu, JS.reshape(p * p, -1)))
    der = Derived(res)
    std_cache = [der._group_stats(theta, g) for g in range(spec.G)]
    out = []
    for c in allc:
        g, mat, i, j, lhs, op, rhs, kind = c
        jm, js = sem.ram.position_jacobian(theta, g, mat, i, j)
        gmu, G, Imm, ISS, Jmu, JSv = pieces[g]
        jsv = js.ravel()
        gc = gmu @ jm + np.sum(G * js)
        Icc = jm @ Imm @ jm + jsv @ ISS @ jsv
        Ict = Jmu.T @ (Imm @ jm) + JSv.T @ (ISS @ jsv)
        Icc_c = Icc - Ict @ I_inv @ Ict
        if Icc_c <= 1e-10:
            continue
        mi = gc * gc / Icc_c
        epc = gc / Icc_c
        A, S, M, var, BM = std_cache[g]
        sdv = lambda k: math.sqrt(var[k]) if var[k] > 0 else float("nan")
        if mat == "A":
            std_epc = epc * sdv(j) / sdv(i)
        elif mat == "S":
            std_epc = epc / math.sqrt(S[i, i] * S[j, j]) if S[i, i] > 0 and S[j, j] > 0 else float("nan")
        else:
            std_epc = epc / sdv(i)
        if mi >= min_mi:
            out.append({"group": g, "lhs": lhs, "op": op, "rhs": rhs, "kind": kind, "mi": float(mi),
                        "epc": float(epc), "std_epc": float(std_epc)})
    out.sort(key=lambda d: -d["mi"])
    return out
