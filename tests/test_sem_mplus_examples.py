"""
SEM engine against published Mplus 8.8 output: Mplus User's Guide examples (statmodel.com/usersguide, chapters 3, 5
and 6 — path analysis, CFA/SEM and growth models with continuous and categorical outcomes; ML and WLSMV with delta
and theta parameterizations, scale factors and categorical growth models).

Each example page holds the Mplus input and its Mplus 8.8 output; the data file is the one used for that output.
The files are not part of the repository — fetch them first:

    python scripts/fetch_mplus_examples.py          # -> tests/data/mplus_ug/ (ignored by git)

The tests are skipped when the files are missing. Tolerances follow Mplus's printed precision (3 decimals) and its
convergence criterion (0.5D-04 on the derivatives: estimates can differ in the third decimal).
"""
import html
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "reconstructed"))
warnings.filterwarnings("ignore")

from pyjamovi.analyses import sem_engine as se  # noqa: E402

DATA = os.path.join(ROOT_DIR, "tests", "data", "mplus_ug")


def _rng(prefix, a, b):
    return [f"{prefix}{k}" for k in range(a, b + 1)]


def _growth(names, scores, fac):
    return f"{fac} =~ " + " + ".join(f"{s}*{n}" for n, s in zip(names, scores))


def _zero_int(names):
    return "\n".join(f"{v} ~ 0*1" for v in names)


def _cat_growth(U, scores, par, nth=1):
    """Mplus defaults of 'i s | u@…' with categorical outcomes (User's Guide ex6.4/6.5): thresholds equal over time,
    intercept-factor mean 0, delta: scale factors free from the 2nd time point, theta: y* residual variances free."""
    m = _growth(U, [1] * len(U), "i") + "\n" + _growth(U, scores, "s") + "\n"
    m += "\n".join(f"{u} | " + " + ".join(f"th{k}*t{k}" for k in range(1, nth + 1)) for u in U) + "\ni ~ 0*1\ns ~ 1\n"
    return m + "\n".join((f"{u} ~*~ {u}" if par == "delta" else f"{u} ~~ {u}") for u in U[1:])


Y4 = ["y11", "y12", "y13", "y14"]
U4 = ["u11", "u12", "u13", "u14"]
I9 = ["11", "21", "31", "12", "22", "32", "13", "23", "33"]
PATH = "1 + 2 ~ x1 + x2 + x3\n3 ~ 1 + 2 + x2"
FAC4 = "f1 =~ y1+y2+y3\nf2 =~ y4+y5+y6\nf3 =~ y7+y8+y9\nf4 =~ y10+y11+y12\n"
MI_LGM = ("f1 =~ {p}11 + l1*{p}21 + l2*{p}31\nf2 =~ {p}12 + l1*{p}22 + l2*{p}32\nf3 =~ {p}13 + l1*{p}23 + l2*{p}33\n"
          "i =~ 1*f1 + 1*f2 + 1*f3\ns =~ 0*f1 + 1*f2 + 2*f3\nf1 ~ 0*1\nf2 ~ 0*1\nf3 ~ 0*1\ni ~ 0*1\ns ~ 1\n")

# example: (column names, model, categorical, parameterization)
CASES = {
    "3.11": (_rng("y", 1, 3) + _rng("x", 1, 3), re.sub(r"\b(\d)\b(?!\*)", r"y\1", PATH), None, "delta"),
    "3.12": (_rng("u", 1, 3) + _rng("x", 1, 3), re.sub(r"\b(\d)\b", r"u\1", PATH), _rng("u", 1, 3), "delta"),
    "3.13": (_rng("u", 1, 3) + _rng("x", 1, 3), re.sub(r"\b(\d)\b", r"u\1", PATH), _rng("u", 1, 3), "theta"),
    "3.14": (["y1", "y2", "u1"] + _rng("x", 1, 3), "y1 + y2 ~ x1 + x2 + x3\nu1 ~ y1 + y2 + x2", ["u1"], "delta"),
    "5.1": (_rng("y", 1, 6), "f1 =~ y1 + y2 + y3\nf2 =~ y4 + y5 + y6", None, "delta"),
    "5.2": (_rng("u", 1, 6), "f1 =~ u1 + u2 + u3\nf2 =~ u4 + u5 + u6", _rng("u", 1, 6), "delta"),
    "5.3": (_rng("u", 1, 3) + _rng("y", 4, 6), "f1 =~ u1 + u2 + u3\nf2 =~ y4 + y5 + y6", _rng("u", 1, 3), "delta"),
    "5.6": (_rng("y", 1, 12), FAC4 + "f5 =~ f1 + f2 + f3 + f4", None, "delta"),
    "5.8": (_rng("y", 1, 6) + _rng("x", 1, 3), "f1 =~ y1+y2+y3\nf2 =~ y4+y5+y6\nf1 + f2 ~ x1 + x2 + x3", None, "delta"),
    "5.9": (["y1a", "y1b", "y1c", "y2a", "y2b", "y2c"],
            "f1 =~ y1a + 1*y1b + 1*y1c\nf2 =~ y2a + 1*y2b + 1*y2c\n"
            "y1a ~ a*1\ny1b ~ a*1\ny1c ~ a*1\ny2a ~ b*1\ny2b ~ b*1\ny2c ~ b*1", None, "delta"),
    "5.10": (["u1a", "u1b", "u1c", "u2a", "u2b", "u2c"],
             "f1 =~ u1a + 1*u1b + 1*u1c\nf2 =~ u2a + 1*u2b + 1*u2c\n"
             "u1a | a*t1\nu1b | a*t1\nu1c | a*t1\nu2a | b*t1\nu2b | b*t1\nu2c | b*t1",
             ["u1a", "u1b", "u1c", "u2a", "u2b", "u2c"], "delta"),
    "5.11": (_rng("y", 1, 12), FAC4 + "f4 ~ f3\nf3 ~ f1 + f2", None, "delta"),
    "5.12": (_rng("y", 1, 12), FAC4 + "f4 ~ b*f3\nf3 ~ a*f1 + f2\nind := a*b", None, "delta"),
    "6.1": (Y4, _growth(Y4, [1] * 4, "i") + "\n" + _growth(Y4, [0, 1, 2, 3], "s") + "\n" + _zero_int(Y4)
            + "\ni ~ 1\ns ~ 1", None, "delta"),
    "6.9": (Y4, _growth(Y4, [1] * 4, "i") + "\n" + _growth(Y4, [0, 1, 2, 3], "s") + "\n" + _growth(Y4, [0, 1, 4, 9], "q")
            + "\n" + _zero_int(Y4) + "\ni ~ 1\ns ~ 1\nq ~ 1", None, "delta"),
    "6.10": (Y4 + ["x1", "x2"] + _rng("a3", 1, 4),
             _growth(Y4, [1] * 4, "i") + "\n" + _growth(Y4, [0, 1, 2, 3], "s") + "\n" + _zero_int(Y4)
             + "\ni ~ 1\ns ~ 1\ni + s ~ x1 + x2\ny11 ~ a31\ny12 ~ a32\ny13 ~ a33\ny14 ~ a34", None, "delta"),
    "6.11": (_rng("y", 1, 5), "i =~ 1*y1+1*y2+1*y3+1*y4+1*y5\ns1 =~ 0*y1+1*y2+2*y3+2*y4+2*y5\n"
             "s2 =~ 0*y1+0*y2+0*y3+1*y4+2*y5\n" + _zero_int(_rng("y", 1, 5)) + "\ni ~ 1\ns1 ~ 1\ns2 ~ 1", None, "delta"),
    "6.13": (Y4 + ["y21", "y22", "y23", "y24"],
             "\n".join([_growth(Y4, [1] * 4, "i1"), _growth(Y4, [0, 1, 2, 3], "s1"),
                        _growth(["y21", "y22", "y23", "y24"], [1] * 4, "i2"),
                        _growth(["y21", "y22", "y23", "y24"], [0, 1, 2, 3], "s2"),
                        _zero_int(Y4 + ["y21", "y22", "y23", "y24"])])
             + "\ni1 ~ 1\ns1 ~ 1\ni2 ~ 1\ns2 ~ 1\ns1 ~ i2\ns2 ~ i1", None, "delta"),
    "6.14": (["y" + k for k in I9], MI_LGM.format(p="y")
             + "y11 ~ c3*1\ny12 ~ c3*1\ny13 ~ c3*1\ny21 ~ c4*1\ny22 ~ c4*1\ny23 ~ c4*1\n"
             "y31 ~ c5*1\ny32 ~ c5*1\ny33 ~ c5*1", None, "delta"),
    "6.4": (U4, _cat_growth(U4, [0, 1, 2, 3], "delta"), U4, "delta"),
    "6.5": (U4, _cat_growth(U4, [0, 1, 2, 3], "theta", nth=3), U4, "theta"),
    "6.15": (["u" + k for k in I9], MI_LGM.format(p="u")
             + "u11 | c3*t1\nu12 | c3*t1\nu13 | c3*t1\nu21 | c4*t1\nu22 | c4*t1\nu23 | c4*t1\n"
             "u31 | c5*t1\nu32 | c5*t1\nu33 | c5*t1\n"
             + "\n".join(f"u{k} ~*~ u{k}" for k in ["12", "22", "32", "13", "23", "33"]),
             ["u" + k for k in I9], "delta"),
}


# ------------------------------------------------------------------------------------- Mplus output parsing
def _text(path):
    # strip tags only (Mplus output contains '<=' that a naive tag regex would eat)
    return html.unescape(re.sub(r"</?[A-Za-z!][^<>]*>", "", open(path, encoding="latin-1").read()))


def _num(pat, txt):
    m = re.search(pat, txt, re.S)
    return float(m.group(1).replace("D", "E")) if m else None


def _block(t, head, n=500):
    i = t.find(head)
    return t[i:i + n] if i >= 0 else ""


def mplus_fit(t):
    chi = _block(t, "Chi-Square Test of Model Fit\n")
    base = _block(t, "Chi-Square Test of Model Fit for the Baseline Model", 300)
    rm = _block(t, "RMSEA (Root", 400)
    return {
        "npar": _num(r"Number of Free Parameters\s+(\d+)", t),
        "chi2": _num(r"Value\s+([\d.]+)", chi), "df": _num(r"Degrees of Freedom\s+(\d+)", chi),
        "rmsea": _num(r"Estimate\s+([\d.]+)", rm), "rmsea_p": _num(r"Probability RMSEA <= .05\s+([\d.]+)", rm),
        "cfi": _num(r"CFI\s+([\d.]+)", _block(t, "CFI/TLI", 200)), "tli": _num(r"TLI\s+([\d.]+)", _block(t, "CFI/TLI", 200)),
        "chi2_base": _num(r"Value\s+([\d.]+)", base), "srmr": _num(r"Value\s+([\d.]+)", _block(t, "SRMR (", 200)),
        "ll0": _num(r"H0 Value\s+(-?[\d.]+)", _block(t, "Loglikelihood", 300)),
        "fmin": _num(r"Weighted Least-Squares Estimator\s+Value\s+(\S+)", t),
    }


SECTIONS = {"Means": "~1", "Intercepts": "~1", "Thresholds": "|", "Variances": "~~", "Residual Variances": "~~",
            "Scales": "~*~"}


def mplus_params(t):
    i = t.find("\nMODEL RESULTS")
    blk = t[i + 14:]
    end = re.search(r"\n(STANDARDIZED MODEL RESULTS|R-SQUARE|QUALITY OF NUMERICAL|TECHNICAL|TOTAL, TOTAL INDIRECT)", blk)
    blk = blk[:end.start()] if end else blk
    out, head = {}, None
    for line in blk.splitlines():
        s = line.strip()
        m = re.match(r"^(\S+)\s+(BY|ON|WITH|\|)$", s)
        if m:
            head = (m.group(1).lower(), m.group(2))
            continue
        if s in SECTIONS or s == "New/Additional Parameters":
            head = (s, None)
            continue
        m = re.match(r"^(\S+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)$", s)
        if not (m and head):
            continue
        name, est, sev = m.group(1).lower(), float(m.group(2)), float(m.group(3))
        h, op = head
        if op in ("BY", "|"):
            key = (h, "=~", name)
        elif op == "ON":
            key = (h, "~", name)
        elif op == "WITH":
            key = tuple([min(h, name), "~~", max(h, name)])
        elif h == "Thresholds":
            v, k = name.split("$")
            key = (v, "|", "t" + k)
        elif h in SECTIONS:
            key = (name, SECTIONS[h], "" if SECTIONS[h] == "~1" else name)
        else:
            continue
        out[key] = (est, sev)
    return out


def ours(res):
    out = {}
    for r in res.spec.rows:
        if r.op not in ("=~", "~", "~~", "~1", "|", "~*~"):
            continue
        a, b = r.lhs.lower(), r.rhs.lower()
        key = (min(a, b), "~~", max(a, b)) if (r.op == "~~" and a != b) else (a, r.op, b)
        out[key] = (res.theta[r.pid] if r.free else r.value, res.se[r.pid] if r.free else 0.0)
    return out


_CACHE = {}


def _run(ex):
    if ex not in _CACHE:
        page, dat = os.path.join(DATA, f"ex{ex}.html"), os.path.join(DATA, f"ex{ex}.dat")
        if not (os.path.exists(page) and os.path.exists(dat)):
            pytest.skip("Mplus User's Guide example files missing — run scripts/fetch_mplus_examples.py")
        names, model, cat, par = CASES[ex]
        df = pd.read_csv(dat, sep=r"\s+", header=None, names=names)
        kw = dict(categorical=cat, estimator="wlsmv", parameterization=par) if cat else {}
        _CACHE[ex] = (se.estimate(df, model, **kw), _text(page))
    return _CACHE[ex]


@pytest.mark.parametrize("ex", list(CASES))
def test_parameters_match_mplus_output(ex):
    res, t = _run(ex)
    assert res.fit.converged
    mp, oo = mplus_params(t), ours(res)
    assert mp, "no MODEL RESULTS parsed"
    for key, (est, sev) in mp.items():
        assert key in oo, (ex, key)
        e, s_ = oo[key]
        assert abs(e - est) <= 8e-4, (ex, key, e, est)            # Mplus convergence criterion 0.5D-04
        assert abs(s_ - sev) <= 6e-4, (ex, key, s_, sev)          # printed with 3 decimals
    free_ours = {k for k in oo if any(r.free for r in res.spec.rows
                                      if (r.lhs.lower(), r.op, r.rhs.lower()) in (k, (k[2], k[1], k[0])))}
    assert free_ours <= set(mp) | {(k[2], k[1], k[0]) for k in mp}, (ex, free_ours - set(mp))


@pytest.mark.parametrize("ex", list(CASES))
def test_fit_matches_mplus_output(ex):
    res, t = _run(ex)
    mf, fs = mplus_fit(t), res.fitstats
    assert fs["npar"] == mf["npar"] and fs["df"] == mf["df"]
    assert abs(fs["chi2_test"] - mf["chi2"]) <= 1.5e-3, (ex, fs["chi2_test"], mf["chi2"])
    for k in ("rmsea", "cfi", "tli", "srmr"):        # TLI truncated at 1; SRMR: Mplus 8.1 definition (also WLSMV)
        assert abs(fs[k] - mf[k]) <= 6e-4, (ex, k, fs[k], mf[k])
    assert abs(fs["rmsea_p"] - mf["rmsea_p"]) <= 6e-4, (ex, "rmsea_p", fs["rmsea_p"], mf["rmsea_p"])  # N, not N − 1
    assert abs(fs["chi2_base"] - mf["chi2_base"]) <= max(1.5e-3, 2e-4 * mf["chi2_base"]), (ex, fs["chi2_base"])
    if mf["ll0"] is not None:
        assert abs(fs["ll0"] - mf["ll0"]) <= 1.5e-3
    if mf["fmin"] is not None:                   # Mplus F = ½(s − σ)'W(s − σ)
        assert abs(fs["fmin"] / 2 - mf["fmin"]) <= 2e-3 * mf["fmin"] + 1e-9


def test_scale_factor_rsquare_matches_mplus_ex6_4():
    """Delta parameterization with free scale factors: R² and residual variances of y* (Mplus R-SQUARE section)."""
    res, t = _run("6.4")
    blk = t[t.find("\nR-SQUARE"):]
    mp = {m.group(1).lower(): (float(m.group(2)), float(m.group(3)))
          for m in re.finditer(r"\n\s+(U1\d)\s+([\d.]+)\s+([\d.]+)", blk)}
    der = se.Derived(res)
    r2 = {v.lower(): val for g, v, val in der.rsquare_values(res.theta)}
    A, S, M = res.sem.ram.matrices(res.theta, 0)
    for v, (r2m, resm) in mp.items():
        k = res.spec.idx[v]
        assert abs(r2[v] - r2m) <= 1e-3, (v, r2[v], r2m)
        assert abs(S[k, k] - resm) <= 1.5e-3, (v, S[k, k], resm)
