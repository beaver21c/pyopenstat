"""
Computation sources of each analysis: the library that computes it, or the published method it follows.

One table serves two purposes:
- `attach(res)` appends a 「계산 방식」 note to the first results table of an analysis (registry.dispatch_analysis), unless
  the analysis already wrote one;
- `rows()` lists the same entries for the 「구현 근거」 table of the user guide (docs/user_guide.html, chapter 25-1).

Entries name what computes the result. A "라이브러리" entry means an established package computes the estimates; a
"계산 방식" entry means no Python library computes the quantity, so the published method (paper and, where it exists,
the Stata / Mplus manual formula) is coded with numpy / scipy. Every entry lists the program it was checked against
and the test file that holds the reference values.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

# id -> (기능, 계산 함수(options) -> str, 대조, 테스트 파일)
Entry = Tuple[str, Callable[[Dict[str, Any]], str], str, str]


def _panel(o: Dict[str, Any]) -> str:
    mt = o.get("modelType", "fixed")
    base = {
        "fixed": "라이브러리: linearmodels PanelOLS (집단 내 추정); R²·sigma_u·corr(u_i, Xb) 정의는 Stata [XT] xtreg Methods and formulas",
        "twoway": "라이브러리: linearmodels PanelOLS + 시점 더미; 시점 효과 결합검정은 Wald 검정 (Stata testparm 제약 선택 규칙)",
        "random": "라이브러리: linearmodels RandomEffects; 분산성분은 Stata [XT] xtreg 기본 일치 추정식 (sa 옵션: Swamy & Arora 1972)",
        "cre": "계산 방식: Mundlak (1978)·Wooldridge (2019) 상관 확률효과 (Stata [XT] xtreg, cre)",
        "between": "라이브러리: linearmodels BetweenOLS",
        "fd": "라이브러리: linearmodels PooledOLS (1차 차분 자료)",
        "pooled": "라이브러리: linearmodels PooledOLS",
        "pa": "라이브러리: statsmodels GEE (Liang & Zeger 1986); AR(1) 작업 상관은 Stata xtgee 적률 추정식",
    }.get(mt, "")
    if o.get("covType") == "dk":
        base += "; Driscoll & Kraay (1998) 표준오차, Hoechle (2007) xtscc 소표본 보정"
    return base


def _sem(o: Dict[str, Any]) -> str:
    est = str(o.get("estimator") or "auto").lower()
    cat = bool(o.get("categorical"))
    if cat or est in ("wlsmv", "wlsm", "wls", "ulsmv", "uls"):
        return ("계산 방식: Muthén (1984) 3단계 추정 (Olsson 1979 다분상관, Olsson, Drasgow & Dorans 1982 다계열상관), "
                "Muthén, du Toit & Spisic (1997) WLSMV, Asparouhov & Muthén (2010) 보정 χ², Asparouhov & Muthén (2018) "
                "SRMR — Mplus 사용자 설명서 기본값")
    return ("계산 방식: Mplus 사용자 설명서(8판 16–18장) 기본값의 최대우도 (FIML), MLR은 Yuan & Bentler (2000), MLM은 "
            "Satorra & Bentler (1994), SRMR은 Asparouhov & Muthén (2018)")


def _count(o: Dict[str, Any]) -> str:
    fam = str(o.get("family") or "poisson")
    lib = "라이브러리: statsmodels 계수 모형 (Poisson·NegativeBinomialP·ZeroInflated·Truncated), OIM 표준오차"
    if fam.startswith("hurdle"):
        lib += "; 장애물 모형 Mullahy (1986)"
    return lib


def _mixed(o: Dict[str, Any]) -> str:
    if str(o.get("family", "gaussian")) in ("binomial", "poisson"):
        return ("계산 방식: 라플라스 근사 우도 (Stata [ME] Methods and formulas, lme4 glmer nAGQ = 1; Bates, Mächler, "
                "Bolker & Walker 2015)")
    return ("라이브러리: statsmodels MixedLM (지원 구조) / 계산 방식: lme4 프로파일 우도 정식화 (Bates, Mächler, Bolker & "
            "Walker 2015) — 결과 표 '계산' 항목 참조")


ENTRIES: Dict[str, Entry] = {
    "panel": ("패널 회귀 (xtreg fe·re·cre·be·fd·pa)", _panel,
              "Stata 19.5 xtreg·regress·xtgee·hausman·xttest0·testparm·predict, R plm",
              "tests/test_panel_extended.py, tests/test_panel_postestimation.py"),
    "dynPanel": ("동적 패널 GMM (xtabond·xtdpdsys)",
                 lambda o: ("계산 방식: Arellano & Bond (1991) 차분 GMM, Arellano & Bover (1995)·Blundell & Bond (1998) "
                            "시스템 GMM, Windmeijer (2005) 보정 — Stata [XT] xtdpd Methods and formulas의 도구변수 구성"),
                 "Stata 19.5 xtabond·xtdpdsys·estat abond·estat sargan, R plm::pgmm (collapse)", "tests/test_panel_extended.py"),
    "panelIV": ("패널 도구변수 (xtivreg)",
                lambda o: ("라이브러리: linearmodels IV2SLS (변환 자료); 계산 방식: Balestra & Varadharajan-Krishnakumar (1987) "
                           "G2SLS, Baltagi (1981) EC2SLS, Swamy & Arora (1972)·Baltagi & Chang (1994) 분산성분"),
                "Stata 19.5 xtivreg fe·re·fd·be", "tests/test_iv_timeseries.py"),
    "ivReg": ("도구변수 회귀 (ivregress)",
              lambda o: ("라이브러리: linearmodels IV2SLS·IVLIML·IVGMM; 계산 방식: Cragg & Donald (1993) 최소 고유값 통계량, "
                         "Stock & Yogo (2005) 임계값, Shea (1997) 편 R²"),
              "Stata 19.5 ivregress·estat firststage·endogenous·overid", "tests/test_iv_timeseries.py"),
    "arima": ("ARIMA·ARMAX (arima)", lambda o: "라이브러리: statsmodels SARIMAX (칼만 필터 정확 우도); Stata [TS] arima 모수화",
              "Stata 19.5 arima", "tests/test_iv_timeseries.py"),
    "unitRoot": ("단위근 검정 (dfuller·pperron·KPSS)",
                 lambda o: ("라이브러리: arch ADF·PhillipsPerron·KPSS (MacKinnon p값); 계산 방식: Fuller (1976) 임계값 보간 "
                            "(Stata dfuller)"),
                 "Stata 19.5 dfuller·pperron", "tests/test_iv_timeseries.py"),
    "corrgram": ("자기상관 (corrgram·wntestq)", lambda o: "라이브러리: statsmodels acf·pacf·acorr_ljungbox (Ljung & Box 1978)",
                 "Stata 19.5 corrgram·wntestq", "tests/test_iv_timeseries.py"),
    "countReg": ("가산자료 회귀 (poisson·nbreg·zip·zinb·tpoisson·tnbreg)", _count,
                 "Stata 19.5 poisson·nbreg·zip·zinb·tpoisson·tnbreg, R pscl::hurdle", "tests/test_count_models.py"),
    "binaryReg": ("이항 선택 (logit·probit·cloglog)",
                  lambda o: "라이브러리: statsmodels Logit·Probit·GLM(CLogLog), 한계효과 get_margeff (Stata margins 관례)",
                  "Stata 19.5 logit·probit·cloglog·margins·estat", "tests/test_limdep_models.py"),
    "ordinalReg": ("순서형 선택 (ologit·oprobit)",
                   lambda o: "라이브러리: statsmodels OrderedModel; 계산 방식: Brant (1990) 평행선 검정",
                   "Stata 19.5 ologit·oprobit, brant 공식", "tests/test_limdep_models.py"),
    "multinomReg": ("다항 로짓 (mlogit)", lambda o: "라이브러리: statsmodels MNLogit", "Stata 19.5 mlogit·margins",
                    "tests/test_limdep_models.py"),
    "tobitReg": ("토빗 (tobit)",
                 lambda o: "계산 방식: Tobin (1958) 중도절단 정규 우도 (Stata [R] tobit), statsmodels GenericLikelihoodModel 최적화",
                 "Stata 19.5 tobit", "tests/test_limdep_models.py"),
    "truncReg": ("절단 회귀 (truncreg)",
                 lambda o: "계산 방식: Hausman & Wise (1977) 절단 정규 우도 (Stata [R] truncreg), statsmodels GenericLikelihoodModel",
                 "Stata 19.5 truncreg", "tests/test_limdep_models.py"),
    "heckman": ("표본선택 (heckman)",
                lambda o: "계산 방식: Heckman (1979) 최대우도·2단계 추정 (Stata [R] heckman Methods and formulas)",
                "Stata 19.5 heckman, heckman twostep", "tests/test_limdep_models.py"),
    "heckprobit": ("표본선택 프로빗 (heckprobit)",
                   lambda o: ("계산 방식: Van de Ven & Van Praag (1981) 우도 (Stata [R] heckprobit), Genz (2004) 이변량 정규 "
                              "분포함수"),
                   "Stata 19.5 heckprobit", "tests/test_limdep_models.py"),
    "mixedModel": ("다층모형 (mixed·melogit·mepoisson)", _mixed, "Stata 19.5 mixed·melogit·mepoisson, R lme4",
                   "tests/test_multilevel.py"),
    "sem": ("구조방정식 (ML·MLR·MLM·WLSMV)", _sem,
            "Mplus 8.8 사용자 설명서 공개 예제 22개 출력, lavaan 0.7.2", "tests/test_sem_mplus_examples.py, tests/test_sem_categorical.py"),
    "lca": ("잠재계층·잠재프로파일 (LCA·LPA)",
            lambda o: ("계산 방식: EM 최대우도 (Mplus TYPE = MIXTURE 기본값: STARTS 20 4, MLR), Lo, Mendell & Rubin (2001) "
                       "LMR 검정, Vermunt (2010)·Bakk, Tekle & Vermunt (2013) 3단계 보조모형"),
            "Mplus 사용자 설명서 예제·공식 문서", "tests/test_mixture.py"),
    "lta": ("잠재전이분석 (LTA)", lambda o: "계산 방식: 1차 마르코프 잠재전이 EM 최대우도 (Mplus TYPE = MIXTURE 기본값)",
            "Mplus 사용자 설명서", "tests/test_mixture.py"),
    "lcga": ("잠재계층 성장분석 (LCGA)", lambda o: "계산 방식: 성장요인 분산 0인 혼합 성장모형 EM 최대우도 (Mplus TYPE = MIXTURE 기본값)",
             "Mplus 사용자 설명서", "tests/test_mixture.py"),
}

def _did(o: Dict[str, Any]) -> str:
    m = o.get("method", "twfe")
    if m == "hdid":
        if o.get("estimator") == "twfe":
            return ("계산 방식: Wooldridge (2025) 확장 이원 고정효과 (코호트×시점 셀, statsmodels OLS) — Stata [CAUSAL] "
                    "xthdidregress Methods and formulas")
        return ("계산 방식: Callaway & Sant'Anna (2021) ATT(g,t)·집계, Sant'Anna & Zhao (2020) 영향함수 (statsmodels OLS·Logit) "
                "— Stata [CAUSAL] xthdidregress Methods and formulas")
    if m == "event":
        return "라이브러리: linearmodels PanelOLS (개체 효과) + 시점·상대 시점 더미, group() 군집 표준오차 (사건연구)"
    return "라이브러리: linearmodels PanelOLS (개체 효과) + 시점 더미 (Stata [CAUSAL] didregress Methods and formulas)"


# result-note entries whose rows the user guide table lists separately (finer rows per option)
ENTRIES.update({
    "panelDescribe": ("패널 자료 요약", lambda o: "계산 방식: Stata [XT] xtdescribe·xtsum·xttab·xttrans 정의 (pandas 집단 연산)",
                      "Stata 19.5 xtdescribe·xtsum·xttab·xttrans", "tests/test_panel_postestimation.py"),
    "panelDiD": ("패널 이중차분", _did, "Stata 19.5 xtdidregress·xthdidregress", "tests/test_panel_did.py"),
})
_DOC_SKIP = {"panelDescribe", "panelDiD"}

_MARKERS = ("계산 방식", "라이브러리:", "라이브러리 :")


def note_for(name: str, options: Dict[str, Any]) -> Optional[str]:
    e = ENTRIES.get(name)
    if e is None:
        return None
    try:
        txt = e[1](options or {})
    except Exception:       # a source note must never break an analysis
        return None
    return txt or None


def attach(res) -> None:
    """Add the 「계산 방식」 note to the first table unless the analysis already states its computation."""
    tables = getattr(res, "tables", None) or []
    if not tables:
        return
    if any(any(str(n).startswith(_MARKERS) for n in (t.notes or [])) for t in tables):
        return
    txt = note_for(getattr(res, "name", ""), getattr(res, "options", {}) or {})
    if txt:
        tables[0].add_note(txt)


def rows() -> List[Dict[str, str]]:
    """Entries for the user guide 「구현 근거」 table (default options)."""
    out = []
    for key, (lab, fn, check, test) in ENTRIES.items():
        if key in _DOC_SKIP:
            continue
        try:
            txt = fn({})
        except Exception:
            txt = ""
        out.append({"id": key, "기능": lab, "계산": txt, "대조": check, "테스트": test})
    return out
