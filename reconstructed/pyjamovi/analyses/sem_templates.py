"""
SEM model templates — ready-made longitudinal models with their conventional default assumptions written in.

The user picks a model type, assigns variables per wave, and receives a complete model string for the existing
SEM analysis (`sem_engine` / `sem_module`; lavaan-style operators, Mplus defaults) together with the list of
assumptions the string encodes. Every assumption that differs from the engine defaults is written explicitly
(observed intercepts fixed at 0 in growth models, measurement-error variances fixed at 0 in the RI-CLPM,
covariances the model excludes fixed at 0, ...), so the same string also runs unchanged in
`lavaan::sem(model, data, missing = "fiml", mimic = "Mplus")`.

    from pyjamovi.analyses.sem_templates import build_template, sem_template, TEMPLATE_CATALOG
    out = build_template("riclpm", {"x_waves": ["우울_1", "우울_2", "우울_3", "우울_4", "우울_5"],
                                    "y_waves": ["외로움_1", "외로움_2", "외로움_3", "외로움_4", "외로움_5"]},
                         {"constrain_lagged": True})
    out["model"], out["assumptions"], out["suggested"]["df"]

Templates
    clpm           교차지연 패널모형 (CLPM)
    riclpm         무선절편 교차지연 패널모형 (RI-CLPM; Hamaker, Kuiper & Grasman, 2015)
    latent_clpm    다중지표 잠재 교차지연모형 (종단 측정동일성 포함)
    lgm_linear     선형 잠재성장모형
    lgm_quadratic  2차 잠재성장모형
    lgm_basis      잠재기저(자유 시간점수) 성장모형
    lgm_conditional 조건부 잠재성장모형 (시간불변·시간가변 공변량)
    lgm_parallel   병행과정 잠재성장모형
    lgm_second_order 2차 요인 잠재성장모형 (요인 곡선 모형, curve-of-factors)
    lcs_univariate 단변량 잠재변화점수 모형 (이중 변화점수)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .base import resolve_choice


class TemplateError(ValueError):
    """Invalid template request (Korean message for the user)."""


# ============================================================================================ references
REF = {
    "hamaker2015": "Hamaker, E. L., Kuiper, R. M., & Grasman, R. P. P. P. (2015). A critique of the cross-lagged "
                   "panel model. Psychological Methods, 20(1), 102–116.",
    "mulder2021": "Mulder, J. D., & Hamaker, E. L. (2021). Three extensions of the random intercept cross-lagged "
                  "panel model. Structural Equation Modeling, 28(4), 638–648.",
    "bollen2006": "Bollen, K. A., & Curran, P. J. (2006). Latent curve models: A structural equation perspective. "
                  "Wiley.",
    "duncan2006": "Duncan, T. E., Duncan, S. C., & Strycker, L. A. (2006). An introduction to latent variable "
                  "growth curve modeling (2nd ed.). Lawrence Erlbaum.",
    "little2013": "Little, T. D. (2013). Longitudinal structural equation modeling. Guilford Press.",
    "mcardle2009": "McArdle, J. J. (2009). Latent variable modeling of differences and changes with longitudinal "
                   "data. Annual Review of Psychology, 60, 577–605.",
    "kievit2018": "Kievit, R. A., Brandmaier, A. M., Ziegler, G., et al. (2018). Developmental cognitive "
                  "neuroscience using latent change score models: A tutorial and applications. Developmental "
                  "Cognitive Neuroscience, 33, 99–117.",
    "mplus": "Muthén, L. K., & Muthén, B. O. (1998–2017). Mplus user's guide (8th ed.). Muthén & Muthén. "
             "(Chapter 6: growth modeling examples)",
}

GROUPS = {
    "crosslag": "교차지연모형 (Cross-lagged panel)",
    "growth": "잠재성장모형 (Latent growth)",
    "change": "잠재변화점수모형 (Latent change score)",
}


# ============================================================================================ catalog
def _role(key, label, rtype, req=True, min_len=0, help_=""):
    return {"key": key, "label": label, "type": rtype, "req": req, "min": min_len, "help": help_}


def _check(key, label, default=False, help_=""):
    return {"key": key, "type": "check", "label": label, "default": default, "help": help_}


def _select(key, label, options, default, help_=""):
    return {"key": key, "type": "select", "label": label, "options": options, "default": default, "help": help_}


def _text(key, label, default="", placeholder="", help_=""):
    return {"key": key, "type": "text", "label": label, "default": default, "placeholder": placeholder,
            "help": help_}


_TIME_SCORES = _text("time_scores", "시간점수 (쉼표 구분, 비우면 0, 1, …, T−1)", "", "예: 0, 1, 2, 4",
                     "측정 간격이 다르면 실제 간격에 비례하는 값 입력")
_RESVAR_EQ = _check("residual_var_equal", "관측변수 잔차분산 시점 간 동일 제약", False)
_COMMENTS = _check("comments", "모형식에 설명 주석(# …) 포함", True)
_MODEL_COVS = _check("model_covariates", "공변량 분포를 모형에 포함 (공변량 결측 사례도 FIML로 사용)", False,
                     "기본은 Mplus처럼 공변량에 조건부 처리(공변량 결측 사례 제외)")
_CORR_RES = _select("correlated_residuals", "같은 문항 잔차의 시점 간 공분산",
                    [["all", "모든 시점 쌍 (Little, 2013 권고)"], ["adjacent", "인접 시점만"], ["none", "없음"]], "all")

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "clpm", "group": "crosslag", "group_title": GROUPS["crosslag"],
        "title": "교차지연 패널모형", "name_en": "Cross-Lagged Panel Model (CLPM)", "short": "CLPM",
        "description": "두 개 이상 변수의 시점별 관측값으로 자기회귀(안정성) 경로와 교차지연 경로를 추정하는 기본 "
                       "패널모형. 개인 간 안정적 차이를 분리하지 않으므로 개인 내 효과 해석에는 RI-CLPM 검토 필요.",
        "min_waves": 2,
        "roles": [
            _role("x_waves", "변수 X — 시점별 관측변수 (1시점부터 순서대로)", "waves", True, 2),
            _role("y_waves", "변수 Y — 시점별 관측변수 (X와 같은 시점 수)", "waves", True, 2),
            _role("z_waves", "변수 Z — 시점별 관측변수 (선택, 3변수 모형)", "waves", False, 2),
            _role("labels", "구성개념 이름 (선택, 쉼표 구분: 예 우울, 외로움)", "labels", False),
        ],
        "options": [
            _check("constrain_lagged", "자기회귀·교차지연 경로 시점 간 동일 제약 (정상성)", False),
            _check("constrain_residual_cov", "2시점 이후 잔차 공분산 시점 간 동일 제약", False),
            _check("constrain_residual_var", "2시점 이후 잔차분산 시점 간 동일 제약", False),
            _COMMENTS,
        ],
        "references": [REF["hamaker2015"], REF["little2013"]],
    },
    {
        "id": "riclpm", "group": "crosslag", "group_title": GROUPS["crosslag"],
        "title": "무선절편 교차지연 패널모형", "name_en": "Random Intercept Cross-Lagged Panel Model (RI-CLPM)",
        "short": "RI-CLPM",
        "description": "관측값을 개인 간 안정적 성분(무선절편)과 시점별 개인 내 성분으로 분해하고, 개인 내 성분 사이의 "
                       "자기회귀·교차지연 경로를 추정 (Hamaker, Kuiper & Grasman, 2015). 3시점 이상 필요.",
        "min_waves": 3,
        "roles": [
            _role("x_waves", "변수 X — 시점별 관측변수 (1시점부터 순서대로)", "waves", True, 3),
            _role("y_waves", "변수 Y — 시점별 관측변수 (X와 같은 시점 수)", "waves", True, 3),
            _role("z_waves", "변수 Z — 시점별 관측변수 (선택, 3변수 모형)", "waves", False, 3),
            _role("covariates", "시간불변 공변량 → 무선절편 (선택)", "vars", False),
            _role("labels", "구성개념 이름 (선택, 쉼표 구분)", "labels", False),
        ],
        "options": [
            _check("constrain_lagged", "개인 내 자기회귀·교차지연 경로 시점 간 동일 제약", False),
            _check("constrain_residual_cov", "2시점 이후 개인 내 잔차(혁신) 공분산 동일 제약", False),
            _check("constrain_residual_var", "2시점 이후 개인 내 잔차(혁신) 분산 동일 제약", False),
            _check("constrain_means", "시점별 평균(관측변수 절편) 시점 간 동일 제약", False),
            _MODEL_COVS,
            _COMMENTS,
        ],
        "references": [REF["hamaker2015"], REF["mulder2021"]],
    },
    {
        "id": "latent_clpm", "group": "crosslag", "group_title": GROUPS["crosslag"],
        "title": "다중지표 잠재 교차지연모형", "name_en": "Multiple Indicator Latent Cross-Lagged Panel Model",
        "short": "잠재 CLPM",
        "description": "구성개념마다 시점별로 같은 문항들로 잠재요인을 정의하고(종단 측정동일성), 같은 문항 잔차의 "
                       "시점 간 공분산을 허용한 뒤 잠재요인 사이의 자기회귀·교차지연 경로를 추정.",
        "min_waves": 2,
        "roles": [
            _role("x_indicators", "구성개념 X — 문항 × 시점 (문항별 한 줄, 시점 순서대로; 첫 문항이 기준지표)",
                  "grid", True, 2),
            _role("y_indicators", "구성개념 Y — 문항 × 시점", "grid", True, 2),
            _role("z_indicators", "구성개념 Z — 문항 × 시점 (선택)", "grid", False, 2),
            _role("labels", "구성개념 이름 (선택, 쉼표 구분)", "labels", False),
        ],
        "options": [
            _select("invariance", "종단 측정동일성 수준",
                    [["configural", "형태 동일성 (제약 없음)"], ["metric", "측정단위 동일성 (요인적재량 동일)"],
                     ["scalar", "절편 동일성 (요인적재량·절편 동일)"],
                     ["strict", "엄격 동일성 (적재량·절편·잔차분산 동일)"]], "scalar"),
            _CORR_RES,
            _check("constrain_lagged", "자기회귀·교차지연 경로 시점 간 동일 제약", False),
            _check("constrain_residual_cov", "2시점 이후 요인 잔차 공분산 동일 제약", False),
            _check("constrain_residual_var", "2시점 이후 요인 잔차분산 동일 제약", False),
            _COMMENTS,
        ],
        "references": [REF["little2013"]],
    },
    {
        "id": "lgm_linear", "group": "growth", "group_title": GROUPS["growth"],
        "title": "선형 잠재성장모형", "name_en": "Linear Latent Growth Model", "short": "선형 LGM",
        "description": "반복측정 관측변수로 절편·기울기 성장요인을 정의 (시간점수 0, 1, …). 3시점 이상 필요.",
        "min_waves": 3,
        "roles": [_role("y_waves", "반복측정 변수 — 시점별 관측변수 (1시점부터 순서대로)", "waves", True, 3),
                  _role("labels", "변수 이름 (선택)", "labels", False)],
        "options": [_TIME_SCORES, _RESVAR_EQ, _COMMENTS],
        "references": [REF["bollen2006"], REF["duncan2006"], REF["mplus"]],
    },
    {
        "id": "lgm_quadratic", "group": "growth", "group_title": GROUPS["growth"],
        "title": "2차 잠재성장모형", "name_en": "Quadratic Latent Growth Model", "short": "2차 LGM",
        "description": "절편·선형·2차 성장요인으로 곡선 변화를 추정 (2차 요인 적재량 = 시간점수²). 4시점 이상 필요.",
        "min_waves": 4,
        "roles": [_role("y_waves", "반복측정 변수 — 시점별 관측변수 (1시점부터 순서대로)", "waves", True, 4),
                  _role("labels", "변수 이름 (선택)", "labels", False)],
        "options": [_TIME_SCORES, _RESVAR_EQ, _COMMENTS],
        "references": [REF["bollen2006"], REF["duncan2006"], REF["mplus"]],
    },
    {
        "id": "lgm_basis", "group": "growth", "group_title": GROUPS["growth"],
        "title": "잠재기저 성장모형", "name_en": "Latent Basis Growth Model (estimated time scores)",
        "short": "잠재기저 LGM",
        "description": "첫 시점 0, 기준 시점 1로 고정하고 나머지 시간점수를 자유 추정하여 변화의 형태를 자료에서 추정. "
                       "3시점이면 포화(자유도 0), 4시점 이상 권장.",
        "min_waves": 3,
        "roles": [_role("y_waves", "반복측정 변수 — 시점별 관측변수 (1시점부터 순서대로)", "waves", True, 3),
                  _role("labels", "변수 이름 (선택)", "labels", False)],
        "options": [
            _select("anchor", "시간점수 1 고정 시점",
                    [["last", "마지막 시점 (기울기 평균 = 전체 변화량)"],
                     ["second", "2시점 (기울기 평균 = 첫 구간 변화량, Mplus 예제 방식)"]], "last"),
            _RESVAR_EQ, _COMMENTS,
        ],
        "references": [REF["bollen2006"], REF["duncan2006"], REF["mplus"]],
    },
    {
        "id": "lgm_conditional", "group": "growth", "group_title": GROUPS["growth"],
        "title": "조건부 잠재성장모형", "name_en": "Conditional Latent Growth Model (time-invariant and time-varying covariates)",
        "short": "조건부 LGM",
        "description": "시간불변 공변량이 절편·기울기를 예측하고, 선택적으로 시간가변 공변량이 같은 시점의 관측변수를 "
                       "예측하는 성장모형 (공변량은 조건부 처리, Mplus 기본).",
        "min_waves": 3,
        "roles": [
            _role("y_waves", "반복측정 변수 — 시점별 관측변수 (1시점부터 순서대로)", "waves", True, 3),
            _role("covariates", "시간불변 공변량 → 성장요인", "vars", False),
            _role("tvc_waves", "시간가변 공변량 — 시점별 변수 (선택, 반복측정 변수와 같은 시점 수)", "waves", False, 3),
            _role("labels", "변수 이름 (선택)", "labels", False),
        ],
        "options": [
            _select("shape", "성장 형태", [["linear", "선형"], ["quadratic", "2차 (4시점 이상)"]], "linear"),
            _TIME_SCORES,
            _check("tvc_equal", "시간가변 공변량 효과 시점 간 동일 제약", False),
            _RESVAR_EQ, _MODEL_COVS, _COMMENTS,
        ],
        "references": [REF["bollen2006"], REF["duncan2006"], REF["mplus"]],
    },
    {
        "id": "lgm_parallel", "group": "growth", "group_title": GROUPS["growth"],
        "title": "병행과정 잠재성장모형", "name_en": "Parallel Process Latent Growth Model", "short": "병행과정 LGM",
        "description": "두 변수의 선형 성장을 동시에 추정하고 성장요인 사이의 공분산 또는 절편 → 기울기 경로를 추정.",
        "min_waves": 3,
        "roles": [
            _role("x_waves", "과정 1 — 시점별 관측변수", "waves", True, 3),
            _role("y_waves", "과정 2 — 시점별 관측변수 (과정 1과 같은 시점 수)", "waves", True, 3),
            _role("labels", "과정 이름 (선택, 쉼표 구분)", "labels", False),
        ],
        "options": [
            _select("structure", "성장요인 관계",
                    [["covariance", "성장요인 4개 모두 공분산 (기본)"],
                     ["regression", "기울기 ← 두 과정 절편 회귀 (과정 간 절편 → 기울기 효과)"]], "covariance"),
            _TIME_SCORES,
            _check("wave_residual_cov", "같은 시점 두 변수 잔차 공분산 허용", False),
            _RESVAR_EQ, _COMMENTS,
        ],
        "references": [REF["bollen2006"], REF["duncan2006"], REF["mplus"]],
    },
    {
        "id": "lgm_second_order", "group": "growth", "group_title": GROUPS["growth"],
        "title": "2차 요인 잠재성장모형 (다중지표)", "name_en": "Second-Order Latent Growth Model (multiple indicators)",
        "short": "2차 요인 LGM",
        "description": "시점마다 같은 문항들로 1차 잠재요인을 정의하고(종단 측정동일성), 1차 요인들 위에 절편·기울기 "
                       "2차 성장요인을 둔 다중지표 성장모형. 측정오차를 분리한 잠재변수 수준의 변화 추정.",
        "min_waves": 3,
        "roles": [
            _role("indicators", "문항 × 시점 (문항별 한 줄, 시점 순서대로; 첫 문항이 기준지표)", "grid", True, 3),
            _role("labels", "구성개념 이름 (선택)", "labels", False),
        ],
        "options": [
            _select("shape", "성장 형태", [["linear", "선형"], ["quadratic", "2차 (4시점 이상)"]], "linear"),
            _TIME_SCORES,
            _select("invariance", "종단 측정동일성 수준",
                    [["scalar", "절편 동일성 (요인적재량·절편 동일)"],
                     ["strict", "엄격 동일성 (적재량·절편·잔차분산 동일)"]], "scalar"),
            _select("mean_identification", "성장요인 평균 식별 방식",
                    [["marker", "기준지표 절편 0 고정, 절편·기울기 평균 추정"],
                     ["intercept_zero", "모든 절편 동일(추정), 절편요인 평균 0 고정 (Mplus 예제 6.14 방식)"]], "marker"),
            _CORR_RES,
            _check("factor_resvar_equal", "1차 요인 잔차분산 시점 간 동일 제약", False),
            _COMMENTS,
        ],
        "references": [REF["bollen2006"], REF["little2013"], REF["mplus"]],
    },
    {
        "id": "lcs_univariate", "group": "change", "group_title": GROUPS["change"],
        "title": "단변량 잠재변화점수 모형", "name_en": "Univariate Latent Change Score Model (dual change score)",
        "short": "LCS",
        "description": "시점별 진점수와 인접 시점 사이의 잠재변화점수를 정의하고, 변화를 일정 변화(상수 변화 요인)와 "
                       "직전 수준에 비례하는 변화(비례 효과)로 분해 (McArdle, 2009).",
        "min_waves": 3,
        "roles": [_role("y_waves", "반복측정 변수 — 시점별 관측변수 (1시점부터 순서대로)", "waves", True, 3),
                  _role("labels", "변수 이름 (선택)", "labels", False)],
        "options": [
            _check("constant_change", "일정 변화 요인 (상수 변화, 모든 변화점수에 적재량 1)", True),
            _check("proportional", "비례 변화 효과 (직전 진점수 → 변화점수, 시점 간 동일)", True),
            _check("residual_var_equal", "관측변수 잔차분산 시점 간 동일 제약", True),
            _COMMENTS,
        ],
        "references": [REF["mcardle2009"], REF["kievit2018"]],
    },
]
_CAT = {t["id"]: t for t in TEMPLATE_CATALOG}
TEMPLATE_ALIASES = {
    "cross_lagged": "clpm", "crosslagged": "clpm", "ri_clpm": "riclpm", "ri-clpm": "riclpm",
    "random_intercept_clpm": "riclpm", "latentclpm": "latent_clpm", "mi_clpm": "latent_clpm",
    "multiple_indicator_clpm": "latent_clpm", "lgm": "lgm_linear", "linear": "lgm_linear",
    "growth": "lgm_linear", "quadratic": "lgm_quadratic", "latent_basis": "lgm_basis", "basis": "lgm_basis",
    "conditional": "lgm_conditional", "parallel": "lgm_parallel", "parallel_process": "lgm_parallel",
    "second_order": "lgm_second_order", "curve_of_factors": "lgm_second_order", "cof": "lgm_second_order",
    "lcs": "lcs_univariate", "dual_change": "lcs_univariate", "latent_change": "lcs_univariate",
}


# English UI labels (the generated model comments and assumption lists are Korean)
_EN = {
    "변수 X — 시점별 관측변수 (1시점부터 순서대로)": "Variable X — one observed variable per wave (wave 1 first)",
    "변수 Y — 시점별 관측변수 (X와 같은 시점 수)": "Variable Y — one observed variable per wave (same waves as X)",
    "변수 Z — 시점별 관측변수 (선택, 3변수 모형)": "Variable Z — one observed variable per wave (optional, 3 variables)",
    "구성개념 이름 (선택, 쉼표 구분: 예 우울, 외로움)": "Construct names (optional, comma-separated)",
    "구성개념 이름 (선택, 쉼표 구분)": "Construct names (optional, comma-separated)",
    "구성개념 이름 (선택)": "Construct name (optional)",
    "변수 이름 (선택)": "Variable name (optional)",
    "과정 이름 (선택, 쉼표 구분)": "Process names (optional, comma-separated)",
    "시간불변 공변량 → 무선절편 (선택)": "Time-invariant covariates → random intercepts (optional)",
    "구성개념 X — 문항 × 시점 (문항별 한 줄, 시점 순서대로; 첫 문항이 기준지표)":
        "Construct X — items × waves (one row per item, waves in order, first item = marker indicator)",
    "구성개념 Y — 문항 × 시점": "Construct Y — items × waves",
    "구성개념 Z — 문항 × 시점 (선택)": "Construct Z — items × waves (optional)",
    "반복측정 변수 — 시점별 관측변수 (1시점부터 순서대로)": "Repeated measure — one observed variable per wave (wave 1 first)",
    "시간불변 공변량 → 성장요인": "Time-invariant covariates → growth factors",
    "시간가변 공변량 — 시점별 변수 (선택, 반복측정 변수와 같은 시점 수)":
        "Time-varying covariate — one variable per wave (optional, same waves)",
    "과정 1 — 시점별 관측변수": "Process 1 — one observed variable per wave",
    "과정 2 — 시점별 관측변수 (과정 1과 같은 시점 수)": "Process 2 — one observed variable per wave (same waves)",
    "문항 × 시점 (문항별 한 줄, 시점 순서대로; 첫 문항이 기준지표)":
        "Items × waves (one row per item, waves in order, first item = marker indicator)",
    "자기회귀·교차지연 경로 시점 간 동일 제약 (정상성)": "Autoregressive and cross-lagged paths constrained over time (stationarity)",
    "2시점 이후 잔차 공분산 시점 간 동일 제약": "Residual covariances constrained over time from wave 2 on",
    "2시점 이후 잔차분산 시점 간 동일 제약": "Residual variances constrained over time from wave 2 on",
    "개인 내 자기회귀·교차지연 경로 시점 간 동일 제약": "Within-person autoregressive and cross-lagged paths constrained over time",
    "2시점 이후 개인 내 잔차(혁신) 공분산 동일 제약": "Covariances between within-person residuals (innovations) constrained over time from wave 2 on",
    "2시점 이후 개인 내 잔차(혁신) 분산 동일 제약": "Variances of within-person residuals (innovations) constrained over time from wave 2 on",
    "시점별 평균(관측변수 절편) 시점 간 동일 제약": "Means at each wave (observed intercepts) constrained over time",
    "공변량 분포를 모형에 포함 (공변량 결측 사례도 FIML로 사용)":
        "Model the covariate distribution (cases with missing covariates kept by FIML)",
    "모형식에 설명 주석(# …) 포함": "Explanatory comments (# …) in the model",
    "종단 측정동일성 수준": "Level of longitudinal measurement invariance",
    "같은 문항 잔차의 시점 간 공분산": "Residual covariances of the same item across waves",
    "자기회귀·교차지연 경로 시점 간 동일 제약": "Autoregressive and cross-lagged paths constrained over time",
    "2시점 이후 요인 잔차 공분산 동일 제약": "Factor residual covariances constrained over time from wave 2 on",
    "2시점 이후 요인 잔차분산 동일 제약": "Factor residual variances constrained over time from wave 2 on",
    "시간점수 (쉼표 구분, 비우면 0, 1, …, T−1)": "Time scores (comma-separated, empty = 0, 1, …, T−1)",
    "관측변수 잔차분산 시점 간 동일 제약": "Residual variances of the observed variables held equal across time",
    "시간점수 1 고정 시점": "Wave with the time score fixed at 1",
    "성장 형태": "Growth shape",
    "시간가변 공변량 효과 시점 간 동일 제약": "Effect of the time-varying covariate held equal across time",
    "성장요인 관계": "Relations among the growth factors",
    "같은 시점 두 변수 잔차 공분산 허용": "Residual covariance of the two variables at the same wave",
    "성장요인 평균 식별 방식": "Identification of the growth factor means",
    "1차 요인 잔차분산 시점 간 동일 제약": "First-order factor residual variances held equal across time",
    "일정 변화 요인 (상수 변화, 모든 변화점수에 적재량 1)": "Constant change factor (loading 1 on every latent change score)",
    "비례 변화 효과 (직전 진점수 → 변화점수, 시점 간 동일)": "Proportional change (previous true score → latent change score, equal over time)",
    # select option labels
    "모든 시점 쌍 (Little, 2013 권고)": "All pairs of waves (recommended by Little, 2013)", "인접 시점만": "Adjacent waves only",
    "없음": "None", "형태 동일성 (제약 없음)": "Configural (no constraints)",
    "측정단위 동일성 (요인적재량 동일)": "Metric (equal loadings)",
    "절편 동일성 (요인적재량·절편 동일)": "Scalar (equal loadings and intercepts)",
    "엄격 동일성 (적재량·절편·잔차분산 동일)": "Strict (equal loadings, intercepts and residual variances)",
    "마지막 시점 (기울기 평균 = 전체 변화량)": "Last wave (slope mean = total change)",
    "2시점 (기울기 평균 = 첫 구간 변화량, Mplus 예제 방식)": "Wave 2 (slope mean = change over the first interval, as in Mplus Example 6.8)",
    "선형": "Linear", "2차 (4시점 이상)": "Quadratic (4+ waves)",
    "성장요인 4개 모두 공분산 (기본)": "Covariances among all four growth factors (default)",
    "기울기 ← 두 과정 절편 회귀 (과정 간 절편 → 기울기 효과)": "Slopes regressed on the intercepts of both processes (intercept → slope effects across processes)",
    "기준지표 절편 0 고정, 절편·기울기 평균 추정": "Marker indicator intercepts fixed at 0, intercept and slope growth factor means estimated",
    "모든 절편 동일(추정), 절편요인 평균 0 고정 (Mplus 예제 6.14 방식)":
        "All intercepts equal (estimated), intercept growth factor mean fixed at 0 (as in Mplus Example 6.14)",
    "예: 0, 1, 2, 4": "e.g., 0, 1, 2, 4",
    "측정 간격이 다르면 실제 간격에 비례하는 값 입력": "With unequal spacing, enter values proportional to the actual intervals",
    "기본은 Mplus처럼 공변량에 조건부 처리(공변량 결측 사례 제외)":
        "Default (as in Mplus): model conditional on the covariates; cases with a missing covariate are excluded",
}
_DESC_EN = {
    "clpm": "Basic panel model with autoregressive (stability) and cross-lagged paths among the observed scores of two "
            "or more variables at each wave. Stable between-person differences are not separated from the "
            "within-person process, so consider the RI-CLPM when within-person effects are to be interpreted.",
    "riclpm": "Separates the observed scores into stable between-person differences (random intercepts) and "
              "within-person centered components at each wave, and estimates autoregressive and cross-lagged paths "
              "among the within-person components (Hamaker, Kuiper & Grasman, 2015). Requires 3 or more waves.",
    "latent_clpm": "A latent factor for each construct at each wave from the same items (longitudinal measurement "
                   "invariance), residual covariances of the same item across waves, and autoregressive and "
                   "cross-lagged paths among the latent factors.",
    "lgm_linear": "Intercept and slope growth factors defined by the repeated observed measures (time scores 0, 1, …). "
                  "Requires 3 or more waves.",
    "lgm_quadratic": "Curvilinear change with intercept, linear slope and quadratic slope growth factors (the quadratic "
                     "time scores are the squared linear time scores). Requires 4 or more waves.",
    "lgm_basis": "Time score of the first wave fixed at 0 and of the reference wave at 1; the remaining time scores are "
                 "estimated, so the shape of change is estimated from the data. Saturated (0 df) with 3 waves; "
                 "4 or more waves are recommended.",
    "lgm_conditional": "Growth model in which time-invariant covariates predict the intercept and slope and, optionally, "
                       "time-varying covariates predict the outcome at the same wave (model conditional on the "
                       "covariates, Mplus default).",
    "lgm_parallel": "Linear growth of two parallel processes estimated together, with covariances among the growth "
                    "factors or regressions of the slopes on the intercepts.",
    "lgm_second_order": "Multiple indicator growth model: first-order latent factors defined by the same items at each "
                        "wave (longitudinal measurement invariance) and second-order intercept and slope growth factors "
                        "on these factors (curve of factors), so change is estimated for latent variables separated "
                        "from measurement error.",
    "lcs_univariate": "A true score at each wave and latent change scores between adjacent waves; change is decomposed "
                      "into constant change (constant change factor) and proportional change (effect of the previous "
                      "true score) (McArdle, 2009).",
}
_SHORT_EN = {"clpm": "CLPM", "riclpm": "RI-CLPM", "latent_clpm": "latent CLPM", "lgm_linear": "linear LGM",
             "lgm_quadratic": "quadratic LGM", "lgm_basis": "latent basis LGM", "lgm_conditional": "conditional LGM",
             "lgm_parallel": "parallel process LGM", "lgm_second_order": "second-order LGM", "lcs_univariate": "LCS"}
_GROUPS_EN = {"crosslag": "Cross-lagged panel models", "growth": "Latent growth models",
              "change": "Latent change score models"}


def template_catalog() -> List[Dict[str, Any]]:
    """JSON-serialisable copy of the catalog (for the UI), with English labels (`label_en`, `description_en`)."""
    import copy
    cat = copy.deepcopy(TEMPLATE_CATALOG)
    for t in cat:
        t["description_en"] = _DESC_EN.get(t["id"], t["description"])
        t["short_en"] = _SHORT_EN.get(t["id"], t["short"])
        t["group_title_en"] = _GROUPS_EN.get(t["group"], t["group_title"])
        for r in t["roles"]:
            r["label_en"] = _EN.get(r["label"], r["label"])
        for o in t["options"]:
            o["label_en"] = _EN.get(o["label"], o["label"])
            if o.get("help"):
                o["help_en"] = _EN.get(o["help"], o["help"])
            if o.get("placeholder"):
                o["placeholder_en"] = _EN.get(o["placeholder"], o["placeholder"])
            if o["type"] == "select":
                o["options_en"] = [[v, _EN.get(lab, lab)] for v, lab in o["options"]]
    return cat


def _resolve_id(template_id: str) -> str:
    raw = str(template_id or "").strip()
    if not raw:
        raise TemplateError(f"모형 예시 지정 필요 (가능: {', '.join(_CAT)})")
    key = raw.lower().replace(" ", "_")
    if key in _CAT:
        return key
    if key in TEMPLATE_ALIASES:
        return TEMPLATE_ALIASES[key]
    raise TemplateError(f"알 수 없는 모형 예시 '{raw}' (가능: {', '.join(_CAT)})")


# ============================================================================================ helpers
def _snake(k: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", str(k)).lower()


def _norm_keys(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        out[_snake(k)] = v
    return out


def _flag(v: Any, default: bool) -> bool:
    if v is None or (isinstance(v, str) and not v.strip()):
        return default
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
        raise TemplateError(f"참/거짓 옵션 값 오류: '{v}'")
    return bool(v)


def _split_names(v: Any) -> List[str]:
    if v is None or v == "" or v is False:
        return []
    if isinstance(v, str):
        return [x.strip() for x in re.split(r"[,\n;]", v) if x.strip()]
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            if x is None or str(x).strip() == "":
                continue
            out.append(str(x).strip())
        return out
    return [str(v).strip()]


def _grid(v: Any, label: str) -> List[List[str]]:
    """indicator × wave grid from a list of lists (or a list of comma-separated strings)."""
    if v is None or v == "" or v is False:
        return []
    if isinstance(v, str):
        rows = [r for r in re.split(r"[\n;]", v) if r.strip()]
    else:
        rows = list(v)
    out = []
    for r in rows:
        names = _split_names(r)
        if names:
            out.append(names)
    if out and any(len(r) != len(out[0]) for r in out):
        lens = ", ".join(str(len(r)) for r in out)
        raise TemplateError(f"{label}: 문항마다 시점 수가 같아야 함 (문항별 시점 수: {lens})")
    return out


def _transpose(rows: List[List[str]]) -> List[List[str]]:
    return [list(c) for c in zip(*rows)] if rows else []


def _num(x: float) -> str:
    x = float(x)
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    return f"{x:.6g}"


def _time_scores(v: Any, T: int, label: str = "시간점수") -> List[float]:
    if v is None or v == "" or (isinstance(v, (list, tuple)) and len(v) == 0):
        return [float(t) for t in range(T)]
    if isinstance(v, str):
        toks = [x for x in re.split(r"[,\s;]+", v.strip()) if x]
    else:
        toks = list(v)
    try:
        vals = [float(x) for x in toks]
    except (TypeError, ValueError):
        raise TemplateError(f"{label}: 숫자만 입력 (입력값: {v})")
    if len(vals) != T:
        raise TemplateError(f"{label}: 시점 수({T})만큼 입력 필요 (입력 {len(vals)}개)")
    if any(not math.isfinite(x) for x in vals):
        raise TemplateError(f"{label}: 유한한 숫자만 입력")
    if len(set(vals)) < len(vals):
        raise TemplateError(f"{label}: 같은 값이 반복됨 — 시점마다 다른 값 필요")
    return vals


_SAFE = re.compile(r"[^\w]", re.UNICODE)


def _clean_label(s: str) -> str:
    s = _SAFE.sub("_", str(s).strip()).strip("_")
    if not s:
        return ""
    if s[0].isdigit():
        s = "V" + s
    return s


def _common_stem(names: Sequence[str]) -> str:
    if not names:
        return ""
    pre = names[0]
    for n in names[1:]:
        while not n.startswith(pre):
            pre = pre[:-1]
            if not pre:
                return ""
    pre = re.sub(r"[_.\-\s]*(?:wave|time|w|t)?[_.\-\s]*\d*$", "", pre, flags=re.I)
    return pre.strip("_.- ")


class _Names:
    """Generated latent names and equality labels that do not collide with data columns or each other."""

    def __init__(self, taken: Iterable[str]):
        self.taken = set(str(x) for x in taken)

    def new(self, base: str) -> str:
        name = base
        while name in self.taken:
            name += "_"
        self.taken.add(name)
        return name


class _Lines:
    def __init__(self, comments: bool):
        self.comments = comments
        self.lines: List[str] = []

    def sec(self, title: str):
        if self.comments:
            if self.lines:
                self.lines.append("")
            self.lines.append(_comment(title))

    def add(self, s: str):
        self.lines.append(s)

    def text(self) -> str:
        return "\n".join(self.lines).strip() + "\n"


def _comment(text: str) -> str:
    """A model-string comment line; ';' would split the line before comments are removed (engine and lavaan)."""
    return "# " + str(text).replace(";", ",").replace('"', "'").replace("!", ".")


def _covariate_block(L: "_Lines", covs: List[str]) -> int:
    """Means, variances and covariances of exogenous covariates as model parameters (FIML for missing x)."""
    L.sec("공변량 평균·분산·공분산을 모형에 포함 (공변량 결측도 FIML 처리)")
    for c in covs:
        L.add(f"{c} ~~ {c}")
    for i, c in enumerate(covs[:-1]):
        L.add(f"{c} ~~ {' + '.join(covs[i + 1:])}")
    C = len(covs)
    return C + C + C * (C - 1) // 2


def _mod(label: Optional[str], name: str) -> str:
    return f"{label}*{name}" if label else name


def _check_unique(all_vars: List[str]):
    seen, dup = set(), []
    for v in all_vars:
        if v in seen and v not in dup:
            dup.append(v)
        seen.add(v)
    if dup:
        raise TemplateError(f"같은 변수가 여러 역할·시점에 지정됨: {', '.join(dup)}")


def _check_columns(names: List[str], columns: Optional[Sequence[str]]):
    if columns is None:
        return
    cols = [str(c) for c in columns]
    miss = [n for n in names if n not in cols]
    if miss:
        raise TemplateError(f"데이터에 없는 변수: {', '.join(miss)}. 사용 가능한 변수: "
                            f"{', '.join(cols[:40])}{' …' if len(cols) > 40 else ''}")


def _labels(roles: Dict[str, Any], n: int, series: List[List[str]], fallback: Sequence[str]) -> List[str]:
    given = _split_names(roles.get("labels"))
    out: List[str] = []
    for k in range(n):
        lab = _clean_label(given[k]) if k < len(given) else ""
        if not lab:
            lab = _clean_label(_common_stem(series[k])) if k < len(series) else ""
        if not lab or lab in out:
            lab = fallback[k] if k < len(fallback) else f"V{k + 1}"
        out.append(lab)
    return out


@dataclass
class _Build:
    lines: _Lines
    assumptions: List[str]
    notes: List[str]
    observed: List[str]          # modelled observed variables (engine "ov")
    covariates: List[str]        # conditioned exogenous covariates (engine "ox")
    latent: List[str]
    npar: int                    # analytically expected number of free parameters
    title_line: str
    references: List[str] = field(default_factory=list)
    layout: Dict[str, Tuple[float, float]] = field(default_factory=dict)   # path-diagram positions (grid units)


def _spread(names: Sequence[str], cx: float, y: float, step: float) -> Dict[str, Tuple[float, float]]:
    n = len(names)
    return {nm: (cx + (i - (n - 1) / 2) * step, y) for i, nm in enumerate(names)}


# ============================================================================================ CLPM
def _series_roles(roles: Dict[str, Any], keys=("x_waves", "y_waves", "z_waves")) -> List[List[str]]:
    series = []
    if roles.get("waves"):
        for s in roles["waves"]:
            series.append(_split_names(s))
    for k in keys:
        s = _split_names(roles.get(k))
        if s:
            series.append(s)
    return [s for s in series if s]


def _check_series(series: List[List[str]], min_k: int, min_t: int, what: str) -> int:
    if len(series) < min_k:
        raise TemplateError(f"{what}: 시점별 변수 목록 {min_k}개 이상 필요 (지정 {len(series)}개)")
    lens = [len(s) for s in series]
    if len(set(lens)) > 1:
        raise TemplateError(f"{what}: 변수마다 시점 수가 같아야 함 (지정된 시점 수: {', '.join(map(str, lens))})")
    T = lens[0]
    if T < min_t:
        raise TemplateError(f"{what}: {min_t}시점 이상 필요 (지정 {T}시점)")
    return T


def _crosslag_block(L: _Lines, nodes: List[List[str]], labs: List[str], T: int, opts: Dict[str, Any],
                    names: _Names, kind: str) -> Tuple[List[str], int]:
    """Lagged paths, wave-1 covariances and later residual (co)variances among `nodes[k][t]`.

    Returns (assumptions, npar of the block)."""
    K = len(nodes)
    cl, ccov, cvar = opts["constrain_lagged"], opts["constrain_residual_cov"], opts["constrain_residual_var"]
    lab_ar = {k: names.new(f"ar_{labs[k]}") for k in range(K)} if cl else {}
    lab_cl = {(k, j): names.new(f"cl_{labs[j]}_{labs[k]}") for k in range(K) for j in range(K) if j != k} if cl else {}
    word = {"obs": "", "within": "개인 내 성분 ", "latent": "잠재요인 "}[kind]
    L.sec(f"자기회귀·교차지연 경로 (시차 1{', 시점 간 동일 제약' if cl else ''})")
    for t in range(1, T):
        for k in range(K):
            terms = []
            for j in range(K):
                lab = (lab_ar[k] if j == k else lab_cl[(k, j)]) if cl else None
                terms.append(_mod(lab, nodes[j][t - 1]))
            L.add(f"{nodes[k][t]} ~ {' + '.join(terms)}")
    L.sec(f"1시점 {word}분산·공분산")
    for k in range(K):
        L.add(f"{nodes[k][0]} ~~ {nodes[k][0]}")
    for k in range(K):
        for j in range(k + 1, K):
            L.add(f"{nodes[k][0]} ~~ {nodes[j][0]}")
    if T > 1:
        lab_rc = {(k, j): names.new(f"rc_{labs[k]}_{labs[j]}") for k in range(K) for j in range(k + 1, K)} if ccov else {}
        lab_rv = {k: names.new(f"rv_{labs[k]}") for k in range(K)} if cvar else {}
        L.sec(f"2시점 이후 {word}잔차분산·잔차 공분산"
              + (" (공분산 동일 제약)" if ccov else "") + (" (분산 동일 제약)" if cvar else ""))
        for t in range(1, T):
            for k in range(K):
                L.add(f"{nodes[k][t]} ~~ {_mod(lab_rv.get(k), nodes[k][t])}")
            for k in range(K):
                for j in range(k + 1, K):
                    L.add(f"{nodes[k][t]} ~~ {_mod(lab_rc.get((k, j)), nodes[j][t])}")
    pairs = K * (K - 1) // 2
    npar = (K * K * (1 if cl else T - 1)) + K + pairs + K * (1 if cvar else T - 1) + pairs * (1 if ccov else T - 1)
    a = []
    a.append(f"{word}시차 1 자기회귀 경로(안정성)와 교차지연 경로 추정" + (" — 시점 간 동일 제약(정상성 가정)" if cl else
                                                         " — 구간마다 자유 추정"))
    a.append(f"1시점 {word}분산·공분산 자유 추정")
    a.append(f"2시점 이후 {word}잔차 공분산(같은 시점 동시적 관계) " + ("시점 간 동일 제약" if ccov else "시점별 자유 추정"))
    a.append(f"2시점 이후 {word}잔차분산 " + ("시점 간 동일 제약" if cvar else "시점별 자유 추정"))
    a.append("시차 2 이상 경로 없음: 직전 시점을 통제하면 그 이전 시점의 직접효과 0 가정 (1차 자기회귀 구조)")
    if cl:
        a.append(f"경로 라벨: ar_{labs[0]} = {labs[0]} 자기회귀, cl_{labs[0]}_{labs[1]} = {labs[0]} → {labs[1]} 교차지연 "
                 "(같은 라벨 = 같은 값)")
    return a, npar


def _build_clpm(roles, opts, columns):
    series = _series_roles(roles)
    T = _check_series(series, 2, 2, "교차지연 패널모형")
    K = len(series)
    allv = [v for s in series for v in s]
    _check_unique(allv)
    _check_columns(allv, columns)
    labs = _labels(roles, K, series, ["X", "Y", "Z", "W"])
    names = _Names(list(columns or []) + allv)
    L = _Lines(opts["comments"])
    a, npar = _crosslag_block(L, series, labs, T, opts, names, "obs")
    npar += K * T   # observed intercepts (Mplus default, free)
    assumptions = [f"시점별 관측변수 {K}개 × {T}시점, 관측변수 절편(시점별 평균) 자유 추정 (Mplus 기본)"] + a + [
        "1시점 변수의 분산·공분산을 모형에 포함 (Mplus에서 1시점 변수를 WITH로 언급한 경우와 같음, 조건부 처리 아님)",
        "측정오차 없음 가정: 관측변수를 오차 없는 측정값으로 취급",
        "개인 간 안정적 차이(특성)를 분리하지 않음: 교차지연 계수는 개인 간·개인 내 관계가 혼합된 값",
    ]
    notes = ["개인 내 인과적 선후관계 해석에는 무선절편 교차지연모형(RI-CLPM) 비교 권장 (Hamaker et al., 2015)"]
    if T == 2:
        notes.append("2시점 모형은 포화모형(자유도 0): 적합도 비교 불가")
    b = _Build(L, assumptions, notes, allv, [], [], npar, f"교차지연 패널모형 (CLPM) — {', '.join(labs)} × {T}시점")
    b.layout = {series[k][t]: (t * 1.6, k * 1.5) for k in range(K) for t in range(T)}
    return b


# ============================================================================================ RI-CLPM
def _build_riclpm(roles, opts, columns):
    series = _series_roles(roles)
    T = _check_series(series, 2, 3, "무선절편 교차지연모형")
    K = len(series)
    covs = _split_names(roles.get("covariates"))
    allv = [v for s in series for v in s]
    _check_unique(allv + covs)
    _check_columns(allv + covs, columns)
    labs = _labels(roles, K, series, ["X", "Y", "Z", "W"])
    names = _Names(list(columns or []) + allv + covs)
    RI = [names.new(f"RI_{labs[k]}") for k in range(K)]
    W = [[names.new(f"W_{labs[k]}{t + 1}") for t in range(T)] for k in range(K)]
    L = _Lines(opts["comments"])
    L.sec("개인 간 성분: 무선절편 (모든 시점 적재량 1 고정)")
    for k in range(K):
        L.add(f"{RI[k]} =~ {' + '.join('1*' + v for v in series[k])}")
    L.sec("개인 내 성분: 시점별 관측변수에 적재량 1")
    for k in range(K):
        for t in range(T):
            L.add(f"{W[k][t]} =~ 1*{series[k][t]}")
    L.sec("관측변수 측정오차분산 0 고정 (관측값 = 시점 평균 + 무선절편 + 개인 내 성분)")
    for k in range(K):
        for t in range(T):
            L.add(f"{series[k][t]} ~~ 0*{series[k][t]}")
    cm = opts["constrain_means"]
    if cm:
        L.sec("시점별 평균(관측변수 절편) 시점 간 동일 제약")
        for k in range(K):
            lab = names.new(f"m_{labs[k]}")
            for t in range(T):
                L.add(f"{series[k][t]} ~ {lab}*1")
    a_block, np_block = _crosslag_block(L, W, labs, T, opts, names, "within")
    L.sec("무선절편 분산·공분산" + (" (시간불변 공변량 통제 후 잔차)" if covs else ""))
    for k in range(K):
        L.add(f"{RI[k]} ~~ {RI[k]}")
    for k in range(K):
        for j in range(k + 1, K):
            L.add(f"{RI[k]} ~~ {RI[j]}")
    ncov = 0
    if covs:
        L.sec("시간불변 공변량 → 무선절편")
        for k in range(K):
            L.add(f"{RI[k]} ~ {' + '.join(covs)}")
        if opts["model_covariates"]:
            ncov = _covariate_block(L, covs)
    L.sec("무선절편과 개인 내 성분의 공분산 0 고정")
    fix_waves = [0] + ([T - 1] if covs else [])
    for k in range(K):
        terms = [f"0*{W[j][t]}" for t in fix_waves for j in range(K)]
        L.add(f"{RI[k]} ~~ {' + '.join(terms)}")
    pairs = K * (K - 1) // 2
    npar = np_block + (K if cm else K * T) + K + pairs + K * len(covs) + ncov
    assumptions = [
        f"관측변수 {K}개 × {T}시점을 개인 간 성분(무선절편 {', '.join(RI)})과 시점별 개인 내 성분(W_)으로 분해",
        "무선절편의 요인적재량 모든 시점 1 고정: 시점 간 안정적인 개인 간 차이(특성 수준)",
        "개인 내 성분은 관측변수 하나에 적재량 1로 정의, 관측변수 측정오차분산 0 고정",
        "관측변수 절편 = 시점별 집단 평균 " + ("(시점 간 동일 제약)" if cm else "(시점별 자유 추정)") + ", 잠재변수 평균 0 고정",
    ] + a_block + [
        "무선절편 분산·공분산 자유 추정" + (" (공변량 통제 후 잔차)" if covs else ""),
        "무선절편과 개인 내 성분 사이 공분산 0 (1시점 포함 모든 시점, 1시점은 명시적 0 고정)",
    ]
    if covs:
        assumptions.append(f"시간불변 공변량({', '.join(covs)})은 무선절편만 예측 (개인 내 성분과 무관), "
                           + ("공변량 평균·분산·공분산은 모형 모수 (결측 FIML)" if opts["model_covariates"]
                              else "공변량은 조건부 처리 (공변량 결측 사례 제외)"))
    notes = ["교차지연 계수는 개인 내 성분 사이의 관계: 개인이 자기 평균보다 높을 때 다음 시점 상대 변화로 해석",
             "무선절편 분산이 0에 가까우면 CLPM과 사실상 같은 모형 (분산 0 검정은 경계값 문제로 χ² 차이검정 주의)"]
    if covs:
        notes.append("공변량 → 무선절편 회귀는 공변량 → 모든 시점 관측변수 효과를 시점 간 동일하게 제약한 모형과 동치 "
                     "(Mulder & Hamaker, 2021)")
    if T == 3 and opts["constrain_lagged"] is False:
        notes.append("3시점 RI-CLPM은 자유도가 작음: 경로 동일 제약 모형과 비교 권장")
    mc = bool(covs) and opts["model_covariates"]
    b = _Build(L, assumptions, notes, allv + (covs if mc else []), [] if mc else covs,
               RI + [w for row in W for w in row], npar,
               f"무선절편 교차지연 패널모형 (RI-CLPM) — {', '.join(labs)} × {T}시점")
    for k in range(K):
        rows = (0, 1, 2) if k == 0 else (3 * k + 2, 3 * k + 1, 3 * k)      # (RI, observed, within)
        b.layout[RI[k]] = (-1.3, rows[0])
        for t in range(T):
            b.layout[series[k][t]] = (t * 1.5, rows[1])
            b.layout[W[k][t]] = (t * 1.5, rows[2])
    b.layout.update({c: (-3.0, 1 + i * (3 * K - 3) / max(len(covs) - 1, 1)) for i, c in enumerate(covs)})
    return b


# ============================================================================================ latent CLPM
def _grid_role(roles, key, label) -> List[List[str]]:
    g = _grid(roles.get(key), label)
    gw = _grid(roles.get(key + "_by_wave"), label)
    if g and gw:
        raise TemplateError(f"{label}: '{key}'와 '{key}_by_wave' 중 하나만 지정")
    if gw:
        g = _transpose(gw)
    return g


def _invariance_block(L: _Lines, F: List[str], grid: List[List[str]], lab: str, inv: str, corr: str,
                      names: _Names, marker_intercept_zero: bool = False) -> Tuple[int, List[str]]:
    """Measurement part for one construct measured by the same J indicators at T waves.

    grid: indicator × wave. Returns npar (loadings, intercepts, residual variances, correlated residuals),
    and the labels of the item intercepts. Factor means / latent intercepts are handled by the caller."""
    J, T = len(grid), len(grid[0])
    eq_load = inv in ("metric", "scalar", "strict")
    eq_int = inv in ("scalar", "strict")
    eq_res = inv == "strict"
    lam = {j: names.new(f"l{j + 1}_{lab}") for j in range(1, J)} if eq_load else {}
    nu = {j: names.new(f"n{j + 1}_{lab}") for j in range(J)} if eq_int else {}
    th = {j: names.new(f"e{j + 1}_{lab}") for j in range(J)} if eq_res else {}
    L.sec(f"{lab}: 시점별 잠재요인 (첫 문항 적재량 1 기준지표"
          + (", 나머지 적재량 시점 간 동일" if eq_load else ", 나머지 적재량 시점별 자유") + ")")
    for t in range(T):
        terms = [f"1*{grid[0][t]}"] + [_mod(lam.get(j), grid[j][t]) for j in range(1, J)]
        L.add(f"{F[t]} =~ {' + '.join(terms)}")
    if eq_int:
        L.sec(f"{lab}: 문항 절편 시점 간 동일" + (" (기준지표 절편 0 고정)" if marker_intercept_zero else ""))
        for j in range(J):
            for t in range(T):
                if j == 0 and marker_intercept_zero:
                    L.add(f"{grid[j][t]} ~ 0*1")
                else:
                    L.add(f"{grid[j][t]} ~ {nu[j]}*1")
    if eq_res:
        L.sec(f"{lab}: 문항 잔차분산 시점 간 동일 (엄격 동일성)")
        for j in range(J):
            for t in range(T):
                L.add(f"{grid[j][t]} ~~ {th[j]}*{grid[j][t]}")
    if corr != "none" and T > 1:
        L.sec(f"{lab}: 같은 문항 잔차의 시점 간 공분산 ({'모든 시점 쌍' if corr == 'all' else '인접 시점'})")
        for j in range(J):
            for t in range(T - 1):
                later = grid[j][t + 1:] if corr == "all" else [grid[j][t + 1]]
                L.add(f"{grid[j][t]} ~~ {' + '.join(later)}")
    n_load = (J - 1) * (1 if eq_load else T)
    if eq_int:
        n_int = J - (1 if marker_intercept_zero else 0)
    else:
        n_int = J * T
    n_res = J * (1 if eq_res else T)
    n_corr = J * (T * (T - 1) // 2 if corr == "all" else (T - 1 if corr == "adjacent" else 0))
    return n_load + n_int + n_res + n_corr, [nu[j] for j in sorted(nu)]


def _build_latent_clpm(roles, opts, columns):
    grids = []
    for key, nm in (("x_indicators", "구성개념 X"), ("y_indicators", "구성개념 Y"), ("z_indicators", "구성개념 Z")):
        g = _grid_role(roles, key, nm)
        if g:
            grids.append((nm, g))
    if len(grids) < 2:
        raise TemplateError("다중지표 잠재 교차지연모형: 구성개념 2개 이상(X·Y 문항 × 시점) 필요")
    Ts = [len(g[0]) for _, g in grids]
    if len(set(Ts)) > 1:
        raise TemplateError(f"다중지표 잠재 교차지연모형: 구성개념마다 시점 수가 같아야 함 (시점 수: {', '.join(map(str, Ts))})")
    T = Ts[0]
    if T < 2:
        raise TemplateError("다중지표 잠재 교차지연모형: 2시점 이상 필요")
    for nm, g in grids:
        if len(g) < 2:
            raise TemplateError(f"{nm}: 문항 2개 이상 필요 (3개 이상 권장)")
    K = len(grids)
    allv = [v for _, g in grids for row in g for v in row]
    _check_unique(allv)
    _check_columns(allv, columns)
    labs = _labels(roles, K, [[v for row in g for v in row] for _, g in grids], ["X", "Y", "Z"])
    inv = resolve_choice("종단 측정동일성", opts.get("invariance"), ["configural", "metric", "scalar", "strict"],
                         {"none": "configural", "weak": "metric", "loadings": "metric", "strong": "scalar",
                          "intercepts": "scalar", "residuals": "strict"}, "scalar")
    corr = resolve_choice("같은 문항 잔차 공분산", opts.get("correlated_residuals"), ["all", "adjacent", "none"],
                          {"true": "all", "false": "none", "lag1": "adjacent"}, "all")
    names = _Names(list(columns or []) + allv)
    F = [[names.new(f"{labs[k]}_T{t + 1}") for t in range(T)] for k in range(K)]
    L = _Lines(opts["comments"])
    npar = 0
    for k, (_, g) in enumerate(grids):
        n, _ = _invariance_block(L, F[k], g, labs[k], inv, corr, names)
        npar += n
    if inv in ("scalar", "strict") and T > 1:
        L.sec("잠재요인 평균: 1시점 0 고정(기준), 2시점 이후 구조방정식 절편 자유 추정")
        for k in range(K):
            L.add(f"{F[k][0]} ~ 0*1")
            for t in range(1, T):
                L.add(f"{F[k][t]} ~ 1")
        npar += K * (T - 1)
    a_block, np_block = _crosslag_block(L, F, labs, T, opts, names, "latent")
    npar += np_block
    inv_txt = {"configural": "형태 동일성: 시점마다 같은 문항 구성, 적재량·절편 시점별 자유",
               "metric": "측정단위 동일성: 같은 문항의 요인적재량 시점 간 동일",
               "scalar": "절편 동일성: 같은 문항의 요인적재량·절편 시점 간 동일",
               "strict": "엄격 동일성: 같은 문항의 요인적재량·절편·잔차분산 시점 간 동일"}[inv]
    Js = [len(g) for _, g in grids]
    assumptions = [
        f"구성개념 {K}개 × {T}시점, 시점별 잠재요인 문항 수 {', '.join(map(str, Js))}개",
        "각 잠재요인의 첫 문항(기준지표) 적재량 1 고정으로 척도 설정",
        f"종단 측정동일성 — {inv_txt}",
        ("잠재요인 평균: 1시점 0 고정, 2시점 이후 절편 자유 추정 (절편 동일성 하에서 시점 간 평균 변화 식별)"
         if inv in ("scalar", "strict") else "잠재요인 평균 0 고정, 문항 절편 시점별 자유 추정 (평균 비교 불가)"),
        {"all": "같은 문항 잔차의 시점 간 공분산 모든 시점 쌍 자유 추정 (문항 특수 분산의 안정성 반영)",
         "adjacent": "같은 문항 잔차의 시점 간 공분산 인접 시점만 자유 추정",
         "none": "같은 문항 잔차의 시점 간 공분산 0 가정"}[corr],
        "서로 다른 문항 잔차 사이 공분산 0 (Mplus 기본)",
    ] + a_block
    notes = ["측정동일성 수준별 모형(형태 → 측정단위 → 절편)을 차례로 적합해 χ² 차이·CFI 변화로 비교 권장"]
    if min(Js) < 3:
        notes.append("문항 2개 요인은 식별이 다른 요인과의 공분산에 의존: 3개 이상 권장")
    if T == 2:
        notes.append("2시점 구조 부분은 포화: 적합도는 측정모형 제약에서만 발생")
    b = _Build(L, assumptions, notes, allv, [], [f for row in F for f in row], npar,
               f"다중지표 잠재 교차지연모형 — {', '.join(labs)} × {T}시점")
    for k, (_, gr) in enumerate(grids):
        fy, iy = (1.0, 0.0) if k == 0 else (1.0 + 1.6 * k, 2.0 + 1.6 * k)
        for t in range(T):
            b.layout[F[k][t]] = (t * 3.5, fy)
            b.layout.update(_spread([row[t] for row in gr], t * 3.5, iy, 1.1))
    return b


# ============================================================================================ growth models
def _growth_factor_names(names: _Names, order: int, prefix: str = "") -> List[str]:
    base = ["I", "S", "Q"][: order + 1]
    return [names.new(f"{b}{('_' + prefix) if prefix else ''}") for b in base]


def _growth_loadings(L: _Lines, G: List[str], ys: List[str], scores: List[float], order: int,
                     free_basis: Optional[List[Optional[float]]] = None):
    L.add(f"{G[0]} =~ {' + '.join('1*' + y for y in ys)}")
    if free_basis is not None:
        terms = []
        for y, s in zip(ys, free_basis):
            terms.append(f"NA*{y}" if s is None else f"{_num(s)}*{y}")
        L.add(f"{G[1]} =~ {' + '.join(terms)}")
        return
    L.add(f"{G[1]} =~ {' + '.join(_num(s) + '*' + y for s, y in zip(scores, ys))}")
    if order >= 2:
        L.add(f"{G[2]} =~ {' + '.join(_num(s * s) + '*' + y for s, y in zip(scores, ys))}")


def _residual_block(L: _Lines, ys: List[str], equal: bool, names: _Names, lab: str) -> int:
    rl = names.new(f"ev_{lab}") if equal else None
    L.sec("관측변수 잔차분산" + (" (시점 간 동일 제약)" if equal else " (시점별 자유 추정)"))
    for y in ys:
        L.add(f"{y} ~~ {_mod(rl, y)}")
    return 1 if equal else len(ys)


def _zero_intercepts(L: _Lines, ys: List[str]):
    L.sec("관측변수 절편 0 고정 (평균 변화는 성장요인 평균으로 표현)")
    for y in ys:
        L.add(f"{y} ~ 0*1")


def _growth_cov(L: _Lines, G: List[str]):
    for a in range(len(G)):
        L.add(f"{G[a]} ~~ {G[a]}")
    for a in range(len(G)):
        for b in range(a + 1, len(G)):
            L.add(f"{G[a]} ~~ {G[b]}")


def _scores_txt(scores: List[float]) -> str:
    return ", ".join(_num(s) for s in scores)


def _single_series(roles, what, min_t) -> List[str]:
    ys = _split_names(roles.get("y_waves") or (roles.get("waves")[0] if roles.get("waves") else None))
    if not ys:
        raise TemplateError(f"{what}: 반복측정 변수(y_waves) 지정 필요")
    if len(ys) < min_t:
        raise TemplateError(f"{what}: {min_t}시점 이상 필요 (지정 {len(ys)}시점)")
    return ys


def _build_lgm(roles, opts, columns, order: int, tid: str):
    what = {1: "선형 잠재성장모형", 2: "2차 잠재성장모형"}[order]
    ys = _single_series(roles, what, 3 if order == 1 else 4)
    T = len(ys)
    _check_unique(ys)
    _check_columns(ys, columns)
    scores = _time_scores(opts.get("time_scores"), T)
    lab = _labels(roles, 1, [ys], ["Y"])[0]
    names = _Names(list(columns or []) + ys)
    G = _growth_factor_names(names, order)
    L = _Lines(opts["comments"])
    L.sec(f"성장요인: 절편(적재량 1)·기울기(시간점수 {_scores_txt(scores)})" + ("·2차(시간점수²)" if order == 2 else ""))
    _growth_loadings(L, G, ys, scores, order)
    L.sec("성장요인 평균 자유 추정")
    for g in G:
        L.add(f"{g} ~ 1")
    L.sec("성장요인 분산·공분산 자유 추정")
    _growth_cov(L, G)
    _zero_intercepts(L, ys)
    nres = _residual_block(L, ys, opts["residual_var_equal"], names, lab)
    nf = order + 1
    npar = nf + nf * (nf + 1) // 2 + nres
    assumptions = [
        f"반복측정 {T}시점, 시간점수 {_scores_txt(scores)} (기울기 = 시간점수 1단위당 변화)",
        "절편요인 적재량 모두 1, 기울기요인 적재량 = 시간점수" + (", 2차요인 적재량 = 시간점수²" if order == 2 else ""),
        f"절편 = 시간점수 0 시점({ys[scores.index(0.0)] if 0.0 in scores else '시간점수 0'})의 기대값",
        "관측변수 절편 0 고정, 성장요인 평균 자유 추정 (평균 궤적은 성장요인 평균이 전담)",
        "성장요인 분산·공분산 자유 추정 (개인차와 초기값-변화율 관계)",
        "관측변수 잔차분산 " + ("시점 간 동일 제약" if opts["residual_var_equal"] else "시점별 자유 추정")
        + ", 잔차 사이 공분산 0 (Mplus 성장모형 기본)",
    ]
    notes = []
    if order == 2:
        notes.append("2차 모형의 선형 기울기 평균은 시간점수 0 시점의 순간 변화율: 시간점수 원점 위치에 따라 해석 변화")
    if T == 3 and order == 1 and not opts["residual_var_equal"]:
        notes.append("3시점 선형 모형은 자유도 1: 형태 검증력이 낮음")
    if T == 4 and order == 2 and not opts["residual_var_equal"]:
        notes.append("4시점 2차 모형은 자유도 1: 형태 검증력이 낮음")
    b = _Build(L, assumptions, notes, ys, [], G, npar, f"{'선형' if order == 1 else '2차'} 잠재성장모형 — {lab} × {T}시점")
    b.layout = {y: (t * 1.3, 1.6) for t, y in enumerate(ys)}
    b.layout.update(_spread(G, (T - 1) * 1.3 / 2, 0, 1.8))
    return b


def _build_lgm_basis(roles, opts, columns):
    ys = _single_series(roles, "잠재기저 성장모형", 3)
    T = len(ys)
    _check_unique(ys)
    _check_columns(ys, columns)
    anchor = resolve_choice("시간점수 1 고정 시점", opts.get("anchor"), ["last", "second"],
                            {"final": "last", "2": "second"}, "last")
    lab = _labels(roles, 1, [ys], ["Y"])[0]
    names = _Names(list(columns or []) + ys)
    G = _growth_factor_names(names, 1)
    fixed1 = T - 1 if anchor == "last" else 1
    basis: List[Optional[float]] = [None] * T
    basis[0] = 0.0
    basis[fixed1] = 1.0
    L = _Lines(opts["comments"])
    L.sec(f"성장요인: 절편(적재량 1)·형태요인(1시점 0, {fixed1 + 1}시점 1 고정, 나머지 자유 추정)")
    _growth_loadings(L, G, ys, [], 1, free_basis=basis)
    L.sec("성장요인 평균 자유 추정")
    for g in G:
        L.add(f"{g} ~ 1")
    L.sec("성장요인 분산·공분산 자유 추정")
    _growth_cov(L, G)
    _zero_intercepts(L, ys)
    nres = _residual_block(L, ys, opts["residual_var_equal"], names, lab)
    npar = 2 + 3 + nres + (T - 2)
    assumptions = [
        f"반복측정 {T}시점, 형태요인 적재량 1시점 0·{fixed1 + 1}시점 1 고정, 나머지 {T - 2}개 자유 추정",
        ("형태요인 평균 = 1시점부터 마지막 시점까지 전체 평균 변화량, 중간 적재량 = 누적 변화 비율"
         if anchor == "last" else "형태요인 평균 = 1→2시점 변화량, 이후 적재량 = 첫 구간 변화 대비 누적 변화 배수"),
        "절편요인 적재량 모두 1, 관측변수 절편 0 고정, 성장요인 평균 자유 추정",
        "성장요인 분산·공분산 자유 추정",
        "관측변수 잔차분산 " + ("시점 간 동일 제약" if opts["residual_var_equal"] else "시점별 자유 추정")
        + ", 잔차 사이 공분산 0",
    ]
    notes = []
    if T == 3 and not opts["residual_var_equal"]:
        notes.append("3시점 잠재기저 모형은 포화(자유도 0): 4시점 이상 권장")
    notes.append("자유 적재량 추정값을 선형 시간점수와 비교하면 비선형 변화의 형태 확인 가능")
    b = _Build(L, assumptions, notes, ys, [], G, npar, f"잠재기저 성장모형 — {lab} × {T}시점")
    b.layout = {y: (t * 1.3, 1.6) for t, y in enumerate(ys)}
    b.layout.update(_spread(G, (T - 1) * 1.3 / 2, 0, 1.8))
    return b


def _build_lgm_conditional(roles, opts, columns):
    shape = resolve_choice("성장 형태", opts.get("shape"), ["linear", "quadratic"], {"quad": "quadratic"}, "linear")
    order = 1 if shape == "linear" else 2
    ys = _single_series(roles, "조건부 잠재성장모형", 3 if order == 1 else 4)
    T = len(ys)
    tic = _split_names(roles.get("covariates"))
    tvc = _split_names(roles.get("tvc_waves"))
    if not tic and not tvc:
        raise TemplateError("조건부 잠재성장모형: 시간불변 공변량(covariates) 또는 시간가변 공변량(tvc_waves) 1개 이상 필요")
    if tvc and len(tvc) != T:
        raise TemplateError(f"시간가변 공변량: 반복측정 변수와 같은 시점 수 필요 (반복측정 {T}시점, 공변량 {len(tvc)}시점)")
    _check_unique(ys + tic + tvc)
    _check_columns(ys + tic + tvc, columns)
    scores = _time_scores(opts.get("time_scores"), T)
    lab = _labels(roles, 1, [ys], ["Y"])[0]
    names = _Names(list(columns or []) + ys + tic + tvc)
    G = _growth_factor_names(names, order)
    L = _Lines(opts["comments"])
    L.sec(f"성장요인: 절편(적재량 1)·기울기(시간점수 {_scores_txt(scores)})" + ("·2차(시간점수²)" if order == 2 else ""))
    _growth_loadings(L, G, ys, scores, order)
    if tic:
        L.sec("시간불변 공변량 → 성장요인")
        for g in G:
            L.add(f"{g} ~ {' + '.join(tic)}")
    L.sec("성장요인 절편 자유 추정" + (" (공변량 0일 때의 성장요인 평균)" if tic else ""))
    for g in G:
        L.add(f"{g} ~ 1")
    L.sec("성장요인 " + ("잔차 " if tic else "") + "분산·공분산 자유 추정")
    _growth_cov(L, G)
    ntvc = 0
    if tvc:
        eq = opts["tvc_equal"]
        bl = names.new(f"b_tvc_{lab}") if eq else None
        L.sec("시간가변 공변량 → 같은 시점 관측변수" + (" (효과 시점 간 동일)" if eq else " (효과 시점별 자유)"))
        for y, z in zip(ys, tvc):
            L.add(f"{y} ~ {_mod(bl, z)}")
        ntvc = 1 if eq else T
    _zero_intercepts(L, ys)
    nres = _residual_block(L, ys, opts["residual_var_equal"], names, lab)
    mc = opts["model_covariates"]
    ncov = _covariate_block(L, tic + tvc) if mc else 0
    nf = order + 1
    npar = nf + nf * len(tic) + nf * (nf + 1) // 2 + ntvc + nres + ncov
    assumptions = [
        f"반복측정 {T}시점, 시간점수 {_scores_txt(scores)}, {'선형' if order == 1 else '2차'} 성장",
        "관측변수 절편 0 고정, 성장요인 절편 자유 추정",
    ]
    if tic:
        assumptions.append(f"시간불변 공변량({', '.join(tic)}) → 모든 성장요인 회귀, 성장요인 잔차 분산·공분산 자유 추정")
        assumptions.append("성장요인 절편 = 공변량이 모두 0일 때의 성장요인 평균 (공변량 중심화 시 평균 수준 해석)")
    if tvc:
        assumptions.append(f"시간가변 공변량({', '.join(tvc)}) → 같은 시점 관측변수만 예측"
                           + (" — 효과 시점 간 동일 제약" if opts["tvc_equal"] else " — 효과 시점별 자유 추정"))
        assumptions.append("시간가변 공변량과 성장요인 사이 관계는 추정하지 않음 (Mplus 예제 6.10과 같은 조건부 구조)")
    assumptions.append("공변량 평균·분산·공분산을 모형 모수로 추정 (공변량 결측 사례도 FIML로 사용)" if mc else
                       "공변량은 외생 관측변수로 조건부 처리 (평균·분산·공분산은 모수 아님, 공변량 결측 사례 제외)")
    assumptions.append("관측변수 잔차분산 " + ("시점 간 동일 제약" if opts["residual_var_equal"] else "시점별 자유 추정")
                       + ", 잔차 사이 공분산 0")
    notes = ["공변량을 평균 중심화하면 성장요인 절편이 평균적 개인의 초기값·변화율로 해석됨 (자료 편집의 계산 기능 사용)"]
    if tvc and not mc:
        notes.append("시간가변 공변량이 탈락으로 결측인 사례는 조건부 처리에서 제외됨: 결측이 많으면 "
                     "'공변량 분포를 모형에 포함' 옵션 사용")
    b = _Build(L, assumptions, notes, ys + (tic + tvc if mc else []), [] if mc else tic + tvc, G, npar,
                  f"조건부 잠재성장모형 — {lab} × {T}시점")
    b.layout = {y: (t * 1.3, 1.8) for t, y in enumerate(ys)}
    b.layout.update(_spread(G, (T - 1) * 1.3 / 2, 0.4, 1.8))
    b.layout.update(_spread(tic, (T - 1) * 1.3 / 2, -0.9, 1.4))
    b.layout.update({z: (t * 1.3, 3.0) for t, z in enumerate(tvc)})
    return b


def _build_lgm_parallel(roles, opts, columns):
    series = _series_roles(roles, ("x_waves", "y_waves"))
    if len(series) != 2:
        raise TemplateError("병행과정 잠재성장모형: 과정 2개(x_waves, y_waves) 필요")
    T = _check_series(series, 2, 3, "병행과정 잠재성장모형")
    allv = series[0] + series[1]
    _check_unique(allv)
    _check_columns(allv, columns)
    scores = _time_scores(opts.get("time_scores"), T)
    structure = resolve_choice("성장요인 관계", opts.get("structure"), ["covariance", "regression"],
                               {"cov": "covariance", "covariances": "covariance", "reg": "regression",
                                "regressions": "regression"}, "covariance")
    labs = _labels(roles, 2, series, ["X", "Y"])
    names = _Names(list(columns or []) + allv)
    Gx = _growth_factor_names(names, 1, labs[0])
    Gy = _growth_factor_names(names, 1, labs[1])
    L = _Lines(opts["comments"])
    for G, ys, lab in ((Gx, series[0], labs[0]), (Gy, series[1], labs[1])):
        L.sec(f"{lab}: 절편(적재량 1)·기울기(시간점수 {_scores_txt(scores)})")
        _growth_loadings(L, G, ys, scores, 1)
    L.sec("성장요인 평균(또는 절편) 자유 추정")
    for g in Gx + Gy:
        L.add(f"{g} ~ 1")
    if structure == "covariance":
        L.sec("성장요인 4개 분산·공분산 자유 추정")
        _growth_cov(L, Gx + Gy)
    else:
        L.sec("기울기 ← 두 과정의 절편 (과정 내·과정 간 절편 → 기울기 효과)")
        L.add(f"{Gx[1]} ~ {Gx[0]} + {Gy[0]}")
        L.add(f"{Gy[1]} ~ {Gy[0]} + {Gx[0]}")
        L.sec("절편 분산·공분산, 기울기 잔차 분산·공분산 자유 추정")
        L.add(f"{Gx[0]} ~~ {Gx[0]}")
        L.add(f"{Gy[0]} ~~ {Gy[0]}")
        L.add(f"{Gx[0]} ~~ {Gy[0]}")
        L.add(f"{Gx[1]} ~~ {Gx[1]}")
        L.add(f"{Gy[1]} ~~ {Gy[1]}")
        L.add(f"{Gx[1]} ~~ {Gy[1]}")
    _zero_intercepts(L, allv)
    nres = 0
    for ys, lab in ((series[0], labs[0]), (series[1], labs[1])):
        nres += _residual_block(L, ys, opts["residual_var_equal"], names, lab)
    nwave = 0
    if opts["wave_residual_cov"]:
        L.sec("같은 시점 두 변수의 잔차 공분산")
        for a, b in zip(series[0], series[1]):
            L.add(f"{a} ~~ {b}")
        nwave = T
    npar = 4 + 10 + nres + nwave
    assumptions = [
        f"두 과정({labs[0]}, {labs[1]}) 각 {T}시점, 시간점수 {_scores_txt(scores)}, 각 과정 선형 성장",
        "관측변수 절편 0 고정, 성장요인 평균 자유 추정",
        ("성장요인 4개(절편·기울기 × 2과정)의 분산·공분산 모두 자유 추정"
         if structure == "covariance" else
         "각 과정 기울기를 두 과정의 절편에 회귀 (과정 간 초기값 → 변화율 효과), 절편 공분산·기울기 잔차 공분산 자유 추정"
         " — 공분산 모형과 같은 적합도의 재모수화"),
        "같은 시점 두 변수의 잔차 공분산 " + ("자유 추정" if opts["wave_residual_cov"] else "0 가정 (Mplus 기본)"),
        "관측변수 잔차분산 " + ("과정별 시점 간 동일 제약" if opts["residual_var_equal"] else "시점별 자유 추정"),
    ]
    notes = ["두 과정의 기울기 공분산(또는 잔차 공분산)은 변화율이 함께 움직이는 정도"]
    b = _Build(L, assumptions, notes, allv, [], Gx + Gy, npar, f"병행과정 잠재성장모형 — {labs[0]}, {labs[1]} × {T}시점")
    cx = (T - 1) * 1.3 / 2
    b.layout = {**_spread(Gx, cx, 0, 1.8), **{y: (t * 1.3, 1.4) for t, y in enumerate(series[0])},
                **{y: (t * 1.3, 2.6) for t, y in enumerate(series[1])}, **_spread(Gy, cx, 4.0, 1.8)}
    return b


def _build_lgm_second_order(roles, opts, columns):
    grid = _grid_role(roles, "indicators", "문항 × 시점")
    if not grid:
        raise TemplateError("2차 요인 잠재성장모형: 문항 × 시점(indicators) 지정 필요")
    J, T = len(grid), len(grid[0])
    if J < 2:
        raise TemplateError("2차 요인 잠재성장모형: 시점별 문항 2개 이상 필요 (3개 이상 권장)")
    shape = resolve_choice("성장 형태", opts.get("shape"), ["linear", "quadratic"], {"quad": "quadratic"}, "linear")
    order = 1 if shape == "linear" else 2
    if T < (3 if order == 1 else 4):
        raise TemplateError(f"2차 요인 잠재성장모형: {'선형' if order == 1 else '2차'} 성장은 "
                            f"{3 if order == 1 else 4}시점 이상 필요 (지정 {T}시점)")
    allv = [v for row in grid for v in row]
    _check_unique(allv)
    _check_columns(allv, columns)
    scores = _time_scores(opts.get("time_scores"), T)
    inv = resolve_choice("종단 측정동일성", opts.get("invariance"), ["scalar", "strict"],
                         {"strong": "scalar", "intercepts": "scalar", "residuals": "strict"}, "scalar")
    mid = resolve_choice("성장요인 평균 식별", opts.get("mean_identification"), ["marker", "intercept_zero"],
                         {"mplus": "intercept_zero", "fixed_intercept": "intercept_zero"}, "marker")
    corr = resolve_choice("같은 문항 잔차 공분산", opts.get("correlated_residuals"), ["all", "adjacent", "none"],
                          {"true": "all", "false": "none", "lag1": "adjacent"}, "all")
    lab = _labels(roles, 1, [allv], ["F"])[0]
    names = _Names(list(columns or []) + allv)
    F = [names.new(f"{lab}_T{t + 1}") for t in range(T)]
    G = _growth_factor_names(names, order)
    L = _Lines(opts["comments"])
    nmeas, _ = _invariance_block(L, F, grid, lab, inv, corr, names, marker_intercept_zero=(mid == "marker"))
    L.sec("1차 요인 절편 0 고정 (평균 변화는 2차 성장요인 평균으로 표현)")
    for f in F:
        L.add(f"{f} ~ 0*1")
    feq = opts["factor_resvar_equal"]
    fl = names.new(f"d_{lab}") if feq else None
    L.sec("1차 요인 잔차분산 (시점별 측정 시점 고유 변동)" + (" — 시점 간 동일 제약" if feq else ""))
    for f in F:
        L.add(f"{f} ~~ {_mod(fl, f)}")
    L.sec(f"2차 성장요인: 절편(적재량 1)·기울기(시간점수 {_scores_txt(scores)})" + ("·2차(시간점수²)" if order == 2 else ""))
    _growth_loadings(L, G, F, scores, order)
    L.sec("성장요인 평균" + (" 자유 추정" if mid == "marker" else ": 절편요인 0 고정(문항 절편이 수준 담당), 나머지 자유 추정"))
    for k, g in enumerate(G):
        L.add(f"{g} ~ 0*1" if (mid == "intercept_zero" and k == 0) else f"{g} ~ 1")
    L.sec("성장요인 분산·공분산 자유 추정")
    _growth_cov(L, G)
    nf = order + 1
    npar = nmeas + T if not feq else nmeas + 1
    npar += (nf - (1 if mid == "intercept_zero" else 0)) + nf * (nf + 1) // 2
    assumptions = [
        f"{T}시점 × 문항 {J}개, 시점마다 같은 문항으로 1차 잠재요인({F[0]} … {F[-1]}) 정의",
        "1차 요인마다 첫 문항(기준지표) 적재량 1 고정, 나머지 적재량 시점 간 동일 (측정단위 동일성)",
        ("문항 절편 시점 간 동일, 기준지표 절편 0 고정 → 성장요인 평균은 기준지표 척도"
         if mid == "marker" else
         "모든 문항 절편 시점 간 동일(추정), 절편요인 평균 0 고정 → 기울기 평균만 추정 (Mplus 예제 6.14 방식)"),
        "1차 요인 절편 0 고정: 시점별 잠재평균의 변화는 2차 성장요인 평균으로만 표현",
    ]
    if inv == "strict":
        assumptions.append("같은 문항 잔차분산 시점 간 동일 (엄격 동일성)")
    assumptions += [
        {"all": "같은 문항 잔차의 시점 간 공분산 모든 시점 쌍 자유 추정",
         "adjacent": "같은 문항 잔차의 시점 간 공분산 인접 시점만 자유 추정",
         "none": "같은 문항 잔차의 시점 간 공분산 0 가정"}[corr],
        "1차 요인 잔차분산 " + ("시점 간 동일 제약" if feq else "시점별 자유 추정") + ", 1차 요인 잔차 사이 공분산 0",
        f"2차 성장요인: 절편 적재량 1, 기울기 적재량 = 시간점수 {_scores_txt(scores)}"
        + (", 2차 적재량 = 시간점수²" if order == 2 else "") + ", 분산·공분산 자유 추정",
    ]
    notes = ["측정동일성 가정(적재량·절편 동일)을 먼저 종단 CFA로 점검한 뒤 성장모형 해석 권장"]
    if J < 3:
        notes.append("시점별 문항 2개는 식별이 약함: 3개 이상 권장")
    b = _Build(L, assumptions, notes, allv, [], F + G, npar, f"2차 요인 잠재성장모형 — {lab} (문항 {J}개) × {T}시점")
    b.layout = _spread(G, (T - 1) * 3.5 / 2, 0, 2.4)
    for t in range(T):
        b.layout[F[t]] = (t * 3.5, 1.4)
        b.layout.update(_spread([row[t] for row in grid], t * 3.5, 2.6, 1.1))
    return b


# ============================================================================================ LCS
def _build_lcs(roles, opts, columns):
    ys = _single_series(roles, "잠재변화점수 모형", 3)
    T = len(ys)
    _check_unique(ys)
    _check_columns(ys, columns)
    cc, pr, req = opts["constant_change"], opts["proportional"], opts["residual_var_equal"]
    if not cc and not pr:
        raise TemplateError("잠재변화점수 모형: 일정 변화 요인과 비례 변화 효과 중 하나 이상 선택 필요")
    lab = _labels(roles, 1, [ys], ["Y"])[0]
    names = _Names(list(columns or []) + ys)
    Ls = [names.new(f"L{t + 1}") for t in range(T)]
    Ds = [None] + [names.new(f"D{t + 1}") for t in range(1, T)]
    Gc = names.new("G") if cc else None
    beta = names.new(f"beta_{lab}") if pr else None
    L = _Lines(opts["comments"])
    L.sec("시점별 진점수 (관측변수 적재량 1)")
    for l, y in zip(Ls, ys):
        L.add(f"{l} =~ 1*{y}")
    L.sec("진점수 자기회귀 1 고정: L(t) = L(t-1) + D(t)")
    for t in range(1, T):
        L.add(f"{Ls[t]} ~ 1*{Ls[t - 1]}")
    L.sec("잠재변화점수: D(t) =~ 1*L(t)")
    for t in range(1, T):
        L.add(f"{Ds[t]} =~ 1*{Ls[t]}")
    if pr:
        L.sec("비례 변화: 직전 진점수 → 변화점수 (시점 간 동일)")
        for t in range(1, T):
            L.add(f"{Ds[t]} ~ {beta}*{Ls[t - 1]}")
    if cc:
        L.sec("일정 변화 요인: 모든 변화점수에 적재량 1")
        L.add(f"{Gc} =~ {' + '.join('1*' + d for d in Ds[1:])}")
    L.sec("평균: 1시점 진점수·일정 변화 요인 평균 자유, 관측변수 절편 0 고정")
    L.add(f"{Ls[0]} ~ 1")
    if cc:
        L.add(f"{Gc} ~ 1")
    for y in ys:
        L.add(f"{y} ~ 0*1")
    L.sec("분산: 1시점 진점수·일정 변화 요인 자유, 2시점 이후 진점수·변화점수 잔차분산 0 고정")
    L.add(f"{Ls[0]} ~~ {Ls[0]}")
    if cc:
        L.add(f"{Gc} ~~ {Gc}")
        L.add(f"{Ls[0]} ~~ {Gc}")
    for t in range(1, T):
        L.add(f"{Ls[t]} ~~ 0*{Ls[t]}")
    for t in range(1, T):
        L.add(f"{Ds[t]} ~~ 0*{Ds[t]}")
    if T > 2:
        L.sec("변화점수 잔차 사이 공분산 0 고정")
        for t in range(1, T - 1):
            L.add(f"{Ds[t]} ~~ {' + '.join('0*' + d for d in Ds[t + 1:])}")
    nres = _residual_block(L, ys, req, names, lab)
    npar = 2 + nres + (3 if cc else 0) + (1 if pr else 0)
    assumptions = [
        f"반복측정 {T}시점, 시점별 진점수 L(t)는 관측변수에 적재량 1 (관측값 = 진점수 + 측정오차)",
        "진점수 구조: L(t) = 1 × L(t−1) + D(t), 2시점 이후 진점수 잔차분산 0",
        "변화점수 D(t) = " + " + ".join(([f"일정 변화 요인 {Gc}"] if cc else []) + (["β × L(t−1) (비례 변화)"] if pr else []))
        + ", 변화점수 잔차분산·잔차 공분산 0 (이중 변화점수 모형)",
    ]
    if pr:
        assumptions.append(f"비례 변화 효과 β({beta}) 시점 간 동일 (음수: 수준이 높을수록 증가폭 감소·수렴)")
    if cc:
        assumptions.append("1시점 진점수와 일정 변화 요인의 평균·분산·공분산 자유 추정")
    else:
        assumptions.append("1시점 진점수 평균·분산 자유 추정, 일정 변화 요인 없음 (비례 변화만)")
    assumptions.append("관측변수 절편 0 고정, 측정오차분산 " + ("시점 간 동일 제약 (McArdle 모형 기본)" if req else "시점별 자유 추정"))
    notes = ["측정 간격이 같다고 가정: 간격이 다르면 결측 시점을 둔 잠재 시점(phantom) 구조 필요"]
    if cc and not pr:
        notes.append("비례 효과 없는 일정 변화 모형은 선형 잠재성장모형과 같은 평균·공분산 구조")
    b = _Build(L, assumptions, notes, ys, [], Ls + Ds[1:] + ([Gc] if cc else []), npar,
                  f"단변량 잠재변화점수 모형 — {lab} × {T}시점")
    b.layout = {**{y: (t * 1.6, 3.0) for t, y in enumerate(ys)}, **{l: (t * 1.6, 2.0) for t, l in enumerate(Ls)},
                **{Ds[t]: (t * 1.6 - 0.55, 1.0) for t in range(1, T)}}
    if cc:
        b.layout[Gc] = ((T - 1) * 1.6 / 2 + 0.25, 0.0)
    return b


_BUILDERS = {
    "clpm": _build_clpm,
    "riclpm": _build_riclpm,
    "latent_clpm": _build_latent_clpm,
    "lgm_linear": lambda r, o, c: _build_lgm(r, o, c, 1, "lgm_linear"),
    "lgm_quadratic": lambda r, o, c: _build_lgm(r, o, c, 2, "lgm_quadratic"),
    "lgm_basis": _build_lgm_basis,
    "lgm_conditional": _build_lgm_conditional,
    "lgm_parallel": _build_lgm_parallel,
    "lgm_second_order": _build_lgm_second_order,
    "lcs_univariate": _build_lcs,
}

_ROLE_ALIASES = {"xwaves": "x_waves", "ywaves": "y_waves", "zwaves": "z_waves", "x": "x_waves", "y": "y_waves",
                 "z": "z_waves", "x_vars": "x_waves", "y_vars": "y_waves", "tvc": "tvc_waves", "tvcs": "tvc_waves",
                 "time_varying": "tvc_waves", "tic": "covariates", "covs": "covariates", "covariate": "covariates",
                 "items": "indicators", "label": "labels", "names": "labels"}


def _effective_options(tid: str, options: Dict[str, Any]) -> Dict[str, Any]:
    spec = _CAT[tid]
    allowed = {o["key"]: o for o in spec["options"]}
    opts = _norm_keys(options)
    unknown = [k for k in opts if k not in allowed]
    if unknown:
        raise TemplateError(f"{spec['title']}: 알 수 없는 옵션 {', '.join(unknown)} (가능: {', '.join(allowed)})")
    out: Dict[str, Any] = {}
    for key, o in allowed.items():
        v = opts.get(key)
        if o["type"] == "check":
            out[key] = _flag(v, o["default"])
        elif o["type"] == "select":
            out[key] = v if v not in (None, "") else o["default"]     # validated by the builder (resolve_choice)
        else:
            out[key] = v if v is not None else o["default"]
    return out


def _effective_roles(tid: str, roles: Dict[str, Any]) -> Dict[str, Any]:
    spec = _CAT[tid]
    allowed = {r["key"] for r in spec["roles"]}
    extra_ok = {"waves"} | {r + "_by_wave" for r in allowed if r.endswith("indicators")}
    out: Dict[str, Any] = {}
    for k, v in (roles or {}).items():
        key = _snake(k)
        key = _ROLE_ALIASES.get(key.replace("_", ""), _ROLE_ALIASES.get(key, key))
        if key not in allowed and key not in extra_ok:
            raise TemplateError(f"{spec['title']}: 알 수 없는 변수 역할 '{k}' (가능: {', '.join(sorted(allowed))})")
        out[key] = v
    for r in spec["roles"]:
        if r["req"] and not out.get(r["key"]) and not (r["key"] in ("x_waves", "y_waves") and out.get("waves")) \
                and not out.get(r["key"] + "_by_wave"):
            raise TemplateError(f"{spec['title']}: '{r['label']}' 지정 필요")
    return out


# ============================================================================================ engine check
def _engine_check(model: str, observed: List[str]) -> Dict[str, Any]:
    """Parse the model with the SEM engine: number of free parameters, df and variable roles."""
    try:
        from . import sem_engine as se
        spec = se.ModelSpec(se.parse_model(model), list(observed), 1)
        py, qx = spec.py, spec.qx
        p_h1 = py + py * (py + 1) // 2 + py * qx
        return {"npar": int(spec.q), "df": int(p_h1 - spec.q), "moments": int(p_h1), "observed": list(spec.ov),
                "covariates": list(spec.ox), "latent": list(spec.latent), "error": ""}
    except Exception as e:  # engine missing or model rejected
        return {"npar": None, "df": None, "moments": None, "observed": [], "covariates": [], "latent": [],
                "error": f"{type(e).__name__}: {e}"}


# ============================================================================================ path diagram
def _diagram(model: str, layout: Dict[str, Tuple[float, float]], covariates: Sequence[str]) -> Dict[str, Any]:
    """Nodes (positions from the template layout) and edges of the generated model for the dialog's preview.

    Edge labels show what the SEM Builder shows on a path: fixed values (@1, @0, …) and equality labels."""
    from . import sem_engine as se
    stmts = se.parse_model(model)
    latent = {st_.lhs for st_ in stmts if st_.op == "=~"}

    def lab(term) -> Tuple[str, bool]:
        m = term.mods[0] if term.mods else None
        if m is None or m[0] in ("free", "start"):
            return "", False
        if m[0] == "fixed":
            return _num(m[1]), True
        return str(m[1]), False

    edges: List[Dict[str, Any]] = []
    zero_var: List[str] = []
    for st_ in stmts:
        for t in st_.terms:
            text, fixed = lab(t)
            if st_.op == "=~":
                edges.append({"from": st_.lhs, "to": t.name, "kind": "path", "label": text, "fixed": fixed})
            elif st_.op == "~":
                edges.append({"from": t.name, "to": st_.lhs, "kind": "path", "label": text, "fixed": fixed})
            elif st_.op == "~~":
                if t.name == st_.lhs:
                    if fixed and text == "0":
                        zero_var.append(st_.lhs)
                    continue
                if fixed and text == "0":
                    continue
                both_obs = st_.lhs not in latent and t.name not in latent
                edges.append({"from": st_.lhs, "to": t.name, "kind": "rescov" if both_obs else "cov",
                              "label": text, "fixed": fixed})
    names = [n for n in layout]
    nodes = [{"id": n, "kind": "lat" if n in latent else ("x" if n in covariates else "obs"),
              "x": round(layout[n][0], 3), "y": round(layout[n][1], 3), "zeroVar": n in zero_var} for n in names]
    known = set(names)
    edges = [e for e in edges if e["from"] in known and e["to"] in known]
    return {"nodes": nodes, "edges": edges}


# ============================================================================================ Stata translation
_STATA_NAME = re.compile(r"^[^\W\d]\w{0,31}$", re.UNICODE)


def to_stata(model: str, missing: str = "fiml", estimator: str = "ml") -> List[str]:
    """Stata `sem` command (do-file lines) equivalent to a template model string.

    Covers the subset the templates use: =~ and ~ paths with fixed values (@v) and equality labels (@name),
    intercepts (_cons@v / _cons@name, means()), (error) variances and covariances. Stata defaults that differ from
    Mplus are overridden explicitly: exogenous observed and exogenous latent variables are made uncorrelated
    (Stata correlates all exogenous variables). Latent names are declared with latent() nocapslatent, so Korean
    names work. Exogenous observed covariates are modelled jointly by method(mlmv) (Stata noxconditional), which
    gives the same estimates as conditioning when the covariates are complete. Models with more latent than
    observed dependent variables get noivstart. Checked in StataNow 19.5 on the sample data: log likelihood and chi-square
    equal to the engine and lavaan; quadratic / latent-basis / second-order growth models may need from() start
    values to converge."""
    from . import sem_engine as se
    stmts = se.parse_model(model)
    if any(st_.op in (":=", "==") for st_ in stmts):
        raise TemplateError("Stata 변환 미지원: 정의 모수(:=)·제약식(==)")
    latent: List[str] = []
    for st_ in stmts:
        if st_.op == "=~" and st_.lhs not in latent:
            latent.append(st_.lhs)
    endo = set()
    for st_ in stmts:
        if st_.op == "=~":
            endo |= {t.name for t in st_.terms}
        elif st_.op == "~":
            endo.add(st_.lhs)
    names: List[str] = []
    for st_ in stmts:
        for n in [st_.lhs] + [t.name for t in st_.terms]:
            if n != "1" and n not in names:
                names.append(n)
    bad = [n for n in names if not _STATA_NAME.match(n)]
    if bad:
        raise TemplateError(f"Stata 변환 미지원 이름: {', '.join(bad)} (Stata 이름 규칙: 문자로 시작, 32자 이하)")

    def at(term) -> str:
        if len(term.mods) != 1:
            raise TemplateError("Stata 변환 미지원: 다집단 c(...) 지정")
        m = term.mods[0]
        if m is None or m[0] in ("free", "start"):
            return ""
        if m[0] == "fixed":
            return "@" + _num(m[1])
        return "@" + str(m[1])

    paths: List[str] = []
    variances: List[str] = []
    covs: List[str] = []
    means: List[str] = []
    explicit_pairs = set()
    for st_ in stmts:
        if st_.op == "=~":
            paths.append(f"({st_.lhs} -> {' '.join(t.name + at(t) for t in st_.terms)})")
        elif st_.op == "~":
            paths.append(f"({st_.lhs} <- {' '.join(t.name + at(t) for t in st_.terms)})")
        elif st_.op == "~1":
            a = at(st_.terms[0])
            if st_.lhs in endo:
                if st_.lhs in latent or a:
                    paths.append(f"({st_.lhs} <- _cons{a})")
            else:
                means.append(f"{st_.lhs}{a}")
        elif st_.op == "~~":
            for t in st_.terms:
                a, b = st_.lhs, t.name
                if a == b:
                    if at(t):
                        variances.append(f"{'e.' if a in endo else ''}{a}{at(t)}")
                    continue
                explicit_pairs.add(frozenset((a, b)))
                if (a in endo) == (b in endo):
                    pre = "e." if a in endo else ""
                    covs.append(f"{pre}{a}*{pre}{b}{at(t)}")
                elif at(t) != "@0":
                    raise TemplateError(f"Stata 변환 미지원: 외생변수와 오차의 공분산 ({a} ~~ {b})")
    obs_exo = [n for n in names if n not in latent and n not in endo]
    lat_exo = [n for n in latent if n not in endo]
    for x in obs_exo:
        for f in lat_exo:
            if frozenset((x, f)) not in explicit_pairs:
                covs.append(f"{x}*{f}@0")
    opts = []
    if latent:
        opts.append(f"latent({' '.join(latent)}) nocapslatent")
    if means:
        opts.append(f"means({' '.join(means)})")
    if variances:
        opts.append(f"var({' '.join(variances)})")
    if covs:
        opts.append(f"cov({' '.join(covs)})")
    opts.append("method(mlmv)" if missing == "fiml" else "method(ml)")
    if estimator == "mlr":
        opts.append("vce(robust)")
    measured = [n for n in names if n not in latent and n in endo]
    if len(latent) > len(measured):
        # Stata's IV starting-value step stops with "too many latent variables" (RI-CLPM, LCS); the Builder's
        # "Skip calculation of starting values" = noivstart
        opts.append("noivstart")
    lines = ["sem " + paths[0] + " ///"] + [f"    {p_} ///" for p_ in paths[1:]]
    lines.append("    , " + " ///\n      ".join(opts))
    out = "\n".join(lines).splitlines()
    return out + ["estat gof, stats(all)"]


# ============================================================================================ public API
def build_template(template_id: str, roles: Optional[Dict[str, Any]] = None, options: Optional[Dict[str, Any]] = None,
                   columns: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Generate the model string of a template.

    roles   : variable roles (see TEMPLATE_CATALOG[*]["roles"]); snake_case or camelCase keys
    options : template options (TEMPLATE_CATALOG[*]["options"]); snake_case or camelCase keys
    columns : data columns (optional) — variables are checked against them and generated latent names avoid them

    Returns {"template", "title", "model", "assumptions", "notes", "references", "suggested"} where
    suggested = {"analysis": "sem", "params": {"model": ...}, "npar", "df", "moments", "observed", "covariates",
    "latent"} (npar/df as interpreted by the SEM engine)."""
    tid = _resolve_id(template_id)
    spec = _CAT[tid]
    r = _effective_roles(tid, roles or {})
    o = _effective_options(tid, options or {})
    b = _BUILDERS[tid](r, o, columns)
    py, qx = len(b.observed), len(b.covariates)
    moments = py + py * (py + 1) // 2 + py * qx
    df_expected = moments - b.npar
    header: List[str] = []
    if o.get("comments", True):
        header.append(_comment(f"모형 예시: {b.title_line}"))
        header.append(_comment(spec["name_en"]))
        header.append(_comment("기본 가정"))
        header += [_comment(f" - {a}") for a in b.assumptions]
    model = ("\n".join(header) + "\n" if header else "") + b.lines.text()
    eng = _engine_check(model, b.observed + b.covariates)
    notes = list(b.notes)
    if df_expected < 0:
        raise TemplateError(f"{spec['title']}: 자유모수 {b.npar}개가 표본 적률 {moments}개보다 많아 식별 불가 "
                            "(시점 수를 늘리거나 동일 제약 선택)")
    if eng["error"]:
        notes.append(f"SEM 엔진 사전 점검 실패: {eng['error']}")
    elif eng["npar"] != b.npar:
        notes.append(f"SEM 엔진 해석 자유모수 {eng['npar']}개, 모형 예시 기대값 {b.npar}개 — 모형식 확인 필요")
    if df_expected == 0:
        notes.append("포화모형(자유도 0): 적합도 지수는 정보 없음")
    notes.append("결측은 SEM 분석 기본 설정(FIML, 가용 자료 전부 사용)으로 처리")
    try:
        diagram = _diagram(model, b.layout, b.covariates)
    except Exception:  # pragma: no cover - preview only
        diagram = {"nodes": [], "edges": []}
    measurement = []
    try:
        from . import sem_engine as _se
        for st_ in _se.parse_model(model):
            if st_.op == "=~":
                measurement.append({"latent": st_.lhs, "indicators": [t.name for t in st_.terms]})
    except Exception:  # pragma: no cover
        pass
    try:
        stata = to_stata(model)
    except TemplateError as e:
        stata = []
        notes.append(f"Stata sem 변환 불가: {e}")
    return {
        "template": tid,
        "title": spec["title"],
        "name_en": spec["name_en"],
        "heading": b.title_line,
        "model": model,
        "assumptions": list(b.assumptions),
        "notes": notes,
        "references": list(spec["references"]),
        "stata": stata,
        "diagram": diagram,
        "measurement": measurement,
        "options": o,
        "roles": {k: v for k, v in r.items()},
        "suggested": {
            "analysis": "sem",
            "params": {"model": model},
            "npar": eng["npar"] if eng["npar"] is not None else b.npar,
            "df": eng["df"] if eng["df"] is not None else df_expected,
            "npar_expected": b.npar,
            "df_expected": df_expected,
            "moments": moments,
            "observed": list(b.observed),
            "covariates": list(b.covariates),
            "latent": list(b.latent),
        },
    }


# ---------------------------------------------------------------------------------- wave detection
_WAVE_RE = re.compile(r"^(?P<stem>.*?[^\d_.\-\s])(?P<sep>[_.\-]?(?:wave|time|w|t)?[_.\-]?)(?P<num>\d{1,2})$", re.I)


def detect_wave_sets(columns: Sequence[str], min_waves: int = 2) -> List[Dict[str, Any]]:
    """Group columns such as 우울_1 … 우울_5, dep1 … dep4, x_w1 … into ordered wave lists.

    Returns [{"stem": "우울", "vars": ["우울_1", ...], "waves": [1, 2, ...]}] in column order."""
    groups: Dict[Tuple[str, str], List[Tuple[int, str, int]]] = {}
    for pos, c in enumerate(str(x) for x in columns):
        m = _WAVE_RE.match(c)
        if not m:
            continue
        key = (m.group("stem"), m.group("sep"))
        groups.setdefault(key, []).append((int(m.group("num")), c, pos))
    out = []
    for (stem, sep), items in groups.items():
        nums = [n for n, _, _ in items]
        if len(items) < min_waves or len(set(nums)) != len(nums):
            continue
        items.sort()
        out.append({"stem": stem, "vars": [c for _, c, _ in items], "waves": [n for n, _, _ in items],
                    "_pos": min(p for _, _, p in items)})
    out.sort(key=lambda d: d["_pos"])
    for d in out:
        d.pop("_pos")
    return out


def _indicator_groups(sets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Wave sets whose stems differ only in a trailing item code (우울A, 우울B, 우울C) → indicator grids."""
    by_prefix: Dict[str, List[Dict[str, Any]]] = {}
    for s in sets:
        m = re.match(r"^(.*?)[_.\-]?([A-Za-z]|\d{1,2})$", s["stem"])
        if not m or not m.group(1):
            continue
        by_prefix.setdefault(m.group(1), []).append(s)
    out = []
    for pre, lst in by_prefix.items():
        if len(lst) >= 2 and len({len(s["vars"]) for s in lst}) == 1:
            out.append({"construct": pre, "grid": [s["vars"] for s in lst]})
    return out


def suggest_roles(template_id: str, columns: Sequence[str]) -> Dict[str, Any]:
    """Role assignment proposed for a dataset (sample datasets first, then detected wave sets)."""
    tid = _resolve_id(template_id)
    cols = [str(c) for c in columns]
    sets = detect_wave_sets(cols, 2)
    by_stem = {s["stem"]: s["vars"] for s in sets}
    grids = _indicator_groups(sets)
    by_construct = {g["construct"]: g["grid"] for g in grids}
    min_t = _CAT[tid]["min_waves"]
    obs_sets = [s for s in sets if len(s["vars"]) >= min_t and not any(s["vars"] in g["grid"] for g in grids)]
    if not obs_sets:
        obs_sets = [s for s in sets if len(s["vars"]) >= min_t]

    def pick(*stems):
        return [by_stem[s] for s in stems if s in by_stem]

    preferred = {
        "clpm": ("건강", "의료이용"), "riclpm": ("우울", "외로움"), "lgm_linear": ("사회참여",),
        "lgm_quadratic": ("인지기능",), "lgm_basis": ("적응",), "lcs_univariate": ("자아존중감",),
        "lgm_parallel": ("사회참여", "여가활동"), "lgm_conditional": ("삶의만족도",),
    }
    roles: Dict[str, Any] = {}
    if tid in ("clpm", "riclpm", "lgm_parallel"):
        chosen = pick(*preferred[tid])
        if len(chosen) < 2:
            same = {}
            for s in obs_sets:
                same.setdefault(len(s["vars"]), []).append(s["vars"])
            best = max(same.values(), key=len) if same else []
            chosen = best[:2]
        if len(chosen) >= 2:
            roles = {"x_waves": chosen[0], "y_waves": chosen[1]}
    elif tid in ("lgm_linear", "lgm_quadratic", "lgm_basis", "lcs_univariate", "lgm_conditional"):
        chosen = pick(*preferred[tid]) or [s["vars"] for s in obs_sets[:1]]
        if chosen:
            roles = {"y_waves": chosen[0]}
            if tid == "lgm_conditional":
                if "삶의만족도" in by_stem:
                    roles["covariates"] = [c for c in ("연령", "소득") if c in cols]
                    if "건강" in by_stem and len(by_stem["건강"]) == len(chosen[0]):
                        roles["tvc_waves"] = by_stem["건강"]
                else:
                    others = [c for c in cols if c not in chosen[0] and not any(c in s["vars"] for s in sets)]
                    roles["covariates"] = others[:1]
    elif tid == "latent_clpm":
        chosen = [by_construct[c] for c in ("우울", "사회적지지") if c in by_construct] or [g["grid"] for g in grids[:2]]
        if len(chosen) >= 2:
            roles = {"x_indicators": chosen[0], "y_indicators": chosen[1]}
    elif tid == "lgm_second_order":
        chosen = [by_construct[c] for c in ("삶의질",) if c in by_construct] or [g["grid"] for g in grids[:1]]
        if chosen:
            roles = {"indicators": chosen[0]}
    return roles


# ---------------------------------------------------------------------------------- convenience object
class SEMTemplate(dict):
    """Result of `sem_template(...)`: a dict (JSON-serialisable) with attribute access and helpers.

        spec = sem_template("riclpm", x_waves=[...], y_waves=[...], constrain_lagged=True)
        print(spec)                 # model string + assumptions
        res = spec.fit(df)          # runs the existing SEM analysis with spec.model
    """

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __str__(self):
        lines = [f"[모형 예시] {self['heading']}", "", self["model"].rstrip(), "", "기본 가정"]
        lines += [f"  - {a}" for a in self["assumptions"]]
        if self.get("notes"):
            lines += ["", "참고"] + [f"  - {n}" for n in self["notes"]]
        s = self["suggested"]
        lines += ["", f"자유모수 {s['npar']}개, 자유도 {s['df']}"]
        return "\n".join(lines)

    def __repr__(self):
        return f"SEMTemplate({self['template']!r}, npar={self['suggested']['npar']}, df={self['suggested']['df']})"

    def fit(self, data, **sem_options):
        from .sem_module import SEM
        return SEM(data, model=self["model"], **sem_options)


def _sem_option_keys() -> set:
    keys = {"std_est", "stdEst", "fit_measures", "fitMeasures", "variances", "path_diagram", "pathDiagram"}
    try:
        from .sem_module import SEM_OPTION_KEYS, _ALIASES
        keys |= set(SEM_OPTION_KEYS) | set(_ALIASES) | set(_ALIASES.values())
    except Exception:  # pragma: no cover
        keys |= {"estimator", "missing", "information", "bootstrap", "indirect", "group", "seed", "ciType"}
    keys.discard("model")
    keys.discard("model_syntax")
    return keys


def split_kwargs(template: str, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Split keyword arguments into (variable roles, template options, SEM analysis options).

    Used by `sem_template` and by `sess.sem_template(...)` (which passes the SEM options on to `sess.sem`)."""
    tid = _resolve_id(template)
    spec = _CAT[tid]
    role_keys = {r["key"] for r in spec["roles"]} | {"waves"} | {
        r["key"] + "_by_wave" for r in spec["roles"] if r["key"].endswith("indicators")}
    opt_keys = {o["key"] for o in spec["options"]}
    sem_keys = _sem_option_keys()
    roles: Dict[str, Any] = {}
    opts: Dict[str, Any] = {}
    sem_opts: Dict[str, Any] = {}
    for k, v in kwargs.items():
        sk = _snake(k)
        sk = _ROLE_ALIASES.get(sk.replace("_", ""), _ROLE_ALIASES.get(sk, sk))
        if sk in role_keys:
            roles[sk] = v
        elif sk in opt_keys:
            opts[sk] = v
        elif k in sem_keys:
            sem_opts[k] = v
        else:
            raise TemplateError(f"{spec['title']}: 알 수 없는 인자 '{k}' (변수 역할: {', '.join(sorted(role_keys))}; "
                                f"옵션: {', '.join(sorted(opt_keys))}; SEM 옵션: estimator, missing, bootstrap 등)")
    return roles, opts, sem_opts


def sem_template(template: str, data=None, *, roles: Optional[Dict[str, Any]] = None,
                 options: Optional[Dict[str, Any]] = None, run: bool = False, **kwargs) -> SEMTemplate:
    """Convenience API (exposed as `ost.sem_template(...)` / `sess.sem_template(...)`).

    Keyword arguments are split into variable roles, template options and SEM options by the template catalog, so
        sem_template("lgm_linear", df, y_waves=["y1", "y2", "y3", "y4"], residual_var_equal=True)
    works. With `data`, variables are checked against its columns; `run=True` also fits the SEM analysis (SEM
    options such as estimator="mlr" go to it) and stores it in `result["analysis"]`."""
    r_kw, o_kw, sem_opts = split_kwargs(template, kwargs)
    r = dict(roles or {})
    r.update(r_kw)
    o = dict(options or {})
    o.update(o_kw)
    columns = list(data.columns) if data is not None and hasattr(data, "columns") else None
    out = SEMTemplate(build_template(template, r, o, columns))
    if run:
        if data is None:
            raise TemplateError("run=True에는 data 필요")
        out["analysis"] = out.fit(data, **sem_opts)
    elif sem_opts:
        out["sem_options"] = sem_opts
    return out


def catalog_payload(columns: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Payload of GET /api/sem/templates: catalog, detected wave sets and per-template role proposals."""
    out: Dict[str, Any] = {"templates": template_catalog(), "groups": GROUPS}
    if columns is not None:
        sets = detect_wave_sets(columns)
        out["wave_sets"] = sets
        out["indicator_sets"] = _indicator_groups(sets)
        sug = {}
        for t in TEMPLATE_CATALOG:
            try:
                sug[t["id"]] = suggest_roles(t["id"], columns)
            except Exception:
                sug[t["id"]] = {}
        out["suggested_roles"] = sug
    return out


# ============================================================================================ extension
EXTENSION = {
    # templates feed the existing `sem` analysis (params["model"]); no separate analysis id
    "catalog": [],
    "aliases": {},
    "api": {"build_template": build_template, "sem_template": sem_template, "split_kwargs": split_kwargs,
            "to_stata": to_stata,
            "catalog": template_catalog,
            "catalog_payload": catalog_payload, "suggest_roles": suggest_roles,
            "detect_wave_sets": detect_wave_sets},
}

__all__ = ["TEMPLATE_CATALOG", "TemplateError", "build_template", "sem_template", "split_kwargs", "to_stata", "SEMTemplate",
           "template_catalog", "detect_wave_sets", "suggest_roles", "catalog_payload", "EXTENSION"]
