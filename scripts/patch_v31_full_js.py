#!/usr/bin/env python3
"""
pyjamovi v3.1 JavaScript Engine Patch
Injects:
- Extended Korean Dictionary (OPTION_LABELS_KO) & Option Panel i18n
- Layout Management (setLayoutMode, toggleLayoutMenu, switchRightView, startHResize)
- Variables Explorer (renderLeftVariablesExplorer, filterLeftVariables, insertVarToSyntax)
- Integrated Syntax & Stata Command Runner (executeIntegratedSyntax, executeInlineCommand, insertSyntaxSnippet)
- JASP Analyses Schemas (summaryTTest, networkAnalysis)
- Synthetic Data Modal & Generation Runner
"""

import re
import sys

HTML_PATH = "/mnt/share/temp/pyjamovi/reconstructed/pyjamovi/web/index.html"

with open(HTML_PATH, "r", encoding="utf-8") as f:
    html = f.read()

# ---------------------------------------------------------------------------
# 1. Update Ribbon Menus with JASP
# ---------------------------------------------------------------------------
# In ko ribbon ttests
if '{ id: "summaryTTest"' not in html:
    html = html.replace(
        '{ id: "bayesTTestIS", label: "베이지안 독립표본 T-검정", badge: "BF₁₀" }',
        '{ id: "summaryTTest", label: "요약 통계량 T-검정 (JASP Summary Stats)", badge: "JASP" },\n            { id: "bayesTTestIS", label: "베이지안 독립표본 T-검정", badge: "BF₁₀" }'
    )
    html = html.replace(
        '{ id: "bayesTTestIS", label: "Bayesian Independent T-Test", badge: "BF₁₀" }',
        '{ id: "summaryTTest", label: "Summary Statistics T-Test (JASP)", badge: "JASP" },\n            { id: "bayesTTestIS", label: "Bayesian Independent T-Test", badge: "BF₁₀" }'
    )
    print("[✓] Added summaryTTest to ttests ribbon menu.")

if '{ id: "networkAnalysis"' not in html:
    html = html.replace(
        '{ id: "efa", label: "탐색적 요인분석 (EFA)" }',
        '{ id: "efa", label: "탐색적 요인분석 (EFA)" },\n            { sep: true },\n            { section: "네트워크 분석 (JASP)" },\n            { id: "networkAnalysis", label: "네트워크 분석 (EBICglasso / 중심성)", badge: "JASP" }'
    )
    html = html.replace(
        '{ id: "efa", label: "Exploratory Factor Analysis (EFA)" }',
        '{ id: "efa", label: "Exploratory Factor Analysis (EFA)" },\n            { sep: true },\n            { section: "Network Analysis (JASP)" },\n            { id: "networkAnalysis", label: "Network Analysis (EBICglasso / Centrality)", badge: "JASP" }'
    )
    print("[✓] Added networkAnalysis to factor ribbon menu.")

# ---------------------------------------------------------------------------
# 2. Update Ribbon Button Labels with data-i18n
# ---------------------------------------------------------------------------
pane_plots_old = """            <!-- TAB 4: Graphs & Design (Standalone 15-Type Graph Builder -> Analysis Results) -->
            <div class="ribbon-pane" id="pane-plots">
                <button class="ribbon-btn" onclick="openAnalysis('graph')">
                    <img class="rb-icon" src="assets/analysis-scatr-jmvbar.svg" alt="Graph Builder">
                    <span class="rb-label">Graph Builder</span>
                </button>
                <button class="ribbon-btn" onclick="openPlotDesignModal(null)">
                    <img class="rb-icon" src="assets/action-palette.svg" alt="Plot Design Studio">
                    <span class="rb-label">Plot Design</span>
                </button>
                <div class="ribbon-sep"></div>
                <button class="ribbon-btn" onclick="openAnalysisWithPreset('graph', {chartType: 'bar', showValues: true})">
                    <img class="rb-icon" src="assets/analysis-scatr-jmvbar.svg" alt="Bar">
                    <span class="rb-label">Bar / Stacked</span>
                </button>
                <button class="ribbon-btn" onclick="openAnalysisWithPreset('graph', {chartType: 'line'})">
                    <img class="rb-icon" src="assets/analysis-linearmodels.svg" alt="Line">
                    <span class="rb-label">Line / Area</span>
                </button>
                <button class="ribbon-btn" onclick="openAnalysisWithPreset('graph', {chartType: 'donut'})">
                    <img class="rb-icon" src="assets/analysis-frequencies.svg" alt="Pie / Donut">
                    <span class="rb-label">Pie / Donut</span>
                </button>
                <button class="ribbon-btn" onclick="openAnalysisWithPreset('graph', {chartType: 'raincloud'})">
                    <img class="rb-icon" src="assets/analysis-scatr-jmvbox.svg" alt="Raincloud">
                    <span class="rb-label">Raincloud / Box</span>
                </button>
                <button class="ribbon-btn" onclick="openAnalysisWithPreset('graph', {chartType: 'histogram'})">
                    <img class="rb-icon" src="assets/analysis-scatr-jmvhist.svg" alt="Histogram">
                    <span class="rb-label">Histogram / KDE</span>
                </button>
                <button class="ribbon-btn" onclick="openAnalysisWithPreset('graph', {chartType: 'scatter'})">
                    <img class="rb-icon" src="assets/analysis-scatr-jmvscatter.svg" alt="Scatter">
                    <span class="rb-label">Scatter / Bubble</span>
                </button>
                <button class="ribbon-btn" onclick="openAnalysisWithPreset('graph', {chartType: 'heatmap'})">
                    <img class="rb-icon" src="assets/analysis-regression.svg" alt="Heatmap">
                    <span class="rb-label">Heatmap / Matrix</span>
                </button>
                <button class="ribbon-btn" onclick="openAnalysisWithPreset('graph', {chartType: 'errorbar'})">
                    <img class="rb-icon" src="assets/analysis-anova.svg" alt="Error-Bar">
                    <span class="rb-label">CI Error-Bar</span>
                </button>
                <button class="ribbon-btn" onclick="openAnalysis('pareto')">
                    <img class="rb-icon" src="assets/analysis-scatr-jmvbar.svg" alt="Pareto">
                    <span class="rb-label">Pareto</span>
                </button>
            </div>"""

pane_plots_new = """            <!-- TAB 4: Graphs & Design (Standalone 15-Type Graph Builder -> Analysis Results) -->
            <div class="ribbon-pane" id="pane-plots">
                <button class="ribbon-btn" onclick="openAnalysis('graph')">
                    <img class="rb-icon" src="assets/analysis-scatr-jmvbar.svg" alt="Graph Builder">
                    <span class="rb-label" data-i18n="p_graph_builder">차트 빌더</span>
                </button>
                <button class="ribbon-btn" onclick="openPlotDesignModal(null)">
                    <img class="rb-icon" src="assets/action-palette.svg" alt="Plot Design Studio">
                    <span class="rb-label" data-i18n="p_plot_design">그래프 디자인</span>
                </button>
                <div class="ribbon-sep"></div>
                <button class="ribbon-btn" onclick="openAnalysisWithPreset('graph', {chartType: 'bar', showValues: true})">
                    <img class="rb-icon" src="assets/analysis-scatr-jmvbar.svg" alt="Bar">
                    <span class="rb-label" data-i18n="p_bar">막대 / 누적 막대</span>
                </button>
                <button class="ribbon-btn" onclick="openAnalysisWithPreset('graph', {chartType: 'line'})">
                    <img class="rb-icon" src="assets/analysis-linearmodels.svg" alt="Line">
                    <span class="rb-label" data-i18n="p_line">선그래프 / 영역</span>
                </button>
                <button class="ribbon-btn" onclick="openAnalysisWithPreset('graph', {chartType: 'donut'})">
                    <img class="rb-icon" src="assets/analysis-frequencies.svg" alt="Pie / Donut">
                    <span class="rb-label" data-i18n="p_donut">원형 / 도넛</span>
                </button>
                <button class="ribbon-btn" onclick="openAnalysisWithPreset('graph', {chartType: 'raincloud'})">
                    <img class="rb-icon" src="assets/analysis-scatr-jmvbox.svg" alt="Raincloud">
                    <span class="rb-label" data-i18n="p_raincloud">레인클라우드 / 상자</span>
                </button>
                <button class="ribbon-btn" onclick="openAnalysisWithPreset('graph', {chartType: 'histogram'})">
                    <img class="rb-icon" src="assets/analysis-scatr-jmvhist.svg" alt="Histogram">
                    <span class="rb-label" data-i18n="p_hist">히스토그램 / 밀도</span>
                </button>
                <button class="ribbon-btn" onclick="openAnalysisWithPreset('graph', {chartType: 'scatter'})">
                    <img class="rb-icon" src="assets/analysis-scatr-jmvscatter.svg" alt="Scatter">
                    <span class="rb-label" data-i18n="p_scatter">산점도 / 버블</span>
                </button>
                <button class="ribbon-btn" onclick="openAnalysisWithPreset('graph', {chartType: 'heatmap'})">
                    <img class="rb-icon" src="assets/analysis-regression.svg" alt="Heatmap">
                    <span class="rb-label" data-i18n="p_heatmap">히트맵 / 상관행렬</span>
                </button>
                <button class="ribbon-btn" onclick="openAnalysisWithPreset('graph', {chartType: 'errorbar'})">
                    <img class="rb-icon" src="assets/analysis-anova.svg" alt="Error-Bar">
                    <span class="rb-label" data-i18n="p_errorbar">오차막대 (95% CI)</span>
                </button>
                <button class="ribbon-btn" onclick="openAnalysis('pareto')">
                    <img class="rb-icon" src="assets/analysis-scatr-jmvbar.svg" alt="Pareto">
                    <span class="rb-label" data-i18n="p_pareto">파레토 품질도표</span>
                </button>
            </div>"""

if pane_plots_old in html:
    html = html.replace(pane_plots_old, pane_plots_new)
    print("[✓] Localized pane-plots buttons.")

pane_edit_old = """            <!-- TAB 5: Python (.py) & Multi-Format Export -->
            <div class="ribbon-pane" id="pane-edit">
                <button class="ribbon-btn" onclick="togglePyDrawer(true)">
                    <img class="rb-icon" src="assets/action-code.svg" alt="Live .py Editor">
                    <span class="rb-label">Live .py Editor</span>
                </button>
                <button class="ribbon-btn" onclick="window.location.href='/api/syntax/export_py'">
                    <img class="rb-icon" src="assets/analysis-r.svg" alt="Save .py">
                    <span class="rb-label">Download .py</span>
                </button>
                <button class="ribbon-btn" onclick="toggleSyntaxModeUI()">
                    <img class="rb-icon" src="assets/action-check.svg" alt="Inline Syntax">
                    <span class="rb-label">Inline Syntax</span>
                </button>
                <div class="ribbon-sep"></div>
                <button class="ribbon-btn" onclick="openRibbonMenu(event, 'export_tables')">
                    <img class="rb-icon" src="assets/action-export.svg" alt="Export Results">
                    <span class="rb-label">Export Results <span class="rb-caret">▾</span></span>
                </button>
                <button class="ribbon-btn" onclick="openPlotDesignModal(null)">
                    <img class="rb-icon" src="assets/action-palette.svg" alt="Plot Design">
                    <span class="rb-label">Plot Design</span>
                </button>
                <button class="ribbon-btn" onclick="window.location.href='/api/dataset/export_omv'">
                    <img class="rb-icon" src="assets/action-save.svg" alt="Save .omv">
                    <span class="rb-label">Save .omv</span>
                </button>
                <button class="ribbon-btn" onclick="clearAllResults()">
                    <img class="rb-icon" src="assets/action-close.svg" alt="Clear Results">
                    <span class="rb-label">Clear Results</span>
                </button>
            </div>"""

pane_edit_new = """            <!-- TAB 5: Python (.py) & Multi-Format Export -->
            <div class="ribbon-pane" id="pane-edit">
                <button class="ribbon-btn" onclick="togglePyDrawer(true)">
                    <img class="rb-icon" src="assets/action-code.svg" alt="Live .py Editor">
                    <span class="rb-label" data-i18n="e_live_py">실시간 .py 에디터</span>
                </button>
                <button class="ribbon-btn" onclick="window.location.href='/api/syntax/export_py'">
                    <img class="rb-icon" src="assets/analysis-r.svg" alt="Save .py">
                    <span class="rb-label" data-i18n="e_dl_py">.py 다운로드</span>
                </button>
                <button class="ribbon-btn" onclick="toggleSyntaxModeUI()">
                    <img class="rb-icon" src="assets/action-check.svg" alt="Inline Syntax">
                    <span class="rb-label" data-i18n="e_inline_syntax">인라인 신택스</span>
                </button>
                <div class="ribbon-sep"></div>
                <button class="ribbon-btn" onclick="openRibbonMenu(event, 'export_tables')">
                    <img class="rb-icon" src="assets/action-export.svg" alt="Export Results">
                    <span class="rb-label"><span data-i18n="e_export_results">결과 내보내기</span> <span class="rb-caret">▾</span></span>
                </button>
                <button class="ribbon-btn" onclick="openPlotDesignModal(null)">
                    <img class="rb-icon" src="assets/action-palette.svg" alt="Plot Design">
                    <span class="rb-label" data-i18n="e_plot_design">그래프 디자인</span>
                </button>
                <button class="ribbon-btn" onclick="window.location.href='/api/dataset/export_omv'">
                    <img class="rb-icon" src="assets/action-save.svg" alt="Save .omv">
                    <span class="rb-label" data-i18n="e_save_omv">.omv 저장</span>
                </button>
                <button class="ribbon-btn" onclick="clearAllResults()">
                    <img class="rb-icon" src="assets/action-close.svg" alt="Clear Results">
                    <span class="rb-label" data-i18n="e_clear_results">결과 지우기</span>
                </button>
            </div>"""

if pane_edit_old in html:
    html = html.replace(pane_edit_old, pane_edit_new)
    print("[✓] Localized pane-edit buttons.")

# ---------------------------------------------------------------------------
# 3. Add JASP schemas to ANALYSIS_SCHEMAS
# ---------------------------------------------------------------------------
jasp_schemas = """    summaryTTest: {
        title: "Summary Statistics T-Test (JASP)",
        slots: [],
        sections: [
            {
                title: "Group 1 Parameters",
                items: [
                    { type: "number", key: "m1", label: "Mean (M₁)", def: 20.0, step: 0.1 },
                    { type: "number", key: "s1", label: "Std. Dev. (s₁)", def: 4.0, step: 0.1 },
                    { type: "number", key: "n1", label: "Sample Size (n₁)", def: 30, step: 1 }
                ]
            },
            {
                title: "Group 2 Parameters",
                items: [
                    { type: "number", key: "m2", label: "Mean (M₂)", def: 16.0, step: 0.1 },
                    { type: "number", key: "s2", label: "Std. Dev. (s₂)", def: 3.5, step: 0.1 },
                    { type: "number", key: "n2", label: "Sample Size (n₂)", def: 30, step: 1 }
                ]
            },
            {
                title: "Prior & Hypothesis",
                items: [
                    { type: "number", key: "priorWidth", label: "Cauchy Prior Width (r)", def: 0.707, step: 0.05 }
                ]
            }
        ]
    },
    networkAnalysis: {
        title: "Network Analysis (JASP / bootnet)",
        slots: [
            { key: "vars", label: "Variables", multi: true, req: true, prefer: "num" }
        ],
        sections: [
            {
                title: "Estimation Method",
                items: [
                    {
                        type: "select", key: "method", label: "Network Estimator",
                        options: [["ebicglasso", "EBICglasso (Regularized Partial Correlation)"], ["ridge", "Ridge Shrinkage Inverse Covariance"]]
                    },
                    { type: "number", key: "gamma", label: "EBIC Tuning Parameter (γ)", def: 0.25, step: 0.05 }
                ]
            },
            {
                title: "Plots & Visualizations",
                items: [
                    { type: "check", key: "plotNetwork", label: "Network Plot (Partial Correlations)", def: true },
                    { type: "check", key: "plotCentrality", label: "Centrality Comparison Plot (z-scores)", def: true }
                ]
            }
        ]
    },"""

if "summaryTTest:" not in html:
    html = html.replace("const ANALYSIS_SCHEMAS = {", "const ANALYSIS_SCHEMAS = {\n" + jasp_schemas)
    print("[✓] Added summaryTTest and networkAnalysis to ANALYSIS_SCHEMAS.")

# ---------------------------------------------------------------------------
# 4. Inject Full Comprehensive OPTION_LABELS_KO and Runtime Engine Code
# ---------------------------------------------------------------------------
js_runtime_code = """
/* ========================================================================= */
/* OpenStat Studio v3.1 Layout, Variables Explorer & Stata Console Engine     */
/* ========================================================================= */

// Extended Korean Statistical Dictionary for All Option Slots, Sections & Controls
const COMPREHENSIVE_KO_LABELS = {
    // Analysis Titles
    "Descriptives": "기술통계량",
    "Independent Samples T-Test": "독립표본 T-검정",
    "Paired Samples T-Test": "대응표본 T-검정",
    "One Sample T-Test": "일표본 T-검정",
    "Bayesian Independent T-Test": "베이지안 독립표본 T-검정",
    "Summary Statistics T-Test (JASP)": "요약 통계량 T-검정 (JASP)",
    "Network Analysis (JASP / bootnet)": "네트워크 분석 (JASP / bootnet)",
    "One-Way ANOVA": "일원배치 분산분석",
    "ANOVA (Factorial)": "요인 분산분석 (Factorial ANOVA)",
    "Repeated Measures ANOVA": "반복측정 분산분석",
    "ANCOVA": "공분산분석 (ANCOVA)",
    "One-Way ANOVA (Non-parametric)": "크루스칼-왈리스 비모수 분산분석",
    "Bayesian ANOVA": "베이지안 분산분석",
    "Correlation Matrix": "상관행렬 분석",
    "Partial Correlation": "편상관분석",
    "Bayesian Correlation": "베이지안 상관행렬",
    "Linear Regression": "다중선형회귀분석",
    "2 Outcomes (Binomial Logistic)": "이항 로지스틱 회귀분석",
    "Independent Samples (χ² Association)": "교차분할표 독립성 검정 (χ²)",
    "Reliability Analysis": "신뢰도 분석 (Cronbach's α & ω)",
    "Principal Component Analysis": "주성분분석 (PCA)",
    "Exploratory Factor Analysis": "탐색적 요인분석 (EFA)",
    "Structural Equation Modeling (SEMLj)": "구조방정식 모형 (SEM / CFA)",
    "Mediation": "매개효과 분석",
    "Moderation": "조절효과 분석",
    "Panel Econometrics (FE / RE / Hausman)": "패널 데이터 계량분석",
    "Linear Mixed Model (Multilevel)": "선형 혼합모형 (다층모형)",
    "Kaplan-Meier & Cox PH Survival": "생존분석 (Kaplan-Meier & Cox)",
    "Graph Builder (15 Chart Types)": "논문용 독립형 차트 빌더 (15종)",

    // Slot Labels
    "Variables": "분석 변수",
    "Split by": "집단 구분 변수 (Split by)",
    "Dependent Variable": "종속변수 (Dependent Variable)",
    "Grouping Variable": "그룹 변수 (Grouping Variable)",
    "Dependent Variables": "종속변수군",
    "Grouping": "집단 구분",
    "Pairs": "대응 변수 쌍 (Pairs)",
    "Factors": "모형 요인 (Factors)",
    "Covariates": "공변량 (Covariates)",
    "Repeated Measures Factors": "반복측정 요인",
    "Between Subject Factors": "개체간 요인",
    "Rows": "행 변수 (Row)",
    "Columns": "열 변수 (Column)",
    "Predictor": "독립변수 (X)",
    "Mediator": "매개변수 (M)",
    "Moderator": "조절변수 (W)",
    "Entity (ID) Variable": "개체 고유 ID 변수",
    "Time Variable (Wave/Year)": "시간/시점 변수",
    "Cluster (Group) Variable": "군집/집락 변수",
    "Time Variable": "생존시간 변수",
    "Status / Event Variable": "사건발생 여부 (0/1)",

    // Sections
    "Statistics": "기술통계량 선택",
    "Dispersion & Range": "산포도 및 범위",
    "Distribution": "분포 특성 (왜도/첨도)",
    "Tests": "가설검정 기법",
    "Hypothesis": "가설 검정 방향 (양측/단측)",
    "Additional Statistics": "추가 통계량",
    "Assumption Checks": "가정 검정 (정규성 / 등분산성)",
    "Plots": "도표 및 시각화",
    "Variances": "분산 동질성 가정",
    "Post Hoc Tests": "사후검정 (Post Hoc)",
    "Model Comparison": "모형 비교 및 베이즈 요인",
    "Correlation Coefficients": "상관계수 기법",
    "Model Fit": "모형 적합도 지수",
    "Coefficients & Diagnostics": "회귀계수 및 다중공선성 진단",
    "Residual Plots": "잔차 진단 플롯",
    "Association Tests": "연관성 검정",
    "Nominal Measures": "명목 척도 연관도",
    "Percentages": "비율 백분율 (%)",
    "Reliability Coefficients": "신뢰도 계수",
    "Item Statistics": "항목별 분석",
    "Extraction & Rotation": "요인 추출 및 회전",
    "Model Syntax": "구조방정식 모형 구문",
    "Estimates & Indices": "모수 추정치 및 적합도",
    "Effect Decomposition": "효과 분해 (직접/간접/총효과)",
    "Model Selection": "패널 모형 선택 (하우스만)",
    "Survival Models": "생존 모형 및 로그순위",
    "Group 1 Parameters": "그룹 1 요약 모수 (M₁, s₁, n₁)",
    "Group 2 Parameters": "그룹 2 요약 모수 (M₂, s₂, n₂)",
    "Prior & Hypothesis": "사전분포 폭 및 가설",
    "Estimation Method": "네트워크 추정 기법",
    "Plots & Visualizations": "네트워크 구조 및 중심성 도표",

    // Items
    "Student's": "스튜던트 t-검정 (등분산 가정)",
    "Welch's": "웰치 t-검정 (등분산 위배 시 권장)",
    "Mann-Whitney U": "맨-휘트니 U 비모수 검정",
    "Wilcoxon W": "윌콕슨 부호순위 검정 (대응)",
    "Mean difference": "평균 차이 (Mean Difference)",
    "Effect size": "효과크기 (Cohen's d)",
    "Homogeneity of variances": "등분산성 검정 (Levene's)",
    "Normality": "정규성 검정 (Shapiro-Wilk)",
    "Descriptives plot": "평균 및 95% 신뢰구간 도표",
    "Descriptives": "기술통계량 표",
    "N": "표본 크기 (N)",
    "Missing": "결측값 수 (Missing)",
    "Mean": "평균 (Mean)",
    "Median": "중앙값 (Median)",
    "Mode": "최빈값 (Mode)",
    "Std. deviation": "표준편차 (SD)",
    "Variance": "분산 (Variance)",
    "Skewness": "왜도 (Skewness)",
    "Kurtosis": "첨도 (Kurtosis)",
    "Shapiro-Wilk": "정규성 검정 (Shapiro-Wilk)",
    "Confidence interval": "95% 신뢰구간",
    "Bar plot": "막대도표 (Bar Plot)",
    "Box plot": "상자도표 (Box Plot)",
    "Violin plot": "바이올린 도표",
    "Raincloud plot": "레인클라우드 도표",
    "Tukey": "Tukey HSD 사후검정",
    "Scheffé": "Scheffé 사후검정",
    "Bonferroni": "Bonferroni 보정",
    "Holm": "Holm 보정",
    "Games-Howell": "Games-Howell 사후검정 (이분산 사후검정)",
    "Collinearity statistics": "다중공선성 진단 (VIF & 공차한계)",
    "Durbin-Watson": "더빈-왓슨 잔차 자기상관",
    "Cook's distance": "Cook의 거리 이상치",
    "Q-Q plot": "정규 Q-Q 도표",
    "Standardized estimate": "표준화 계수 (Beta)",
    "Path diagram": "경로 모형도 (Path Diagram)",
    "Odds ratio": "승산비 (Odds Ratio, Exp(B))",
    "ROC curve": "ROC / AUC 곡선",
    "Chi-square": "카이제곱 검정 (χ²)",
    "Fisher's exact test": "피셔 직접확률 검정",
    "Cramér's V": "크래머 V 계수",
    "Cronbach's alpha": "크론바흐 알파 (Cronbach α)",
    "McDonald's omega": "맥도날드 오메가 (McDonald ω)",
    "Scree plot": "스크리 고윳값 감쇠도표",
    "Hausman test": "하우스만 사양검정 (FE vs RE)",
    "Network Plot (Partial Correlations)": "네트워크 구조망 (양:청색, 음:적색)",
    "Centrality Comparison Plot (z-scores)": "중심성 지표 비교 플롯 (Z-Score)"
};

Object.assign(OPTION_LABELS_KO, COMPREHENSIVE_KO_LABELS);

// --- Window Layout Controller ---
function setLayoutMode(mode) {
    document.body.setAttribute("data-layout", mode);
    document.body.setAttribute("data-split", mode);
    localStorage.setItem("openstat_layout", mode);

    const labels = {
        "default": "🏛️ 기본 (변수+신택스 | 결과)",
        "jamovi": "📊 자모비 클래식 (데이터 | 결과)",
        "syntax": "📝 신택스 집중 (Do-file | 결과)",
        "quad": "▦ 4분할 (변수/신택스 | 데이터/결과)",
        "results-only": "📄 결과 전체화면",
        "data-only": "📋 데이터 전체화면"
    };

    const lbl = document.getElementById("current-layout-label");
    if (lbl) lbl.innerText = labels[mode] || mode;

    document.querySelectorAll(".layout-dd-item").forEach(el => {
        el.classList.toggle("active", el.id === "layout-item-" + mode);
    });

    closeLayoutMenu();

    if (mode === "jamovi" || mode === "quad" || mode === "data-only") {
        renderGrid();
    }
    if (mode === "default" || mode === "quad") {
        renderLeftVariablesExplorer();
        syncLeftSyntaxEditor();
    }
}

function toggleLayoutMenu(ev) {
    if (ev) ev.stopPropagation();
    const dd = document.getElementById("layout-dropdown");
    if (dd) dd.classList.toggle("open");
}

function closeLayoutMenu() {
    const dd = document.getElementById("layout-dropdown");
    if (dd) dd.classList.remove("open");
}

document.addEventListener("click", (e) => {
    if (!e.target.closest(".layout-selector-wrap")) closeLayoutMenu();
    if (!e.target.closest("#syntax-snippets-dropdown") && !e.target.closest(".py-btn")) {
        const sn = document.getElementById("syntax-snippets-dropdown");
        if (sn) sn.style.display = "none";
    }
});

// --- Left Variables Explorer ---
function renderLeftVariablesExplorer() {
    const tbody = document.getElementById("left-var-tbody");
    if (!tbody) return;
    const filterText = (document.getElementById("left-var-filter")?.value || "").toLowerCase().trim();
    const badge = document.getElementById("left-var-count-badge");
    if (badge) badge.innerText = `${APP.columns.length}개 변수`;

    let html = "";
    APP.columns.forEach((col, idx) => {
        if (filterText && !col.name.toLowerCase().includes(filterText)) return;
        const missing = col.missing !== undefined ? col.missing : 0;
        const dtype = col.type || (col.measureType === "Continuous" ? "float" : "obj");
        html += `<tr onclick="selectLeftVarRow('${col.name.replace(/'/g, "\\'")}', this)" ondblclick="insertVarToSyntax('${col.name.replace(/'/g, "\\'")}')" title="더블클릭 시 신택스에 변수명 삽입">
            <td style="color:#94A3B8;">${idx + 1}</td>
            <td><img src="${measureIcon(col.measureType)}" style="width:14px;height:14px;vertical-align:-2px;" alt=""></td>
            <td style="font-weight:600;color:#0F172A;">${escapeHtml(col.name)}</td>
            <td style="color:#64748B;font-size:10.5px;">${escapeHtml(dtype)}</td>
            <td style="text-align:right;color:${missing > 0 ? '#E11D48' : '#94A3B8'};">${missing}</td>
            <td style="text-align:center;"><span class="var-quick-add" onclick="event.stopPropagation(); insertVarToSyntax('${col.name.replace(/'/g, "\\'")}')" title="신택스에 삽입">+</span></td>
        </tr>`;
    });
    tbody.innerHTML = html;
}

function filterLeftVariables(query) {
    renderLeftVariablesExplorer();
}

function selectLeftVarRow(varName, rowEl) {
    document.querySelectorAll(".left-var-table tr").forEach(r => r.classList.remove("selected"));
    if (rowEl) rowEl.classList.add("selected");
    APP.selectedSupplierVar = varName;
    if (APP.activeSlotKey) {
        assignSelectedToSlot(APP.activeSlotKey);
    }
}

function insertVarToSyntax(varName) {
    const ed = document.getElementById("integrated-syntax-editor");
    if (!ed) return;
    const start = ed.selectionStart;
    const end = ed.selectionEnd;
    const text = ed.value;
    const before = text.substring(0, start);
    const after = text.substring(end, text.length);
    ed.value = before + "'" + varName + "'" + after;
    ed.selectionStart = ed.selectionEnd = start + varName.length + 2;
    ed.focus();
    showToast(`변수 '${varName}' 신택스 에디터에 삽입됨`);
}

// --- Horizontal Splitter Dragging ---
let isHResizing = false;
function startHResize(e) {
    e.preventDefault();
    isHResizing = true;
    document.addEventListener("mousemove", handleHResize);
    document.addEventListener("mouseup", stopHResize);
}
function handleHResize(e) {
    if (!isHResizing) return;
    const leftPane = document.getElementById("left-pane");
    if (!leftPane) return;
    const rect = leftPane.getBoundingClientRect();
    const offsetY = e.clientY - rect.top;
    const pct = Math.max(15, Math.min(80, (offsetY / rect.height) * 100));
    document.body.style.setProperty("--left-top-height", pct + "%");
    const topBox = document.getElementById("left-var-explorer");
    if (topBox) topBox.style.height = pct + "%";
}
function stopHResize() {
    isHResizing = false;
    document.removeEventListener("mousemove", handleHResize);
    document.removeEventListener("mouseup", stopHResize);
}

// --- Integrated Syntax & Command Console Execution ---
function syncLeftSyntaxEditor() {
    const ed = document.getElementById("integrated-syntax-editor");
    if (ed && !ed.value.trim() && APP.pythonCode) {
        ed.value = APP.pythonCode;
    }
}

async function executeIntegratedSyntax(selectedOnly) {
    const ed = document.getElementById("integrated-syntax-editor");
    if (!ed) return;
    let code = ed.value;
    if (selectedOnly) {
        const start = ed.selectionStart;
        const end = ed.selectionEnd;
        if (start !== end) {
            code = code.substring(start, end).trim();
        } else {
            // Pick current line
            const lines = code.substring(0, start).split("\\n");
            const allLines = code.split("\\n");
            const lineIdx = lines.length - 1;
            code = allLines[lineIdx].trim();
        }
    }

    if (!code) {
        showToast("실행할 신택스 코드를 입력하거나 선택하십시오.");
        return;
    }

    const badge = document.getElementById("syntax-status-badge");
    if (badge) { badge.innerText = "실행 중..."; badge.className = "badge-pill"; }

    try {
        const res = await fetch("/api/syntax/execute", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code: code, replace_existing: !selectedOnly })
        });
        if (res.ok) {
            const data = await res.json();
            applyDatasetPayload(data);
            showToast(selectedOnly ? "선택 블록 실행 완료" : "전체 신택스 파이프라인 실행 완료");
            if (badge) { badge.innerText = "정상 완료"; badge.className = "badge-pill green"; }
        } else {
            const err = await res.json();
            showToast("실행 오류: " + (err.detail || "알 수 없는 오류"), 4000);
            if (badge) { badge.innerText = "오류 발생"; badge.className = "badge-pill"; }
        }
    } catch (e) {
        showToast("실행 통신 오류: " + e, 4000);
        if (badge) { badge.innerText = "네트워크 오류"; badge.className = "badge-pill"; }
    }
}

async function executeInlineCommand() {
    const input = document.getElementById("stata-inline-cmd");
    if (!input || !input.value.trim()) return;
    const cmd = input.value.trim();

    const ed = document.getElementById("integrated-syntax-editor");
    if (ed) {
        ed.value += (ed.value.endsWith("\\n") ? "" : "\\n") + cmd;
        ed.scrollTop = ed.scrollHeight;
    }

    try {
        const res = await fetch("/api/syntax/execute", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code: cmd, replace_existing: false })
        });
        if (res.ok) {
            const data = await res.json();
            applyDatasetPayload(data);
            showToast(`명령어 실행 완료: ${cmd}`);
            input.value = "";
        } else {
            const err = await res.json();
            showToast(`명령 오류: ${err.detail || "실행 실패"}`, 4000);
        }
    } catch (e) {
        showToast("명령 전송 오류: " + e, 4000);
    }
}

function clearIntegratedSyntax() {
    const ed = document.getElementById("integrated-syntax-editor");
    if (ed) ed.value = "";
    showToast("신택스 에디터가 비워졌습니다.");
}

function saveIntegratedSyntax() {
    const ed = document.getElementById("integrated-syntax-editor");
    if (!ed) return;
    const blob = new Blob([ed.value], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = (APP.datasetName || "analysis_script").replace(/\\.[^.]+$/, "") + ".py";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast("스크립트 파일(.py) 저장 완료");
}

function toggleSyntaxSnippetsDropdown(ev) {
    ev.stopPropagation();
    const dd = document.getElementById("syntax-snippets-dropdown");
    if (!dd) return;
    if (dd.style.display === "block") {
        dd.style.display = "none";
        return;
    }
    const rect = ev.currentTarget.getBoundingClientRect();
    dd.style.top = (rect.bottom + 4) + "px";
    dd.style.left = Math.min(rect.left, window.innerWidth - 310) + "px";
    dd.style.display = "block";
}

function insertSyntaxSnippet(type) {
    const ed = document.getElementById("integrated-syntax-editor");
    const numCols = APP.columns.filter(c => c.measureType === "Continuous").map(c => c.name);
    const catCols = APP.columns.filter(c => c.measureType !== "Continuous").map(c => c.name);
    const c1 = numCols[0] || "len";
    const c2 = numCols[1] || "dose";
    const g1 = catCols[0] || "supp";

    const snippets = {
        "desc": `sess.descriptives(vars=['${c1}', '${c2}'])`,
        "ttest": `sess.ttest(dep='${c1}', group='${g1}', welchs=True)`,
        "anova": `sess.anova(dep='${c1}', group='${g1}', welchs=True)`,
        "reg": `sess.regression(dep='${c1}', covs=['${c2}'])`,
        "corr": `sess.corr_matrix(vars=['${c1}', '${c2}'])`,
        "sem": `sess.sem(formula='Visual =~ x1 + x2 + x3\\nOutcome ~ Visual')`,
        "panel": `sess.panel(entity='id', time='year', dep='${c1}', covs=['${c2}'], model_type='fixed')`,
        "survival": `sess.survival(time_col='time', event_col='status', group_col='treat')`,
        "network": `sess.network_analysis(vars=['${c1}', '${c2}'], method='ebicglasso')`,
        "summary_ttest": `sess.summary_ttest(m1=22.0, s1=4.5, n1=35, m2=18.0, s2=4.0, n2=35)`,
        "stata_ttest": `ttest ${c1}, by(${g1})`,
        "stata_reg": `regress ${c1} ${g1} ${c2}`,
        "stata_gen": `gen log_${c1} = log(${c1})`,
        "export_hwpx": `sess.export_results('reports/analysis_report.hwpx', fmt='hwpx')`,
        "export_pdf": `sess.export_results('reports/analysis_report.pdf', fmt='pdf')`
    };

    const snippet = snippets[type] || `# Snippet: ${type}`;
    if (ed) {
        ed.value += (ed.value.endsWith("\\n") || !ed.value ? "" : "\\n") + snippet + "\\n";
        ed.scrollTop = ed.scrollHeight;
    }
    const dd = document.getElementById("syntax-snippets-dropdown");
    if (dd) dd.style.display = "none";
    showToast(`스니펫 삽입됨: ${type}`);
}

// --- Right View Mode Switching (Results vs Data Grid) ---
function switchRightView(mode) {
    document.querySelectorAll(".v-tab").forEach(t => t.classList.remove("active"));
    const activeTab = document.getElementById("v-tab-" + mode);
    if (activeTab) activeTab.classList.add("active");

    const scrollWrap = document.getElementById("results-scroll-wrap");
    const gridWrap = document.getElementById("right-grid-wrap");

    if (mode === "results") {
        if (scrollWrap) scrollWrap.style.display = "block";
        if (gridWrap) gridWrap.style.display = "none";
    } else if (mode === "data") {
        if (scrollWrap) scrollWrap.style.display = "none";
        if (gridWrap) {
            gridWrap.style.display = "block";
            renderRightSpreadsheet();
        }
    } else if (mode === "split") {
        if (scrollWrap) { scrollWrap.style.display = "block"; scrollWrap.style.height = "50%"; }
        if (gridWrap) {
            gridWrap.style.display = "block";
            gridWrap.style.height = "50%";
            renderRightSpreadsheet();
        }
    }
}

function renderRightSpreadsheet() {
    const thead = document.getElementById("right-grid-thead");
    const tbody = document.getElementById("right-grid-tbody");
    if (!thead || !tbody) return;

    let thHtml = "<tr><th style='width:36px;'>#</th>";
    for (const c of APP.columns) {
        thHtml += `<th><img src="${measureIcon(c.measureType)}" style="width:13px;height:13px;vertical-align:-2px;margin-right:4px;">${escapeHtml(c.name)}</th>`;
    }
    thHtml += "</tr>";
    thead.innerHTML = thHtml;

    let tbHtml = "";
    (APP.rows || []).slice(0, 100).forEach((r, idx) => {
        tbHtml += `<tr><td style="color:#94A3B8;text-align:center;">${idx + 1}</td>`;
        for (const c of APP.columns) {
            const v = r[c.name] !== undefined && r[c.name] !== null ? r[c.name] : "";
            tbHtml += `<td>${escapeHtml(String(v))}</td>`;
        }
        tbHtml += "</tr>";
    });
    tbody.innerHTML = tbHtml;
}

// --- Synthetic Data Generator Modal & Runner ---
function openSyntheticDataModal() {
    const modal = document.getElementById("synthetic-modal");
    if (modal) modal.style.display = "flex";
}

function closeSyntheticModal() {
    const modal = document.getElementById("synthetic-modal");
    if (modal) modal.style.display = "none";
}

async function runSyntheticGeneration() {
    const n = parseInt(document.getElementById("synth-n-samples")?.value || "100", 10);
    const epsVal = document.getElementById("synth-epsilon")?.value;
    const eps = epsVal === "none" ? null : parseFloat(epsVal);

    showToast("가우시안 코퓰러 & DP 엔진으로 합성데이터 생성 중...");
    try {
        const res = await fetch("/api/data/synthetic", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                n_samples: n,
                epsilon: eps,
                dataset_name: "synthetic_" + (APP.datasetName || "dataset.csv"),
                as_new_dataset: true
            })
        });
        if (res.ok) {
            const data = await res.json();
            applyDatasetPayload(data);
            closeSyntheticModal();
            showToast(`[✓] 합성데이터 생성 완료 (N=${n}, ${APP.columns.length}개 변수)`, 4000);
            renderLeftVariablesExplorer();
        } else {
            const err = await res.json();
            showToast("합성데이터 생성 오류: " + (err.detail || "알 수 없음"), 4000);
        }
    } catch (e) {
        showToast("통신 오류: " + e, 4000);
    }
}

// Auto-run layout initialization on load
window.addEventListener("DOMContentLoaded", () => {
    const savedLayout = localStorage.getItem("openstat_layout") || "default";
    setLayoutMode(savedLayout);
    renderLeftVariablesExplorer();
    syncLeftSyntaxEditor();
});
"""

# Inject before the final script closing tag </script>
if "/* OpenStat Studio v3.1 Layout, Variables Explorer & Stata Console Engine */" not in html:
    html = html.replace("</script>\n</body>", js_runtime_code + "\n</script>\n</body>")
    print("[✓] Injected V3.1 Layout & Stata Engine JavaScript.")

# Also update applyDatasetPayload to refresh LeftVariablesExplorer
if "renderLeftVariablesExplorer();" not in html:
    html = html.replace(
        "renderGrid();\n    renderResults();",
        "renderGrid();\n    renderResults();\n    renderLeftVariablesExplorer();\n    syncLeftSyntaxEditor();"
    )
    print("[✓] Hooked renderLeftVariablesExplorer into applyDatasetPayload.")

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print("[Step 2 Complete] All JavaScript modules injected successfully.")
