#!/usr/bin/env python3
"""
Inject bilingual i18n dictionary, SPSS Korean terms, and dynamic translation hooks into index.html.
"""

import os

HTML_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reconstructed", "pyjamovi", "web", "index.html")

with open(HTML_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# Prepare I18N definitions
i18n_code = '''
/* ==========================================================================
   Bilingual Internationalization (I18N) — Korean Default & English Switcher
   SPSS / jamovi Standard Korean Statistical Terminology
   ========================================================================== */

const I18N = {
    ko: {
        top_user_guide: "사용 설명서",
        tab_variables: "변수",
        tab_data: "데이터",
        tab_analyses: "분석",
        tab_plots: "그래프 &amp; 디자인",
        tab_python: "파이썬 (.py) &amp; 내보내기",

        live_py: "&lt;/&gt; 실시간 .py",
        graph_builder: "📊 차트 빌더",
        plot_design: "🎨 그래프 디자인",
        export_results_btn: "📄 결과 내보내기 ▾",

        var_compute: "변수 계산",
        var_recode: "변수 재코딩",
        var_add: "변수 추가",
        var_delete: "변수 삭제",
        var_undo: "실행 취소",

        data_library: "데이터 라이브러리",
        data_filter: "데이터 필터",
        data_merge: "데이터 병합",
        data_reshape: "형태 변환",
        data_transform: "표준화 (Z)",
        data_add_row: "행 추가",
        data_delete_row: "행 삭제",

        ana_exploration: "기술통계",
        ana_ttests: "T-검정",
        ana_anova: "분산분석",
        ana_regression: "회귀분석",
        ana_frequencies: "빈도분석",
        ana_factor: "요인분석",
        ana_sem: "SEM / CFA",
        ana_medmod: "매개/조절",
        ana_panel: "패널 &amp; 혼합",
        ana_survival: "생존분석",
        ana_doctor: "자가진단 &amp; 동기화",
        ana_modules: "추가 모듈",

        results_header: "분석 결과 (Analysis Results)",
        results_empty_tip: "데이터를 불러오고 상단 리본 메뉴에서 원하는 통계 분석을 선택하면 여기에 APA 양식 표와 고해상도 그래프가 실시간으로 나타납니다. (우측 상단에서 한국어/English 전환 가능)",
        btn_pin_results: "📌 결과 시트에 고정",
        btn_design: "🎨 디자인",
        apa_narrative: "APA 7판 학술 해석:",

        hdr_num: "#",
        hdr_scale: "척도",
        hdr_var_name: "변수명",
        hdr_measure: "측정 척도",
        hdr_type: "데이터 유형",
        hdr_valid: "유효값",
        hdr_missing: "결측치",
        hdr_summary: "요약 통계 / 범주 목록",
        hdr_actions: "작업",
        btn_recode_action: "재코딩",
        btn_delete_action: "삭제",

        opt_panel_header: "분석 옵션",
        opt_btn_new_card: "+ 새 결과 카드",
        opt_btn_auto_fill: "자동 입력",
        opt_btn_plot: "🎨 그래프 설정",

        modal_recode_title: "변수 재코딩 / 범주화 (Recode)",
        modal_filter_title: "데이터 행 필터링 (Filter)",
        modal_merge_title: "데이터 병합 및 추가 (Merge / Append)",
        modal_reshape_title: "데이터 형태 변환 (Wide ↔ Long Reshape)",
        modal_export_title: "분석 결과 보고서 내보내기 (Export Results)",

        toast_saved: "저장되었습니다.",
        toast_executed: "실행이 완료되었습니다.",
        toast_undone: "이전 상태로 복원되었습니다."
    },
    en: {
        top_user_guide: "User Guide",
        tab_variables: "Variables",
        tab_data: "Data",
        tab_analyses: "Analyses",
        tab_plots: "Graphs &amp; Design",
        tab_python: "Python (.py) &amp; Export",

        live_py: "&lt;/&gt; Live .py",
        graph_builder: "📊 Graph Builder",
        plot_design: "🎨 Plot Design",
        export_results_btn: "📄 Export Results ▾",

        var_compute: "Compute",
        var_recode: "Recode",
        var_add: "Add Var",
        var_delete: "Delete Var",
        var_undo: "Undo",

        data_library: "Data Library",
        data_filter: "Filter",
        data_merge: "Merge Data",
        data_reshape: "Reshape",
        data_transform: "Transform (Z)",
        data_add_row: "Add Row",
        data_delete_row: "Delete Row",

        ana_exploration: "Exploration",
        ana_ttests: "T-Tests",
        ana_anova: "ANOVA",
        ana_regression: "Regression",
        ana_frequencies: "Frequencies",
        ana_factor: "Factor",
        ana_sem: "SEM / CFA",
        ana_medmod: "MedMod",
        ana_panel: "Panel &amp; Mixed",
        ana_survival: "Survival",
        ana_doctor: "Doctor &amp; Sync",
        ana_modules: "Modules",

        results_header: "Analysis Results",
        results_empty_tip: "Load a dataset and select an analysis from the top ribbon menu. Formatted APA tables and figures will appear here live.",
        btn_pin_results: "📌 To Results",
        btn_design: "🎨 Design",
        apa_narrative: "APA 7th Interpretation:",

        hdr_num: "#",
        hdr_scale: "Scale",
        hdr_var_name: "Variable Name",
        hdr_measure: "Measure Type",
        hdr_type: "Data Type",
        hdr_valid: "Valid",
        hdr_missing: "Missing",
        hdr_summary: "Summary / Levels",
        hdr_actions: "Actions",
        btn_recode_action: "Recode",
        btn_delete_action: "Delete",

        opt_panel_header: "Analysis Options",
        opt_btn_new_card: "+ New Card",
        opt_btn_auto_fill: "Auto-Fill",
        opt_btn_plot: "🎨 Plot Design",

        modal_recode_title: "Recode / Bin Variable (Recode)",
        modal_filter_title: "Filter Rows (Filter)",
        modal_merge_title: "Merge / Append Data",
        modal_reshape_title: "Reshape Dataset (Wide ↔ Long)",
        modal_export_title: "Export Analysis Results",

        toast_saved: "Saved successfully.",
        toast_executed: "Executed successfully.",
        toast_undone: "Restored to previous state."
    }
};

const RIBBON_MENUS_I18N = {
    ko: {
        exploration: [
            { section: "기초 기술통계 (Descriptives)" },
            { id: "descriptives", label: "기술통계량 (평균, 표준편차, 왜도/첨도)" },
            { sep: true },
            { section: "탐색적 도표 (scatr)" },
            { id: "scatr", label: "산점도 및 주변 분포도 (Scatterplot)", badge: "scatr" },
            { id: "pareto", label: "품질관리 파레토 차트 (Pareto Chart)", badge: "scatr" }
        ],
        ttests: [
            { id: "ttestIS", label: "독립표본 T-검정 (Independent Samples)" },
            { id: "ttestPS", label: "대응표본 T-검정 (Paired Samples)" },
            { id: "ttestOneS", label: "일표본 T-검정 (One Sample)" },
            { sep: true },
            { section: "베이지안 가설검정 (Bayesian)" },
            { id: "bayesTTestIS", label: "베이지안 독립표본 T-검정", badge: "BF₁₀" }
        ],
        anova: [
            { id: "anovaOneW", label: "일원배치 분산분석 (One-Way ANOVA)" },
            { id: "anovaFactorial", label: "요인 분산분석 (Factorial ANOVA)" },
            { id: "anovaRM", label: "반복측정 분산분석 (Repeated Measures)" },
            { id: "ancova", label: "공분산분석 (ANCOVA)" },
            { sep: true },
            { section: "비모수 검정 (Non-Parametric)" },
            { id: "kruskal", label: "비모수 분산분석 (Kruskal-Wallis)" },
            { sep: true },
            { section: "베이지안 분산분석 (Bayesian)" },
            { id: "bayesANOVA", label: "베이지안 분산분석", badge: "BF_M" }
        ],
        regression: [
            { id: "corrMatrix", label: "상관행렬 분석 (Correlation Matrix)" },
            { id: "partCorr", label: "편상관분석 (Partial Correlation)" },
            { id: "bayesCorr", label: "베이지안 상관행렬", badge: "BF₁₀" },
            { sep: true },
            { id: "linReg", label: "다중선형회귀분석 (Linear Regression)" },
            { sep: true },
            { section: "로지스틱 회귀분석 (Logistic)" },
            { id: "logRegBin", label: "이항 로지스틱 회귀분석 (Binomial Logistic)" }
        ],
        frequencies: [
            { section: "분할표 및 범주형 데이터" },
            { id: "conttables", label: "교차분할표 독립성 검정 (χ² / Fisher)" }
        ],
        factor: [
            { id: "reliability", label: "신뢰도 분석 (Cronbach's α & ω)" },
            { sep: true },
            { id: "pca", label: "주성분분석 (PCA)" },
            { id: "efa", label: "탐색적 요인분석 (EFA)" }
        ],
        medmod: [
            { id: "mediation", label: "매개효과 분석 (부트스트랩 간접효과)" },
            { id: "moderation", label: "조절효과 분석 (단순기울기 상호작용)" }
        ],
        linearmodels: [
            { section: "패널 데이터 계량분석 (Panel Econometrics)" },
            { id: "panel", label: "패널 고정/확률효과 모형 (Within FE / RE / Hausman)" },
            { sep: true },
            { section: "다층모형 (Multilevel Models)" },
            { id: "mixedModel", label: "선형 혼합모형 (Linear Mixed Model - REML & ICC)" }
        ],
        survival: [
            { id: "survival", label: "생존분석 (Kaplan-Meier & Cox 비례위험)" }
        ],
        export_tables: [
            { section: "문서 및 논문 보고서 (.hwpx, .docx, .pdf)" },
            { action: "exportFullSessionResults('hwpx')", label: "한글 문서 (.hwpx) — 네이티브 편집 가능 표 + 그림", badge: "한글" },
            { action: "exportFullSessionResults('pdf')", label: "출판용 PDF (.pdf) — APA 3선 표 + 고해상도 그림", badge: "인쇄" },
            { action: "exportFullSessionResults('docx')", label: "마이크로소프트 워드 (.docx)", badge: "Word" },
            { sep: true },
            { section: "스프레드시트 및 데이터" },
            { action: "exportFullSessionResults('xlsx')", label: "엑셀 워크북 (.xlsx) — APA 테두리 표 + 차트 시트", badge: "Excel" },
            { action: "exportFullSessionResults('tex')", label: "LaTeX 논문 테이블 (.tex + PDF 벡터 그림)", badge: "LaTeX" },
            { action: "exportFullSessionResults('html')", label: "웹 HTML 보고서 (.html)", badge: "HTML" },
            { action: "exportFullSessionResults('md')", label: "마크다운 (.md)", badge: "Markdown" },
            { sep: true },
            { action: "exportFullSessionResults('csv')", label: "CSV 표 데이터 (.csv)", badge: "CSV" },
            { action: "exportFullSessionResults('omv')", label: "jamovi 바이너리 컨테이너 (.omv)", badge: "OMV" }
        ]
    },
    en: {
        exploration: [
            { section: "jmv" },
            { id: "descriptives", label: "Descriptives" },
            { sep: true },
            { section: "scatr" },
            { id: "scatr", label: "Scatterplot", badge: "scatr" },
            { id: "pareto", label: "Pareto Chart", badge: "scatr" }
        ],
        ttests: [
            { id: "ttestIS", label: "Independent Samples T-Test" },
            { id: "ttestPS", label: "Paired Samples T-Test" },
            { id: "ttestOneS", label: "One Sample T-Test" },
            { sep: true },
            { section: "Bayesian (jsq)" },
            { id: "bayesTTestIS", label: "Bayesian Independent Samples T-Test", badge: "BF₁₀" }
        ],
        anova: [
            { id: "anovaOneW", label: "One-Way ANOVA" },
            { id: "anovaFactorial", label: "ANOVA (Factorial)" },
            { id: "anovaRM", label: "Repeated Measures ANOVA" },
            { id: "ancova", label: "ANCOVA" },
            { sep: true },
            { section: "Non-Parametric" },
            { id: "kruskal", label: "One-Way ANOVA (Kruskal-Wallis)" },
            { sep: true },
            { section: "Bayesian (jsq)" },
            { id: "bayesANOVA", label: "Bayesian ANOVA", badge: "BF_M" }
        ],
        regression: [
            { id: "corrMatrix", label: "Correlation Matrix" },
            { id: "partCorr", label: "Partial Correlation" },
            { id: "bayesCorr", label: "Bayesian Correlation Matrix", badge: "BF₁₀" },
            { sep: true },
            { id: "linReg", label: "Linear Regression" },
            { sep: true },
            { section: "Logistic Regression" },
            { id: "logRegBin", label: "2 Outcomes (Binomial)" }
        ],
        frequencies: [
            { section: "Contingency Tables" },
            { id: "conttables", label: "Contingency Tables (χ² / Fisher)" }
        ],
        factor: [
            { id: "reliability", label: "Reliability Analysis (Cronbach's α & ω)" },
            { sep: true },
            { id: "pca", label: "Principal Component Analysis" },
            { id: "efa", label: "Exploratory Factor Analysis" }
        ],
        medmod: [
            { id: "mediation", label: "Mediation (Bootstrap Indirect)" },
            { id: "moderation", label: "Moderation (Simple Slopes)" }
        ],
        linearmodels: [
            { section: "Panel Econometrics" },
            { id: "panel", label: "Panel Fixed & Random Effects (Hausman)" },
            { sep: true },
            { section: "Multilevel Models" },
            { id: "mixedModel", label: "Linear Mixed Model (Multilevel ICC)", badge: "GAMLj" }
        ],
        survival: [
            { id: "survival", label: "Survival Analysis (Kaplan-Meier & Cox)" }
        ],
        export_tables: [
            { section: "Full Results (Tables + Embedded Graphs)" },
            { action: "exportFullSessionResults('pdf')", label: "Publication PDF Report (.pdf)", badge: "PDF" },
            { action: "exportFullSessionResults('hwpx')", label: "Hancom Hangul + Graphs (.hwpx)", badge: "한글" },
            { action: "exportFullSessionResults('docx')", label: "Word Document + Graphs (.docx)", badge: "APA" },
            { action: "exportFullSessionResults('xlsx')", label: "Excel Workbook + Charts (.xlsx)", badge: "APA" },
            { action: "exportFullSessionResults('tex')", label: "LaTeX Booktabs + Figures (.tex)", badge: "TeX" },
            { sep: true },
            { section: "Data, Web & Archive" },
            { action: "exportFullSessionResults('html')", label: "Standalone HTML Report (.html)" },
            { action: "exportFullSessionResults('md')", label: "Markdown + Figures (.md)" },
            { action: "exportFullSessionResults('csv')", label: "CSV Table (.csv)" },
            { action: "exportFullSessionResults('tsv')", label: "TSV Table (.tsv)" },
            { action: "exportFullSessionResults('json')", label: "Structured JSON (.json)" },
            { action: "exportFullSessionResults('omv')", label: "jamovi Archive (.omv)", badge: "omv" }
        ]
    }
};

const OPTION_LABELS_KO = {
    // Analysis Titles
    "Descriptives": "기술통계량 (Descriptives)",
    "Independent Samples T-Test": "독립표본 T-검정 (Independent Samples T-Test)",
    "Paired Samples T-Test": "대응표본 T-검정 (Paired Samples T-Test)",
    "One Sample T-Test": "일표본 T-검정 (One Sample T-Test)",
    "One-Way ANOVA": "일원배치 분산분석 (One-Way ANOVA)",
    "ANOVA (Factorial)": "요인 분산분석 (Factorial ANOVA)",
    "Repeated Measures ANOVA": "반복측정 분산분석 (Repeated Measures ANOVA)",
    "ANCOVA": "공분산분석 (ANCOVA)",
    "One-Way ANOVA (Kruskal-Wallis)": "비모수 분산분석 (Kruskal-Wallis)",
    "Correlation Matrix": "상관행렬 분석 (Correlation Matrix)",
    "Partial Correlation": "편상관분석 (Partial Correlation)",
    "Linear Regression": "다중선형회귀분석 (Linear Regression)",
    "Logistic Regression (Binomial)": "이항 로지스틱 회귀분석 (Logistic Regression)",
    "Contingency Tables": "교차분할표 독립성 검정 (Contingency Tables)",
    "Reliability Analysis": "신뢰도 분석 (Reliability Analysis)",
    "Principal Component Analysis": "주성분분석 (PCA)",
    "Exploratory Factor Analysis": "탐색적 요인분석 (EFA)",
    "Structural Equation Modeling (SEM)": "구조방정식 모형 (SEM / CFA)",
    "Mediation Analysis": "매개효과 분석 (Mediation Analysis)",
    "Moderation Analysis": "조절효과 분석 (Moderation Analysis)",
    "Panel Data Econometrics": "패널 데이터 계량분석 (Panel Econometrics)",
    "Linear Mixed Model": "선형 혼합모형 (Multilevel Linear Mixed Model)",
    "Survival Analysis": "생존분석 (Survival Analysis)",
    "Standalone Graph Builder": "독립형 15종 학술 차트 빌더 (Graph Builder)",

    // Slots
    "Variables": "변수 목록 (Variables)",
    "Dependent Variable": "종속변수 (Dependent Variable)",
    "Dependent Variables": "종속변수 (Dependent Variables)",
    "Grouping Variable": "집단변수 (Grouping Variable)",
    "Fixed Factors": "고정요인 (Fixed Factors)",
    "Covariates": "공변량 (Covariates)",
    "Split by": "집단 분할 (Split by)",
    "Pairs": "대응 변수 쌍 (Pairs)",
    "Independent Variables": "독립변수 (Independent Variables)",
    "Predictors": "예측변수 (Predictors)",
    "Rows": "행 변수 (Rows)",
    "Columns": "열 변수 (Columns)",
    "Mediator": "매개변수 (Mediator)",
    "Moderator": "조절변수 (Moderator)",
    "Entity (ID)": "개체 식별 변수 (Entity ID)",
    "Time": "시간 변수 (Time)",
    "Cluster (Group)": "군집/집단 변수 (Cluster)",
    "Time Variable": "시간 변수 (Time Variable)",
    "Event / Status": "사건/상태 변수 (Event Status)",

    // Common Option Sections & Items
    "Hypothesis": "가설 설정 (Hypothesis)",
    "Tests": "검정 방법 (Tests)",
    "Additional Statistics": "추가 통계량 (Additional Statistics)",
    "Assumptions": "가정 검정 (Assumptions)",
    "Plots": "도표 및 그래프 (Plots)",
    "Post-Hoc Tests": "사후검정 (Post-Hoc Tests)",
    "Model Fit": "모형 적합도 (Model Fit)",
    "Residuals": "잔차 진단 (Residuals)",
    "Coefficients": "회귀계수 (Coefficients)",

    // Options Checkboxes
    "Student's": "Student t-검정",
    "Welch's": "Welch t-검정 (등분산 기각 시)",
    "Mann-Whitney U": "Mann-Whitney U 검정 (비모수)",
    "Wilcoxon W": "Wilcoxon 부호순위 검정 (비모수)",
    "Mean difference": "평균 차이 (Mean Difference)",
    "Effect size": "효과크기 (Cohen's d)",
    "Homogeneity of variances": "등분산성 검정 (Levene's)",
    "Normality": "정규성 검정 (Shapiro-Wilk)",
    "Descriptives plot": "기술통계 도표 (평균 및 95% 신뢰구간)",
    "Descriptives": "기술통계량 표",
    "N": "표본 크기 (N)",
    "Missing": "결측치 수 (Missing)",
    "Mean": "평균 (Mean)",
    "Median": "중앙값 (Median)",
    "Mode": "최빈값 (Mode)",
    "Std. deviation": "표준편차 (Std. Deviation)",
    "Variance": "분산 (Variance)",
    "Skewness": "왜도 (Skewness)",
    "Kurtosis": "첨도 (Kurtosis)",
    "Shapiro-Wilk": "정규성 검정 (Shapiro-Wilk)",
    "Confidence interval": "95% 신뢰구간 (CI)",
    "Bar plot": "막대도표 (Bar Plot)",
    "Box plot": "상자수염도 (Box Plot)",
    "Violin plot": "바이올린 도표 (Violin Plot)",
    "Raincloud plot": "레인클라우드 도표 (Raincloud)",
    "Tukey": "Tukey HSD 사후검정",
    "Scheffé": "Scheffé 사후검정",
    "Bonferroni": "Bonferroni 보정",
    "Holm": "Holm 보정",
    "Games-Howell": "Games-Howell 사후검정 (등분산 기각 시 권장)",
    "Collinearity statistics": "다중공선성 진단 (VIF & Tolerance)",
    "Durbin-Watson": "Durbin-Watson 잔차 독립성",
    "Cook's distance": "Cook의 거리 (이상치 진단)",
    "Q-Q plot": "정규 Q-Q 도표",
    "Standardized estimate": "표준화 계수 (Beta)",
    "Path diagram": "경로 모형도 (Path Diagram)",
    "Odds ratio": "오즈비 (Odds Ratio, Exp(B))",
    "ROC curve": "ROC / AUC 곡선",
    "Chi-square": "카이제곱 검정 (χ²)",
    "Fisher's exact test": "Fisher 직접확률 검정",
    "Cramér's V": "Cramér의 V (연관성 크기)",
    "Cronbach's alpha": "Cronbach's α 계수",
    "McDonald's omega": "McDonald's ω 계수",
    "Scree plot": "스크리 도표 (Scree Plot)",
    "Hausman test": "하우스만 사양검정 (Fixed vs Random Effects)",
};

function setLanguage(lang) {
    APP.lang = lang;
    localStorage.setItem("openstat_lang", lang);
    applyLanguage();
}

function applyLanguage() {
    const lang = APP.lang || "ko";
    const dict = I18N[lang] || I18N.ko;

    const btnKo = document.getElementById("btn-lang-ko");
    const btnEn = document.getElementById("btn-lang-en");
    if (btnKo && btnEn) {
        btnKo.classList.toggle("active", lang === "ko");
        btnEn.classList.toggle("active", lang === "en");
    }

    document.querySelectorAll("[data-i18n]").forEach(el => {
        const key = el.getAttribute("data-i18n");
        if (dict[key]) el.innerHTML = dict[key];
    });

    document.querySelectorAll("[data-i18n-title]").forEach(el => {
        const key = el.getAttribute("data-i18n-title");
        if (dict[key]) el.setAttribute("title", dict[key]);
    });

    // Update variables view if open
    const vc = document.getElementById("variables-container");
    if (vc && vc.style.display !== "none") {
        renderVariablesView();
    }

    // Update active options panel if open
    if (APP.currentType) {
        renderOptionsPanel();
    }
}

// Global Image Error Auto-Recovery
window.addEventListener("error", function(e) {
    if (e.target && e.target.tagName === "IMG") {
        const src = e.target.getAttribute("src");
        if (src && src.startsWith("/assets/")) {
            e.target.src = src.substring(1);
        } else if (src && src.startsWith("assets/")) {
            e.target.src = "/" + src;
        }
    }
}, true);
'''

# Replace RIBBON_MENUS declaration with our bilingual I18N
target_ribbon = "const RIBBON_MENUS = {"
if target_ribbon in content:
    # Find where RIBBON_MENUS ends
    end_marker = "const PALETTE_SWATCHES = {"
    start_pos = content.find(target_ribbon)
    end_pos = content.find(end_marker)
    if start_pos != -1 and end_pos != -1:
        content = content[:start_pos] + i18n_code + "\n\n" + content[end_pos:]

# Update APP definition to have default lang = 'ko'
app_def_target = "    plotTargetAnalysisId: null,\n    runTimer: null\n};"
app_def_replace = "    plotTargetAnalysisId: null,\n    runTimer: null,\n    lang: localStorage.getItem('openstat_lang') || 'ko'\n};"
if app_def_target in content:
    content = content.replace(app_def_target, app_def_replace, 1)

# Update openRibbonMenu to use RIBBON_MENUS_I18N
old_orm = "    const items = RIBBON_MENUS[menuKey] || [];"
new_orm = "    const lang = APP.lang || 'ko';\n    const menus = RIBBON_MENUS_I18N[lang] || RIBBON_MENUS_I18N.ko;\n    const items = menus[menuKey] || [];"
if old_orm in content:
    content = content.replace(old_orm, new_orm, 1)

# Update measureIcon to be robust against case, nominal-text, and path
old_mi = """function measureIcon(mType) {
    if (mType === "Continuous") return "/assets/variable-continuous.svg";
    if (mType === "Ordinal") return "/assets/variable-ordinal.svg";
    if (mType === "ID") return "/assets/variable-id.svg";
    return "/assets/variable-nominal.svg";
}"""

new_mi = """function measureIcon(mType) {
    if (!mType) return "assets/variable-nominal.svg";
    const mt = String(mType).toLowerCase();
    if (mt.includes("cont") || mt.includes("scale") || mt.includes("연속")) return "assets/variable-continuous.svg";
    if (mt.includes("ord") || mt.includes("순서") || mt.includes("서열")) return "assets/variable-ordinal.svg";
    if (mt.includes("id")) return "assets/variable-id.svg";
    if (mt.includes("text") || mt.includes("문자")) return "assets/variable-nominal-text.svg";
    return "assets/variable-nominal.svg";
}"""
if old_mi in content:
    content = content.replace(old_mi, new_mi, 1)

# Update renderVariablesView with Korean / English headers
old_rv = """function renderVariablesView() {
    const wrap = document.getElementById("variables-container");
    let html = `<div class="var-row-item var-row-hdr">
        <div>#</div><div>Scale</div><div>Variable Name</div><div>Measure Type</div><div>Data Type</div><div style="text-align:right;">Valid</div><div style="text-align:right;">Missing</div><div>Summary / Range / Levels</div><div style="text-align:center;">Actions</div>
    </div>`;"""

new_rv = """function renderVariablesView() {
    const wrap = document.getElementById("variables-container");
    const lang = APP.lang || "ko";
    const dict = I18N[lang] || I18N.ko;
    let html = `<div class="var-row-item var-row-hdr">
        <div>${dict.hdr_num}</div><div>${dict.hdr_scale}</div><div>${dict.hdr_var_name}</div><div>${dict.hdr_measure}</div><div>${dict.hdr_type}</div><div style="text-align:right;">${dict.hdr_valid}</div><div style="text-align:right;">${dict.hdr_missing}</div><div>${dict.hdr_summary}</div><div style="text-align:center;">${dict.hdr_actions}</div>
    </div>`;"""
if old_rv in content:
    content = content.replace(old_rv, new_rv, 1)

# Update renderOptionsPanel to translate titles, slots and items
old_rop_title = """function renderOptionsPanel() {
    const schema = ANALYSIS_SCHEMAS[APP.currentType];
    if (!schema) return;
    const container = document.getElementById("opt-panel-body");

    let html = "";
    document.getElementById("opt-panel-title").textContent = schema.title;"""

new_rop_title = """function renderOptionsPanel() {
    const schema = ANALYSIS_SCHEMAS[APP.currentType];
    if (!schema) return;
    const container = document.getElementById("opt-panel-body");
    const lang = APP.lang || "ko";
    const isKo = lang === "ko";

    let html = "";
    const titleText = (isKo && OPTION_LABELS_KO[schema.title]) ? OPTION_LABELS_KO[schema.title] : schema.title;
    document.getElementById("opt-panel-title").textContent = titleText;"""
if old_rop_title in content:
    content = content.replace(old_rop_title, new_rop_title, 1)

# Call applyLanguage at DOMContentLoaded or init
init_call = "    applyLanguage();\n    initApp();"
if "initApp();" in content and "applyLanguage();" not in content:
    content = content.replace("initApp();", init_call, 1)

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(content)

print("[OK] Injected I18N dictionaries and Korean SPSS terms into index.html")
