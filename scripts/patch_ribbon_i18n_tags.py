#!/usr/bin/env python3
"""
Add data-i18n tags to tabs, ribbon buttons, and analysis buttons in index.html.
"""

import os

HTML_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reconstructed", "pyjamovi", "web", "index.html")

with open(HTML_PATH, "r", encoding="utf-8") as f:
    c = f.read()

# 1. Tabs
c = c.replace('<button class="jmv-ribbon-tab" data-tab="variables" onclick="switchRibbonTab(\'variables\')">Variables</button>',
              '<button class="jmv-ribbon-tab" data-tab="variables" data-i18n="tab_variables" onclick="switchRibbonTab(\'variables\')">변수</button>')
c = c.replace('<button class="jmv-ribbon-tab" data-tab="data" onclick="switchRibbonTab(\'data\')">Data</button>',
              '<button class="jmv-ribbon-tab" data-tab="data" data-i18n="tab_data" onclick="switchRibbonTab(\'data\')">데이터</button>')
c = c.replace('<button class="jmv-ribbon-tab selected" data-tab="analyses" onclick="switchRibbonTab(\'analyses\')">Analyses</button>',
              '<button class="jmv-ribbon-tab selected" data-tab="analyses" data-i18n="tab_analyses" onclick="switchRibbonTab(\'analyses\')">분석</button>')
c = c.replace('<button class="jmv-ribbon-tab" data-tab="plots" onclick="switchRibbonTab(\'plots\')">Graphs &amp; Design</button>',
              '<button class="jmv-ribbon-tab" data-tab="plots" data-i18n="tab_plots" onclick="switchRibbonTab(\'plots\')">그래프 &amp; 디자인</button>')
c = c.replace('<button class="jmv-ribbon-tab" data-tab="edit" onclick="switchRibbonTab(\'edit\')">Python (.py) &amp; Export</button>',
              '<button class="jmv-ribbon-tab" data-tab="edit" data-i18n="tab_python" onclick="switchRibbonTab(\'edit\')">파이썬 (.py) &amp; 내보내기</button>')

# 2. Right tools
c = c.replace('<button class="ribbon-pill-btn" id="btn-toggle-py" onclick="togglePyDrawer()" title="Open Bidirectional Python (.py) Live Editor & Runner">&lt;/&gt; Live .py</button>',
              '<button class="ribbon-pill-btn" id="btn-toggle-py" onclick="togglePyDrawer()" title="실시간 파이썬 (.py) 코드 스튜디오 열기"><span data-i18n="live_py">&lt;/&gt; 실시간 .py</span></button>')
c = c.replace('<button class="ribbon-pill-btn" onclick="openAnalysis(\'graph\')" title="Open Standalone Graph Builder (15 Chart Types -> Analysis Results)">📊 Graph Builder</button>',
              '<button class="ribbon-pill-btn" onclick="openAnalysis(\'graph\')" title="독립형 15종 논문용 차트 빌더 열기"><span data-i18n="graph_builder">📊 차트 빌더</span></button>')
c = c.replace('<button class="ribbon-pill-btn" onclick="openPlotDesignModal(null)" title="Customize Graph Palette, Theme, Fonts & Labels">🎨 Plot Design</button>',
              '<button class="ribbon-pill-btn" onclick="openPlotDesignModal(null)" title="그래프 컬러 팔레트 및 테마 설정"><span data-i18n="plot_design">🎨 그래프 디자인</span></button>')
c = c.replace('<button class="ribbon-pill-btn" onclick="openRibbonMenu(event, \'export_tables\')" title="Export Analysis Results (Tables + Embedded Graphs) to .pdf, .hwpx, .docx, .xlsx, .tex, .csv, .md, .html, .omv">📄 Export Results ▾</button>',
              '<button class="ribbon-pill-btn" onclick="openRibbonMenu(event, \'export_tables\')" title="분석 결과 및 그래프를 한글(HWPX), 엑셀, PDF 등으로 내보내기"><span data-i18n="export_results_btn">📄 결과 내보내기 ▾</span></button>')

# 3. Variables Pane Buttons
c = c.replace('<span class="rb-label">Compute</span>', '<span class="rb-label" data-i18n="var_compute">변수 계산</span>')
c = c.replace('<span class="rb-label">Recode</span>', '<span class="rb-label" data-i18n="var_recode">변수 재코딩</span>')
c = c.replace('<span class="rb-label">Add Var</span>', '<span class="rb-label" data-i18n="var_add">변수 추가</span>')
c = c.replace('<span class="rb-label">Delete Var</span>', '<span class="rb-label" data-i18n="var_delete">변수 삭제</span>')
c = c.replace('<span class="rb-label">Undo</span>', '<span class="rb-label" data-i18n="var_undo">실행 취소</span>')

# 4. Data Pane Buttons
c = c.replace('<span class="rb-label">Data Library</span>', '<span class="rb-label" data-i18n="data_library">데이터 라이브러리</span>')
c = c.replace('<span class="rb-label">Filter</span>', '<span class="rb-label" data-i18n="data_filter">데이터 필터</span>')
c = c.replace('<span class="rb-label">Merge Data</span>', '<span class="rb-label" data-i18n="data_merge">데이터 병합</span>')
c = c.replace('<span class="rb-label">Reshape</span>', '<span class="rb-label" data-i18n="data_reshape">형태 변환</span>')
c = c.replace('<span class="rb-label">Transform (Z)</span>', '<span class="rb-label" data-i18n="data_transform">표준화 (Z)</span>')
c = c.replace('<span class="rb-label">Add Row</span>', '<span class="rb-label" data-i18n="data_add_row">행 추가</span>')
c = c.replace('<span class="rb-label">Delete Row</span>', '<span class="rb-label" data-i18n="data_delete_row">행 삭제</span>')

# 5. Analyses Pane Buttons
c = c.replace('<span class="rb-label">Exploration <span class="rb-caret">▾</span></span>', '<span class="rb-label"><span data-i18n="ana_exploration">기술통계</span> <span class="rb-caret">▾</span></span>')
c = c.replace('<span class="rb-label">T-Tests <span class="rb-caret">▾</span></span>', '<span class="rb-label"><span data-i18n="ana_ttests">T-검정</span> <span class="rb-caret">▾</span></span>')
c = c.replace('<span class="rb-label">ANOVA <span class="rb-caret">▾</span></span>', '<span class="rb-label"><span data-i18n="ana_anova">분산분석</span> <span class="rb-caret">▾</span></span>')
c = c.replace('<span class="rb-label">Regression <span class="rb-caret">▾</span></span>', '<span class="rb-label"><span data-i18n="ana_regression">회귀분석</span> <span class="rb-caret">▾</span></span>')
c = c.replace('<span class="rb-label">Frequencies <span class="rb-caret">▾</span></span>', '<span class="rb-label"><span data-i18n="ana_frequencies">빈도분석</span> <span class="rb-caret">▾</span></span>')
c = c.replace('<span class="rb-label">Factor <span class="rb-caret">▾</span></span>', '<span class="rb-label"><span data-i18n="ana_factor">요인분석</span> <span class="rb-caret">▾</span></span>')
c = c.replace('<span class="rb-label">SEM / CFA</span>', '<span class="rb-label" data-i18n="ana_sem">SEM / CFA</span>')
c = c.replace('<span class="rb-label">MedMod <span class="rb-caret">▾</span></span>', '<span class="rb-label"><span data-i18n="ana_medmod">매개/조절</span> <span class="rb-caret">▾</span></span>')
c = c.replace('<span class="rb-label">Panel &amp; Mixed <span class="rb-caret">▾</span></span>', '<span class="rb-label"><span data-i18n="ana_panel">패널 &amp; 혼합</span> <span class="rb-caret">▾</span></span>')
c = c.replace('<span class="rb-label">Survival</span>', '<span class="rb-label" data-i18n="ana_survival">생존분석</span>')
c = c.replace('<span class="rb-label">Doctor &amp; Sync</span>', '<span class="rb-label" data-i18n="ana_doctor">자가진단 &amp; 동기화</span>')
c = c.replace('<span class="rb-label">Modules</span>', '<span class="rb-label" data-i18n="ana_modules">추가 모듈</span>')

# 6. Options panel title and buttons
c = c.replace('<span class="silky-options-title" id="opt-panel-title">Analysis</span>',
              '<span class="silky-options-title" id="opt-panel-title" data-i18n="opt_panel_header">분석 옵션</span>')
c = c.replace('<button class="silky-small-btn" onclick="duplicateAsNewResultCard()" title="Append as a new card in Analysis Results">+ New Card</button>',
              '<button class="silky-small-btn" onclick="duplicateAsNewResultCard()" title="분석 결과 시트에 새 카드로 추가"><span data-i18n="opt_btn_new_card">+ 새 결과 카드</span></button>')
c = c.replace('<button class="silky-small-btn" onclick="smartFillCurrentAnalysis()" title="Auto-fill valid sample variables & run">Auto-Fill</button>',
              '<button class="silky-small-btn" onclick="smartFillCurrentAnalysis()" title="유효한 변수를 자동으로 채우고 분석 실행"><span data-i18n="opt_btn_auto_fill">자동 입력</span></button>')
c = c.replace('<button class="silky-small-btn" onclick="openPlotDesignModal(APP.selectedAnalysisId)" title="Customize Graph Design">🎨 Plot</button>',
              '<button class="silky-small-btn" onclick="openPlotDesignModal(APP.selectedAnalysisId)" title="그래프 디자인 설정"><span data-i18n="opt_btn_plot">🎨 그래프 설정</span></button>')

# 7. Results empty placeholder
c = c.replace('Select an analysis from the ribbon above to begin.',
              '상단 리본 메뉴에서 원하는 통계 분석을 선택하면 여기에 APA 양식 표와 고해상도 그래프가 실시간으로 나타납니다. (우측 상단에서 한국어/English 전환 가능)')

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(c)

print("[OK] Replaced ribbon labels with Korean defaults and data-i18n tags.")
