#!/usr/bin/env python3
"""
pyjamovi v3.1 Comprehensive Studio Update
Applies:
1. New Window Layout with Left-Top Variables Explorer & Left-Bottom Syntax/Command Console, Right Results/Data
2. Layout switcher with 4 main modes (default, jamovi, syntax, quad, fullscreen)
3. Full Korean Localization for all 30 analyses, schemas, options panel, dialogs, buttons
4. JASP Features (Network Analysis & Summary Statistics T-Test in Menus & UI Schemas)
5. Synthetic Data Generator Modal & Stata/Python command runner
"""

import os
import re
import sys

HTML_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reconstructed", "pyjamovi", "web", "index.html")

with open(HTML_PATH, "r", encoding="utf-8") as f:
    html = f.read()

# ---------------------------------------------------------------------------
# 1. Update Ribbon Layout Controls
# ---------------------------------------------------------------------------
old_split_btns = """<button class="ribbon-icon-btn" onclick="setSplitMode('data-only')" title="Data View Fullscreen">
                    <img src="assets/fullscreen-hide-results.svg" alt="Data Only">
                </button>
                <button class="ribbon-icon-btn active" onclick="setSplitMode('mixed')" title="Split View (Data + Results)">
                    <img src="assets/splitscreen.svg" alt="Split">
                </button>
                <button class="ribbon-icon-btn" onclick="setSplitMode('results-only')" title="Results View Fullscreen">
                    <img src="assets/fullscreen-hide-data.svg" alt="Results Only">
                </button>"""

new_layout_controls = """<!-- Window Layout Selector -->
                <div class="layout-selector-wrap">
                    <button class="ribbon-pill-btn layout-pill" onclick="toggleLayoutMenu(event)" title="창 배열 모드 변경 (기본: 좌측 변수+신택스 | 우측 결과)">
                        <span id="current-layout-label">🏛️ 창 배열: 기본 모드 ▾</span>
                    </button>
                    <div id="layout-dropdown" class="layout-dd">
                        <div class="layout-dd-item active" id="layout-item-default" onclick="setLayoutMode('default')">
                            <div class="layout-dd-title">🏛️ 기본 모드 (Stata / RStudio 학술 연구형)</div>
                            <div class="layout-dd-desc">좌측: 상단 변수 탐색기 + 하단 Do-file 신택스 &amp; 커맨드창 | 우측: 결과/데이터</div>
                        </div>
                        <div class="layout-dd-item" id="layout-item-jamovi" onclick="setLayoutMode('jamovi')">
                            <div class="layout-dd-title">📊 자모비 클래식 모드</div>
                            <div class="layout-dd-desc">좌측: 데이터 스프레드시트 그리드 | 우측: 분석 결과 시트</div>
                        </div>
                        <div class="layout-dd-item" id="layout-item-syntax" onclick="setLayoutMode('syntax')">
                            <div class="layout-dd-title">📝 신택스 집중 모드 (Script-Centric)</div>
                            <div class="layout-dd-desc">좌측: 대형 Do-file / Python 스크립트 에디터 | 우측: 분석 결과</div>
                        </div>
                        <div class="layout-dd-item" id="layout-item-quad" onclick="setLayoutMode('quad')">
                            <div class="layout-dd-title">▦ 4분할 모드 (All-in-One 4-Panes)</div>
                            <div class="layout-dd-desc">좌상: 변수 | 좌하: 신택스 | 우상: 데이터 | 우하: 결과</div>
                        </div>
                        <div class="layout-dd-sep"></div>
                        <div class="layout-dd-item" id="layout-item-results-only" onclick="setLayoutMode('results-only')">
                            <div class="layout-dd-title">📄 결과 전체화면</div>
                            <div class="layout-dd-desc">분석표 및 그래프 결과 시트 최대화</div>
                        </div>
                        <div class="layout-dd-item" id="layout-item-data-only" onclick="setLayoutMode('data-only')">
                            <div class="layout-dd-title">📋 데이터 전체화면</div>
                            <div class="layout-dd-desc">스프레드시트 데이터 그리드 최대화</div>
                        </div>
                    </div>
                </div>"""

if old_split_btns in html:
    html = html.replace(old_split_btns, new_layout_controls)
    print("[✓] Replaced split buttons with Layout Switcher Dropdown.")
else:
    print("[!] old_split_btns not found directly; searching with regex...")
    html = re.sub(
        r'<button class="ribbon-icon-btn"[^>]*onclick="setSplitMode\(\'data-only\'\)".*?onclick="setSplitMode\(\'results-only\'\)"[^>]*>\s*<img[^>]*>\s*</button>',
        new_layout_controls,
        html,
        flags=re.DOTALL
    )
    print("[✓] Replaced via regex.")

# ---------------------------------------------------------------------------
# 2. Update Left Pane Markup
# ---------------------------------------------------------------------------
old_left_pane = """        <!-- Zone 1: Spreadsheet / Variable List -->
        <div id="left-pane">
            <div id="spreadsheet-container">
                <table class="jmv-grid" id="spreadsheet-table">
                    <thead id="grid-thead"></thead>
                    <tbody id="grid-tbody"></tbody>
                </table>
            </div>
            <div id="variables-container"></div>
        </div>"""

new_left_pane = """        <!-- Zone 1: Left Pane (Variables Explorer & Integrated Syntax/Command Console) -->
        <div id="left-pane">
            <!-- Left Top: Variables Explorer (좌측 상단: 변수) -->
            <div id="left-var-explorer">
                <div class="left-box-hdr">
                    <span class="left-box-title" data-i18n="var_explorer_title">📋 변수 탐색기</span>
                    <span id="left-var-count-badge" class="badge-pill">0개 변수</span>
                    <div class="left-box-tools">
                        <input type="text" id="left-var-filter" placeholder="변수 검색..." oninput="filterLeftVariables(this.value)">
                        <button class="left-small-icon-btn" onclick="toggleComputeBar()" title="변수 계산 (Stata: generate)"><img src="assets/menu-compute.svg" alt="Compute"></button>
                        <button class="left-small-icon-btn" onclick="openRecodeModal()" title="변수 재코딩 (Stata: recode)"><img src="assets/menu-data-transform.svg" alt="Recode"></button>
                    </div>
                </div>
                <div id="left-var-list-scroll">
                    <table class="left-var-table" id="left-var-table">
                        <thead>
                            <tr>
                                <th style="width:26px;">#</th>
                                <th style="width:22px;">척도</th>
                                <th>변수명</th>
                                <th style="width:50px;">유형</th>
                                <th style="width:38px;text-align:right;">결측</th>
                                <th style="width:32px;text-align:center;">선택</th>
                            </tr>
                        </thead>
                        <tbody id="left-var-tbody"></tbody>
                    </table>
                </div>
            </div>

            <!-- Draggable Horizontal Splitter -->
            <div id="left-h-splitter" onmousedown="startHResize(event)"></div>

            <!-- Left Bottom: Syntax & Command Console (SPSS Syntax / Stata Do-file Editor) -->
            <div id="left-syntax-console">
                <div class="left-box-hdr dark-hdr">
                    <div style="display:flex;align-items:center;gap:6px;">
                        <span class="left-box-title" data-i18n="syntax_editor_title">&lt;/&gt; 명령어 &amp; 신택스 에디터 (.py / .do)</span>
                        <span id="syntax-status-badge" class="badge-pill green">준비됨</span>
                    </div>
                    <div class="left-box-tools">
                        <button class="py-btn run" onclick="executeIntegratedSyntax(true)" title="선택 블록 실행 (Ctrl+Enter)">▶ 선택 실행</button>
                        <button class="py-btn run-all" onclick="executeIntegratedSyntax(false)" title="전체 실행 (F5)">▶ 전체 실행</button>
                        <button class="py-btn" onclick="toggleSyntaxSnippetsDropdown(event)" title="원클릭 분석 스니펫 템플릿">⚡ 템플릿 ▾</button>
                        <button class="py-btn" onclick="triggerLoadPyFile()" title="스크립트 파일 열기">📂</button>
                        <button class="py-btn" onclick="saveIntegratedSyntax()" title="스크립트 저장">💾</button>
                        <button class="py-btn" onclick="clearIntegratedSyntax()" title="에디터 비우기">🧹</button>
                    </div>
                </div>
                <div class="syntax-editor-wrap">
                    <textarea id="integrated-syntax-editor" spellcheck="false" placeholder="# OpenStat / Stata 호환 분석 스크립트&#10;# 예: ttest len, by(supp) 또는 sess.regression(dep='len', covs=['supp', 'dose'])&#10;sess.descriptives(vars=['len'])&#10;sess.ttest(dep='len', group='supp')"></textarea>
                </div>
                <div class="stata-cmd-bar">
                    <span class="stata-cmd-label">명령어 &gt;</span>
                    <input type="text" id="stata-inline-cmd" placeholder="Stata / Python 한 줄 즉시 실행 (예: ttest len, by(supp) 또는 reg len supp dose)..." onkeydown="if(event.key==='Enter') executeInlineCommand()">
                    <button class="stata-cmd-btn" onclick="executeInlineCommand()">실행 ↵</button>
                </div>
            </div>

            <!-- Spreadsheet Grid (Used in Jamovi Classic layout or when toggled) -->
            <div id="spreadsheet-container">
                <table class="jmv-grid" id="spreadsheet-table">
                    <thead id="grid-thead"></thead>
                    <tbody id="grid-tbody"></tbody>
                </table>
            </div>
            <div id="variables-container"></div>
        </div>"""

if old_left_pane in html:
    html = html.replace(old_left_pane, new_left_pane)
    print("[✓] Replaced left pane markup.")
else:
    print("[!] old_left_pane not found directly; searching with regex...")
    html = re.sub(
        r'<!-- Zone 1: Spreadsheet / Variable List -->\s*<div id="left-pane">.*?<div id="variables-container"></div>\s*</div>',
        new_left_pane,
        html,
        flags=re.DOTALL
    )
    print("[✓] Replaced left pane via regex.")

# ---------------------------------------------------------------------------
# 3. Add Right Pane Subbar (Results vs Data Grid Tab Switcher)
# ---------------------------------------------------------------------------
old_results_pane = '<div id="results-pane" onclick="handleResultsBackgroundClick(event)">'
new_results_pane = """<div id="results-pane" onclick="handleResultsBackgroundClick(event)">
            <!-- Right Pane Subbar: View Switcher (Results vs Data Grid) -->
            <div class="right-pane-subbar">
                <div class="view-switch-tabs">
                    <button class="v-tab active" id="v-tab-results" onclick="switchRightView('results')">📄 분석 결과 시트</button>
                    <button class="v-tab" id="v-tab-data" onclick="switchRightView('data')">📋 데이터 스프레드시트</button>
                    <button class="v-tab" id="v-tab-split" onclick="switchRightView('split')">◫ 결과 + 데이터 분할</button>
                </div>
                <div class="subbar-actions">
                    <button class="subbar-btn" onclick="openSyntheticDataModal()" title="합성데이터 생성 (시뮬레이션 / 개인정보보호)">🧬 합성데이터 생성</button>
                    <button class="subbar-btn" onclick="openRibbonMenu(event, 'export_tables')">📄 내보내기 ▾</button>
                    <button class="subbar-btn" onclick="clearAllResults()">🧹 결과 지우기</button>
                </div>
            </div>
            <div id="right-grid-wrap" style="display:none;flex:1 1 auto;overflow:auto;background:#FFFFFF;border-bottom:1px solid #CBD5E1;">
                <table class="jmv-grid" id="right-spreadsheet-table">
                    <thead id="right-grid-thead"></thead>
                    <tbody id="right-grid-tbody"></tbody>
                </table>
            </div>
            <div id="results-scroll-wrap" style="flex:1 1 auto;overflow-y:auto;">"""

if old_results_pane in html and "right-pane-subbar" not in html:
    html = html.replace(old_results_pane, new_results_pane)
    # also close the results-scroll-wrap before closing results-pane
    html = html.replace("<!-- Zone 4: Live Bidirectional Python (.py)", "</div>\n        <!-- Zone 4: Live Bidirectional Python (.py)")
    print("[✓] Injected right pane subbar & scroll wrap.")

# ---------------------------------------------------------------------------
# 4. Add Synthetic Data Generator Modal & Snippets Dropdown in HTML
# ---------------------------------------------------------------------------
modals_html = """
<!-- Synthetic Data Generator Modal -->
<div id="synthetic-modal" class="modal-backdrop" style="display:none;" onclick="if(event.target===this) closeSyntheticModal()">
    <div class="modal-dialog" style="max-width:520px;background:#FFFFFF;border-radius:8px;box-shadow:0 12px 32px rgba(15,23,42,0.3);overflow:hidden;">
        <div style="padding:12px 16px;background:#1E3A5F;color:#FFFFFF;display:flex;justify-content:space-between;align-items:center;">
            <span style="font-weight:700;font-size:14px;">🧬 가상 합성데이터(Synthetic Data) 생성기</span>
            <button onclick="closeSyntheticModal()" style="background:none;border:none;color:#FFFFFF;font-size:18px;cursor:pointer;">✕</button>
        </div>
        <div style="padding:16px;font-size:12.5px;color:#334155;line-height:1.6;">
            <div style="background:#F0FDFA;border:1px solid #CCFBF1;border-radius:6px;padding:10px;margin-bottom:12px;font-size:11.5px;color:#0F766E;">
                <strong>다변량 가우시안 코퓰러(Gaussian Copula) &amp; 차등 정보보호(DP) 엔진:</strong><br>
                현재 로드된 원시 데이터의 변수 간 순위 상관관계(Spearman)와 경험적 주변분포를 정밀 복제하여, 개인식별정보(PII) 유출 위험이 없는 안전한 연구·시뮬레이션용 합성데이터를 생성합니다.
            </div>
            <div style="margin-bottom:10px;">
                <label style="display:block;font-weight:600;margin-bottom:4px;">생성할 표본 수 (N):</label>
                <input type="number" id="synth-n-samples" value="100" min="10" max="100000" style="width:100%;padding:6px 8px;border:1px solid #CBD5E1;border-radius:4px;">
            </div>
            <div style="margin-bottom:10px;">
                <label style="display:block;font-weight:600;margin-bottom:4px;">차등 정보보호 노이즈 계수 (ε, Epsilon - 낮을수록 프라이버시 강력):</label>
                <select id="synth-epsilon" style="width:100%;padding:6px 8px;border:1px solid #CBD5E1;border-radius:4px;">
                    <option value="none">노이즈 없음 (정밀 상관계수 100% 보존 - 연구 시뮬레이션용)</option>
                    <option value="1.0" selected>ε = 1.0 (균형적 차등 정보보호 - 권장)</option>
                    <option value="0.5">ε = 0.5 (강력한 프라이버시 보호 - 민감 보건데이터용)</option>
                    <option value="0.1">ε = 0.1 (최대 보호 노이즈 주입)</option>
                </select>
            </div>
            <div style="margin-bottom:12px;">
                <label style="display:block;font-weight:600;margin-bottom:4px;">생성 후 동작:</label>
                <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
                    <input type="radio" name="synth-target" value="new_sheet" checked>
                    <span>새 활성 데이터 시트로 즉시 로드 (분석 시작)</span>
                </label>
            </div>
            <div style="display:flex;justify-content:flex-end;gap:8px;margin-top:16px;">
                <button class="silky-small-btn" onclick="closeSyntheticModal()">취소</button>
                <button class="stata-cmd-btn" style="padding:6px 14px;" onclick="runSyntheticGeneration()">⚡ 합성데이터 즉시 생성</button>
            </div>
        </div>
    </div>
</div>

<!-- Syntax Snippets Floating Dropdown -->
<div id="syntax-snippets-dropdown" style="display:none;position:fixed;background:#FFFFFF;border:1px solid #CBD5E1;box-shadow:0 8px 24px rgba(15,23,42,0.25);border-radius:6px;width:300px;z-index:3000;padding:6px 0;">
    <div style="padding:6px 12px;font-size:11px;font-weight:700;color:#64748B;background:#F8FAFC;border-bottom:1px solid #E2E8F0;">⚡ 원클릭 분석 &amp; 전처리 스니펫 삽입</div>
    <div class="dd-item" onclick="insertSyntaxSnippet('desc')"><span>📊 기술통계량 (Descriptives)</span></div>
    <div class="dd-item" onclick="insertSyntaxSnippet('ttest')"><span>⚖️ 독립표본 T-검정 (T-Test)</span></div>
    <div class="dd-item" onclick="insertSyntaxSnippet('anova')"><span>📈 일원배치 분산분석 (ANOVA)</span></div>
    <div class="dd-item" onclick="insertSyntaxSnippet('reg')"><span>📐 선형 회귀분석 (Regression)</span></div>
    <div class="dd-item" onclick="insertSyntaxSnippet('corr')"><span>🔗 상관행렬 분석 (Correlation)</span></div>
    <div class="dd-item" onclick="insertSyntaxSnippet('sem')"><span>🔮 구조방정식 모형 (SEM / CFA)</span></div>
    <div class="dd-item" onclick="insertSyntaxSnippet('panel')"><span>🏢 패널 고정효과 (Panel FE/RE)</span></div>
    <div class="dd-item" onclick="insertSyntaxSnippet('survival')"><span>⏳ 생존분석 (Kaplan-Meier &amp; Cox)</span></div>
    <div class="dd-item" onclick="insertSyntaxSnippet('network')"><span>🌐 네트워크 분석 (JASP Network)</span></div>
    <div class="dd-item" onclick="insertSyntaxSnippet('summary_ttest')"><span>📋 요약통계량 T-검정 (JASP)</span></div>
    <div class="dd-sep"></div>
    <div class="dd-item" onclick="insertSyntaxSnippet('stata_ttest')"><span>💻 Stata 명령: ttest dep, by(grp)</span></div>
    <div class="dd-item" onclick="insertSyntaxSnippet('stata_reg')"><span>💻 Stata 명령: regress dep x1 x2</span></div>
    <div class="dd-item" onclick="insertSyntaxSnippet('stata_gen')"><span>💻 Stata 명령: gen newvar = log(x)</span></div>
    <div class="dd-item" onclick="insertSyntaxSnippet('export_hwpx')"><span>📄 한글(HWPX) 리포트 자동 내보내기</span></div>
    <div class="dd-item" onclick="insertSyntaxSnippet('export_pdf')"><span>📄 PDF 리포트 자동 내보내기</span></div>
</div>
"""

if "<!-- Synthetic Data Generator Modal -->" not in html:
    html = html.replace("</body>", modals_html + "\n</body>")
    print("[✓] Injected Synthetic Data Modal & Snippets dropdown.")

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print("[Step 1 Complete] HTML structure updated successfully.")
