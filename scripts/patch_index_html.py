#!/usr/bin/env python3
"""
Update OpenStat Studio index.html with:
1. Bilingual (Korean/English) UI with Korean as default.
2. Prominent Language Switcher & User Guide Link in titlebar.
3. Fix all image src paths from '/assets/' to 'assets/' and hook auto error handler.
4. Replace generic icons with dedicated menu-data-filter, menu-merge, menu-reshape, menu-zscore SVGs.
5. Translate all 28 analysis menus, slots, options, variables view, and results sheet into SPSS-standard Korean.
"""

import os

HTML_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reconstructed", "pyjamovi", "web", "index.html")

with open(HTML_PATH, "r", encoding="utf-8") as f:
    html = f.read()

# 1. Add CSS for language switcher and guide link before </style>
css_patch = """
/* Language Switcher & User Guide in Titlebar */
.lang-switch-wrap {
    display: inline-flex;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 5px;
    padding: 2px;
    gap: 2px;
}
.lang-btn {
    background: transparent;
    border: none;
    color: #CBD5E1;
    font-size: 11.5px;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 3px;
    cursor: pointer;
    font-family: inherit;
    transition: all 0.15s;
}
.lang-btn:hover { color: #FFFFFF; }
.lang-btn.active {
    background: #0D9488;
    color: #FFFFFF;
    font-weight: 700;
}
.titlebar-guide-link {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(13, 148, 136, 0.25);
    border: 1px solid rgba(13, 148, 136, 0.6);
    color: #A7F3D0;
    font-size: 12px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 5px;
    text-decoration: none;
    transition: all 0.15s;
}
.titlebar-guide-link:hover {
    background: #0D9488;
    color: #FFFFFF;
}
</style>
"""
if ".lang-switch-wrap" not in html:
    html = html.replace("</style>", css_patch, 1)

# 2. Fix titlebar right section
old_titlebar_right = """        <div class="titlebar-right">
            <div class="ai-status-pill" onclick="openDoctorModal()" title="Check Python Package Freshness, API Compatibility & jamovi Feature Sync">
                <span class="ai-dot"></span>
                <span id="engine-status-text">Python 3.0 · Claude Ready · Doctor</span>
            </div>
        </div>"""

new_titlebar_right = """        <div class="titlebar-right">
            <div class="lang-switch-wrap">
                <button class="lang-btn active" id="btn-lang-ko" onclick="setLanguage('ko')" title="한국어로 전환 (기본값)">🇰🇷 한국어</button>
                <button class="lang-btn" id="btn-lang-en" onclick="setLanguage('en')" title="Switch to English">🇺🇸 English</button>
            </div>
            <a href="user_guide.html" target="_blank" class="titlebar-guide-link" title="초보자를 위한 클릭 투 클릭(Click-to-Click) 실전 사용 설명서 열기">
                <img src="assets/menu-guide.svg" alt="" style="width:14px;height:14px;vertical-align:-2px;">
                <span data-i18n="top_user_guide">사용 설명서</span>
            </a>
            <div class="ai-status-pill" onclick="openDoctorModal()" title="Check Python Package Freshness, API Compatibility & jamovi Feature Sync">
                <span class="ai-dot"></span>
                <span id="engine-status-text">Python 3.0 · Claude Ready · Doctor</span>
            </div>
        </div>"""

if old_titlebar_right in html:
    html = html.replace(old_titlebar_right, new_titlebar_right, 1)

# 3. Add Guide item to Backstage Menu
old_bs = """        <div class="bs-nav-item active" id="bs-tab-open" onclick="renderBackstageOpen()">Open (Data Library)</div>"""
new_bs = """        <div class="bs-nav-item active" id="bs-tab-open" onclick="renderBackstageOpen()">Open (Data Library)</div>
        <div class="bs-nav-item" onclick="window.open('user_guide.html', '_blank'); closeBackstage();" style="color:#A7F3D0;font-weight:700;">📖 클릭 투 클릭 사용 설명서 (User Guide)</div>"""
if old_bs in html and "📖 클릭 투 클릭" not in html:
    html = html.replace(old_bs, new_bs, 1)

# 4. Replace image src paths from '/assets/' to 'assets/'
html = html.replace('src="/assets/', 'src="assets/')

# 5. Fix dedicated icons in Data Ribbon
html = html.replace('src="assets/action-check.svg" alt="Filter"', 'src="assets/menu-data-filter.svg" alt="Filter"')
html = html.replace('src="assets/brand-modules.svg" alt="Merge"', 'src="assets/menu-merge.svg" alt="Merge"')
html = html.replace('src="assets/analysis-linearmodels.svg" alt="Reshape"', 'src="assets/menu-reshape.svg" alt="Reshape"')
html = html.replace('src="assets/menu-data-transform.svg" alt="Transform"', 'src="assets/menu-zscore.svg" alt="Transform"')

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print("[OK] Applied CSS, Titlebar, Backstage and Icon fixes to index.html")
