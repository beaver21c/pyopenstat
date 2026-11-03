#!/usr/bin/env python3
"""Update welcome screen and results interpretation in index.html"""
import os

HTML_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reconstructed", "pyjamovi", "web", "index.html")

with open(HTML_PATH, "r", encoding="utf-8") as f:
    text = f.read()

old_welcome = '<div style="font-size:20px;color:#1E3A5F;font-weight:700;">OpenStat Studio <span style="font-size:12.5px;color:#0D9488;font-weight:600;">v3.0 Open-Source Python Engine (MIT)</span></div>'

new_welcome = '''<div style="font-size:20px;color:#1E3A5F;font-weight:700;">OpenStat Studio <span style="font-size:12.5px;color:#0D9488;font-weight:600;">v3.0 순수 파이썬 통계 분석 &amp; 시각화 스튜디오 (MIT)</span></div>
                <div style="margin-top:6px;font-size:13px;color:#334155;max-width:640px;line-height:1.6;">
                    순수 파이썬(Pure Python) 기반 28종 통계 분석, Stata 수준 데이터 전처리(계산/재코딩/필터/병합), 15종 학술 차트 빌더, <strong>Claude Code</strong> 자연어 AI 연동, 그리고 <strong>한글(HWPX) / 엑셀(XLSX) / Word / PDF</strong> 출판용 리포트 자동 생성.<br>
                    <span style="font-size:11.5px;color:#64748B;">자모비(jamovi)의 직관적 UI와 모형 체계를 참고하여 개발된 100% 클린룸 독자 오픈소스 소프트웨어입니다.</span>
                </div>'''

if old_welcome in text:
    pos1 = text.find(old_welcome)
    pos2 = text.find('<div class="welcome-quick-pills">')
    if pos1 != -1 and pos2 != -1:
        text = text[:pos1] + new_welcome + "\n                " + text[pos2:]

# Update pills to Korean
text = text.replace("ToothGrowth (T-Test / ANOVA / Bayes)", "ToothGrowth (T-검정 / ANOVA / 베이지안)")
text = text.replace("Iris (ANOVA / Reg / PCA)", "Iris 붓꽃 (ANOVA / 회귀분석 / PCA)")
text = text.replace("SEM_Latent3Factor (CFA / SEM)", "SEM_Latent3Factor (구조방정식 SEM / CFA)")
text = text.replace("Grunfeld_Panel (Panel FE / RE / Hausman)", "Grunfeld_Panel (패널 고정효과 FE / 하우스만 검정)")
text = text.replace("MedMod_WorkStress (Mediation / Logistic)", "MedMod_WorkStress (매개효과 / 조절효과)")
text = text.replace("Clinical_Survival (Kaplan-Meier / Cox)", "Clinical_Survival (생존분석 Kaplan-Meier / Cox)")
text = text.replace("Big5 (Reliability / EFA)", "Big5 성격검사 (신뢰도 α / 탐색적 요인분석 EFA)")

# Ensure isKo is defined in renderAllResults
target_render = 'function renderAllResults() {\n    const welcome = document.getElementById("welcome-screen");'
replace_render = 'function renderAllResults() {\n    const lang = APP.lang || "ko";\n    const isKo = lang === "ko";\n    const welcome = document.getElementById("welcome-screen");'
if target_render in text and "const isKo = lang === \"ko\";" not in text:
    text = text.replace(target_render, replace_render, 1)

text = text.replace("<strong>APA 7th Interpretation:</strong>", "<strong>${(APP.lang || 'ko') === 'ko' ? 'APA 7판 학술 해석:' : 'APA 7th Interpretation:'}</strong>")

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(text)

print("[OK] Welcome screen & Results interpretation updated to Korean defaults.")
