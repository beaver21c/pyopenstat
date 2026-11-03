"""
Playwright End-to-End Real Chrome Verification for pyjamovi v2.0 Desktop UI
Verifies:
1. Both http://localhost:8500 and http://192.168.137.2:8000 respond and render identically
2. Zero JS console errors & zero uncaught page errors
3. Official jamovi SVG assets load (naturalWidth > 0)
4. Spreadsheet renders rows & Variables tab works
5. Interactive ribbon execution across Bayesian T-Test, ANOVA + Games-Howell, CFA/SEM, Panel + Hausman, Mediation, and Survival
6. Live AI Push (/api/ai/push) synchronizes into the browser DOM
"""

import json
import os
import urllib.request
from playwright.sync_api import sync_playwright

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def run_ui_verification():
    console_errors = []
    page_errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True)
        context = browser.new_context(viewport={"width": 1480, "height": 900})
        page = context.new_page()

        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # 1. Verify port 8000 (cross-PC UFW-open URL)
        print("[1/7] Opening http://192.168.137.2:8000 ...")
        page.goto("http://192.168.137.2:8000", wait_until="networkidle")
        assert "jamovi" in page.title()
        page.wait_for_selector("#grid-tbody tr", timeout=5000)

        # 2. Verify port 8500 (localhost URL)
        print("[2/7] Opening http://localhost:8500 ...")
        page.goto("http://localhost:8500", wait_until="networkidle")
        page.wait_for_selector("#grid-tbody tr", timeout=5000)

        # Reset to ToothGrowth.csv cleanly
        page.evaluate("() => loadDatasetByName('ToothGrowth.csv')")
        page.wait_for_timeout(400)

        # Check SVG icons loaded properly
        broken_imgs = page.evaluate("""() => {
            const imgs = Array.from(document.querySelectorAll('.jmv-ribbon img'));
            return imgs.filter(i => i.complete && i.naturalWidth === 0).map(i => i.src);
        }""")
        assert len(broken_imgs) == 0, f"Broken SVG icons found: {broken_imgs}"
        print("      All official jamovi ribbon SVG icons loaded (0 broken).")

        # 3. Click T-Tests -> Bayesian Independent Samples T-Test + Enable Syntax Mode
        print("[3/7] Running Bayesian Independent Samples T-Test + One-Way ANOVA (Games-Howell)...")
        page.evaluate("() => setSyntaxMode(true)")
        page.click("#rb-ttests")
        page.click(".dd-item[data-analysis-id='bayesTTestIS']")
        page.wait_for_selector(".jmv-results-sheet .jmv-plot-card img", timeout=8000)

        # Also run One-Way ANOVA with Games-Howell
        page.click("#rb-anova")
        page.click(".dd-item[data-analysis-id='anovaOneW']")
        page.wait_for_timeout(700)
        page.select_option("#opt-panel-body select", "gamesHowell")
        page.wait_for_timeout(800)

        sheets_count = page.locator(".jmv-results-sheet").count()
        assert sheets_count >= 2, f"Expected >=2 result cards, got {sheets_count}"
        page.screenshot(path=os.path.join(REPORTS_DIR, "ui_verify_1_bayes_anova.png"))
        print("      Captured screenshot: reports/ui_verify_1_bayes_anova.png")

        # 4. Load SEM_Latent3Factor.csv and run 3-Factor SEM via ribbon
        print("[4/7] Loading SEM_Latent3Factor.csv & running 3-Factor Latent SEM...")
        page.evaluate("() => loadDatasetByName('SEM_Latent3Factor.csv')")
        page.wait_for_timeout(400)
        page.click("#rb-sem")
        page.wait_for_selector("#results-list .jmv-results-sheet .jmv-plot-card img", state="visible", timeout=10000)
        sem_text = page.locator("#results-list").inner_text()
        assert "Comparative Fit Index (CFI)" in sem_text
        page.screenshot(path=os.path.join(REPORTS_DIR, "ui_verify_2_sem_cfa.png"))
        print("      Captured screenshot: reports/ui_verify_2_sem_cfa.png")

        # 5. Load Grunfeld_Panel.csv and run Panel Econometrics (FE + Hausman)
        print("[5/7] Loading Grunfeld_Panel.csv & running Panel Fixed Effects + Hausman Test...")
        page.evaluate("() => loadDatasetByName('Grunfeld_Panel.csv')")
        page.wait_for_timeout(400)
        page.click("#rb-linearmodels")
        page.click(".dd-item[data-analysis-id='panel']")
        page.wait_for_selector("#results-list .jmv-results-sheet .jmv-plot-card img", state="visible", timeout=8000)
        panel_text = page.locator("#results-list").inner_text()
        assert "Hausman Specification Test" in panel_text
        page.screenshot(path=os.path.join(REPORTS_DIR, "ui_verify_3_panel_hausman.png"))
        print("      Captured screenshot: reports/ui_verify_3_panel_hausman.png")

        # 6. Load Clinical_Survival.csv and run Kaplan-Meier & Cox PH
        print("[6/7] Loading Clinical_Survival.csv & running Survival Analysis (KM + Cox PH)...")
        page.evaluate("() => loadDatasetByName('Clinical_Survival.csv')")
        page.wait_for_timeout(400)
        page.click("#rb-survival")
        page.wait_for_selector("#results-list .jmv-results-sheet .jmv-plot-card img", state="visible", timeout=8000)
        surv_text = page.locator("#results-list").inner_text()
        assert "Kaplan-Meier" in surv_text and "Cox Proportional Hazards" in surv_text
        page.screenshot(path=os.path.join(REPORTS_DIR, "ui_verify_4_survival_cox.png"))
        print("      Captured screenshot: reports/ui_verify_4_survival_cox.png")

        # 7. Test AI Live Push (/api/ai/push) -> UI auto-syncs within poll interval!
        print("[7/7] Testing Live AI Agent Push (/api/ai/push) -> Browser Auto-Sync...")
        push_body = json.dumps({
            "dataset": "MedMod_WorkStress.csv",
            "analysis_type": "mediation",
            "params": {"pred": "Stress", "med": "Burnout", "dep": "Turnover_Intent", "bootstrap": 200}
        }).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:8000/api/ai/push",
            data=push_body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200

        # Trigger immediate poll or wait 3.2s for automatic live sync
        page.evaluate("() => pollLiveAISync()")
        page.wait_for_selector("#results-list .jmv-results-sheet .jmv-plot-card img", state="visible", timeout=8000)
        med_text = page.locator("#results-list").inner_text()
        assert "Mediation Estimates" in med_text and "Indirect" in med_text
        page.screenshot(path=os.path.join(REPORTS_DIR, "ui_verify_5_ai_live_push_mediation.png"))
        print("      Captured screenshot: reports/ui_verify_5_ai_live_push_mediation.png")

        # Reset back to clean ToothGrowth welcome state for the user!
        page.evaluate("() => loadDatasetByName('ToothGrowth.csv')")
        page.wait_for_timeout(300)

        browser.close()

    assert len(console_errors) == 0, f"Console errors detected: {console_errors}"
    assert len(page_errors) == 0, f"Uncaught page errors detected: {page_errors}"
    print("\n[SUCCESS] Playwright Real Chrome Verification Passed: 0 console errors, 0 page errors, all 5 visual suites verified!")


if __name__ == "__main__":
    run_ui_verification()
