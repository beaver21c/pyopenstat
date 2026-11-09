"""Korean / English interface and documentation.

The web UI is written partly in Korean and partly in English. Fixed text in the other language lives in
web/assets/i18n_dict.js (UI_EN: Korean source -> English, UI_KO: English source -> Korean) and is applied by
localizeDom() in index.html. These tests keep every Korean label of the analysis forms translated and the
English README / user guide in step with the Korean ones.
"""
import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "reconstructed", "pyjamovi", "web")
HANGUL = re.compile(r"[가-힣]")


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def _norm(s):
    return re.sub(r"\s+", " ", s).strip()


def _dicts():
    js = _read("reconstructed", "pyjamovi", "web", "assets", "i18n_dict.js")
    en = json.loads(js[js.index("window.UI_EN = ") + len("window.UI_EN = "):js.index(";\nwindow.UI_KO")])
    ko = json.loads(js[js.index("window.UI_KO = ") + len("window.UI_KO = "):js.rindex(";")])
    return en, ko


def test_dictionary_loaded_and_applied():
    html = _read("reconstructed", "pyjamovi", "web", "index.html")
    assert '<script src="assets/i18n_dict.js' in html
    assert html.index('<script src="assets/i18n_dict.js') < html.index("const I18N = {")
    for name in ("function uiText(", "function localizeDom(", "function startUiI18nObserver(", "const TOAST_EN = [",
                 "function guideHref("):
        assert name in html, name
    # applyLanguage translates the static page, starts the observer and points guide links at the right file
    body = html[html.index("function applyLanguage() {"):]
    body = body[:body.index("\n}\n")]
    assert "localizeDom(document.body, lang)" in body and "startUiI18nObserver()" in body
    assert "user_guide_en.html" in body
    en, ko = _dicts()
    assert len(en) > 1500 and len(ko) > 150
    assert all(isinstance(v, str) and v.strip() for v in list(en.values()) + list(ko.values()))
    # keys are stored normalised, so lookups of rendered text (collapsed whitespace) hit
    assert all(k == _norm(k) for k in list(en) + list(ko))


def test_every_korean_form_label_has_english():
    """Titles, labels, help lines, hints, placeholders and select entries of the analysis forms."""
    html = _read("reconstructed", "pyjamovi", "web", "index.html")
    region = html[html.index("const TEXT_PREP_SECTION"):html.index("const REPORT_STYLE_KEY")]
    lits = re.findall(r'\b(?:label|title|help|hint|intro|text|placeholder)\s*:\s*"((?:[^"\\]|\\.)*)"', region)
    lits += re.findall(r'\[\s*"[A-Za-z0-9_.+:-]*"\s*,\s*"((?:[^"\\]|\\.)*)"\s*\]', region)
    korean = {_norm(json.loads('"' + s + '"')) for s in lits if HANGUL.search(s)}
    en, _ = _dicts()
    missing = sorted(k for k in korean if k not in en)
    assert len(korean) > 800
    assert not missing, f"Korean form text without English in assets/i18n_dict.js: {missing[:20]}"


def test_english_values_keep_hangul_only_for_data_names():
    """English entries may keep Korean only for sample-data names and Korean text-mining examples."""
    en, _ = _dicts()
    left = [v for v in en.values() if HANGUL.search(v)]
    assert len(left) < 40, left


def test_english_user_guide_matches_korean_structure():
    ko = _read("docs", "user_guide.html")
    en = _read("docs", "user_guide_en.html")
    assert _read("reconstructed", "pyjamovi", "web", "user_guide_en.html") == en
    assert _read("reconstructed", "pyjamovi", "web", "user_guide.html") == ko
    assert '<html lang="en">' in en and '<html lang="ko">' in ko
    ids = lambda s: re.findall(r'<section id="(ch-[\w-]+)"', s)
    assert ids(en) == ids(ko) and len(ids(ko)) >= 38
    assert 'href="user_guide.html"' in en and 'href="user_guide_en.html"' in ko
    assert "@page" in en and "window.print()" in en          # PDF output as in the Korean guide


def test_readme_language_selector():
    assert "[English](README_EN.md)" in _read("README.md")
    assert "[English](README_EN.md)" in _read("README_KR.md")
    readme_en = _read("README_EN.md")
    assert "[한국어](README.md)" in readme_en
    assert "docs/user_guide_en.html" in readme_en and 'id="disclaimer"' in readme_en


def test_server_serves_english_guide():
    try:
        from fastapi.testclient import TestClient
    except (ImportError, RuntimeError):
        pytest.skip("Starlette TestClient requires httpx")
    from pyjamovi.server import app
    c = TestClient(app)
    r = c.get("/user_guide_en.html")
    assert r.status_code == 200 and '<html lang="en">' in r.text
    r = c.get("/assets/i18n_dict.js")
    assert r.status_code == 200 and "window.UI_EN" in r.text
