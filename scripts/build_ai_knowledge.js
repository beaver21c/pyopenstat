#!/usr/bin/env node
/* Builds reconstructed/pyjamovi/ai/analysis_schemas.json — the option reference the AI assistant (Gemini) reads.

   The analysis forms of the web UI (ANALYSIS_SCHEMAS in web/index.html) are the single source of every analysis
   option: role slots, option names, defaults and choices. This script opens index.html in headless Chromium,
   reads those objects and writes them as JSON, so the assistant is taught the option names the program really has.

   Run after changing an analysis form:   node scripts/build_ai_knowledge.js
   (needs the `playwright` package with Chromium; tests/test_ai_assistant.py fails when the file is out of date)
*/
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const ROOT = path.resolve(__dirname, "..");
const INDEX = path.join(ROOT, "reconstructed", "pyjamovi", "web", "index.html");
const OUT = path.join(ROOT, "reconstructed", "pyjamovi", "ai", "analysis_schemas.json");

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto("file://" + INDEX);
  await page.waitForFunction(() => typeof ANALYSIS_SCHEMAS === "object" && typeof RIBBON_MENUS_I18N === "object");
  const out = await page.evaluate(() => {
    const plain = (o) => JSON.parse(JSON.stringify(o));          // drops functions
    const labels = (lang) => {
      const m = {};
      for (const g of Object.values(RIBBON_MENUS_I18N[lang] || {}))
        if (Array.isArray(g)) for (const it of g) if (it && it.id && it.label) m[it.id] = it.label;
      return m;
    };
    return { schemas: plain(ANALYSIS_SCHEMAS), menu_ko: labels("ko"), menu_en: labels("en") };
  });
  await browser.close();
  const ids = Object.keys(out.schemas).sort();
  const doc = { source: "reconstructed/pyjamovi/web/index.html (ANALYSIS_SCHEMAS, RIBBON_MENUS_I18N)", n: ids.length,
                menu_ko: out.menu_ko, menu_en: out.menu_en, schemas: Object.fromEntries(ids.map((k) => [k, out.schemas[k]])) };
  fs.writeFileSync(OUT, JSON.stringify(doc, null, 1) + "\n");
  console.log(`wrote ${path.relative(ROOT, OUT)} — ${ids.length} analyses`);
})().catch((e) => { console.error(e); process.exit(1); });
