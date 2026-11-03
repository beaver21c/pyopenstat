#!/usr/bin/env python3
"""
Patch pyjamovi Web Studio for:
1. New Window Arrangement with Default: Left-Top Variables Explorer + Left-Bottom Syntax/Command Console, Right Results/Data
2. Comprehensive Korean Localization (SPSS/jamovi Standard Korean for all analysis options, buttons, and dialogs)
3. JASP Features Integration (Network Analysis & Summary Statistics T-Test in Menus & UI Schemas)
4. Stata Command Execution & Synthetic Data Generator UI
"""

import re
import os

HTML_PATH = "/mnt/share/temp/pyjamovi/reconstructed/pyjamovi/web/index.html"

with open(HTML_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# ---------------------------------------------------------------------------
# 1. Add Layout & Sub-Pane CSS
# ---------------------------------------------------------------------------
layout_css = """
/* === V3.1 Layout & Multi-Pane Styling === */
body {
    --left-top-height: 42%;
}

/* Default Layout: Left Column has Variables (Top) and Syntax Editor (Bottom) */
body[data-layout="default"] #left-pane,
body:not([data-layout]) #left-pane {
    width: 42%;
    min-width: 320px;
    display: flex;
    flex-direction: column;
}
body[data-layout="default"] #left-var-explorer,
body:not([data-layout]) #left-var-explorer {
    display: flex;
    flex-direction: column;
    height: var(--left-top-height, 42%);
    min-height: 140px;
    background: #FFFFFF;
    border-bottom: 1px solid #CBD5E1;
    overflow: hidden;
}
body[data-layout="default"] #left-h-splitter,
body:not([data-layout]) #left-h-splitter {
    display: block;
    height: 6px;
    background: #E2E8F0;
    cursor: row-resize;
    flex-shrink: 0;
    border-top: 1px solid #CBD5E1;
    border-bottom: 1px solid #CBD5E1;
    transition: background 0.15s;
}
body[data-layout="default"] #left-h-splitter:hover,
body:not([data-layout]) #left-h-splitter:hover {
    background: #0D9488;
}
body[data-layout="default"] #left-syntax-console,
body:not([data-layout]) #left-syntax-console {
    display: flex;
    flex-direction: column;
    flex: 1 1 auto;
    min-height: 160px;
    background: #0F172A;
    color: #E2E8F0;
    overflow: hidden;
}
body[data-layout="default"] #spreadsheet-container,
body:not([data-layout]) #spreadsheet-container {
    display: none;
}
body[data-layout="default"] #variables-container,
body:not([data-layout]) #variables-container {
    display: none;
}

/* Jamovi Classic Layout: Left Data Grid, Right Results */
body[data-layout="jamovi"] #left-pane {
    width: 44%;
    display: flex;
}
body[data-layout="jamovi"] #left-var-explorer,
body[data-layout="jamovi"] #left-h-splitter,
body[data-layout="jamovi"] #left-syntax-console {
    display: none !important;
}
body[data-layout="jamovi"] #spreadsheet-container {
    display: block;
    flex: 1 1 auto;
}

/* Syntax Focused Layout: Left Column is 100% Syntax Editor */
body[data-layout="syntax"] #left-pane {
    width: 46%;
    display: flex;
}
body[data-layout="syntax"] #left-var-explorer,
body[data-layout="syntax"] #left-h-splitter,
body[data-layout="syntax"] #spreadsheet-container {
    display: none !important;
}
body[data-layout="syntax"] #left-syntax-console {
    display: flex;
    flex: 1 1 100%;
}

/* 4-Panes Layout (Quad / RStudio Style) */
body[data-layout="quad"] #left-pane {
    width: 42%;
    display: flex;
    flex-direction: column;
}
body[data-layout="quad"] #left-var-explorer {
    display: flex;
    flex-direction: column;
    height: 45%;
}
body[data-layout="quad"] #left-h-splitter {
    display: block;
    height: 6px;
    cursor: row-resize;
}
body[data-layout="quad"] #left-syntax-console {
    display: flex;
    flex-direction: column;
    flex: 1 1 auto;
}
body[data-layout="quad"] #right-pane-wrap {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
body[data-layout="quad"] #quad-grid-container {
    display: block;
    height: 45%;
    border-bottom: 1px solid #CBD5E1;
    overflow: auto;
}
body[data-layout="quad"] #results-pane {
    flex: 1 1 auto;
}

/* Fullscreen Modes */
body[data-layout="results-only"] #left-pane,
body[data-layout="results-only"] #main-splitter,
body[data-layout="results-only"] #options-pane {
    display: none !important;
}
body[data-layout="results-only"] #results-pane {
    width: 100% !important;
}

body[data-layout="data-only"] #results-pane,
body[data-layout="data-only"] #options-pane,
body[data-layout="data-only"] #main-splitter {
    display: none !important;
}
body[data-layout="data-only"] #left-pane {
    width: 100% !important;
}
body[data-layout="data-only"] #spreadsheet-container {
    display: block !important;
}
body[data-layout="data-only"] #left-var-explorer,
body[data-layout="data-only"] #left-h-splitter,
body[data-layout="data-only"] #left-syntax-console {
    display: none !important;
}

/* Pane Header Toolbars */
.left-box-hdr {
    padding: 7px 10px;
    background: #F1F5F9;
    border-bottom: 1px solid #E2E8F0;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    font-weight: 600;
    color: #1E293B;
    flex-shrink: 0;
}
.left-box-hdr.dark-hdr {
    background: #1E293B;
    color: #F8FAFC;
    border-bottom: 1px solid #334155;
}
.left-box-title {
    font-size: 12px;
    font-weight: 700;
    white-space: nowrap;
}
.badge-pill {
    padding: 2px 7px;
    border-radius: 10px;
    font-size: 10.5px;
    font-weight: 600;
    background: #E2E8F0;
    color: #475569;
}
.badge-pill.green {
    background: #064E3B;
    color: #6EE7B7;
}
.left-box-tools {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 5px;
}
#left-var-filter {
    padding: 3px 7px;
    border: 1px solid #CBD5E1;
    border-radius: 4px;
    font-size: 11.5px;
    width: 110px;
}
.left-small-icon-btn {
    padding: 3px 5px;
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 4px;
    cursor: pointer;
    display: flex;
    align-items: center;
}
.left-small-icon-btn img {
    width: 13px;
    height: 13px;
}
.left-small-icon-btn:hover {
    background: #E2E8F0;
}

/* Variable Table in Left Box */
#left-var-list-scroll {
    flex: 1 1 auto;
    overflow-y: auto;
    background: #FFFFFF;
}
.left-var-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 11.5px;
}
.left-var-table th {
    position: sticky;
    top: 0;
    background: #F8FAFC;
    padding: 5px 6px;
    text-align: left;
    color: #64748B;
    font-weight: 600;
    border-bottom: 1px solid #E2E8F0;
    z-index: 2;
}
.left-var-table td {
    padding: 5px 6px;
    border-bottom: 1px solid #F1F5F9;
    color: #334155;
}
.left-var-table tr:hover {
    background: #F0FDFA;
    cursor: pointer;
}
.left-var-table tr.selected {
    background: #CCFBF1;
    font-weight: 600;
}
.var-quick-add {
    padding: 1px 5px;
    background: #E0F2FE;
    color: #0369A1;
    border-radius: 3px;
    font-size: 11px;
    cursor: pointer;
    display: inline-block;
}
.var-quick-add:hover {
    background: #0284C7;
    color: #FFFFFF;
}

/* Integrated Syntax Console */
.syntax-editor-wrap {
    flex: 1 1 auto;
    display: flex;
    background: #0F172A;
    overflow: hidden;
    position: relative;
}
#integrated-syntax-editor {
    width: 100%;
    height: 100%;
    background: #0B1120;
    color: #E2E8F0;
    border: none;
    padding: 8px 10px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 12px;
    line-height: 1.5;
    resize: none;
    outline: none;
}
.stata-cmd-bar {
    padding: 6px 10px;
    background: #1E293B;
    border-top: 1px solid #334155;
    display: flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
}
.stata-cmd-label {
    font-size: 11.5px;
    font-weight: 700;
    color: #38BDF8;
    white-space: nowrap;
}
#stata-inline-cmd {
    flex: 1 1 auto;
    padding: 4px 8px;
    background: #0F172A;
    color: #F8FAFC;
    border: 1px solid #475569;
    border-radius: 3px;
    font-size: 12px;
    font-family: monospace;
}
#stata-inline-cmd:focus {
    border-color: #38BDF8;
    outline: none;
}
.stata-cmd-btn {
    padding: 4px 10px;
    background: #0284C7;
    color: #FFFFFF;
    border: none;
    border-radius: 3px;
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
}
.stata-cmd-btn:hover {
    background: #0369A1;
}

/* Layout Switcher UI */
.layout-selector-wrap {
    position: relative;
    display: inline-block;
}
.layout-dd {
    position: absolute;
    top: 100%;
    right: 0;
    margin-top: 4px;
    width: 320px;
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.2);
    display: none;
    z-index: 2500;
    padding: 6px 0;
}
.layout-dd.open {
    display: block;
}
.layout-dd-item {
    padding: 8px 12px;
    cursor: pointer;
    border-left: 3px solid transparent;
}
.layout-dd-item:hover {
    background: #F1F5F9;
}
.layout-dd-item.active {
    background: #F0FDFA;
    border-left-color: #0D9488;
}
.layout-dd-title {
    font-size: 12.5px;
    font-weight: 700;
    color: #1E293B;
}
.layout-dd-desc {
    font-size: 11px;
    color: #64748B;
    margin-top: 2px;
    line-height: 1.35;
}
.layout-dd-sep {
    height: 1px;
    background: #E2E8F0;
    margin: 4px 0;
}

/* Right View Switch Tabs */
.right-pane-subbar {
    padding: 4px 10px;
    background: #F8FAFC;
    border-bottom: 1px solid #E2E8F0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
}
.view-switch-tabs {
    display: flex;
    gap: 4px;
}
.v-tab {
    padding: 4px 10px;
    border: 1px solid #CBD5E1;
    border-radius: 4px;
    background: #FFFFFF;
    font-size: 11.5px;
    font-weight: 600;
    color: #475569;
    cursor: pointer;
}
.v-tab:hover {
    background: #F1F5F9;
}
.v-tab.active {
    background: #0D9488;
    color: #FFFFFF;
    border-color: #0F766E;
}
.subbar-btn {
    padding: 3px 8px;
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 4px;
    font-size: 11px;
    color: #334155;
    cursor: pointer;
}
.subbar-btn:hover {
    background: #F1F5F9;
}
"""

# Insert CSS into <style> before </style>
if "/* === V3.1 Layout & Multi-Pane Styling === */" not in content:
    content = content.replace("</style>", layout_css + "\n</style>", 1)

print("[1/5] Injected layout CSS rules.")

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(content)
