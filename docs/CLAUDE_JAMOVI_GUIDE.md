# Claude Code × OpenStat Studio v3.0 Integration Guide

See root [`CLAUDE.md`](../CLAUDE.md) for the auto-loaded Claude Code instructions and `.mcp.json` (generated per machine by [`scripts/setup_claude_code.py`](../scripts/setup_claude_code.py)) for zero-config MCP server registration.

## 4 Ways Claude Code Directly Operates OpenStat Studio

| Mode | Entrypoint | Best Use Case |
| :--- | :--- | :--- |
| **1. MCP Server** | `.venv/bin/openstat-mcp` (via `.mcp.json`) | Native tool calls (`openstat_run_analysis`, `openstat_exec_python_syntax`, `openstat_doctor`, `openstat_sync_jamovi`) directly inside Claude Code chat |
| **2. CLI (`openstat`)** | `.venv/bin/openstat` | One-liner bash execution with `--out-py`, `--export-table` (`.xlsx`/`.docx`/`.hwpx`/`.tex`), and `--export-plot` (`.svg`/`.pdf`) |
| **3. Live `.py` Sync** | `workspace/live_session.py` | Bidirectional Python script authoring shared between Claude Code and the OpenStat Studio Results Sheet |
| **4. Python SDK** | `import openstat as ost` | Complex multi-model pipelines, custom `PlotStyle` loops, Panel FE/RE Hausman comparisons, and `lavaan` SEM models |

## Maintenance & Evolution Commands

1. **Check Package Freshness & Upstream Python API Drift (`doctor`)**:
   ```bash
   .venv/bin/openstat doctor --json
   ```
2. **Detect & Scaffold Newly Added jamovi Features (`sync-jamovi`)**:
   ```bash
   .venv/bin/openstat sync-jamovi
   .venv/bin/openstat sync-jamovi --scaffold friedman
   ```
3. **Rebuild Catalog, Clean Icons & Run Verification (`rebuild`)**:
   ```bash
   .venv/bin/openstat rebuild
   ```
