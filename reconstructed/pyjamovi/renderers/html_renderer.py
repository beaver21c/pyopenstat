"""
OpenStat Studio — HTML & APA 7th Report Renderer (v3.0)
License: MIT
Produces standalone, printable HTML reports with APA 3-line tables, vector/high-DPI figures, and reproducible Python syntax.
"""

from typing import List, Any
import html


def render_jamovi_html_report(analyses: List[Any], title: str = "OpenStat Studio — Statistical Analysis Report") -> str:
    """Generate a self-contained, publication-ready HTML report."""
    cards_html = []

    for idx, a in enumerate(analyses, start=1):
        a_title = getattr(a, "title", f"Analysis {idx}")
        a_content = getattr(a, "to_html_snippet", lambda: "")()
        syn_py = getattr(a, "syntax_python", "")
        syn_block = (
            f"<pre class='code-box'>{html.escape(syn_py)}</pre>"
            if syn_py else ""
        )
        card = f"""
        <section class="ost-card">
            <header class="ost-card-header">
                <span class="ost-badge">ANALYSIS {idx}</span>
                <h2>{html.escape(a_title)}</h2>
            </header>
            <div class="ost-card-body">
                {syn_block}
                {a_content}
            </div>
        </section>
        """
        cards_html.append(card)

    cards_joined = "\n".join(cards_html) if cards_html else "<p style='color:#64748B;'>No analyses executed yet.</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        :root {{
            --ost-slate: #1E3A5F;
            --ost-teal: #0D9488;
            --ost-bg: #F4F6F9;
            --ost-card-bg: #FFFFFF;
            --text-primary: #1E293B;
            --text-secondary: #64748B;
            --border-color: #E2E8F0;
            --table-rule: #334155;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "KoPubDotum Bold", "Noto Sans KR", Arial, sans-serif;
            background-color: var(--ost-bg);
            color: var(--text-primary);
            line-height: 1.55;
            margin: 0;
            padding: 32px 24px;
        }}
        .container {{
            max-width: 1080px;
            margin: 0 auto;
        }}
        .header-bar {{
            background: linear-gradient(135deg, var(--ost-slate) 0%, #155E75 100%);
            color: white;
            padding: 24px 32px;
            border-radius: 8px;
            margin-bottom: 24px;
            box-shadow: 0 4px 10px rgba(15, 23, 42, 0.12);
        }}
        .header-bar h1 {{
            margin: 0 0 6px 0;
            font-size: 24px;
            font-weight: 700;
        }}
        .header-bar p {{
            margin: 0;
            opacity: 0.9;
            font-size: 13.5px;
        }}
        .ost-card {{
            background: var(--ost-card-bg);
            border-radius: 8px;
            border: 1px solid var(--border-color);
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
            margin-bottom: 24px;
            overflow: hidden;
        }}
        .ost-card-header {{
            background: #F8FAFC;
            border-bottom: 1px solid var(--border-color);
            padding: 15px 24px;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .ost-badge {{
            background: var(--ost-teal);
            color: white;
            font-size: 11px;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 4px;
            letter-spacing: 0.5px;
        }}
        .ost-card-header h2 {{
            margin: 0;
            font-size: 18px;
            color: var(--ost-slate);
        }}
        .ost-card-body {{
            padding: 20px 24px;
        }}
        .code-box {{
            background: #0F172A;
            color: #E2E8F0;
            padding: 10px 14px;
            border-radius: 6px;
            font-family: "Consolas", "SFMono-Regular", Menlo, monospace;
            font-size: 12px;
            overflow-x: auto;
            margin: 0 0 18px 0;
        }}
        .jamovi-table-container {{
            margin-bottom: 20px;
            overflow-x: auto;
        }}
        .jamovi-table-container h4 {{
            margin: 0 0 8px 0;
            font-size: 14.5px;
            color: var(--ost-slate);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13.5px;
            margin-bottom: 8px;
        }}
        th, td {{
            padding: 7px 12px;
            text-align: left;
        }}
        th.num, td.num {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        th {{
            border-top: 2px solid var(--table-rule);
            border-bottom: 1px solid var(--table-rule);
            font-weight: 600;
            color: #0F172A;
            background: #F8FAFC;
        }}
        td {{
            border-bottom: 1px solid var(--border-color);
        }}
        tr:last-child td {{
            border-bottom: 2px solid var(--table-rule);
        }}
        .footnote {{
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 4px;
            font-style: italic;
        }}
        .apa-summary {{
            background: #F0FDFA;
            border-left: 4px solid var(--ost-teal);
            padding: 12px 16px;
            border-radius: 0 6px 6px 0;
            font-size: 13.5px;
            color: #134E4A;
            margin-top: 16px;
        }}
        footer {{
            text-align: center;
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 36px;
            line-height: 1.6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header-bar">
            <h1>{html.escape(title)}</h1>
            <p>OpenStat Studio v3.0 (MIT License) · Pure-Python Statistical Engine &amp; Claude Code Runtime</p>
        </div>
        {cards_joined}
        <footer>
            Generated by <strong>OpenStat Studio v3.0</strong> (<code>openstat</code> / <code>pyjamovi</code>, MIT License).<br>
            Analytical models, menu taxonomy, and <code>.omv</code> interoperability designed with reference to <a href="https://www.jamovi.org" style="color:#0D9488;">The jamovi project</a>.
        </footer>
    </div>
</body>
</html>
"""
