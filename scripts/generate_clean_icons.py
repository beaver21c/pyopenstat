"""
OpenStat Studio — Clean-Room Original Vector Icon & Brand Emblem Generator
License: MIT / CC0-1.0 Public Domain Dedication
Generates 100% original geometric SVG icons for OpenStat Studio with zero third-party copyrighted graphics.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "reconstructed", "pyjamovi", "web", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

HEADER = '<!-- Original Vector Asset by OpenStat Studio Contributors | License: MIT / CC0-1.0 -->\n'

# Clean academic palette for original icons
C_SLATE = "#1E3A5F"
C_BLUE = "#2563EB"
C_TEAL = "#0D9488"
C_AMBER = "#D97706"
C_CORAL = "#E11D48"
C_VIOLET = "#7C3AED"
C_MUTED = "#64748B"


def wrap_svg(inner: str, size: int = 24, viewbox: str = "0 0 24 24") -> str:
    return (
        f'{HEADER}<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="{viewbox}" fill="none">\n{inner}\n</svg>\n'
    )


ICONS = {
    # 1. Original OpenStat Studio Brand Logo (Hex-Squircle + Ascending Bars + Bell/Node Curve — NO 'V' logo!)
    "brand-logo.svg": wrap_svg(
        f'  <rect x="1.5" y="1.5" width="21" height="21" rx="5.5" fill="{C_TEAL}" stroke="#FFFFFF" stroke-width="1.4"/>\n'
        '  <rect x="5.5" y="13" width="2.8" height="5.5" rx="1" fill="#FFFFFF"/>\n'
        '  <rect x="10.6" y="9.2" width="2.8" height="9.3" rx="1" fill="#FFFFFF"/>\n'
        '  <rect x="15.7" y="6.2" width="2.8" height="12.3" rx="1" fill="#A7F3D0"/>\n'
        '  <path d="M5.5 11.2 C8.5 6.5, 13.0 5.5, 18.2 4.8" stroke="#FEF08A" stroke-width="1.8" stroke-linecap="round"/>\n'
        '  <circle cx="18.2" cy="4.8" r="1.6" fill="#FEF08A"/>',
        size=28
    ),
    "brand-logo-mark.svg": wrap_svg(
        f'  <rect x="2" y="2" width="20" height="20" rx="5" fill="{C_SLATE}"/>\n'
        f'  <rect x="6" y="13" width="2.6" height="5" rx="0.8" fill="#38BDF8"/>\n'
        f'  <rect x="10.7" y="9.5" width="2.6" height="8.5" rx="0.8" fill="#2DD4BF"/>\n'
        f'  <rect x="15.4" y="6.5" width="2.6" height="11.5" rx="0.8" fill="#FBBF24"/>',
        size=24
    ),
    "brand-modules.svg": wrap_svg(
        f'  <rect x="3" y="3" width="7.5" height="7.5" rx="1.8" stroke="{C_SLATE}" stroke-width="1.7" fill="#E0F2FE"/>\n'
        f'  <rect x="13.5" y="3" width="7.5" height="7.5" rx="1.8" stroke="{C_TEAL}" stroke-width="1.7" fill="#CCFBF1"/>\n'
        f'  <rect x="3" y="13.5" width="7.5" height="7.5" rx="1.8" stroke="{C_SLATE}" stroke-width="1.7" fill="#E0F2FE"/>\n'
        f'  <path d="M17.25 14v6.5M14 17.25h6.5" stroke="{C_AMBER}" stroke-width="2.1" stroke-linecap="round"/>',
        size=28
    ),

    # 2. Ribbon Analysis Category Icons (Original Clean-Room Designs)
    "analysis-descriptives.svg": wrap_svg(
        f'  <rect x="3.5" y="12" width="3.8" height="8" rx="1" fill="{C_BLUE}"/>\n'
        f'  <rect x="9.8" y="6" width="3.8" height="14" rx="1" fill="{C_TEAL}"/>\n'
        f'  <rect x="16.1" y="9.5" width="3.8" height="10.5" rx="1" fill="{C_AMBER}"/>\n'
        f'  <line x1="2.5" y1="20.5" x2="21.5" y2="20.5" stroke="{C_SLATE}" stroke-width="1.6" stroke-linecap="round"/>',
        size=32
    ),
    "analysis-ttest.svg": wrap_svg(
        f'  <path d="M3 18.5 C5.5 18.5, 6.5 6.5, 9.5 6.5 C12.5 6.5, 13.5 18.5, 16 18.5" stroke="{C_BLUE}" stroke-width="2" fill="none"/>\n'
        f'  <path d="M8 18.5 C10.5 18.5, 11.5 8.5, 14.5 8.5 C17.5 8.5, 18.5 18.5, 21 18.5" stroke="{C_AMBER}" stroke-width="2" stroke-dasharray="2.2 1.5" fill="none"/>\n'
        f'  <line x1="2.5" y1="19.5" x2="21.5" y2="19.5" stroke="{C_SLATE}" stroke-width="1.5"/>',
        size=32
    ),
    "analysis-anova.svg": wrap_svg(
        f'  <line x1="6" y1="6" x2="6" y2="18" stroke="{C_BLUE}" stroke-width="1.8" stroke-linecap="round"/>\n'
        f'  <circle cx="6" cy="12" r="2.5" fill="#FFFFFF" stroke="{C_BLUE}" stroke-width="1.8"/>\n'
        f'  <line x1="12" y1="4.5" x2="12" y2="15.5" stroke="{C_TEAL}" stroke-width="1.8" stroke-linecap="round"/>\n'
        f'  <circle cx="12" cy="9.5" r="2.5" fill="#FFFFFF" stroke="{C_TEAL}" stroke-width="1.8"/>\n'
        f'  <line x1="18" y1="8" x2="18" y2="19.5" stroke="{C_AMBER}" stroke-width="1.8" stroke-linecap="round"/>\n'
        f'  <circle cx="18" cy="14" r="2.5" fill="#FFFFFF" stroke="{C_AMBER}" stroke-width="1.8"/>',
        size=32
    ),
    "analysis-regression.svg": wrap_svg(
        f'  <line x1="4" y1="19" x2="20" y2="5.5" stroke="{C_TEAL}" stroke-width="2.1" stroke-linecap="round"/>\n'
        f'  <circle cx="6" cy="15.5" r="1.7" fill="{C_BLUE}"/>\n'
        f'  <circle cx="10" cy="11.2" r="1.7" fill="{C_AMBER}"/>\n'
        f'  <circle cx="14.2" cy="12.5" r="1.7" fill="{C_BLUE}"/>\n'
        f'  <circle cx="17.8" cy="6.8" r="1.7" fill="{C_AMBER}"/>\n'
        f'  <path d="M3.5 4.5v15.5h17" stroke="{C_SLATE}" stroke-width="1.5" stroke-linecap="round"/>',
        size=32
    ),
    "analysis-frequencies.svg": wrap_svg(
        f'  <rect x="3.5" y="3.5" width="7.5" height="7.5" rx="1.2" fill="#DBEAFE" stroke="{C_BLUE}" stroke-width="1.6"/>\n'
        f'  <rect x="13" y="3.5" width="7.5" height="7.5" rx="1.2" fill="#FEF3C7" stroke="{C_AMBER}" stroke-width="1.6"/>\n'
        f'  <rect x="3.5" y="13" width="7.5" height="7.5" rx="1.2" fill="#CCFBF1" stroke="{C_TEAL}" stroke-width="1.6"/>\n'
        f'  <rect x="13" y="13" width="7.5" height="7.5" rx="1.2" fill="#DBEAFE" stroke="{C_BLUE}" stroke-width="1.6"/>',
        size=32
    ),
    "analysis-factor.svg": wrap_svg(
        f'  <circle cx="12" cy="6" r="3" fill="#DBEAFE" stroke="{C_BLUE}" stroke-width="1.8"/>\n'
        f'  <rect x="3" y="16" width="4.4" height="4" rx="0.8" stroke="{C_SLATE}" stroke-width="1.5" fill="#FFFFFF"/>\n'
        f'  <rect x="9.8" y="16" width="4.4" height="4" rx="0.8" stroke="{C_SLATE}" stroke-width="1.5" fill="#FFFFFF"/>\n'
        f'  <rect x="16.6" y="16" width="4.4" height="4" rx="0.8" stroke="{C_SLATE}" stroke-width="1.5" fill="#FFFFFF"/>\n'
        f'  <path d="M10.5 8.8L5.2 15.8M12 9v6.8M13.5 8.8l5.3 7" stroke="{C_TEAL}" stroke-width="1.5" stroke-linecap="round"/>',
        size=32
    ),
    "analysis-sem.svg": wrap_svg(
        f'  <ellipse cx="7.2" cy="7.5" rx="4.2" ry="2.7" fill="#CCFBF1" stroke="{C_TEAL}" stroke-width="1.7"/>\n'
        f'  <ellipse cx="16.8" cy="7.5" rx="4.2" ry="2.7" fill="#DBEAFE" stroke="{C_BLUE}" stroke-width="1.7"/>\n'
        f'  <path d="M11.6 7.5h0.8" stroke="{C_AMBER}" stroke-width="2" stroke-linecap="round"/>\n'
        f'  <rect x="3.5" y="15.5" width="4.2" height="3.8" rx="0.7" stroke="{C_SLATE}" stroke-width="1.4" fill="#FFFFFF"/>\n'
        f'  <rect x="9.9" y="15.5" width="4.2" height="3.8" rx="0.7" stroke="{C_SLATE}" stroke-width="1.4" fill="#FFFFFF"/>\n'
        f'  <rect x="16.3" y="15.5" width="4.2" height="3.8" rx="0.7" stroke="{C_SLATE}" stroke-width="1.4" fill="#FFFFFF"/>\n'
        f'  <path d="M6.5 10.3L5.6 15.3M8.2 10.3L11.5 15.3M16.8 10.3L18.4 15.3" stroke="{C_MUTED}" stroke-width="1.3"/>',
        size=32
    ),
    "analysis-medmod.svg": wrap_svg(
        f'  <rect x="10" y="3.5" width="4.5" height="4" rx="1" fill="#FEF3C7" stroke="{C_AMBER}" stroke-width="1.6"/>\n'
        f'  <rect x="3" y="15.5" width="4.5" height="4" rx="1" fill="#DBEAFE" stroke="{C_BLUE}" stroke-width="1.6"/>\n'
        f'  <rect x="16.5" y="15.5" width="4.5" height="4" rx="1" fill="#CCFBF1" stroke="{C_TEAL}" stroke-width="1.6"/>\n'
        f'  <path d="M6.2 15L10.8 8M13.8 8L18 15M8 17.5h8" stroke="{C_SLATE}" stroke-width="1.5" stroke-linecap="round"/>',
        size=32
    ),
    "analysis-linearmodels.svg": wrap_svg(
        f'  <path d="M4 17L19.5 9.5" stroke="{C_BLUE}" stroke-width="1.9" stroke-linecap="round"/>\n'
        f'  <path d="M4 13L19.5 5.5" stroke="{C_TEAL}" stroke-width="1.9" stroke-linecap="round"/>\n'
        f'  <path d="M4 20L19.5 13.5" stroke="{C_AMBER}" stroke-width="1.9" stroke-linecap="round"/>\n'
        f'  <path d="M3.5 4v16.5h17" stroke="{C_SLATE}" stroke-width="1.5" stroke-linecap="round"/>',
        size=32
    ),
    "analysis-survival.svg": wrap_svg(
        f'  <path d="M4 6h5v4h5v4.5h6" stroke="{C_BLUE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>\n'
        f'  <path d="M4 8.5h3.5v4.5h4.5v5h7" stroke="{C_CORAL}" stroke-width="1.7" stroke-dasharray="2 1.5" stroke-linejoin="round"/>\n'
        f'  <path d="M3.5 4v16.5h17" stroke="{C_SLATE}" stroke-width="1.5" stroke-linecap="round"/>',
        size=32
    ),

    # 3. Scatter / Plot Icons
    "analysis-scatr-jmvscatter.svg": wrap_svg(
        f'  <path d="M3.5 4v16.5h17" stroke="{C_SLATE}" stroke-width="1.5"/>\n'
        f'  <circle cx="7.5" cy="15" r="2" fill="{C_BLUE}"/>\n'
        f'  <circle cx="11.5" cy="11" r="2" fill="{C_TEAL}"/>\n'
        f'  <circle cx="15.5" cy="13" r="2" fill="{C_AMBER}"/>\n'
        f'  <circle cx="17.5" cy="7.5" r="2" fill="{C_BLUE}"/>',
        size=32
    ),
    "analysis-scatr-jmvbar.svg": wrap_svg(
        f'  <rect x="4.5" y="6" width="3.5" height="14" rx="0.8" fill="{C_BLUE}"/>\n'
        f'  <rect x="10" y="10.5" width="3.5" height="9.5" rx="0.8" fill="{C_TEAL}"/>\n'
        f'  <rect x="15.5" y="14.5" width="3.5" height="5.5" rx="0.8" fill="{C_AMBER}"/>\n'
        f'  <path d="M6.2 6 L11.7 4.8 L17.2 4.2" stroke="{C_CORAL}" stroke-width="1.7" stroke-linecap="round"/>\n'
        f'  <line x1="3" y1="20.2" x2="21" y2="20.2" stroke="{C_SLATE}" stroke-width="1.5"/>',
        size=32
    ),
    "analysis-scatr-jmvbox.svg": wrap_svg(
        f'  <rect x="6" y="8.5" width="4.5" height="7" fill="#DBEAFE" stroke="{C_BLUE}" stroke-width="1.6"/>\n'
        f'  <line x1="8.25" y1="5" x2="8.25" y2="8.5" stroke="{C_BLUE}" stroke-width="1.6"/>\n'
        f'  <line x1="8.25" y1="15.5" x2="8.25" y2="19" stroke="{C_BLUE}" stroke-width="1.6"/>\n'
        f'  <rect x="13.5" y="7" width="4.5" height="7.5" fill="#CCFBF1" stroke="{C_TEAL}" stroke-width="1.6"/>\n'
        f'  <line x1="15.75" y1="4" x2="15.75" y2="7" stroke="{C_TEAL}" stroke-width="1.6"/>\n'
        f'  <line x1="15.75" y1="14.5" x2="15.75" y2="18.5" stroke="{C_TEAL}" stroke-width="1.6"/>',
        size=32
    ),
    "analysis-scatr-jmvhist.svg": wrap_svg(
        f'  <rect x="4" y="13" width="3.8" height="7" fill="#DBEAFE" stroke="{C_BLUE}" stroke-width="1.4"/>\n'
        f'  <rect x="7.8" y="7" width="4" height="13" fill="#93C5FD" stroke="{C_BLUE}" stroke-width="1.4"/>\n'
        f'  <rect x="11.8" y="9.5" width="4" height="10.5" fill="#CCFBF1" stroke="{C_TEAL}" stroke-width="1.4"/>\n'
        f'  <rect x="15.8" y="15" width="3.8" height="5" fill="#DBEAFE" stroke="{C_BLUE}" stroke-width="1.4"/>',
        size=32
    ),
    "analysis-r.svg": wrap_svg(
        f'  <rect x="3" y="4" width="18" height="16" rx="3" fill="#0F172A" stroke="{C_TEAL}" stroke-width="1.6"/>\n'
        '  <path d="M7 9l3.2 3L7 15M12.2 15h4.8" stroke="#38BDF8" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/>',
        size=32
    ),

    # 4. Variable Measurement Scale Icons (100% Original Geometric Badges)
    "variable-continuous.svg": wrap_svg(
        f'  <rect x="3" y="6.5" width="18" height="11" rx="2" fill="#FEF3C7" stroke="{C_AMBER}" stroke-width="1.7"/>\n'
        f'  <path d="M7 6.5v3.5M11 6.5v5M15 6.5v3.5M18 6.5v2.5" stroke="{C_AMBER}" stroke-width="1.5" stroke-linecap="round"/>',
        size=18
    ),
    "variable-ordinal.svg": wrap_svg(
        f'  <rect x="3.5" y="13" width="4" height="7" rx="0.8" fill="{C_TEAL}"/>\n'
        f'  <rect x="10" y="9" width="4" height="11" rx="0.8" fill="{C_BLUE}"/>\n'
        f'  <rect x="16.5" y="5" width="4" height="15" rx="0.8" fill="{C_SLATE}"/>',
        size=18
    ),
    "variable-nominal.svg": wrap_svg(
        f'  <circle cx="8" cy="9" r="4.2" fill="#93C5FD" stroke="{C_BLUE}" stroke-width="1.5"/>\n'
        f'  <circle cx="15.5" cy="9.5" r="4.2" fill="#FDE68A" stroke="{C_AMBER}" stroke-width="1.5"/>\n'
        f'  <circle cx="11.8" cy="15.5" r="4.2" fill="#99F6E4" stroke="{C_TEAL}" stroke-width="1.5"/>',
        size=18
    ),
    "variable-nominal-text.svg": wrap_svg(
        f'  <circle cx="9" cy="12" r="6" fill="#DBEAFE" stroke="{C_BLUE}" stroke-width="1.5"/>\n'
        f'  <path d="M14 8h6M17 8v8" stroke="{C_SLATE}" stroke-width="1.8" stroke-linecap="round"/>',
        size=18
    ),
    "variable-id.svg": wrap_svg(
        f'  <rect x="2.5" y="5" width="19" height="14" rx="2.5" fill="#F1F5F9" stroke="{C_MUTED}" stroke-width="1.6"/>\n'
        f'  <path d="M7.5 9v6M11.5 9h2.2a2.8 2.8 0 0 1 0 6h-2.2V9z" stroke="{C_SLATE}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>',
        size=18
    ),
    "variable-computed.svg": wrap_svg(
        f'  <circle cx="12" cy="12" r="9" fill="#CCFBF1" stroke="{C_TEAL}" stroke-width="1.7"/>\n'
        f'  <path d="M14 7.5c-1.8 0-2.4 1.2-2.7 3.8l-.6 4c-.3 1.5-.9 2.2-2.2 2.2M8.8 11.2h5.2" stroke="{C_SLATE}" stroke-width="1.7" stroke-linecap="round"/>',
        size=20
    ),

    # 5. Data / Menu / Action / Studio Icons
    "menu-compute.svg": wrap_svg(
        f'  <rect x="4" y="3" width="16" height="18" rx="2.2" fill="#F8FAFC" stroke="{C_SLATE}" stroke-width="1.6"/>\n'
        f'  <rect x="7" y="6" width="10" height="3.5" rx="0.8" fill="#CCFBF1" stroke="{C_TEAL}" stroke-width="1.2"/>\n'
        f'  <circle cx="8.5" cy="13" r="1.1" fill="{C_SLATE}"/><circle cx="12" cy="13" r="1.1" fill="{C_SLATE}"/><circle cx="15.5" cy="13" r="1.1" fill="{C_BLUE}"/>\n'
        f'  <circle cx="8.5" cy="17" r="1.1" fill="{C_SLATE}"/><circle cx="12" cy="17" r="1.1" fill="{C_SLATE}"/><circle cx="15.5" cy="17" r="1.1" fill="{C_AMBER}"/>',
        size=28
    ),
    "menu-data-transform.svg": wrap_svg(
        f'  <path d="M5 8h11m0 0l-3-3m3 3l-3 3M19 16H8m0 0l3-3m-3 3l3 3" stroke="{C_TEAL}" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/>',
        size=28
    ),
    "menu-data-filter.svg": wrap_svg(
        f'  <path d="M4 5h16l-6.2 7.2v5.3l-3.6 1.8v-7.1L4 5z" fill="#DBEAFE" stroke="{C_BLUE}" stroke-width="1.6" stroke-linejoin="round"/>',
        size=28
    ),
    "menu-row-add.svg": wrap_svg(
        f'  <rect x="3" y="5" width="13" height="14" rx="1.5" stroke="{C_SLATE}" stroke-width="1.5"/>\n'
        f'  <line x1="3" y1="10" x2="16" y2="10" stroke="{C_SLATE}" stroke-width="1.4"/>\n'
        f'  <path d="M19 13v6M16 16h6" stroke="{C_TEAL}" stroke-width="2" stroke-linecap="round"/>',
        size=28
    ),
    "menu-row-remove.svg": wrap_svg(
        f'  <rect x="3" y="5" width="13" height="14" rx="1.5" stroke="{C_SLATE}" stroke-width="1.5"/>\n'
        f'  <line x1="3" y1="10" x2="16" y2="10" stroke="{C_SLATE}" stroke-width="1.4"/>\n'
        f'  <path d="M16 16h6" stroke="{C_CORAL}" stroke-width="2.1" stroke-linecap="round"/>',
        size=28
    ),
    "menu-variable-add.svg": wrap_svg(
        f'  <rect x="4" y="4" width="11" height="16" rx="1.5" stroke="{C_SLATE}" stroke-width="1.5"/>\n'
        f'  <line x1="9.5" y1="4" x2="9.5" y2="20" stroke="{C_SLATE}" stroke-width="1.4"/>\n'
        f'  <path d="M19 9v6M16 12h6" stroke="{C_BLUE}" stroke-width="2" stroke-linecap="round"/>',
        size=28
    ),
    "menu-variable-remove.svg": wrap_svg(
        f'  <rect x="4" y="4" width="11" height="16" rx="1.5" stroke="{C_SLATE}" stroke-width="1.5"/>\n'
        f'  <line x1="9.5" y1="4" x2="9.5" y2="20" stroke="{C_SLATE}" stroke-width="1.4"/>\n'
        f'  <path d="M16 12h6" stroke="{C_CORAL}" stroke-width="2.1" stroke-linecap="round"/>',
        size=28
    ),
    "menu-variable-edit.svg": wrap_svg(
        f'  <path d="M15.5 5.5l3 3L8.5 18.5H5.5v-3L15.5 5.5z" fill="#FEF3C7" stroke="{C_AMBER}" stroke-width="1.6" stroke-linejoin="round"/>',
        size=28
    ),
    "action-save.svg": wrap_svg(
        f'  <path d="M5 4h11l4 4v11a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 19V5.5A1.5 1.5 0 0 1 5.5 4z" fill="#DBEAFE" stroke="{C_BLUE}" stroke-width="1.6"/>\n'
        f'  <rect x="8" y="13" width="8" height="7" fill="#FFFFFF" stroke="{C_BLUE}" stroke-width="1.4"/>\n'
        f'  <rect x="8" y="4" width="6" height="4.5" fill="#FFFFFF" stroke="{C_BLUE}" stroke-width="1.4"/>',
        size=28
    ),
    "action-code.svg": wrap_svg(
        f'  <path d="M8.5 8L4 12l4.5 4M15.5 8L20 12l-4.5 4M13.5 6l-3 12" stroke="{C_TEAL}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
        size=24
    ),
    "action-palette.svg": wrap_svg(
        f'  <path d="M12 3a9 9 0 0 0 0 18c1.5 0 2.5-1 2.5-2.2 0-.6-.2-1.1-.6-1.5-.4-.4-.5-1-.5-1.5 0-1.2 1-2.2 2.3-2.2H17a4 4 0 0 0 4-4c0-3.8-4-6.6-9-6.6z" fill="#FEF3C7" stroke="{C_SLATE}" stroke-width="1.6"/>\n'
        f'  <circle cx="8" cy="10" r="1.3" fill="{C_BLUE}"/>\n'
        f'  <circle cx="11.5" cy="7.5" r="1.3" fill="{C_TEAL}"/>\n'
        f'  <circle cx="15.5" cy="9.5" r="1.3" fill="{C_CORAL}"/>\n'
        f'  <circle cx="7.5" cy="14" r="1.3" fill="{C_VIOLET}"/>',
        size=24
    ),
    "action-export.svg": wrap_svg(
        f'  <path d="M12 4v11m0-11L8 8m4-4l4 4M4.5 15.5v3A1.5 1.5 0 0 0 6 20h12a1.5 1.5 0 0 0 1.5-1.5v-3" stroke="{C_BLUE}" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/>',
        size=24
    ),
    "action-back.svg": wrap_svg(
        f'  <path d="M19 12H5m0 0l6-6m-6 6l6 6" stroke="{C_SLATE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
        size=20
    ),
    "action-close.svg": wrap_svg(
        f'  <path d="M6 6l12 12M18 6L6 18" stroke="{C_SLATE}" stroke-width="2" stroke-linecap="round"/>',
        size=18
    ),
    "action-check.svg": wrap_svg(
        f'  <path d="M5 12.5l4.5 4.5L19 7.5" stroke="{C_TEAL}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
        size=18
    ),
    "variable-list-arrow-right.svg": wrap_svg(
        f'  <path d="M5 12h13m0 0l-5-5m5 5l-5 5" stroke="{C_SLATE}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
        size=14
    ),
    "variable-list-arrow.svg": wrap_svg(
        f'  <path d="M19 12H6m0 0l5-5m-5 5l5 5" stroke="{C_SLATE}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
        size=14
    ),
    "menu-button-arrow.svg": wrap_svg(
        f'  <path d="M6 9l6 6 6-6" stroke="{C_SLATE}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
        size=12
    ),
    "splitscreen.svg": wrap_svg(
        '  <rect x="3" y="4" width="18" height="16" rx="2" stroke="#FFFFFF" stroke-width="1.8"/>\n'
        '  <line x1="12" y1="4" x2="12" y2="20" stroke="#FFFFFF" stroke-width="1.8"/>',
        size=18
    ),
    "fullscreen-hide-data.svg": wrap_svg(
        '  <rect x="3" y="4" width="18" height="16" rx="2" stroke="#FFFFFF" stroke-width="1.8"/>\n'
        '  <rect x="9" y="4" width="12" height="16" rx="1" fill="#FFFFFF" fill-opacity="0.35"/>',
        size=18
    ),
    "fullscreen-hide-results.svg": wrap_svg(
        '  <rect x="3" y="4" width="18" height="16" rx="2" stroke="#FFFFFF" stroke-width="1.8"/>\n'
        '  <rect x="3" y="4" width="12" height="16" rx="1" fill="#FFFFFF" fill-opacity="0.35"/>',
        size=18
    ),
    "menu-merge.svg": wrap_svg(
        f'  <rect x="3" y="4" width="8" height="16" rx="1.5" stroke="{C_SLATE}" stroke-width="1.6" fill="#E0F2FE"/>\n'
        f'  <rect x="13" y="4" width="8" height="16" rx="1.5" stroke="{C_TEAL}" stroke-width="1.6" fill="#CCFBF1"/>\n'
        f'  <path d="M9 12h6m-2-3l3 3-3 3" stroke="{C_AMBER}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
        size=24
    ),
    "menu-reshape.svg": wrap_svg(
        f'  <rect x="3" y="4" width="18" height="6" rx="1.2" stroke="{C_SLATE}" stroke-width="1.5" fill="#E0F2FE"/>\n'
        f'  <path d="M12 11v5m-3-2l3 3 3-3" stroke="{C_BLUE}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>\n'
        f'  <rect x="5" y="17" width="6" height="4" rx="1" fill="{C_TEAL}"/>\n'
        f'  <rect x="13" y="17" width="6" height="4" rx="1" fill="{C_AMBER}"/>',
        size=24
    ),
    "menu-zscore.svg": wrap_svg(
        f'  <path d="M3 19c3-1 6-13 9-13s6 12 9 13" stroke="{C_SLATE}" stroke-width="1.8" fill="none"/>\n'
        f'  <line x1="12" y1="5" x2="12" y2="19" stroke="{C_CORAL}" stroke-width="1.5" stroke-dasharray="2 2"/>\n'
        f'  <line x1="3" y1="19" x2="21" y2="19" stroke="{C_MUTED}" stroke-width="1.5"/>',
        size=24
    ),
    "menu-guide.svg": wrap_svg(
        f'  <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" stroke="{C_SLATE}" stroke-width="1.8" stroke-linecap="round"/>\n'
        f'  <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" stroke="{C_BLUE}" stroke-width="1.8" fill="#F8FAFC"/>\n'
        f'  <line x1="8" y1="7" x2="16" y2="7" stroke="{C_TEAL}" stroke-width="1.6" stroke-linecap="round"/>\n'
        f'  <line x1="8" y1="11" x2="14" y2="11" stroke="{C_MUTED}" stroke-width="1.5" stroke-linecap="round"/>',
        size=24
    ),
    "lang-globe.svg": wrap_svg(
        f'  <circle cx="12" cy="12" r="9" stroke="{C_SLATE}" stroke-width="1.7"/>\n'
        f'  <ellipse cx="12" cy="12" rx="4.5" ry="9" stroke="{C_TEAL}" stroke-width="1.5"/>\n'
        f'  <line x1="3" y1="12" x2="21" y2="12" stroke="{C_SLATE}" stroke-width="1.5"/>',
        size=20
    ),
}


def main():
    for fname, content in ICONS.items():
        with open(os.path.join(ASSETS_DIR, fname), "w", encoding="utf-8") as f:
            f.write(content)
    print(f"Generated {len(ICONS)} clean-room MIT/CC0 SVG icons in {ASSETS_DIR}")


if __name__ == "__main__":
    main()
