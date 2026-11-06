# Open-Source Attribution & License Notice (`NOTICE.md`)

## 1. Reference to The jamovi Project

**OpenStat Studio** (`openstat` / `pyjamovi`) is an open-source, Python-based statistical computing suite released under the **MIT License**.

The statistical menu organization, the layout of the results tables, and the `.omv` file interoperability of OpenStat Studio were designed with reference to [The jamovi project](https://www.jamovi.org).

> *Reference:* The jamovi project (2026). *jamovi* [Computer Software]. Retrieved from https://www.jamovi.org

---

## 2. Trademarks and Non-Affiliation

- Product names mentioned in this repository (for example SPSS, Stata, SAS, Mplus, jamovi, JASP, Hangul/Hancom Office, Microsoft Word/Excel and Claude) are trademarks of their respective owners.
- OpenStat Studio is an independent project developed by an individual. It is not affiliated with, sponsored by or endorsed by any of these owners, and product names are used only to describe compatibility and features.
- The OpenStat Studio brand mark and the SVG icons in `reconstructed/pyjamovi/web/assets/` were created for this project (`scripts/generate_clean_icons.py`) and are released under the MIT License.
- See the disclaimer section of [`README.md`](README.md#disclaimer) for warranty and liability terms.

---

## 3. Third-Party Python Dependencies

OpenStat Studio does not bundle or vendor any third-party package source. All dependencies listed in `requirements.txt` / `setup.py` are installed separately from PyPI by `pip`, and each remains under its own license.

### 3.1 Permissive-licensed dependencies (BSD / MIT / NCSA / PSF)

- `numpy`, `scipy`, `pandas`, `statsmodels`, `scikit-learn`, `seaborn` (BSD-3-Clause)
- `semopy` (MIT) — Reference implementation for the SEM test suite. SEM/CFA estimation (ML, FIML, MLR, multiple groups, indirect effects, modification indices) and the mixture models (LCA, LPA, LTA, LCGA) are implemented in OpenStat itself with defaults that follow Mplus; Mplus is not used or required
- `linearmodels` (NCSA / Permissive) — Panel Econometrics (FE / Two-Way FE / RE / Pooled OLS / Hausman)
- `lifelines` (MIT) — Kaplan-Meier & Cox Proportional Hazards Survival Analysis
- `matplotlib` (PSF/BSD-compatible) — Publication Vector/Raster Graphics (`PlotStyle`)
- `openpyxl` (MIT), `python-docx` (MIT) — Multi-format APA Table Export (`.xlsx`, `.docx`, plus built-in `.hwpx` OWPML & `.tex` exporters)
- `fastapi`, `uvicorn`, `pydantic` (MIT) — Optional on-demand local Web Studio & REST API
- `kiwipiepy`, `kiwipiepy_model` (Apache-2.0) — Korean morphological analysis for text mining
- `wordcloud` (MIT) — Word clouds
- `networkx` (BSD-3-Clause) — Keyword co-occurrence & relational networks, centrality, Louvain communities
- `umap-learn` (BSD-3-Clause) with `pynndescent` (BSD-2-Clause), `numba` (BSD-2-Clause), `llvmlite` (BSD-2-Clause / Apache-2.0 with LLVM exception) — Dimension reduction for document clustering
- `pypdf` (BSD-3-Clause) — Text extraction from PDF files
- `pyreadstat` (Apache-2.0; wheels bundle the ReadStat C library, MIT) — Stata .dta and SPSS .sav/.por import and export with value and variable labels
- `geopandas` (BSD-3-Clause), `pyogrio` (MIT; wheels bundle GDAL, MIT/X), `shapely` (BSD-3-Clause; wheels bundle the GEOS library, LGPL-2.1, dynamically linked and unmodified), `pyproj` (MIT; wheels bundle PROJ, MIT) — Boundary files, geometry and coordinate reference systems
- `libpysal`, `mapclassify`, `mgwr`, `spglm`, `spreg` (BSD-3-Clause, PySAL project) — Contiguity weights, map class breaks, geographically weighted regression. The spatial econometric estimators (SAR/SEM/SDM/SDEM/SAC, spatial panels, effects, LM tests) are implemented in OpenStat itself; `spreg` (installed with `mgwr`) serves as an independent reference implementation in the test suite
- Optional (`requirements-embedding.txt`, not installed by default): `sentence-transformers` (Apache-2.0), `torch` (BSD-3-Clause); embedding models are downloaded from Hugging Face under their own model licenses (e.g. `BAAI/bge-m3`: MIT)

### 3.2 GPL-licensed dependencies (copyleft)

The following packages are imported at runtime by specific analysis modules and are licensed under the GNU General Public License:

| Package | License | Imported by |
|---|---|---|
| `pingouin` | GPL-3.0 | `analyses/anova_advanced.py`, `analyses/bayesian.py`, `analyses/factor_suite.py`, `analyses/posthoc.py`, `analyses/regression_advanced.py` |
| `factor_analyzer` | GPL-2.0 | `analyses/factor_suite.py`, `diagnostics.py` |

- The MIT License in `LICENSE` covers the OpenStat Studio source code in this repository only. It does not relicense `pingouin` or `factor_analyzer`.
- Anyone who redistributes OpenStat Studio together with these packages (for example, a bundled installer, frozen executable, or archive that includes a populated `.venv`) must comply with the terms of the GPL for those packages.
- License identifiers above were taken from the installed package metadata (`pingouin` 0.7.0, `factor_analyzer` 0.5.1). Check the upstream projects for the license of the version you install.
