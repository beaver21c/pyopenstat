# OpenStat Studio v3.0 (`openstat` / `pyjamovi`)

**Pure-Python Statistical Computing Suite · 15-Type Standalone Graph Builder · Live `.py` Syntax Studio · Claude Code Runtime · 11-Format Analysis Results & Graph Exporter**

[![License: MIT](https://img.shields.io/badge/License-MIT-0D9488.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-1E3A5F.svg)](setup.py)
[![Claude Code MCP](https://img.shields.io/badge/Claude_Code-MCP_%26_CLI_Ready-D97706.svg)](scripts/setup_claude_code.py)
[![1-Click Launcher](https://img.shields.io/badge/1--Click_Launch-Windows_%26_macOS_%26_Linux-2563EB.svg)](START_OPENSTAT.bat)

[한국어 안내서 (README_KR.md)](README_KR.md) | [Claude Code 가이드 (CLAUDE.md)](CLAUDE.md) | [출처 및 라이선스 고지 (NOTICE.md)](NOTICE.md)

---

## ⚡ 빠른 시작 (3가지 사용 방식)

OpenStat Studio는 일반 사용자, AI 에이전트 사용자, 파이썬 연구자 모두를 위해 완벽하게 최적화된 3가지 실행 방식을 제공합니다.

### 🌟 방식 1. 일반 사용자 (마우스 1클릭 무설치형 실행)
복잡한 터미널 명령어 입력 없이, 마우스 클릭만으로 자모비(jamovi) 스타일의 인터랙티브 GUI 프로그램을 엽니다.

1. GitHub 저장소 우측 상단의 **[Code] → [Download ZIP]** 을 클릭하여 다운로드한 뒤 원하는 폴더에 압축을 풉니다.
2. **실행 파일 더블클릭**:
   - **Windows**: `OPENSTAT_실행.bat` (또는 `START_OPENSTAT.bat`) 더블클릭!
   - **macOS / Linux**: 터미널에서 `./START_OPENSTAT.sh` 실행
3. **자동 환경 구성 및 앱 실행**:
   - 시스템에 Python이 설치되어 있으면, 최초 1회 필요한 가상환경(`.venv`)과 패키지를 자동으로 설치한 뒤 브라우저 독립 앱 창(`http://localhost:8500`)을 자동으로 띄웁니다.
   - 2회차 실행부터는 1초 만에 즉시 실행됩니다.
   - *종료 방법*: 검은색 콘솔 창을 닫거나 콘솔에서 아무 키나 누르면 서버 종료.
   - *공유 폴더 실행*: 네트워크 공유 폴더에서 실행하면 이 PC의 `%USERPROFILE%\OpenStatStudio`로 복사한 뒤 실행.

---

### 🤖 방식 2. AI 에이전트 사용자 (Claude Code로 말하듯이 분석하기)
터미널에서 **Claude Code**, Antigravity, Cursor 등 AI 에이전트를 열어 대화하듯 데이터를 분석하고 보고서를 생성합니다.

1. **원클릭 연동 등록**:
   - **Windows**: `SETUP_CLAUDE_CODE.bat` 더블클릭!
   - **macOS / Linux**: `./SETUP_CLAUDE_CODE.sh` 실행
   - *(또는 파이썬: `python scripts/setup_claude_code.py`)*
2. **Claude Code 실행 및 자연어 지시**:
   터미널에서 `claude`를 실행한 후 자연어로 입력하세요:
   > *"sample_data/ToothGrowth.csv 데이터 불러와서 dose별 t-검정하고 APA 표랑 박스플롯 그려줘"*  
   > *"패널 데이터 Grunfeld_Panel.csv로 고정효과 분석하고 결과 표를 HWPX, Excel로 만들어줘"*  
   > *"age가 65세 이상인 사람만 필터링해서 고혈압 유병률 교차분석해줘"*  
3. Claude Code가 OpenStat의 28종 통계 분석, 15종 그래프 빌더, 데이터 변환(Recode/Filter/Reshape/Compute), 한글(HWPX)/엑셀(XLSX)/워드(DOCX)/PDF 내보내기를 직접 실시간 실행합니다.

---

### 💻 방식 3. 파이썬 연구자 & 데이터 과학자 (`import openstat as ost`)
Jupyter Notebook, Google Colab 또는 파이썬 스크립트에서 Stata/R보다 직관적인 분석 코드를 작성합니다.

```python
import openstat as ost

# 1. 데이터 불러오기 (CSV, Excel, OMV, Parquet)
sess = ost.Session("sample_data/Grunfeld_Panel.csv")

# 2. 데이터 가공 (Stata do-file 동등 기능)
sess.compute("inv_cap_ratio", "invest / capital")
sess.filter("year >= 1940")

# 3. 고급 계량경제 패널 분석 (Entity Fixed Effects + Hausman Test)
res = sess.panel(
    entity="firm", time="year", dep="invest", covs=["value", "capital"],
    model_type="fixed", hausman=True,
    plotStyle={"palette": "kihasa", "theme": "apa_classic", "title": "패널 고정효과 모형 추세선"}
)

# 4. 11개 포맷 결과 표 및 고해상도 벡터 그래프 내보내기
res.export_results("reports/panel_report.hwpx")  # 한글 OWPML 네이티브 표 + 그림 삽입
res.export_results("reports/panel_report.xlsx")  # 엑셀 APA 3선 서식 표
res.save_plot("reports/panel_plot.svg")          # 출판용 벡터 SVG
sess.save_py("reports/analysis_script.py")       # 100% 재현 가능한 파이썬 스크립트
```

---

## 📋 사전 필수 요구사항 (Prerequisites)

- **Python 3.10 이상** (권장: Python 3.11, 3.12, 3.13, 3.14)
- **Windows 사용자 주의사항**:
  - Python이 설치되어 있지 않다면 [python.org/downloads](https://www.python.org/downloads/) 에서 다운로드하여 설치하세요.
  - **★ 설치 창 맨 아래의 `[Add python.exe to PATH]` 체크박스를 반드시 체크해야 합니다.**
  - `OPENSTAT_실행.bat` 실행 시 Python이 없으면 자동으로 다운로드 링크를 열어 안내합니다.
  - 공유 폴더(네트워크 드라이브)에서 실행 시 `%USERPROFILE%\OpenStatStudio`로 자동 복사·갱신 후 로컬 사본 실행 (위치 변경: 환경변수 `OPENSTAT_LOCAL_DIR`). PC2 등 원본 PC에서 바뀐 파일은 다음 실행 때 자동 반영.
  - 서버는 `127.0.0.1:8500`에만 연결되며, 서버 기록은 `workspace\server.log`에 저장.

---

## 🏛️ 오픈소스 라이선스 및 출처 고지 (Attribution)

**OpenStat Studio**는 **[MIT License](LICENSE)** 하에 완전히 자유롭게 배포되는 독자 오픈소스 프로젝트입니다.

- **개념적 참조 및 호환성 설계**: 분석 모형 체계, 통계 메뉴 구성, APA 7판 결과 시트 표현 양식 및 `.omv` 파일 호환 구조는 **[The jamovi project](https://www.jamovi.org)**를 참고하여 설계되었습니다. 상세 라이선스 및 저작권 정보는 [`NOTICE.md`](NOTICE.md)에 상세히 표기되어 있습니다.
- **100% 클린룸 코드 및 저작권 무결성**: GPL/AGPL 외부 코드가 일절 포함되지 않았으며, jamovi의 `V` 로고 및 원본 아이콘을 전혀 사용하지 않았습니다. 45개의 모든 벡터 아이콘(`reconstructed/pyjamovi/web/assets/`)과 파이썬 통계/시각화/내보내기 엔진은 상업적/비상업적 연구 및 배포에 안전한 순수 창작물입니다.
- **GPL 의존 패키지 고지**: 일부 분석 모듈은 실행 시 GPL 라이선스 패키지(`pingouin` GPL-3.0, `factor_analyzer` GPL-2.0)를 불러옴. 저장소 소스코드에는 이들 패키지가 포함되지 않으며, 이들 패키지를 함께 묶어 재배포하는 경우 해당 GPL 조건을 따라야 함. 상세 내용은 [`NOTICE.md`](NOTICE.md) 3.2절 참조.

---

## 🔬 주요 기능 상세 (v3.0)

### 1. 28+ 순수 파이썬 통계 분석 엔진 (R 언어 설치 불필요)
- **기술통계 및 탐색**: 요약통계, Violin/Raincloud 플롯, 산점도(LOESS 추세선 + 주변 밀도도/박스플롯), Pareto 차트, Q-Q 플롯.
- **T-검정 & 분산분석(ANOVA)**: 독립표본 / 대응표본 / 일표본 T-검정, Welch & Fisher 일원배치 분산분석, 요인분산분석(Type I/II/III SS, $\eta^2$, $\eta^2_p$, $\omega^2$), 공분산분석(ANCOVA), 반복측정 분산분석(Mauchly 구형성 검정 + Greenhouse-Geisser/Huynh-Feldt), Kruskal-Wallis, **5종 사후검정**(Tukey HSD, Games-Howell, Scheffé, Bonferroni, Holm).
- **회귀분석 & 범주형 데이터**: 상관행렬, 편상관, OLS 다중선형회귀(VIF 다중공선성, Durbin-Watson 잔차독립성, Cook's Distance, 2×2 진단도), 이항 로지스틱 회귀(Odds Ratio, ROC/AUC 곡선), 교차분할표($\chi^2$, Fisher 직접확률, Cramér's $V$).
- **심리측정 & 요인분석**: 신뢰도 분석(Cronbach's $\alpha$, McDonald's $\omega$), 주성분분석(PCA), 탐색적 요인분석(EFA - MinRes / ML / Principal Axis + Varimax / Promax / Oblimin 회전 + 평행분석 스크리 도표).
- **구조방정식 모형 (SEM / CFA)**: `lavaan` 문법 지원(`=~` 잠재요인 측정, `~` 구조 회귀, `~~` 공분산), 표준화 경로계수($\beta$), 적합도 지수($\chi^2$, CFI, TLI, RMSEA, SRMR, AIC, BIC) 및 잠재 경로 다이어그램 자동 렌더링.
- **계량경제 패널 분석 & 다층모형**: 개체 고정효과(Fixed Effects Within), 이원 고정효과(Two-Way FE), 확률효과(Random Effects GLS), 통합 OLS(Pooled OLS), 클러스터 로버스트 표준오차, **하우스만 사양검정(Hausman Test, $\chi^2$)**, 선형 혼합모형(Multilevel Linear Mixed Models - REML, ICC).
- **베이지안 통계 & 생존분석**: JZS 코시 베이지안 T-검정/ANOVA/상관분석 ($BF_{10}$), 부트스트랩 매개효과(직접/간접/총효과 95% 신뢰구간), 조절효과(Simple Slopes), Kaplan-Meier 생존곡선 및 Cox 비례위험 모형.

### 2. Stata `.do` 파일 수준의 데이터 전처리 GUI 리본 & 모달
- **변수 보기 (Variables View)**: 변수 척도(명목/서열/연속), 데이터 유형, 결측치 수치, 요약 통계(평균±표준편차, 범주 목록) 및 변수별 즉시 삭제/리코드 버튼.
- **데이터 처리 버튼**:
  - `[Compute]`: 파이썬 수식 기반 새 변수 생성 (`weight / (height/100)**2`)
  - `[Recode]`: 구간별 / 개별값 리코드 및 레이블 지정 (모달 대화상자)
  - `[Filter]`: 조건식 기반 행 필터링 (`age >= 20 and gender == 1`)
  - `[Merge Data]`: 두 데이터셋 간 Key 변수 기준 결합 (Inner, Left, Right, Outer)
  - `[Reshape]`: Wide 형 $\leftrightarrow$ Long 형 변환
  - `[Transform (Z)]`: 표준화 (Z-score, Min-Max, Robust)
  - `[Undo]`: 1단계 작업 취소 및 이전 상태 복원

### 3. 독립형 15종 학술 그래프 빌더 (`GraphBuilder`)
통계 검정과 완전히 독립적으로 논문용 차트를 그리고 결과 시트에 추가할 수 있습니다:
- `bar`, `stacked_bar`, `line`, `area`, `pie`, `donut`, `histogram`, `density`, `box`, `violin`, `raincloud`, `scatter` (버블 + OLS/LOESS 회귀선), `pairplot`, `heatmap`, `errorbar` (Forest CI 플롯).

### 3-1. 텍스트 분석 & 관계망 분석 (리본 [텍스트 분석]·[관계망 분석])
- **텍스트 파일 불러오기**: `.txt` `.md` `.docx` `.pdf` `.hwpx` 파일·폴더 → 문서 1건 = 1행 데이터 (문서ID·파일명·폴더·형식·글자수·본문). SDK: `ost.load_texts("폴더")`
- **키워드 분석** (`textKeywords`): kiwipiepy 형태소 분석, 사용자 사전·불용어·유의어 통합, 빈도·문서 빈도·TF-IDF 표, 집단별 특징 키워드(c-TF-IDF), 워드클라우드
- **키워드 동시출현 관계망** (`textNetwork`): 문장·문서 단위 동시출현, 빈도·Jaccard·NPMI 연결 강도, 중심성, Louvain 묶음, 관계망 그림
- **문서 군집 분석** (`textCluster`): TF-IDF+LSA(기본) 또는 문장 임베딩(`BAAI/bge-m3` 등, `requirements-embedding.txt` 선택 설치) → UMAP → HDBSCAN·k-평균·Ward, 군집별 대표 키워드(c-TF-IDF), 미분류 재배정, 상위 주제 묶음, 시드 3개 안정성(ARI), 군집 분포도·군집 관계망·문서 유사도 관계망
- **관계망 분석** (`graphNetwork`): 연결 목록(출발·도착·가중치) → 밀도·경로 길이·군집계수, 중심성 6종(연결·가중 연결정도·매개·근접·고유벡터·PageRank), Louvain 묶음, 관계망 그림
- 예제 데이터: `sample_data/Text_PolicyDocs.csv` (합성 정책문서 150건), `sample_data/Network_Collaboration.csv` (합성 기관 협력 86건)

### 4. 11개 포맷 결과 보고서 내보내기 (통계 표 + 고해상도 그림)
- **`.hwpx`**: 한글 OWPML 네이티브 편집 가능 표(`<hp:tbl>`) **+ 삽입된 PNG 그림(`<hp:pic>`)**
- **`.pdf`**: 다면 출판 품질의 APA 3선 표 및 고해상도 그래프 리포트
- **`.docx`**: 마이크로소프트 워드 네이티브 APA 표 + 차트
- **`.xlsx`**: 엑셀 전용 APA 테두리 서식 시트 + 차트 이미지 시트
- **`.tex`**: LaTeX `booktabs` 표 + `\includegraphics` 벡터 PDF 그림 세트
- **`.html`**, **`.md`**, **`.csv`**, **`.tsv`**, **`.json`**, **`.omv`**

---

## 🛠️ 자체 점검 및 유지보수 CLI (`openstat`)

```bash
# 1. 패키지 최신성 점검 및 upstream 문법 변화 자가 진단 (Doctor)
.venv/bin/openstat doctor        # Linux / macOS
.venv\Scripts\openstat doctor    # Windows

# 2. jamovi 최신 버전 기능 동기화 점검 및 빌드
.venv/bin/openstat sync-jamovi
.venv/bin/openstat rebuild

# 3. 전체 분석 모듈 및 색상 팔레트 목록
.venv/bin/openstat catalog
.venv/bin/openstat palettes
```

---

## 📁 프로젝트 폴더 구조

```
OpenStat-Studio/
├── OPENSTAT_실행.bat        # [Windows] 마우스 1클릭 자동 부트스트랩 실행기
├── START_OPENSTAT.bat       # [Windows] 로컬 서버 구동 및 앱 화면 런처
├── STOP_OPENSTAT.bat        # [Windows] 백그라운드 서버 안전 종료 스크립트
├── START_OPENSTAT.sh        # [macOS/Linux] 자동 부트스트랩 실행기
├── STOP_OPENSTAT.sh         # [macOS/Linux] 서버 종료 스크립트
├── SETUP_CLAUDE_CODE.bat    # [Windows] Claude Code AI 연동 원클릭 스크립트
├── SETUP_CLAUDE_CODE.sh     # [macOS/Linux] Claude Code AI 연동 스크립트
├── .mcp.json                # Claude Code MCP 설정 (SETUP_CLAUDE_CODE 실행 시 PC별 생성, Git 추적 제외)
├── CLAUDE.md                # Claude Code 에이전트 지침서
├── README.md                # 영문 & 국문 통합 안내서
├── README_KR.md             # 한국어 상세 매뉴얼
├── NOTICE.md                # 오픈소스 출처 및 라이선스 고지
├── LICENSE                  # MIT 라이선스 전문
├── requirements.txt         # 패키지 의존성 목록
├── setup.py                 # pip 설치 및 CLI 등록 스크립트
├── reconstructed/           # 100% 클린룸 파이썬 소스 코드
│   ├── pyjamovi/            # 통계 엔진, 데이터 전처리, 시각화, 웹 UI
│   │   ├── analyses/        # 28종 통계 분석 모듈
│   │   ├── data_ops.py      # Stata 수준 데이터 변환 엔진
│   │   ├── exporters/       # 11개 포맷 내보내기 엔진
│   │   └── web/             # 인터랙티브 웹 UI 및 45종 클린룸 SVG 아이콘
│   └── openstat/            # openstat 네임스페이스 및 CLI/MCP 엔트리포인트
├── sample_data/             # 패널, SEM, 분산분석 등 실습용 데이터셋
└── workspace/               # 실시간 파이썬 스크립트(live_session.py) 동기화 공간
```
