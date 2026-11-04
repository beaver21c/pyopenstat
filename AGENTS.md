# OpenStat Studio v3.0 (`openstat` / `pyjamovi`) — 워크스페이스 운영 및 아키텍처 규율

## 1. 개요 및 오픈소스 라이선스 원칙
- 본 워크스페이스(저장소 루트)는 **독자 오픈소스(MIT License) 통계 패키지 및 Claude Code 실시간 제어 런타임인 `OpenStat Studio v3.0`**의 표준 개발·검증 환경이다.
- **참조 표기 및 저작권 클린룸 원칙**:
  - 통계 분석 모형 구성, 메뉴 체계, APA 7판 결과 시트 구조 및 `.omv` 바이너리 컨테이너 호환성은 **[The jamovi project](https://www.jamovi.org)를 참고하여 설계**되었음을 `LICENSE`, `NOTICE.md`, `README.md`, 웹 UI 하단에 명시한다.
  - jamovi의 `V` 로고(`logo-v.svg`) 및 GPL/AGPL 원천 파일·아이콘은 **일절 사용하지 않으며**, 자체 제작 브랜드 로고(`brand-logo.svg`, `brand-logo-mark.svg`)와 `scripts/generate_clean_icons.py`로 생성한 45종의 독자 MIT/CC0 벡터 아이콘 및 100% 자체 Python 코드를 사용한다.
- **온디맨드 실행 원칙**:
  - 웹 서버(`run_web_server.py`)를 상시 백그라운드 데몬으로 띄워 두지 않으며, 사용자가 브라우저 UI를 원할 때만 `START_OPENSTAT.sh` / `START_OPENSTAT.bat`으로 온디맨드 실행한다.

## 2. 핵심 아키텍처 구성 (`reconstructed/openstat` & `reconstructed/pyjamovi`)
- **28종 순수 파이썬 통계 엔진 (`analyses/`)**:
  - 기술통계, T-검정(독립/대응/단일표본), 분산분석(일원/다원/공분산/반복측정/비모수 Kruskal-Wallis) 및 5종 사후검정(Tukey, Games-Howell, Scheffé, Bonferroni, Holm)
  - 회귀분석(위계적 OLS, 이항 로지스틱 ROC/AUC), 상관/편상관, 교차분석($\chi^2$/Fisher/Cramér's $V$)
  - 신뢰도($\alpha$, $\omega$), 주성분/탐색적 요인분석(PCA/EFA), **확인적 요인분석 및 구조방정식(`sem_module.py`, `lavaan` 문법)**, 매개/조절분석(`medmod_module.py`)
  - **패널 계량경제 분석(`panel_module.py`: 개체/이원 고정효과 FE, 확률효과 RE, 합동 OLS, Hausman 검정)**, 다층모형(`MixedLM`), 생존분석(`Kaplan-Meier`, `Cox PH`), 베이지안 통계(`BF10`)
- **실시간 `.py` 코드 작성 및 Claude Code 직접 제어 (`syntax_engine.py`, `mcp_server.py`, `cli.py`)**:
  - UI 조작과 `workspace/live_session.py` 양방향 동기화, `.py` 직접 실행, `.mcp.json` 기반 `openstat-mcp` 서버 및 `openstat` CLI 제공.
- **10종 분석표 내보내기 (`exporters.py`)**:
  - `.xlsx`(Excel), `.docx`(Word), `.hwpx`(한글 OWPML 표), `.tex`(LaTeX `booktabs`), `.csv`, `.tsv`, `.md`, `.html`, `.json`, `.omv`.
- **자유로운 그래프 디자인 조작 (`analyses/plot_style.py`)**:
  - 10종 학술 팔레트 + 커스텀 Hex 색상, 5종 테마, 한글 폰트(`KoPubDotum Bold`), 제목·축 라벨·해상도(DPI)·크기 조정 및 벡터 `.svg`/`.pdf`/`.png` 저장.
- **패키지 최신성 및 문법/API 변화 자가 점검 (`diagnostics.py` — `openstat doctor`)**:
  - 패키지 실행 시 의존성 버전(최소 요구치 및 실시간 PyPI 최신 버전)을 점검하고, 상위 라이브러리 문법/API 변경으로 인한 오류 여부를 6개 서브시스템 프로브 및 런타임 호환성 심(`apply_runtime_compat_shims`)으로 자가 진단한다.
- **jamovi 신규 기능 반영 및 재빌드 (`upstream_sync.py` — `openstat sync-jamovi` & `openstat rebuild`)**:
  - 비정기적으로 jamovi에 추가된 신규 분석 모듈을 탐지하여 `reconstructed/pyjamovi/plugins/`에 클린룸 MIT 플러그인으로 스캐폴딩하고, 아이콘·카탈로그·휠·자가점검·테스트를 단일 명령으로 재빌드한다.
