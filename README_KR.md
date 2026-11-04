# OpenStat Studio v3.0 한국어 사용자 가이드

**순수 파이썬 기반 통계 분석 소프트웨어 · 독립 15종 학술 차트 빌더 · 실시간 `.py` 코드 스튜디오 · Claude Code AI 런타임 · 11개 포맷 결과 보고서 내보내기**

본 프로그램은 jamovi의 직관적인 UI와 분석 체계를 참고하여, R 언어나 복잡한 환경 설정 없이 **순수 파이썬(Pure Python)**만으로 연구 논문 통계 분석과 시각화를 완결할 수 있도록 개발된 오픈소스(MIT 라이선스) 프로그램입니다.

---

## 🚀 1. 설치 및 실행 (3초 퀵스타트)

OpenStat Studio는 파이썬 초보자부터 데이터 사이언티스트까지 누구나 바로 쓸 수 있도록 설계되었습니다.

### 🌟 1단계: 다운로드
1. GitHub 페이지 오른쪽 위의 초록색 **[Code]** 버튼을 누르고 **[Download ZIP]** 을 클릭합니다.
2. 다운로드받은 압축 파일의 압축을 원하는 폴더에 풉니다.

### 🌟 2단계: 마우스 더블클릭 실행
- **Windows 사용자**: 폴더 안에 있는 **`OPENSTAT_실행.bat`** (또는 `START_OPENSTAT.bat`)을 마우스로 더블클릭합니다!
- **Mac / Linux 사용자**: 터미널을 열고 `./START_OPENSTAT.sh` 를 실행합니다.

> **최초 1회 실행 시 자동 환경 구성**:
> 컴퓨터에 Python 3.10 이상이 설치되어 있으면, 프로그램이 스스로 가상환경(`.venv`)을 만들고 필요한 통계 패키지들을 알아서 설치합니다 (약 1분 소요).  
> 설치가 끝나면 Microsoft Edge 또는 Chrome 브라우저를 독립 앱 창으로 띄워 오픈스탯 스튜디오가 실행됩니다.  
> 2회차 실행부터는 기다림 없이 1초 만에 바로 실행됩니다!
>
> **공유 폴더에서 실행하는 경우**: 다른 PC의 공유 폴더(네트워크 드라이브)에서 더블클릭하면 이 PC의 `%USERPROFILE%\OpenStatStudio`로 프로그램을 복사·갱신한 뒤 로컬 사본을 실행 (위치 변경: 환경변수 `OPENSTAT_LOCAL_DIR`).

### 🛑 프로그램 종료하기
- 검은색 콘솔 창을 닫거나, 콘솔 창에서 아무 키나 누르면 백그라운드 프로세스가 남지 않고 안전하게 종료됩니다.
- 필요할 경우 `STOP_OPENSTAT.bat` (Mac/Linux는 `./STOP_OPENSTAT.sh`)을 실행하여 강제 종료할 수도 있습니다.

---

## 🤖 2. Claude Code AI 에이전트로 분석하기

터미널에서 AI 에이전트(Claude Code, Antigravity, Cursor 등)에게 말로 지시하여 데이터를 분석할 수 있습니다.

### 🌟 1회성 연동 설정
폴더 안에 있는 **`SETUP_CLAUDE_CODE.bat`** (Mac/Linux는 `./SETUP_CLAUDE_CODE.sh`)을 더블클릭합니다.  
프로그램이 현재 컴퓨터 경로를 스스로 인식하여 Claude Code에 OpenStat 도구를 자동 등록합니다.

### 🌟 실전 대화 예시
프로젝트 폴더에서 터미널을 열고 `claude`를 실행한 후 다음과 같이 입력해 보세요:

1. **데이터 탐색 및 요약**:
   > *"sample_data/ToothGrowth.csv 데이터 열어서 변수 요약해 주고, dose별 len 기술통계량 구해줘"*
2. **가설 검정 및 분산분석**:
   > *"dose를 독립변수로, len을 종속변수로 일원배치 분산분석(ANOVA) 돌려줘. 등분산성이 깨졌을 때 쓸 Games-Howell 사후검정도 같이 넣고, 박스플롯도 그려줘"*
3. **데이터 가공 (Stata do-file 동등 기능)**:
   > *"환자 데이터에서 체중(weight)과 신장(height)으로 BMI 변수를 새로 계산하고, 연령(age)을 20대/30대/40대/50대이상으로 범주화(recode)해줘"*
4. **고급 패널 계량경제학 분석**:
   > *"Grunfeld_Panel.csv 패널 데이터로 고정효과(Fixed Effects) 모형 돌리고 하우스만 검정(Hausman test) 결과 해석해 줘"*
5. **보고서 파일 내보내기**:
   > *"지금까지 분석한 모든 표와 그래프를 논문 작성용 한글(HWPX) 파일과 엑셀(XLSX) 파일로 저장해줘"*

---

## 📊 3. 지원하는 통계 분석 모형 (28종)

| 분류 | 분석 모듈 명칭 | 상세 기능 및 특징 |
| :--- | :--- | :--- |
| **탐색 및 기술통계** | `descriptives` | 평균, 표준편차, 왜도/첨도, 사분위수, Shapiro-Wilk 정규성 검정 |
| | `scatr` | 산점도, 선형/LOESS 평활선, 주변 밀도곡선, 주변 박스플롯 |
| | `pareto`, `qqplot` | 품질관리 파레토 차트, Q-Q 정규확률도 |
| **T-검정 & 분산분석** | `ttestIS`, `ttestPS`, `ttestOneS` | 독립표본/대응표본/일표본 T-검정, Cohen's d 효과크기 |
| | `anovaOneW` | Fisher & Welch 일원분산분석, 등분산 검정(Levene) |
| | `anovaFactorial` | 다원 요인분산분석 (Type I/II/III SS, $\eta^2$, $\eta^2_p$, $\omega^2$) |
| | `anovaANCOVA` | 공분산분석, 공변량 통제 후 수정평균 산출 |
| | `anovaRM` | 반복측정 분산분석, Mauchly 구형성 검정, GG/HF 보정 |
| | `post_hoc` | **5종 사후검정** (Tukey HSD, Games-Howell, Scheffé, Bonferroni, Holm) |
| | `kruskal` | 비모수 분산분석 (Kruskal-Wallis H 검정) |
| **회귀 & 범주형** | `corrmatrix`, `partial_corr` | Pearson/Spearman 상관행렬, 편상관분석 |
| | `linReg` | OLS 다중회귀, VIF 다중공선성, Durbin-Watson 잔차독립성, 2×2 진단도 |
| | `logRegBin` | 이항 로지스틱 회귀, 오즈비(Odds Ratio), ROC/AUC 곡선 |
| | `crosstab` | 교차분할표, $\chi^2$ 독립성 검정, Fisher 직접확률, Cramér's V |
| **요인 & 신뢰도** | `reliability` | Cronbach's $\alpha$, McDonald's $\omega$, 항목 제거 시 신뢰도 |
| | `pca`, `efa` | 주성분분석, 탐색적 요인분석(Varimax/Promax/Oblimin, Scree Plot) |
| **구조방정식 (SEM)** | `sem` | `lavaan` 문법 지원(`=~`, `~`, `~~`), CFI/TLI/RMSEA 적합도, 경로도 |
| **패널 & 다층모형** | `panel` | 개체 고정효과(Entity FE), 이원 고정효과, 확률효과(RE), 하우스만 검정 |
| | `mixedModel` | 선형 혼합모형(Multilevel Linear Mixed Model), REML, ICC 계산 |
| **베이지안 & 생존분석** | `bayesTTestIS`, `bayesANOVA` | JZS 코시 사전분포 기반 베이지안 가설검정 ($BF_{10}$) |
| | `mediation`, `moderation` | 부트스트랩 매개효과, 상호작용 조절효과 및 단순기울기(Simple Slopes) |
| | `survival` | Kaplan-Meier 생존곡선 분석, Cox 비례위험 회귀분석 |

| **텍스트 분석** | `textKeywords` | 형태소 분석(kiwipiepy), 빈도·TF-IDF, 집단별 특징 키워드(c-TF-IDF), 워드클라우드 |
| | `textNetwork` | 키워드 동시출현 관계망, 중심성, Louvain 묶음 |
| | `textCluster` | 문서 임베딩(TF-IDF+LSA 또는 bge-m3) → UMAP → HDBSCAN 군집, 군집 키워드, 주제 묶음, 관계망 |
| **관계망 분석** | `graphNetwork` | 연결 목록 기반 중심성 6종, 밀도·경로 길이, Louvain 묶음, 관계망 그림 |
| **GIS 공간분석** | `spatialWeights` | 공간가중행렬(퀸·룩 인접, k-최근접, 거리, 역거리, 외부 파일), 정규화·고립 지역 처리, .gal·.gwt·.csv 저장 |
| | `spatial_moran` | 전역 Moran's I(정규 근사·순열), LISA 군집(FDR·Bonferroni), Getis–Ord Gi* |
| | `spatial` | 횡단면 공간회귀 OLS·SLX·SAR·SEM·SDM·SDEM·SAC, LM 진단, 모형 비교·LR 검정, 직접·간접·총효과 |
| | `spatialPanel` | 공간패널회귀 고정효과(지역·시점·양방향, Lee–Yu 보정)·확률효과, Hausman, 패널 LM (Stata xsmle·spxtregress 대응) |
| | `gwr` | 지리가중회귀(커널·대역폭 선택, 국지 계수 지도, 공간 변이 검정) |
| | `spatialMap` | 주제도(분위수·자연 구분·등간격 단계구분도) |

> **텍스트 파일 불러오기**: [텍스트 분석 ▸ 텍스트 파일 불러오기]에서 `.txt` `.md` `.docx` `.pdf` `.hwpx` 파일이나 폴더를 선택하면 문서 1건 = 1행 데이터로 변환 (하위 폴더 이름은 '폴더' 변수에 기록).

> **지도 경계 연결**: [데이터 ▸ 지도 경계 연결]에서 `.shp`·`.shx`·`.dbf`(선택 `.prj`·`.cpg`) 또는 `.zip`·`.geojson`·`.gpkg`를 선택하고 데이터의 지역 코드 변수와 경계 속성을 연결 (연결 변수 자동 추천, 일치 지역 수 표시). 경계 없이 경도·위도 좌표 변수로도 분석 가능.

---

## 📈 4. 독립형 15종 학술 그래프 빌더

통계 분석을 돌리지 않고도 데이터만으로 즉시 출판 수준의 그래프를 작성할 수 있습니다:
- **차트 유형**: 막대그래프(`bar`), 누적막대(`stacked_bar`), 선그래프(`line`), 영역도(`area`), 원그래프(`pie`), 도넛차트(`donut`), 히스토그램(`histogram`), 커널밀도곡선(`density`), 상자수염도(`box`), 바이올린플롯(`violin`), 레인클라우드플롯(`raincloud`), 산점도(`scatter`), 다변량 산점도행렬(`pairplot`), 상관 히트맵(`heatmap`), 오차막대 포레스트플롯(`errorbar`).
- **학술 팔레트**: `openstat`, `apa_bw`(흑백 APA), `nature`, `lancet`, `kihasa`(한국보건사회연구원 스타일), `okabe_ito`(색각이상자 배려), `viridis` 등 10종 지원.
- **포맷 내보내기**: 무손실 벡터 SVG, 인쇄용 PDF, 고해상도(300~600 DPI) PNG.

---

## 📄 5. 11개 포맷 결과 내보내기

분석된 APA 통계 표와 렌더링된 모든 그래프를 다음 형식으로 한 번에 내보낼 수 있습니다:
1. **`.hwpx`**: 한글 표준 규격(OWPML)의 편집 가능한 네이티브 표(`<hp:tbl>`)와 본문에 자동 삽입된 그래프 이미지(`<hp:pic>`)
2. **`.pdf`**: 출판 논문 형식의 다면 리포트
3. **`.docx`**: 마이크로소프트 워드 문서 (APA 3선 테두리 표 + 차트)
4. **`.xlsx`**: 엑셀 워크북 (통계 표 시트 + 차트 전용 시트)
5. **`.tex`**: LaTeX `booktabs` 테이블 코드 + PDF 벡터 그림 파일 연동
6. **기타**: `.html`, `.md`, `.csv`, `.tsv`, `.json`, `.omv`

---

## ❓ 자주 묻는 질문 (FAQ) & 문제 해결

### Q1. "파이썬이 설치되어 있지 않다"는 메시지가 뜹니다.
- **해결책**: [https://www.python.org/downloads/](https://www.python.org/downloads/) 에 접속하여 파이썬 최신 버전을 다운로드하여 설치하세요.
- **★ 중요**: 파이썬 설치 화면 맨 첫 창 아래쪽에 있는 **`[Add python.exe to PATH]`** 체크박스에 꼭 체크를 하고 설치를 진행해야 합니다!

### Q2. 사내망이나 학교망에서 패키지 다운로드가 느리거나 방화벽에 막힙니다.
- `requirements.txt`에 명시된 패키지들을 다운로드받는 과정에서 보안 프로그램이 차단할 경우, 프록시 설정이나 네트워크 예외 처리를 확인해 주세요.

### Q3. R 언어를 따로 설치해야 하나요?
- **아닙니다.** OpenStat Studio는 scipy, statsmodels, pingouin, linearmodels, semopy 등 파이썬 자체의 과학 계산 생태계로 100% 재작성되었으므로, **R 언어를 설치할 필요가 전혀 없습니다.**

### Q4. 분석 결과가 자모비나 Stata, SPSS와 일치하나요?
- 네, 모든 28개 통계 분석 엔진은 표준 알고리즘과 수식에 기반하여 검증되었으며, 소수점 이하 수치까지 일치하도록 정밀 검증(3-Gate Verification)을 완료하였습니다.
