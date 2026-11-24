# AI 분석 지원 실측 — baseline_no_knowledge

- 모형: gemini-3.1-flash-lite · 프로그램 지식 제공: 아니오(기준선) · 문항 28개
- 통과 10/28 (36%) · 고쳐 쓰기 없이 통과 8/28 · 모형 호출 96회 · 596.5초

| 문항 | 자료 | 판정 | 고쳐 쓰기 | 초 | 비고 |
|---|---|---|---:|---:|---|
| T01 | ToothGrowth.csv | wrong | 2 | 9.8 | 분석 anovaOneW |
| T02 | ToothGrowth.csv | wrong | 0 | 53.3 | 옵션 ttestIS.welchs=True (작성: None) |
| T03 | ToothGrowth.csv | wrong | 0 | 2.4 | 옵션 descriptives.splitBy='dose' (작성: None); 옵션 descriptives.skew=True (작성: None); 옵션 descriptives.kurt=True (작성: None); 옵 |
| T04 | ToothGrowth.csv | pass | 0 | 5.1 |  |
| T05 | ToothGrowth.csv | pass | 0 | 2.9 |  |
| T06 | ToothGrowth.csv | pass | 0 | 3.3 |  |
| T07 | ToothGrowth.csv | not_run | 0 | 1.6 | 코드 없음 |
| M01 | MedMod_WorkStress.csv | not_run | 2 | 3.2 | 실행 오류: KeyError: '[None] not in index' |
| M02 | MedMod_WorkStress.csv | not_run | 3 | 48.7 | 실행 오류: KeyError: '[None] not in index' |
| M03 | MedMod_WorkStress.csv | pass | 0 | 4.4 |  |
| M04 | MedMod_WorkStress.csv | wrong | 0 | 9.8 | 분석 compareModels |
| M05 | MedMod_WorkStress.csv | pass | 2 | 107.9 |  |
| M06 | MedMod_WorkStress.csv | wrong | 0 | 4.5 | 옵션 corrMatrix.spearman=True (작성: None) |
| P01 | Grunfeld_Panel.csv | wrong | 2 | 9.9 | 옵션 panel.modelType='fixed' (작성: None); 옵션 panel.covType='robust' (작성: None) |
| P02 | Grunfeld_Panel.csv | wrong | 0 | 3.5 | 옵션 panel.modelType='twoway' (작성: None) |
| P03 | Dynamic_Panel_Welfare.csv | pass | 1 | 7.0 |  |
| S01 | SEM_Latent3Factor.csv | pass | 0 | 5.7 |  |
| S02 | SEM_Latent3Factor.csv | not_run | 4 | 56.8 | 실행 오류: ValueError: PCA / EFA requires at least 2 continuous variables. |
| S03 | SEM_Latent3Factor.csv | not_run | 2 | 12.7 | 실행 오류: ValueError: Reliability Analysis requires at least 2 scale items. |
| C01 | Count_HealthVisits.csv | not_run | 0 | 5.4 | 코드 없음 |
| C02 | Count_HealthVisits.csv | wrong | 0 | 21.4 | 옵션 countReg.inflate=['만성질환수'] (작성: None) |
| L01 | LimDep_Employment_Wage.csv | pass | 0 | 15.7 |  |
| L02 | LimDep_Employment_Wage.csv | wrong | 1 | 69.7 | 분석 binaryReg |
| H01 | Multilevel_School3L.csv | wrong | 1 | 27.6 | 옵션 mixedModel.cluster='학급ID' (작성: None); 옵션 mixedModel.cluster3='학교ID' (작성: None) |
| I01 | IV_Schooling.csv | wrong | 0 | 23.4 | 옵션 ivReg.endog=['educ'] (작성: None) |
| A01 | TS_Monthly_Welfare.csv | not_run | 1 | 76.1 | 실행 오류: ValueError: Unsupported analysis type: 'adfTest' |
| X01 | ToothGrowth.csv | pass | 0 | 1.5 |  |
| X02 | MedMod_WorkStress.csv | pass | 0 | 2.5 |  |

## 통과하지 못한 문항

### T01 · wrong
- 요청: 용량에 따라 치아 길이에 차이가 있는지 분석하고 Games-Howell 사후검정과 등분산 검정도 해줘
- 사유: 분석 anovaOneW

```python
sess.set_measure(['dose'], 'Nominal')
sess.run('linReg', dep='len', covs=['dose'])
```

### T02 · wrong
- 요청: 보충제 종류에 따라 치아 길이 평균이 다른지 Welch 검정으로 확인
- 사유: 옵션 ttestIS.welchs=True (작성: None)

```python
sess.run('ttestIS', dep='len', group='supp', varEqual=False)
```

### T03 · wrong
- 요청: len의 기술통계를 dose별로 나눠서, 왜도 첨도랑 정규성 검정 포함
- 사유: 옵션 descriptives.splitBy='dose' (작성: None); 옵션 descriptives.skew=True (작성: None); 옵션 descriptives.kurt=True (작성: None); 옵션 descriptives.sw=True (작성: None)

```python
sess.run('descriptives', dep='len', group='dose', stats=['mean', 'sd', 'min', 'max', 'skew', 'kurt'], normality=True)
```

### T07 · not_run
- 요청: 치아 길이 분포가 용량 집단 간에 다른지 비모수 검정으로
- 사유: 코드 없음

```python
(코드 없음)
```

### M01 · not_run
- 요청: 직무 스트레스가 이직의도에 미치는 영향을 소진이 매개하는지 부트스트랩 1000회로 검정
- 사유: 실행 오류: KeyError: '[None] not in index'

```python
(코드 없음)
```

### M02 · not_run
- 요청: 스트레스와 이직의도의 관계를 사회적 지지가 조절하는지 보고 단순기울기 그림도
- 사유: 실행 오류: KeyError: '[None] not in index'

```python
(코드 없음)
```

### M04 · wrong
- 요청: 모형1은 스트레스만, 모형2는 소진과 사회적 지지를 추가해서 이직의도 회귀를 하고 두 모형을 나란히 비교표로
- 사유: 분석 compareModels

```python
m1 = sess.run('linReg', dep='Turnover_Intent', covs=['Stress'])
m2 = sess.run('linReg', dep='Turnover_Intent', covs=['Stress', 'Burnout', 'Social_Support'])
sess.compare_models(m1, m2, names=['모형 1', '모형 2'])
```

### M06 · wrong
- 요청: Stress, Burnout, Social_Support, Turnover_Intent 상관행렬을 스피어만으로, 유의성 별표 표시
- 사유: 옵션 corrMatrix.spearman=True (작성: None)

```python
sess.run('corrMatrix', covs=['Stress', 'Burnout', 'Social_Support', 'Turnover_Intent'], method='spearman', sigStars=True)
```

### P01 · wrong
- 요청: xtreg invest value capital, fe vce(robust) 해주고 하우스만 검정도
- 사유: 옵션 panel.modelType='fixed' (작성: None); 옵션 panel.covType='robust' (작성: None)

```python
sess.run('panel', dep='invest', covs=['value', 'capital'], entity='firm', time='year', model='fe', vce='robust', hausman=True)
```

### P02 · wrong
- 요청: 기업과 연도 이원 고정효과로 투자를 기업가치와 자본에 회귀
- 사유: 옵션 panel.modelType='twoway' (작성: None)

```python
sess.run('panel', dep='invest', covs=['value', 'capital'], entity='firm', time='year', model='fixed')
```

### S02 · not_run
- 요청: x1_vis부터 x9_spd까지 9개 문항으로 탐색적 요인분석, 요인 3개, 사각 회전
- 사유: 실행 오류: ValueError: PCA / EFA requires at least 2 continuous variables.

```python
(코드 없음)
```

### S03 · not_run
- 요청: x1_vis x2_vis x3_vis 세 문항의 크론바흐 알파와 오메가
- 사유: 실행 오류: ValueError: Reliability Analysis requires at least 2 scale items.

```python
(코드 없음)
```

### C01 · not_run
- 요청: 외래방문 횟수를 연령, 여성, 만성질환수로 설명하는 음이항 회귀. 발생률비로 보고
- 사유: 코드 없음

```python
(코드 없음)
```

### C02 · wrong
- 요청: 응급실방문은 0이 너무 많아서 영과잉 포아송으로. 영과잉 부분은 만성질환수로
- 사유: 옵션 countReg.inflate=['만성질환수'] (작성: None)

```python
sess.run('countReg', dep='응급실방문', covs=['연령', '여성', '만성질환수', '소득분위', '지역유형', '의료급여', '관찰개월'], zi_covs=['만성질환수'], modelType='zip')
```

### L02 · wrong
- 요청: 취업여부를 프로빗으로 추정하고 평균 한계효과
- 사유: 분석 binaryReg

```python
sess.run('logRegBin', dep='취업여부', covs=['연령', '교육연수', '경력', '미취학자녀수', '배우자소득', '지역유형'], link='probit', marginalEffects=True)
```

### H01 · wrong
- 요청: 수학성취도를 가정SES와 학습시간으로 설명하는 3수준 다층모형(학생-학급-학교), ICC도
- 사유: 옵션 mixedModel.cluster='학급ID' (작성: None); 옵션 mixedModel.cluster3='학교ID' (작성: None)

```python
sess.run('mixedModel', dep='수학성취도', covs=['가정SES', '학습시간'], levels=['학교ID', '학급ID'], icc=True)
```

### I01 · wrong
- 요청: 교육의 임금 수익률을 nearc4를 도구변수로 2SLS 추정. 외생변수는 exper, expersq, black, south
- 사유: 옵션 ivReg.endog=['educ'] (작성: None)

```python
sess.run('ivReg', dep='lwage', endogenous='educ', instruments='nearc4', exogenous=['exper', 'expersq', 'black', 'south'])
```

### A01 · not_run
- 요청: lvisits에 대해 ADF 단위근 검정
- 사유: 실행 오류: ValueError: Unsupported analysis type: 'adfTest'

```python
(코드 없음)
```

