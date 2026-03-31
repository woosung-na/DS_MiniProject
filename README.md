# ESS 배터리 수명 예측 
ESS는 다수의 배터리 셀과 모듈로 구성되며, 개별 셀의 성능 저하나 이상은 전체 시스템의 안정성과 신뢰성에 직접적인 영향을 미친다. 
특히 ESS는 전력 공급 연속성이 중요한 설비에 적용되며, 최근에는 데이터센터용 UPS 등으로 활용 범위가 확대되고 있어 개별 셀 상태를 조기에 진단하고 수명을 예측하는 기술의 중요성이 커지고 있다. 
본 프로젝트는 이러한 필요성에 기반하여, 초기 충·방전 데이터를 활용한 셀 단위 수명 예측 가능성을 확인하는 것을 목적으로 한다.

## 프로젝트 개요
- 데이터셋 : MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 학습 데이터 : Batch 1 (2017-05-12)
- 평가 데이터 : Batch 2 (2018-02-20), Batch 3 (2018-04-12)
- 태스크 : Regression (Cycle Life 예측)

## 파일 구조
```
├── data/
│   └── README.md
├── eda.ipynb
├── results/
│   └── model_performance.csv
├── src/
│   ├── preprocess.py
│   ├── features.py
│   └── train.py
├── requirements.txt
└── README.md
```

## 파일 설명

### `src/preprocess.py`
배터리 `.mat` 데이터를 로드하고 사이클별 DataFrame으로 변환한다.

| 함수 | 역할 |
|------|------|
| `load_batch` | MATLAB v7.3(mat73) / v7.2(scipy) 자동 감지 로드 |
| `batch_to_summary_df` | 사이클별 시계열 데이터(QD, IR, Tavg 등) 생성 |

### `src/features.py`
EDA 결과 기반 최종 7개 피처를 추출한다.

- **최종 피처셋 (v2)**:
  - `dQ_min`, `dQ_mean`, `dQ_var` : ΔQ(V) 곡선 기반 열화 신호
  - **`dQ_zone_ratio_low`, `dQ_zone_ratio_high`** : 전압 구간별 열화 비중
  - `mean_chargetime`, `mean_Tavg` : 초기 100사이클 운용 요약값

### `src/train.py`
모델 학습 및 평가 전체 파이프라인. `python src/train.py` 로 실행한다.

| 단계 | 내용 |
|------|------|
| Hold-out 분리 | Batch 1 셀 단위 80/20 (train ≈37셀 / valid ≈9셀, random_state=42) |
| 스케일링 | `StandardScaler` — X_train 기준으로만 fit, valid·test는 transform만 적용 |
| 후보 모델 | `RidgeCV`, `LassoCV` (α 자동선택), `GradientBoostingRegressor(max_depth=3)` |
| 평가 구조 | Train 5-fold CV · Valid Hold-out · Test(Batch 2) — RMSE / R² / MAE |
| 결과 저장 | `results/model_performance.csv` 자동 생성 |

**최종 모델 선택 기준** : Valid RMSE 최소 모델. Ridge vs 최선 모델 차이 ≤ 5%이면 해석 가능성 우선으로 Ridge 채택한다.

### `results/model_performance.csv`
train.py 실행 시 자동 생성되는 모델별 성능 비교표.

| 컬럼 | 설명 |
|------|------|
| `model_name` | Ridge / Lasso / GradientBoosting |
| `train_cv_rmse/r2/mae` | Batch 1 Train 5-fold CV 평균 |
| `valid_rmse/r2/mae` | Batch 1 Hold-out 검증 |
| `test_rmse/r2/mae` | Batch 2 최종 평가 |

## 환경 설정
```bash
git clone https://github.com/woosung-na/DS_MiniProject.git
cd DS_MiniProject
pip install -r requirements.txt
```

## 실행 방법

### 1단계: 데이터 준비
`archive/` 폴더를 프로젝트 루트에 생성하고, 제공된 `.mat` 원본 데이터 2개를 해당 경로에 위치시켜야 한다.
- 파일명이 `src/preprocess.py`에 정의된 `BATCH1_PATH`, `BATCH2_PATH`와 일치해야 한다.

### 2단계: 파이프라인 실행
```bash
# 전체 파이프라인 실행 (src/ 디렉터리 기준)
cd src
python train.py          # 학습 + 평가 + results/model_performance.csv 자동 생성

# 단계별 단독 실행 (디버깅 및 중간 확인용)
python preprocess.py     # 데이터 로드 정상 확인 (설치 환경 및 경로 테스트)
python features.py       # 변경한 피처 정상 추출 및 상관계수 테스트
```

## EDA 

- Cycle Life 분포
	- 분포 형태 및 장단수명 비율 요약
	- 핵심 발견 : Batch1은 Normal(500~999) 구간에, Batch2는 Short(<500) 구간에, Batch3는 Long(>1000) 구간에 집중되며, 특히 Batch3는 고수명 셀의 편차가 커 장수명 셀일수록 초기 특성·충전 조건 차이에 더 민감하게 반응할 가능성을 시사한다.

- 열화 곡선 분석
	- 장수명 vs 단수명 셀의 열화 속도 차이
	- Knee point 존재 여부 및 발생 시점
	- 핵심 발견 : 단수명 셀일수록 열화 곡선이 더 깊고 빠르게 꺾이며, Batch2는 초반부터 급격한 용량 감소가 나타나 조기 열화 특성이 가장 뚜렷하다. Knee point를 통해 급격한 열화 시작 시점이 늦을수록 전체 cycle life가 길어지는 경향을 확인했다.

- ΔQ(V) 곡선 분석
	- Cycle 100 - Cycle 10 차이 곡선 형태
	- 장단수명 셀 간 ΔQ 형태 비교
	- 핵심 발견 : 단수명 셀일수록 2.8~3.1V 구간에서 더 깊은 음의 valley가 나타난다. 이때, batch 3 내 일부 장수명 셀에서 3.0~3.1V 부근 ΔQ(V) 곡선의 상승·변동이 나타나 동일 수명군 내부 이질성이 커지고, 이로 인해 단순 선형 규칙만으로는 모델링이 어려울 가능성을 시사한다.

- 충전 속도(C-rate)와 수명의 관계
	- 충전 프로토콜별 평균 수명 비교 결과
	- 핵심 발견 :Batch1은 고속충전일수록 수명이 짧고, Batch2는 상관이 약하며, Batch3는 평균 수명보다 최대 수명을 제한하는 형태로 작용했다.

## Modeling 

### 피처 엔지니어링 전략
본 프로젝트의 피처 엔지니어링은 철저하게 탐색적 데이터 분석(EDA) 과정에서 도출된 인사이트를 기반으로 진행되었다.

- 초기 핵심 Feature 선정
	- 내부 열화 지표 기반 (dQ_min, dQ_mean, dQ_var)
			근거: 초기 사이클 구간의 추세만으로는 장/단수명 셀을 구분할 수 없었으나, 전압별 차이 곡선 분석에서 명확한 패턴 차이 (단수명 셀일수록 곡선이 깊게 파이는 현상)를 발견했다.
	- 외부 지표 기반 (mean_chargetime, mean_Tavg)
			근거: 초기 100 사이클 요약 데이터 분석 결과, 장수명 배터리들은 충전 시간이 길어 발열이 통제되는 경향을 보였다.
			참고: mean_Tmax는 mean_Tavg와 피어슨 상관계수가 0.96으로 매우 높아 다중공선성으로 판단하여 제외하였다.

-	신규 Feature 도입 및 변환
	- 열화 패턴 구체화 시도 (dQ_mean_low, dQ_mean_mid, dQ_mean_high 등)
			근거: 기존 지표는 열화의 전체 규모만 보여줄 뿐, 어느 전압 구간에서 열화가 집중되었는지에 대한 패턴을 파악하기 어려워 전체 전압을 3등분하여 절대값을 추출했다.
			참고: 해당 절대값들은 전체 열화량과 완전한 선형 관계(r>0.99)를 가져 심각한 다중공선성을 유발하므로 학습 변수에서 제외하였다.
	- 다중공선성 해소를 위한 파생 Feature 생성 (dQ_zone_ratio_low, dQ_zone_ratio_high)
			근거: 절대값을 전체 열화량으로 나눈 비율로 변환함으로써, 중복되는 크기 정보를 수학적으로 상쇄하고 순수한 구간별 열화 비중 정보만 독립적으로 모델에 전달하였다.
			참고: 세 구간 비율의 합은 항상 1이 되므로, 선형 종속성을 방지하기 위해 통계적 정석에 따라 Mid 구간을 생략하고 2개 변수만 최종 채택하였다.

따라서 최종적으로 배터리의 내부 열화 패턴과 외부 스트레스 요인을 학습할 수 있는 총 7개의 피처(dQ_min, dQ_mean, dQ_var, dQ_zone_ratio_low, dQ_zone_ratio_high, mean_chargetime, mean_Tavg)를 최종 feature로 채택하였다.

### 모델 선택 및 근거
- 후보 모델 : Ridge, Lasso, ElasticNet, GradientBoosting
- 최종 모델 : Ridge
- 선택 이유 :
	- 샘플 수가 46개로 매우 적어 트리 기반 모델은 과적합 위험이 크다. 실제로 GradientBoosting의 Test B2 MAPE는 35.26%로 Ridge(26.23%) 대비 9%p 이상 높아 과적합이 확인되었다.
	- 핵심 피처들이 cycle_life와 선형 관계를 보여(dQ_min r=0.882, mean_chargetime r=0.577) 선형 모델로 충분한 설명력이 확보된다. 
	- 또한 dQ_min, dQ_mean, dQ_var 간 상관이 높아 Lasso보다 피처를 고르게 유지하는 Ridge가 더 안정적이다.
	- ElasticNet과 성능이 유사하나, 46개 소규모 샘플에서 하이퍼파라미터가 많아질수록 결과가 불안정해지는 점을 고려해 alpha 단일 튜닝으로 안정적인 Ridge를 최종 선택하였다.

## 성능 결과

| 구분 | | MAPE (%) | 비고 |
| :--- | :--- | :--- | :--- |
| Train (Batch 1 CV) | | 8.48 | |
| Valid (Batch 1 Hold-out) | | 6.56 | |
| Test (Batch 2) | | 25.13 | |
| | Gap (Train-Valid) | +1.92%p | 과적합 없음 (Valid 성능 우수) |
| | Gap (Valid-Test) | -18.57%p | 배치 간 일반화 성능 저하 뚜렷 |
| | Gap (Target-Test) | +16.03%p | 원논문(9.1%) 대비 성능 크게 하회 |
| Test (Batch 3) | | 13.73 | |
| | Gap (Batch2-Batch3) | +11.41%p | Batch 3에서 성능 일부 회복 |
| | Gap (Target-Test) | +4.63%p | 원논문 성능에 근접 |

- Target, Test 배치별 데이터 분포가 크게 달라 큰 예측 오류 발생

결과 해석
- Gap (Train - Valid) : +1.92%p (Ridge 기준)
Train CV MAPE(8.48%) > Valid MAPE(6.56%) 로 양수지만 차이가 작다.
과적합 없이 Hold-out 검증 셋에서도 안정적으로 예측하고 있다.
모든 모델이 1~2%p 수준으로 정상 범위다.

- Gap (Valid - Test B2) : -18.57%p (Ridge 기준)
Valid MAPE(6.56%) << Test B2 MAPE(25.13%) 로 가장 큰 문제다.
모델이 Batch 1 셀들의 패턴은 잘 학습했지만, 새로운 배치의 셀에 대한 일반화가 크게 떨어진다.
이 Gap은 하이퍼파라미터 튜닝으로는 해결이 어렵고, 피처 자체의 배치 간 분포 차이에서 기인한다.

- Gap (Target - Test B2) : +16.03%p (Ridge 기준)
원논문 목표 9.1% 대비 25.13%로 약 2.8배 수준이다.
다만 원논문은 Batch 1만 사용해 학습/테스트를 동일 배치 내에서 수행한 반면, 본 실험은 **배치 간 예측(Batch 1 → Batch 2)**이므로 직접 비교는 공정하지 않다.
Test B3(13.72%)는 원논문 목표에 4.62%p 차이로 훨씬 근접해, 배치 간 격차가 줄어들면 목표 달성 가능성이 있다.

## ESS 도메인 해석
분석 결과를 실제 ESS 운영 관점에서 해석

- 이 모델을 실제 BESS에 적용한다면 어떤 의사결정에 활용 가능한가?
1. 조기 불량 셀 선별 (batch 2와 같이 급격한 초기 열화)를 보이는 셀을 조기 식별 가능-> 랙/모듈 단위 교체 혹은 증설이 가능하다.
2. 예측 수명이 짧게 나오는 셀에 대해서는 낮은 c-rate 할당하여 열화 속도를 늦춰 운영 측면에서 수명 연장 가능하다. 또는, BMS 등으로 접촉기(contactor)제어하여, 고장 셀을 물리적으로 차단 후 운영한다.
 
- 어떤 한계가 있으며, 실 배포를 위해 추가로 필요한 것은 무엇인가?
Batch 1 검증(Valid) MAPE는 7.09%로 비교적 낮지만, Batch 2 테스트 MAPE는 25.24%로 무려 18.14%p나 성능이 였다.g
따라서, 실제 현장에서는, 운영 데이터의 분포를 감시하고 스케일러를 업데이트하는 체계가 필요하다.
실시간으로 보간(Interpolation)하여 Q(V)곡선으로 변환하고 ΔQ를 계산하는 전처리 모듈이 필요하다.

## 참고문헌
- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391.
- 2차전지 수명예측방법에 대한 고찰 (2018.02).(이창호) A study on life estimation system for a Secondary Battery

## 팀 구성
- 김도익 : EDA, 피처 엔지니어링, 모델 개발, 성능 평가(Batch2)
- 나우성 : EDA, 피처 엔지니어링, 모델 개발, 성능 평가(Batch1)
- 배세은 : EDA, 피처 엔지니어링, 모델 개발, 성능 평가(Batch2)
- 인유진 : EDA, 피처 엔지니어링, 모델 개발, 성능 평가(Batch3)
