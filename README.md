# ESS 배터리 수명 예측 
목적 작성 

## 프로젝트 개요
- 데이터셋 : MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 학습 데이터 : Batch 1 (2017-05-12)
- 평가 데이터 : Batch 2 (2018-02-20)
- 태스크 : Regression (Cycle Life 예측) / Classification (장단수명 분류)  ← 택1 


## 파일 구조
```
├── data/
│   └── README.md
├── src/
│   ├── preprocess.py
│   ├── features.py
│   └── train.py
├── results/
│   └── model_performance.csv
├── requirements.txt
└── README.md
```

## 파일 설명

### `src/preprocess.py`
배터리 `.mat` 데이터를 로드하고 사이클별 DataFrame으로 변환한다.

| 함수 | 역할 |
|------|------|
| `load_batch(path)` | MATLAB v7.3(mat73) / v7.2(scipy) 자동 감지 로드 → 셀 리스트 반환 |
| `batch_to_summary_df(batch, batch_id)` | 셀 리스트를 사이클별 long-format DataFrame으로 변환 (cell_id, cycle, QD, IR, Tavg, Tmax, chargetime, cycle_life) |

### `src/features.py`
EDA 결과 기반 최종 5개 피처를 추출한다.

| 함수 | 역할 |
|------|------|
| `extract_dq_features(batch)` | ΔQ(V) = Qdlin[cycle=100] − Qdlin[cycle=10] → `dQ_min`, `dQ_mean`, `dQ_var` |
| `extract_summary_features(summary_df)` | 초기 100사이클 평균 → `mean_chargetime`, `mean_Tavg` |
| `build_feature_matrix(batch, batch_id)` | 두 함수 결과를 병합하여 셀 단위 최종 피처 행렬 반환 |

**최종 피처셋** : `dQ_min`(r=0.882), `dQ_mean`(r=0.854), `dQ_var`(r=−0.838), `mean_chargetime`(r=0.577), `mean_Tavg`(r=−0.482)
**제외 피처** : `mean_Tmax`(mean_Tavg와 r=0.96), `std_QD`(mean_QD와 Spearman r=0.93), `mean_IR`(r=0.18)

### `src/train.py`
모델 학습 및 평가 전체 파이프라인. `python src/train.py` 로 실행.

| 단계 | 내용 |
|------|------|
| Hold-out 분리 | Batch 1 셀 단위 80/20 (train ≈37셀 / valid ≈9셀, random_state=42) |
| 스케일링 | `StandardScaler` — X_train 기준으로만 fit, valid·test는 transform만 적용 |
| 후보 모델 | `RidgeCV`, `LassoCV` (α 자동선택), `GradientBoostingRegressor(max_depth=3)` |
| 평가 구조 | Train 5-fold CV · Valid Hold-out · Test(Batch 2) — RMSE / R² / MAE |
| 결과 저장 | `results/model_performance.csv` 자동 생성 |

**최종 모델 선택 기준** : Valid RMSE 최소 모델. Ridge vs 최선 모델 차이 ≤ 5%이면 해석 가능성 우선으로 Ridge 채택.

### `results/model_performance.csv`
train.py 실행 시 자동 생성되는 모델별 성능 비교표.

| 컬럼 | 설명 |
|------|------|
| `model_name` | Ridge / Lasso / GradientBoosting |
| `train_cv_rmse/r2/mae` | Batch 1 Train 5-fold CV 평균 |
| `valid_rmse/r2/mae` | Batch 1 Hold-out 검증 |
| `test_rmse/r2/mae` | Batch 2 최종 평가 |


## 환경 설정 (sample)
```bash
git clone https://github.com/팀명/ess-battery-project
cd ess-battery-project
pip install -r requirements.txt
```

## 실행 방법

### 1단계: 데이터 준비 (필수)
`archive/` 폴더를 프로젝트 루트에 생성하고, 제공된 `.mat` 원본 데이터 2개를 해당 경로에 위치시켜야 합니다.
- 파일명이 `src/preprocess.py`에 정의된 `BATCH1_PATH`, `BATCH2_PATH`와 일치해야 합니다.

### 2단계: 코드 수정 (팀원 개별 작업)
팀원들은 각자의 분석 및 모델링 전략에 맞춰 아래 파일들을 직접 수정해야 합니다.

1. **피처 엔지니어링 변경 (`src/features.py`)**
   - 새로운 파생변수(피처) 추출 함수를 구현하세요.
   - 스크립트 하단의 `FEATURES` 리스트를 본인이 선택한 최종 피처들로 업데이트하세요.
2. **모델 및 파라미터 변경 (`src/train.py`)**
   - `build_models()` 함수 내에 있는 분류/회귀 알고리즘을 본인이 선택한 모델로 교체하세요. (예: `RandomForest`, `XGBoost` 등)
   - 파라미터 그리드(alphas, max_depth 등)를 실험 목적에 맞게 변경하세요.
   - (필요한 경우) `sklearn` 외의 패키지 사용 시 `requirements.txt`에 해당 패키지를 추가하세요.

### 3단계: 파이프라인 실행
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
	- 핵심 발견 : (팀이 발견한 인사이트를 한 줄로)

- 열화 곡선 분석
	- 장수명 vs 단수명 셀의 열화 속도 차이
	- Knee point 존재 여부 및 발생 시점
	- 핵심 발견 :

- ΔQ(V) 곡선 분석
	- Cycle 100 - Cycle 10 차이 곡선 형태
	- 장단수명 셀 간 ΔQ 형태 비교
	- 핵심 발견 :

- 충전 속도(C-rate)와 수명의 관계
	- 충전 프로토콜별 평균 수명 비교 결과
	- 핵심 발견 :

- (추가 확인한 내용 작성) 


## Modeling 

### 피처 엔지니어링 전략
EDA 결과를 바탕으로 선택한 피처와 그 근거를 기술


### 모델 선택 및 근거
- 후보 모델 : 
- 최종 모델 :
- 선택 이유 :


## 성능 결과
Format에 맞춰 작성


## 오류 분석
- 모델이 가장 크게 틀린 셀의 공통점
- 원인 가설 및 개선 방향


## ESS 도메인 해석
분석 결과를 실제 ESS 운영 관점에서 해석

- 이 모델을 실제 BESS에 적용한다면 어떤 의사결정에 활용 가능한가?
- 어떤 한계가 있으며, 실 배포를 위해 추가로 필요한 것은 무엇인가?


## 참고문헌
- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391.


## 팀 구성
- 김영희 : EDA, 피처 엔지니어링, 모델 개발, 성능 평가(Batch2)
- 박철수 : EDA, 피처 엔지니어링, 모델 개발, 성능 평가(Batch3)
