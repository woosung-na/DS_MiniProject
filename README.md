# ESS 배터리 수명 예측 
목적 작성 

## 프로젝트 개요
- 데이터셋 : MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 학습 데이터 : Batch 1 (2017-05-12)
- 평가 데이터 : Batch 2 (2018-02-20)
- 태스크 : Regression (Cycle Life 예측) / Classification (장단수명 분류)  ← 택1 


## 파일 구조 (sample) 
```
├── data/
│   └── README.md          
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── preprocess.py
│   ├── features.py
│   └── train.py
├── results/
│   └── model_performance.csv
├── requirements.txt
└── README.md
```


## 환경 설정 (sample) 
```bash
git clone https://github.com/팀명/ess-battery-project
cd ess-battery-project
pip install -r requirements.txt
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


## 오류 분석 -> 안해도 ㄱㅊ음
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
