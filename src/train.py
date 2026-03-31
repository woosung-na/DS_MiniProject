"""
모델 학습 스크립트

평가 구조:
  - Train  : Batch 1 5-fold CV 평균 성능
  - Valid  : Batch 1 Hold-out 검증 성능  (20%, cell 단위 분리)
  - Test   : Batch 2 최종 평가 성능      (모델 선택 완료 후 단 1회)

  Valid를 CV가 아닌 Hold-out으로 설정한 이유:
    배터리 데이터는 셀 단위로 독립적이며 각 셀이 서로 다른 충전 프로토콜로 실험된다.
    CV 적용 시 동일 프로토콜 셀이 train/valid에 나뉘어 데이터 누수 위험이 잔존한다.
    Hold-out은 셀 단위 분리를 명확히 보장하며 배치 간 일반화 평가에 더 적합하다.

후보 모델:
  1. Ridge Regression  — Baseline (선형, 해석 용이)
  2. Lasso Regression  — 피처 희소화 (ΔQ 피처 간 중복 정보 제거)
  3. GradientBoosting  — 비선형 패턴 포착 (max_depth≤3, 과적합 방지)

평가 지표: MAPE (%)
최종 모델 선택 기준: Valid MAPE 최소. Ridge와 GBM 차이 ≤ 5%p 이면 Ridge 우선.
원논문 Target: Regression 9.1% MAPE (Severson et al., Nature Energy 2019)
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_absolute_percentage_error

# features.py가 같은 src/ 디렉터리에 있으므로 경로 추가
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_batch, BATCH1_PATH, BATCH2_PATH
from features import build_feature_matrix, FEATURES, TARGET

RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'model_performance.csv')

HOLD_OUT_RATIO = 0.2
CV_FOLDS       = 5
RANDOM_STATE   = 42


# ---------------------------------------------------------------------------
# 평가 지표 계산
# ---------------------------------------------------------------------------

TARGET_MAPE = 9.1  # 원논문 (Severson et al., 2019) Regression MAPE


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE (%) 반환."""
    return float(mean_absolute_percentage_error(y_true, y_pred) * 100)


def cv_mape(model, X: np.ndarray, y: np.ndarray, cv: int = CV_FOLDS) -> float:
    """
    5-fold CV를 수행하고 MAPE (%) 평균을 반환한다.
    sklearn의 neg_mean_absolute_percentage_error는 소수 단위이므로 ×100 변환.
    """
    scores = cross_validate(
        model, X, y, cv=cv,
        scoring='neg_mean_absolute_percentage_error',
        return_train_score=False,
    )
    return float(-scores['test_score'].mean() * 100)


def print_report(name: str, train_mape: float, valid_mape: float, test_mape: float) -> None:
    """모델별 MAPE 결과를 요구사항 테이블 형식으로 출력한다."""
    gap_tv  = train_mape - valid_mape
    gap_vt  = valid_mape - test_mape
    gap_tgt = test_mape  - TARGET_MAPE

    def note_gap(val: float, label_pos: str) -> str:
        sign = '+' if val > 0 else ''
        flag = f'  ← {label_pos}' if val > 0 else ''
        return f"{sign}{val:.2f}%p{flag}"

    rows = [
        ("Train (Batch 1 CV)",      f"{train_mape:.2f}",  ""),
        ("Valid (Batch 1 Hold-out)", f"{valid_mape:.2f}",  ""),
        ("Test (Batch 2)",           f"{test_mape:.2f}",   ""),
        ("Gap (Train-Valid)",        note_gap(gap_tv,  "과적합 의심"),           "(+) : 과적합 의심"),
        ("Gap (Valid-Test)",         note_gap(gap_vt,  "배치간 일반화 저하 의심"), "(+) : 배치간 일반화 저하 의심"),
        ("Gap (Target-Test)",        note_gap(gap_tgt, "원논문 대비 성능 저하"),  f"Target : 원논문 {TARGET_MAPE}%"),
    ]

    col_w = [26, 12, 30]
    sep   = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    hdr   = f"| {'구분':<{col_w[0]}} | {'MAPE (%)':<{col_w[1]}} | {'비고':<{col_w[2]}} |"

    print(f"\n  [{name}]")
    print("  " + sep)
    print("  " + hdr)
    print("  " + sep)
    for label, val, note in rows:
        print(f"  | {label:<{col_w[0]}} | {val:<{col_w[1]}} | {note:<{col_w[2]}} |")
    print("  " + sep)


# ---------------------------------------------------------------------------
# 모델 정의
# ---------------------------------------------------------------------------

def build_models() -> dict:
    """
    후보 모델 딕셔너리를 반환한다.
    알파는 CV로 자동 선택, GBM은 과적합 방지를 위해 보수적 하이퍼파라미터 적용.
    """
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

    return {
        'Ridge': RidgeCV(alphas=alphas),
        'Lasso': LassoCV(alphas=alphas, cv=CV_FOLDS, max_iter=10_000, random_state=RANDOM_STATE),
        'GradientBoosting': GradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.05,
            n_estimators=100,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
    }


# ---------------------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Battery Cycle Life Prediction — Training Pipeline")
    print("=" * 60)

    # --- 1. 데이터 로드 ---
    print("\n[1/5] 데이터 로드 중...")
    print(f"  Batch 1: {os.path.basename(BATCH1_PATH)}")
    batch1 = load_batch(BATCH1_PATH)
    print(f"  Batch 2: {os.path.basename(BATCH2_PATH)}")
    batch2 = load_batch(BATCH2_PATH)

    # --- 2. 피처 행렬 생성 ---
    print("\n[2/5] 피처 행렬 생성 중...")
    df_b1 = build_feature_matrix(batch1, batch_id=1)
    df_b2 = build_feature_matrix(batch2, batch_id=2)
    print(f"  Batch 1 피처 행렬: {df_b1.shape}  (셀 수: {len(df_b1)})")
    print(f"  Batch 2 피처 행렬: {df_b2.shape}  (셀 수: {len(df_b2)})")
    print(f"  최종 피처: {FEATURES}")

    # --- 3. Hold-out 분리 ---
    print("\n[3/5] Hold-out 분리 (Batch 1, test_size=0.2) ...")
    X_all = df_b1[FEATURES].values
    y_all = df_b1[TARGET].values

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_all, y_all,
        test_size=HOLD_OUT_RATIO,
        random_state=RANDOM_STATE,
    )
    X_test = df_b2[FEATURES].values
    y_test = df_b2[TARGET].values

    print(f"  Train: {len(X_train)}셀  |  Valid: {len(X_valid)}셀  |  Test(Batch 2): {len(X_test)}셀")

    # --- 4. 스케일링 (X_train 기준 fit, 누수 방지) ---
    print("\n[4/5] StandardScaler 적용 (fit: X_train only) ...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_valid_s = scaler.transform(X_valid)
    X_test_s  = scaler.transform(X_test)

    # --- 5. 모델 학습 및 평가 ---
    print("\n[5/5] 모델 학습 및 평가 중...")
    models  = build_models()
    results = []

    for name, model in models.items():
        # Train: 5-fold CV MAPE
        train_mape = cv_mape(model, X_train_s, y_train, cv=CV_FOLDS)

        # Valid: Hold-out MAPE
        model.fit(X_train_s, y_train)
        valid_mape = compute_mape(y_valid, model.predict(X_valid_s))

        # Test: Batch 2 MAPE (단 1회)
        test_mape = compute_mape(y_test, model.predict(X_test_s))

        # Lasso: 선택된 피처 출력
        if name == 'Lasso' and hasattr(model, 'coef_'):
            nonzero = [f for f, c in zip(FEATURES, model.coef_) if abs(c) > 1e-6]
            print(f"\n  [Lasso 선택 피처]: {nonzero} (총 {len(nonzero)}개)")

        print_report(name, train_mape, valid_mape, test_mape)

        results.append({
            'model_name':      name,
            'train_cv_mape':   round(train_mape, 2),
            'valid_mape':      round(valid_mape,  2),
            'test_mape':       round(test_mape,   2),
            'gap_train_valid': round(train_mape - valid_mape, 2),
            'gap_valid_test':  round(valid_mape  - test_mape,  2),
            'gap_target_test': round(test_mape   - TARGET_MAPE, 2),
        })

    # --- 결과 저장 ---
    df_results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    df_results.to_csv(RESULTS_PATH, index=False)
    print(f"\n결과 저장 완료: {RESULTS_PATH}")

    # --- 최종 모델 선택 권고 ---
    print("\n" + "=" * 60)
    print("최종 모델 선택 기준: Valid MAPE 최소")
    best_idx   = df_results['valid_mape'].idxmin()
    best_name  = df_results.loc[best_idx, 'model_name']
    best_mape  = df_results.loc[best_idx, 'valid_mape']

    ridge_mape = df_results.loc[df_results['model_name'] == 'Ridge', 'valid_mape'].values[0]
    gap_pp     = best_mape - ridge_mape  # %p 차이 (음수 = best가 Ridge보다 낮음)

    print(f"  Valid MAPE 최소 모델: {best_name} ({best_mape:.2f}%)")
    if best_name != 'Ridge' and abs(gap_pp) <= 5.0:
        print(f"  → Ridge와 차이 {abs(gap_pp):.2f}%p ≤ 5%p : 해석 가능성 우선 → Ridge 권고")
    else:
        print(f"  → 최종 권고 모델: {best_name}")

    print("\n전체 결과:")
    print(df_results.to_string(index=False))
    print("=" * 60)


if __name__ == '__main__':
    main()
