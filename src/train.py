"""
모델 학습 스크립트

평가 구조:
  - Train  : Batch 1 5-fold CV 평균 성능
  - Valid  : Batch 1 Hold-out 검증 성능  (20%, cell 단위 분리)
  - Test   : Batch 2 최종 평가 성능      (모델 선택 완료 후 단 1회)

후보 모델:
  1. Ridge Regression  — Baseline (선형, 해석 용이)
  2. Lasso Regression  — 피처 희소화 (ΔQ 피처 간 중복 정보 제거)
  3. GradientBoosting  — 비선형 패턴 포착 (max_depth≤3, 과적합 방지)

최종 모델 선택 기준:
  Valid RMSE 최소 모델. Ridge와 GBM 차이 ≤ 5% 이면 Ridge 우선.
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    mae  = float(mean_absolute_error(y_true, y_pred))
    return {'rmse': rmse, 'r2': r2, 'mae': mae}


def cv_metrics(model, X: np.ndarray, y: np.ndarray, cv: int = CV_FOLDS) -> dict:
    """
    cross_validate로 5-fold CV를 수행하고 RMSE·R²·MAE 평균을 반환한다.
    """
    scoring = {
        'neg_rmse': 'neg_root_mean_squared_error',
        'r2':       'r2',
        'neg_mae':  'neg_mean_absolute_error',
    }
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
    return {
        'rmse': float(-scores['test_neg_rmse'].mean()),
        'r2':   float(scores['test_r2'].mean()),
        'mae':  float(-scores['test_neg_mae'].mean()),
    }


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
        print(f"\n  [{name}]")

        # 5-fold CV (train set 기준)
        train_cv = cv_metrics(model, X_train_s, y_train, cv=CV_FOLDS)
        print(f"    Train CV   RMSE={train_cv['rmse']:.1f}  R²={train_cv['r2']:.3f}  MAE={train_cv['mae']:.1f}")

        # Hold-out valid
        model.fit(X_train_s, y_train)
        valid_pred = model.predict(X_valid_s)
        valid_met  = compute_metrics(y_valid, valid_pred)
        print(f"    Valid      RMSE={valid_met['rmse']:.1f}  R²={valid_met['r2']:.3f}  MAE={valid_met['mae']:.1f}")

        # Test (Batch 2) — 단 1회 평가
        test_pred = model.predict(X_test_s)
        test_met  = compute_metrics(y_test, test_pred)
        print(f"    Test(B2)   RMSE={test_met['rmse']:.1f}  R²={test_met['r2']:.3f}  MAE={test_met['mae']:.1f}")

        # Lasso: 선택된 피처 출력
        if name == 'Lasso' and hasattr(model, 'coef_'):
            nonzero = [(f, c) for f, c in zip(FEATURES, model.coef_) if abs(c) > 1e-6]
            print(f"    Lasso 선택 피처: {[f for f, _ in nonzero]} (총 {len(nonzero)}개)")

        results.append({
            'model_name':    name,
            'train_cv_rmse': round(train_cv['rmse'], 2),
            'train_cv_r2':   round(train_cv['r2'],   4),
            'train_cv_mae':  round(train_cv['mae'],   2),
            'valid_rmse':    round(valid_met['rmse'],  2),
            'valid_r2':      round(valid_met['r2'],    4),
            'valid_mae':     round(valid_met['mae'],   2),
            'test_rmse':     round(test_met['rmse'],   2),
            'test_r2':       round(test_met['r2'],     4),
            'test_mae':      round(test_met['mae'],    2),
        })

    # --- 결과 저장 ---
    df_results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    df_results.to_csv(RESULTS_PATH, index=False)
    print(f"\n결과 저장 완료: {RESULTS_PATH}")

    # --- 최종 모델 선택 권고 ---
    print("\n" + "=" * 60)
    print("최종 모델 선택 기준: Valid RMSE 최소")
    best_idx  = df_results['valid_rmse'].idxmin()
    best_name = df_results.loc[best_idx, 'model_name']
    best_rmse = df_results.loc[best_idx, 'valid_rmse']

    ridge_rmse = df_results.loc[df_results['model_name'] == 'Ridge', 'valid_rmse'].values[0]
    gap_pct    = (best_rmse - ridge_rmse) / ridge_rmse * 100 if best_name != 'Ridge' else 0.0

    print(f"  Valid RMSE 최소 모델: {best_name} ({best_rmse:.1f})")
    if best_name != 'Ridge' and abs(gap_pct) <= 5.0:
        print(f"  → Ridge와 차이 {abs(gap_pct):.1f}% ≤ 5% : 해석 가능성 우선 → Ridge 권고")
    else:
        print(f"  → 최종 권고 모델: {best_name}")

    print("\n전체 결과:")
    print(df_results.to_string(index=False))
    print("=" * 60)


if __name__ == '__main__':
    main()
