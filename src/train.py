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
  3. ElasticNet        — Ridge + Lasso 결합
  4. GradientBoosting  — 비선형 패턴 포착 (max_depth≤3, 과적합 방지)

평가 지표: MAPE (%)
최종 모델 선택 기준: Valid MAPE 최소. Ridge와 GBM 차이 ≤ 5%p 이면 Ridge 우선.
원논문 Target: Regression 9.1% MAPE (Severson et al., Nature Energy 2019)

"""

import os
import sys
import numpy as np
import pandas as pd

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_absolute_percentage_error

# features.py가 같은 src/ 디렉터리에 있으므로 경로 추가
sys.path.insert(0, os.path.dirname(__file__))
import preprocess
from preprocess import load_batch, BATCH1_PATH, BATCH2_PATH
from features import build_feature_matrix_v2, FEATURES_V2, TARGET

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


def print_report(
    name: str,
    train_mape: float,
    valid_mape: float,
    test_b2_mape: float,
    test_b3_mape: float | None = None,
) -> None:
    """모델별 MAPE 결과를 요구사항 테이블 형식으로 출력한다."""
    gap_tv      = train_mape - valid_mape
    gap_vt      = valid_mape - test_b2_mape
    gap_tgt_b2  = test_b2_mape - TARGET_MAPE

    gap_b2_b3  = None
    gap_tgt_b3 = None
    if test_b3_mape is not None:
        gap_b2_b3  = test_b2_mape - test_b3_mape
        gap_tgt_b3 = test_b3_mape - TARGET_MAPE

    def note_gap(val: float, label_pos: str) -> str:
        sign = '+' if val > 0 else ''
        flag = f'  ← {label_pos}' if val > 0 else ''
        return f"{sign}{val:.2f}%p{flag}"

    rows = [
        ("Train (Batch 1 CV)",      f"{train_mape:.2f}",  ""),
        ("Valid (Batch 1 Hold-out)", f"{valid_mape:.2f}",  ""),
        ("Test (Batch 2)",           f"{test_b2_mape:.2f}",   ""),
        ("Gap (Train-Valid)",        note_gap(gap_tv,  "과적합 의심"),           "(+) : 과적합 의심"),
        ("Gap (Valid-Test)",         note_gap(gap_vt,  "배치간 일반화 저하 의심"), "(+) : 배치간 일반화 저하 의심"),
        ("Gap (Target-Test)",        note_gap(gap_tgt_b2, "원논문 대비 성능 저하"),  f"Target : 원논문 {TARGET_MAPE}%"),
    ]

    if test_b3_mape is not None and gap_b2_b3 is not None and gap_tgt_b3 is not None:
        rows.extend([
            ("Test (Batch 3)",           f"{test_b3_mape:.2f}",   ""),
            ("Gap (Batch2-Batch3)",      note_gap(gap_b2_b3, "Test 성능 차이"), "Test 성능 간 비교"),
            ("Gap (Target-Test B3)",     note_gap(gap_tgt_b3, "원논문 대비 성능 저하"), f"Batch 3 기준, Target {TARGET_MAPE}%"),
        ])

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
    # alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    alphas = np.logspace(-3, 3, 50)

    return {
        'Ridge': RidgeCV(alphas=alphas),
        'Lasso': LassoCV(alphas=alphas, cv=CV_FOLDS, max_iter=10_000, random_state=RANDOM_STATE),
        'ElasticNet': ElasticNetCV(alphas=alphas, l1_ratio=[0.1, 0.5, 0.9], cv=CV_FOLDS, max_iter=10_000, random_state=RANDOM_STATE),
        'GradientBoosting': GradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.05,
            n_estimators=100,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
    }


# ---------------------------------------------------------------------------
# Optuna 하이퍼파라미터 튜닝
# ---------------------------------------------------------------------------

def tune_with_optuna(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    cv: int = CV_FOLDS,
    n_trials: int = 100,
) -> tuple[dict, dict]:
    """
    Ridge / Lasso / ElasticNet / GradientBoosting 하이퍼파라미터를 Optuna로 튜닝하고
    (best_params, best_values) 를 반환한다.
    best_values 를 Train MAPE 로 직접 사용해 선택 편향을 방지한다.
    """

    def objective_ridge(trial):
        alpha = trial.suggest_float('alpha', 1e-4, 1e4, log=True)
        scores = cross_validate(
            Ridge(alpha=alpha), X_train_s, y_train, cv=cv,
            scoring='neg_mean_absolute_percentage_error',
        )
        return float(-scores['test_score'].mean() * 100)

    def objective_lasso(trial):
        alpha = trial.suggest_float('alpha', 1e-4, 1e4, log=True)
        scores = cross_validate(
            Lasso(alpha=alpha, max_iter=10_000, random_state=RANDOM_STATE),
            X_train_s, y_train, cv=cv,
            scoring='neg_mean_absolute_percentage_error',
        )
        return float(-scores['test_score'].mean() * 100)

    def objective_elasticnet(trial):
        alpha    = trial.suggest_float('alpha',    1e-4, 1e4,  log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.01, 0.99)
        scores = cross_validate(
            ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                       max_iter=10_000, random_state=RANDOM_STATE),
            X_train_s, y_train, cv=cv,
            scoring='neg_mean_absolute_percentage_error',
        )
        return float(-scores['test_score'].mean() * 100)

    def objective_gbm(trial):
        params = dict(
            max_depth    =trial.suggest_int  ('max_depth',     2, 5),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            n_estimators =trial.suggest_int  ('n_estimators',  50, 300),
            subsample    =trial.suggest_float('subsample',     0.5, 1.0),
            random_state =RANDOM_STATE,
        )
        scores = cross_validate(
            GradientBoostingRegressor(**params),
            X_train_s, y_train, cv=cv,
            scoring='neg_mean_absolute_percentage_error',
        )
        return float(-scores['test_score'].mean() * 100)

    objectives = {
        'Ridge'           : objective_ridge,
        'Lasso'           : objective_lasso,
        'ElasticNet'      : objective_elasticnet,
        'GradientBoosting': objective_gbm,
    }

    best_params: dict = {}
    best_values: dict = {}
    for model_name, objective in objectives.items():
        print(f"  [{model_name}] Optuna 튜닝 중... (n_trials={n_trials})")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best_params[model_name] = study.best_params
        best_values[model_name] = study.best_value
        print(f"  [{model_name}] 최적 파라미터: {study.best_params}  CV MAPE: {study.best_value:.2f}%")

    return best_params, best_values


def build_models_tuned(best_params: dict) -> dict:
    """Optuna 튜닝 결과로 모든 모델을 생성한다."""
    ridge_p = best_params.get('Ridge', {})
    lasso_p = best_params.get('Lasso', {})
    en_p    = best_params.get('ElasticNet', {})
    gbm_p   = best_params.get('GradientBoosting', {})

    return {
        'Ridge': Ridge(
            alpha=ridge_p.get('alpha', 1.0),
        ),
        'Lasso': Lasso(
            alpha=lasso_p.get('alpha', 1.0),
            max_iter=10_000,
            random_state=RANDOM_STATE,
        ),
        'ElasticNet': ElasticNet(
            alpha=en_p.get('alpha', 1.0),
            l1_ratio=en_p.get('l1_ratio', 0.5),
            max_iter=10_000,
            random_state=RANDOM_STATE,
        ),
        'GradientBoosting': GradientBoostingRegressor(
            max_depth    =gbm_p.get('max_depth',     3),
            learning_rate=gbm_p.get('learning_rate', 0.05),
            n_estimators =gbm_p.get('n_estimators',  100),
            subsample    =gbm_p.get('subsample',     0.8),
            random_state =RANDOM_STATE,
        ),
    }


# ---------------------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------------------

def main():
    feat_list = FEATURES_V2

    print("=" * 60)
    print("Battery Cycle Life Prediction — Training Pipeline")
    print(f"  피처 ({len(feat_list)}개): {feat_list}")
    print("=" * 60)

    # --- 1. 데이터 로드 ---
    print("\n[1/5] 데이터 로드 중...")
    print(f"  Batch 1: {os.path.basename(BATCH1_PATH)}")
    batch1 = load_batch(BATCH1_PATH)
    print(f"  Batch 2: {os.path.basename(BATCH2_PATH)}")
    batch2 = load_batch(BATCH2_PATH)
    batch3 = None
    b3_path = getattr(preprocess, 'BATCH3_PATH', None)
    if isinstance(b3_path, str):
        try:
            print(f"  Batch 3: {os.path.basename(b3_path)}")
            batch3 = load_batch(b3_path)
        except FileNotFoundError:
            batch3 = None

    # --- 2. 피처 행렬 생성 ---
    print("\n[2/5] 피처 행렬 생성 중...")
    df_b1 = build_feature_matrix_v2(batch1, batch_id=1)
    df_b2 = build_feature_matrix_v2(batch2, batch_id=2)
    df_b3 = build_feature_matrix_v2(batch3, batch_id=3) if batch3 is not None else None

    # NaN 행 제거 (신규 피처 계산 실패 셀 방어)
    df_b1 = df_b1.dropna(subset=feat_list).reset_index(drop=True)
    df_b2 = df_b2.dropna(subset=feat_list).reset_index(drop=True)
    if df_b3 is not None:
        df_b3 = df_b3.dropna(subset=feat_list).reset_index(drop=True)

    print(f"  Batch 1 피처 행렬: {df_b1.shape}  (셀 수: {len(df_b1)})")
    print(f"  Batch 2 피처 행렬: {df_b2.shape}  (셀 수: {len(df_b2)})")
    if df_b3 is not None:
        print(f"  Batch 3 피처 행렬: {df_b3.shape}  (셀 수: {len(df_b3)})")
    print(f"  최종 피처 ({len(feat_list)}개): {feat_list}")

    # --- 3. Hold-out 분리 ---
    print("\n[3/5] Hold-out 분리 (Batch 1, test_size=0.2) ...")
    X_all = df_b1[feat_list].values
    y_all = df_b1[TARGET].values

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_all, y_all,
        test_size=HOLD_OUT_RATIO,
        random_state=RANDOM_STATE,
    )
    X_test_b2 = df_b2[feat_list].values
    y_test_b2 = df_b2[TARGET].values
    X_test_b3 = df_b3[feat_list].values if df_b3 is not None else None
    y_test_b3 = df_b3[TARGET].values if df_b3 is not None else None

    msg = f"  Train: {len(X_train)}셀  |  Valid: {len(X_valid)}셀  |  Test(Batch 2): {len(X_test_b2)}셀"
    if X_test_b3 is not None:
        msg += f"  |  Test(Batch 3): {len(X_test_b3)}셀"
    print(msg)

    # --- 4. 스케일링 (X_train 기준 fit, 누수 방지) ---
    print("\n[4/5] StandardScaler 적용 (fit: X_train only) ...")
    scaler = StandardScaler()
    CLIP_Z = 5.0  # 배치간 분포 이탈로 인한 극단 z-score 방어
    X_train_s    = np.clip(scaler.fit_transform(X_train), -CLIP_Z, CLIP_Z)
    X_valid_s    = np.clip(scaler.transform(X_valid),     -CLIP_Z, CLIP_Z)
    X_test_b2_s  = np.clip(scaler.transform(X_test_b2),  -CLIP_Z, CLIP_Z)
    X_test_b3_s  = np.clip(scaler.transform(X_test_b3), -CLIP_Z, CLIP_Z) if X_test_b3 is not None else None

    # --- 4.5 Optuna 하이퍼파라미터 튜닝 (선형 모델만) ---
    print("\n[Optuna] 하이퍼파라미터 튜닝 중 (Ridge / Lasso / ElasticNet / GBM)...")
    best_params, best_values = tune_with_optuna(X_train_s, y_train, n_trials=100)

    # --- 5. 모델 학습 및 평가 ---
    print("\n[5/5] 모델 학습 및 평가 중...")
    models  = build_models_tuned(best_params)
    results = []

    for name, model in models.items():
        # Train: Optuna CV MAPE (선형) or 별도 CV (GBM)
        if name in best_values:
            train_mape = best_values[name]   # 선택 편향 없이 Optuna best_value 재사용
        else:
            train_mape = cv_mape(model, X_train_s, y_train, cv=CV_FOLDS)

        # Valid: Hold-out MAPE
        model.fit(X_train_s, y_train)
        valid_mape = compute_mape(y_valid, model.predict(X_valid_s))

        # Test: Batch 2 MAPE (단 1회)
        test_b2_mape = compute_mape(y_test_b2, model.predict(X_test_b2_s))

        # Test: Batch 3 MAPE (단 1회)
        test_b3_mape = None
        if X_test_b3_s is not None and y_test_b3 is not None:
            test_b3_mape = compute_mape(y_test_b3, model.predict(X_test_b3_s))

        # Lasso, ElasticNet: 선택된 피처 출력
        if name in ['Lasso', 'ElasticNet'] and hasattr(model, 'coef_'):
            nonzero = [f for f, c in zip(feat_list, model.coef_) if abs(c) > 1e-6]
            print(f"\n  [{name} 선택 피처]: {nonzero} (총 {len(nonzero)}개)")

        print_report(name, train_mape, valid_mape, test_b2_mape, test_b3_mape)

        results.append({
            'feature_version':    'v2_optuna',
            'model_name':         name,
            'train_cv_mape':      round(train_mape, 2),
            'valid_mape':         round(valid_mape,  2),
            'test_b2_mape':       round(test_b2_mape,   2),
            'test_b3_mape':       round(test_b3_mape,   2) if test_b3_mape is not None else np.nan,
            'gap_train_valid':    round(train_mape - valid_mape, 2),
            'gap_valid_test_b2':  round(valid_mape  - test_b2_mape,  2),
            'gap_target_test_b2': round(test_b2_mape   - TARGET_MAPE, 2),
            'gap_b2_b3':          round(test_b2_mape - test_b3_mape, 2) if test_b3_mape is not None else np.nan,
            'gap_target_test_b3': round(test_b3_mape - TARGET_MAPE, 2) if test_b3_mape is not None else np.nan,
        })

    # --- 결과 저장 (기존 파일에 append) ---
    df_new = pd.DataFrame(results)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    if os.path.exists(RESULTS_PATH):
        df_old = pd.read_csv(RESULTS_PATH)
        df_results = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_results = df_new
    df_results.to_csv(RESULTS_PATH, index=False)
    print(f"\n결과 저장 완료 (append): {RESULTS_PATH}")

    # --- 최종 모델 선택 권고 ---
    print("\n" + "=" * 60)
    print("최종 모델 선택 기준: Valid MAPE 최소 [피처 버전: v2]")
    best_idx  = df_new['valid_mape'].idxmin()
    best_name = df_new.loc[best_idx, 'model_name']
    best_mape = df_new.loc[best_idx, 'valid_mape']

    ridge_rows = df_new[df_new['model_name'] == 'Ridge']
    if not ridge_rows.empty:
        ridge_mape = ridge_rows['valid_mape'].values[0]
        gap_pp = best_mape - ridge_mape
        print(f"  Valid MAPE 최소 모델: {best_name} ({best_mape:.2f}%)")
        if best_name != 'Ridge' and abs(gap_pp) <= 5.0:
            print(f"  → Ridge와 차이 {abs(gap_pp):.2f}%p ≤ 5%p : 해석 가능성 우선 → Ridge 권고")
        else:
            print(f"  → 최종 권고 모델: {best_name}")

    print("\n전체 결과 (이번 실행):")
    print(df_new.to_string(index=False))
    print("=" * 60)


if __name__ == '__main__':
    main()