"""
피처 엔지니어링 스크립트

피처셋:
  dQ_min              : ΔQ(V) = Q(V, c=100) - Q(V, c=10) 의 최솟값  (r=+0.882)
  dQ_mean             : ΔQ(V) 의 평균값                               (r=+0.854)
  dQ_var              : ΔQ(V) 의 분산                                 (r=-0.838)
  dQ_zone_ratio_low   : 저전압 구간 dQ 비중 (비율, VIF 해소)
  dQ_zone_ratio_high  : 고전압 구간 dQ 비중 (비율, VIF 해소)
  mean_chargetime     : 초기 100사이클 평균 충전시간                   (r=+0.577)
  mean_Tavg           : 초기 100사이클 평균 온도                       (r=-0.482)
"""

import numpy as np
import pandas as pd

from preprocess import load_batch, batch_to_summary_df, BATCH1_PATH, BATCH2_PATH

CYCLE_A = 10    # ΔQ 기준 사이클
CYCLE_B = 100   # ΔQ 비교 사이클
EARLY_CYCLES = 100  # summary 피처 계산 기준

FEATURES_V2 = [
    'dQ_min', 'dQ_mean', 'dQ_var',
    'dQ_zone_ratio_low',
    'dQ_zone_ratio_high',
    'mean_chargetime', 'mean_Tavg',
]

TARGET = 'cycle_life'


# ─────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────
def _get_qdlin(cell: dict, cycle_num: int):
    """특정 사이클의 Qdlin 배열을 추출한다."""
    cycles = cell['cycles']
    idx = cycle_num - 1

    if isinstance(cycles, dict):
        qdlin_list = cycles.get('Qdlin', [])
        if idx >= len(qdlin_list):
            return None
        val = qdlin_list[idx]
    else:
        if idx >= len(cycles):
            return None
        val = cycles[idx].get('Qdlin') if isinstance(cycles[idx], dict) else None

    if val is None:
        return None

    arr = np.array(val).flatten()
    return arr if len(arr) > 0 else None


# ─────────────────────────────────────────────────────────────
# ① ΔQ 기반 피처
# ─────────────────────────────────────────────────────────────
def extract_dq_features(batch: list, cycle_a: int = CYCLE_A, cycle_b: int = CYCLE_B) -> pd.DataFrame:
    """
    ΔQ(V) 피처를 추출한다.

    반환 컬럼:
      기본 3개:  dQ_min, dQ_mean, dQ_var
      구간 비율: dQ_zone_ratio_low, dQ_zone_ratio_high
        (low = 전압 하위 1/3, high = 전압 상위 1/3)
    """
    records = []

    for cell_idx, cell in enumerate(batch):
        q_a = _get_qdlin(cell, cycle_a)
        q_b = _get_qdlin(cell, cycle_b)

        if q_a is None or q_b is None:
            print(f"  [경고] cell_id={cell_idx}: cycle {cycle_a} 또는 {cycle_b} Qdlin 누락 — 스킵")
            continue

        min_len = min(len(q_a), len(q_b))
        dq = q_b[:min_len] - q_a[:min_len]

        n = len(dq)
        t1, t2 = n // 3, 2 * (n // 3)

        global_mean = float(np.mean(dq))

        row = {
            'cell_id': cell_idx,
            'dQ_min':  float(np.min(dq)),
            'dQ_mean': global_mean,
            'dQ_var':  float(np.var(dq)),
        }

        if global_mean != 0 and len(dq[:t1]) > 0 and len(dq[t2:]) > 0:
            row['dQ_zone_ratio_low']  = float(np.mean(dq[:t1])) / global_mean
            row['dQ_zone_ratio_high'] = float(np.mean(dq[t2:])) / global_mean
        else:
            row['dQ_zone_ratio_low']  = np.nan
            row['dQ_zone_ratio_high'] = np.nan

        records.append(row)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# ② summary 기반 피처
# ─────────────────────────────────────────────────────────────
def extract_summary_features(summary_df: pd.DataFrame) -> pd.DataFrame:
    """초기 100사이클 summary 기반 피처를 추출한다."""
    early = summary_df[summary_df['cycle'] <= EARLY_CYCLES]

    feat = (
        early
        .groupby('cell_id')
        .agg(
            cycle_life      =('cycle_life',  'first'),
            mean_chargetime =('chargetime',  'mean'),
            mean_Tavg       =('Tavg',        'mean'),
        )
        .reset_index()
    )
    return feat


# ─────────────────────────────────────────────────────────────
# ③ 최종 행렬 빌더
# ─────────────────────────────────────────────────────────────
def build_feature_matrix_v2(batch: list, batch_id: int = 1) -> pd.DataFrame:
    """
    dQ 피처와 summary 피처를 합쳐 최종 피처 행렬을 반환한다.

    반환 컬럼: cell_id, batch_id,
               dQ_min, dQ_mean, dQ_var,
               dQ_zone_ratio_low, dQ_zone_ratio_high,
               mean_chargetime, mean_Tavg,
               cycle_life
    """
    summary_df   = batch_to_summary_df(batch, batch_id=batch_id)
    dq_feat      = extract_dq_features(batch)
    summary_feat = extract_summary_features(summary_df)

    df = dq_feat.merge(summary_feat, on='cell_id', how='inner')
    df.insert(1, 'batch_id', batch_id)
    return df


# ─────────────────────────────────────────────────────────────
# 단독 실행 테스트
# ─────────────────────────────────────────────────────────────
def _vif(X: np.ndarray, i: int) -> float:
    """feature i의 VIF를 numpy lstsq로 계산한다."""
    y = X[:, i]
    X_rest = np.column_stack([np.ones(len(y)), np.delete(X, i, axis=1)])
    y_hat = X_rest @ np.linalg.lstsq(X_rest, y, rcond=None)[0]
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return 1 / (1 - r2) if r2 < 1 else np.inf


if __name__ == '__main__':
    print("=== features.py 단독 실행 테스트 ===")
    batch1 = load_batch(BATCH1_PATH)
    df = build_feature_matrix_v2(batch1, batch_id=1)
    print(f"  shape: {df.shape}, 컬럼: {list(df.columns)}")

    print(f"\n  [cycle_life와의 Pearson 상관]")
    for feat in FEATURES_V2:
        r = df[feat].corr(df[TARGET])
        print(f"    {feat:25s}: r = {r:+.3f}")

    feat_df = df[FEATURES_V2].dropna()

    print(f"\n  [피처 간 Pearson 상관 행렬]")
    print(feat_df.corr().round(3).to_string())

    X_mat = feat_df.values
    print(f"\n  [VIF]  (기준: >10 다중공선성 의심, >30 심각)")
    for i, name in enumerate(FEATURES_V2):
        print(f"    {name:25s}: VIF = {_vif(X_mat, i):.1f}")
