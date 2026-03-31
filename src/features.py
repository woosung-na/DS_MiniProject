"""
피처 엔지니어링 스크립트

EDA 결과를 바탕으로 최종 5개 피처를 추출한다.

최종 피처셋:
  - dQ_min      : ΔQ(V) = Q(V, cycle=100) - Q(V, cycle=10) 의 최솟값  (r=0.882)
  - dQ_mean     : ΔQ(V) 의 평균값                                       (r=0.854)
  - dQ_var      : ΔQ(V) 의 분산                                         (r=-0.838)
  - mean_chargetime : 초기 100사이클 평균 충전시간                        (r=0.577)
  - mean_Tavg   : 초기 100사이클 평균 온도                               (r=-0.482)

제외 피처:
  - mean_Tmax   : mean_Tavg와 Pearson r=0.96 (다중공선성)
  - std_QD      : mean_QD와 Spearman r=0.93  (비선형 다중공선성)
  - mean_IR     : cycle_life와 r=0.18         (낮은 상관)
"""

import numpy as np
import pandas as pd

from preprocess import load_batch, batch_to_summary_df, BATCH1_PATH, BATCH2_PATH

CYCLE_A = 10    # ΔQ 기준 사이클
CYCLE_B = 100   # ΔQ 비교 사이클
EARLY_CYCLES = 100  # summary 피처 계산 기준


def _get_qdlin(cell: dict, cycle_num: int):
    """
    특정 사이클의 Qdlin 배열(1000포인트)을 추출한다.

    Parameters
    ----------
    cell : dict
        배터리 셀 dict
    cycle_num : int
        1-indexed 사이클 번호

    Returns
    -------
    np.ndarray or None
        shape=(1000,) 배열. 사이클이 없으면 None.
    """
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


def extract_dq_features(batch: list, cycle_a: int = CYCLE_A, cycle_b: int = CYCLE_B) -> pd.DataFrame:
    """
    ΔQ(V) 피처를 추출한다.
    ΔQ(V) = Q(V, cycle=cycle_b) - Q(V, cycle=cycle_a)

    Parameters
    ----------
    batch : list of dict
    cycle_a : int
        기준 사이클 (기본값 10)
    cycle_b : int
        비교 사이클 (기본값 100)

    Returns
    -------
    pd.DataFrame
        컬럼: cell_id, dQ_min, dQ_mean, dQ_var
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

        records.append({
            'cell_id':  cell_idx,
            'dQ_min':   float(np.min(dq)),
            'dQ_mean':  float(np.mean(dq)),
            'dQ_var':   float(np.var(dq)),
        })

    return pd.DataFrame(records)


def extract_summary_features(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    초기 100사이클 summary 기반 피처를 추출한다.

    Parameters
    ----------
    summary_df : pd.DataFrame
        batch_to_summary_df()가 반환한 long-format DataFrame

    Returns
    -------
    pd.DataFrame
        컬럼: cell_id, cycle_life, mean_chargetime, mean_Tavg
    """
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


def build_feature_matrix(batch: list, batch_id: int = 1) -> pd.DataFrame:
    """
    dQ 피처와 summary 피처를 합쳐 최종 피처 행렬을 반환한다.

    Parameters
    ----------
    batch : list of dict
    batch_id : int

    Returns
    -------
    pd.DataFrame
        컬럼: cell_id, batch_id, dQ_min, dQ_mean, dQ_var,
               mean_chargetime, mean_Tavg, cycle_life
    """
    summary_df = batch_to_summary_df(batch, batch_id=batch_id)

    dq_feat      = extract_dq_features(batch)
    summary_feat = extract_summary_features(summary_df)

    df = dq_feat.merge(summary_feat, on='cell_id', how='inner')
    df.insert(1, 'batch_id', batch_id)

    return df


FEATURES = ['dQ_min', 'dQ_mean', 'dQ_var', 'mean_chargetime', 'mean_Tavg']
TARGET   = 'cycle_life'


if __name__ == '__main__':
    print("=== features.py 단독 실행 테스트 ===")
    print(f"\nBatch 1 로드 중...")
    batch1 = load_batch(BATCH1_PATH)

    print("피처 행렬 생성 중...")
    df = build_feature_matrix(batch1, batch_id=1)

    print(f"\n피처 행렬 shape: {df.shape}")
    print(f"컬럼: {list(df.columns)}")
    print(f"\n피처 통계:\n{df[FEATURES].describe().round(4)}")

    print(f"\ncycle_life와의 Pearson 상관:")
    for feat in FEATURES:
        r = df[feat].corr(df[TARGET])
        print(f"  {feat:20s}: r = {r:+.3f}")

    print(f"\nSpot-check — dQ_min 음수 여부 (열화 패턴):")
    print(f"  dQ_min < 0 인 셀: {(df['dQ_min'] < 0).sum()} / {len(df)}")
