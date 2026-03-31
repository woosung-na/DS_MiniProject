"""
데이터 전처리 스크립트

배터리 .mat 파일을 로드하고, 사이클별 long-format DataFrame으로 변환한다.
"""

import os
import numpy as np
import pandas as pd

try:
    import mat73
    HAS_MAT73 = True
except ImportError:
    HAS_MAT73 = False

import scipy.io as sio


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'archive')

BATCH1_PATH = os.path.join(DATA_DIR, '2017-05-12_batchdata_updated_struct_errorcorrect.mat')
BATCH2_PATH = os.path.join(DATA_DIR, '2018-02-20_batchdata_updated_struct_errorcorrect.mat')
BATCH3_PATH = os.path.join(DATA_DIR, '2018-04-12_batchdata_updated_struct_errorcorrect.mat')


def _dict_of_lists_to_list_of_dicts(d) -> list:
    """
    mat73이 반환하는 dict-of-lists 구조를 list-of-dicts로 변환한다.
    예: {'cycle_life': [v0, v1, ...], 'policy': [p0, p1, ...]}
        → [{'cycle_life': v0, 'policy': p0}, {'cycle_life': v1, 'policy': p1}, ...]
    """
    if not isinstance(d, dict):
        return list(d)
    keys = list(d.keys())
    n = len(d[keys[0]])
    return [{k: d[k][i] for k in keys} for i in range(n)]


def load_batch(path: str) -> list:
    """
    .mat 파일을 로드하여 배터리 셀 리스트를 반환한다.

    Parameters
    ----------
    path : str
        .mat 파일 경로

    Returns
    -------
    list of dict
        각 원소는 하나의 배터리 셀을 나타내는 dict.
        주요 키: cycle_life, cycles, summary, Vdlin, policy
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")

    if HAS_MAT73:
        try:
            data = mat73.loadmat(path)
            print(f"  로드 완료 (MATLAB v7.3 / HDF5): {os.path.basename(path)}")
            # mat73은 batch를 dict-of-lists로 반환 → list-of-dicts로 변환
            return _dict_of_lists_to_list_of_dicts(data['batch'])
        except Exception:
            pass

    data = sio.loadmat(path, simplify_cells=True)
    print(f"  로드 완료 (MATLAB v7.2 이하): {os.path.basename(path)}")
    return list(data['batch'])


def _get_summary_array(cell: dict, key: str) -> np.ndarray:
    """
    셀의 summary에서 특정 키의 배열을 추출한다.
    mat73은 dict-of-lists, scipy는 ndarray를 반환하므로 두 형식 모두 처리한다.
    """
    summary = cell['summary']
    val = summary[key]
    return np.array(val).flatten()


def batch_to_summary_df(batch: list, batch_id: int = 1) -> pd.DataFrame:
    """
    배터리 셀 리스트를 사이클별 long-format DataFrame으로 변환한다.

    Parameters
    ----------
    batch : list of dict
        load_batch()가 반환한 셀 리스트
    batch_id : int
        배치 식별자 (Batch 1 → 1, Batch 2 → 2)

    Returns
    -------
    pd.DataFrame
        컬럼: cell_id, batch_id, cycle, cycle_life, QD, IR, Tavg, Tmax, chargetime
    """
    records = []

    for cell_idx, cell in enumerate(batch):
        raw_life = np.array(cell['cycle_life']).flatten()[0]
        if np.isnan(raw_life):
            continue  # 실험 미완료 셀 스킵
        cycle_life = int(raw_life)

        qd = _get_summary_array(cell, 'QDischarge')
        ir = _get_summary_array(cell, 'IR')
        tavg = _get_summary_array(cell, 'Tavg')
        tmax = _get_summary_array(cell, 'Tmax')
        chargetime = _get_summary_array(cell, 'chargetime')

        n_cycles = len(qd)

        for c in range(n_cycles):
            records.append({
                'cell_id':     cell_idx,
                'batch_id':    batch_id,
                'cycle':       c + 1,
                'cycle_life':  cycle_life,
                'QD':          float(qd[c]),
                'IR':          float(ir[c]) if c < len(ir) else np.nan,
                'Tavg':        float(tavg[c]) if c < len(tavg) else np.nan,
                'Tmax':        float(tmax[c]) if c < len(tmax) else np.nan,
                'chargetime':  float(chargetime[c]) if c < len(chargetime) else np.nan,
            })

    return pd.DataFrame(records)


if __name__ == '__main__':
    print("=== preprocess.py 단독 실행 테스트 ===")
    print(f"\nBatch 1 로드 중: {BATCH1_PATH}")
    batch1 = load_batch(BATCH1_PATH)
    print(f"  셀 수: {len(batch1)}")

    df = batch_to_summary_df(batch1, batch_id=1)
    print(f"  summary DataFrame shape: {df.shape}")
    print(f"  컬럼: {list(df.columns)}")
    print(f"  셀별 사이클 수:\n{df.groupby('cell_id')['cycle'].count().describe()}")
    print(f"\n  cycle_life 분포:")
    lives = df.drop_duplicates('cell_id')['cycle_life']
    print(f"    min={lives.min()}, max={lives.max()}, mean={lives.mean():.1f}")
