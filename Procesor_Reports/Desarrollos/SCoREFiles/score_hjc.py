#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_hjc.py
Calcula el Hip Joint Center (HJC) usando SCoRE/SARA (Ehrig et al. 2006-2007)
Lee la configuracion desde score.cfg

Uso:
    python score_hjc.py <path_carpeta_paciente>

Ejemplo:
    python score_hjc.py "C:/My captures/0128~aa"

Salida:
    - Imprime HJC R y L en consola
    - Escribe los marcadores HJC en una copia del standing .tdf
"""

import sys
import logging
import numpy as np
from pathlib import Path
from score_config import ScoreConfig
from scipy.signal import butter, filtfilt
from basictdf import Tdf
from extraFunctions import rigidbody_OCST, SCORE_SARA
from Add_marker_valen import add_marker_to_tdf

# ===================== LOGGING ====================== #
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def _fail(msg):
    log.error(msg)
    raise RuntimeError(msg)

# ===================== LEER CONFIGURACION ====================== #
'''
def load_config(cfg_path='score.cfg'):
    """Lee el archivo de configuracion score.cfg"""
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    return cfg
'''


def build_maps(cfg):
    """
    Construye los mapas de nombres a partir del .cfg
    Devuelve funcional_map
    """
    funcional_map = {
        cfg['funcional']['thigh'] : 'A',
        cfg['funcional']['clup']  : 'B',
        cfg['funcional']['cldown']: 'C',
        cfg['funcional']['sacrum']: 'E',
        cfg['funcional']['r_asis']: 'F',
        cfg['funcional']['l_asis']: 'G',
    }

    return funcional_map


# ===================== FUNCIONES ====================== #

def load_tdf(path):
    """Carga markers desde .tdf. Devuelve dict {label: (3, n_frames)}"""
    with Tdf(str(path)) as f:
        data3d = f.data3D
    markers = {}
    for track in data3d.tracks:
        markers[track.label] = track.data.T.astype(float)  # (3, n_frames) NaNs preservados
    log.info(f'Cargado: {Path(path).name}')
    return markers


def rename(markers, name_map):
    """Renombra markers usando el mapa. Solo incluye los que estan en el mapa."""
    return {name_map[k]: v for k, v in markers.items() if k in name_map}

ABSTRACT_LABELS = ['A', 'B', 'C', 'E', 'F', 'G']


def validate_marker_map(markers, name_map, source, side=None):
    """
    Comprueba que el archivo tenga todos los marcadores definidos en name_map.
    name_map: {nombre_en_tdf: etiqueta_abstracta}
    """
    missing = [src for src in name_map if src not in markers]
    if not missing:
        return

    side_txt = f' lado {side}' if side else ''
    detail = ', '.join(f'{name_map[src]!r} ({src!r})' for src in missing)
    available = ', '.join(sorted(markers))
    _fail(
        f'Faltan marcadores en {source}{side_txt}: {detail}. '
        f'Disponibles en {source}: {available}'
    )


def require_abstract(markers, context):
    """Exige las etiquetas abstractas A-G tras renombrar."""
    missing = [label for label in ABSTRACT_LABELS if label not in markers]
    if not missing:
        return {k: markers[k] for k in ABSTRACT_LABELS}

    _fail(f'Faltan etiquetas abstractas en {context}: {", ".join(missing)}')



def interpolate_short_gaps(data, max_gap):
    """Interpola NaN gaps de longitud <= max_gap. Gaps mas largos quedan como NaN."""
    data = data.copy()
    for axis in range(3):
        signal   = data[axis, :]
        nan_mask = np.isnan(signal)
        if not np.any(nan_mask):
            continue
        changes = np.diff(nan_mask.astype(int))
        starts  = np.where(changes == 1)[0] + 1
        ends    = np.where(changes == -1)[0] + 1
        if nan_mask[0]:
            starts = np.concatenate([[0], starts])
        if nan_mask[-1]:
            ends = np.concatenate([ends, [len(signal)]])
        for s, e in zip(starts, ends):
            gap_len = e - s
            if gap_len <= max_gap and s > 0 and e < len(signal):
                y0, y1 = signal[s - 1], signal[e]
                if not np.isnan(y0) and not np.isnan(y1):
                    signal[s:e] = np.linspace(y0, y1, gap_len + 2)[1:-1]
        data[axis, :] = signal
    return data


def prepare(markers, max_gap):
    """Interpola gaps cortos y aplica nan_to_num."""
    return {k: np.nan_to_num(interpolate_short_gaps(v, max_gap))
            for k, v in markers.items()}


def butterworth_filter(data, cutoff, fs, order):
    """Filtro Butterworth zero-phase. data: (3, n_frames)."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    filtered = np.zeros_like(data)
    for axis in range(3):
        try:
            filtered[axis, :] = filtfilt(b, a, data[axis, :])
        except Exception:
            filtered[axis, :] = data[axis, :]
    return filtered


def calculate_hjc(func_path, funcional_map,
                  cluster_marker, virtual_marker,
                  max_gap, butter_cutoff, butter_fs, butter_order, reference_frame, side):
    """
    Calcula el HJC para un lado dado.
    Devuelve H_thigh, H_pelvis, hjc (promedio).
    """
    log.info(f'Calculando HJC lado {side}...')

    # Cargar funcional
    func_raw = load_tdf(func_path)

    # Interpolar gaps y nan_to_num
    func_prep = prepare(func_raw, max_gap)

    # Filtro Butterworth
    func_filt = {k: butterworth_filter(v, butter_cutoff, butter_fs, butter_order)
                 for k, v in func_prep.items()}

    # Renombrar con nombres abstractos
    validate_marker_map(func_filt, funcional_map, func_path.name, side)
    func = rename(func_filt, funcional_map)
    func = require_abstract(func, f'funcional {func_path.name}')

    # Extraer frame de referencia como standing
    standing_raw_frame = {k: v[:, reference_frame:reference_frame+1] for k, v in func_raw.items()}
    validate_marker_map(standing_raw_frame, funcional_map, 'standing', side)
    standing_side = rename(standing_raw_frame, funcional_map)
    standing_side = require_abstract(standing_side, f'standing lado {side}')

    # OCST
    new_markers, R_OCST, d_OCST = rigidbody_OCST(
        func, standing_side, '2', cluster_marker, virtual_marker
    )

    # SCoRE
    sco, sara, err = SCORE_SARA(R_OCST, d_OCST, ['1', '2'])

    H_thigh  = sco[0:3]
    H_pelvis = sco[3:6]
    hjc      = (H_thigh + H_pelvis) / 2

    log.info(f'HJC {side} muslo  : {np.round(H_thigh,  4)} m')
    log.info(f'HJC {side} pelvis : {np.round(H_pelvis, 4)} m')
    log.info(f'HJC {side} prom   : {np.round(hjc,      4)} m  (Ehrig err: {err:.4f})')

    return H_thigh, H_pelvis, hjc


# ===================== MAIN ====================== #

def main(folder_path):
    """
    Funcion principal. Recibe la carpeta del paciente y calcula el HJC.
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        _fail(f'Carpeta no encontrada: {folder_path}')

    # Leer configuracion
    cfg = ScoreConfig().config

    # Parametros
    max_gap      = int(cfg['interpolacion']['max_gap'])
    butter_order  = int(cfg['filtro']['butter_order'])
    butter_cutoff = float(cfg['filtro']['butter_cutoff'])
    butter_fs     = float(cfg['filtro']['butter_fs'])
    file_func_R  = cfg['archivos']['funcional_r']
    file_func_L  = cfg['archivos']['funcional_l']
    reference_frame = int(cfg['archivos']['reference_frame'])

    # Construir mapas desde cfg
    funcional_map = build_maps(cfg)

    # Clusters (fijos — la estructura no cambia)
    #cluster_marker = {0: ['A', 'B', 'C', 'D'], 1: ['E', 'F', 'G']}
    cluster_marker = {0: ['A', 'B', 'C'], 1: ['E', 'F', 'G']}
    virtual_marker = {0: ['n'], 1: ['n']}

    # Construir paths funcionales
    # Deduce el prefijo del paciente desde el nombre del standing (ej: 0128~aa)
    tdf_files = list(folder_path.glob('*.tdf'))
    if not tdf_files:
        _fail(f'No se encontraron archivos .tdf en la carpeta')
    prefix = '~'.join(tdf_files[0].stem.split('~')[:2])

    
    func_path_R = folder_path / f'{prefix}~{file_func_R}.tdf'
    func_path_L = folder_path / f'{prefix}~{file_func_L}.tdf'

    if not func_path_R.exists():
        _fail(f'No se encontro funcional R: {func_path_R}')
    if not func_path_L.exists():
        _fail(f'No se encontro funcional L: {func_path_L}')

    # Calcular HJC
    H_thigh_R, H_pelvis_R, hjc_R = calculate_hjc(
        func_path_R, funcional_map,
        cluster_marker, virtual_marker,
        max_gap, butter_cutoff, butter_fs, butter_order, reference_frame, 'R'
    )

    H_thigh_L, H_pelvis_L, hjc_L = calculate_hjc(
        func_path_L, funcional_map,
        cluster_marker, virtual_marker,
        max_gap, butter_cutoff, butter_fs, butter_order, reference_frame, 'L'
    )

    # Resultado final
    log.info('===== RESULTADOS FINALES =====')
    log.info(f'HJC R: {np.round(hjc_R, 5)} m')
    log.info(f'HJC L: {np.round(hjc_L, 5)} m')

    
    # Los resultados los tengo que grabar en el standing que ya copiamos !
    nombre_copia = cfg['archivos']['standing_copia']
    standing_score_files = list(folder_path.glob(f'*{nombre_copia}.tdf'))
    if len(standing_score_files) != 1:
        _fail(f'No se encontro {nombre_copia}.tdf — correr copy_tdf.py primero')
    standing_copy = standing_score_files[0]

    add_marker_to_tdf(str(standing_copy), 'RHPF', hjc_R)
    add_marker_to_tdf(str(standing_copy), 'LHPF', hjc_L)
    log.info('Marcadores HJC escritos en TDF')
    log.info('Pipeline completo')

    
    return hjc_R, hjc_L


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Uso: python score_hjc.py <path_carpeta_paciente>')
        print('Ejemplo: python score_hjc.py "C:/My captures/0128~aa"')
        sys.exit(1)

    folder = sys.argv[1]
    try:
        main(folder)
    except RuntimeError:
        sys.exit(1)

