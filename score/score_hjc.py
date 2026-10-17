import shutil
import sys
import logging
import numpy as np
from pathlib import Path
from score_config import ScoreConfig
from scipy.signal import butter, filtfilt
from basictdf import Tdf
from extraFunctions import rigidbody_OCST, SCORE_SARA
from Add_marker import add_marker_to_tdf

# ===================== LOGGING ====================== #
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def _fail(msg):
    log.error(msg)
    raise RuntimeError(msg)


# ===================== LEER CONFIGURACION ====================== #
def build_maps(cfg):
    """
    Construye los mapas de nombres a partir del .cfg
    Devuelve standing_map_R, standing_map_L, funcional_map
    """
    standing_map_R = {
        cfg['standing_R']['thigh'] : 'A',
        cfg['standing_R']['clup']  : 'B',
        cfg['standing_R']['cldown']: 'C',
        #cfg['standing_R']['knee']  : 'D',
        cfg['standing_R']['sacrum']: 'E',
        cfg['standing_R']['r_asis']: 'F',
        cfg['standing_R']['l_asis']: 'G',
    }

    standing_map_L = {
        cfg['standing_L']['thigh'] : 'A',
        cfg['standing_L']['clup']  : 'B',
        cfg['standing_L']['cldown']: 'C',
        #cfg['standing_L']['knee']  : 'D',
        cfg['standing_L']['sacrum']: 'E',
        cfg['standing_L']['r_asis']: 'F',
        cfg['standing_L']['l_asis']: 'G',
    }

    funcional_map = {
        cfg['funcional']['thigh'] : 'A',
        cfg['funcional']['clup']  : 'B',
        cfg['funcional']['cldown']: 'C',
        #cfg['funcional']['knee']  : 'D',
        cfg['funcional']['sacrum']: 'E',
        cfg['funcional']['r_asis']: 'F',
        cfg['funcional']['l_asis']: 'G',
    }

    return standing_map_R, standing_map_L, funcional_map


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


#ABSTRACT_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
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


def calculate_hjc(standing, func_path, standing_map, funcional_map,
                  cluster_marker, virtual_marker,
                  max_gap, butter_cutoff, butter_fs, butter_order, side):
    """
    Calcula el HJC para un lado dado usando SCoRE, lo expresa
    en el sistema local de la pelvis y ajusta la orientación/signos
    para corregir la inversión derecha/izquierda y alinear con Davis.
    """
    log.info(f'Calculando HJC lado {side}...')

    # Cargar funcional
    func_raw = load_tdf(func_path)
    func_prep = prepare(func_raw, max_gap)
    func_filt = {k: butterworth_filter(v, butter_cutoff, butter_fs, butter_order)
                 for k, v in func_prep.items()}

    validate_marker_map(func_filt, funcional_map, func_path.name, side)
    func = rename(func_filt, funcional_map)
    func = require_abstract(func, f'funcional {func_path.name}')

    validate_marker_map(standing, standing_map, 'standing', side)
    standing_side = rename(standing, standing_map)
    standing_side = require_abstract(standing_side, f'standing lado {side}')

    # =========================================================================
    # CONSTRUCCIÓN DEL SISTEMA DE REFERENCIA PÉLVICO LOCAL (ESTÁTICO)
    # =========================================================================
    # En el mapa abstracto: 'F' es r_asis, 'G' es l_asis, 'E' es sacrum
    asis_r = np.mean(standing_side['F'], axis=1)
    asis_l = np.mean(standing_side['G'], axis=1)
    sacrum = np.mean(standing_side['E'], axis=1)

    # 1. Origen: Punto medio entre las ASIS
    origin = (asis_r + asis_l) / 2.0
    
    # 2. Eje mediolateral (Y local) - Invertimos el orden (l_asis - asis_r o ajustando signo)
    # para corregir el espejo de derecha/izquierda
    y_axis = asis_l - asis_r  # Ajustado para alinear correctamente la lateralidad
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # 3. Vector hacia el sacro para definir el plano
    v_sacrum = sacrum - origin
    
    # 4. Eje anteroposterior (X local) ortogonal
    x_axis = np.cross(y_axis, v_sacrum)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # 5. Eje vertical (Z local) producto vectorial final
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Matriz de rotación de la pelvis estática (Global a Local)
    R_pelvis_stat = np.column_stack((x_axis, y_axis, z_axis))
    # =========================================================================

    # OCST
    new_markers, R_OCST, d_OCST = rigidbody_OCST(
        func, standing_side, '2', cluster_marker, virtual_marker
    )

    # SCoRE
    sco, sara, err = SCORE_SARA(R_OCST, d_OCST, ['1', '2'])

    H_thigh = sco[0:3]
    H_pelvis_global = sco[3:6]  # Coordenada devuelta por SCoRE en global

    # Transformar el HJC global al sistema local de la pelvis
    H_pelvis_local_raw = np.dot(R_pelvis_stat.T, (H_pelvis_global - origin))

    # Factor de corrección final de polaridad para X y Z
    H_pelvis_local = H_pelvis_local_raw * np.array([1, 1, -1])

    log.info(f'HJC {side} muslo (global): {np.round(H_thigh, 4)} m')
    log.info(f'HJC {side} pelvis (local): {np.round(H_pelvis_local, 4)} m (Ehrig err: {err:.4f})')

    return H_thigh, H_pelvis_local
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

    # Construir mapas desde cfg
    standing_map_R, standing_map_L, funcional_map = build_maps(cfg)

    # clústeres (fijos — la estructura no cambia)
    #cluster_marker = {0: ['A', 'B', 'C', 'D'], 1: ['E', 'F', 'G']}
    cluster_marker = {0: ['A', 'B', 'C'], 1: ['E', 'F', 'G']}
    virtual_marker = {0: ['n'], 1: ['n']}

    # Encontrar la copia del standing creada por copiar_tdf.py
    nombre_copia = cfg['archivos']['standing_copia'] 
    standing_files = list(folder_path.glob(f'*{nombre_copia}.tdf')) #busco el standing con el nombre que defini en el cfg 
    if len(standing_files) != 1:
        _fail(f'Se esperaba 1 standing copia, se encontraron {len(standing_files)}')
    file_standing = standing_files[0]

    # Encontrar el standing que se usa para CALCULAR (puede ser distinto al de grabado)
    nombre_calculo = cfg['archivos']['standing_calculo']
    calculo_files = list(folder_path.glob(f'*{nombre_calculo}.tdf'))
    if len(calculo_files) != 1:
        _fail(f'Se esperaba 1 standing de calculo, se encontraron {len(calculo_files)}')
    file_standing_calculo = calculo_files[0]

    # Lo cargo
    log.info('Cargando standing para el calculo...')
    standing_raw = load_tdf(file_standing_calculo)
    standing     = {k: np.nan_to_num(v) for k, v in standing_raw.items()}

    # Construir paths funcionales
    # Deduce el prefijo del paciente desde el nombre del standing (ej: 0128~aa)
    prefix = '~'.join(file_standing.stem.split('~')[:2])
    func_path_R = folder_path / f'{prefix}~{file_func_R}.tdf'
    func_path_L = folder_path / f'{prefix}~{file_func_L}.tdf'

    if not func_path_R.exists():
        _fail(f'No se encontro funcional R: {func_path_R}')
    if not func_path_L.exists():
        _fail(f'No se encontro funcional L: {func_path_L}')

    # Calcular HJC (ahora devuelve H_thigh y H_pelvis_local)
    H_thigh_R, H_pelvis_R = calculate_hjc(
        standing, func_path_R, standing_map_R, funcional_map,
        cluster_marker, virtual_marker,
        max_gap, butter_cutoff, butter_fs, butter_order, 'R'
    )

    H_thigh_L, H_pelvis_L = calculate_hjc(
        standing, func_path_L, standing_map_L, funcional_map,
        cluster_marker, virtual_marker,
        max_gap, butter_cutoff, butter_fs, butter_order, 'L'
    )

    # Resultado final en consola
    log.info('===== RESULTADOS FINALES (SCoRE en Marco Pélvico Local) =====')
    log.info(f'HJC R Pelvis Local: {np.round(H_pelvis_R, 5)} m')
    log.info(f'HJC L Pelvis Local: {np.round(H_pelvis_L, 5)} m')

    
    # Escribir las coordenadas locales de la pelvis en el TDF usando H_pelvis_R y H_pelvis_L
    standing_copy = file_standing
    add_marker_to_tdf(str(standing_copy), 'RHPF', H_pelvis_R)
    add_marker_to_tdf(str(standing_copy), 'LHPF', H_pelvis_L)
    log.info('Marcadores HJC (SCoRE pelvis local) escritos en TDF')
    log.info('Pipeline completo')
    
    
    return H_pelvis_R, H_pelvis_L


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
