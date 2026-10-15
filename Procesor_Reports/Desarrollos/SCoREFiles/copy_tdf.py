#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
copiar_tdf.py
Crea una copia del standing para ser usada por score_hjc.py

Uso:
    python copiar_tdf.py <path_carpeta_paciente>

Ejemplo:
    python copiar_tdf.py "C:/My captures/0128~aa"
"""

import sys
import shutil
import logging
from pathlib import Path
from score_config import ScoreConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def _fail(msg):
    log.error(msg)
    raise RuntimeError(msg)


def main(folder_path):

    folder_path = Path(folder_path)

    if not folder_path.exists():
        _fail(f'Carpeta no encontrada: {folder_path}')

    # Leer configuracion
    cfg = ScoreConfig()
    nombre_copia = cfg.config['archivos']['standing_copia']

    # Encontrar standing original
    standing_files = list(folder_path.glob('*Standing.tdf'))
    if len(standing_files) != 1:
        _fail(f'Se esperaba 1 standing, se encontraron {len(standing_files)}')
    file_standing = standing_files[0]

    # Crear copia
    standing_copy = folder_path / file_standing.name.replace('Standing', nombre_copia)
    shutil.copyfile(file_standing, standing_copy)
    log.info(f'Copia creada: {standing_copy.name}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Uso: python copiar_tdf.py <path_carpeta_paciente>')
        print('Ejemplo: python copiar_tdf.py "C:/My captures/0128~aa"')
        sys.exit(1)

    folder = sys.argv[1]
    cfg    = sys.argv[2] if len(sys.argv) > 2 else 'score.cfg'
    try:
        main(folder, cfg)
    except RuntimeError:
        sys.exit(1)
