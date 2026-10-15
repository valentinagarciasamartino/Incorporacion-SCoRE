#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
delete_tdf.py
Elimina la copia del standing creada por copy_tdf.py

Uso:
    python delete_tdf.py <path_carpeta_paciente>

Ejemplo:
    python delete_tdf.py "C:/My captures/0128~aa"
"""

import sys
import configparser
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def main(folder_path, cfg_path='ScoreFiles/score.cfg'):

    folder_path = Path(folder_path)

    if not folder_path.exists():
        log.error(f'Carpeta no encontrada: {folder_path}')
        sys.exit(1)

    # Leer configuracion
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    nombre_copia = cfg['archivos']['standing_copia']

    # Encontrar la copia
    copia_files = list(folder_path.glob(f'*{nombre_copia}.tdf'))
    if len(copia_files) != 1:
        log.error(f'Se esperaba 1 copia, se encontraron {len(copia_files)}')
        sys.exit(1)

    # Eliminar
    copia_files[0].unlink()
    log.info(f'Eliminado: {copia_files[0].name}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Uso: python delete_tdf.py <path_carpeta_paciente>')
        print('Ejemplo: python delete_tdf.py "C:/My captures/0128~aa"')
        sys.exit(1)

    folder = sys.argv[1]
    cfg    = sys.argv[2] if len(sys.argv) > 2 else 'score.cfg'
    main(folder, cfg)