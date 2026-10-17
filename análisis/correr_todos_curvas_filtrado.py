#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
correr_todos_curvas_filtrado.py

Genera:
  --out-dir : los mismos 12 PNG que correr_todos_curvas.py (6 diff_<angulo>.png
              + 6 diff_promedio_pacientes_<angulo>.png), pero calculados
              solo con los walkings que NO superaron el umbral.
  --out-csv : una fila por paciente/walking/lado/angulo, IGUAL que antes,
              pero con una columna extra "excluido_umbral" (True/False)
              para que quede registro de que se saco y que no, sin perder
              el dato.
  --out     : log de texto, incluye al final el detalle de que walkings
              se excluyeron en que angulo/lado y por que RMSE.
"""

import argparse
import csv
import logging
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comparar_hh_score as chs
import correr_todos_curvas as ctc


def recolectar_patient(carpeta_paciente, registros, log):
    """
    Junta, para cada angulo/lado, un registro por walking encontrado en la
    carpeta de este paciente (sin filtrar todavia por umbral). Cada
    registro es un dict con paciente, walking, diff, hh, score, rmse.
    Devuelve la cantidad de walkings que aportaron al menos un dato.
    """
    pares = chs.encontrar_pares(carpeta_paciente)
    if not pares:
        log.warning(f'  {carpeta_paciente.name}: no se encontraron pares _hh/_score.emt')
        return 0

    walkings_ok = 0
    for par in pares:
        base = par["base"]
        paciente, walking = chs.id_paciente_walking(base)

        try:
            n_hh, data_hh = chs.parse_emt(par["hh"])
            n_score, data_score = chs.parse_emt(par["score"])
        except Exception as e:
            log.error(f'  {carpeta_paciente.name} / {base}: ERROR al parsear ({e})')
            continue

        log.info(f'  {carpeta_paciente.name} / {base}: hh {n_hh} ciclos, score {n_score} ciclos')
        aporto_algo = False

        for angulo in chs.ANGULOS:
            for lado, col in (("R", angulo["col_R"]), ("L", angulo["col_L"])):
                curva_hh = chs.curva_media(data_hh, col)
                curva_score = chs.curva_media(data_score, col)
                if curva_hh is None or curva_score is None:
                    continue

                diff = curva_hh - curva_score
                if np.all(np.isnan(diff)):
                    continue

                aporto_algo = True
                rmse = float(np.sqrt(np.nanmean(diff ** 2)))
                registros[angulo["id"]][lado].append({
                    "paciente": paciente,
                    "walking": walking,
                    "diff": diff,
                    "hh": curva_hh,
                    "score": curva_score,
                    "rmse": rmse,
                })

        if aporto_algo:
            walkings_ok += 1

    return walkings_ok


def filtrar_y_armar_acumuladores(registros, umbral_rmse, log):
    diffs = {a["id"]: {"R": [], "L": []} for a in chs.ANGULOS}
    curvas_hh = {a["id"]: {"R": [], "L": []} for a in chs.ANGULOS}
    curvas_score = {a["id"]: {"R": [], "L": []} for a in chs.ANGULOS}
    diffs_por_paciente = {a["id"]: {"R": {}, "L": {}} for a in chs.ANGULOS}
    filas_csv = []

    for angulo in chs.ANGULOS:
        for lado in ("R", "L"):
            lista = registros[angulo["id"]][lado]
            excluidos = [r for r in lista if r["rmse"] > umbral_rmse]
            incluidos = [r for r in lista if r["rmse"] <= umbral_rmse]

            if excluidos:
                log.warning(f'{angulo["id"]} / {lado}: se excluyeron {len(excluidos)} walking(s) por RMSE > {umbral_rmse} grados:')
                for r in sorted(excluidos, key=lambda r: r["rmse"], reverse=True):
                    log.warning(f'    {r["paciente"]} / {r["walking"]}: RMSE = {r["rmse"]:.2f} grados')

            for r in incluidos:
                diffs[angulo["id"]][lado].append(r["diff"])
                curvas_hh[angulo["id"]][lado].append(r["hh"])
                curvas_score[angulo["id"]][lado].append(r["score"])
                diffs_por_paciente[angulo["id"]][lado].setdefault(r["paciente"], []).append(r["diff"])

            for r in lista:
                diff = r["diff"]
                media_diff = float(np.nanmean(diff))
                idx_max = int(np.nanargmax(np.abs(diff)))
                max_abs = float(diff[idx_max])
                filas_csv.append({
                    "paciente": r["paciente"],
                    "walking": r["walking"],
                    "lado": lado,
                    "angulo": angulo["id"],
                    "rmse_deg": round(r["rmse"], 3),
                    "diff_media_deg": round(media_diff, 3),
                    "diff_max_abs_deg": round(max_abs, 3),
                    "frame_max_abs": idx_max,
                    "excluido_umbral": r["rmse"] > umbral_rmse,
                })

    return diffs, curvas_hh, curvas_score, diffs_por_paciente, filas_csv


def main(carpeta, out_dir, out_csv, out_log, prefijo_excluido='no', umbral_rmse=15.0):
    carpeta = Path(carpeta)
    if not carpeta.exists():
        raise FileNotFoundError(f'No se encontro la carpeta: {carpeta}')

    Path(out_log).resolve().parent.mkdir(parents=True, exist_ok=True)
    Path(out_csv).resolve().parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(out_log, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

    logger = logging.getLogger('correr_todos_curvas_filtrado')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    subcarpetas = sorted(
        p for p in carpeta.iterdir()
        if p.is_dir() and not p.name.lower().startswith(prefijo_excluido.lower())
    )

    logger.info('')
    logger.info(f'===== Encontradas {len(subcarpetas)} carpetas de pacientes en {carpeta} =====')
    logger.info(f'Umbral de exclusion: RMSE > {umbral_rmse} grados (por angulo y lado)')

    registros = {a["id"]: {"R": [], "L": []} for a in chs.ANGULOS}

    ok, fallidos, sin_datos = [], [], []
    for sub in subcarpetas:
        logger.info('')
        logger.info(f'########## Paciente: {sub.name} ##########')
        try:
            n_walkings = recolectar_patient(sub, registros, logger)
            if n_walkings > 0:
                ok.append(sub.name)
            else:
                sin_datos.append(sub.name)
        except Exception as e:
            logger.error(f'FALLO en {sub.name}: {e}')
            logger.error(traceback.format_exc())
            fallidos.append(sub.name)
            continue

    logger.info('')
    logger.info('===== Procesamiento completo =====')
    logger.info(f'OK ({len(ok)}): {", ".join(ok) if ok else "-"}')
    logger.info(f'SIN DATOS ({len(sin_datos)}): {", ".join(sin_datos) if sin_datos else "-"}')
    logger.info(f'FALLIDOS ({len(fallidos)}): {", ".join(fallidos) if fallidos else "-"}')

    logger.info('')
    logger.info('===== Filtrado por umbral de RMSE =====')
    diffs, curvas_hh, curvas_score, diffs_por_paciente, filas_csv = filtrar_y_armar_acumuladores(
        registros, umbral_rmse, logger
    )

    if filas_csv:
        campos = ["paciente", "walking", "lado", "angulo", "rmse_deg",
                  "diff_media_deg", "diff_max_abs_deg", "frame_max_abs", "excluido_umbral"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(filas_csv)
        logger.info(f'Tabla CSV guardada en: {out_csv}')
    else:
        logger.warning('No se genero CSV: ningun paciente aporto datos validos.')

    ctc.graficar(diffs, curvas_hh, curvas_score, out_dir, logger)
    ctc.graficar_por_paciente(diffs_por_paciente, out_dir, logger)

    print(f'\nListo. Log en: {out_log}')
    print(f'CSV en: {out_csv}')
    print(f'Graficos en: {out_dir}')
    print(f'OK: {len(ok)}  |  Sin datos: {len(sin_datos)}  |  Fallidos: {len(fallidos)}')
    if fallidos:
        print(f'Fallidos: {", ".join(fallidos)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Como correr_todos_curvas.py, pero excluyendo walkings con RMSE alto (por angulo/lado) antes de graficar'
    )
    parser.add_argument('--carpeta', required=True,
                         help='Carpeta que contiene las subcarpetas de cada paciente')
    parser.add_argument('--out-dir', required=True,
                         help='Carpeta de salida para los 12 PNG')
    parser.add_argument('--out-csv', required=True,
                         help='Archivo .csv con el resumen (incluye columna excluido_umbral)')
    parser.add_argument('--out', required=True, dest='out_log',
                         help='Archivo de texto con el log detallado')
    parser.add_argument('--prefijo-excluido', default='no',
                         help='Prefijo de carpetas a saltear, sin importar mayus/minus (default "no")')
    parser.add_argument('--umbral-rmse', type=float, default=15.0,
                         help='RMSE en grados por encima del cual se excluye un walking de un angulo/lado (default 15)')
    args = parser.parse_args()

    main(args.carpeta, args.out_dir, args.out_csv, args.out_log, args.prefijo_excluido, args.umbral_rmse)
