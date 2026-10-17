#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corre comparar_hh_score.py para TODAS las subcarpetas de pacientes de una
carpeta madre, salteando las que arrancan con "no".

Tiene que estar en la MISMA carpeta que comparar_hh_score.py, porque lo
importa como modulo (no le toca nada, solo usa sus funciones).

  --out-dir : 12 PNG en total, dos por angulo:
              - diff_<angulo>.png: el promedio HH, el promedio SCoRE y su
                diferencia, un subplot por lado (R/L). 
              - diff_promedio_pacientes_<angulo>.png: solo la curva de
                diferencia, con su media +- 1 SD 
  --out-csv : una fila por paciente/walking/lado/angulo con RMSE, diferencia
              media y diferencia maxima.
  --out     : log de texto con el detalle paciente por paciente.

Uso:
    python correr_todos_curvas.py --carpeta "/ruta/a/Carpetas de pacientes" \\
        --out-dir graficos --out-csv resumen.csv --out log.txt
"""

import argparse
import csv
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Se importa como modulo: tiene que estar en la misma carpeta.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import comparar_hh_score as chs


def procesar_patient(carpeta_paciente, diffs, curvas_hh, curvas_score, diffs_por_paciente, filas_csv, log):
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
                diffs[angulo["id"]][lado].append(diff)
                curvas_hh[angulo["id"]][lado].append(curva_hh)
                curvas_score[angulo["id"]][lado].append(curva_score)
                diffs_por_paciente[angulo["id"]][lado].setdefault(paciente, []).append(diff)

                rmse = np.sqrt(np.nanmean(diff ** 2))
                media_diff = np.nanmean(diff)
                idx_max = int(np.nanargmax(np.abs(diff)))
                max_abs = diff[idx_max]

                filas_csv.append({
                    "paciente": paciente,
                    "walking": walking,
                    "lado": lado,
                    "angulo": angulo["id"],
                    "rmse_deg": round(float(rmse), 3),
                    "diff_media_deg": round(float(media_diff), 3),
                    "diff_max_abs_deg": round(float(max_abs), 3),
                    "frame_max_abs": idx_max,
                })

        if aporto_algo:
            walkings_ok += 1

    return walkings_ok


def graficar(diffs, curvas_hh, curvas_score, out_dir, log):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = np.arange(chs.N_FRAMES)

    for angulo in chs.ANGULOS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

        for ax, lado in zip(axes, ("R", "L")):
            lista_hh = curvas_hh[angulo["id"]][lado]
            lista_score = curvas_score[angulo["id"]][lado]
            if not lista_hh or not lista_score:
                ax.set_title(f"{lado}: sin datos")
                continue

            media_hh = np.nanmean(np.array(lista_hh), axis=0)
            media_score = np.nanmean(np.array(lista_score), axis=0)
            media_diff = media_hh - media_score
            n = len(lista_hh)

            ax.plot(frames, media_hh, color="#1f77b4", linewidth=2, label="Promedio HH")
            ax.plot(frames, media_score, color="#ff7f0e", linewidth=2, linestyle="--", label="Promedio SCoRE")
            ax.plot(frames, media_diff, color="#2ca02c", linewidth=2, label="Diferencia (HH - SCoRE)")
            ax.axhline(0, color="black", linewidth=0.8, linestyle=":")

            ax.set_title(f"Lado {lado} (n={n} walkings)")
            ax.set_xlabel("% del ciclo de marcha")
            ax.legend(fontsize=8)

        axes[0].set_ylabel("Angulo (grados)")
        fig.suptitle(angulo["titulo"])
        fig.tight_layout()

        out_path = out_dir / f"diff_{angulo['id']}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log.info(f'Guardado: {out_path}')


def graficar_por_paciente(diffs_por_paciente, out_dir, log):
    """
    Para cada angulo: un panel por lado (R y L por separado, no superpuestos).
    En cada panel, primero colapsa los walkings de CADA paciente a una sola
    curva (promedio dentro del paciente), y despues grafica solo la curva
    de diferencia: media entre pacientes +- 1 SD entre pacientes (n =
    cantidad de pacientes, no de walkings).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = np.arange(chs.N_FRAMES)
    color = "#2ca02c"

    for angulo in chs.ANGULOS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

        for ax, lado in zip(axes, ("R", "L")):
            por_paciente = diffs_por_paciente[angulo["id"]][lado]
            if not por_paciente:
                ax.set_title(f"{lado}: sin datos")
                continue

            # una curva por paciente = promedio de sus walkings
            curvas_paciente = [np.nanmean(np.array(lista), axis=0) for lista in por_paciente.values()]
            matriz = np.array(curvas_paciente)  # (n_pacientes, 101)

            media = np.nanmean(matriz, axis=0)
            sd = np.nanstd(matriz, axis=0)
            n_pacientes = matriz.shape[0]

            ax.plot(frames, media, color=color, linewidth=2.5, label="Diferencia (HH - SCoRE)")
            ax.fill_between(frames, media - sd, media + sd, color=color, alpha=0.25, label="+-1 SD entre pacientes")
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

            ax.set_title(f"Lado {lado} (n={n_pacientes} pacientes)")
            ax.set_xlabel("% del ciclo de marcha")
            ax.legend(fontsize=8)

        axes[0].set_ylabel("Diferencia HH - SCoRE (grados)")
        fig.suptitle(angulo["titulo"])
        fig.tight_layout()

        out_path = out_dir / f"diff_promedio_pacientes_{angulo['id']}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log.info(f'Guardado: {out_path}')


def main(carpeta, out_dir, out_csv, out_log, prefijo_excluido='no'):
    carpeta = Path(carpeta)
    if not carpeta.exists():
        raise FileNotFoundError(f'No se encontro la carpeta: {carpeta}')

    # Crea las carpetas de salida si no existen (log, csv, graficos)
    Path(out_log).resolve().parent.mkdir(parents=True, exist_ok=True)
    Path(out_csv).resolve().parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(out_log, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

    logger = logging.getLogger('correr_todos_curvas')
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

    diffs = {a["id"]: {"R": [], "L": []} for a in chs.ANGULOS}
    curvas_hh = {a["id"]: {"R": [], "L": []} for a in chs.ANGULOS}
    curvas_score = {a["id"]: {"R": [], "L": []} for a in chs.ANGULOS}
    diffs_por_paciente = {a["id"]: {"R": {}, "L": {}} for a in chs.ANGULOS}
    filas_csv = []

    ok, fallidos, sin_datos = [], [], []
    for sub in subcarpetas:
        logger.info('')
        logger.info(f'########## Paciente: {sub.name} ##########')
        try:
            n_walkings = procesar_patient(sub, diffs, curvas_hh, curvas_score, diffs_por_paciente, filas_csv, logger)
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

    if filas_csv:
        campos = ["paciente", "walking", "lado", "angulo", "rmse_deg",
                  "diff_media_deg", "diff_max_abs_deg", "frame_max_abs"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(filas_csv)
        logger.info(f'Tabla CSV guardada en: {out_csv}')
    else:
        logger.warning('No se genero CSV: ningun paciente aporto datos validos.')

    graficar(diffs, curvas_hh, curvas_score, out_dir, logger)
    graficar_por_paciente(diffs_por_paciente, out_dir, logger)

    print(f'\nListo. Log en: {out_log}')
    print(f'CSV en: {out_csv}')
    print(f'Graficos en: {out_dir}')
    print(f'OK: {len(ok)}  |  Sin datos: {len(sin_datos)}  |  Fallidos: {len(fallidos)}')
    if fallidos:
        print(f'Fallidos: {", ".join(fallidos)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Corre comparar_hh_score.py para todas las subcarpetas de una carpeta de pacientes'
    )
    parser.add_argument('--carpeta', required=True,
                         help='Carpeta que contiene las subcarpetas de cada paciente')
    parser.add_argument('--out-dir', required=True,
                         help='Carpeta de salida para los 6 PNG agregados')
    parser.add_argument('--out-csv', required=True,
                         help='Archivo .csv con el resumen por paciente/walking/lado/angulo')
    parser.add_argument('--out', required=True, dest='out_log',
                         help='Archivo de texto con el log detallado paciente por paciente')
    parser.add_argument('--prefijo-excluido', default='no',
                         help='Prefijo de carpetas a saltear, sin importar mayus/minus (default "no")')
    args = parser.parse_args()

    main(args.carpeta, args.out_dir, args.out_csv, args.out_log, args.prefijo_excluido)
