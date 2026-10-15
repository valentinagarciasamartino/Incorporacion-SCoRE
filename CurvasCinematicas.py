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

# Tamanos de fuente mas grandes para las figuras finales de comparacion de curvas
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.titlesize': 18,
})

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comparar_hh_score as chs
import correr_todos_curvas_filtrado as ctf


def curvas_ejemplo_paciente(registros, paciente_ejemplo, umbral_rmse):
    """
    Para el paciente de ejemplo, arma el promedio de HH y el promedio de
    SCoRE (a traves de sus propios walkings incluidos, es decir los que no
    superan el umbral de RMSE), por angulo y lado.
    Devuelve dict: angulo_id -> lado -> (media_hh, media_score, n_walkings)
    o None si el paciente no tiene datos incluidos para esa combinacion.
    """
    resultado = {a["id"]: {"R": None, "L": None} for a in chs.ANGULOS}
    for angulo in chs.ANGULOS:
        for lado in ("R", "L"):
            registros_paciente = [
                r for r in registros[angulo["id"]][lado]
                if r["paciente"] == paciente_ejemplo and r["rmse"] <= umbral_rmse
            ]
            if not registros_paciente:
                continue
            media_hh = np.nanmean(np.array([r["hh"] for r in registros_paciente]), axis=0)
            media_score = np.nanmean(np.array([r["score"] for r in registros_paciente]), axis=0)
            resultado[angulo["id"]][lado] = (media_hh, media_score, len(registros_paciente))
    return resultado


def graficar_con_ejemplo(diffs, curvas_hh, curvas_score, diffs_por_paciente,
                          curvas_ejemplo, paciente_ejemplo, out_dir, log):
    """
    Grilla de 2x2 por angulo:
      cuadro 1 (arriba-izq):  diferencia HH-SCoRE, media +-1 SD entre pacientes, lado L
      cuadro 2 (arriba-der):  idem, lado R
      cuadro 3 (abajo-izq):   promedio HH y promedio SCoRE del paciente de ejemplo
                              (todos sus walkings), lado L
      cuadro 4 (abajo-der):   idem, lado R
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = np.arange(chs.N_FRAMES)

    for angulo in chs.ANGULOS:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)

        # --- Fila 1: diferencia, media +- 1 SD entre pacientes (cuadro 1 = L, cuadro 2 = R) ---
        for col, lado in enumerate(("L", "R")):
            ax = axes[0, col]
            por_paciente = diffs_por_paciente[angulo["id"]][lado]
            if por_paciente:
                curvas_paciente = [np.nanmean(np.array(lista), axis=0) for lista in por_paciente.values()]
                matriz = np.array(curvas_paciente)
                media = np.nanmean(matriz, axis=0)
                sd = np.nanstd(matriz, axis=0)
                ax.plot(frames, media, color=chs.COLOR_DIFF, linewidth=2.5, label="Diferencia (HHMM - SCoRE)")
                ax.fill_between(frames, media - sd, media + sd, color=chs.COLOR_DIFF, alpha=0.25, label="+-1 SD entre pacientes")
                ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
                ax.set_title(f"{lado}", fontsize=16)
            else:
                ax.set_title(f"{lado}: sin datos", fontsize=16)
            ax.legend(fontsize=12)
            chs.aplicar_estilo_eje(ax, angulo, tipo="diff")
        axes[0, 0].set_ylabel("Diferencia HHMM - SCoRE (grados)")

        # --- Fila 2: paciente de ejemplo, promedio HH y SCoRE (cuadro 3 = L, cuadro 4 = R) ---
        for col, lado in enumerate(("L", "R")):
            ax = axes[1, col]
            dato = curvas_ejemplo[angulo["id"]][lado]
            color_lado = chs.COLOR_LADO[lado]
            if dato is not None:
                media_hh, media_score, n = dato
                ax.plot(frames, media_hh, color=color_lado, linewidth=2, label="HHMM")
                ax.plot(frames, media_score, color=color_lado, linewidth=2, linestyle="--", label="SCoRE")
                ax.set_title(f"{lado} - paciente {paciente_ejemplo}", fontsize=16)
            else:
                ax.set_title(f"{lado}: paciente {paciente_ejemplo} sin datos", fontsize=16)
            ax.set_xlabel("% del ciclo de marcha")
            ax.legend(fontsize=12)
            chs.aplicar_estilo_eje(ax, angulo)
        axes[1, 0].set_ylabel("Ángulo (grados)")

        fig.suptitle(angulo["titulo"])
        fig.tight_layout()

        out_path = out_dir / f"resumen_{angulo['id']}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log.info(f'Guardado: {out_path}')


def main(carpeta, out_dir, out_csv, out_log, prefijo_excluido='no', umbral_rmse=15.0, paciente_ejemplo='0128'):
    carpeta = Path(carpeta)
    if not carpeta.exists():
        raise FileNotFoundError(f'No se encontro la carpeta: {carpeta}')

    Path(out_log).resolve().parent.mkdir(parents=True, exist_ok=True)
    Path(out_csv).resolve().parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(out_log, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

    logger = logging.getLogger('correr_todos_curvas_con_ejemplo')
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
    logger.info(f'Paciente de ejemplo: {paciente_ejemplo}')

    registros = {a["id"]: {"R": [], "L": []} for a in chs.ANGULOS}

    ok, fallidos, sin_datos = [], [], []
    for sub in subcarpetas:
        logger.info('')
        logger.info(f'########## Paciente: {sub.name} ##########')
        try:
            n_walkings = ctf.recolectar_patient(sub, registros, logger)
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
    diffs, curvas_hh, curvas_score, diffs_por_paciente, filas_csv = ctf.filtrar_y_armar_acumuladores(
        registros, umbral_rmse, logger
    )

    if not any(r['paciente'] == paciente_ejemplo for lado_dict in registros.values() for lista in lado_dict.values() for r in lista):
        logger.warning(f'Aviso: el paciente de ejemplo "{paciente_ejemplo}" no aparece en los datos procesados.')

    curvas_ejemplo = curvas_ejemplo_paciente(registros, paciente_ejemplo, umbral_rmse)

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

    graficar_con_ejemplo(diffs, curvas_hh, curvas_score, diffs_por_paciente,
                          curvas_ejemplo, paciente_ejemplo, out_dir, logger)

    print(f'\nListo. Log en: {out_log}')
    print(f'CSV en: {out_csv}')
    print(f'Graficos en: {out_dir}')
    print(f'OK: {len(ok)}  |  Sin datos: {len(sin_datos)}  |  Fallidos: {len(fallidos)}')
    if fallidos:
        print(f'Fallidos: {", ".join(fallidos)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Como correr_todos_curvas_filtrado.py, pero con una tercera fila de ejemplo (curvas HH/SCoRE de un paciente puntual) en cada grafico'
    )
    parser.add_argument('--carpeta', required=True,
                         help='Carpeta que contiene las subcarpetas de cada paciente')
    parser.add_argument('--out-dir', required=True,
                         help='Carpeta de salida para los 6 PNG')
    parser.add_argument('--out-csv', required=True,
                         help='Archivo .csv con el resumen (incluye columna excluido_umbral)')
    parser.add_argument('--out', required=True, dest='out_log',
                         help='Archivo de texto con el log detallado')
    parser.add_argument('--prefijo-excluido', default='no',
                         help='Prefijo de carpetas a saltear, sin importar mayus/minus (default "no")')
    parser.add_argument('--umbral-rmse', type=float, default=15.0,
                         help='RMSE en grados por encima del cual se excluye un walking de un angulo/lado (default 15)')
    parser.add_argument('--paciente-ejemplo', default='0128',
                         help='Nombre del paciente a mostrar como ejemplo en la fila 3 (default 0128)')
    args = parser.parse_args()

    main(args.carpeta, args.out_dir, args.out_csv, args.out_log,
         args.prefijo_excluido, args.umbral_rmse, args.paciente_ejemplo)
