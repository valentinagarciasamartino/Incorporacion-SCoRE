#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comparar_hh_score.py

Compara, para cada walking, la curva de angulo obtenida con HJC=Harrington/
Davis (HH) contra la curva obtenida con HJC=SCoRE, para las 6 curvas de
interes (cadera FE/AA/rotacion, rodilla FE/varo-valgo/rotacion), lado por
lado.

Que hace, paso a paso:
1) Busca en una carpeta todos los pares de archivos .emt que comparten el
   mismo nombre base y difieren solo en el sufijo _hh / _score
   (ej: "0128_walk03_hh.emt" y "0128_walk03_score.emt").
2) Para cada archivo .emt, parsea el formato BTS ASCII "Angle cycle
   sequences": header con "Cycle sequences: N" + fila de columnas + N
   bloques de 101 filas (frames 0-100 de cada zancada normalizada).
3) Para cada archivo, colapsa las N zancadas a UNA sola curva de 101 puntos
   por columna, tomando la media (cycle-sequence-mean, igual a lo que
   BTS SmartAnalyzer llama "Cycle sequence mean").
4) Para cada walking (par hh/score) y cada una de las 12 columnas
   (6 angulos x 2 lados), calcula la curva de diferencia:
       diff(frame) = curva_HH(frame) - curva_SCoRE(frame)
   Un valor positivo significa "HH da mas [flexion/aduccion/rotacion
   interna, segun el signo del angulo] que SCoRE" en ese frame.
5) Agrega todas las curvas de diferencia (de todos los pacientes y
   walkings) en 6 graficos -uno por tipo de angulo-, cada uno con las
   curvas de lado R y lado L en colores distintos: la nube de curvas
   individuales en fondo (semi-transparente) + la curva media +- 1 SD
   en trazo grueso encima.
6) Escribe ademas una tabla CSV con un resumen numerico por
   paciente/walking/lado/angulo (RMSE, diferencia media, diferencia
   maxima absoluta y en que frame ocurre).

IMPORTANTE - convencion de signo R/L:
Este script asume que la convencion de signo de ad/aduccion y rotacion
ya es "anatomica" (ej: aduccion positiva en R significa lo mismo que
aduccion positiva en L), que es lo habitual en un modelo tipo Davis/PiG
de BTS. Antes de confiar en los graficos combinados de R+L, conviene
revisar con 2-3 pacientes que esto sea asi. Si no lo es, hay que invertir
el signo de un lado antes de correr este script (se puede hacer editando
el diccionario ANGULOS mas abajo, agregando un flip por columna).

Uso:
    python comparar_hh_score.py /ruta/a/carpeta/con/emt --out-dir graficos --out-csv resumen.csv

    La carpeta puede tener subcarpetas (se busca de forma recursiva).
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Tamanos de fuente mas grandes para todas las figuras generadas con este modulo
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.titlesize': 18,
})


# ---------------------------------------------------------------------------
# Definicion de las 6 curvas de interes: nombre para mostrar, columna R,
# columna L (nombres de columna tal cual aparecen en el .emt de BTS).
# ---------------------------------------------------------------------------
ANGULOS = [
    {"id": "cadera_abaduccion", "titulo": "Ab-aduccion de cadera",   "col_R": "acRHPAA", "col_L": "acLHPAA",
     "ylim": (-10, 10), "ylim_diff": (-10, 10), "label_pos": "Add", "label_neg": "Abd"},
    {"id": "cadera_flexoext",   "titulo": "Flexoextension de cadera", "col_R": "acRHPFE", "col_L": "acLHPFE",
     "ylim": (-15, 40), "ylim_diff": (-15, 40), "label_pos": "Flex", "label_neg": "Ext"},
    {"id": "cadera_rotacion",   "titulo": "Rotacion de cadera",       "col_R": "acRHPIE", "col_L": "acLHPIE",
     "ylim": (-15, 15), "ylim_diff": (-15, 15), "label_pos": "Int", "label_neg": "Ext"},
    {"id": "rodilla_varovalgo", "titulo": "Varo-valgo de rodilla",    "col_R": "acRKAA",  "col_L": "acLKAA",
     "ylim": (-10, 10), "ylim_diff": (-10, 10), "label_pos": "Var", "label_neg": "Valg"},
    {"id": "rodilla_flexoext",  "titulo": "Flexoextension de rodilla","col_R": "acRKFE",  "col_L": "acLKFE",
     "ylim": (-10, 80), "ylim_diff": (-10, 80), "label_pos": "Flex", "label_neg": "Ext"},
    {"id": "rodilla_rotacion",  "titulo": "Rotacion de rodilla",      "col_R": "acRKIE",  "col_L": "acLKIE",
     "ylim": (-40, 5), "ylim_diff": (-40, 5), "label_pos": "Int", "label_neg": "Ext"},
]

N_FRAMES = 101  # 0% a 100% del ciclo, siempre 101 puntos tras normalizar
SUFIJOS = ("hh", "score")

# Paleta de colores comun: lado L = rojo, lado R = verde (HH solido, SCoRE
# discontinuo del mismo color que su lado), y la curva de diferencia en azul.
COLOR_LADO = {"L": "red", "R": "green"}
COLOR_DIFF = "blue"


def aplicar_estilo_eje(ax, angulo, tipo="angulo"):
    """
    Aplica el rango de eje Y definido en ANGULOS para este angulo.
    tipo="angulo": usa "ylim" (rango del angulo real, ej. curvas de HH/SCoRE
        de un paciente) y agrega las etiquetas +/- (Add/Abd, Flex/Ext,
        Int/Ext, Var/Valg) con el mismo estilo que un reporte clinico de
        marcha tradicional: etiqueta arriba-derecha para el valor positivo,
        abajo-derecha para el negativo.
    tipo="diff": usa "ylim_diff" (rango mas chico, para paneles que
        muestran la diferencia HH-SCoRE), sin las etiquetas +/-.
    """
    clave_ylim = "ylim_diff" if tipo == "diff" else "ylim"
    ylim = angulo.get(clave_ylim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Deja las marcas del eje Y, pero etiqueta (numero) si y blanco si no,
    # alternando, para que no se amontonen los valores.
    ticks = ax.get_yticks()
    etiquetas = []
    for i, t in enumerate(ticks):
        if i % 2 != 0:
            etiquetas.append("")
        else:
            etiquetas.append(f"{t:g}")
    ax.set_yticks(ticks)
    ax.set_yticklabels(etiquetas)

    if tipo == "diff":
        return

    pos = angulo.get("label_pos")
    neg = angulo.get("label_neg")
    if pos:
        ax.text(1.01, 0.93, pos, transform=ax.transAxes, fontsize=13,
                color="#444444", ha="left", va="top")
    if neg:
        ax.text(1.01, 0.07, neg, transform=ax.transAxes, fontsize=13,
                color="#444444", ha="left", va="bottom")


# ---------------------------------------------------------------------------
# Parseo de archivos .emt (formato BTS ASCII, "Angle cycle sequences")
# ---------------------------------------------------------------------------
def parse_emt(path):
    """
    Lee un .emt de BTS tipo "Angle cycle sequences" y devuelve:
        n_cycles: cantidad de zancadas (cycles) del archivo
        data: dict columna -> np.ndarray de forma (n_cycles, 101)
    """
    with open(path, "r", encoding="latin-1") as f:
        lines = f.readlines()

    n_cycles = None
    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower().startswith("cycle sequences"):
            n_cycles = int(stripped.split("\t")[-1].strip())
        # la fila de encabezado tiene "Cycle" y "Sample" pero no es la
        # linea de "Cycle sequences: N"
        if "sequences" not in stripped.lower() and stripped.startswith("Cycle") and "Sample" in line:
            header_idx = i

    if header_idx is None or n_cycles is None:
        raise ValueError(f"No se pudo interpretar el formato de {path} (no se encontro header o 'Cycle sequences')")

    header_cols = [c.strip() for c in lines[header_idx].split("\t")]
    header_cols = [c for c in header_cols if c != ""]
    angle_cols = header_cols[2:]  # se saltea "Cycle" y "Sample"

    data = {col: np.full((n_cycles, N_FRAMES), np.nan) for col in angle_cols}

    current_cycle = -1
    for line in lines[header_idx + 1:]:
        if not line.strip():
            continue
        fields = line.rstrip("\n").split("\t")
        fields = [f.strip() for f in fields][: len(header_cols)]
        if len(fields) < 2:
            continue
        if fields[0] != "":
            current_cycle += 1
        try:
            sample_idx = int(fields[1])
        except ValueError:
            continue
        if current_cycle < 0 or current_cycle >= n_cycles or sample_idx >= N_FRAMES:
            continue
        for col_idx, col in enumerate(angle_cols):
            raw = fields[2 + col_idx] if 2 + col_idx < len(fields) else ""
            if raw == "":
                continue
            try:
                data[col][current_cycle, sample_idx] = float(raw)
            except ValueError:
                pass

    return n_cycles, data


def curva_media(data, col):
    """Cycle-sequence-mean: promedio a traves de todas las zancadas, frame a frame."""
    if col not in data:
        return None
    return np.nanmean(data[col], axis=0)  # forma (101,)


# ---------------------------------------------------------------------------
# Busqueda de pares hh/score
# ---------------------------------------------------------------------------
def encontrar_pares(carpeta):
    """
    Busca recursivamente archivos *_hh.emt y *_score.emt y arma pares por
    nombre base (mismo nombre salvo el sufijo _hh / _score).
    Devuelve una lista de dicts: {"base": str, "hh": Path, "score": Path}
    """
    carpeta = Path(carpeta)
    candidatos = defaultdict(dict)

    patron = re.compile(r"^(?P<base>.+)_(?P<suf>hh|score)$", re.IGNORECASE)

    for p in carpeta.rglob("*.emt"):
        m = patron.match(p.stem)
        if not m:
            continue
        base = m.group("base")
        suf = m.group("suf").lower()
        candidatos[base][suf] = p

    pares = []
    incompletos = []
    for base, d in sorted(candidatos.items()):
        if "hh" in d and "score" in d:
            pares.append({"base": base, "hh": d["hh"], "score": d["score"]})
        else:
            incompletos.append((base, list(d.keys())))

    if incompletos:
        print("Aviso: archivos sin su par (se ignoran):")
        for base, tienen in incompletos:
            print(f"  {base}: solo tiene {tienen}")

    return pares


def id_paciente_walking(base):
    """De un nombre base tipo '0128_walk03' saca (paciente, walking)."""
    partes = base.split("_")
    paciente = partes[0] if partes else base
    walking = "_".join(partes[1:]) if len(partes) > 1 else ""
    return paciente, walking


# ---------------------------------------------------------------------------
# Analisis principal
# ---------------------------------------------------------------------------
def analizar(carpeta, out_dir, out_csv):
    pares = encontrar_pares(carpeta)
    if not pares:
        print("No se encontraron pares _hh/_score.emt en la carpeta indicada.")
        return

    print(f"Se encontraron {len(pares)} walkings (pares hh/score).")

    # acumuladores: para cada angulo, para cada lado, lista de curvas
    diffs = {a["id"]: {"R": [], "L": []} for a in ANGULOS}
    curvas_hh = {a["id"]: {"R": [], "L": []} for a in ANGULOS}
    curvas_score = {a["id"]: {"R": [], "L": []} for a in ANGULOS}
    filas_csv = []

    for par in pares:
        base = par["base"]
        paciente, walking = id_paciente_walking(base)

        try:
            n_hh, data_hh = parse_emt(par["hh"])
            n_score, data_score = parse_emt(par["score"])
        except Exception as e:
            print(f"  [{base}] ERROR al parsear: {e}")
            continue

        print(f"  [{base}] hh: {n_hh} ciclos, score: {n_score} ciclos")

        for angulo in ANGULOS:
            for lado, col in (("R", angulo["col_R"]), ("L", angulo["col_L"])):
                curva_hh = curva_media(data_hh, col)
                curva_score = curva_media(data_score, col)
                if curva_hh is None or curva_score is None:
                    print(f"    Aviso: columna {col} no encontrada en {base}")
                    continue

                diff = curva_hh - curva_score  # HH - SCoRE, frame a frame
                if np.all(np.isnan(diff)):
                    continue

                diffs[angulo["id"]][lado].append(diff)
                curvas_hh[angulo["id"]][lado].append(curva_hh)
                curvas_score[angulo["id"]][lado].append(curva_score)

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

    # --- graficos: uno por angulo, con un subplot por lado (R y L) ---
    # En cada subplot: promedio de HH, promedio de SCoRE, y la diferencia
    # entre ambos promedios (que, al ser un promedio de diferencias por
    # walking, matematicamente da lo mismo que restar los dos promedios).
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = np.arange(N_FRAMES)

    for angulo in ANGULOS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        hay_datos = False

        for ax, lado in zip(axes, ("R", "L")):
            lista_hh = curvas_hh[angulo["id"]][lado]
            lista_score = curvas_score[angulo["id"]][lado]
            if not lista_hh or not lista_score:
                ax.set_title(f"{lado}: sin datos")
                continue
            hay_datos = True

            media_hh = np.nanmean(np.array(lista_hh), axis=0)
            media_score = np.nanmean(np.array(lista_score), axis=0)
            media_diff = media_hh - media_score
            n = len(lista_hh)

            color_lado = COLOR_LADO[lado]
            ax.plot(frames, media_hh, color=color_lado, linewidth=2, label="Promedio HHMM")
            ax.plot(frames, media_score, color=color_lado, linewidth=2, linestyle="--", label="Promedio SCoRE")
            ax.plot(frames, media_diff, color=COLOR_DIFF, linewidth=2, label="Diferencia (HHMM - SCoRE)")
            ax.axhline(0, color="black", linewidth=0.8, linestyle=":")

            ax.set_title(f"{lado} (n={n} walkings)")
            ax.set_xlabel("% del ciclo de marcha")
            ax.legend(fontsize=13)
            aplicar_estilo_eje(ax, angulo)

        axes[0].set_ylabel("Ángulo (grados)")
        fig.suptitle(angulo["titulo"])
        fig.tight_layout()

        out_path = out_dir / f"diff_{angulo['id']}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Guardado: {out_path}")
        if not hay_datos:
            print(f"  (aviso: {angulo['id']} no tuvo datos en ningun lado)")

    # --- CSV resumen ---
    if filas_csv:
        import csv
        out_csv = Path(out_csv)
        campos = ["paciente", "walking", "lado", "angulo", "rmse_deg",
                  "diff_media_deg", "diff_max_abs_deg", "frame_max_abs"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(filas_csv)
        print(f"Guardado: {out_csv}")
    else:
        print("No se genero CSV: no hubo datos validos para ninguna columna.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compara curvas de angulo HH vs SCoRE (walkings pareados).")
    parser.add_argument("carpeta", help="Carpeta con los .emt (_hh y _score), busca en subcarpetas tambien")
    parser.add_argument("--out-dir", default="graficos_hh_vs_score", help="Carpeta de salida para los 6 PNG")
    parser.add_argument("--out-csv", default="resumen_hh_vs_score.csv", help="Archivo CSV de salida")
    args = parser.parse_args()

    analizar(args.carpeta, args.out_dir, args.out_csv)
