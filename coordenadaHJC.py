import argparse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Tamanos de fuente mas grandes para todas las figuras generadas por este script
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 22,
})

PATRON_PACIENTE = re.compile(r'^(\d+)\s*$')
PATRON_RESULTADO = re.compile(r'^(?:INFO - )?(SCoRE|Davis 91|Harrington)\s*([RL])\s*:\s*\[\s*([^\]]+)\]')
NOMBRE_METODO = {'SCoRE': 'SCoRE', 'Davis 91': 'Davis', 'Harrington': 'Harrington'}


def parse_resultados(path):
    """
    Devuelve dict: paciente -> lado -> metodo -> np.array([x,y,z])
    """
    datos = defaultdict(lambda: defaultdict(dict))
    paciente_actual = None
    with open(path, 'r', encoding='utf-8') as f:
        for linea in f:
            linea = linea.rstrip('\n')
            m_pac = PATRON_PACIENTE.match(linea.strip())
            if m_pac:
                paciente_actual = m_pac.group(1)
                continue
            m_res = PATRON_RESULTADO.match(linea.strip())
            if m_res and paciente_actual is not None:
                metodo_raw, lado, numeros_str = m_res.groups()
                metodo = NOMBRE_METODO.get(metodo_raw, metodo_raw)
                try:
                    valores = np.array([float(x) for x in numeros_str.split()])
                except ValueError:
                    continue
                datos[paciente_actual][lado][metodo] = valores
    return datos


METODOS = ["SCoRE", "Davis", "Harrington"]
COLOR_METODO = {"SCoRE": "#e377c2", "Davis": "#ff7f0e", "Harrington": "#2ca02c"}


def calcular_valores(datos):
    """
    Valores CRUDOS (no diferencias): el "0" es el origen real del sistema
    de coordenadas de la pelvis (el mismo 0 de score_hjc.py), no un
    promedio ni un metodo elegido como referencia.

    - distancia 3D (mm) de cada metodo al origen: R+L pooleados
    - X, Z (mm): R+L pooleados (no dependen del lado)
    - Y (mm): SIN corregir signo, separado en dos grupos (R y L), porque
      el signo de Y depende del lado (negativo en R, positivo en L)

    El lado de cada valor es el que indica el archivo de resultados, sin
    ningun intercambio R/L (los tres metodos son internamente consistentes
    en el signo de Y por lado, como se verifico con los datos crudos).
    """
    resultado = {
        "dist": {m: [] for m in METODOS},
        "dX": {m: [] for m in METODOS},
        "dZ": {m: [] for m in METODOS},
        "dY_R": {m: [] for m in METODOS},
        "dY_L": {m: [] for m in METODOS},
    }

    for paciente, lados in datos.items():
        for lado, metodos in lados.items():
            if not all(m in metodos for m in METODOS):
                continue

            for m in METODOS:
                val = metodos[m]  # [x, y, z], en metros, crudo
                dist_mm = float(np.linalg.norm(val) * 1000)
                x_mm = val[0] * 1000
                z_mm = val[2] * 1000
                y_mm = val[1] * 1000

                resultado["dist"][m].append(dist_mm)
                resultado["dX"][m].append(x_mm)
                resultado["dZ"][m].append(z_mm)
                resultado[f"dY_{lado}"][m].append(y_mm)

    return resultado


def _boxplot_por_metodo(ax, valores_por_metodo, metodos=METODOS, posiciones=None):
    """Dibuja una caja por metodo, coloreada segun COLOR_METODO."""
    datos_box = [valores_por_metodo[m] for m in metodos]
    bp = ax.boxplot(datos_box, positions=posiciones, patch_artist=True,
                     widths=0.6)
    for patch, m in zip(bp['boxes'], metodos):
        patch.set_facecolor(COLOR_METODO[m])
        patch.set_alpha(0.5)
    return bp


def graficar_estilo_sangeux(valores, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # --- arriba-izq: distancia 3D al origen (R+L pooleados) ---
    ax = axes[0, 0]
    _boxplot_por_metodo(ax, valores["dist"])
    ax.set_xticks(range(1, len(METODOS) + 1))
    ax.set_xticklabels(METODOS)
    ax.set_title("Distancia al origen (3D)", pad=30)
    ax.set_ylabel("mm")
    ax.yaxis.set_label_coords(-0.14, 0.5)

    # --- arriba-der: X (R+L pooleados) ---
    ax = axes[0, 1]
    _boxplot_por_metodo(ax, valores["dX"])
    ax.set_xticks(range(1, len(METODOS) + 1))
    ax.set_xticklabels(METODOS)
    ax.axhline(0, color="red", linewidth=1)
    ax.set_title("Coordenada X", pad=30)
    ax.set_ylabel("mm")
    ax.yaxis.set_label_coords(-0.14, 0.5)

    # --- abajo-izq: Y, agrupado por lado (L a la izquierda, R a la derecha) ---
    ax = axes[1, 0]
    n = len(METODOS)
    pos_izq = list(range(1, n + 1))
    pos_der = list(range(n + 2, 2 * n + 2))
    _boxplot_por_metodo(ax, valores["dY_L"], posiciones=pos_izq)
    _boxplot_por_metodo(ax, valores["dY_R"], posiciones=pos_der)
    ax.set_xticks(pos_izq + pos_der)
    ax.set_xticklabels([])  # los metodos los aclara la leyenda de color
    ax.axhline(0, color="red", linewidth=1)
    # separador vertical entre el grupo L y el grupo R
    ax.axvline((pos_izq[-1] + pos_der[0]) / 2, color="gray", linewidth=1, linestyle=":")
    ax.set_title("Coordenada Y", pad=30)
    ax.set_ylabel("mm")
    ax.yaxis.set_label_coords(-0.14, 0.5)
    # etiquetas de grupo L / R (en coordenadas de ejes para no chocar con el titulo)
    trans = ax.get_xaxis_transform()
    ax.text(sum(pos_izq) / n, 1.02, "L", ha="center", va="bottom",
            fontweight="bold", fontsize=13, transform=trans)
    ax.text(sum(pos_der) / n, 1.02, "R", ha="center", va="bottom",
            fontweight="bold", fontsize=13, transform=trans)
    # leyenda de metodos por color (aclara cual caja es cual metodo, en R y en L)
    handles = [plt.matplotlib.patches.Patch(facecolor=COLOR_METODO[m], alpha=0.5,
                                             edgecolor="black", label=m) for m in METODOS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=3, frameon=False, fontsize=16)

    # --- abajo-der: Z (R+L pooleados) ---
    ax = axes[1, 1]
    _boxplot_por_metodo(ax, valores["dZ"])
    ax.set_xticks(range(1, len(METODOS) + 1))
    ax.set_xticklabels(METODOS)
    ax.axhline(0, color="red", linewidth=1)
    ax.set_title("Coordenada Z", pad=30)
    ax.set_ylabel("mm")
    ax.yaxis.set_label_coords(-0.14, 0.5)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.subplots_adjust(hspace=0.5)
    out_path = out_dir / "comparacion_estilo_sangeux.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Guardado: {out_path}")


def graficar_dispersion_por_paciente(datos, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pacientes = sorted(datos.keys())
    metodos = ["SCoRE", "Davis", "Harrington"]
    colores = {"SCoRE": "#e377c2", "Davis": "#ff7f0e", "Harrington": "#2ca02c"}
    marcador = {"R": "o", "L": "s"}

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    ejes_nombres = ["X [m]", "Y [m]", "Z [m]"]

    for i, ax in enumerate(axes):
        for j, paciente in enumerate(pacientes):
            for lado in ("R", "L"):
                if lado not in datos[paciente]:
                    continue
                for metodo in metodos:
                    if metodo not in datos[paciente][lado]:
                        continue
                    valor = datos[paciente][lado][metodo][i]
                    ax.scatter(j, valor, color=colores[metodo], marker=marcador[lado], s=60,
                               label=f"{metodo} {lado}" if (i == 0 and j == 0) else None)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel(ejes_nombres[i])
        ax.grid(axis="x", linestyle=":", alpha=0.5)

    # leyenda combinada, una sola vez
    handles = []
    labels = []
    for metodo in metodos:
        for lado in ("R", "L"):
            handles.append(plt.Line2D([0], [0], marker=marcador[lado], color="w",
                                       markerfacecolor=colores[metodo], markersize=9))
            labels.append(f"{metodo} {lado}")
    axes[0].legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=16)

    # los IDs de paciente son numericos; se muestran siempre con 4 digitos
    # (ej. "094" -> "0094") para que queden parejos con el resto
    etiquetas_pacientes = [p.zfill(4) if p.isdigit() else p for p in pacientes]
    axes[-1].set_xticks(range(len(pacientes)))
    axes[-1].set_xticklabels(etiquetas_pacientes, rotation=45, ha="right")
    fig.tight_layout()

    out_path = out_dir / "dispersion_por_paciente.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Guardado: {out_path}")


def main(path, out_dir):
    datos = parse_resultados(path)
    print(f"Pacientes encontrados: {sorted(datos.keys())}")

    valores = calcular_valores(datos)
    graficar_estilo_sangeux(valores, out_dir)
    graficar_dispersion_por_paciente(datos, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compara HJC (SCoRE/Davis/Harrington) estilo Sangeux et al. 2011")
    parser.add_argument("resultados", help="Archivo de texto con el log de resultados")
    parser.add_argument("--out-dir", default="graficos_hjc", help="Carpeta de salida")
    args = parser.parse_args()

    main(args.resultados, args.out_dir)
