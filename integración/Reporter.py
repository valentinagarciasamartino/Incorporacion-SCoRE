import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from matplotlib.pyplot import get_cmap
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation
from simplemdx import mdxParser
from weasyprint import CSS, HTML
from snapshot_pyppeteer import snapshot
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Gauge, Pie
from pyecharts.render import make_snapshot

from Figures import (
    EMG_Figure,
    KM_Figure,
    KM_Figure_Running,
    KT_Ang_Torque_Power_Figure,
    KT_Frontal_Plane_Figure,
    KT_Ground_Reaction_Figure,
)
from functions import *
logging.basicConfig(level=logging.INFO)

# Render engine can be either kaleido. Kaleido is faster, but needs some more work
RENDER_ENGINE = "kaleido"


class Report:
    def __init__(self, trial, session, standing=None, normative=None, anonymize=False):
        print("Report src")
        self.trial_path = Path(trial)
        self.session_path = Path(session)
        self.session_mdx = mdxParser(self.session_path.as_posix())
        self.trial_mdx = mdxParser(self.trial_path.as_posix())
        self.anonymize = anonymize
        ### NUEVO
        self.score_mdx = None
        ### fin nuevo
        age = self.session_mdx.trial.static.age
        if not normative:
            if age < 18:
                self.normative_path = Path(
                    r"Q:\SMARTData\_normatives\en\Gait-Normal-children.enb"
                )

            else:
                self.normative_path = Path(
                    r"Q:\SMARTData\_normatives\en\Gait-Normal-adults.enb"
                )
        else:
            if "Running" in normative:
                s = next((x for x in self.trial_mdx.trial.streams[0]._scalars if x.label == "Velocity [m/s]"), None)
                veloc_mps = s.data[0][-1] if s else s
                if veloc_mps:
                    if veloc_mps <= 2.5:
                        normative = normative+"2,5mps.enb"
                    elif 2.5 < veloc_mps <= 3.5:
                        normative = normative+"3,5mps.enb"
                    else:
                        normative = normative+"4,5mps.enb"
                else: normative = normative+"3,5mps.enb"
            print(normative)
            self.normative_path = Path(normative)

        if standing:
            self.standing_path = Path(standing)
            self.standing_mdx = mdxParser(self.standing_path.as_posix())

        self.normative_mdx = mdxParser(self.normative_path.as_posix())

        name = self.session_mdx.trial.static.name
        lastname = self.session_mdx.trial.static.lastname

        if self.anonymize:
            name = name[0].capitalize()
            lastname = lastname[0].capitalize()
        date = self.session_mdx.trial.static.date.strftime("%d/%m/%Y")

        if self.anonymize:
            name = name[0]
            lastname = lastname[0]

        self.ntrial = self.trial_mdx.trial.label.split("~")[-1][8:-4]

        self.root_path = Path(__file__).parent
        self.images = self.root_path / "images"
        self.css = self.root_path / "templates" / "EMG.css"
        self.fonts = list((self.root_path / "templates").glob("*.ttf"))

        """ Can't use Interstate (FLENI's font) for plots, since the axis use
         a real unicode minus sign, instead of hyphens, and Interstate has no
         Unicode. Calibri should do.
         """

        self.plotly_size = 18
        self.plotly_size_m = 14
        self.plotly_size_sm = 12

        self.boot_css = self.root_path / "templates" / "bootstrap.css"
        self.font_css = [
            "@font-face {font-family: "
            + i.stem
            + ';src: url("'
            + i.resolve().as_uri()
            + '")}'
            for i in self.fonts
        ]
        self.font_css = [CSS(string=i) for i in self.font_css]
        self.styles = self.font_css
        self.styles.append(CSS(self.css.as_posix()))

        self.footer = f"""<footer style="font-size:{self.plotly_size_sm}px">
            <hr>
            <table style="width:100%;table-layout:fixed">
            <tr>
            <td style="text-align:left;">
                {lastname}, {name}
            </td>
            <td style="text-align:center;">
               {self.session_mdx.trial.label} {self.ntrial}
            </td>
            <td style="text-align:right">
                    {date}
            </td>
            </tr>
            </table>
            </footer>"""
        self.pages = []

    ### NUEVO
    @property
    def has_score(self) -> bool:
        return self.score_mdx is not None

    def _score_cycle(self):
        return self.score_mdx.trial.cycle if self.has_score else None

    def _score_value(self, section: str, key: str):
        """section: 'angles' | 'torques1d' | 'powers' | 'scalars' | 'angle_bands'.
        Returns .data or None if missing."""
        cycle = self._score_cycle()
        if cycle is None:
            return None
        data = getattr(cycle, section)
        return data[key].data if key in data else None

    @staticmethod
    def _score_side_style(side: str, width: int = 2):
        is_right = side.upper().startswith("R")
        name = "SCORE Derecha" if is_right else "SCORE Izquierda"
        line = dict(width=width, color="green" if is_right else "red", dash="dash")
        return name, line

    def _add_score_scatter(
        self,
        fig,
        y,
        x,
        side,
        row,
        col,
        legendgroup,
        showlegend=False,
        width=2,
    ):
        name, line = self._score_side_style(side, width=width)
        fig.add_trace(
            go.Scatter(
                y=y,
                x=x,
                name=name,
                line=line,
                legendgroup=legendgroup,
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )
    ### fin nuevo

    def add_normality_region_emg(self, muscle, side, normalized=False):
        """Checks the normative for EMG normality region, returns it
        as a list of rectangles"""
        if not normalized:
            event_data = self.trial_mdx.trial.static.events[f"e{side}HS"].data
        else:
            event_data = [0, 1]

        data = self.normative_mdx.norm.cycle.emg[muscle]
        shapes = [
            go.layout.Shape(
                type="rect",
                xref="x",
                # yref="paper",
                x0=event_data[0] + i["start"] * (event_data[1] - event_data[0]) / 100,
                y0=-1,
                x1=event_data[0]
                + (i["start"] + i["length"]) * (event_data[1] - event_data[0]) / 100,
                y1=1,
                fillcolor="LightSalmon",
                opacity=0.3,
                layer="below",
                line_width=0,
            )
            for i in data
        ]
        return shapes

    def RUN_header(self, title="Reporte"):
        """Page with the main logo and table with trial metadata"""

        if not self.trial_mdx.trial.has_kinematics:
            print(f"KM report: NO angles in {self.trial_path}. Return")
            return

        root_path = Path(__file__).parent
        images = root_path / "images"
        css = root_path / "templates" / "EMG.css"
        boot = root_path / "templates" / "bootstrap.css"
        header = images / "Gait.bmp"
        logo = images / "logo_fleni_60-anios-vertical.svg"

        name = self.session_mdx.trial.static.name
        lastname = self.session_mdx.trial.static.lastname

        if self.anonymize:
            name = name[0].capitalize()
            lastname = lastname[0].capitalize()

        date = self.session_mdx.trial.static.date.strftime("%d/%m/%Y")
        birthdate = self.session_mdx.trial.static.birthday.strftime("%d/%m/%Y")
        session = self.session_mdx.trial.static.session_notes
        ntrial = self.trial_mdx.trial.label.split("~")[-1][7:-4]
        patid = self.trial_mdx.trial.label.split("~")[0]

        if self.anonymize:
            name = name[0]
            lastname = lastname[0]
            clinician = "FLENI"

        template = f"""
        <body>
            <img src='{logo.resolve().as_uri()}' class='main_logo'>
            <img src='{header.resolve().as_uri()}'>
            <h1>Análisis de carrera</h1>
            <h2>{title}</h2>
            <table class='table table-striped' style='width:100%'>
                <tr>
                    <th scope='row'>Apellido:</th>
                    <td>{lastname}</td>
                </tr>
                <tr>
                    <th scope='row'>Nombre:</th>
                    <td>{name}</td>
                </tr>
                <tr>
                    <th scope='row'>ID paciente:</th>
                    <td>{patid}</td>
                </tr>
                <tr>
                    <th scope='row'>Fecha de nacimiento:</th>
                    <td>{birthdate}</td>
                </tr>
                <tr>
                    <th scope='row'>Fecha de estudio:</th>
                    <td>{date}</td>
                </tr>
                <tr>
                    <th scope='row'>Condición:</th>
                    <td>{session}</td>
                </tr>
                <tr>
                    <th scope='row'>Pasada:</th>
                    <td>{ntrial}</td>
                </tr>
            </table>

        </body>
        """
        # Weasyprint the template
        page = HTML(string=template).render(stylesheets=self.styles)
        self.pages.append(page)

    def KM_static_header(self, title="Reporte"):
        """Page with the main logo and table with trial metadata"""

        if not self.trial_mdx.trial.has_kinematics:
            print(f"KM report: NO angles in {self.trial_path}. Return")
            return

        root_path = Path(__file__).parent
        images = root_path / "images"
        css = root_path / "templates" / "EMG.css"
        boot = root_path / "templates" / "bootstrap.css"
        header = images / "Gait.bmp"
        logo = images / "logo_fleni_60-anios-vertical.svg"

        name = self.session_mdx.trial.static.name
        lastname = self.session_mdx.trial.static.lastname

        if self.anonymize:
            name = name[0].capitalize()
            lastname = lastname[0].capitalize()
        
        date = self.session_mdx.trial.static.date.strftime("%d/%m/%Y")
        birthdate = self.session_mdx.trial.static.birthday.strftime("%d/%m/%Y")
        clinician = self.session_mdx.trial.static.clinician
        clinician = clinician.replace(" KLGO TOLEDO", "")
        diagnostic = self.session_mdx.trial.static.taxid
        pathology = self.session_mdx.trial.static.pathology
        session = self.session_mdx.trial.static.session_notes
        ntrial = self.trial_mdx.trial.label.split("~")[-1][7:-4]
        patid = self.trial_mdx.trial.label.split("~")[0]

        if self.anonymize:
            name = name[0]
            lastname = lastname[0]
            clinician = "FLENI"

        template = f"""
        <body>
            <img src='{logo.resolve().as_uri()}' class='main_logo'>
            <img src='{header.resolve().as_uri()}'>
            <h1>ANÁLISIS CLÍNICO DE MARCHA</h1>
            <h2>{title}</h2>
            <h3>Datos del paciente</h3>
            <table class='table table-striped' style='width:100%'>
                <tr>
                    <th scope='row'>Apellido:</th>
                    <td>{lastname}</td>
                </tr>
                <tr>
                    <th scope='row'>Nombre:</th>
                    <td>{name}</td>
                </tr>
                <tr>
                    <th scope='row'>ID paciente:</th>
                    <td>{patid}</td>
                </tr>
                <tr>
                    <th scope='row'>Fecha de nacimiento:</th>
                    <td>{birthdate}</td>
                </tr>
                <tr>
                    <th scope='row'>Fecha de estudio:</th>
                    <td>{date}</td>
                </tr>
                <tr>
                    <th scope='row'>Médico de referencia:</th>
                    <td>{clinician}</td>
                </tr>
                <tr>
                    <th scope='row'>Diagnóstico:</th>
                    <td>{diagnostic}</td>
                </tr>
                <tr>
                    <th scope='row'>Compromiso:</th>
                    <td>{pathology}</td>
                </tr>
                <tr>
                    <th scope='row'>Condición:</th>
                    <td>{session}</td>
                </tr>
                <tr>
                    <th scope='row'>Pasada:</th>
                    <td>{ntrial}</td>
                </tr>
            </table>

        </body>
        """
        # Weasyprint the template
        page = HTML(string=template).render(stylesheets=self.styles)
        self.pages.append(page)

    def KM_static_table(self):
        if not self.trial_mdx.trial.has_kinematics:
            print(f"KM report: NO angles in {self.trial_path}. Return")
            return

        stat = self.trial_mdx.trial.static
        norm = self.normative_mdx.norm.static
        data = {
            "Cadencia": {"code": "fq_CADENCE"},
            "Tiempo de ciclo (s)": {"code": "tm_STRIDE"},
            "Tiempo de apoyo (s)": {"code": "tm_STANCE"},
            "Tiempo de balanceo (s)": {"code": "tm_SWING"},
            "Fase de apoyo (%)": {"code": "s_STANCE"},
            "Fase de balanceo (%)": {"code": "s_SWING"},
            "Fase de apoyo simple (%)": {"code": "s_SINGSTANCE"},
            "Fase de apoyo doble (%)": {"code": "s_DBLSTANCE"},
            "Velocidad media (m/s)": {"code": "velm_HSTRIDE"},
            "Velocidad de balanceo media (m/s)": {"code": "velm_HSWING"},
            "Largo de ciclo (m)": {"code": "d_STRIDE"},
            "Largo de paso (m)": {"code": "dm_STEP"},
            "Ancho de paso (m)": {"code": "dmSTEPWIDTH"},
            "Gait Deviation Index": {"code": "_GDI"},
            "Gait Profile Score": {"code": "a_GPS"},
        }

        for k, v in data.items():
            if "_" in v["code"]:
                for side in ["R", "L"]:
                    item = stat.find(v["code"].replace("_", side))
                    if "M" in item.coords:
                        data[k][side] = {"data": item.data[0], "error": item.data[1]}
                    elif "V" in item.coords:
                        data[k][side] = {"data": item.data[0]}

                    else:
                        raise Exception(f"No data for {side} {v['code']}")
                try:
                    item = stat.find(v["code"].replace("_", ""))
                    if "M" in item.coords:
                        data[k]["B"] = {"data": item.data[0], "error": item.data[1]}
                    elif "V" in item.coords:
                        data[k]["B"] = {"data": item.data[0]}
                except:
                    pass
            else:
                item = stat.find(v["code"])
                if "M" in item.coords:
                    data[k]["B"] = {"data": item.data[0], "error": item.data[1]}
                elif "V" in item.coords:
                    data[k]["B"] = {"data": item.data[0]}
                else:
                    raise Exception(f"No data for {v['code']}")
            try:
                name = (
                    v["code"]
                    .replace("_", "")
                    .replace("tm", "t")
                    .replace("dm", "d")
                    .replace("velm", "vel")
                    .replace("velHSTRIDE", "velMEAN")
                )
                norm_item = norm.find(name)
                data[k]["N"] = {"data": norm_item.data[0], "error": norm_item.data[1]}
            except:
                print(f"No norm for {name}")

        specs = [[{}] * 3] * 3
        specs.append([{"colspan": 3}, {}, {}])

        self.KM_table_fig = make_subplots(
            rows=4,
            cols=3,
            # vertical_spacing=0.05
            horizontal_spacing=0.1,
            subplot_titles=[
                "Tiempo de apoyo/balanceo (s)",
                "Fase de apoyo/balanceo (%)",
                "Fase de apoyo (%)",
                "Velocidad (m/s)",
                "Cadencia (pasos por minuto)",
                "Longitud (m)",
                "Ancho de paso (m)",
                "Gait Deviation Index (GDI)†",
                "Gait Profile Score (GPS)‡",
                "Gait Variable Scores (GVS)",
            ]
            + [None] * 2,
            specs=specs,
        )
        self.KM_table_fig.update_layout(
            template="plotly_white", font={"size": self.plotly_size_sm}
        )

        # Add step time graphs:
        vals = {
            k: v
            for k, v in data.items()
            if k in ["Tiempo de apoyo (s)", "Tiempo de balanceo (s)"]
        }
        for trace in stance_swing_time_subfig(vals):
            self.KM_table_fig.add_trace(trace, row=1, col=1)

        self.KM_table_fig.update_yaxes(visible=True, row=1, col=1)
        vals = {
            k: v
            for k, v in data.items()
            if k in ["Fase de apoyo (%)", "Fase de balanceo (%)"]
        }

        for trace in stance_swing_phase_subfig(vals):
            self.KM_table_fig.add_trace(trace, row=1, col=2)

        self.KM_table_fig.update_yaxes(visible=False, row=1, col=2)

        # Add single/double stance graph
        vals = {
            k: v
            for k, v in data.items()
            if k in ["Fase de apoyo simple (%)", "Fase de apoyo doble (%)"]
        }

        for trace in stance_single_double_phase_subfig(vals):
            self.KM_table_fig.add_trace(trace, row=1, col=3)

        self.KM_table_fig.update_yaxes(visible=False, row=1, col=3)

        # Add velocity
        vals = {
            k: v
            for k, v in data.items()
            if k in ["Velocidad media (m/s)", "Velocidad de balanceo media (m/s)"]
        }

        for trace in speeds_subfig(vals):
            self.KM_table_fig.add_trace(trace, row=2, col=1)

        self.KM_table_fig.update_yaxes(visible=True, row=2, col=1)
        vals = {k: v for k, v in data.items() if k in ["Cadencia"]}
        for trace in cadence_subfig(vals):
            self.KM_table_fig.add_trace(trace, row=2, col=2)
        self.KM_table_fig.update_yaxes(visible=True, row=2, col=2)
        vals = {
            k: v
            for k, v in data.items()
            if k in ["Largo de ciclo (m)", "Largo de paso (m)"]
        }

        for n, m in enumerate(vals.items()):
            marker_opacity = 0.6
            k, v = m
            dat = v["N"]["data"]
            err = [v["N"]["error"], v["N"]["error"]]
            self.KM_table_fig.add_trace(
                go.Bar(
                    x=["Ciclo" if "ciclo" in k else "Paso"],
                    y=dat,
                    error_y={"array": err},
                    text=f"{dat[0]:.2f}",
                    textfont={"color": "black"},
                    textposition="auto",
                    insidetextanchor="middle",
                    marker={"color": norm_color(marker_opacity)},
                    marker_line_color=norm_color(1),
                    marker_line_width=1.5,
                    offsetgroup=0,
                ),
                row=2,
                col=3,
            )

            for side in ["R", "L"]:
                dat = v[side]["data"]
                err = [v[side]["error"], v[side]["error"]]

                if side == "R":
                    marker_color = right_color(marker_opacity)
                    marker_line_color = right_color(1)
                    offsetgroup = 1
                elif side == "L":
                    marker_color = left_color(marker_opacity)
                    marker_line_color = left_color(1)
                    offsetgroup = 2

                self.KM_table_fig.add_trace(
                    go.Bar(
                        x=["Ciclo" if "ciclo" in k else "Paso"],
                        y=dat,
                        marker={"color": marker_color},
                        marker_line_color=marker_line_color,
                        text=f"{dat[0]:.2f}",
                        textfont={"color": "black"},
                        textposition="auto",
                        error_y={"array": err, "color": marker_line_color},
                        insidetextanchor="middle",
                        marker_line_width=1.5,
                        offsetgroup=offsetgroup,
                    ),
                    row=2,
                    col=3,
                )

        self.KM_table_fig.update_yaxes(visible=True, row=2, col=3)

        # Add step width
        vals = {k: v for k, v in data.items() if k in ["Ancho de paso (m)"]}

        for n, m in enumerate(vals.items()):
            marker_opacity = 0.6
            k, v = m
            dat = v["N"]["data"]
            err = [v["N"]["error"], v["N"]["error"]]
            self.KM_table_fig.add_trace(
                go.Bar(
                    x=["Control"],
                    y=dat,
                    error_y={"array": err},
                    text=f"{dat[0]:.2f}",
                    textfont={"color": "black"},
                    textposition="auto",
                    insidetextanchor="middle",
                    marker={"color": norm_color(marker_opacity)},
                    marker_line_color=norm_color(1),
                    marker_line_width=1.5,
                    offsetgroup=0,
                ),
                row=3,
                col=1,
            )

            for side in ["B"]:
                dat = v[side]["data"]
                # err = [v[side]["error"], v[side]["error"]]

                marker_color = both_color(marker_opacity)
                marker_line_color = both_color(1)
                offsetgroup = 0

                self.KM_table_fig.add_trace(
                    go.Bar(
                        x=["Total"],
                        y=dat,
                        marker={"color": marker_color},
                        marker_line_color=marker_line_color,
                        text=f"{dat[0]:.2f}",
                        textfont={"color": "black"},
                        textposition="auto",
                        # error_y={"array": err, "color": marker_line_color,},
                        insidetextanchor="middle",
                        marker_line_width=1.5,
                        offsetgroup=offsetgroup,
                    ),
                    row=3,
                    col=1,
                )

        self.KM_table_fig.update_yaxes(visible=True, row=3, col=1)

        # Add GDI
        vals = {k: v for k, v in data.items() if k in ["Gait Deviation Index"]}
        for n, m in enumerate(vals.items()):
            k, v = m
            marker_opacity = 0.6

            for side in ["R", "L"]:
                dat = v[side]["data"]
                if side == "R":
                    x = "Derecha"
                    marker_color = right_color(marker_opacity)
                    marker_line_color = right_color(1)
                    offsetgroup = 0
                elif side == "L":
                    x = "Izquierda"
                    marker_color = left_color(marker_opacity)
                    marker_line_color = left_color(1)
                    offsetgroup = 0

                self.KM_table_fig.add_trace(
                    go.Bar(
                        x=[x],
                        y=dat,
                        marker={"color": marker_color},
                        marker_line_color=marker_line_color,
                        text=f"{dat[0]:.1%}",
                        textfont={"color": "black"},
                        textposition="auto",
                        insidetextanchor="middle",
                        marker_line_width=1.5,
                        offsetgroup=offsetgroup,
                    ),
                    row=3,
                    col=2,
                )

        self.KM_table_fig.update_yaxes(visible=False, row=3, col=2)

        # Add GPS
        vals = {k: v for k, v in data.items() if k in ["Gait Profile Score"]}
        for n, m in enumerate(vals.items()):
            k, v = m
            marker_opacity = 0.6

            for side in ["R", "L"]:
                dat = v[side]["data"]
                if side == "R":
                    x = "Derecha"
                    marker_color = right_color(marker_opacity)
                    marker_line_color = right_color(1)
                    offsetgroup = 0
                elif side == "L":
                    x = "Izquierda"
                    marker_color = left_color(marker_opacity)
                    marker_line_color = left_color(1)
                    offsetgroup = 0

                self.KM_table_fig.add_trace(
                    go.Bar(
                        x=[x],
                        y=dat,
                        marker={"color": marker_color},
                        marker_line_color=marker_line_color,
                        text=f"{dat[0]:.2f}",
                        textfont={"color": "black"},
                        textposition="auto",
                        insidetextanchor="middle",
                        marker_line_width=1.5,
                        offsetgroup=offsetgroup,
                    ),
                    row=3,
                    col=3,
                )

        self.KM_table_fig.update_yaxes(visible=False, row=3, col=3)

        # Add GVS
        GPS_data = {i.label: i for i in stat._angles if i.label.startswith("aGVS")}
        GPS_data_translated = {}

        translations = (
            ("AFE", "Tobillo Flex/Ext"),
            ("AIE", "Pie Progres."),
            ("KFE", "Rodilla Flex/Ext"),
            ("HPFE", "Cadera Flex/Ext"),
            ("HPAA", "Cadera Ab/Ad"),
            ("HPIE", "Cadera Rot."),
            ("PTILT", "Pelvis Incl."),
            ("PROT", "Pelvis Rot."),
            ("POBLI", "Pelvis Obli."),
        )

        for k, v in GPS_data.items():
            for old, new in translations:
                if k.endswith(old):
                    GPS_data_translated[k] = new
                    break

        for n, m in enumerate(GPS_data.items()):
            marker_opacity = 0.6
            k, v = m
            dat = v.data[0]
            x = GPS_data_translated.pop(k, None)
            side = k[5]
            if side == "R":
                marker_color = right_color(marker_opacity)
                marker_line_color = right_color(1)
                offsetgroup = 0
            else:
                marker_color = left_color(marker_opacity)
                marker_line_color = left_color(1)
                offsetgroup = 1

            self.KM_table_fig.add_trace(
                go.Bar(
                    x=[x],
                    y=dat,
                    text=f"{dat[0]:.2f}",
                    textfont={"color": "black"},
                    textposition="auto",
                    insidetextanchor="middle",
                    marker={"color": marker_color},
                    marker_line_color=marker_line_color,
                    marker_line_width=1.5,
                    offsetgroup=offsetgroup,
                ),
                row=4,
                col=1,
            )

        self.KM_table_fig.update_yaxes(visible=False, row=4, col=1)

        for i in self.KM_table_fig["layout"]["annotations"]:
            i["font"] = {"size": self.plotly_size_m}
        self.KM_table_fig.update_layout(
            margin=go.layout.Margin(l=0, r=0, b=0, t=30),
            height=950,
            width=1000,
            showlegend=False,
        )
        root_path = Path(__file__).parent
        images = root_path / "images"
        css = root_path / "templates" / "EMG.css"
        header = images / "HeaderCIN170X23.png"

        # Weasyprint cannot handle embedded SVGs in the HTML yet, so we write
        # the SVG into a temporary file. Cannot use NamedTemporaryFile context
        # manager because the file couldn't be accessed by both Orca and Weasyprint
        # at the same time. Doing it manually...

        temp_svg = tempfile.mkstemp(suffix=".svg", text=True)

        # write SVG to tempfile
        self.KM_table_fig.write_image(temp_svg[1], scale=3, engine=RENDER_ENGINE)

        # simple template for the PDF page
        template = f"""
        <body>
            <img src='{header.resolve().as_uri()}'>
            <h2>Parámetros lineales</h2>
            <img src='{Path(temp_svg[1]).as_uri()}' class='main_plot'>
            <p style='font-size:xx-small'>
            Referencias:
            </p>
            <p style='font-size:xx-small'>
            <a href='https://doi.org/10.1016/j.gaitpost.2008.05.001'>†Schwartz, M. H., & Rozumalski, A. (2008). The gait deviation index: A new comprehensive index of gait pathology. Gait & Posture, 28(3), 351–357</a></p>
            <p style='font-size:xx-small'>
            <a href='https://doi.org/https://doi.org/10.1016/j.gaitpost.2009.05.020'>‡Baker, R., McGinley, J. L., Schwartz, M. H., Beynon, S., Rozumalski, A., Graham, H. K., & Tirosh, O. (2009). The Gait Profile Score and Movement Analysis Profile. Gait & Posture, 30(3), 265–269</a></p>
            {self.footer}
        </body>
        """

        # Weasyprint the template
        page = HTML(string=template, encoding="utf-8").render(stylesheets=self.styles)
        self.pages.append(page)

        # close ande remove tempfile
        os.close(temp_svg[0])
        os.unlink(temp_svg[1])

    def ACV_lissajous(self):
        events = self.trial_mdx.trial.static.events
        freq = self.trial_mdx.trial.markerStream.frequency
        heel_strikes = events["eRHS"].data + events["eLHS"].data
        rang = [min(heel_strikes), max(heel_strikes)]
        rang = [int(i * freq) for i in rang]
        print(f"RANG: {rang}")
        mark = self.trial_mdx.trial.markerStream.tracks
        # if not is_facing_forward(mark):
        #     pivot = mark["MIDASIS"].data[:, rang[0] : rang[1]]
        #     print("ROTATING")
        #     r = Rotation.from_euler("Y", 180, degrees=True)
        #     # TODO: check that this is a valid rotation
        #     mark = {
        #         k: r.apply(v.data[:, rang[0] : rang[1]].T - pivot.T).T + pivot
        #         for k, v in mark.items()
        #     }
        # else:
        paired_cmap = get_cmap("Paired")
        colors = [f"rgba{paired_cmap(i)}" for i in np.linspace(0, 1, 11)]
        if is_facing_forward(mark):
            mark = {k: v.data[:, rang[0] : rang[1]] for k, v in mark.items()}
        else:
            pivot = mark["MIDASIS"].data[:, rang[0] : rang[1]]
            print("ROTATING")
            r = Rotation.from_euler("y", 180, degrees=True)
            # TODO: check that this is a valid rotation
            mark = {
                k: r.apply(v.data[:, rang[0] : rang[1]].T - pivot.T).T + pivot
                for k, v in mark.items()
            }
            for k, v in mark.items():
                mark[k][0] = np.flip(v[0])

        sacrum = mark["sacrum"]
        center = (mark["MIDASIS"] - sacrum) / 2 + sacrum
        markers = {
            "shoulder": {
                "data": [mark[f"{s} should s"] - center for s in ["r", "l"]],
                "color": colors[0:2],
            },
            "hip": {
                "data": [mark[f"{s}HP"] - center for s in ["R", "L"]],
                "color": colors[2:4],
            },
            "knee": {
                "data": [mark[f"{s}K"] - center for s in ["R", "L"]],
                "color": colors[4:6],
            },
            "ankle": {
                "data": [mark[f"{s}A"] - center for s in ["R", "L"]],
                "color": colors[6:8],
            },
            "met": {
                "data": [mark[f"{s} met s"] - center for s in ["r", "l"]],
                "color": colors[8:10],
            },
        }

        self.ACV_lissajous_fig = make_subplots(
            rows=4,
            cols=2,
            specs=[
                [{"rowspan": 3}, {"rowspan": 3}],
                [{}, {}],
                [{}, {}],
                [{"colspan": 1}, None],
            ],
        )
        s = sacrum - center
        # plot center
        self.ACV_lissajous_fig.add_trace(
            go.Scatter(
                x=s[0], y=s[1], name=f"COP", mode="lines", marker={"color": colors[10]}
            ),
            row=1,
            col=1,
        )

        self.ACV_lissajous_fig.add_trace(
            go.Scatter(
                x=s[2],
                y=s[1],
                name=f"COP",
                mode="lines",
                marker={"color": colors[10]},
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        self.ACV_lissajous_fig.add_trace(
            go.Scatter(
                x=s[2],
                y=s[0],
                name=f"COP",
                mode="lines",
                marker={"color": colors[10]},
                showlegend=False,
            ),
            row=4,
            col=1,
        )

        for k, v in markers.items():
            for n, dat in enumerate(v["data"]):
                side = "L" if n else "R"
                self.ACV_lissajous_fig.add_trace(
                    go.Scatter(
                        x=dat[0],
                        y=dat[1],
                        name=f"{side} {k}",
                        mode="lines",
                        marker={"color": v["color"][n]},
                    ),
                    row=1,
                    col=1,
                )

                self.ACV_lissajous_fig.add_trace(
                    go.Scatter(
                        x=dat[2],
                        y=dat[1],
                        name=f"{side} {k}",
                        mode="lines",
                        marker={"color": v["color"][n]},
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )

                self.ACV_lissajous_fig.add_trace(
                    go.Scatter(
                        x=dat[2],
                        y=dat[0],
                        name=f"{side} {k}",
                        mode="lines",
                        marker={"color": v["color"][n]},
                        showlegend=False,
                    ),
                    row=4,
                    col=1,
                )

        self.ACV_lissajous_fig.update_xaxes(range=[-0.7, 0.7], row=1, col=1)
        self.ACV_lissajous_fig.update_xaxes(range=[-0.7, 0.7], row=1, col=2)
        self.ACV_lissajous_fig.update_xaxes(range=[-0.7, 0.7], row=4, col=1)
        self.ACV_lissajous_fig.update_yaxes(range=[-0.7, 0.7], row=4, col=1)

        self.ACV_lissajous_fig.update_layout(
            margin=go.layout.Margin(l=0, r=0, b=0, t=30),
            height=1000,
            width=1000,
            showlegend=True,
            legend={"x": 0.75, "y": 0},
            # template="seaborn",
        )
        root_path = Path(__file__).parent
        images = root_path / "images"
        css = root_path / "templates" / "EMG.css"
        header = images / "HeaderCIN170X23.png"

        # Weasyprint cannot handle embedded SVGs in the HTML yet, so we write
        # the SVG into a temporary file. Cannot use NamedTemporaryFile context
        # manager because the file couldn't be accessed by both Orca and Weasyprint
        # at the same time. Doing it manually...

        temp_svg = tempfile.mkstemp(suffix=".svg", text=True)

        # write SVG to tempfile
        self.ACV_lissajous_fig.write_image(temp_svg[1], scale=3, engine=RENDER_ENGINE)

        # simple template for the PDF page
        template = f"""
        <body>
            <img src='{header.resolve().as_uri()}'>
            <h2>Diagramas de Lissajous</h2>
            <img src='{Path(temp_svg[1]).as_uri()}' class='main_plot'>
            {self.footer}
        </body>
        """

        # Weasyprint the template
        page = HTML(string=template, encoding="utf-8").render(stylesheets=self.styles)
        self.pages.append(page)

        # close ande remove tempfile
        os.close(temp_svg[0])
        os.unlink(temp_svg[1])

    def ACV_abnormal_gait(self):
        self.ACV_abnormal_fig = go.Figure(
            data=go.Scatterpolar(
                r=[80, 20, 100, 60, 85, 55, 75, 110, 45, 35, 80],
                theta=[
                    "&nbsp;&nbsp;&nbsp;&nbsp;Desplazamiento<br>&nbsp;&nbsp;&nbsp;&nbsp;lateral<br>&nbsp;&nbsp;&nbsp;&nbsp;del tronco",
                    "Contacto antepie",
                    "Golpe extensor de rodilla",
                    "Retropulsión <br>de cadera",
                    "Rodilla<br>rígida",
                    "Vaulting<br>contralateral",
                    "Fexión de rodilla<br>insuficiente en balanceo",
                    "Latigazo medial",
                    "Circunducción",
                    "Hip hiking",
                    "&nbsp;&nbsp;&nbsp;&nbsp;Desplazamiento<br>&nbsp;&nbsp;&nbsp;&nbsp;lateral<br>&nbsp;&nbsp;&nbsp;&nbsp;del tronco",
                ],
                fill="toself",
                connectgaps=True,
                mode="markers",
            )
        )

        self.ACV_abnormal_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 120])), showlegend=False
        )

        self.ACV_abnormal_fig.update_layout(
            margin=go.layout.Margin(l=150, r=150, b=0, t=30),
            height=1100,
            width=1100,
            # template="plotly_white",
            font={"size": self.plotly_size_sm}
            # showlegend=True,
            # legend={"x": 0.75, "y": 0},
            # template="seaborn",
        )
        print(self.ACV_abnormal_fig["layout"])
        root_path = Path(__file__).parent
        images = root_path / "images"
        css = root_path / "templates" / "EMG.css"
        header = images / "HeaderCIN170X23.png"

        # Weasyprint cannot handle embedded SVGs in the HTML yet, so we write
        # the SVG into a temporary file. Cannot use NamedTemporaryFile context
        # manager because the file couldn't be accessed by both Orca and Weasyprint
        # at the same time. Doing it manually...

        temp_svg = tempfile.mkstemp(suffix=".svg", text=True)

        # write SVG to tempfile
        self.ACV_abnormal_fig.write_image(temp_svg[1], scale=3, engine=RENDER_ENGINE)

        # simple template for the PDF page
        template = f"""
        <body>
            <img src='{header.resolve().as_uri()}'>
            <h2>Indices de abnormalidad de la marcha</h2>
            <img src='{Path(temp_svg[1]).as_uri()}' class='main_plot'>
            {self.footer}
        </body>
        """

        # Weasyprint the template
        page = HTML(string=template, encoding="utf-8").render(stylesheets=self.styles)
        self.pages.append(page)

        # close ande remove tempfile
        os.close(temp_svg[0])
        os.unlink(temp_svg[1])
        # gt = GaitIndices(self.trial_mdx.trial)
        # r_hiking, l_hiking = gt.hip_hiking()
        # r_hiking_mean = np.mean(r_hiking)  # If more than one cycle, average them
        # r_hiking_std = np.std(r_hiking)
        # l_hiking_mean = np.mean(l_hiking)  # If more than one cycle, average them
        # l_hiking_std = np.std(l_hiking)
        #
        # Wait for Crespo. Try empty Radial graph
        # print(r_hiking_mean, r_hiking_std, l_hiking)

    def EMG_static_report(
        self,
        color_from_envelope=False,
        plot_data=True,
        plot_envelope=False,
        emg_labels=[],
    ):
        """Prints EMG static report for the current patient"""
        if not self.trial_mdx.trial.has_emg:
            print(f"EMG report: NO EMG in {self.trial_path}. Return")
            return

        emg_channels = self.trial_mdx.trial.emgStream.emgChannels
        emg_freq = int(self.trial_mdx.trial.emgStream.frequency)
        nFrames = self.trial_mdx.trial.emgStream._nFrames
        events = self.trial_mdx.trial.static.events

        # Plot between the first heel strike and the last heel strike
        try:
            events_range = [
                min(events["eRHS"].data + events["eLHS"].data),
                max(events["eRHS"].data + events["eLHS"].data),
            ]
        except:
            print("Problems detecting events. Might not have any!")
            return

        # Remove processed emgs
        emg_channels = {k: v.data[0] for k, v in emg_channels.items() if "~" not in k}

        # cut to only the selected step
        for k, v in emg_channels.items():
            if f"Left " in k:
                ev = [int(i * emg_freq) for i in sorted(events["eLHS"].data)]
            elif f"Right " in k:
                ev = [int(i * emg_freq) for i in sorted(events["eRHS"].data)]
            emg_channels[k] = [v[ev[i] : ev[i + 1]] for i in range(len(ev) - 1)]

        # apply the bandpass filter
        emg_filtered = {k: [bandpass(i) for i in v] for k, v in emg_channels.items()}

        # The desired labels for this page
        if not emg_labels:
            emg_labels = [
                "Rectus femoris",
                "Semimembranosus",
                "Tibialis anterior",
                "Gastrocnemius medialis",
                "Vastus lateralis",
                "Vastus medialis",
            ]

        right_emg_labels = [
            f"Right {i}" for i in emg_labels if f"Right {i}" in emg_channels.keys()
        ]

        left_emg_labels = [
            f"Left {i}" for i in emg_labels if f"Left {i}" in emg_channels.keys()
        ]

        emg_channels = {
            k: v
            for k, v in emg_channels.items()
            if k in (right_emg_labels + left_emg_labels)
        }

        def translate_label(old_label):
            """Translates labels into Spanish"""
            w = {
                "Rectus femoris": "Recto femoral",
                "Semimembranosus": "Semimembranoso",
                "Semitendinosus": "Semitendinoso",
                "Soleus": "Sóleo",
                "Tibialis anterior": "Tibial anterior",
                "Gastrocnemius medialis": "Gastrocnemio medial",
                "Vastus lateralis": "Vasto lateral",
                "Gluteus medius": "Gluteo medio",
                "Vastus medialis": "Vasto medial",
                "Biceps femoris caput longus": "Biceps femoral",
            }
            side = {"Right": "derecho", "Left": "izquierdo"}
            new_label = ""
            for eng, spa in w.items():
                if eng in old_label:
                    new_label += spa
                    break
            for eng, spa in side.items():
                if old_label.startswith(eng):
                    new_label += f" {spa}"
            if new_label == "":
                raise Exception(f"Can't translate {old_label} into Spanish")
            return new_label

        # Create the empty figure
        self.EMG_stat_fig = EMG_Figure(
            right_emg_labels=right_emg_labels,
            left_emg_labels=left_emg_labels,
            max_value_all_channels=0.0005,
            max_energy_all_channels=0.0004,
            freq=emg_freq,
        )

        # Add normality values to all labels
        for k in self.EMG_stat_fig.mapa.keys():
            self.EMG_stat_fig.add_normality(k, self.normative_mdx)

        # Add toe offs
        RTO = self.trial_mdx.trial.cycle.events["enRTO"].data[0][0][0]
        for k in right_emg_labels:
            self.EMG_stat_fig.add_event(
                k, RTO, dict(color="green", width=2, dash="longdash")
            )

        LTO = self.trial_mdx.trial.cycle.events["enLTO"].data[0][0][0]
        for k in left_emg_labels:
            self.EMG_stat_fig.add_event(
                k, LTO, dict(color="red", width=2, dash="longdash")
            )

        # Add contralateral toe offs
        try:
            LTORCycle = self.trial_mdx.trial.cycle.events["enLTORCycle"].data[0][0][0]
        except Exception as e:
            LTORCycle = []
            print("Sin LTORCycle")

        for k in right_emg_labels:
            self.EMG_stat_fig.add_event(
                k, LTORCycle, dict(color="red", width=2, dash="dash"), y1=1, y0=0
            )

        try:
            LHSRCycle = self.trial_mdx.trial.cycle.events["enLHSRCycle"].data[0][0][0]
        except Exception as e:
            LHSRCycle = []
            print("Sin LTORCycle")

        for k in right_emg_labels:
            self.EMG_stat_fig.add_event(
                k, LHSRCycle, dict(color="red", width=2, dash="solid"), y1=1, y0=0
            )

        try:
            RTOLCycle = self.trial_mdx.trial.cycle.events["enRTOLCycle"].data[0][0][0]
        except Exception as e:
            RTOLCycle = []
            print("Sin RTOLCycle")

        for k in left_emg_labels:
            self.EMG_stat_fig.add_event(
                k, RTOLCycle, dict(color="green", width=2, dash="dash"), y1=1, y0=0
            )

        try:
            RHSLCycle = self.trial_mdx.trial.cycle.events["enRHSLCycle"].data[0][0][0]
        except Exception as e:
            RHSLCycle = []
            print("Sin RHSLCycle")

        for k in left_emg_labels:
            self.EMG_stat_fig.add_event(
                k, RHSLCycle, dict(color="green", width=2, dash="solid"), y1=1, y0=0
            )

        for k in left_emg_labels:
            self.EMG_stat_fig.add_event(
                k, RTOLCycle, dict(color="green", width=2, dash="dash"), y1=1, y0=0
            )

        # add traces
        for k in left_emg_labels:
            for i in emg_filtered[k]:
                self.EMG_stat_fig.add_trace(
                    k,
                    i,
                    plot_envelope=plot_envelope,
                    plot_data=plot_data,
                    color_from_envelope=color_from_envelope,
                    legendgroup="Left",
                )
        for k in right_emg_labels:
            for i in emg_filtered[k]:
                self.EMG_stat_fig.add_trace(
                    k,
                    i,
                    plot_envelope=plot_envelope,
                    plot_data=plot_data,
                    color_from_envelope=color_from_envelope,
                    legendgroup="Right",
                )

        self.EMG_stat_fig.adjust_xaxes()
        self.EMG_stat_fig.colorscale(color_from_envelope=True)

        def title_color(t):
            if "Right " in t or "derecho" in t:
                return "green"
            elif "Left " in t or "izquierdo" in t:
                return "red"
            else:
                return "black"

        # Update title colours according to left and right
        for i in self.EMG_stat_fig.fig["layout"]["annotations"]:
            i["font"] = {
                "size": self.plotly_size,
                "color": title_color(i["text"]),
                
            }
            i["text"] = translate_label(i["text"])

        self.EMG_stat_fig.fig.update_layout(
            title=None,
            showlegend=False,
            template="plotly_white",
            font={"size": self.plotly_size_sm},
            margin=go.layout.Margin(l=0, r=0, b=0, t=30),
            height=1060,
            width=1000,
        )

        header = self.images / "HeaderEMG170X23.png"

        # Weasyprint cannot handle embedded SVGs in the HTML yet, so we write
        # the SVG into a temporary file. Cannot use NamedTemporaryFile context
        # manager because the file couldn't be accessed by both Orca and Weasyprint
        # at the same time. Doing it manually...

        temp_svg = tempfile.mkstemp(suffix=".svg", text=True)

        # write SVG to tempfile
        self.EMG_stat_fig.fig.write_image(temp_svg[1], scale=3, engine=RENDER_ENGINE)

        # simple template for the PDF page
        template = f"""
        <body>
            <img src='{header.resolve().as_uri()}'>
            <h2>Electromiografía dinámica de superficie</h2>
            <img src='{Path(temp_svg[1]).as_uri()}' class='main_plot'>
            {self.footer}
        </body>
        """
        # Weasyprint the template
        page = HTML(string=template).render(stylesheets=self.styles)
        self.pages.append(page)

        # close ande remove tempfile
        os.close(temp_svg[0])
        os.unlink(temp_svg[1])

    def KM_static_report(
        self,
        autorange=False,
        mark_out_of_range=False,
        out_of_range_opts=None,
        mark_static_angles=False,
        plot_sides={},
        plot_events={},
        plot_mean_events={},
        plot_normative=True,
        normative_options={},
        title="",
        legend_options={},
        legend_orientation = "v",
        running =False
    ):
        """Prints the kinematic report for this trial into a PDF"""

        if not self.trial_mdx.trial.has_kinematics:
            print(f"KM report: NO angles in {self.trial_path}. Return")
            return

        cycle_data = self.trial_mdx.trial.cycle
        # cycle_data_score = self.trial_mdx.trial.cycle
        cycle_normative = self.normative_mdx.norm.cycle

        trial_angles_bands = cycle_data.angle_bands
        trial_angles = cycle_data.angles

        normative_angles = cycle_normative.angle_bands

        if running:
            self.KM_stat_fig = KM_Figure_Running()
            required_angle_codes = KM_Figure_Running.angles_required.keys()
            
            if plot_normative:
                for i in required_angle_codes:
                    if (i != "FAA" and  i != "SHTILTOFF" and  i != "SHOBLI" and  i != "SHROT" 
                        and  i != "PTILT" and  i != "POBLI" and  i != "PROT"): #TODO borrar
                            self.KM_stat_fig.add_norm(
                                i,
                                normative_angles[f"a{i}"]["mean"],
                                normative_angles[f"a{i}"]["sd"],
                            )
        else:
            self.KM_stat_fig = KM_Figure()
            required_angle_codes = KM_Figure.angles_required.keys()

            if plot_normative:
                for i in required_angle_codes:
                    self.KM_stat_fig.add_norm(
                        i,
                        normative_angles[f"a{i}"]["mean"],
                        normative_angles[f"a{i}"]["sd"],
                    )

        for side, opts in plot_sides.items():
            use_mean = opts.pop("mean", False)
            ### NUEVO
            # Si es un side de SCORE, leer datos del mdx de SCORE
            if side.startswith("SCORE_"):
                if not self.has_score:
                    continue
                real_side = side.replace("SCORE_", "")
                for i in required_angle_codes:
                    if use_mean:
                        band = self._score_value("angle_bands", f"acm{real_side}{i}")
                        if band is not None:
                            mean, sd = band
                            self.KM_stat_fig.add_mean_trace(
                                i, f"SCORE {real_side}", mean, sd, line_opts=opts
                            )
                    else:
                        c_data = self._score_value("angles", f"ac{real_side}{i}")
                        if c_data is not None:
                            for cycle in c_data:
                                self.KM_stat_fig.add_trace(
                                    i, f"SCORE {real_side}", cycle[0], line_opts=opts
                                )
            else:
            ### fin nuevo
                for i in required_angle_codes:
                    if use_mean:
                        mean, sd = trial_angles_bands[f"acm{side}{i}"].data
                        self.KM_stat_fig.add_mean_trace(
                            i, f"{side}", mean, sd, line_opts=opts
                        )
                    else:
                        c_data = trial_angles[f"ac{side}{i}"].data
                        for cycle in c_data:
                            self.KM_stat_fig.add_trace(
                                i, f"{side}", cycle[0], line_opts=opts
                            )
        self.KM_stat_fig.adjust_range(auto=autorange)

        # Add mean events
        for name, opts in plot_mean_events.items():
            pos = opts.pop("position", "top")
            length = opts.pop("length", 1)
            legend_text = opts.pop("legend_text", "")
            show_mean = opts.pop("show_mean", True)
            show_sd = opts.pop("show_sd", True)
            mean_ev, sd_ev = cycle_data.event_bands[f"enm{name}"].data
            print(f"Adding mean event {name} to {side}")
            self.KM_stat_fig.add_mean_event(
                mean_ev[0],
                sd_ev[0],
                pos=pos,
                length=length,
                legend_text=legend_text,
                show_mean=show_mean,
                show_sd=show_sd,
                line_opts=opts,
            )

        # Add individual events
        for name, opts in plot_events.items():
            pos = opts.pop("position", "top")
            length = opts.pop("length", 1)
            legend_text = opts.pop("legend_text", "")
            ev = cycle_data.events[f"en{name}"].data
            to_set = {
                i[0][0] for i in ev
            }  # make it a set, so as not to plot the same event more than once
            print(f"Adding {len(to_set)} unique {name} events to side {side}")
            self.KM_stat_fig.add_event(
                to_set, pos=pos, length=length, legend_text=legend_text, linestyle=opts
            )

        self.KM_stat_fig.add_annotations()
        if mark_out_of_range:
            legend_text = out_of_range_opts.pop("legend_text", "Fuera de escala")
            self.KM_stat_fig.mark_out_of_scale(
                legend_text=legend_text, line_opts=out_of_range_opts
            )

        # if mark_static_angles:
        #     stat_angles = {
        #         i.label: i for i in self.standing_mdx.trial.static._angle_bands
        #     }
        #     for k, v in current_angles.items():
        #         v["stat_data"] = {
        #             "right": stat_angles[f"amR{v['code']}"].data,
        #             "left": stat_angles[f"amL{v['code']}"].data,
        #         }

        # Adjust figure size, margins, and legend location
        self.KM_stat_fig.fig.update_layout(
            template="plotly_white", hovermode="x unified"
        )
        # self.KM_stat_fig.fig.write_html("cinematica.html")
        self.KM_stat_fig.fig.update_layout(
            margin=go.layout.Margin(l=0, r=0, b=0, t=30),
            height=1060,
            width=1000,legend_orientation= legend_orientation,
            legend={
                **{
                    "xanchor": "left",
                    "yanchor": "bottom",
                    "x": 0.05,
                    "y": -0.02,
                    "font": {"size": self.plotly_size_m},
                },
                **legend_options,
            },
        )

        # print(self.KM_stat_fig["layout"])
        # self.KN_stat_fig.show()
        root_path = Path(__file__).parent
        images = root_path / "images"
        css = root_path / "templates" / "EMG.css"
        header = images / "HeaderCIN170X23.png"

        # Weasyprint cannot handle embedded SVGs in the HTML yet, so we write
        # the SVG into a temporary file. Cannot use NamedTemporaryFile context
        # manager because the file couldn't be accessed by both Orca and Weasyprint
        # at the same time. Doing it manually...

        temp_svg = tempfile.mkstemp(suffix=".svg", text=True)

        # write SVG to tempfile
        self.KM_stat_fig.fig.write_image(temp_svg[1], scale=3, engine=RENDER_ENGINE)
        # self.KM_stat_fig.fig.write_json("final_fig.json",encod)
        # self.KM_stat_fig.write_image("salida.png", scale=3, engine="orca")
        # simple template for the PDF page

        if not title:
            title = f"Cinemática{' (escala automática)' if autorange else ''}"
        template = f"""
        <body>
            <img src='{header.resolve().as_uri()}'>
            <h2>{title}</h2>
            <img src='{Path(temp_svg[1]).as_uri()}' class='main_plot'>
            {self.footer}
        </body>
        """
        # Weasyprint the template
        page = HTML(string=template).render(stylesheets=self.styles)
        self.pages.append(page)

        # close ande remove tempfile
        os.close(temp_svg[0])
        os.unlink(temp_svg[1])

    def SYNC_static_report(self, side):
        """Prints the synced kinematic + kinetic + emg report for this trial into a PDF"""

        side = side.upper()

        # proportional heights between graphs and emgs
        heights = (3, 1)

        if not self.trial_mdx.trial.has_emg:
            print(f"SYNC report: NO EMG in {self.trial_path}. Return")
            return

        if not self.trial_mdx.trial.has_kinematics:
            print(f"SYNC report: NO angles in {self.trial_path}. Return")
            return

        if not self.trial_mdx.trial.has_kinetics:
            print(f"SYNC report: NO forces in {self.trial_path}. Return")
            return

        angles = self.trial_mdx.trial.cycle.angles
        torques = self.trial_mdx.trial.cycle.torques1d
        if f"tc{side}HPFE" not in torques:
            print(f"SYNC report: NO torques for side {side}")
            return
        powers = self.trial_mdx.trial.cycle.powers
        emg = self.trial_mdx.trial.emgStream.emgChannels

        side_check = "Left" if side == "L" else "Right"
        if not any(i for i in emg.keys() if i.startswith(side_check)):
            print(f"SYNC report: NO EMG for side {side}")
            return

        joints_to_plot = {
            "cadera": {
                "code": "HP",
                "col": 0,
                "min": -20,
                "max": 60,
                "texts": ["Flex", "Ext"],
            },
            "rodilla": {
                "code": "K",
                "col": 1,
                "min": -20,
                "max": 80,
                "texts": ["Flex", "Ext"],
            },
            "tobillo": {
                "code": "A",
                "col": 2,
                "min": -30,
                "max": 30,
                "texts": ["Dors", "Plant"],
            },
        }
        ### NUEVO
        has_score = self.has_score
        ### fin nuevo

        for k, v in joints_to_plot.items():
            v["angle"] = {
                "min": v["min"],
                "max": v["max"],
                "texts": v["texts"],
                "data": angles[f"ac{side}{v['code']}FE"].data[0][0],
                "norm": self.normative_mdx.norm.cycle.angle_bands[f"a{v['code']}FE"],
            }
            v["power"] = {
                "min": -2,
                "max": 3,
                "texts": ["Gen", "Abs"],
                "data": powers[f"pc{side}{v['code']}FE"].data[0][0],
                "norm": self.normative_mdx.norm.cycle.power_bands[f"pcm{v['code']}FE"],
            }

            v["torque"] = {
                "min": -1,
                "max": 2,
                "texts": ["Ext", "Flex"],
                "data": torques[f"tc{side}{v['code']}FE"].data[0][0],
                "norm": self.normative_mdx.norm.cycle.torque1d_bands[
                    f"tcm{v['code']}FE"
                ],
            }

            ### NUEVO
            # Datos SCORE por articulación (si existen)
            v["angle"]["score_data"] = None
            v["torque"]["score_data"] = None
            v["power"]["score_data"] = None
            if has_score:
                angle_data = self._score_value("angles", f"ac{side}{v['code']}FE")
                if angle_data is not None:
                    v["angle"]["score_data"] = angle_data[0][0]
                torque_data = self._score_value("torques1d", f"tc{side}{v['code']}FE")
                if torque_data is not None:
                    v["torque"]["score_data"] = torque_data[0][0]
                power_data = self._score_value("powers", f"pc{side}{v['code']}FE")
                if power_data is not None:
                    v["power"]["score_data"] = power_data[0][0]
            ### fin nuevo

        titles = (
            [
                "Flexoextensión de cadera",
                "Flexoextensión de rodilla",
                "Dors/plant. de tobillo",
            ]
            + [None] * 6
            + [f"Momento de {i}" for i in joints_to_plot.keys()]
            + [None] * 6
            + [f"Potencia de {i}" for i in joints_to_plot.keys()]
        )

        x_values = np.linspace(0, 1, 100)

        self.SYNC_stat_fig = make_subplots(
            rows=7,
            cols=3,
            subplot_titles=titles,
            shared_xaxes=False,
            shared_yaxes=False,
            vertical_spacing=0.06,
            horizontal_spacing=0.10,
            # print_grid=True,  # Just to get the axis names
            row_heights=[
                heights[0],
                heights[1],
                heights[1],
                heights[0],
                heights[1],
                heights[1],
                heights[0],
            ],
        )
        self.SYNC_stat_fig.update_layout(
            template="plotly_white", font={"size": self.plotly_size_sm}
        )

        # Add the traces
        new_annotations = []
        for n, m in enumerate(joints_to_plot.items()):
            k, v = m
            for nsec, section in enumerate(["angle", "torque", "power"]):
                row = nsec * 3 + 1
                self.SYNC_stat_fig.add_trace(
                    go.Scatter(
                        y=v[section]["norm"]["mean"] + v[section]["norm"]["sd"],
                        name=k,
                        x=x_values,
                        line=dict(width=0, color="grey"),
                        legendgroup="mean",
                        showlegend=False,
                        # marker={"coloraxis": "coloraxis"},
                    ),
                    row=row,
                    col=v["col"] + 1,
                )
                self.SYNC_stat_fig.add_trace(
                    go.Scatter(
                        y=v[section]["norm"]["mean"] - v[section]["norm"]["sd"],
                        x=x_values,
                        line=dict(width=0, color="grey"),
                        legendgroup="mean",
                        showlegend=False,
                        fill="tonexty",
                        fillcolor="rgba(125,125,125,0.3)"
                        # marker={"coloraxis": "coloraxis"},
                    ),
                    row=row,
                    col=v["col"] + 1,
                )
                self.SYNC_stat_fig.add_trace(
                    go.Scatter(
                        y=v[section]["norm"]["mean"],
                        name="Valores normales",
                        x=x_values,
                        line=dict(width=2, color="grey"),
                        legendgroup="mean",
                        showlegend=False,
                        # showlegend=True if nsec + v["col"] == 0 else False,
                        # marker={"coloraxis": "coloraxis"},
                    ),
                    row=row,
                    col=v["col"] + 1,
                )
                self.SYNC_stat_fig.add_trace(
                    go.Scatter(
                        y=v[section]["data"],
                        x=x_values,
                        name="Derecha" if side == "R" else "Izquierda",
                        line=dict(width=3, color="green" if side == "R" else "red"),
                        legendgroup="right" if side == "R" else "left",
                        showlegend=False,
                        # showlegend=True if nsec + v["col"] == 0 else False,
                        # marker={"coloraxis": "coloraxis"},
                    ),
                    row=row,
                    col=v["col"] + 1,
                )

                ### NUEVO
                # Curva SCORE (si existe) con línea punteada del mismo color
                if v[section]["score_data"] is not None:
                    self._add_score_scatter(
                        self.SYNC_stat_fig,
                        y=v[section]["score_data"],
                        x=x_values,
                        side=side,
                        row=row,
                        col=v["col"] + 1,
                        legendgroup="score_right" if side == "R" else "score_left",
                        showlegend=False,
                        width=2,
                    )
                ### fin nuevo

                for i in add_event_line(
                    self.trial_mdx.trial,
                    f"en{side}TO",
                    line=dict(
                        color="red" if side == "L" else "green",
                        width=2,
                        dash="longdash",
                    ),
                    normalized=True,
                    y1=v[section]["max"],
                    y0=v[section]["min"],
                ):
                    self.SYNC_stat_fig.add_shape(i, row=row, col=v["col"] + 1)

                if section == "angle":
                    y_title = "(°)"
                elif section == "torque":
                    y_title = "N m/Kg"
                elif section == "power":
                    y_title = "W/Kg"

                self.SYNC_stat_fig.update_yaxes(
                    range=[v[section]["min"], v[section]["max"]],
                    row=row,
                    col=v["col"] + 1,
                    title_text=y_title,
                    # tickmode="linear"
                )

                self.SYNC_stat_fig.update_xaxes(
                    # range=[0, 1],
                    tickformat=",.0%",
                    dtick=0.2,
                    row=row,
                    col=v["col"] + 1,
                    title_text="Ciclo (%)",
                )

                # distance beteween
                dist = (v[section]["max"] - v[section]["min"]) * 0.1
                index = 3 * (row - 1) + n + 1
                for i, text in enumerate(v[section]["texts"]):
                    # Exception: anke moment texts are Plant and Dors, not
                    # Ext and Flex
                    if index in [3, 12]:
                        if text == "Flex":
                            text = "Dors"
                        elif text == "Ext":
                            text = "Plant"
                    new_annotations.append(
                        {
                            "text": text,
                            "xref": f"x{index}",
                            "y": v[section]["min"] + dist
                            if i
                            else v[section]["max"] - dist,
                            "yref": f"y{index}",
                            "x": 0.9,
                            "yanchor": "middle",
                            "xanchor": "auto",
                            "font": {"size": self.plotly_size_m},
                            "showarrow": False,
                            "align": "left",
                        }
                    )

                    # Add EMG traces
        emg_to_plot = {
            "semimembranosus": {"short": "SM", "pos": ((2, 1), (5, 2))},
            "rectus femoris": {"short": "RF", "pos": ((2, 2), (5, 1))},
            "gastrocnemius medialis": {"short": "GS", "pos": ((3, 3), (6, 2))},
            "vastus lateralis": {"short": "VL", "pos": ((3, 2),)},
            "tibialis anterior": {"short": "AT", "pos": ((5, 3),)},
        }
        events = self.trial_mdx.trial.static.events
        emg_freq = self.trial_mdx.trial.emgStream.frequency
        ev = sorted([int(i * emg_freq) for i in events[f"e{side}HS"].data])
        logging.debug(f"events: {ev}")
        # Clean emg channels not available in this trial
        emg_to_plot = {
            k: v
            for k, v in emg_to_plot.items()
            if f"{'Right' if side == 'R' else 'Left'} {k.capitalize()}" in emg
        }

        for k, v in emg_to_plot.items():
            logging.debug(k)
            # apply the bandpass filter
            dat = emg[f"{'Right' if side == 'R' else 'Left'} {k.capitalize()}"].data[0][
                ev[0] : ev[1]
            ]

            logging.debug(dat)

            v["data"] = bandpass(dat)

        for k, v in emg_to_plot.items():
            for row, col in v["pos"]:
                max_value_scale = max(v["data"])
                min_value_scale = min(v["data"])
                # remove this axis name, annotations, etc
                self.SYNC_stat_fig.update_xaxes(visible=False, row=row, col=col)
                self.SYNC_stat_fig.update_yaxes(
                    visible=True,
                    row=row,
                    col=col,
                    title=v["short"],
                    showticklabels=False,
                )
                for shape in self.add_normality_region_emg(
                    k.capitalize(), side, normalized=True
                ):
                    self.SYNC_stat_fig.add_shape(shape, row=row, col=col)
                self.SYNC_stat_fig.update_yaxes(
                    range=[min_value_scale, max_value_scale],
                    row=row,
                    col=col,
                    showgrid=False,
                    # nticks=8,
                    # tickmode="auto",
                )
                for i in add_event_line(
                    self.trial_mdx.trial,
                    f"en{side}TO",
                    line=dict(
                        color="red" if side == "L" else "green",
                        width=2,
                        dash="longdash",
                    ),
                    normalized=True,
                    y1=max_value_scale,
                    y0=min_value_scale,
                ):
                    self.SYNC_stat_fig.add_shape(i, row=row, col=col)
                self.SYNC_stat_fig.add_trace(
                    go.Scatter(
                        y=v["data"],
                        x=np.linspace(0, 1, len(v["data"])),
                        # name=i,
                        line=dict(width=1, color="black"),
                        # legendgroup="filtered",
                        # showlegend=False if n else True,
                        # marker={"coloraxis": "coloraxis"},
                    ),
                    row=row,
                    col=col,
                )

        for i in self.SYNC_stat_fig["layout"]["annotations"]:
            i["font"] = {"size": self.plotly_size}
            new_annotations.append(i)

        self.SYNC_stat_fig["layout"]["annotations"] = new_annotations

        # Adjust figure size, margins, and legend location
        self.SYNC_stat_fig.update_layout(
            margin=go.layout.Margin(l=0, r=0, b=0, t=30),
            height=950,
            width=1000,
            showlegend=False,
            # legend={
            #     "xanchor": "left",
            #     "yanchor": "bottom",
            #     "x": 0.05,
            #     "y": -0.5,
            #     "traceorder": "reversed",
            #     "font": {"size": self.plotly_size},
            # },
        )
        # # print(self.KM_stat_fig["layout"])
        # # self.KN_stat_fig.show()
        root_path = Path(__file__).parent
        images = root_path / "images"
        css = root_path / "templates" / "EMG.css"
        header = images / "HeaderCIN+DIN+EMG170X23.png"

        # Weasyprint cannot handle embedded SVGs in the HTML yet, so we write
        # the SVG into a temporary file. Cannot use NamedTemporaryFile context
        # manager because the file couldn't be accessed by both Orca and Weasyprint
        # at the same time. Doing it manually...

        temp_svg = tempfile.mkstemp(suffix=".svg", text=True)

        refs = [
            f"<p>{emg_to_plot[k]['short']}:  {k.capitalize()}</p>" for k in emg_to_plot
        ]
        refs = "\n".join(refs)

        # write SVG to tempfile
        self.SYNC_stat_fig.write_image(temp_svg[1], scale=3, engine=RENDER_ENGINE)
        # simple template for the PDF page
        head_title = f"Lado {'derecho' if side == 'R' else 'izquierdo'}"
        template = f"""
        <body>
            <img src='{header.resolve().as_uri()}'>
            <h2 style='color:{'red' if side == 'L' else 'green'}'>Cinemática y cinética sagital con electomiografía integrada. {head_title}</h2>
            <img src='{Path(temp_svg[1]).as_uri()}' class='main_plot'>
            <div class='container'>
            {refs}
            </div>
            {self.footer}
        </body>
        """
        # Weasyprint the template
        page = HTML(string=template).render(stylesheets=self.styles)
        self.pages.append(page)

        # close ande remove tempfile
        os.close(temp_svg[0])
        os.unlink(temp_svg[1])

    def KT_static_report(self, autorange=False, mark_out_of_range=False):
        """Prints the kinetic report for this trial into a PDF"""

        print(f"KT report")
        if not self.trial_mdx.trial.has_kinematics:
            print(f"KT report: NO angles in {self.trial_path}. Return")
            return
        if not self.trial_mdx.trial.has_kinetics:
            print(f"KT report: NO forces in {self.trial_path}. Return")
            return
        self.KT_stat_fig1 = KT_Ang_Torque_Power_Figure()

        angles = self.trial_mdx.trial.cycle.angles
        forces = self.trial_mdx.trial.cycle.forces1d
        powers = self.trial_mdx.trial.cycle.powers
        torques = self.trial_mdx.trial.cycle.torques1d
        norm = self.normative_mdx.norm.cycle

        joints_to_plot = KT_Ang_Torque_Power_Figure.joints_to_plot.keys()

        # Adding normatives
        for joint in joints_to_plot:
            ang = norm.angle_bands[f"a{joint}FE"]
            self.KT_stat_fig1.add_norm_trace(
                joint=joint, kind="angle", mean=ang["mean"], std=ang["sd"]
            )
            torq = norm.torque1d_bands[f"tcm{joint}FE"]
            self.KT_stat_fig1.add_norm_trace(
                joint=joint, kind="torque", mean=torq["mean"], std=torq["sd"]
            )
            power = norm.power_bands[f"pcm{joint}FE"]
            self.KT_stat_fig1.add_norm_trace(
                joint=joint, kind="power", mean=power["mean"], std=power["sd"]
            )

        # Adding traces. This will add ALL traces for the cycle
        for joint in joints_to_plot:
            for side in ["r", "l"]:
                for i in angles[f"ac{side.upper()}{joint}FE"].data[0]:
                    self.KT_stat_fig1.add_trace(joint, "angle", side, i)
                if f"tc{side.upper()}{joint}FE" in torques:
                    for i in torques[f"tc{side.upper()}{joint}FE"].data[0]:
                        self.KT_stat_fig1.add_trace(joint, "torque", side, i)
                if f"pc{side.upper()}{joint}FE" in powers:
                    for i in powers[f"pc{side.upper()}{joint}FE"].data[0]:
                        self.KT_stat_fig1.add_trace(joint, "power", side, i)
        ### NUEVO
        # Agregar curvas SCORE si existe el MDX
        if self.has_score:
            x_values = np.linspace(0, 1, 100)
            for joint in joints_to_plot:
                col = KT_Ang_Torque_Power_Figure.joints_to_plot[joint]["col"]
                for side in ["r", "l"]:
                    angle_data = self._score_value(
                        "angles", f"ac{side.upper()}{joint}FE"
                    )
                    if angle_data is not None:
                        for i in angle_data[0]:
                            self._add_score_scatter(
                                self.KT_stat_fig1.fig,
                                y=i,
                                x=x_values,
                                side=side,
                                row=1,
                                col=col + 1,
                                legendgroup=f"score_{side}",
                                showlegend=True if col == 0 else False,
                            )
                    torque_data = self._score_value(
                        "torques1d", f"tc{side.upper()}{joint}FE"
                    )
                    if torque_data is not None:
                        for i in torque_data[0]:
                            self._add_score_scatter(
                                self.KT_stat_fig1.fig,
                                y=i,
                                x=x_values,
                                side=side,
                                row=2,
                                col=col + 1,
                                legendgroup=f"score_{side}",
                                showlegend=False,
                            )
                    power_data = self._score_value(
                        "powers", f"pc{side.upper()}{joint}FE"
                    )
                    if power_data is not None:
                        for i in power_data[0]:
                            self._add_score_scatter(
                                self.KT_stat_fig1.fig,
                                y=i,
                                x=x_values,
                                side=side,
                                row=3,
                                col=col + 1,
                                legendgroup=f"score_{side}",
                                showlegend=False,
                            )
        ### fin nuevo
        self.KT_stat_fig1.adjust_range(auto=False)
        self.KT_stat_fig1.add_annotations()

        RTO = self.trial_mdx.trial.cycle.events["enRTO"].data[0][0][0]
        LTO = self.trial_mdx.trial.cycle.events["enLTO"].data[0][0][0]

        self.KT_stat_fig1.add_event(RTO, dict(color="green", width=2, dash="longdash"))
        self.KT_stat_fig1.add_event(LTO, dict(color="red", width=2, dash="longdash"))

        self.KT_stat_fig1.fig.update_layout(
            margin=go.layout.Margin(l=0, r=0, b=0, t=30),
            height=700,
            width=1000,
            showlegend=False,
            # legend={
            #     "xanchor": "left",
            #     "yanchor": "bottom",
            #     "x": 0.05,
            #     "y": -0.5,
            #     "traceorder": "reversed",
            #     "font": {"size": self.plotly_size},
            # },
        )
        self.KT_stat_fig1.fig.update_layout(
            template="plotly_white", font={"size": self.plotly_size_sm}
        )
        # 2nd figure: frontal plane kinetics
        self.KT_stat_fig2 = KT_Frontal_Plane_Figure()

        # Add nomatives
        for joint in joints_to_plot:
            torq = norm.torque1d_bands[f"tcm{joint}AA"]
            self.KT_stat_fig2.add_norm_trace(
                joint=joint, mean=torq["mean"], std=torq["sd"]
            )

        # Adding traces. This will add ALL traces for the cycle
        for joint in joints_to_plot:
            for side in ["r", "l"]:
                if f"tc{side.upper()}{joint if joint != 'A' else 'F'}AA" in torques:
                    for i in torques[
                        f"tc{side.upper()}{joint if joint != 'A' else 'F'}AA"
                    ].data[0]:
                        self.KT_stat_fig2.add_trace(joint, side, i)

        # Adjusting ranges and annotations
        self.KT_stat_fig2.adjust_range(auto=False)
        self.KT_stat_fig2.add_annotations()

        # Adding events
        self.KT_stat_fig2.add_event(RTO, dict(color="green", width=2, dash="longdash"))
        self.KT_stat_fig2.add_event(LTO, dict(color="red", width=2, dash="longdash"))

        # Adjust figure size, margins, and legend location
        self.KT_stat_fig2.fig.update_layout(
            margin=go.layout.Margin(l=0, r=0, b=0, t=30),
            height=700 / 3.0,
            width=1000,
            showlegend=False,
            legend={
                "xanchor": "left",
                "yanchor": "bottom",
                "x": 0.05,
                "y": 0.5,
                "traceorder": "reversed",
                "font": {"size": self.plotly_size},
            },
        )
        self.KT_stat_fig2.fig.update_layout(
            template="plotly_white", font={"size": self.plotly_size_sm}
        )

        # # 3nd figure: ground reaction force
        forces_to_plot = KT_Ground_Reaction_Figure.forces_to_plot.keys()
        self.KT_stat_fig3 = KT_Ground_Reaction_Figure()

        # Add nomatives
        for force in forces_to_plot:
            forc = norm.force1d_bands[f"fcm{force}"]
            self.KT_stat_fig3.add_norm_trace(
                force=force, mean=forc["mean"], std=forc["sd"]
            )

        # Adding traces. This will add ALL traces for the cycle
        for force in forces_to_plot:
            for side in ["r", "l"]:
                if f"fc{side.upper()}{force}" in forces:
                    for i in forces[f"fc{side.upper()}{force}"].data[0]:
                        self.KT_stat_fig3.add_trace(force, side, i)

        # Adjusting ranges and annotations
        self.KT_stat_fig3.adjust_range(auto=False)
        self.KT_stat_fig3.add_annotations()

        # Adding events
        self.KT_stat_fig3.add_event(RTO, dict(color="green", width=2, dash="longdash"))
        self.KT_stat_fig3.add_event(LTO, dict(color="red", width=2, dash="longdash"))

        self.KT_stat_fig3.fig.update_layout(
            template="plotly_white", font={"size": self.plotly_size_sm}
        )

        # Adjust figure size, margins, and legend location
        self.KT_stat_fig3.fig.update_layout(
            margin=go.layout.Margin(l=0, r=0, b=0, t=30),
            height=700 / 3.0,
            width=1000,
            showlegend=False,
            legend={
                "xanchor": "left",
                "yanchor": "bottom",
                "x": 0.05,
                "y": 0.5,
                "traceorder": "reversed",
                "font": {"size": self.plotly_size},
            },
        )
        root_path = Path(__file__).parent
        images = root_path / "images"
        css = root_path / "templates" / "EMG.css"
        header = images / "HeaderDIN170X23.png"

        # Weasyprint cannot handle embedded SVGs in the HTML yet, so we write
        # the SVG into a temporary file. Cannot use NamedTemporaryFile context
        # manager because the file couldn't be accessed by both Orca and Weasyprint
        # at the same time. Doing it manually...

        temp_svg = tempfile.mkstemp(suffix=".svg", text=True)
        temp_svg2 = tempfile.mkstemp(suffix=".svg", text=True)
        temp_svg3 = tempfile.mkstemp(suffix=".svg", text=True)
        # write SVG to tempfile
        self.KT_stat_fig1.fig.write_image(temp_svg[1], scale=3, engine=RENDER_ENGINE)
        self.KT_stat_fig2.fig.write_image(temp_svg2[1], scale=3, engine=RENDER_ENGINE)
        self.KT_stat_fig3.fig.write_image(temp_svg3[1], scale=3, engine=RENDER_ENGINE)
        # simple template for the PDF page
        template = f"""
        <body>
            <img src='{header.resolve().as_uri()}'>
            <h2>Cinética. Plano sagital</h2>
            <img src='{Path(temp_svg[1]).as_uri()}' class='main_plot'>
            <h2>Cinética. Plano frontal</h2>
            <img src='{Path(temp_svg2[1]).as_uri()}' class='main_plot'>
            <h2>Fuerza de reacción del piso</h2>
            <img src='{Path(temp_svg3[1]).as_uri()}' class='main_plot'>
            {self.footer}
        </body>
        """

        # Weasyprint the template
        page = HTML(string=template).render(stylesheets=self.styles)
        self.pages.append(page)

        # close ande remove tempfile
        os.close(temp_svg[0])
        os.unlink(temp_svg[1])

        os.close(temp_svg2[0])
        os.unlink(temp_svg2[1])

        os.close(temp_svg3[0])
        os.unlink(temp_svg3[1])

    def MUS_static_report(self, autorange=False, mark_out_of_range=False):
        """Prints the muscle length/speed report for this trial into a PDF"""

        if not self.trial_mdx.trial.has_muscles:
            print(f"MUS report: NO muscles in {self.trial_path}. Return")
            return

        scalars = self.trial_mdx.trial.cycle.scalars
        # scalars_score = self.trial_score_mdx.trial.cycle.scalars
        normatives = self.trial_mdx.trial.cycle.scalar_bands
        muscles_to_plot = {
            "Semitendinoso": {"code": "semiten"},
            "Recto femoral": {"code": "rect_fem"},
            "Aductor largo": {"code": "add_long"},
            "Gastrocnemio medial": {"code": "med_gas"},
        }

        titles = []
        for title in muscles_to_plot.keys():
            titles.append(title + " (longitud)")
            titles.append(title + " (velocidad)")

        for k, v in muscles_to_plot.items():
            v["length"] = {
                "right": scalars[f"len_{v['code']}_r"].data[0],
                "left": scalars[f"len_{v['code']}_l"].data[0],
                # "right_score": scalars_score[f"len_{v['code']}_r"].data[0],
                # "left_score": scalars[f"len_{v['code']}_l"].data[0],
                "norm": normatives[f"{v['code']}lenCycle"].data,
            }
            v["speed"] = {
                "right": scalars[f"vel_{v['code']}_r"].data[0],
                "left": scalars[f"vel_{v['code']}_l"].data[0],
                "norm": normatives[f"{v['code']}velCycle"].data,
            }

            # self.KM_static_fig = make_subplots(rows=4,cols=3,shared_xaxes=True,shared_yaxes=False,subplot_titles=)
        # for k, v in angles.items():
        #     print(k, v._description)

        
        ### NUEVO
        # Construir muscles_to_plot para SCORE
        score_muscles_to_plot = None
        if self.has_score:
            score_muscles_to_plot = {
                k: {"code": v["code"]} for k, v in muscles_to_plot.items()
            }
            for k, v in score_muscles_to_plot.items():
                code = v["code"]
                length_r = self._score_value("scalars", f"len_{code}_r")
                length_l = self._score_value("scalars", f"len_{code}_l")
                vel_r = self._score_value("scalars", f"vel_{code}_r")
                vel_l = self._score_value("scalars", f"vel_{code}_l")
                if None in (length_r, length_l, vel_r, vel_l):
                    score_muscles_to_plot[k] = None
                else:
                    v["length"] = {"right": length_r[0], "left": length_l[0]}
                    v["speed"] = {"right": vel_r[0], "left": vel_l[0]}
        ### fin nuevo

        x_values = np.linspace(0, 1, 100)

        self.MUS_stat_fig = make_subplots(
            rows=len(muscles_to_plot),
            cols=2,
            subplot_titles=titles,
            shared_xaxes=True,
            shared_yaxes=False,
            vertical_spacing=0.1,
            horizontal_spacing=0.10,
        )
        self.MUS_stat_fig.update_layout(
            template="plotly_white", font={"size": self.plotly_size_m}
        )
        for col, item in enumerate(["length", "speed"]):
            rang = [0.75, 1.15] if item == "length" else [-1.3, 1.3]

            for n, mus in enumerate(muscles_to_plot.items()):
                k, v = mus

                # Add the traces. Muscle normatives included in every .mdx... might want to change
                # that in the future

                # Plotting normative
                mean, sd = v[item]["norm"]
                self.MUS_stat_fig.add_trace(
                    go.Scatter(
                        y=mean + sd,
                        name=k,
                        x=x_values,
                        line=dict(width=0, color="grey"),
                        legendgroup="mean",
                        showlegend=False,
                        # marker={"coloraxis": "coloraxis"},
                    ),
                    row=n + 1,
                    col=col + 1,
                )

                self.MUS_stat_fig.add_trace(
                    go.Scatter(
                        y=mean - sd,
                        name=k,
                        x=x_values,
                        line=dict(width=0, color="grey"),
                        legendgroup="mean",
                        showlegend=False,
                        fill="tonexty",
                        fillcolor="rgba(125,125,125,0.3)"
                        # marker={"coloraxis": "coloraxis"},
                    ),
                    row=n + 1,
                    col=col + 1,
                )

                self.MUS_stat_fig.add_trace(
                    go.Scatter(
                        y=mean,
                        name="Control",
                        x=x_values,
                        line=dict(width=2, color="grey"),
                        legendgroup="mean",
                        showlegend=True if n + col == 0 else False,
                        # marker={"coloraxis": "coloraxis"},
                    ),
                    row=n + 1,
                    col=col + 1,
                )

                # Plot both traces and events
                for side in ["right", "left"]:
                # for side in ["right", "left","right_score", "left_score"]:
                    color = "green" if side == "right" else "red"
                    ev = "enLTO" if side == "right" else "enRTO"
                    data = v[item][side]
                    # if _score
                    # self.MUS_stat_fig.add_trace(
                    #     go.Scatter(
                    #         y=data,
                    #         name="Derecha" if side == "right" else "Izquierda",
                    #         x=x_values,
                    #         line=dict(width=3, color=color),
                    #         legendgroup="mean",
                    #         showlegend=True if n + col == 0 else False,
                    #         # marker={"coloraxis": "coloraxis"},
                    #     ),
                    #     row=n + 1,
                    #     col=col + 1,
                    # )
                    # else
                    self.MUS_stat_fig.add_trace(
                        go.Scatter(
                            y=data,
                            name="Derecha" if side == "right" else "Izquierda",
                            x=x_values,
                            line=dict(width=3, color=color),
                            legendgroup="mean",
                            showlegend=True if n + col == 0 else False,
                            # marker={"coloraxis": "coloraxis"},
                        ),
                        row=n + 1,
                        col=col + 1,
                    )

                    for i in add_event_line(
                        self.trial_mdx.trial,
                        ev,
                        line=dict(color=color, width=2, dash="longdash"),
                        normalized=True,
                        y1=2,
                        y0=-2,
                    ):
                        self.MUS_stat_fig.add_shape(i, row=n + 1, col=col + 1)
                ### NUEVO
                # Curvas SCORE
                if (
                    score_muscles_to_plot is not None
                    and score_muscles_to_plot.get(k) is not None
                ):
                    for side in ["right", "left"]:
                        try:
                            score_data = score_muscles_to_plot[k][item][side]
                            self._add_score_scatter(
                                self.MUS_stat_fig,
                                y=score_data,
                                x=x_values,
                                side=side,
                                row=n + 1,
                                col=col + 1,
                                legendgroup="score",
                                showlegend=True if n + col == 0 else False,
                                width=2,
                            )
                        except (KeyError, TypeError):
                            pass
                ### fin nuevo

                # Detect ranges
                self.MUS_stat_fig.update_yaxes(
                    range=rang, row=n + 1, col=col + 1, tickformat=",.0%"
                )
                self.MUS_stat_fig.update_xaxes(
                    tickformat=",.0%",
                    dtick=0.2,
                    row=n + 1,
                    col=col + 1,
                    title_text="Ciclo (%)" if n == len(muscles_to_plot) - 1 else "",
                )

                for i in self.MUS_stat_fig["layout"]["annotations"]:
                    i["font"] = {"size": self.plotly_size_sm}

        self.MUS_stat_fig.update_layout(
            margin=go.layout.Margin(l=0, r=0, b=0, t=30),
            height=1060,
            width=1000,
            legend={
                "x": 1.01,
                "y": 0.5,
                # "x": 1.0 / 3,
                "orientation": "h",
                "xanchor": "center",
                "yanchor": "middle",
                "font": {"size": self.plotly_size},
            },
        )

        # self.KN_stat_fig.show()
        root_path = Path(__file__).parent
        images = root_path / "images"
        css = root_path / "templates" / "EMG.css"
        header = images / "HeaderCIN170X23.png"

        # Weasyprint cannot handle embedded SVGs in the HTML yet, so we write
        # the SVG into a temporary file. Cannot use NamedTemporaryFile context
        # manager because the file couldn't be accessed by both Orca and Weasyprint
        # at the same time. Doing it manually...

        temp_svg = tempfile.mkstemp(suffix=".svg", text=True)

        # write SVG to tempfile
        self.MUS_stat_fig.write_image(temp_svg[1], scale=3, engine=RENDER_ENGINE)

        # simple template for the PDF page
        template = f"""
        <body>
            <img src='{header.resolve().as_uri()}'>
            <h2>Longitud y velocidad muscular</h2>
            <img src='{Path(temp_svg[1]).as_uri()}' class='main_plot'>
            {self.footer}
        </body>
        """
        # Weasyprint the template
        page = HTML(string=template).render(stylesheets=self.styles)
        self.pages.append(page)

        # close ande remove tempfile
        os.close(temp_svg[0])
        os.unlink(temp_svg[1])

    def print(self, filename):
        if not self.pages:
            print("Empty PDF, generate pages first.")
            return
        a = self.pages[0].copy(
            [page for document in self.pages for page in document.pages]
        )
        a.write_pdf(filename)

    def KM_static_table_running(self):
        if not self.trial_mdx.trial.has_kinematics:
            print(f"KM report: NO angles in {self.trial_path}. Return")
            return
        # processor = xml.dictionary(
        #     "emxDataFile",
        #     [
        #         xml.array(
        #             xml.dictionary(
        #                 "trial",
        #                 [
        #                     xml.string("stream"),
        #                     xml.string("stream"),
        #                     xml.integer("static"),
        #                     xml.integer("stream"),
        #                     xml.integer("cycle"),
        #                     xml.integer("calibration"),
        #                     xml.integer("stick_model"),
        #                 ],
        #             ),
        #             alias="_trial",
        #         )
        #     ],
        # )
        # dat = xml.parse_from_file(processor, self.trial_path)
        stat = self.trial_mdx.trial.static
        norm = self.normative_mdx.norm.static
        session = self.session_mdx.trial.static
        data = {
            "Cadencia": {"code": "Cadence SPM"},
            "Numero total de pasos": {"code": "TotalStepNumber"},
            "Tiempo de ciclo (s)": {"code": "m_CycleTime"},
            "Tiempo de apoyo (s)": {"code": "m_StanceTime"},
            "Tiempo de balanceo (s)": {"code": "m_SwingTime"},
            "Fase de apoyo (%)": {"code": "_%Stance"},
            "Fase de balanceo (%)": {"code": "_%Swing"},
            # "Fase de apoyo simple (%)": {"code": "s_SINGSTANCE"},
            # "Fase de apoyo doble (%)": {"code": "s_DBLSTANCE"},
            # "Velocidad media (m/s)": {"code": "VEL1DPELVIS"},
            # "Velocidad de balanceo media (m/s)": {"code": "velm_HSWING"},
            # "Largo de ciclo (m)": {"code": "d_STRIDE"},
            # "Ancho de paso (m)": {"code": "dmSTEPWIDTH"},
            # "Gait Deviation Index": {"code": "_GDI"},
            # "Gait Profile Score": {"code": "a_GPS"},
            "Oscilación (m)": {"code": "Oscillation"},
        }

        # "Largo de paso (m)": {"code": "mStepLength"},
        for k, v in data.items():
            if "_" in v["code"]:
                for side in ["R", "L"]:
                    item = stat.find(v["code"].replace("_", side))
                    if "M" in item.coords:
                        data[k][side] = {"data": item.data[0], "error": item.data[1]}
                    elif "V" in item.coords:
                        data[k][side] = {"data": item.data[0]}

                    else:
                        raise Exception(f"No data for {side} {v['code']}")
                try:
                    item = stat.find(v["code"].replace("_", ""))
                    if "M" in item.coords:
                        data[k]["B"] = {"data": item.data[0], "error": item.data[1]}
                    elif "V" in item.coords:
                        data[k]["B"] = {"data": item.data[0]}
                except:
                    pass
            else:
                item = stat.find(v["code"])
                if "M" in item.coords:
                    data[k]["B"] = {"data": item.data[0], "error": item.data[1]}
                elif "V" in item.coords:
                    data[k]["B"] = {"data": item.data[0]}
                else:
                    raise Exception(f"No data for {v['code']}")
            try:
                name = (
                    v["code"]
                    .replace("_", "")
                    .replace("tm", "t")
                    .replace("dm", "d")
                    .replace("velm", "vel")
                    .replace("velHSTRIDE", "velMEAN")
                )
                norm_item = norm.find(name)
                data[k]["N"] = {"data": norm_item.data[0], "error": norm_item.data[1]}
            except:
                print(f"No norm for {name}")

        largo_pierna = max(
            session.find("dRLL").data[0][0],
            session.find("dLLL").data[0][0],
        )
        largo_pierna = largo_pierna * 100
        altura = session.find("dTH").data[0][0] * 100
        peso = session.find("mTB").data[0][0]

        tiempo_total = round(
            int(data["Numero total de pasos"]["B"]["data"][0])
            / float(data["Cadencia"]["B"]["data"][0]),
            1,
        )

        svg_temp = {}
        html_temp = {}
        temp_dir = tempfile.mkdtemp()
        
        # Add step time graphs:
        vals = {"L": {}, "R": {}}
        vals["L"] = {
            k: v["L"]["data"][0]
            for k, v in data.items()
            if "B" not in v.keys() and v["code"].startswith("_%")
        }
        vals["R"] = {
            k: v["R"]["data"][0]
            for k, v in data.items()
            if "B" not in v.keys() and v["code"].startswith("_%")
        }

        categories = {"L": {}, "R": {}}
        categories["L"] = {
            k: round((v["L"]["data"][0] * 1000))
            for k, v in data.items()
            if "B" not in v.keys()
            and v["code"] != "_%Stance"
            and v["code"] != "_%Swing"
        }

        categories["R"] = {
            k: round((v["R"]["data"][0] * 1000))
            for k, v in data.items()
            if "B" not in v.keys()
            and v["code"] != "_%Stance"
            and v["code"] != "_%Swing"
        }
        # --- Graficos i / d
        graficos = {}
        for side in ["L", "R"]:
            if side == "R":
                marker_color = [right_color(0.5), right_color(0.7)]
            else:
                marker_color = [left_color(0.5), left_color(0.7)]
            # Add pie
            name = "pie_" + side
            value = Pie(
                init_opts=opts.InitOpts(width="400px", height="400px")
            )
            value.options = {
                "height": 400,
                "width": 400,
                "margin": 0,
                "padding": 0,
                "series": [
                    {
                        "name": "Access From",
                        "type": "pie",
                        "radius": ["42%", "65%"],
                        "itemStyle": {
                            "borderRadius": 10,
                            "borderWidth": 4,
                            "borderColor": "#fff",
                        },
                        "label": {
                            "show": True,
                            "position": "inner",
                            "fontWeight": "bolder",
                            "fontSize": 35,
                            "color": "black",
                            "formatter": " {d}% ",
                        },
                        "labelLine": {"show": True},
                        "data": [
                            {
                                "value": data["Fase de apoyo (%)"][side]["data"][0],
                                "name": "Apoyo",
                            },
                            {
                                "value": data["Fase de balanceo (%)"][side]["data"][0],
                                "name": "Balanceo",
                            },
                        ],
                    }
                ],
            }
            graficos[name] = value
            max_bar = max(max(categories["L"].values()), max(categories["R"].values()))
            # add bars for the categories
            for k, v in categories[side].items():
                t = (
                    "apoyo"
                    if "apoyo" in k
                    else "balanceo"
                    if "balanceo" in k
                    else "ciclo"
                )
                name = "bar_" + t + "_" + side
                print(name, categories[side][k], " - ", max_bar)
                value = Bar(
                    init_opts=opts.InitOpts(
                        width="900px", height="40px"
                    )
                )
                value.options = {
                    "xAxis": {
                        "type": "value",
                        "max": max_bar,
                        "interval": 30,
                        "offset": 20,
                        "axisLabel": {"show": False},
                        "axisLine": {"lineStyle": {"width": 10}},
                        "axisTick": {"length": 0},
                        "splitLine": {"show": False},
                    },
                    "yAxis": {
                        "type": "value",
                        "interval": 1,
                        "xscale": True,
                        "show": False,
                    },
                    "grid": {
                        "height": 21,
                        "left": 0,
                        "top": 0,
                        "right": 0,
                        "bottom": 0,
                    },
                    "series": [
                        {
                            "type": "line",
                            "data": [[0, 1], [categories[side][k], 1]],
                            "lineStyle": {"color": "#5470c6", "width": 40},
                            "symbol": "none",
                        },
                    ],
                }
                graficos[name] = value
        for name in graficos:
            n = name
            svg_temp[n] = tempfile.mkstemp(suffix=".svg", text=True)
            make_snapshot(
                snapshot, graficos[name].render(), svg_temp[n][1], is_remove_html=True
            )
            print(n,svg_temp[n][1])
            # svg_temp[n] = os.path.join(temp_dir, n+".png")
            # html_temp[n] = os.path.join(temp_dir, n+".html")
            # graficos[name].render(html_temp[n]) 
        
        # --- cadencia
        d = 0
        for k, v in data.items():
            if v["code"] == "Cadence SPM":
                d = round(v["B"]["data"][0], 1)
        cadencia = {
            "nombre": "Cadence SPM",
            "data": d,
        }

        gaug = Gauge(
            init_opts=opts.InitOpts(width="400px", height="400px")
        )

        gaug.options = {
            "series": [
                {
                    "type": "gauge",
                    "center": ["50%", "60%"],
                    "startAngle": 200,
                    "endAngle": -20,
                    "min": 0,
                    "max": 250,
                    "splitNumber": 5,
                    "itemStyle": {"color": "#5470c6"},
                    "progress": {"show": True, "width": 30},
                    "pointer": {"show": False},
                    "axisLine": {"lineStyle": {"width": 30}},
                    "axisTick": {
                        "distance": -45,
                        "splitNumber": 5,
                        "lineStyle": {"width": 3, "color": "#999"},
                    },
                    "splitLine": {
                        "distance": -52,
                        "length": 17,
                        "lineStyle": {"width": 4, "color": "#999"},
                    },
                    "axisLabel": {"distance": -13, "color": "#999", "fontSize": 23},
                    "anchor": {"show": False},
                    "title": {"show": False},
                    "detail": {
                        "valueAnimation": True,
                        "width": "60%",
                        "lineHeight": 40,
                        "borderRadius": 8,
                        "offsetCenter": [0, "0%"],
                        "fontSize": 45,
                        "fontWeight": "bolder",
                        "formatter": "{value}\nSPM",
                        "color": "inherit",
                    },
                    "data": [{"value": cadencia["data"]}],
                },
                {
                    "type": "gauge",
                    "center": ["50%", "60%"],
                    "startAngle": 200,
                    "endAngle": -20,
                    "min": 0,
                    "max": 250,
                    "itemStyle": {"color": "#30468a"},
                    "progress": {"show": True, "width": 8},
                    "pointer": {"show": False},
                    "axisLine": {"show": False},
                    "axisTick": {"show": False},
                    "splitLine": {"show": False},
                    "axisLabel": {"show": False},
                    "detail": {"show": False},
                    "data": [{"value": cadencia["data"]}],
                },
            ]
        }
        n = "gauge"
        svg_temp[n] = tempfile.mkstemp(suffix=".svg", text=True)
        make_snapshot(snapshot, gaug.render(), svg_temp[n][1], is_remove_html=True)
        # svg_temp[n] = os.path.join(temp_dir, n+".png")
        # html_temp[n] = os.path.join(temp_dir, n+".html")
        # gaug.render(html_temp[n]) 
        
        # --- Oscilacion
        d = 0
        e = 0
        for k, v in data.items():
            if v["code"] == "Oscillation":
                d = v["B"]["data"][0] * 1000
                e = v["B"]["error"][0] * 1000
        oscilacion = {"nombre": "Oscillation", "data": d, "error": e}
        #              {'code': 'Oscillation', 'B': {'data': array([0.1099]), 'error': array([0.0044])}}])
        osc_value = round(oscilacion["data"], 2)
        oscil = Line(
            init_opts=opts.InitOpts(width="125px", height="400px")
        )
        oscil.options = {
            "yAxis": {
                "type": "value",
                "max": 150,
                "interval": 30,
                "offset": 20,
                "axisLabel": {"fontSize": 23},
                "axisLine": {"lineStyle": {"width": 3}},
                "axisTick": {"length": 25, "lineStyle": {"width": 3}},
                "minorTick": {"show": True, "length": 15, "lineStyle": {"width": 1}},
            },
            "xAxis": {"type": "value", "interval": 1, "yscale": True, "show": False},
            "grid": {"width": 22, "right": 80, "left": 10, "top": 20, "bottom": 10},
            "series": [
                {
                    "type": "line",
                    "data": [[1, 0], [1, 150]],
                    "lineStyle": {"color": "#e6ebf8", "width": 40},
                    "symbol": "none",
                },
                {
                    "type": "line",
                    "data": [[1, 66], [1, 86]],
                    "lineStyle": {"color": "#5470c6", "width": 40},
                    "symbol": "none",
                },
                {
                    "type": "line",
                    "data": [[1, osc_value], [1, osc_value]],
                    "color": "#696969",
                    "symbolSize": 40,
                    "symbol": "triangle",
                    "symbolRotate": 90,
                    "symbolOffset": [60, 0],
                },
            ],
        }
        n = "oscil"
        svg_temp[n] = tempfile.mkstemp(suffix=".svg", text=True)
        make_snapshot(snapshot, oscil.render(), svg_temp[n][1], is_remove_html=True)
        # svg_temp[n] = os.path.join(temp_dir, n+".png")
        # html_temp[n] = os.path.join(temp_dir, n+".html")
        # oscil.render(html_temp[n]) 

        # for n in html_temp:
        #     imgkit.from_file(html_temp[n], svg_temp[n])
        
        root_path = Path(__file__).parent
        images = root_path / "images"
        css = root_path / "Running_reporter" / "assets" / "style-cards.css"
        header = images / "HeaderCIN170X23.png"
        icon_root = (
            root_path
            / "Running_reporter"
            / "assets"
            / "fontawesomefree"
            / "svgs"
            / "solid"
        )
        
        # Weasyprint cannot handle embedded SVGs in the HTML yet, so we write
        # the SVG into a temporary file. Cannot use NamedTemporaryFile context
        # manager because the file couldn't be accessed by both Orca and Weasyprint
        # at the same time. Doing it manually...

        # simple template for the PDF page

        template = f"""
        <body>
        <img src='{header.resolve().as_uri()}'>
        <h2>Parámetros lineales</h2>
        <div class="main_plot">
            <div class="cards">
            <div class="cards__item_1">
                <div class="card">
                <div class="card__content">
                    <div class="icons">
                        <img class="icons__img" src="{Path(icon_root/"user-color.svg").as_uri()}" />
                    </div>
                    <div class="card__half">
                        <div class="info">
                            <div class="info__titles">
                                <div class="info__text">Altura</div>
                                <div class="info__value">{altura} cm</div>
                            </div>
                        </div>
                        <div class="info">
                            <div class="info__titles">
                                <div class="info__text">Peso</div>
                                <div class="info__value">{peso} kg</div>
                            </div>
                        </div>
                        <div class="info">
                            <div class="info__titles">
                                <div class="info__text">Largo de pierna</div>
                                <div class="info__value">{largo_pierna} cm</div>
                            </div>
                        </div>
                        </div>
                        <div>       </div>
                        <div class="card__half">
                        <div class="info">
                            <div class="info__titles">
                                <div class="info__text">Largo de paso</div>
                                <div class="info__value">- cm</div>
                            </div>
                        </div>
                        <div class="info">
                            <div class="info__titles">
                                <div class="info__text">Ancho de paso</div>
                                <div class="info__value">- cm</div>
                            </div>
                        </div>
                        <div class="info">
                            <div class="info__titles">
                                <div class="info__text">Zancada</div>
                                <div class="info__value">- cm</div>
                        </div>
                    </div>
                    </div>
                </div>
                </div>
            </div>
            <div class="cards__item_2">
                <div class="card">
                <div class="card__title">Derecha</div>
                <div class="card__content">
                    <div class="card__half">
                    <div class="img__svg">
                        <img class="imgs"  src="{Path(svg_temp["pie_R"][1]).as_uri()}" />
                    </div>
                    </div>
                    <div class="card__half">
                    <div class="bar">
                        <div class="bar__titles">
                        <div class="bar__text">Apoyo</div>
                        <div class="bar__value">{categories["R"]["Tiempo de apoyo (s)"]} ms</div>
                        </div>
                        <div class="img__svg">
                        <img src="{Path(svg_temp["bar_apoyo_R"][1]).as_uri()}" />
                        </div>
                    </div>
                    <div class="bar">
                        <div class="bar__titles">
                        <div class="bar__text">Vuelo</div>
                        <div class="bar__value">{categories["R"]["Tiempo de balanceo (s)"]} ms</div>
                        </div>
                        <div class="img__svg">
                        <img src="{Path(svg_temp["bar_balanceo_R"][1]).as_uri()}" />
                        </div>
                    </div>
                    <div class="bar">
                        <div class="bar__titles">
                        <div class="bar__text">Tiempo de ciclo</div>
                        <div class="bar__value">{categories["R"]["Tiempo de ciclo (s)"]} ms</div>
                        </div>
                        <div class="img__svg">
                        <img src="{Path(svg_temp["bar_ciclo_R"][1]).as_uri()}" />
                        </div>
                    </div>
                    </div>
                </div>
                </div>
            </div>
            <div class="cards__item_2">
                <div class="card">
                <div class="card__title">Izquierda</div>
                <div class="card__content">
                    <div class="card__half">
                    <div class="img__svg">
                        <img class="imgs" src="{Path(svg_temp["pie_L"][1]).as_uri()}" />
                    </div>
                    </div>
                    <div class="card__half">
                    <div class="bar">
                        <div class="bar__titles">
                        <div class="bar__text">Apoyo</div>
                        <div class="bar__value">{categories["L"]["Tiempo de apoyo (s)"]} ms</div>
                        </div>
                        <div class="img__svg">
                        <img src="{Path(svg_temp["bar_apoyo_L"][1]).as_uri()}" />
                        </div>
                    </div>
                    <div class="bar">
                        <div class="bar__titles">
                        <div class="bar__text">Vuelo</div>
                        <div class="bar__value">{categories["L"]["Tiempo de balanceo (s)"]} ms</div>
                        </div>
                        <div class="img__svg">
                        <img src="{Path(svg_temp["bar_balanceo_L"][1]).as_uri()}" />
                        </div>
                    </div>
                    <div class="bar">
                        <div class="bar__titles">
                        <div class="bar__text">Tiempo de ciclo</div>
                        <div class="bar__value">{categories["L"]["Tiempo de ciclo (s)"]} ms</div>
                        </div>
                        <div class="img__svg">
                        <img src="{Path(svg_temp["bar_ciclo_L"][1]).as_uri()}" />
                        </div>
                    </div>
                    </div>
                </div>
                </div>
            </div>
            <div class="cards__item">
                <div class="card">
                <div class="card__content">
                    <div class="grid__values">
                    <div class="half">
                    
                        <div class="cell uno">
                        <div class="cell-logo">
                            <img class="logo-img" src="{Path(icon_root/"clock-color.svg").as_uri()}" />
                        </div>
                        <div class="cell-info">
                            <div class="cell-name">Tiempo total</div>
                            <div class="cell-value">{tiempo_total} min</div>
                            <div class="cell-subname">Detalle</div>
                        </div>
                        </div>
                        
                        <div class="cell tres">
                        <div class="cell-logo">
                            <img class="logo-img" src="{Path(icon_root/"bolt-color.svg").as_uri()}" />
                        </div>
                        <div class="cell-info">
                            <div class="cell-name">Energía consumida</div>
                            <div class="cell-value">3.3 J/kg x m</div>
                            <div class="cell-subname">Detalle</div>
                        </div>
                        </div>
                        
                    </div>
                    <div class="half">
                    
                        <div class="cell dos">
                        <div class="cell-logo">
                            <img class="logo-img" src="{Path(icon_root/"ruler-color.svg").as_uri()}" />
                        </div>
                        <div class="cell-info">
                            <div class="cell-name">Distancia total</div>
                            <div class="cell-value">- km</div>
                            <div class="cell-subname">Detalle</div>
                        </div>
                        </div>
                        
                        <div class="cell cuatro">
                        <div class="cell-logo">
                            <img class="logo-img" src="{Path(icon_root/"person-running-color.svg").as_uri()}" />
                        </div>
                        <div class="cell-info">
                            <div class="cell-name">Velocidad</div>
                            <div class="cell-value">- km/h</div>
                            <div class="cell-subname">Detalle</div>
                        </div>
                        </div>
                        
                    </div>
                    </div>
                </div>
                </div>
            </div>
            <div class="cards__item">
                <div class="card">
                <div class="card__title">Cadencia</div>
                <div class="card__content">
                    <div class="card__half">
                    <div class="img__svg">
                        <img class="gauge" src="{Path(svg_temp["gauge"][1]).as_uri()}" />
                    </div>
                    </div>
                </div>
                </div>
            </div>
            <div class="cards__item">
                <div class="card">
                <div class="card__title">Oscilación</div>
                <div class="card__content">
                    <div class="card__half">
                    <div class="graph">
                        <img class="graph__img" src="{Path(svg_temp["oscil"][1]).as_uri()}" />
                    </div>
                    </div>
                    <div class="text">
                    <div class="graph__text">{osc_value} mm</div>
                    </div>
                </div>
                </div>
            </div>
            </div>
        </div>
            {self.footer}
        </body>

        """

        # Get CSS

        css = root_path / "Running_reporter" / "assets" / "style-cards.css"
        css_string = open(css, encoding="UTF-8").read()
        style_cards = CSS(string=css_string)

        # self.styles.append(CSS(self.css.as_posix()))

        # Weasyprint the template

        page = HTML(string=template, encoding="utf-8").render(stylesheets=[style_cards])
        # stylesheets=[base_css],
        # )

        self.pages.append(page)
        # # close ande remove tempfile
        # for k in svg_temp:
        #     os.close(svg_temp[k][0])
        #     os.unlink(svg_temp[k][1])

        #     # Eliminar los archivos HTML temporales y el directorio temporal
        #     os.remove(svg_temp[k][1])
        # # ... Elimina los otros archivos temporales
        # os.rmdir(temp_dir)
