from itertools import zip_longest
from operator import index, sub
from typing import Iterable
from matplotlib.pyplot import legend, title
from matplotlib import cm, colors
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from functions import add_KM_norms, add_KM_traces, get_emg_envelope, mpl_to_plotly


class Equinus_Figure:
    def __init__(self):
        self.fig = make_subplots(
            6,
            2,
            shared_xaxes=True,
            shared_yaxes=False,
            subplot_titles=[
                "Right rectus femoris",
                "Left rectus femoris",
                "Right tibialis anterior",
                "Left tibialis anterior",
                "Right gastrocnemius medialis",
                "Left gastrocnemius medialis",
                "Right rectus femoris (with resistance)",
                "Left rectus femoris (with resistance)",
                "Right tibialis anterior (with resistance)",
                "Left tibialis anterior (with resistance)",
                "Right gastrocnemius medialis (with resistance)",
                "Left gastrocnemius medialis (with resistance)",
            ],
        )
        self.fig.update_xaxes(tickformat=",.0%", range=[0, 1])
        # self.fig.update_layout(width=1600, height=1600, autosize=True)
        # self.fig.update_xaxes(automargin=True)
        # self.fig.update_yaxes(automargin=True)


class KT_Ang_Torque_Power_Figure:
    x_values = np.linspace(0, 1, 100)
    joints_to_plot = {
        "HP": {
            "title": "cadera",
            "col": 0,
            "angle_data": {"min": -20, "max": 60, "texts": ["Flex", "Ext"]},
            "torque_data": {"min": -1, "max": 2, "texts": ["Ext", "Flex"]},
            "power_data": {"min": -2, "max": 3, "texts": ["Gen", "Abs"]},
        },
        "K": {
            "title": "rodilla",
            "col": 1,
            "angle_data": {"min": -20, "max": 80, "texts": ["Flex", "Ext"]},
            "torque_data": {"min": -1, "max": 2, "texts": ["Ext", "Flex"]},
            "power_data": {"min": -2, "max": 3, "texts": ["Gen", "Abs"]},
        },
        "A": {
            "title": "tobillo",
            "col": 2,
            "angle_data": {"min": -30, "max": 30, "texts": ["Dors", "Plant"]},
            "torque_data": {"min": -1, "max": 2, "texts": ["Plant", "Dors"]},
            "power_data": {"min": -2, "max": 3, "texts": ["Gen", "Abs"]},
        },
    }

    titles = (
        [f"Flexoextensión de {i['title']}" for i in joints_to_plot.values()]
        + [f"Momento de {i['title']}" for i in joints_to_plot.values()]
        + [f"Potencia de {i['title']}" for i in joints_to_plot.values()]
    )

    def __init__(self, title_font={"size": 21}, sub_args={}):
        self.title_font = title_font
        self.fig = make_subplots(
            rows=3,
            cols=3,
            subplot_titles=self.titles,
            shared_xaxes=False,
            shared_yaxes=False,
            vertical_spacing=0.15,
            horizontal_spacing=0.10,
            **sub_args,
        )

    def adjust_range(self, auto=False):
        if not auto:
            for k, v in self.joints_to_plot.items():
                for n, kind in enumerate(["angle", "torque", "power"]):
                    self.fig.update_yaxes(
                        range=[v[f"{kind}_data"]["min"], v[f"{kind}_data"]["max"]],
                        row=n + 1,
                        col=v["col"] + 1,
                    )
        else:
            for k, v in self.joints_to_plot.items():
                for row in range(3):
                    index = row * 3 + v["col"] + 1
                    if index == 1:
                        index = ""
                    mx = max(
                        (
                            np.nanmax(i["y"])
                            for i in self.fig.data
                            if i["yaxis"] == f"y{index}"
                        ),
                        default=1,
                    )
                    mn = min(
                        (
                            np.nanmin(i["y"])
                            for i in self.fig.data
                            if i["yaxis"] == f"y{index}"
                        ),
                        default=0,
                    )
                    self.fig.update_yaxes(range=[mn, mx], row=row + 1, col=v["col"] + 1)

    def add_annotations(self, font={"size": 14}):
        for i in self.fig["layout"]["annotations"]:
            i["font"] = self.title_font

        for k, v in self.joints_to_plot.items():
            for row, kind in enumerate(["angle", "torque", "power"]):
                self.fig.update_xaxes(
                    tickformat=",.0%",
                    dtick=0.2,
                    row=row + 1,
                    col=v["col"] + 1,
                    title_text="Ciclo (%)",
                )
                if kind == "angle":
                    y_unit = "(°)"
                elif kind == "torque":
                    y_unit = "N m/Kg"
                elif kind == "power":
                    y_unit = "W/Kg"
                self.fig.update_yaxes(title_text=y_unit, row=row + 1, col=v["col"] + 1)

        for k, v in self.joints_to_plot.items():
            for row, val in enumerate(["angle", "torque", "power"]):
                index = row * 3 + v["col"] + 1
                if index == 1:
                    index = ""
                ran = self.fig.layout[f"yaxis{index}"]["range"]
                dist = (ran[1] - ran[0]) * 0.15
                texts = v[f"{val}_data"]["texts"]
                for i, text in enumerate(texts):
                    annot = {
                        "text": text,
                        "xref": f"x{index}",
                        "y": ran[0] + dist if i else ran[1] - dist,
                        "yref": f"y{index}",
                        "x": 1.05,
                        "yanchor": "middle",
                        "xanchor": "auto",
                        "font": font,
                        "showarrow": False,
                        "align": "left",
                    }
                    self.fig.layout.annotations += (annot,)

    def add_trace(self, joint, kind, side, data):
        if kind == "angle":
            row = 0
        elif kind == "torque":
            row = 1
        elif kind == "power":
            row = 2
        else:
            raise ValueError("The kind of trace added can be angle, torque or power")
        col = self.joints_to_plot[joint]["col"]
        self.fig.add_trace(
            go.Scatter(
                y=data,
                x=self.x_values,
                name="Derecha" if side.lower() == "r" else "Izquierda",
                line=dict(width=3, color="green" if side.lower() == "r" else "red"),
                legendgroup="right" if side.lower() == "r" else "left",
                showlegend=True if row + col == 0 else False,
                # marker={"coloraxis": "coloraxis"},
            ),
            row=row + 1,
            col=col + 1,
        )
        
    def add_consistency_trace(self, joint, kind, side, data):
        if kind == "angle":
            row = 0
        elif kind == "torque":
            row = 1
        elif kind == "power":
            row = 2
        else:
            raise ValueError("The kind of trace added can be angle, torque or power")
        col = self.joints_to_plot[joint]["col"]
        self.fig.add_trace(
            go.Scatter(
                y=data,
                x=self.x_values,
                name="Derecha" if side.lower() == "r" else "Izquierda",
                line=dict(width=2, color="green" if side.lower() == "r" else "red"),
                legendgroup="right" if side.lower() == "r" else "left",
                showlegend=True if row + col == 0 else False,
                # marker={"coloraxis": "coloraxis"},
            ),
            row=row + 1,
            col=col + 1,
        )
        
    def add_norm_trace(self, joint, kind, mean, std):
        if kind == "angle":
            row = 0
        elif kind == "torque":
            row = 1
        elif kind == "power":
            row = 2
        else:
            raise ValueError(
                "The kind of normative trace added can be angle, torque or power"
            )

        col = self.joints_to_plot[joint]["col"]

        self.fig.add_trace(
            go.Scatter(
                y=mean + std,
                name="Valores normales",
                x=self.x_values,
                line=dict(width=0, color="grey"),
                legendgroup="mean",
                showlegend=False,
                hoverinfo="skip"
                # marker={"coloraxis": "coloraxis"},
            ),
            row=row + 1,
            col=col + 1,
        )
        self.fig.add_trace(
            go.Scatter(
                y=mean - std,
                name="Valores normales",
                x=self.x_values,
                line=dict(width=0, color="grey"),
                legendgroup="mean",
                showlegend=False,
                fill="tonexty",
                fillcolor="rgba(125,125,125,0.3)",
                hoverinfo="skip"
                # marker={"coloraxis": "coloraxis"},
            ),
            row=row + 1,
            col=col + 1,
        )
        self.fig.add_trace(
            go.Scatter(
                y=mean,
                name="Valores normales",
                x=self.x_values,
                line=dict(width=2, color="grey"),
                legendgroup="mean",
                showlegend=True if row + col == 0 else False,
                hoverinfo="skip"
                # marker={"coloraxis": "coloraxis"},
            ),
            row=row + 1,
            col=col + 1,
        )

    def add_event(self, data, linestyle):
        for k, v in self.joints_to_plot.items():
            for row in range(3):
                self.fig.add_shape(
                    go.layout.Shape(
                        type="line", x0=data, x1=data, y0=-999, y1=999, line=linestyle
                    ),
                    row=row + 1,
                    col=v["col"] + 1,
                )


class KT_Ground_Reaction_Figure:
    x_values = np.linspace(0, 1, 100)
    forces_to_plot = {
        "GRAP": {
            "title": "anteroposterior",
            "col": 0,
            "ground_reaction_data": {
                "min": -22.5,
                "max": 22.5,
                "texts": ["Ant", "Post"],
            },
        },
        "GRML": {
            "title": "mediolateral",
            "col": 1,
            "ground_reaction_data": {"min": -20, "max": 20, "texts": ["Med", "Lat"]},
        },
        "GRVE": {
            "title": "vertical",
            "col": 2,
            "ground_reaction_data": {"min": 0, "max": 150, "texts": ["Vert", ""]},
        },
    }

    titles = [f"Fuerza {i['title']}" for i in forces_to_plot.values()]

    def __init__(self, title_font={"size": 21}, sub_args={}):
        self.title_font = title_font
        self.fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=self.titles,
            shared_xaxes=False,
            shared_yaxes=False,
            vertical_spacing=0.15,
            horizontal_spacing=0.10,
            **sub_args,
        )

    def adjust_range(self, auto=False):
        if not auto:
            for k, v in self.forces_to_plot.items():
                self.fig.update_yaxes(
                    range=[
                        v[f"ground_reaction_data"]["min"],
                        v[f"ground_reaction_data"]["max"],
                    ],
                    row=1,
                    col=v["col"] + 1,
                )
        else:
            for k, v in self.forces_to_plot.items():
                index = v["col"] + 1
                if index == 1:
                    index = ""
                mx = max(
                    (
                        np.nanmax(i["y"])
                        for i in self.fig.data
                        if i["yaxis"] == f"y{index}"
                    ),
                    default=1,
                )
                mn = min(
                    (
                        np.nanmin(i["y"])
                        for i in self.fig.data
                        if i["yaxis"] == f"y{index}"
                    ),
                    default=0,
                )
                self.fig.update_yaxes(range=[mn, mx], row=1, col=v["col"] + 1)

    def add_annotations(self, font={"size": 14}):
        for i in self.fig["layout"]["annotations"]:
            i["font"] = self.title_font

        for k, v in self.forces_to_plot.items():
            self.fig.update_xaxes(
                tickformat=",.0%",
                dtick=0.2,
                row=1,
                col=v["col"] + 1,
                title_text="Ciclo (%)",
            )
        self.fig.update_yaxes(title_text="%Peso")

        for k, v in self.forces_to_plot.items():
            index = v["col"] + 1
            if index == 1:
                index = ""
            ran = self.fig.layout[f"yaxis{index}"]["range"]
            dist = (ran[1] - ran[0]) * 0.15
            texts = v[f"ground_reaction_data"]["texts"]
            for i, text in enumerate(texts):
                annot = {
                    "text": text,
                    "xref": f"x{index}",
                    "y": ran[0] + dist if i else ran[1] - dist,
                    "yref": f"y{index}",
                    "x": 1.05,
                    "yanchor": "middle",
                    "xanchor": "auto",
                    "font": font,
                    "showarrow": False,
                    "align": "left",
                }
                self.fig.layout.annotations += (annot,)

    def add_trace(self, force, side, data):
        col = self.forces_to_plot[force]["col"]
        self.fig.add_trace(
            go.Scatter(
                y=data,
                x=self.x_values,
                name="Derecha" if side.lower() == "r" else "Izquierda",
                line=dict(width=3, color="green" if side.lower() == "r" else "red"),
                legendgroup="right" if side.lower() == "r" else "left",
                showlegend=True if col == 0 else False,
                # marker={"coloraxis": "coloraxis"},
            ),
            row=1,
            col=col + 1,
        )

    def add_consistency_trace(self, force, side, data):
        col = self.forces_to_plot[force]["col"]
        self.fig.add_trace(
            go.Scatter(
                y=data,
                x=self.x_values,
                name="Derecha" if side.lower() == "r" else "Izquierda",
                line=dict(width=2, color="green" if side.lower() == "r" else "red"),
                legendgroup="right" if side.lower() == "r" else "left",
                showlegend=True if col == 0 else False,
                # marker={"coloraxis": "coloraxis"},
            ),
            row=1,
            col=col + 1,
        )

    def add_norm_trace(self, force, mean, std):
        col = self.forces_to_plot[force]["col"]
        self.fig.add_trace(
            go.Scatter(
                y=mean + std,
                name="Valores normales",
                x=self.x_values,
                line=dict(width=0, color="grey"),
                legendgroup="mean",
                showlegend=False,
                hoverinfo="skip"
                # marker={"coloraxis": "coloraxis"},
            ),
            row=1,
            col=col + 1,
        )
        self.fig.add_trace(
            go.Scatter(
                y=mean - std,
                name="Valores normales",
                x=self.x_values,
                line=dict(width=0, color="grey"),
                legendgroup="mean",
                showlegend=False,
                fill="tonexty",
                fillcolor="rgba(125,125,125,0.3)",
                hoverinfo="skip"
                # marker={"coloraxis": "coloraxis"},
            ),
            row=1,
            col=col + 1,
        )
        self.fig.add_trace(
            go.Scatter(
                y=mean,
                name="Valores normales",
                x=self.x_values,
                line=dict(width=2, color="grey"),
                legendgroup="mean",
                showlegend=True if col == 0 else False,
                hoverinfo="skip"
                # marker={"coloraxis": "coloraxis"},
            ),
            row=1,
            col=col + 1,
        )

    def add_event(self, data, linestyle):
        for k, v in self.forces_to_plot.items():
            self.fig.add_shape(
                go.layout.Shape(
                    type="line", x0=data, x1=data, y0=-999, y1=999, line=linestyle
                ),
                row=1,
                col=v["col"] + 1,
            )


class KT_Frontal_Plane_Figure:
    x_values = np.linspace(0, 1, 100)
    joints_to_plot = {
        "HP": {
            "title": "cadera",
            "col": 0,
            "torque_frontal_data": {"min": -1, "max": 2, "texts": ["Abd", "Add"]},
        },
        "K": {
            "title": "rodilla",
            "col": 1,
            "torque_frontal_data": {"min": -1, "max": 2, "texts": ["Val", "Var"]},
        },
        "A": {
            "title": "tobillo",
            "col": 2,
            "torque_frontal_data": {"min": -1, "max": 2, "texts": ["Abd", "Add"]},
        },
    }

    titles = [f"Momento de {i['title']}" for i in joints_to_plot.values()]

    def __init__(self, title_font={"size": 21}, sub_args={}):
        self.title_font = title_font
        self.fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=self.titles,
            shared_xaxes=False,
            shared_yaxes=False,
            vertical_spacing=0.15,
            horizontal_spacing=0.10,
            **sub_args,
        )

    def adjust_range(self, auto=False):
        if not auto:
            for k, v in self.joints_to_plot.items():
                self.fig.update_yaxes(
                    range=[
                        v[f"torque_frontal_data"]["min"],
                        v[f"torque_frontal_data"]["max"],
                    ],
                    row=1,
                    col=v["col"] + 1,
                )
        else:
            for k, v in self.joints_to_plot.items():
                index = v["col"] + 1
                if index == 1:
                    index = ""
                mx = max(
                    (
                        np.nanmax(i["y"])
                        for i in self.fig.data
                        if i["yaxis"] == f"y{index}"
                    ),
                    default=1,
                )
                mn = min(
                    (
                        np.nanmin(i["y"])
                        for i in self.fig.data
                        if i["yaxis"] == f"y{index}"
                    ),
                    default=0,
                )
                self.fig.update_yaxes(range=[mn, mx], row=1, col=v["col"] + 1)

    def add_annotations(self, font={"size": 14}):
        for i in self.fig["layout"]["annotations"]:
            i["font"] = self.title_font

        for k, v in self.joints_to_plot.items():
            self.fig.update_xaxes(
                tickformat=",.0%",
                dtick=0.2,
                row=1,
                col=v["col"] + 1,
                title_text="Ciclo (%)",
            )
        self.fig.update_yaxes(title_text="N m/Kg")

        for k, v in self.joints_to_plot.items():
            index = v["col"] + 1
            if index == 1:
                index = ""
            ran = self.fig.layout[f"yaxis{index}"]["range"]
            dist = (ran[1] - ran[0]) * 0.15
            texts = v[f"torque_frontal_data"]["texts"]
            for i, text in enumerate(texts):
                annot = {
                    "text": text,
                    "xref": f"x{index}",
                    "y": ran[0] + dist if i else ran[1] - dist,
                    "yref": f"y{index}",
                    "x": 1.05,
                    "yanchor": "middle",
                    "xanchor": "auto",
                    "font": font,
                    "showarrow": False,
                    "align": "left",
                }
                self.fig.layout.annotations += (annot,)

    def add_trace(self, joint, side, data):
        col = self.joints_to_plot[joint]["col"]
        self.fig.add_trace(
            go.Scatter(
                y=data,
                x=self.x_values,
                name="Derecha" if side.lower() == "r" else "Izquierda",
                line=dict(width=3, color="green" if side.lower() == "r" else "red"),
                legendgroup="right" if side.lower() == "r" else "left",
                showlegend=True if col == 0 else False,
                # marker={"coloraxis": "coloraxis"},
            ),
            row=1,
            col=col + 1,
        )

    def add_consistency_trace(self, joint, side, data):
        col = self.joints_to_plot[joint]["col"]
        self.fig.add_trace(
            go.Scatter(
                y=data,
                x=self.x_values,
                name="Derecha" if side.lower() == "r" else "Izquierda",
                line=dict(width=2, color="green" if side.lower() == "r" else "red"),
                legendgroup="right" if side.lower() == "r" else "left",
                showlegend=True if col == 0 else False,
                # marker={"coloraxis": "coloraxis"},
            ),
            row=1,
            col=col + 1,
        )

    def add_norm_trace(self, joint, mean, std):
        col = self.joints_to_plot[joint]["col"]
        self.fig.add_trace(
            go.Scatter(
                y=mean + std,
                name="Valores normales",
                x=self.x_values,
                line=dict(width=0, color="grey"),
                legendgroup="mean",
                showlegend=False,
                hoverinfo="skip"
                # marker={"coloraxis": "coloraxis"},
            ),
            row=1,
            col=col + 1,
        )
        self.fig.add_trace(
            go.Scatter(
                y=mean - std,
                name="Valores normales",
                x=self.x_values,
                line=dict(width=0, color="grey"),
                legendgroup="mean",
                showlegend=False,
                fill="tonexty",
                fillcolor="rgba(125,125,125,0.3)",
                hoverinfo="skip"
                # marker={"coloraxis": "coloraxis"},
            ),
            row=1,
            col=col + 1,
        )
        self.fig.add_trace(
            go.Scatter(
                y=mean,
                name="Valores normales",
                x=self.x_values,
                line=dict(width=2, color="grey"),
                legendgroup="mean",
                showlegend=True if col == 0 else False,
                hoverinfo="skip"
                # marker={"coloraxis": "coloraxis"},
            ),
            row=1,
            col=col + 1,
        )

    def add_event(self, data, linestyle):
        for k, v in self.joints_to_plot.items():
            self.fig.add_shape(
                go.layout.Shape(
                    type="line", x0=data, x1=data, y0=-999, y1=999, line=linestyle
                ),
                row=1,
                col=v["col"] + 1,
            )


class KM_Figure:
    x_values = np.linspace(0, 1, 100)
    angles_required = {
        "SHOBLI": {
            "title": "Oblicuidad de tronco",
            "row": 0,
            "col": 0,
            "min": -20,
            "max": 20,
            "texts": ["Arriba", "Abajo"],
        },
        "SHTILTOFF": {
            "title": "Inclinación de tronco",
            "row": 0,
            "col": 1,
            "min": -20,
            "max": 35,
            "texts": ["Ant", "Post"],
        },
        "SHROT": {
            "title": "Rotación de tronco",
            "row": 0,
            "col": 2,
            "min": -30,
            "max": 30,
            "texts": ["Int", "Ext"],
        },
        "POBLI": {
            "title": "Oblicuidad pélvica",
            "row": 1,
            "col": 0,
            "min": -15,
            "max": 20,
            "texts": ["Arriba", "Abajo"],
        },
        "PTILT": {
            "title": "Inclinación pélvica",
            "row": 1,
            "col": 1,
            "min": 0,
            "max": 30,
            "texts": ["Ant", "Post"],
        },
        "PROT": {
            "title": "Rotación pélvica",
            "row": 1,
            "col": 2,
            "min": -20,
            "max": 30,
            "texts": ["Int", "Ext"],
        },
        "HPAA": {
            "title": "Ab-aducción de cadera",
            "row": 2,
            "col": 0,
            "min": -15,
            "max": 15,
            "texts": ["Add", "Abd"],
        },
        "HPFE": {
            "title": "Flexoextensión de cadera",
            "row": 2,
            "col": 1,
            "min": -10,
            "max": 55,
            "texts": ["Flex", "Ext"],
        },
        "HPIE": {
            "title": "Rotación de cadera",
            "row": 2,
            "col": 2,
            "min": -30,
            "max": 30,
            "texts": ["Int", "Ext"],
        },
        "KAA": {
            "title": "Varo-valgo de rodilla",
            "row": 3,
            "col": 0,
            "min": -30,
            "max": 30,
            "texts": ["Var", "Valg"],
        },
        "KFE": {
            "title": "Flexoextensión de rodilla",
            "row": 3,
            "col": 1,
            "min": -10,
            "max": 80,
            "texts": ["Flex", "Ext"],
        },
        "KIE": {
            "title": "Rotación de rodilla",
            "row": 3,
            "col": 2,
            "min": -50,
            "max": 20,
            "texts": ["Int", "Ext"],
        },
        "AFE": {
            "title": "Dorsi-plantarflex. de tobillo",
            "row": 4,
            "col": 1,
            "min": -30,
            "max": 30,
            "texts": ["Dors", "Plant"],
        },
        "AIE": {
            "title": "Progresión de pie",
            "row": 4,
            "col": 2,
            "min": -30,
            "max": 30,
            "texts": ["Int", "Ext"],
        },
    }

    def __init__(self, title_font={"size": 18}, sub_args={}):
        titles = [i["title"] for i in self.angles_required.values()]
        titles.insert(12, None)
        self.fig = make_subplots(
            rows=5,
            cols=3,
            subplot_titles=titles,
            shared_xaxes=False,
            shared_yaxes=False,
            **sub_args
            # print_grid=True,  # Just to get the axis names
        )
        for i in self.fig["layout"]["annotations"]:
            i["font"] = title_font
        for k, v in self.angles_required.items():
            self.fig.update_xaxes(
                tickformat=",.0%",
                dtick=0.2,
                row=v["row"] + 1,
                col=v["col"] + 1,
                title_text="Ciclo (%)",
            )
        # self.fig.write_json("initial_fig.json")

    def add_annotations(self, font={"size": 14}):
        for k, v in self.angles_required.items():
            index = v["row"] * 3 + v["col"] + 1
            if index == 1:
                index = ""
            range = self.fig.layout[f"yaxis{index}"]["range"]
            dist = (range[1] - range[0]) * 0.15
            for i, text in enumerate(v["texts"]):
                annot = {
                    "text": text,
                    "xref": f"x{index}",
                    "y": range[0] + dist if i else range[1] - dist,
                    "yref": f"y{index}",
                    "x": 1.05,
                    "yanchor": "middle",
                    "xanchor": "auto",
                    "font": font,
                    "showarrow": False,
                    "align": "left",
                }
                self.fig.layout.annotations += (annot,)

    def mark_out_of_scale(
        self,
        legend_text="Fuera de rango",
        line_opts={"color": "orange", "width": 3},
    ):
        for k, v in self.angles_required.items():
            index = v["row"] * 3 + v["col"] + 1
            if index == 1:
                index = ""
            mx = max(
                (np.nanmax(i["y"]) for i in self.fig.data if i["yaxis"] == f"y{index}"),
                default=1,
            )
            mn = min(
                (np.nanmin(i["y"]) for i in self.fig.data if i["yaxis"] == f"y{index}"),
                default=0,
            )
            if mx > v["max"] or mn < v["min"]:
                self.fig.update_xaxes(
                    showline=True,
                    linewidth=line_opts.get("width", 3),
                    linecolor=line_opts.get("color", "orange"),
                    mirror="all",
                    row=v["row"] + 1,
                    col=v["col"] + 1,
                )
                self.fig.update_yaxes(
                    showline=True,
                    linewidth=line_opts.get("width", 3),
                    linecolor=line_opts.get("color", "orange"),
                    mirror="all",
                    row=v["row"] + 1,
                    col=v["col"] + 1,
                )

        # add dummy trace to add legend item
        if legend_text:
            self.fig.add_trace(
                go.Scatter(
                    y=[0],
                    name=legend_text,
                    x=[0],
                    showlegend=True,
                    line=line_opts,
                    mode="lines",
                )
            )

    def adjust_range(self, auto=False):
        if not auto:
            for k, v in self.angles_required.items():
                self.fig.update_yaxes(
                    range=[v["min"], v["max"]], row=v["row"] + 1, col=v["col"] + 1
                )
        else:
            for k, v in self.angles_required.items():
                index = v["row"] * 3 + v["col"] + 1
                if index == 1:
                    index = ""
                mx = max(
                    (max(i["y"]) for i in self.fig.data if i["yaxis"] == f"y{index}"),
                    default=1,
                )
                mn = min(
                    (min(i["y"]) for i in self.fig.data if i["yaxis"] == f"y{index}"),
                    default=0,
                )
                self.fig.update_yaxes(
                    range=[mn, mx], row=v["row"] + 1, col=v["col"] + 1
                )

    def add_norm(self, angle, mean, std):
        row = self.angles_required[angle]["row"]
        col = self.angles_required[angle]["col"]
        title = self.angles_required[angle]["title"]

        self.fig.add_trace(
            go.Scatter(
                y=mean + std,
                name="Valores normales",
                x=self.x_values,
                line=dict(width=0, color="grey"),
                legendgroup="mean",
                showlegend=False,
                hoverinfo="skip"
                # marker={"coloraxis": "coloraxis"},
            ),
            row=row + 1,
            col=col + 1,
        )
        self.fig.add_trace(
            go.Scatter(
                y=mean - std,
                name="Valores normales",
                x=self.x_values,
                line=dict(width=0, color="grey"),
                legendgroup="mean",
                showlegend=False,
                fill="tonexty",
                fillcolor="rgba(125,125,125,0.3)",
                hoverinfo="skip"
                # marker={"coloraxis": "coloraxis"},
            ),
            row=row + 1,
            col=col + 1,
        )
        self.fig.add_trace(
            go.Scatter(
                y=mean,
                name="Valores normales",
                x=self.x_values,
                line=dict(width=2, color="grey"),
                legendgroup="mean",
                showlegend=True if row + col == 0 else False,
                hoverinfo="skip"
                # marker={"coloraxis": "coloraxis"},
            ),
            row=row + 1,
            col=col + 1,
        )

    def add_event(self, data, pos, length, legend_text, linestyle):

        axes = list(range(1, 16))
        axes.remove(13)
        axes[0] = ""

        if pos == "top":
            y0 = 1 - length
            y1 = 1
        elif pos == "bottom":
            y0 = 0
            y1 = length
        else:
            raise ValueError(
                f"Unknown position {pos} for the event. It's either 'top' or 'bottom'"
            )

        if isinstance(data, Iterable):
            lines = tuple(
                {
                    "line": linestyle,
                    "type": "line",
                    "x0": j,
                    "x1": j,
                    "xref": f"x{i}",
                    "y0": y0,
                    "y1": y1,
                    "yref": f"y{i} domain",
                }
                for i in axes
                for j in data
            )
        else:
            lines = tuple(
                {
                    "line": linestyle,
                    "type": "line",
                    "x0": data,
                    "x1": data,
                    "xref": f"x{i}",
                    "y0": y0,
                    "y1": y1,
                    "yref": f"y{i} domain",
                }
                for i in axes
            )

        if legend_text:
            self.fig.add_trace(
                go.Scatter(
                    y=[0],
                    name=legend_text,
                    x=[0],
                    showlegend=True,
                    line=linestyle,
                    mode="lines",
                ),
            )

        self.fig.layout.shapes += lines

    def add_mean_event(
        self,
        mean,
        sd,
        pos="top",
        length=1,
        legend_text="",
        show_mean=True,
        show_sd=True,
        line_opts={"color": "red"},
    ):

        axes = list(range(1, 16))
        axes.remove(13)
        axes[0] = ""

        if pos == "top":
            y0 = 1 - length
            y1 = 1
        elif pos == "bottom":
            y0 = 0
            y1 = length
        else:
            raise ValueError(
                f"Unknown position {pos} for the event. It's either 'top' or 'bottom'"
            )

        new_rects = (
            [
                {
                    "fillcolor": line_opts["color"],
                    "line": {"width": 0},
                    "opacity": 0.25,
                    "type": "rect",
                    "x0": mean - sd,
                    "x1": mean + sd,
                    "xref": f"x{i}",
                    "y0": y0,
                    "y1": y1,
                    "yref": f"y{i} domain",
                }
                for i in axes
            ]
            if show_sd
            else []
        )

        new_lines = (
            [
                {
                    "line": line_opts,
                    "type": "line",
                    "x0": mean,
                    "x1": mean,
                    "xref": f"x{i}",
                    "y0": y0,
                    "y1": y1,
                    "yref": f"y{i} domain",
                }
                for i in axes
            ]
            if show_mean
            else []
        )

        # Add dummy trace to make the legend work

        if legend_text:
            self.fig.add_trace(
                go.Scatter(
                    y=[0],
                    name=legend_text,
                    x=[0],
                    showlegend=True,
                    line={**line_opts, **{"width": 3}},
                    opacity=0.25,
                    mode="lines",
                )
            )

        self.fig.layout.shapes += tuple(new_rects + new_lines)

    def _is_in_legend(self, text):
        """Checks if text is already in the legend for the figure

        Args:
            text (_type_): _description_
        """
        return any(i["name"] == text for i in self.fig.data)

    def add_trace(self, angle, side, data, line_opts={}):
        row = self.angles_required[angle]["row"]
        col = self.angles_required[angle]["col"]

        name = f"Ciclo {'derecho' if side.lower() == 'r' else 'izquierdo'}"

        show_legend = False if self._is_in_legend(name) else True

        self.fig.add_trace(
            go.Scatter(
                y=data,
                name=name,
                x=self.x_values,
                legendgroup="right" if side.lower() == "r" else "left",
                showlegend=show_legend,
                line=line_opts,
            ),
            row=row + 1,
            col=col + 1,
        )

    def add_mean_trace(self, angle, side, mean, std, line_opts={}):
        self.add_trace(angle, side, mean, line_opts=line_opts)
        row = self.angles_required[angle]["row"]
        col = self.angles_required[angle]["col"]

        name = "Ciclo" + "derecho" if side.lower() == "r" else "izquierdo"

        self.fig.add_trace(
            go.Scatter(
                y=mean + std,
                x=self.x_values,
                name=name + "(media + std)",
                line={**line_opts, **{"width": 0}},
                legendgroup="right" if side.lower() == "r" else "left",
                showlegend=False,
            ),
            row=row + 1,
            col=col + 1,
        )
        self.fig.add_trace(
            go.Scatter(
                y=mean - std,
                x=self.x_values,
                name=name + "(media - std)",
                line={**line_opts, **{"width": 0}},
                legendgroup="right" if side.lower() == "r" else "left",
                showlegend=False,
                fill="tonexty",
            ),
            row=row + 1,
            col=col + 1,
        )


class KM_Figure_Running:
    x_values = np.linspace(0, 1, 100)
    angles_required = {
        "SHOBLI": {
            "title": "Oblicuidad de tronco",
            "row": 0,
            "col": 0,
            "min": -15,
            "max": 15,
            "texts": ["Arriba", "Abajo"],
        },
        "SHTILTOFF": {
            "title": "Inclinación de tronco",
            "row": 0,
            "col": 1,
            "min": -20,
            "max": 20,
            "texts": ["Ant", "Post"],
        },
        "SHROT": {
            "title": "Rotación de tronco",
            "row": 0,
            "col": 2,
            "min": -30,
            "max": 30,
            "texts": ["Int", "Ext"],
        },
        "POBLI": {
            "title": "Oblicuidad pélvica",
            "row": 1,
            "col": 0,
            "min": -15,
            "max": 20,
            "texts": ["Arriba", "Abajo"],
        },
        "PTILT": {
            "title": "Inclinación pélvica",
            "row": 1,
            "col": 1,
            "min": 0,
            "max": 30,
            "texts": ["Ant", "Post"],
        },
        "PROT": {
            "title": "Rotación pélvica",
            "row": 1,
            "col": 2,
            "min": -20,
            "max": 30,
            "texts": ["Int", "Ext"],
        },
        "HPAA": {
            "title": "Ab-aducción de cadera",
            "row": 2,
            "col": 0,
            "min": -15,
            "max": 15,
            "texts": ["Add", "Abd"],
        },
        "HPFE": {
            "title": "Flexoextensión de cadera",
            "row": 2,
            "col": 1,
            "min": -10,
            "max": 55,
            "texts": ["Flex", "Ext"],
        },
        "HPIE": {
            "title": "Rotación de cadera",
            "row": 2,
            "col": 2,
            "min": -30,
            "max": 30,
            "texts": ["Int", "Ext"],
        },
        "KAA": {
            "title": "Varo-valgo de rodilla",
            "row": 3,
            "col": 0,
            "min": -30,
            "max": 30,
            "texts": ["Var", "Valg"],
        },
        "KFE": {
            "title": "Flexoextensión de rodilla",
            "row": 3,
            "col": 1,
            "min": -10,
            "max": 80,
            "texts": ["Flex", "Ext"],
        },
        "KIE": {
            "title": "Rotación de rodilla",
            "row": 3,
            "col": 2,
            "min": -50,
            "max": 20,
            "texts": ["Int", "Ext"],
        },
        "FAA": {
            "title": "Varo-valgo de retropie",
            "row": 4,
            "col": 0,
            "min": -20,
            "max": 25,
            "texts": ["Var", "Valg"],
        },
        "AFE": {
            "title": "Dorsi-plantarflex. de tobillo",
            "row": 4,
            "col": 1,
            "min": -30,
            "max": 30,
            "texts": ["Dors", "Plant"],
        },
        "AIE": {
            "title": "Progresión de pie",
            "row": 4,
            "col": 2,
            "min": -30,
            "max": 30,
            "texts": ["Int", "Ext"],
        },
    }

    def __init__(self, title_font={"size": 21}, sub_args={}):
        titles = [i["title"] for i in self.angles_required.values()]
        titles.insert(15, None)
        self.fig = make_subplots(
            rows=6,
            cols=3,
            subplot_titles=titles,
            shared_xaxes=False,
            shared_yaxes=False,
            **sub_args
            # print_grid=True,  # Just to get the axis names
        )
        for i in self.fig["layout"]["annotations"]:
            i["font"] = title_font
        for k, v in self.angles_required.items():
            self.fig.update_xaxes(
                tickformat=",.0%",
                dtick=0.2,
                row=v["row"] + 1,
                col=v["col"] + 1,
                title_text="Ciclo (%)",
            )
        # self.fig.write_json("initial_fig.json")

    def add_annotations(self, font={"size": 14}):
        for k, v in self.angles_required.items():
            index = v["row"] * 3 + v["col"] + 1
            if index == 1:
                index = ""
            range = self.fig.layout[f"yaxis{index}"]["range"]
            dist = (range[1] - range[0]) * 0.15
            for i, text in enumerate(v["texts"]):
                annot = {
                    "text": text,
                    "xref": f"x{index}",
                    "y": range[0] + dist if i else range[1] - dist,
                    "yref": f"y{index}",
                    "x": 1.05,
                    "yanchor": "middle",
                    "xanchor": "auto",
                    "font": font,
                    "showarrow": False,
                    "align": "left",
                }
                self.fig.layout.annotations += (annot,)

    def mark_out_of_scale(
        self,
        legend_text="Fuera de rango",
        line_opts={"color": "orange", "width": 3},
    ):
        for k, v in self.angles_required.items():
            index = v["row"] * 3 + v["col"] + 1
            if index == 1:
                index = ""
            mx = max(
                (np.nanmax(i["y"]) for i in self.fig.data if i["yaxis"] == f"y{index}"),
                default=1,
            )
            mn = min(
                (np.nanmin(i["y"]) for i in self.fig.data if i["yaxis"] == f"y{index}"),
                default=0,
            )
            if mx > v["max"] or mn < v["min"]:
                self.fig.update_xaxes(
                    showline=True,
                    linewidth=line_opts.get("width", 3),
                    linecolor=line_opts.get("color", "orange"),
                    mirror="all",
                    row=v["row"] + 1,
                    col=v["col"] + 1,
                )
                self.fig.update_yaxes(
                    showline=True,
                    linewidth=line_opts.get("width", 3),
                    linecolor=line_opts.get("color", "orange"),
                    mirror="all",
                    row=v["row"] + 1,
                    col=v["col"] + 1,
                )

        # add dummy trace to add legend item
        if legend_text:
            self.fig.add_trace(
                go.Scatter(
                    y=[0],
                    name=legend_text,
                    x=[0],
                    showlegend=True,
                    line=line_opts,
                    mode="lines",
                )
            )

    def adjust_range(self, auto=False):
        if not auto:
            for k, v in self.angles_required.items():
                self.fig.update_yaxes(
                    range=[v["min"], v["max"]], row=v["row"] + 1, col=v["col"] + 1
                )
        else:
            for k, v in self.angles_required.items():
                index = v["row"] * 3 + v["col"] + 1
                if index == 1:
                    index = ""
                mx = max(
                    (max(i["y"]) for i in self.fig.data if i["yaxis"] == f"y{index}"),
                    default=1,
                )
                mn = min(
                    (min(i["y"]) for i in self.fig.data if i["yaxis"] == f"y{index}"),
                    default=0,
                )
                self.fig.update_yaxes(
                    range=[mn, mx], row=v["row"] + 1, col=v["col"] + 1
                )
        for k, v in self.angles_required.items():
            if k == "SHOBLI" or k == "SHTILTOFF" or k == "SHROT":
                self.fig.update_yaxes(
                    range=[v["min"], v["max"]], row=v["row"] + 1, col=v["col"] + 1
                )

    def add_norm(self, angle, mean, std):
        row = self.angles_required[angle]["row"]
        col = self.angles_required[angle]["col"]
        title = self.angles_required[angle]["title"]

        self.fig.add_trace(
            go.Scatter(
                y=mean + std,
                name="Valores normales",
                x=self.x_values,
                line=dict(width=0, color="grey"),
                legendgroup="mean",
                showlegend=False,
                hoverinfo="skip"
                # marker={"coloraxis": "coloraxis"},
            ),
            row=row + 1,
            col=col + 1,
        )
        self.fig.add_trace(
            go.Scatter(
                y=mean - std,
                name="Valores normales",
                x=self.x_values,
                line=dict(width=0, color="grey"),
                legendgroup="mean",
                showlegend=False,
                fill="tonexty",
                fillcolor="rgba(125,125,125,0.3)",
                hoverinfo="skip"
                # marker={"coloraxis": "coloraxis"},
            ),
            row=row + 1,
            col=col + 1,
        )
        self.fig.add_trace(
            go.Scatter(
                y=mean,
                name="Valores normales",
                x=self.x_values,
                line=dict(width=2, color="grey"),
                legendgroup="mean",
                showlegend=True if row + col == 0 else False,
                hoverinfo="skip"
                # marker={"coloraxis": "coloraxis"},
            ),
            row=row + 1,
            col=col + 1,
        )

    def add_event(self, data, pos, length, legend_text, linestyle):

        axes = list(range(1, 16))
        axes.remove(13)
        axes[0] = ""

        if pos == "top":
            y0 = 1 - length
            y1 = 1
        elif pos == "bottom":
            y0 = 0
            y1 = length
        else:
            raise ValueError(
                f"Unknown position {pos} for the event. It's either 'top' or 'bottom'"
            )

        if isinstance(data, Iterable):
            lines = tuple(
                {
                    "line": linestyle,
                    "type": "line",
                    "x0": j,
                    "x1": j,
                    "xref": f"x{i}",
                    "y0": y0,
                    "y1": y1,
                    "yref": f"y{i} domain",
                }
                for i in axes
                for j in data
            )
        else:
            lines = tuple(
                {
                    "line": linestyle,
                    "type": "line",
                    "x0": data,
                    "x1": data,
                    "xref": f"x{i}",
                    "y0": y0,
                    "y1": y1,
                    "yref": f"y{i} domain",
                }
                for i in axes
            )

        if legend_text:
            self.fig.add_trace(
                go.Scatter(
                    y=[0],
                    name=legend_text,
                    x=[0],
                    showlegend=True,
                    line=linestyle,
                    mode="lines",
                ),
            )

        self.fig.layout.shapes += lines

    def add_mean_event(
        self,
        mean,
        sd,
        pos="top",
        length=1,
        legend_text="",
        show_mean=True,
        show_sd=True,
        line_opts={"color": "red"},
    ):

        axes = list(range(1, 16))
        axes.remove(13)
        axes[0] = ""

        if pos == "top":
            y0 = 1 - length
            y1 = 1
        elif pos == "bottom":
            y0 = 0
            y1 = length
        else:
            raise ValueError(
                f"Unknown position {pos} for the event. It's either 'top' or 'bottom'"
            )

        new_rects = (
            [
                {
                    "fillcolor": line_opts["color"],
                    "line": {"width": 0},
                    "opacity": 0.25,
                    "type": "rect",
                    "x0": mean - sd,
                    "x1": mean + sd,
                    "xref": f"x{i}",
                    "y0": y0,
                    "y1": y1,
                    "yref": f"y{i} domain",
                }
                for i in axes
            ]
            if show_sd
            else []
        )

        new_lines = (
            [
                {
                    "line": line_opts,
                    "type": "line",
                    "x0": mean,
                    "x1": mean,
                    "xref": f"x{i}",
                    "y0": y0,
                    "y1": y1,
                    "yref": f"y{i} domain",
                }
                for i in axes
            ]
            if show_mean
            else []
        )

        # Add dummy trace to make the legend work

        if legend_text:
            self.fig.add_trace(
                go.Scatter(
                    y=[0],
                    name=legend_text,
                    x=[0],
                    showlegend=True,
                    line={**line_opts, **{"width": 3}},
                    opacity=0.25,
                    mode="lines",
                )
            )

        self.fig.layout.shapes += tuple(new_rects + new_lines)

    def _is_in_legend(self, text):
        """Checks if text is already in the legend for the figure

        Args:
            text (_type_): _description_
        """
        return any(i["name"] == text for i in self.fig.data)

    def add_trace(self, angle, side, data, line_opts={}):
        row = self.angles_required[angle]["row"]
        col = self.angles_required[angle]["col"]

        name = f"Ciclo {'derecho' if side.lower() == 'r' else 'izquierdo'}"

        show_legend = False if self._is_in_legend(name) else True

        self.fig.add_trace(
            go.Scatter(
                y=data,
                name=name,
                x=self.x_values,
                legendgroup="right" if side.lower() == "r" else "left",
                showlegend=show_legend,
                line=line_opts,
            ),
            row=row + 1,
            col=col + 1,
        )

    def add_mean_trace(self, angle, side, mean, std, line_opts={}):
        self.add_trace(angle, side, mean, line_opts=line_opts)
        row = self.angles_required[angle]["row"]
        col = self.angles_required[angle]["col"]

        name = "Ciclo" + "derecho" if side.lower() == "r" else "izquierdo"

        self.fig.add_trace(
            go.Scatter(
                y=mean + std,
                x=self.x_values,
                name=name + "(media + std)",
                line={**line_opts, **{"width": 0}},
                legendgroup="right" if side.lower() == "r" else "left",
                showlegend=False,
            ),
            row=row + 1,
            col=col + 1,
        )
        self.fig.add_trace(
            go.Scatter(
                y=mean - std,
                x=self.x_values,
                name=name + "(media - std)",
                line={**line_opts, **{"width": 0}},
                legendgroup="right" if side.lower() == "r" else "left",
                showlegend=False,
                fill="tonexty",
            ),
            row=row + 1,
            col=col + 1,
        )



def get_normality_shapes_emg(label, norm_mdx, rang=[0, 1]):
    """Checks the normative for EMG normality region, returns it
    as a list of rectangles"""
    data = norm_mdx.norm.cycle.emg[label]
    shapes = [
        go.layout.Shape(
            type="rect",
            xref="x",
            # yref="paper",
            x0=rang[0] + i["start"] * (rang[1] - rang[0]) / 100,
            y0=-1,
            x1=rang[0] + (i["start"] + i["length"]) * (rang[1] - rang[0]) / 100,
            y1=1,
            fillcolor="LightSalmon",
            opacity=0.3,
            layer="below",
            line_width=0,
        )
        for i in data
    ]
    return shapes


class EMG_Figure:
    # Create a blue colorscale form light blue to dark blue, for EMG amplitude

    def __init__(
        self,
        right_emg_labels=[],
        left_emg_labels=[],
        title_font={"size": 21},
        sub_args={},
        max_value_all_channels=0.0005,
        max_energy_all_channels=0.0004,
        freq=1000,
    ):
        self.rows = max(len(right_emg_labels), len(left_emg_labels))
        self.cols = 2 if right_emg_labels and left_emg_labels else 1
        if self.cols == 2:
            subplot_titles = [
                z for i in zip_longest(right_emg_labels, left_emg_labels) for z in i
            ]
            pos = [(r + 1, c + 1) for r in range(self.rows) for c in range(self.cols)]
        elif self.cols == 1:
            subplot_titles = right_emg_labels or left_emg_labels
            pos = [i + 1 for i in range(self.rows)]
        self.mapa = {k: v for k, v in zip(subplot_titles, pos) if k}
        indices = [f"y{i+1 if i!=0 else ''}" for i in range(len(subplot_titles))]
        self.axes_indices = [
            f"yaxis{i+1 if i!=0 else ''}" for i in range(len(subplot_titles))
        ]
        self.mapa_indices = {k: v for k, v in zip(subplot_titles, indices) if k}
        self.mapa_axes_indices = {
            k: v for k, v in zip(subplot_titles, self.axes_indices) if k
        }

        self.fig = make_subplots(
            rows=self.rows,
            cols=self.cols,
            subplot_titles=subplot_titles,
            shared_xaxes=False,
            shared_yaxes=False,
            vertical_spacing=0.09,
        )
        # print(self.fig)
        self.mpl_colorscale = cm.get_cmap("Blues")(np.linspace(0, 1, 100))
        self.mpl_colorscale = colors.ListedColormap(self.mpl_colorscale[30:, :-1])
        self.ply_colorscale = mpl_to_plotly(self.mpl_colorscale)
        self.right_emg_labels = right_emg_labels
        self.left_emg_labels = left_emg_labels
        self.freq = freq
        self.max_value_all_channels = max_value_all_channels
        self.max_energy_all_channels = max_energy_all_channels

    def colorscale(self, color_from_envelope=True):
        # Adding dummy maker trace for the colorscale
        self.fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    colorscale=self.ply_colorscale,
                    coloraxis="coloraxis",
                    showscale=True,
                    colorbar=dict(thickness=5),
                ),
                hoverinfo="none",
            )
        )

        self.fig.update_layout(
            coloraxis={
                "colorscale": self.ply_colorscale,
                "showscale": True,
                "cmin": 0,
                "cmax": self.max_value_all_channels
                if not color_from_envelope
                else self.max_energy_all_channels,
                "colorbar": dict(
                    thickness=20,
                    tickvals=list(
                        np.linspace(
                            0,
                            self.max_value_all_channels
                            if not color_from_envelope
                            else self.max_energy_all_channels,
                            11,
                        )
                    ),
                    tickformat=".2s",
                    len=0.7,
                ),
            }
        )

    def add_normality(self, label, norm_mdx, rang=[0, 1]):
        n_label = label.replace("Right ", "").replace("Left ", "")
        for shape in get_normality_shapes_emg(n_label, norm_mdx, rang):
            self.fig.add_shape(shape, row=self.mapa[label][0], col=self.mapa[label][1])

    def add_event(self, label, data, linestyle, **kwargs):
        def_opts = dict(y0=-1, y1=1, yref="paper")
        def_opts.update(kwargs)
        self.fig.add_shape(
            go.layout.Shape(type="line", x0=data, x1=data, line=linestyle, **def_opts),
            row=self.mapa[label][0],
            col=self.mapa[label][1],
        )

    def adjust_xaxes(self):
        # # adjust xaxes to fit the cycle
        for k, v in self.mapa.items():
            self.fig.update_xaxes(
                range=[0, 1],
                tickformat=",.0%",
                row=v[0],
                col=v[1],
                title_text=None if v[0] < self.rows else "Ciclo (%)",
            )

    def has_traces(self, label):
        axis = self.mapa_indices[label]
        val = [i["yaxis"] for i in self.fig.data]
        return True if axis in val else False

    def add_trace(
        self,
        label,
        data,
        plot_envelope=False,
        plot_data=True,
        color_from_envelope=True,
        legendgroup=None,
    ):
        if not legendgroup:
            legendgroup = "filtered"
        first_trace = not self.has_traces(label)
        emg_time = np.linspace(0, 1, len(data))
        envelope = get_emg_envelope(data, self.freq)  # check max value
        max_value_scale = np.max(np.abs(data))
        max_envelope = np.max(envelope)
        if color_from_envelope:
            # print(f"Max envelope value: {max_envelope}")
            color_line = self.mpl_colorscale(
                max_envelope / self.max_energy_all_channels
            )
        else:
            color_line = self.mpl_colorscale(
                max_value_scale / self.max_value_all_channels
            )
        color_line = f"rgb({color_line[0]},{color_line[1]},{color_line[2]})"

        if plot_data:
            self.fig.add_trace(
                go.Scatter(
                    y=data,
                    x=emg_time,
                    name=label,
                    line=dict(width=1, color=color_line),
                    legendgroup=legendgroup,
                    showlegend=True if first_trace else False,
                    marker={"coloraxis": "coloraxis"},
                ),
                row=self.mapa[label][0],
                col=self.mapa[label][1],
            )
        # add envelope
        if plot_envelope:
            self.fig.add_trace(
                go.Scatter(
                    y=envelope,
                    x=emg_time,
                    name=label,
                    line=dict(width=1, color="magenta"),
                    legendgroup="envelope",
                    showlegend=True if first_trace else False,
                    marker={"coloraxis": "coloraxis"},
                ),
                row=self.mapa[label][0],
                col=self.mapa[label][1],
            )
        current_range = self.fig.layout[self.mapa_axes_indices[label]]["range"]
        if not current_range:
            current_range = [0, 0]

        max_to_use = (
            max_envelope if (plot_envelope and not plot_data) else max_value_scale
        )

        replace = [-max_to_use, max_to_use] if plot_data else [0, max_to_use]

        self.fig.update_yaxes(
            range=replace if max_to_use > current_range[1] else current_range,
            row=self.mapa[label][0],
            col=self.mapa[label][1],
            nticks=8,
            tickmode="auto",
            title_text="Amplitud (V)" if self.mapa[label][1] == 1 else "",
        )


# if __name__ == "__main__":
#     a = KT_Figure()
#     a.add_annotations()
#     a.fig.show()
