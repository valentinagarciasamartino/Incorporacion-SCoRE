import numpy as np
from itertools import zip_longest
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, sosfiltfilt

KM_angles_required = {
    "Oblicuidad de tronco": {
        "code": "SHOBLI",
        "row": 0,
        "col": 0,
        "min": -20,
        "max": 20,
        "texts": ["Arriba", "Abajo"],
    },
    "Inclinación de tronco": {
        "code": "SHTILTOFF",
        "row": 0,
        "col": 1,
        "min": -20,
        "max": 35,
        "texts": ["Ant", "Post"],
    },
    "Rotación de tronco": {
        "code": "SHROT",
        "row": 0,
        "col": 2,
        "min": -30,
        "max": 30,
        "texts": ["Int", "Ext"],
    },
    "Oblicuidad pélvica": {
        "code": "POBLI",
        "row": 1,
        "col": 0,
        "min": -15,
        "max": 20,
        "texts": ["Arriba", "Abajo"],
    },
    "Inclinación pélvica": {
        "code": "PTILT",
        "row": 1,
        "col": 1,
        "min": 0,
        "max": 30,
        "texts": ["Ant", "Post"],
    },
    "Rotación pélvica": {
        "code": "PROT",
        "row": 1,
        "col": 2,
        "min": -20,
        "max": 30,
        "texts": ["Int", "Ext"],
    },
    "Ab-aducción de cadera": {
        "code": "HPAA",
        "row": 2,
        "col": 0,
        "min": -15,
        "max": 15,
        "texts": ["Add", "Abd"],
    },
    "Flexoextensión de cadera": {
        "code": "HPFE",
        "row": 2,
        "col": 1,
        "min": -10,
        "max": 55,
        "texts": ["Flex", "Ext"],
    },
    "Rotación de cadera": {
        "code": "HPIE",
        "row": 2,
        "col": 2,
        "min": -30,
        "max": 30,
        "texts": ["Int", "Ext"],
    },
    "Varo-valgo de rodilla": {
        "code": "KAA",
        "row": 3,
        "col": 0,
        "min": -30,
        "max": 30,
        "texts": ["Var", "Valg"],
    },
    "Flexoextensión de rodilla": {
        "code": "KFE",
        "row": 3,
        "col": 1,
        "min": -10,
        "max": 80,
        "texts": ["Flex", "Ext"],
    },
    "Rotación de rodilla": {
        "code": "KIE",
        "row": 3,
        "col": 2,
        "min": -50,
        "max": 20,
        "texts": ["Int", "Ext"],
    },
    "Dorsi-plantarflex. de tobillo": {
        "code": "AFE",
        "row": 4,
        "col": 1,
        "min": -30,
        "max": 30,
        "texts": ["Dors", "Plant"],
    },
    "Progresión de pie": {
        "code": "AIE",
        "row": 4,
        "col": 2,
        "min": -30,
        "max": 30,
        "texts": ["Int", "Ext"],
    },
}


def add_event_line(trial, event, line, normalized=False, *args, **kwargs):
    """ Returns the event as a list of vertical lines """
    if normalized:
        try:
            data = trial.cycle.events[event].data[0][0]
        except:
            print(f"No {event}")
            return []
    else:
        data = trial.static.events[event].data

    def_opts = dict(y0=-1, y1=1, yref="paper")
    def_opts.update(kwargs)
    # Line Vertical
    shapes = [
        go.layout.Shape(type="line", x0=i, x1=i, line=line, **def_opts) for i in data
    ]
    return shapes


def add_KM_traces(fig, angs):
    """ adds all the traces to a KM graph """
    x_values = np.linspace(0, 1, 100)
    for n, m in enumerate(angs.items(), start=1):

        # don't use axis 13 (row 5, col1), there's nothing to be plotted there
        if n >= 13:
            continue

        k, v = m

        fig.add_trace(
            go.Scatter(
                y=v["data"]["right"],
                x=x_values,
                name="Derecha",
                line=dict(width=3, color="green"),
                legendgroup="right",
                showlegend=True if v["row"] + v["col"] == 0 else False,
                # marker={"coloraxis": "coloraxis"},
            ),
            row=v["row"] + 1,
            col=v["col"] + 1,
        )
        fig.add_trace(
            go.Scatter(
                y=v["data"]["left"],
                name="Izquierda",
                x=x_values,
                line=dict(width=3, color="red"),
                legendgroup="left",
                showlegend=True if v["row"] + v["col"] == 0 else False,
                # marker={"coloraxis": "coloraxis"},
            ),
            row=v["row"] + 1,
            col=v["col"] + 1,
        )
    return fig


def add_KM_norms(fig, angs):
    """ adds all the norms to a KM graph"""
    x_values = np.linspace(0, 1, 100)
    for n, m in enumerate(angs.items(), start=1):
        if n >= 13:
            continue
        k, v = m
        fig.add_trace(
            go.Scatter(
                y=v["norm"]["mean"] + v["norm"]["sd"],
                name=k,
                x=x_values,
                line=dict(width=0, color="grey"),
                legendgroup="mean",
                showlegend=False,
                # marker={"coloraxis": "coloraxis"},
            ),
            row=v["row"] + 1,
            col=v["col"] + 1,
        )
        fig.add_trace(
            go.Scatter(
                y=v["norm"]["mean"] - v["norm"]["sd"],
                name=k,
                x=x_values,
                line=dict(width=0, color="grey"),
                legendgroup="mean",
                showlegend=False,
                fill="tonexty",
                fillcolor="rgba(125,125,125,0.3)"
                # marker={"coloraxis": "coloraxis"},
            ),
            row=v["row"] + 1,
            col=v["col"] + 1,
        )
        fig.add_trace(
            go.Scatter(
                y=v["norm"]["mean"],
                name="Valores normales",
                x=x_values,
                line=dict(width=2, color="grey"),
                legendgroup="mean",
                showlegend=True if v["row"] + v["col"] == 0 else False,
                # marker={"coloraxis": "coloraxis"},
            ),
            row=v["row"] + 1,
            col=v["col"] + 1,
        )
    return fig


def add_KM_events(fig, trial, angs, autorange=False):
    print(fig.layout)
    for n, m in enumerate(angs.items(), start=1):
        k, v = m
        for i in add_event_line(
            trial,
            "enLTO",
            line=dict(color="red", width=2, dash="longdash"),
            normalized=True,
            y1=1,
            y0=0,
            yref="paper",
        ):
            fig.add_shape(i, row=v["row"] + 1, col=v["col"] + 1)
        for i in add_event_line(
            trial,
            "enRTO",
            line=dict(color="green", width=2, dash="longdash"),
            normalized=True,
            y1=1,
            y0=0,
            yref="paper",
        ):
            fig.add_shape(i, row=v["row"] + 1, col=v["col"] + 1)
    for i in fig.layout.shapes:
        i["yref"] = "paper"

    return fig


def mpl_to_plotly(cmap, pl_entries=11, rdigits=2):
    """Converts a continuous Matplotlib colormap to a plotly colormap.
    Used beacause plotly won't provide a nice way to interpolate colour
    ranges
    """

    scale = np.linspace(0, 1, pl_entries)
    colors = (cmap(scale)[:, :3] * 255).astype(np.uint8)
    pl_colorscale = [
        [round(s, rdigits), f"rgb{tuple(color)}"] for s, color in zip(scale, colors)
    ]
    return pl_colorscale


def cut_to_cycles(arr, hs, freq):
    """Cuts arrays into chunks from heel strike to heel strike"""
    ind = sorted([int(i * freq) for i in hs])
    return [arr[..., ind[i] : ind[i + 1]] for i in range(len(ind) - 1)]


def bandpass(data, lowcut=50.0, highcut=450.0, fs=1000, order=2):
    """Bandpass filter for EMG"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype="bandpass", analog=False, output="sos")
    return sosfiltfilt(sos, data)


def lowpass(data, highcut=450.0, fs=1000, order=2):
    """Lowpass filter for EMG"""
    nyq = 0.5 * fs
    high = highcut / nyq
    sos = butter(order, [high], btype="lowpass", analog=False, output="sos")
    return sosfiltfilt(sos, data)


def get_emg_envelope(val, freq, cut=50, order=4):
    return lowpass(np.abs(val - np.mean(val)), cut, fs=freq, order=order)


def is_facing_forward(markers):
    """ Returns walking direction. Positive if in the x direction of the subject"""
    r_hip_x = markers["MIDASIS"].data[0]
    sacrum_x = markers["sacrum"].data[0]
    # Checks the first non null pair of r_hip and sacrum, x coordinate. If hip>sacrum, is facing forward
    print(r_hip_x)
    pos = next(
        (i > j for i, j in zip(r_hip_x, sacrum_x) if i != np.NaN and j != np.NaN)
    )

    print(pos)

    if pos:
        print("IS FORWARD")
        return True
    print("IS BACKWARD")
    return False


class GaitIndices:
    def __init__(self, trial):
        self.trial = trial
        self.markers = self.trial.markerStream.tracks
        self.freq = self.trial.markerStream.frequency
        self.events = self.trial.static.events
        self.r_hs = sorted(self.events["eRHS"].data)
        self.r_to = sorted(self.events["eRTO"].data)
        a = [j for i in zip_longest(self.r_hs, self.r_to) for j in i][:-1]
        assert a == sorted(self.r_hs + self.r_to), "RHS and RTO not in alternated order"

        self.l_hs = sorted(self.events["eLHS"].data)
        self.l_to = sorted(self.events["eLTO"].data)
        a = [j for i in zip_longest(self.l_hs, self.l_to) for j in i][:-1]
        assert a == sorted(self.l_hs + self.l_to), "LHS and LTO not in alternated order"

        self.r_hs_ind = sorted([int(i * self.freq) for i in self.r_hs])
        self.r_to_ind = sorted([int(i * self.freq) for i in self.r_to])

        self.l_hs_ind = sorted([int(i * self.freq) for i in self.l_hs])
        self.l_to_ind = sorted([int(i * self.freq) for i in self.l_to])

    def hip_hiking(self):
        """ Returns the difference between the maximum height of each hip joint during
        swing phase and its contralateral hip joint, minus the average height difference
        during stance"""

        r_hip = self.markers[f"RHP"].data
        l_hip = self.markers[f"LHP"].data

        r_hs = self.r_hs_ind
        l_hs = self.l_hs_ind

        r_to = self.r_to_ind
        l_to = self.l_to_ind

        max_r_hip_swing = []
        max_l_hip_swing = []
        stance_avg_r_hip = []
        stance_avg_l_hip = []

        for i in range(len(r_hs) - 1):
            swing = slice(r_to[i], r_hs[i + 1])  # swing is from RTO to next RHS
            stance = slice(r_hs[i], r_to[i])  # stance is from RHS to RTO
            r_hip_swing = r_hip[1, swing]  # hip height at swing
            max_val = max(r_hip_swing)
            max_pos = np.where(r_hip_swing == max_val)[0][0] + r_hs[i]
            l_hip_val = l_hip[1][max_pos]
            max_r_hip_swing.append(max_val - l_hip_val)
            stance_avg_r_hip.append(np.mean(r_hip[1, stance]))

        for i in range(len(l_hs) - 1):
            swing = slice(l_to[i], l_hs[i + 1])  # swing is from RTO to next RHS
            stance = slice(l_hs[i], l_to[i])  # stance is from RHS to RTO
            l_hip_swing = l_hip[1, swing]  # hip height at swing
            max_val = max(l_hip_swing)
            max_pos = np.where(l_hip_swing == max_val)[0][0] + l_hs[i]
            r_hip_val = r_hip[1][max_pos]
            max_l_hip_swing.append(max_val - r_hip_val)
            stance_avg_l_hip.append(np.mean(l_hip[1, stance]))

        r_res = [
            max_r_hip_swing[i] / (stance_avg_r_hip[i] - stance_avg_l_hip)
            for i in range(len(max_r_hip_swing))
        ]
        l_res = [
            max_l_hip_swing[i] / (stance_avg_l_hip[i] - stance_avg_r_hip)
            for i in range(len(max_l_hip_swing))
        ]

        return (r_res, l_res)

    def circumduction(self):
        """the difference in distance between the lateral-most X coordinate of the ankle joint marker during
        25 - 75 % of the swing phase and the medial-most X coordinate during 25 - 75 % of the stance
        phase"""

        r_ank = self.markers[f"RA"].data
        l_ank = self.markers[f"LA"].data

        r_hs = self.r_hs_ind
        l_hs = self.l_hs_ind

        r_to = self.r_to_ind
        l_to = self.l_to_ind

        for i in range(len(r_hs) - 1):
            swing = slice(r_to[i], r_hs[i + 1])  # swing is from RTO to next RHS
            stance = slice(r_hs[i], r_to[i])  # stance is from RHS to RTO
            r_hip_swing = r_hip[1, swing]  # hip height at swing
            max_val = max(r_hip_swing)
            max_pos = np.where(r_hip_swing == max_val)[0][0] + r_hs[i]
            l_hip_val = l_hip[1][max_pos]
            max_r_hip_swing.append(max_val - l_hip_val)
            stance_avg_r_hip.append(np.mean(r_hip[1, stance]))

        for i in range(len(l_hs) - 1):
            swing = slice(l_to[i], l_hs[i + 1])  # swing is from RTO to next RHS
            stance = slice(l_hs[i], l_to[i])  # stance is from RHS to RTO
            l_hip_swing = l_hip[1, swing]  # hip height at swing
            max_val = max(l_hip_swing)
            max_pos = np.where(l_hip_swing == max_val)[0][0] + l_hs[i]
            r_hip_val = r_hip[1][max_pos]
            max_l_hip_swing.append(max_val - r_hip_val)
            stance_avg_l_hip.append(np.mean(l_hip[1, stance]))

        # r_hip_cut = cut_to_cycles(r_hip, events["eRHS"].data, freq)

        # l_hip = markers[f"LHP"].data
        # l_hip_cut = cut_to_cycles(r_hip, events["eLHS"].data, freq)


norm_color = lambda opacity: f"rgba(125,125,125,{opacity})"
left_color = lambda opacity: f"rgba(255,0,0,{opacity})"
right_color = lambda opacity: f"rgba(0,128,0,{opacity})"
both_color = lambda opacity: f"rgba(0,0,255,{opacity})"


def stance_swing_time_subfig(vals):
    traces = []
    for n, m in enumerate(vals.items()):
        k, v = m
        marker_opacity = 0.7 if not n else 0.5
        err = [v["N"]["error"], v["N"]["error"]]
        traces.append(
            go.Bar(
                x=["Control"],
                y=v["N"]["data"],
                error_y={"array": err},
                text=f"{v['N']['data'][0]:.2f}",
                textfont={"color": "black"},
                textposition="auto",
                insidetextanchor="middle",
                marker={"color": norm_color(marker_opacity)},
                marker_line_color=norm_color(1),
                marker_line_width=1.5,
                offsetgroup=0,
                base=0 if not n else list(vals.values())[n - 1]["N"]["data"],
            )
        )

        for side in ["R", "L"]:
            err = [v[side]["error"], v[side]["error"]]
            marker_color = (
                right_color(marker_opacity)
                if side == "R"
                else left_color(marker_opacity)
            )
            marker_line_color = right_color(1) if side == "R" else left_color(1)

            traces.append(
                go.Bar(
                    x=["Derecha" if side == "R" else "Izquierda"],
                    y=v[side]["data"],
                    error_y={"array": err, "color": marker_line_color},
                    text=f"{v[side]['data'][0]:.2f}",
                    textfont={"color": "black"},
                    textposition="auto",
                    insidetextanchor="middle",
                    marker_line_width=1.5,
                    marker_color=marker_color,
                    marker_line_color=marker_line_color,
                    offsetgroup=0,
                    base=0 if not n else list(vals.values())[n - 1][side]["data"],
                )
            )

    return traces


def stance_swing_phase_subfig(vals):
    traces = []
    for n, m in enumerate(vals.items()):
        marker_opacity = 0.7 if not n else 0.5
        k, v = m
        err = [v["N"]["error"], v["N"]["error"]]
        traces.append(
            go.Bar(
                x=["Control"],
                y=v["N"]["data"],
                error_y={"array": err},
                text=f"{v['N']['data'][0]:.0%}",
                textfont={"color": "black"},
                textposition="auto",
                insidetextanchor="middle",
                marker={"color": norm_color(marker_opacity)},
                marker_line_color=norm_color(1),
                marker_line_width=1.5,
                offsetgroup=0,
                base=0 if not n else list(vals.values())[n - 1]["N"]["data"],
            )
        )

        for side in ["R", "L"]:
            marker_color = (
                right_color(marker_opacity)
                if side == "R"
                else left_color(marker_opacity)
            )
            marker_line_color = right_color(1) if side == "R" else left_color(1)

            traces.append(
                go.Bar(
                    x=["Derecha" if side == "R" else "Izquierda"],
                    y=v[side]["data"],
                    marker={"color": marker_color},
                    marker_line_color=marker_line_color,
                    text=f"{v[side]['data'][0]:.0%}",
                    textfont={"color": "black"},
                    textposition="auto",
                    insidetextanchor="middle",
                    marker_line_width=1.5,
                    offsetgroup=0,
                    base=0 if not n else list(vals.values())[n - 1][side]["data"],
                )
            )
    return traces


def cadence_subfig(vals):
    traces = []
    for n, m in enumerate(vals.items()):
        k, v = m
        marker_opacity = 0.6
        dat = v["N"]["data"] * 60
        err = [v["N"]["error"] * 60]

        traces.append(
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
            )
        )

        for side in ["B"]:
            dat = v[side]["data"] * 60
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
            else:
                x = "Total"
                marker_color = both_color(marker_opacity)
                marker_line_color = both_color(1)
                offsetgroup = 0

            traces.append(
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
                )
            )

    return traces


def stance_single_double_phase_subfig(vals):
    traces = []
    for n, m in enumerate(vals.items()):

        marker_opacity = 0.6
        k, v = m
        dat = v["N"]["data"]
        err = [v["N"]["error"], v["N"]["error"]]
        traces.append(
            go.Bar(
                x=["Doble" if "doble" in k else "Simple"],
                y=dat,
                error_y={"array": err},
                text=f"{dat[0]:.0%}",
                textfont={"color": "black"},
                textposition="auto",
                insidetextanchor="middle",
                marker={"color": norm_color(marker_opacity)},
                marker_line_color=norm_color(1),
                marker_line_width=1.5,
                offsetgroup=0,
            )
        )

        for side in ["R", "L"]:
            dat = v[side]["data"]
            marker_color = (
                right_color(marker_opacity)
                if side == "R"
                else left_color(marker_opacity)
            )
            marker_line_color = right_color(1) if side == "R" else left_color(1)

            traces.append(
                go.Bar(
                    x=["Doble" if "doble" in k else "Simple"],
                    y=dat,
                    marker={"color": marker_color},
                    marker_line_color=marker_line_color,
                    text=f"{dat[0]:.0%}",
                    textfont={"color": "black"},
                    textposition="auto",
                    insidetextanchor="middle",
                    marker_line_width=1.5,
                    offsetgroup=1 if side == "R" else 2,
                )
            )

    return traces


def speeds_subfig(vals):
    traces = []
    for n, m in enumerate(vals.items()):
        marker_opacity = 0.6
        k, v = m
        dat = v["N"]["data"]
        err = [v["N"]["error"], v["N"]["error"]]
        traces.append(
            go.Bar(
                x=["Balanceo" if "balanceo" in k else "Media"],
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
            )
        )

        for side in ["R", "L"]:
            dat = v[side]["data"]
            err = [v[side]["error"], v[side]["error"]]
            marker_color = (
                right_color(marker_opacity)
                if side == "R"
                else left_color(marker_opacity)
            )
            marker_line_color = right_color(1) if side == "R" else left_color(1)

            traces.append(
                go.Bar(
                    x=["Balanceo" if "balanceo" in k else "Media"],
                    y=dat,
                    marker={"color": marker_color},
                    marker_line_color=marker_line_color,
                    text=f"{dat[0]:.2f}",
                    textfont={"color": "black"},
                    textposition="auto",
                    error_y={"array": err, "color": marker_line_color},
                    insidetextanchor="middle",
                    marker_line_width=1.5,
                    offsetgroup=1 if side == "R" else 2,
                )
            )
    return traces
