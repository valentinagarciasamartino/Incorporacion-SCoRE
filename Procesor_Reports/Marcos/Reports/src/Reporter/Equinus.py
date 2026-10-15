import numpy as np
from Figures import Equinus_Figure
from pathlib import Path
from simplemdx.declparser import mdxParser
from functions import bandpass
import plotly.graph_objects as go


def get_muscles_activations(trial, muscle, cut=True, filter=True):
    side = "Right" if muscle.startswith("Right") else "Left"
    emg = trial.trial.emgStream.emgChannels[muscle].data[0]
    if filter:
        emg = bandpass(emg, highcut=200)
    if not cut:
        return [emg]

    events = list(
        zip(
            sorted(trial.trial.static.events[f"{side} start"].data),
            sorted(trial.trial.static.events[f"{side} end"].data),
        )
    )
    freq = trial.trial.emgStream.frequency
    segments = [emg[int(start * freq) : int(end * freq)] for start, end in events]
    return segments


class Equinus_report:
    muscles = {
        "Rectus femoris": {"row": 0},
        "Tibialis anterior": {"row": 1},
        "Gastrocnemius medialis": {"row": 2},
    }

    def __init__(self, normal_trials, resistance_trials):
        self.normal_left_trials = [
            i
            for i in normal_trials
            if "Left start" in i.trial.static.events
            and i.trial.static.events["Left start"].nItems != 0
        ]
        self.normal_right_trials = [
            i for i in normal_trials if i.trial.static.events["Right start"].nItems != 0
        ]
        self.resistance_left_trials = [
            i
            for i in resistance_trials
            if "Left start" in i.trial.static.events
            and i.trial.static.events["Left start"].nItems != 0
        ]
        self.resistance_right_trials = [
            i
            for i in resistance_trials
            if i.trial.static.events["Right start"].nItems != 0
        ]
        self.figure = Equinus_Figure()

    def plot_all(self):
        for k, v in self.muscles.items():
            for i in self.normal_left_trials:
                self.plot_muscle_activations(
                    i, "Left " + k, "rgba(255,0,0,0.5)", v["row"], 1
                )
            for i in self.normal_right_trials:
                self.plot_muscle_activations(
                    i, "Right " + k, "rgba(0,128,0,0.5)", v["row"], 0
                )
            for i in self.resistance_left_trials:
                self.plot_muscle_activations(
                    i, "Left " + k, "rgba(229,9,127,0.5)", v["row"] + 3, 1
                )
            for i in self.resistance_right_trials:
                self.plot_muscle_activations(
                    i, "Right " + k, "rgba(0,0,255,0.5)", v["row"] + 3, 0
                )

    def plot_muscle_activations(self, trial, muscle, color, row, col):
        segments = get_muscles_activations(trial, muscle)
        for n, i in enumerate(segments):
            self.figure.fig.add_trace(
                go.Scatter(
                    y=i,
                    x=np.linspace(0, 1, len(i)),
                    name=f"Ciclo {n}",
                    line=dict(width=1, color=color),
                    legendgroup=f"{n}",
                    showlegend=True if not row else False,
                    marker={"coloraxis": "coloraxis"},
                ),
                row=row + 1,
                col=col + 1,
            )

    # def plot_events(self,trial):
    #     for k,v in self.muscles.items():


normal_trials = ["2598~aa~Walking 13.mdx", "2598~aa~Walking 14.mdx"]
normal_trials = [Path("./test_files/New folder/") / i for i in normal_trials]
normal_trials = [mdxParser(i) for i in normal_trials]
normal_left_trials = [
    i for i in normal_trials if i.trial.static.events["Left start"].nItems != 0
]
normal_right_trials = [
    i for i in normal_trials if i.trial.static.events["Right start"].nItems != 0
]
# # indentify left from right trials:
# # get all normal rectus femoris
# normal_left_rf = []
# for i in normal_left_trials:
#     y = i.trial.emgStream.emgChannels["Left Rectus femoris"].data[0]

#     self.fig.add_trace(
#     )

resistance_trials = ["2598~aa~Walking 15.mdx", "2598~aa~Walking 16.mdx"]
resistance_trials = [Path("./test_files/New folder/") / i for i in resistance_trials]
resistance_trials = [mdxParser(i) for i in resistance_trials]
a = Equinus_report(normal_trials, resistance_trials)
a.plot_all()
a.figure.fig.show()
a.figure.fig.write_html("Kaliroff.html")
a.figure.fig.update_layout(width=1600, height=1600, autosize=True)
a.figure.fig.write_image("Kaliroff.png")
