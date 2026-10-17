import logging
import os
import tempfile
from pathlib import Path

import plotly.graph_objects as go
from simplemdx.declparser import mdxParser
from weasyprint import CSS, HTML

from Figures import (KT_Ang_Torque_Power_Figure, KT_Frontal_Plane_Figure,
                     KT_Ground_Reaction_Figure)
from functions import *

logging.basicConfig(level=logging.INFO)

# Render engine can be either kaleido. Kaleido is faster, but needs some more work
RENDER_ENGINE = "kaleido"


class Consist_repo:
    def __init__(self, trials, session, standing=None, normative=None, anonymize=False):
        self.trial_paths = []
        for trial in trials:
            self.trial_paths.append(Path(trial))
        self.session_path = Path(session)
        self.session_mdx = mdxParser(self.session_path.as_posix())
        self.trial_mdxs = []
        for trial_path in self.trial_paths:
            trial_mdx = mdxParser(trial_path.as_posix())
            if trial_mdx.trial.has_kinetics:
                self.trial_mdxs.append(trial_mdx)
        
        separador = os.path.sep
        session_file_name =  str(self.session_path).split(separador)[-1]
        if len(self.trial_mdxs) == 0:
            print(f"No forces in {session_file_name}")       
                
        self.anonymize = anonymize
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

        if standing:
            self.standing_path = Path(standing)
            self.standing_mdx = mdxParser(self.standing_path.as_posix())

        self.normative_mdx = mdxParser(self.normative_path.as_posix())

        self.name = self.session_mdx.trial.static.name
        self.lastname = self.session_mdx.trial.static.lastname

        if self.anonymize:
            self.name = self.name[0].capitalize()
            self.lastname = self.lastname[0].capitalize()
        self.date = self.session_mdx.trial.static.date.strftime("%d/%m/%Y")
        
        self.session_name =  session_file_name.split("~")[1]+"~"+session_file_name.split("~")[2].split(".")[0]
        self.patid = session_file_name.rsplit("~")[0]
                
        self.ntrial = ""
        for i, trial_mdx in enumerate(self.trial_mdxs) :
            self.ntrial = self.ntrial + trial_mdx.trial.label.split("~")[-1][8:-4]
            if i != (len(self.trial_mdxs)-1) :
                self.ntrial = self.ntrial + ", "         

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
                {self.lastname}, {self.name}
            </td>
            <td style="text-align:center;">
               {self.session_name}
            </td>
            <td style="text-align:right">
                    {self.date}
            </td>
            </tr>
            </table>
            </footer>"""
        self.pages = []


    def KT_static_report(self, autorange=False, mark_out_of_range=False):
        """Prints the kinetic report for all trials into a PDF"""

        self.KT_stat_fig1r = KT_Ang_Torque_Power_Figure()
        self.KT_stat_fig1l = KT_Ang_Torque_Power_Figure()
        joints_to_plot = KT_Ang_Torque_Power_Figure.joints_to_plot.keys()
        
        norm = self.normative_mdx.norm.cycle
        # Adding normatives
        for joint in joints_to_plot:
            ang = norm.angle_bands[f"a{joint}FE"]
            self.KT_stat_fig1r.add_norm_trace(
                joint=joint, kind="angle", mean=ang["mean"], std=ang["sd"]
            )
            self.KT_stat_fig1l.add_norm_trace(
                joint=joint, kind="angle", mean=ang["mean"], std=ang["sd"]
            )
            torq = norm.torque1d_bands[f"tcm{joint}FE"]
            self.KT_stat_fig1r.add_norm_trace(
                joint=joint, kind="torque", mean=torq["mean"], std=torq["sd"]
            )
            self.KT_stat_fig1l.add_norm_trace(
                joint=joint, kind="torque", mean=torq["mean"], std=torq["sd"]
            )
            power = norm.power_bands[f"pcm{joint}FE"]
            self.KT_stat_fig1r.add_norm_trace(
                joint=joint, kind="power", mean=power["mean"], std=power["sd"]
            )
            self.KT_stat_fig1l.add_norm_trace(
                joint=joint, kind="power", mean=power["mean"], std=power["sd"]
            )
                
        # 2nd figure: frontal plane kinetics
        self.KT_stat_fig2r = KT_Frontal_Plane_Figure()
        self.KT_stat_fig2l = KT_Frontal_Plane_Figure()
        
        # Add nomatives
        for joint in joints_to_plot:
            torq = norm.torque1d_bands[f"tcm{joint}AA"]
            self.KT_stat_fig2r.add_norm_trace(
                joint=joint, mean=torq["mean"], std=torq["sd"]
            )
            self.KT_stat_fig2l.add_norm_trace(
                joint=joint, mean=torq["mean"], std=torq["sd"]
            )
                
        # # 3nd figure: ground reaction force
        forces_to_plot = KT_Ground_Reaction_Figure.forces_to_plot.keys()
        self.KT_stat_fig3r = KT_Ground_Reaction_Figure()
        self.KT_stat_fig3l = KT_Ground_Reaction_Figure()
        
        # Add nomatives
        for force in forces_to_plot:
            forc = norm.force1d_bands[f"fcm{force}"]
            self.KT_stat_fig3r.add_norm_trace(
                force=force, mean=forc["mean"], std=forc["sd"]
            )
            self.KT_stat_fig3l.add_norm_trace(
                force=force, mean=forc["mean"], std=forc["sd"]
            )
        
        for trial_mdx in self.trial_mdxs: 
             
            angles = trial_mdx.trial.cycle.angles
            forces = trial_mdx.trial.cycle.forces1d
            powers = trial_mdx.trial.cycle.powers
            torques = trial_mdx.trial.cycle.torques1d

            # Adding traces. This will add ALL traces for the cycle
            for joint in joints_to_plot:
                side="r"
                for i in angles[f"ac{side.upper()}{joint}FE"].data[0]:
                    self.KT_stat_fig1r.add_consistency_trace(joint, "angle", side, i)
                if f"tc{side.upper()}{joint}FE" in torques:
                    for i in torques[f"tc{side.upper()}{joint}FE"].data[0]:
                        self.KT_stat_fig1r.add_consistency_trace(joint, "torque", side, i)
                if f"pc{side.upper()}{joint}FE" in powers:
                    for i in powers[f"pc{side.upper()}{joint}FE"].data[0]:
                        self.KT_stat_fig1r.add_consistency_trace(joint, "power", side, i)
                side="l"
                for i in angles[f"ac{side.upper()}{joint}FE"].data[0]:
                    self.KT_stat_fig1l.add_consistency_trace(joint, "angle", side, i)
                if f"tc{side.upper()}{joint}FE" in torques:
                    for i in torques[f"tc{side.upper()}{joint}FE"].data[0]:
                        self.KT_stat_fig1l.add_consistency_trace(joint, "torque", side, i)
                if f"pc{side.upper()}{joint}FE" in powers:
                    for i in powers[f"pc{side.upper()}{joint}FE"].data[0]:
                        self.KT_stat_fig1l.add_consistency_trace(joint, "power", side, i)

            RTO = trial_mdx.trial.cycle.events["enRTO"].data[0][0][0]
            LTO = trial_mdx.trial.cycle.events["enLTO"].data[0][0][0]

            self.KT_stat_fig1r.add_event(RTO, dict(color="green", width=2, dash="longdash"))
            self.KT_stat_fig1l.add_event(LTO, dict(color="red", width=2, dash="longdash"))

            # ------ 2nd figure: frontal plane kinetics


            # Adding traces. This will add ALL traces for the cycle
            for joint in joints_to_plot:
                for side in ["r", "l"]:
                    if f"tc{side.upper()}{joint if joint != 'A' else 'F'}AA" in torques:
                        for i in torques[
                            f"tc{side.upper()}{joint if joint != 'A' else 'F'}AA"
                        ].data[0]:
                            if side == "r":
                                self.KT_stat_fig2r.add_consistency_trace(joint, side, i)
                            if side == "l":
                                self.KT_stat_fig2l.add_consistency_trace(joint, side, i)

            # Adding events
            self.KT_stat_fig2r.add_event(RTO, dict(color="green", width=2, dash="longdash"))
            self.KT_stat_fig2l.add_event(LTO, dict(color="red", width=2, dash="longdash"))

            

            # ------ 3nd figure: ground reaction force


            # Adding traces. This will add ALL traces for the cycle
            for force in forces_to_plot:
                for side in ["r", "l"]:
                    if f"fc{side.upper()}{force}" in forces:
                        for i in forces[f"fc{side.upper()}{force}"].data[0]:
                            if side == "r":
                                self.KT_stat_fig3r.add_consistency_trace(force, side, i)
                            if side == "l":
                                self.KT_stat_fig3l.add_consistency_trace(force, side, i)

            # Adding events
            self.KT_stat_fig3r.add_event(RTO, dict(color="green", width=2, dash="longdash"))
            self.KT_stat_fig3l.add_event(LTO, dict(color="red", width=2, dash="longdash"))

        # Adjusting ranges and annotations
        self.KT_stat_fig1r.adjust_range(auto=False)
        self.KT_stat_fig1r.add_annotations()
        self.KT_stat_fig1l.adjust_range(auto=False)
        self.KT_stat_fig1l.add_annotations()

        self.KT_stat_fig2r.adjust_range(auto=False)
        self.KT_stat_fig2r.add_annotations()
        self.KT_stat_fig2l.adjust_range(auto=False)
        self.KT_stat_fig2l.add_annotations()

        self.KT_stat_fig3r.adjust_range(auto=False)
        self.KT_stat_fig3r.add_annotations()
        self.KT_stat_fig3l.adjust_range(auto=False)
        self.KT_stat_fig3l.add_annotations()

        # update_layout fig 1
        self.KT_stat_fig1r.fig.update_layout(
            margin=go.layout.Margin(l=0, r=0, b=0, t=30),
            height=700,
            width=1000,
            showlegend=False
        )
        self.KT_stat_fig1r.fig.update_layout(
            template="plotly_white", font={"size": self.plotly_size_sm}
        )
        self.KT_stat_fig1l.fig.update_layout(
            margin=go.layout.Margin(l=0, r=0, b=0, t=30),
            height=700,
            width=1000,
            showlegend=False
        )
        self.KT_stat_fig1l.fig.update_layout(
            template="plotly_white", font={"size": self.plotly_size_sm}
        )
        # update_layout fig 2
        # Adjust figure size, margins, and legend location
        self.KT_stat_fig2r.fig.update_layout(
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
                "font": {"size": 24},
            },
        )
        self.KT_stat_fig2r.fig.update_layout(
            template="plotly_white", font={"size": self.plotly_size_sm}
        )
        self.KT_stat_fig2l.fig.update_layout(
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
                "font": {"size": 24},
            },
        )
        self.KT_stat_fig2l.fig.update_layout(
            template="plotly_white", font={"size": self.plotly_size_sm}
        )
        
        # update_layout fig 3
        self.KT_stat_fig3r.fig.update_layout(
            template="plotly_white", font={"size": self.plotly_size_sm}
        )

        # Adjust figure size, margins, and legend location
        self.KT_stat_fig3r.fig.update_layout(
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
                "font": {"size": 24},
            },
        )
        self.KT_stat_fig3l.fig.update_layout(
            template="plotly_white", font={"size": self.plotly_size_sm}
        )

        # Adjust figure size, margins, and legend location
        self.KT_stat_fig3l.fig.update_layout(
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
                "font": {"size": 24},
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

        temp_svgr = tempfile.mkstemp(suffix=".svg", text=True)
        temp_svg2r = tempfile.mkstemp(suffix=".svg", text=True)
        temp_svg3r = tempfile.mkstemp(suffix=".svg", text=True)
        temp_svgl = tempfile.mkstemp(suffix=".svg", text=True)
        temp_svg2l = tempfile.mkstemp(suffix=".svg", text=True)
        temp_svg3l = tempfile.mkstemp(suffix=".svg", text=True)
        # write SVG to tempfile
        self.KT_stat_fig1r.fig.write_image(temp_svgr[1], scale=3, engine=RENDER_ENGINE)
        self.KT_stat_fig2r.fig.write_image(temp_svg2r[1], scale=3, engine=RENDER_ENGINE)
        self.KT_stat_fig3r.fig.write_image(temp_svg3r[1], scale=3, engine=RENDER_ENGINE)
        self.KT_stat_fig1l.fig.write_image(temp_svgl[1], scale=3, engine=RENDER_ENGINE)
        self.KT_stat_fig2l.fig.write_image(temp_svg2l[1], scale=3, engine=RENDER_ENGINE)
        self.KT_stat_fig3l.fig.write_image(temp_svg3l[1], scale=3, engine=RENDER_ENGINE)
        # simple template for the PDF page
        template1 = f"""
        <body>
            <img src='{header.resolve().as_uri()}'>            
            <h1 style='font-size: 19px;'>Estudio de Consistencia: Cinética</h1>
            <table class='table table-striped' style='width:100%;font-size: 14px;'>
                <tr>
                    <th scope='row'>Paciente:</th>
                    <td>{self.lastname}, {self.name}</td>
                </tr>
                <tr>
                    <th scope='row'>ID paciente:</th>
                    <td>{self.patid}</td>
                </tr>
                <tr>
                    <th scope='row'>Fecha de estudio:</th>
                    <td>{self.date}</td>
                </tr>
                <tr>
                    <th scope='row'>Sesión:</th>
                    <td>{self.session_name}</td>
                </tr>
                <tr>
                    <th scope='row'>Archivos seleccionados:</th>
                    <td>{self.ntrial}</td>
                </tr>
            </table>

            <h2 style='font-size: 18px;'>  </h2>
            <h2 style='font-size: 18px;'>Cinética. Plano sagital</h2>
            <img src='{Path(temp_svgr[1]).as_uri()}' class='main_plot'>
            
            {self.footer}
        </body>
        """

        # Weasyprint the template
        page = HTML(string=template1).render(stylesheets=self.styles)
        self.pages.append(page)
        
        template2 = f"""
        <body>
            <img src='{header.resolve().as_uri()}'>    
            <img src='{Path(temp_svgl[1]).as_uri()}' class='main_plot'>
            
            {self.footer}
        </body>
        """

        # Weasyprint the template
        page = HTML(string=template2).render(stylesheets=self.styles)
        self.pages.append(page)


        template3 = f"""
        <body>
            <img src='{header.resolve().as_uri()}'> 
            <h2 style='font-size: 18px;'>Cinética. Plano frontal</h2>
            <img src='{Path(temp_svg2r[1]).as_uri()}' class='main_plot'>            
            <img src='{Path(temp_svg2l[1]).as_uri()}' class='main_plot'>
            
            <h2 style='font-size: 18px;'>Fuerza de reacción del piso</h2>
            <img src='{Path(temp_svg3r[1]).as_uri()}' class='main_plot'>
            <img src='{Path(temp_svg3l[1]).as_uri()}' class='main_plot'>
            {self.footer}
        </body>
        """

        # Weasyprint the template
        page = HTML(string=template3).render(stylesheets=self.styles)
        self.pages.append(page)
        # close ande remove tempfile
        os.close(temp_svgr[0])
        os.unlink(temp_svgr[1])
        os.close(temp_svgl[0])
        os.unlink(temp_svgl[1])

        os.close(temp_svg2r[0])
        os.unlink(temp_svg2r[1])
        os.close(temp_svg2l[0])
        os.unlink(temp_svg2l[1])

        os.close(temp_svg3r[0])
        os.unlink(temp_svg3r[1])
        os.close(temp_svg3l[0])
        os.unlink(temp_svg3l[1])

    def print(self, filename):
        if not self.pages:
            print("Empty PDF, generate pages first.")
            return
        a = self.pages[0].copy(
            [page for document in self.pages for page in document.pages]
        )
        a.write_pdf(filename)
