from pathlib import Path
from Reporter import Report
import click
import click_pathlib
from Consist import Consist_repo
from itertools import groupby


### NUEVO
import sys
def get_score_suffix():
    sys.path.append(r"C:\Users\marcha\Desktop\Desarrollos\SCoREFiles")
    from score_config import ScoreConfig

    return ScoreConfig().mdx_score_suffix
### fin nuevo


def get_CSF_in_dir(ctx, args, incomplete):
    return list(Path(incomplete).glob("*.csf"))


def is_valid_dir(ctx, param, value):
    try:
        a = Path(value)
        if not a.is_dir():
            raise click.BadParameter("Has to be an existing directory")
        return a
    except Exception:
        raise click.BadParameter("Has to be an existing directory")


def is_valid_file(file_to_check):
    try:
        a = Path(file_to_check)
        if not a.is_file():
            raise click.BadParameter(f"{file_to_check} doesn't exist")
    except Exception:
        raise click.BadParameter(f"{file_to_check} doesn't exist")


def are_valid_files(ctx, param, value):

    if isinstance(value, tuple):
        for i in value:
            is_valid_file(i)
    else:
        is_valid_file(value)

    return value


@click.group("Reporter")
@click.option(
    "--envelope/--no-envelope",
    is_flag=True,
    required=False,
    default=False,
    help="Adds an envelope to the EMG page",
)
@click.option(
    "--anonymize/--no-anonymize",
    is_flag=True,
    required=False,
    default=False,
    help="Anonimizes all reports printed",
)
@click.pass_context
def cli(ctx, envelope, anonymize):
    if not ctx.obj:
        ctx.obj = {}
    ctx.obj["ENVELOPE"] = envelope
    ctx.obj["ANONYMIZE"] = anonymize


@cli.command("print_CSF")
@click.argument(
    "csf_files",
    nargs=-1,
    required=True,
    callback=are_valid_files,
    type=click_pathlib.Path(exists=True, file_okay=True),
    # autocompletion=get_CSF_in_dir,
)
@click.argument(
    "output_dir",
    type=click_pathlib.Path(dir_okay=True, exists=True),
    callback=is_valid_dir,
    nargs=1,
    required=True,
)
@click.option(
    "-s",
    "--studies-folder",
    "studies_folder",
    type=click_pathlib.Path(exists=True),
    nargs=1,
    required=False,
    callback=is_valid_dir,
    default=Path("Q:/SMARTData/DataSMARTClinic"),
    help="Folder where the patient data is located. Tipically Data[mydbname]",
)
@click.pass_context
def print_CSF(ctx, csf_files, output_dir, studies_folder):
    for csf_file in csf_files:
        print(type(csf_file))
        mdx_list = []
        print(f"Printing reports from {csf_file}")
        with csf_file.open("r") as f:
            for line in f:
                if ".mdx" in line:
                    mdx_list.append(line.rstrip())

        # Group mdx by conditions
        grouped_condition = [
            list(x[1]) for x in groupby(mdx_list, lambda x: x.rsplit("~")[1])
        ]

        # logging.info("Encontradas " + str(len(grouped_condition)) + " condicion(es)")
        for group in grouped_condition:
            session_code = group[0].rsplit("~")[1]
            pat_id = group[0].rsplit("~")[0]
            dire = [
                i for i in studies_folder.iterdir() if i.name.startswith(pat_id + "~")
            ]
            if len(dire) > 1:
                raise Exception(
                    f"Ambiguous patient code, more than one folder possible: {dire}"
                )
            dire = dire[0]
            subdire = [
                i.resolve()
                for i in dire.iterdir()
                if i.name.startswith(pat_id + "~" + session_code)
            ]
            if len(subdire) > 1:
                raise Exception(
                    f"Ambiguous patient code, more than one folder possible: {subdire}"
                )
            subdire = subdire[0]
            # logging.info(str(subdire[0].rsplit('~')[2]))
            session_mdx = [
                i
                for i in subdire.glob("*.mdx")
                if "Walking" not in i.name and "Standing" not in i.name
            ]
            if len(session_mdx) > 1:
                raise Exception(
                    f"Ambiguous patient session mdx, more than one MDX possible: {session_mdx}"
                )
            session_mdx = session_mdx[0]
            standing_mdx = Path(
                subdire / Path(f"{pat_id}~{session_code}~Standing.mdx")
            )
            trials_mdx = [subdire / Path(i) for i in group]
            # logging.info("\tWalking MDX: " + x)
            # logging.info("\tImprimiendo informe...")
            ctx.invoke(
                print_report,
                session=session_mdx,
                standing=standing_mdx,
                trials=trials_mdx,
                output_dir=output_dir,
            )


@cli.command("print_MDX")
@click.argument("trials", type=click_pathlib.Path(exists=True), nargs=-1, required=True)
@click.argument(
    "output_dir",
    type=click_pathlib.Path(dir_okay=True, exists=True),
    callback=is_valid_dir,
    nargs=1,
    required=True,
)
@click.option(
    "-s",
    "--studies-folder",
    "studies_folder",
    type=click_pathlib.Path(exists=True),
    nargs=1,
    required=False,
    callback=is_valid_dir,
    default=Path("Q:/SMARTData/DataSMARTClinic"),
    help="Folder where the patient data is located. Tipically Data[mydbname]",
)
@click.option(
    "-n",
    "--normative",
    "normative",
    type=click_pathlib.Path(exists=True),
    nargs=1,
    required=False,
    default=None,
    help="Normative ENB for the reports. Chosen from subjet age if not set",
)
@click.option(
    "-t",
    "--type",
    "report_type",
    type=click.Choice(["gait", "ACV", "running"]),
    nargs=1,
    required=False,
    default="gait",
    help="Type of report",
)
@click.pass_context
def print_MDX(ctx, trials, output_dir, studies_folder, normative, report_type):
    grouped_condition = [
        list(x[1]) for x in groupby(trials, lambda x: x.name.rsplit("~")[1])
    ]
    for group in grouped_condition:
        session_code = group[0].name.rsplit("~")[1]
        pat_id = group[0].name.rsplit("~")[0]
        dire = [i for i in studies_folder.iterdir() if i.name.startswith(pat_id + "~")]
        if len(dire) > 1:
            raise Exception(
                f"Ambiguous patient code, more than one folder possible: {dire}"
            )
        dire = dire[0]
        subdire = [
            i.resolve()
            for i in dire.iterdir()
            if i.name.startswith(pat_id + "~" + session_code)
        ]
        if len(subdire) > 1:
            raise Exception(
                f"Ambiguous patient code, more than one folder possible: {subdire}"
            )
        subdire = subdire[0]
        # logging.info(str(subdire[0].rsplit('~')[2]))
        session_mdx = [
            i
            for i in subdire.glob("*.mdx")
            if "Walking" not in i.name and "Standing" not in i.name and "Running" not in i.name
        ]
        if len(session_mdx) > 1:
            raise Exception(
                f"Ambiguous patient session mdx, more than one MDX possible: {session_mdx}"
            )
        session_mdx = session_mdx[0]
        standing_mdx = Path(
            subdire / Path(f"{pat_id}~{session_code}~Standing.mdx")
        )
        trials_mdx = [subdire / Path(i) for i in group]
        # logging.info("\tWalking MDX: " + x)
        # logging.info("\tImprimiendo informe...")
        if report_type == "ACV":
            fun = print_ACV
        elif report_type == "gait":
            fun = print_report
        elif report_type == "running":
            fun = print_running
        else:
            raise ValueError(f"Invalid report type: {report_type}")

        ctx.invoke(
            fun,
            session=session_mdx,
            standing=standing_mdx,
            trials=trials_mdx,
            output_dir=output_dir,
            normative=normative,
        )


@cli.command("print_report")
@click.option(
    "-s",
    "standing",
    "--standing",
    multiple=False,
    type=click_pathlib.Path(exists=True),
    default=None,
    help="Standing MDX. Optional for kinematic static angles.",
)
@click.argument("session", type=click_pathlib.Path(exists=True), nargs=1, required=True)
@click.option(
    "-n",
    "--normative",
    "normative",
    type=click_pathlib.Path(exists=True),
    nargs=1,
    required=False,
    default=None,
    help="Normative ENB for the reports. Chosen from subjet age if not set",
)
@click.argument("trials", type=click_pathlib.Path(exists=True), nargs=-1, required=True)
@click.argument(
    "output_dir",
    type=click_pathlib.Path(dir_okay=True, exists=True),
    callback=is_valid_dir,
    nargs=1,
    required=True,
)
@click.pass_context
def print_report(ctx, session, standing, trials, normative, output_dir):
    ### NUEVO
    score_suffix = get_score_suffix()
    ### fin nuevo
    for i in trials: 
        ### NUEVO
        if score_suffix in i.name:
            continue
        ### fin nuevo

        print(f"Building report for {i.name}")

        ### NUEVO
        # Buscar MDX del protocolo SCORE si existe
        score_mdx = i.with_name(i.stem + score_suffix + ".mdx")
        has_score = score_mdx.exists()
        if has_score:
            print(f"   SCORE MDX encontrado: {score_mdx.name}")
        else:
            print("   SCORE MDX no encontrado — solo se graficara HH")
        ### fin nuevo

        a = Report(
            trial=i,
            # traial_score=i.name + "-score.mdx",
            session=session,
            normative=normative,
            standing=standing,
            anonymize=ctx.obj["ANONYMIZE"],
        )
        ### NUEVO
        from simplemdx.declparser import mdxParser

        a.score_mdx = mdxParser(score_mdx) if has_score else None
        ### fin nuevo
        print("Header...")
        a.KM_static_header()
        a.KM_static_table()
        a.KM_static_report(
            autorange=False,
            mark_out_of_range=False,
            plot_sides={
                "L": {"color": "red", "width": 2},
                "R": {"color": "green", "width": 2},
                ### NUEVO
                "SCORE_L": {"color": "red", "width": 2, "dash": "dash"},
                "SCORE_R": {"color": "green", "width": 2, "dash": "dash"},
                ### fin nuevo
            },
            plot_normative=True,
            normative_options={},
            legend_options={
                "y": -0.04,
            },
            plot_events={
                "LTO": {
                    "color": "red",
                    "width": 1,
                    "position": "top",
                    "length": 1,
                    "legend_text": "Toe off izquierdo",
                    "dash": "longdash",
                },
                "RTO": {
                    "color": "Green",
                    "width": 1,
                    "position": "top",
                    "length": 1,
                    "legend_text": "Toe off derecho",
                    "dash": "longdash",
                },
            },
        )
        a.KM_static_report(
            autorange=True,
            mark_out_of_range=True,
            out_of_range_opts={
                "color": "orange",
                "width": 2,
                "legend_text": "Fuera de escala",
            },
            plot_sides={
                "L": {"color": "red", "width": 2},
                "R": {"color": "green", "width": 2},
                ### NUEVO
                "SCORE_L": {"color": "red", "width": 1, "dash": "dash"},
                "SCORE_R": {"color": "green", "width": 1, "dash": "dash"},
                ### fin nuevo
            },
            plot_normative=True,
            normative_options={},
            legend_options={
                "y": -0.04,
            },
            plot_events={
                "LTO": {
                    "color": "red",
                    "width": 1,
                    "position": "top",
                    "length": 1,
                    "legend_text": "Toe off izquierdo",
                    "dash": "longdash",
                },
                "RTO": {
                    "color": "green",
                    "width": 1,
                    "position": "top",
                    "length": 1,
                    "legend_text": "Toe off derecho",
                    "dash": "longdash",
                },
            },
        )
        a.KT_static_report()
        print("Muscles...")
        a.MUS_static_report()
        print("EMG...1")
        a.EMG_static_report(color_from_envelope=True, plot_envelope=ctx.obj["ENVELOPE"])
        # a.EMG_static_report(color_from_envelope=False)
        a.SYNC_static_report(side="R")
        a.SYNC_static_report(side="L")
        name = f"{a.session_mdx.trial.label}~{a.ntrial}.pdf"
        if ctx.obj["ANONYMIZE"]:
            name = f"{a.session_mdx.trial.label}~{a.ntrial}_anon_.pdf"
        out_name = Path(output_dir / name).resolve()
        a.print(out_name)


@cli.command("print_ACV")
@click.argument("session", type=click_pathlib.Path(exists=True), nargs=1, required=True)
@click.option(
    "-n",
    "--normative",
    "normative",
    type=click_pathlib.Path(exists=True),
    nargs=1,
    required=False,
    default=None,
    help="Normative ENB for the reports. Chosen from subjet age if not set",
)
@click.argument("trials", type=click_pathlib.Path(exists=True), nargs=-1, required=True)
@click.argument(
    "output_dir",
    type=click_pathlib.Path(dir_okay=True, exists=True),
    callback=is_valid_dir,
    nargs=1,
    required=True,
)
@click.option(
    "-s",
    "standing",
    "--standing",
    multiple=False,
    type=click_pathlib.Path(exists=True),
    default=None,
    help="Standing MDX. Optional for kinematic static angles.",
)
@click.pass_context
def print_ACV(ctx, session, trials, normative, standing, output_dir):
    for i in trials:
        print(f"Building ACV report for {i.name}")
        a = Report(trial=i, session=session, normative=normative)
        print("Header...")
        a.KM_static_header(title="Reporte ACV")
        a.KM_static_table()
        a.ACV_lissajous()
        # a.ACV_abnormal_gait()
        # a.KM_static_report(
        #     autorange=False, mark_out_of_range=False, mark_static_angles=False
        # )
        a.KM_static_report(autorange=True, mark_out_of_range=True)
        # a.KT_static_report()
        print("Muscles...")
        a.MUS_static_report()
        # print("EMG...")
        a.EMG_static_report(color_from_envelope=True, plot_envelope=ctx.obj["ENVELOPE"])
        # a.EMG_static_report(color_from_envelope=False)
        # a.SYNC_static_report(side="R")
        # a.SYNC_static_report(side="L")
        name = f"{a.session_mdx.trial.label}~{a.ntrial}_ACV_.pdf"
        out_name = Path(output_dir / name).resolve()
        a.print(out_name)


@cli.command("print_running")
@click.argument("session", type=click_pathlib.Path(exists=True), nargs=1, required=True)
@click.option(
    "-n",
    "--normative",
    "normative",
    type=click_pathlib.Path(exists=True),
    nargs=1,
    required=False,
    default=None,
    help="Normative ENB for the reports. Chosen from subjet age if not set",
)
@click.argument("trials", type=click_pathlib.Path(exists=True), nargs=-1, required=True)
@click.argument(
    "output_dir",
    type=click_pathlib.Path(dir_okay=True, exists=True),
    callback=is_valid_dir,
    nargs=1,
    required=True,
)
@click.option(
    "-s",
    "standing",
    "--standing",
    multiple=False,
    type=click_pathlib.Path(exists=True),
    default=None,
    help="Standing MDX. Optional for kinematic static angles.",
)
@click.pass_context
def print_running(ctx, session, trials, normative, standing, output_dir):
    for i in trials:
        print(f"Building running report for {i.name}")
        normative = r"Q:\SMARTData\_normatives\en\Running-Normal-adults-"
        a = Report(trial=i, session=session, normative=normative)
        # print("Header...")
        # a.RUN_header(title="Análisis de carrera")
        # a.KM_static_table_running()
        
        ## a.KM_static_table()
        # a.KM_static_report(
        #     autorange=True,
        #     mark_out_of_range=False,
        #     title="Cinemática izquierda",
        #     plot_sides={
        #         "L": {"color": "red", "width": 0.15},
        #     },
        #     plot_normative=True,
        #     normative_options={},
        #     legend_options={
        #         "y": -0.04,
        #     },
        #     plot_events={
        #         "LTO": {
        #             "color": "red",
        #             "width": 0.5,
        #             "position": "top",
        #             "length": 0.2,
        #             "legend_text": "Toe off",
        #         },
        #         "RHSLCycle": {
        #             "color": "green",
        #             "width": 0.5,
        #             "position": "bottom",
        #             "length": 0.2,
        #             "dash": "dot",
        #             "legend_text": "Heel strike (derecho)",
        #         },
        #         "RTOLCycle": {
        #             "color": "green",
        #             "width": 0.5,
        #             "position": "bottom",
        #             "length": 0.2,
        #             "legend_text": "Toe off (derecho)",
        #         },
        #     },
        # )
        # a.KM_static_report(
        #     autorange=True,
        #     mark_out_of_range=False,
        #     title="Cinemática derecha",
        #     plot_sides={
        #         "R": {"color": "green", "width": 0.15},
        #     },
        #     plot_normative=True,
        #     normative_options={},
        #     legend_options={
        #         "y": -0.04,
        #     },
        #     plot_events={
        #         "RTO": {
        #             "color": "green",
        #             "width": 0.5,
        #             "position": "top",
        #             "length": 0.2,
        #             "legend_text": "Toe off",
        #         },
        #         "LHSRCycle": {
        #             "color": "red",
        #             "width": 0.5,
        #             "position": "bottom",
        #             "length": 0.2,
        #             "dash": "dot",
        #             "legend_text": "Heel strike (izquierdo)",
        #         },
        #         "LTORCycle": {
        #             "color": "red",
        #             "width": 0.5,
        #             "position": "bottom",
        #             "length": 0.2,
        #             "legend_text": "Toe off (izquierdo)",
        #         },
        #     },
        # )
        
        # Cinematica bilateral con envolvente:
        a.KM_static_report(
            autorange=True,
            mark_out_of_range=False,
            title="Cinemática bilateral",
            plot_sides={
                "R": {"mean": True, "color": "green", "width": 0.5},
                "L": {"mean": True, "color": "red", "width": 0.5},
            },
            plot_normative=True,
            normative_options={},
            legend_options={
                "y": 0.04,
            },
            plot_mean_events={
                "RTO": {
                    "color": "green",
                    "width": 0.3,
                    "position": "top",
                    "length": 0.2,
                    "legend_text": "Toe off (derecho)",
                },
                "LTO": {
                    "color": "red",
                    "width": 0.3,
                    "position": "top",
                    "length": 0.2,
                    "legend_text": "Toe off (izquierdo)",
                },
            },
            legend_orientation = "h",
            running =True
        )

        # a.KM_static_report(
        #     autorange=True,
        #     mark_out_of_range=False,
        #     sides=["R"],
        #     title="Cinemática derecha",
        #     line_opts={"width": 1, "color": None},
        #     plot_events={
        #         "RTO": {
        #             "color": "green",
        #             "width": 0.5,
        #             "position": "top",
        #             "length": 0.2,
        #         },
        #     },
        # )
        # a.KM_static_report(
        #     autorange=True,
        #     mark_out_of_range=False,
        #     use_mean=True,
        #     title="Cinemática bilateral (media y desvío estándar)",
        #     use_mean_events=True,
        #     line_opts={"width": 1},
        #     plot_mean_events={
        #         "RTO": {
        #             "color": "green",
        #             "width": 0.5,
        #             "position": "top",
        #             "length": 0.2,
        #         },
        #         "LTO": {
        #             "color": "red",
        #             "width": 0.5,
        #             "position": "bottom",
        #             "length": 0.2,
        #         },
        #     },
        # )
        
        # a.EMG_static_report(
        #     color_from_envelope=True,
        #     plot_envelope=True,
        #     plot_data=False,
        #     emg_labels=[
        #         "Gluteus medius",
        #         "Vastus medialis",
        #         "Biceps femoris caput longus",
        #         "Semitendinosus",
        #         "Soleus",
        #     ],
        # )
        
        # EMG:
        a.EMG_static_report(
            color_from_envelope=True,
            plot_envelope=True,
            # plot_data=False,
            emg_labels=[
                "Gluteus maximus",
                "Gluteus medius",
                "Vastus medialis",
                "Biceps femoris caput longus",
                "Semitendinosus",
                "Soleus",
            ],
        )
        # a.EMG_static_report(color_from_envelope=False)
        # sincronizacion con plataformas
        # a.SYNC_static_report(side="R")
        # a.SYNC_static_report(side="L")

        name = f"{a.session_mdx.trial.label}~{a.ntrial}_RUNNING.pdf"
        out_name = Path(output_dir / name).resolve()
        a.print(out_name)


@cli.command("print_kinetic_consistency")
@click.argument(
    "csf_files",
    nargs=-1,
    required=True,
    callback=are_valid_files,
    type=click_pathlib.Path(exists=True, file_okay=True),
    # autocompletion=get_CSF_in_dir,
)
@click.argument(
    "output_dir",
    type=click_pathlib.Path(dir_okay=True, exists=True),
    callback=is_valid_dir,
    nargs=1,
    required=True,
)
@click.option(
    "-s",
    "--studies-folder",
    "studies_folder",
    type=click_pathlib.Path(exists=True),
    nargs=1,
    required=False,
    callback=is_valid_dir,
    default=Path("Q:/SMARTData/DataSMARTClinic"),
    help="Folder where the patient data is located. Tipically Data[mydbname]",
)

@click.pass_context
def print_kinetic_consistency(ctx, csf_files, output_dir, studies_folder):
    for csf_file in csf_files:
        mdx_list = []
        print(f"Printing reports from {csf_file}")
        with csf_file.open("r") as f:
            for line in f:
                if ".mdx" in line:
                    mdx_list.append(line.rstrip())

        # Group mdx by conditions
        grouped_condition = [
            list(x[1]) for x in groupby(mdx_list, lambda x: x.rsplit("~")[1])
        ]
        
        # logging.info("Encontradas " + str(len(grouped_condition)) + " condicion(es)")
        for group in grouped_condition:
            session_code = group[0].rsplit("~")[1]
            pat_id = group[0].rsplit("~")[0]
            dire = [
                i for i in studies_folder.iterdir() if i.name.startswith(pat_id + "~")
            ]
            if len(dire) > 1:
                raise Exception(
                    f"Ambiguous patient code, more than one folder possible: {dire}"
                )
            dire = dire[0]
            subdire = [
                i.resolve()
                for i in dire.iterdir()
                if i.name.startswith(pat_id + "~" + session_code)
            ]
            if len(subdire) > 1:
                raise Exception(
                    f"Ambiguous patient code, more than one folder possible: {subdire}"
                )
            subdire = subdire[0]
            # logging.info(str(subdire[0].rsplit('~')[2]))
            session_mdx = [
                i
                for i in subdire.glob("*.mdx")
                if "Walking" not in i.name and "Standing" not in i.name
            ]
            if len(session_mdx) > 1:
                raise Exception(
                    f"Ambiguous patient session mdx, more than one MDX possible: {session_mdx}"
                )
            session_mdx = session_mdx[0]
            standing_mdx = Path(
                subdire / Path(f"{pat_id}~{session_code}~Standing.mdx")
            )
            trials_mdx = [subdire / Path(i) for i in group]
            
            print(f"Building kinetic consistency report for {pat_id}~{session_code}")
            a = Consist_repo(trials=trials_mdx, session=session_mdx)
            
            a.KT_static_report(
                autorange=True,
                mark_out_of_range=False,
            )
            
            name = f"{a.session_mdx.trial.label}~Consist_CINETICA.pdf"
            out_name = Path(output_dir / name).resolve()
            a.print(out_name)
        
if __name__ == "__main__":
    cli(obj={})
