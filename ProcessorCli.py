import click
import click_pathlib
from pathlib import Path

from Processor import Processor, ValidatedTdf
from simplemdx.declparser import mdxParser

### NUEVO
import sys

def run_score_pipeline(folder_path: Path) -> None:
    """Copia el standing y calcula HJC (SCoRE) sin subprocess."""
    sys.path.append(r"C:\Users\marcha\Desktop\Desarrollos\SCoREFiles")
    from copy_tdf import main as copy_tdf_main
    from score_hjc import main as score_hjc_main

    folder = str(folder_path)
    try:
        copy_tdf_main(folder)
        score_hjc_main(folder)
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e
### fin nuevo


### NUEVO
def get_score_suffix():
    sys.path.append(r"C:\Users\marcha\Desktop\Desarrollos\SCoREFiles")
    from ScoreFiles.score_config import ScoreConfig

    return ScoreConfig().mdx_score_suffix
### fin nuevo



def is_valid_dir(ctx, param, value):
    try:
        a = Path(value)
        if not a.is_dir():
            raise click.BadParameter("Has to be an existing directory")
        return a
    except Exception:
        raise click.BadParameter("Has to be valid directory")


def is_valid_file(file_to_check):
    try:
        a = Path(file_to_check)
        if not a.is_file():
            raise click.BadParameter(f"{file_to_check} doesn't exist")
        return a
    except Exception:
        raise click.BadParameter(f"Invalid path {file_to_check}")


def is_valid_file_list(ctx, param, value):
    if isinstance(value, tuple):
        for i in value:
            is_valid_file(i)
        return tuple(Path(i) for i in value)
    else:
        is_valid_file(value)
        return Path(value)


@click.group("Processor")
@click.pass_context
def cli(ctx):
    """
    Processor CLI
    """
    pass


@cli.command("process_folder")
@click.argument(
    "folder_path",
    type=click_pathlib.Path(exists=True, dir_okay=True),
    callback=is_valid_dir,
    nargs=1,
    required=True,
)
@click.option(
    "-m", "--min-kinematics", default=6, help="Minimum kinematics to be processed"
)

### NUEVO
@click.option(
    "--score/--no-score",
    default=False,
    help="Corre también el protocolo SCORE (HJC) además del protocolo HH estándar",
)
### fin nuevo
def process_folder(folder_path, min_kinematics, score):
    a = Processor()
    a.scan_folder(folder_path)
    a.track_trials()
    a.classify_trials()
    print("classify_trials")
    a.mark_events(min_kinematics)
    print("mark_events")

    ### NUEVO
    if score:
        run_score_pipeline(folder_path)
    a.process_trials_all(run_score=score)
    ### fin nuevo
    a.process_videos(all_trials=True)


@cli.command("process_tdf")
@click.option(
    "-f",
    "--my-folder",
    type=click_pathlib.Path(exists=True, dir_okay=True),
    # callback=is_valid_dir,
    nargs=1,
    required=False,
    help="Path to the My captures folder, needed for SMARTProcessor. If unspecified, I will be deduced from the standing tdf's drive",
)
@click.option(
    "-S",
    "--session-mdx",
    nargs=1,
    required=False,
    callback=is_valid_file,
    type=click.Path(exists=True),
    help="Session MDX file",
)
@click.option(
    "-s",
    "--standing-tdf",
    nargs=1,
    required=False,
    callback=is_valid_file,
    type=click.Path(exists=True),
    help="Standing TDF file",
)
@click.argument(
    "walking_tdfs",
    nargs=-1,
    required=True,
    callback=is_valid_file_list,
    type=click.Path(exists=True),
)
### NUEVO
@click.option(
    "--score/--no-score",
    default=True,
    help="Corre también el protocolo SCORE (HJC) además del protocolo HH estándar",
)
### fin nuevo
def process_tdf(my_folder, session_mdx, standing_tdf, walking_tdfs, score):

    a = Processor()
    if not my_folder:
        if walking_tdfs[0].drive.lower() == "c:":
            a.my_folder = Path(r"C:\My captures")
        else:
            a.my_folder = Path(r"Q:\SMARTData")
    else:
        a.my_folder = my_folder

    a.models_folder = a.my_folder / "_models"

    if standing_tdf:
        a.standing_tdf = ValidatedTdf(standing_tdf)
    else:
        try:
            s = "~".join(walking_tdfs[0].name.split("~")[0:2]) + "~Standing.tdf"
            standing = walking_tdfs[0].with_name(s)
            a.standing_tdf = ValidatedTdf(standing)
        except Exception as e:
            raise click.BadParameter(
                f"Can't find standing for {walking_tdfs[0]}. Error: {e}"
            )

    if session_mdx:
        a.session_mdx = mdxParser(session_mdx)
    else:
        sess = [
            i
            for i in walking_tdfs[0].parent.glob("*.mdx")
            if "Walking" not in i.name and "Standing" not in i.name
        ]
        if len(sess) != 1:
            raise click.BadParameter(
                f"Can't find inambiguous session for {walking_tdfs[0]}, found {sess}"
            )
        a.session_mdx = mdxParser(sess[0])

    a.trial_tdfs = [ValidatedTdf(i) for i in walking_tdfs]

    a.protocol = a.session_mdx.trial.static.protocol

    if a.protocol == "HHMM - VM":
        a.standing_tracking_model = a.models_folder / "HelenHayesVM STD.xmf"
        a.walking_tracking_model = a.models_folder / "HelenHayesVM WLK.xmf"
    elif a.protocol == "HHMM - VP":
        a.standing_tracking_model = a.models_folder / "HelenHayesVP STD.xmf"
        a.walking_tracking_model = a.models_folder / "HelenHayesVP WLK.xmf"
    elif a.protocol == "Helen Hayes MM":
        a.standing_tracking_model = a.models_folder / "HelenHayesMM.xmf"
        a.walking_tracking_model = a.standing_tracking_model
    else:
        raise ValueError(f"Unknown protocol {a.protocol}")

    try:
        a.birthday = a.session_mdx.trial.static.birthday
    except ValueError:
        raise ValueError(
            f"Birthday not found in {a.session_mdx.filename}. It's most likely empty"
        )

    a.age = a.session_mdx.trial.static.age

    if a.age < 1:
        raise ValueError(f"Age is {a.age} years old. Date of birth is probably wrong")

    a.track_trials()
    a.classify_trials()
    a.mark_events(min_trials=None)
    
    ### NUEVO
    if score:
        run_score_pipeline(walking_tdfs[0].parent)

    a.process_trials_all(run_score=score)
    ### fin nuevo
    
    a.process_videos(all_trials=False)
