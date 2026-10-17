
import logging
import os
import subprocess
import sys
from itertools import chain, zip_longest
from pathlib import Path
from typing import List

import colorlog
from basictdf import Tdf
from simplemdx.declparser import mdxParser

### NUEVO
sys.path.append(r"C:\Users\marcha\Desktop\Desarrollos\SCoREFiles")
from score_config import ScoreConfig
### fin nuevo

PROTOCOL_EPX_FILES = {
    "HHMM - VM": {
        "manual_events": "Manual_events_VMALL.epx",
        "full_no_events": "FULL_noevents_VMALL.epx",
        ### NUEVO
        "full_no_events_score": "FULL_noevents_VMALL_SCORE.epx",
        ### fin nuevo
    },
    "HHMM - VP": {
        "manual_events": "Manual_events_VMALL_VP.epx",
        "full_no_events": "FULL_noevents_VM_VP.epx",
    },
}

def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    format = "%(asctime)s - %(levelname)-8s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    if "colorlog" in sys.modules and os.isatty(2):
        cformat = "%(log_color)s" + format
        f = colorlog.ColoredFormatter(
            cformat,
            date_format,
            log_colors={
                "DEBUG": "reset",
                "INFO": "green",
                "WARNING": "bold_yellow",
                "ERROR": "bold_red",
                "CRITICAL": "bold_red",
            },
        )
    else:
        f = logging.Formatter(format, date_format)
    ch = logging.StreamHandler()
    ch.setFormatter(f)
    root.addHandler(ch)


setup_logging()

log = logging.getLogger(__name__)


class ValidatedTdf(Tdf):
    """Add events validation for basictdf Tdf class"""

    def has_valid_lateral_events(self, side, single_cycle=False):
        if not self.has_events:
            log.warning(f"{self.filePath.name} has no events")
            return False

        if side.lower() == "right":
            labels = ("eRHS", "eRTO")
        elif side.lower() == "left":
            labels = ("eLHS", "eLTO")
        else:
            raise ValueError(f"{side} is not a valid side")

        with self:
            if not all(i in self.events for i in labels):
                log.warning(f"{self.filePath.name} has {side} events missing")
                return False

            heel_strikes = sorted(self.events[labels[0]].values)
            toe_offs = sorted(self.events[labels[1]].values)

        if single_cycle:
            if len(heel_strikes) != 2:
                log.warning(
                    f"{self.filePath.name} has {len(heel_strikes)} {side} heel strikes, expected 2"
                )
                return False
            if len(toe_offs) != 1:
                log.warning(
                    f"{self.filePath.name} has {len(toe_offs)} {side} toe offs, expected 1"
                )
                return False
            if not heel_strikes[0] < toe_offs[0] < heel_strikes[1]:
                log.warning(
                    f"{self.filePath.name}'s {side} toe off isn't between heel strikes"
                )
                return False
        else:
            if len(heel_strikes) < 2:
                log.warning(
                    f"{self.filePath.name} has {len(heel_strikes)} {side} heel strikes, expected at least 2"
                )
                return False
            if len(toe_offs) < 1:
                log.warning(
                    f"{self.filePath.name} has {len(toe_offs)} {side} toe offs, expected at least 1"
                )
                return False

            if heel_strikes[0] < toe_offs[0]:
                l = zip_longest(heel_strikes, toe_offs)
            elif heel_strikes[0] > toe_offs[0]:
                l = zip_longest(toe_offs, heel_strikes)

            l = list(filter(None, chain.from_iterable(l)))
            if sorted(l) != l:
                log.warning(
                    f"{self.filePath.name}'s {side} toe offs aren't between heel strikes"
                )
                return False

        return True

    @property
    def has_valid_right_events(self):
        return self.has_valid_lateral_events("right", single_cycle=True)

    @property
    def has_valid_left_events(self):
        return self.has_valid_lateral_events("left", single_cycle=True)

    @property
    def has_valid_events(self):
        return self.has_valid_right_events and self.has_valid_left_events


class Processor:
    def __init__(
        self,
        tracker_path=r"C:\Program Files (x86)\BTS Bioengineering\SMART 1.10\SMARTtracker.exe",
        processor_path=r"C:\Program Files (x86)\BTS Bioengineering\SMART 1.10\SMARTprocessor.exe",
    ):

        if not (tracker_path := Path(tracker_path)).exists():
            raise ValueError(f"Cannot find SMARTtracklab.exe at {tracker_path}")

        self.tracker_path = tracker_path

        if not (processor_path := Path(processor_path)).exists():
            raise ValueError(f"Cannot find SMARTprocessor.exe at {processor_path}")

        self.processor_path = processor_path

    def get_standing_tdf(self, path: Path) -> Tdf:
        """
        Returns a valid parsed Tdf file.
        If input is a folder, it will return the first tdf file in said folder that
        contains the word "Standing" in its name.



        Args:
            path (Path): input Path. Can be a folder or a file.

        Raises:
            ValueError: - If path is invalid
                        - If multiple standings are found
                        - If no standing at all is found
                        - Input tdf name doesn't correspond to a "id~ss~Walking nn.tdf" pattern

        Returns:
            Tdf: a Tdf object
        """
        if not path.exists():
            raise ValueError(f"Cannot find {path}")

        if path.is_dir():
            standing_tdf = [i for i in path.glob("*.tdf") if "Standing" in i.name]
            if len(standing_tdf) > 1:
                raise ValueError(
                    f"Found {len(standing_tdf)} standing tdfs in {path}. There should only be one"
                )
            elif len(standing_tdf) == 0:
                raise ValueError(f"Cannot find standing tdf in {path}")
            return Tdf(standing_tdf[0])

        try:
            id, session_code, type = path.stem.split("~")
        except ValueError:
            raise ValueError(
                f"File name {path.name} is not in the correct format (e.g.: 1234~aa~Walking.tdf"
            )
        standing_tdf = path.parent / f"{id}~{session_code}~Standing.tdf"

        if not standing_tdf.exists():
            raise ValueError(
                f"Cannot find a valid standing tdf for {path} in {path.parent}"
            )

        return Tdf(standing_tdf)

    def get_all_trial_tdfs(self, folder: Path) -> List[ValidatedTdf]:
        if not folder.is_dir():
            raise ValueError(f"{folder} is not a directory")
        trials = [ValidatedTdf(i) for i in folder.glob("*.tdf") if "Walking" in i.name]
        log.info(f"Found {len(trials)} trial tdfs in {folder}")
        return trials

    def get_all_videos(self, folder: Path) -> List[Path]:
        if not folder.is_dir():
            raise ValueError(f"{folder} is not a directory")
        videos = list(folder.glob("*_tvc1.mjpeg")) + list(folder.glob("*_tvc2.mjpeg"))
        log.info(f"Found {len(videos)} videos in {folder}")
        return sorted(videos)

    def get_session_mdx(self, path: Path) -> Path:

        if path.is_dir():
            session_mdx = [
                i
                for i in path.glob("*.mdx")
                if "Walking" not in i.name and "Standing" not in i.name
            ]
        else:
            try:
                id, session_code, type = path.stem.split("~")
            except ValueError:
                raise ValueError(
                    f"File name {path.name} is not in the correct format (e.g.: 1234~aa~Walking.tdf"
                )
            session_mdx = [
                i
                for i in path.parent.glob("*.mdx")
                if "Walking" not in i.name
                and "Standing" not in i.name
                and i.stem.startswith(f"{id}~{session_code}")
            ]

        if len(session_mdx) > 1:
            raise ValueError(
                f"Found {len(session_mdx)} session mdx in {path}->{[i.name for i in session_mdx]}. There should only be one"
            )
        elif len(session_mdx) == 0:
            raise ValueError(f"Cannot find session mdx in {path}")

        log.info(f"Found session mdx {session_mdx[0].name}")
        return mdxParser(session_mdx[0])

    def scan_folder(self, folder_path: Path):
        log.info(f"Processing folder {folder_path}")

        if not (folder_path := Path(folder_path)).exists():
            raise ValueError(f"Cannot find folder {folder_path}")

        if folder_path.drive.lower() == "c:":
            self.my_folder = Path(r"C:\My captures")
        elif folder_path.drive.lower() == r"\\e-bts-station\c": 
            self.my_folder = Path(r"\\e-bts-station\C\My captures")
        else:
            self.my_folder = Path(r"Q:\SMARTData")

        self.models_folder = self.my_folder / "_models"
        if not self.models_folder.exists():
            raise ValueError(f"Cannot find models folder {self.models_folder}")

        self.session_mdx = self.get_session_mdx(folder_path)
        self.protocol = self.session_mdx.trial.static.protocol
        if self.protocol == "HHMM - VM":
            self.standing_tracking_model = self.models_folder / "HelenHayesVM STD.xmf"
            self.walking_tracking_model = self.models_folder / "HelenHayesVM WLK.xmf"
        elif self.protocol == "HHMM - VP":
            self.standing_tracking_model = self.models_folder / "HelenHayesVP STD.xmf"
            self.walking_tracking_model = self.models_folder / "HelenHayesVP WLK.xmf"
        elif self.protocol == "Helen Hayes MM":
            self.standing_tracking_model = self.models_folder / "HelenHayesMM.xmf"
            self.walking_tracking_model = self.standing_tracking_model
        else:
            raise ValueError(f"Unknown protocol {self.protocol}")

        try:
            self.birthday = self.session_mdx.trial.static.birthday
        except ValueError:
            raise ValueError(
                f"Birthday not found in {self.session_mdx.filename}. It's most likely empty"
            )

        self.age = self.session_mdx.trial.static.age

        if self.age < 1:
            raise ValueError(f"Age is {self.age}. Date of birth is probably wrong")

        self.standing_tdf = self.get_standing_tdf(folder_path)
        self.trial_tdfs = self.get_all_trial_tdfs(folder_path)
        self.videos = self.get_all_videos(folder_path)

        if not self.trial_tdfs:
            log.info(f"No trial tdfs found in {folder_path}. Nothing to do")
            return

    def track_trials(self):
        log.info(f"Tracking {len(self.trial_tdfs)} trials...")

        ready_trials = []

        # First, make sure that the standing is tracked
        while not self.standing_tdf.has_data3D:
            log.info(f"Tracking standing {self.standing_tdf.filePath.name}...")
            self.track(self.standing_tdf, self.standing_tracking_model)

        # Then, track the trials
        total_tracks = len(self.trial_tdfs)

        omit_all = False

        for ntrial, trial in enumerate(self.trial_tdfs):
            if trial.has_data3D:
                log.info(f"Trial {trial.filePath.name} already has 3D data. Adding...")
                ready_trials.append(trial)
                continue

            if omit_all:
                log.info(f"Skipping untracked trial {trial.filePath.name}...")
                continue

            log.info(
                f"Tracking trial [{ntrial+1}/{total_tracks}] {trial.filePath.name}"
            )
            self.track(trial, self.walking_tracking_model)

            if trial.has_data3D:
                ready_trials.append(trial)
            else:
                while not trial.has_data3D:
                    answer = input(
                        f"Trial {trial.filePath.name} has no 3D data yet. Try again? ([Y]es, [O]mmit, O[m]mit all, [A]bort): "
                    ).lower()
                    if answer == "y":
                        self.track(trial, self.walking_tracking_model)
                    elif answer == "o":
                        break
                    elif answer == "m":
                        omit_all = True
                        break
                    elif answer == "a":
                        log.info("Exiting...")
                        exit(0)
                ready_trials.append(trial) if trial.has_data3D else None

        log.info(f"Done. {len(ready_trials)} tracked trials ready to be processed")
        self.trial_tdfs = ready_trials

    def track(self, tdf, model):
        subprocess.call([self.tracker_path, tdf.filePath, f"\\M:{model}", "\\T", "\\O"])

    def classify_trials(self):
        log.info("Classifying tracked trials...")
        self.kinematic_trials = []
        self.right_kinetic_trials = []
        self.left_kinetic_trials = []
        self.full_kinetic_trials = []

        for trial in self.trial_tdfs:
            if trial.has_force_and_torque:
                with trial as t:
                    force_labels = [i.label for i in t.force_and_torque]
                if "r gr" in force_labels and not "l gr" in force_labels:
                    self.right_kinetic_trials.append(trial)
                elif "l gr" in force_labels and not "r gr" in force_labels:
                    self.left_kinetic_trials.append(trial)
                elif "r gr" in force_labels and "l gr" in force_labels:
                    self.full_kinetic_trials.append(trial)
                else:
                    self.kinematic_trials.append(trial)
            else:
                self.kinematic_trials.append(trial)

        log.info(f"Found {len(self.full_kinetic_trials)} full kinetic trials")
        log.info(f"Found {len(self.right_kinetic_trials)} right kinetic trials")
        log.info(f"Found {len(self.left_kinetic_trials)} left kinetic trials")
        log.info(f"Found {len(self.kinematic_trials)} kinematic trials")

    @property
    def _GDI_table(self):
        if self.age < 3:
            while True:
                try:
                    answ = input(
                        "There is no GItable for age < 3. Force the use of the 3-12 years old GItable? (Y/N): "
                    ).lower()
                    assert answ in ["y", "n"]
                    if answ.lower() == "y":
                        return r"_tables\en\GI (03-12 yrs).mdx"
                    elif answ.lower() == "n":
                        exit(1)
                    break
                except AssertionError:
                    continue
        elif 3 <= self.age < 13:
            return r"_tables\en\GI (03-12 yrs).mdx"
        elif 13 <= self.age < 55:
            return r"_tables\en\GI (13-54 yrs).mdx"
        elif 55 <= self.age < 65:
            return r"_tables\en\GI (55-64 yrs).mdx"
        elif self.age >= 65:
            return r"_tables\en\GI (over 65 yrs).mdx"

    def processor(self, trial_list, mode, GDI=False):
        args = [self.processor_path, self.my_folder]
        args += [
            str(j.filePath.relative_to(self.my_folder)) + "#M#Walking"
            for j in trial_list
        ]
        # copy file 
        args += [
            str(self.standing_tdf.filePath.relative_to(self.my_folder)) + "#S#Standing"
        ]
        args += [str(self.session_mdx.filename.relative_to(self.my_folder)) + "#S#Info"]

        if GDI:
            args += [self._GDI_table + "#S#Table"]

        if mode == "auto_events":
            args += [r"\P:Auto_events.epx"]
        else:
            try:
                args += [r"\P:" + PROTOCOL_EPX_FILES[self.protocol][mode]]
            except KeyError:
                raise ValueError(
                    f"Mode {mode} not supported for protocol {self.protocol}"
                )

        args = [f'"{i}"' for i in args]
        error = subprocess.call(f"{args[0]} {''.join(args[1:])}" + " \\O")
        print(error)
        
    def mark_events(self, min_trials=6) -> None:

        self.final_trials = []

        def missing_platform_events():
            return (
                [t for t in self.full_kinetic_trials if not t.has_valid_events]
                + [t for t in self.right_kinetic_trials if not t.has_valid_right_events]
                + [t for t in self.left_kinetic_trials if not t.has_valid_left_events]
            )

        # Mark all automatic events = all full kinematic trials + all right kinetic trials + all left kinetic trials

        first_try = True

        while any(missing_events := missing_platform_events()):
            if not first_try:
                for trial in missing_events:
                    answer = input(
                        f"Automatic events still missing from {trial.filePath.name}. ([R]etry,r[e]track,re[m]ove,[a]bort): "
                    ).lower()
                    if answer == "e":
                        self.track(trial, self.walking_tracking_model)
                    elif answer == "e":
                        continue
                    elif answer == "m":
                        self.trial_tdfs.remove(trial)
                    elif answer == "a":
                        exit(0)
                self.classify_trials()
                self.mark_events(min_trials=min_trials)
                return
            self.processor(missing_events, "auto_events")
            first_try = False

        self.final_trials += self.full_kinetic_trials

        # Mark manually the contralateral sides of the mixed trials

        def missing_contralateral_manual_events():
            return [
                t for t in self.right_kinetic_trials if not t.has_valid_left_events
            ] + [t for t in self.left_kinetic_trials if not t.has_valid_right_events]

        first_try = True
        while any(missing_events := missing_contralateral_manual_events()):
            if not first_try:
                for trial in missing_events:
                    answer = input(
                        f"Contralateral manual events still missing from {trial.filePath.name}. ([R]etry,r[e]track,re[m]ove,[a]bort): "
                    ).lower()
                    if answer == "e":
                        self.track(trial, self.walking_tracking_model)
                    elif answer == "e":
                        continue
                    elif answer == "m":
                        self.trial_tdfs.remove(trial)
                    elif answer == "a":
                        exit(0)
                self.classify_trials()
                self.mark_events(min_trials=min_trials)
                return
            self.processor(missing_events, "manual_events")
            first_try = False

        self.final_trials += self.right_kinetic_trials + self.left_kinetic_trials

        # Add kinematic trials if needed

        append_kinematic_trials = True

        if min_trials:
            if len(self.final_trials) < min_trials:
                log.info(
                    f"Cant't get to {min_trials} trials with just {len(self.final_trials)} kinetic trials. Adding all remaining kinematic trials..."
                )
            else:
                log.info(f"Got to {min_trials} trials with just kinetic trials")
                append_kinematic_trials = False

        def missing_kinematic_events():
            return (
                [t for t in self.kinematic_trials if not t.has_valid_events]
                if append_kinematic_trials
                else []
            )

        first_try = True
        while any(missing_events := missing_kinematic_events()):
            if not first_try:
                for trial in missing_events:
                    answer = input(
                        f"Manual events still missing from {trial.filePath.name}. ([R]etry,r[e]track,re[m]ove,[a]bort): "
                    ).lower()
                    if answer == "e":
                        self.track(trial, self.walking_tracking_model)
                    elif answer == "e":
                        continue
                    elif answer == "m":
                        self.trial_tdfs.remove(trial)
                    elif answer == "a":
                        exit(0)
                self.classify_trials()
                self.mark_events(min_trials=min_trials)
                return
            self.processor(missing_events, "manual_events")
            first_try = False

        self.final_trials += self.kinematic_trials

    def process_trials(self):
        # filtrar nombre sin score
        self.processor(self.final_trials, "full_no_events", GDI=True)
        # nombbre con score
        # self.processor(self.final_trials, "full_no_events_score", GDI=True)

    ### NUEVO
    def process_trials_score(self):
        score_cfg = ScoreConfig()
        nombre_copia = score_cfg.standing_copia
        score_suffix = score_cfg.mdx_score_suffix

        # Usar la copia del standing con los HJC
        original_standing = self.standing_tdf
        score_standing = self.standing_tdf.filePath.parent / self.standing_tdf.filePath.name.replace(
            "Standing", nombre_copia
        )
        self.standing_tdf = ValidatedTdf(score_standing)

        # 1. Renombrar todos los MDX de HH a _HH_temp
        for trial in self.final_trials:
            mdx = trial.filePath.with_suffix(".mdx")
            if mdx.exists():
                mdx_temp = mdx.with_name(mdx.stem + "_HH_temp.mdx")
                if mdx_temp.exists():
                    mdx_temp.unlink()
                mdx.rename(mdx_temp)

        # 2. Correr protocolo SCORE
        self.processor(self.final_trials, "full_no_events_score", GDI=True)

        # 3. Renombrar MDX de SCORE al nombre con sufijo (reemplaza si ya existe)
        for trial in self.final_trials:
            mdx = trial.filePath.with_suffix(".mdx")
            if mdx.exists():
                mdx_score = mdx.with_name(mdx.stem + score_suffix + ".mdx")
                if mdx_score.exists():
                    mdx_score.unlink()
                mdx.rename(mdx_score)

        # 4. Restaurar MDX de HH al nombre original
        for trial in self.final_trials:
            mdx = trial.filePath.with_suffix(".mdx")
            mdx_temp = mdx.with_name(mdx.stem + "_HH_temp.mdx")
            if mdx_temp.exists():
                if mdx.exists():
                    mdx.unlink()
                mdx_temp.rename(mdx)

        # Restaurar el standing original
        self.standing_tdf = original_standing

    def process_trials_all(self, run_score: bool = True) -> None:
        self.process_trials()
        if run_score:
            self.process_trials_score()
    ### fin nuevo

    def process_videos(self, all_trials=False) -> None:
        """Compress the mjpeg videos to h264 avi format with FLENI logo overlay

        Args:
            all_trials (bool, optional): Compress the video of all tdf trials, not
            just the ones selected for processing. Defaults to False.
        """

        videos = (
            self.videos
            if all_trials
            else sorted(
                [
                    i.filePath.with_name(i.filePath.stem + j).with_suffix(".mjpeg")
                    for i in (self.trial_tdfs + [self.standing_tdf])
                    for j in ["_tvc1", "_tvc2"]
                ]
            )
        )

        videos = [i for i in videos if i.exists()]

        for video in videos:
            log.info(f"Compressing {video.name}...")
            ret = subprocess.run(
                [
                    "c:\\ffmpeg\\bin\\ffmpeg.exe",
                    "-loglevel",
                    "error",
                    "-stats",
                    "-i",
                    video,
                    "-i",
                    "c:\\FLENISmartDX\\fleni-logo.png",
                    "-filter_complex",
                    r"[0:v] setpts=PTS-STARTPTS [video];[1:v]setpts=PTS-STARTPTS,scale=250:-1,format=argb,geq=r='r(X,Y)':a='0.5*alpha(X,Y)' [logo];[video][logo] overlay=1:x=720:y=690",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "slow",
                    "-crf",
                    "20",
                    "-an",
                    "-pix_fmt",
                    "yuv420p",
                    video.with_suffix(".avi"),
                ]
            )
            try:
                ret.check_returncode()
            except subprocess.CalledProcessError as e:
                log.error(f"Error while compressing {video.name}: {e}")
            else:
                try:
                    log.info(f"Success. Deleting original {video.name}...")
                    video.unlink()
                except FileExistsError as f:
                    log.error(f"Error while deleting {video.name}: {f}")
