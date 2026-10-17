import configparser
from pathlib import Path

DEFAULT_CFG_PATH = Path(__file__).resolve().parent / "score.cfg"


class ScoreConfig:
    def __init__(self, cfg_path: Path = None):
        self.cfg_path = Path(cfg_path) if cfg_path else DEFAULT_CFG_PATH

    @property
    def config(self):
        cfg = configparser.ConfigParser()
        cfg.read(self.cfg_path)
        return cfg

    @property
    def mdx_score_suffix(self) -> str:
        return self.config["archivos"]["mdx_score_suffix"]

    @property
    def standing_copia(self) -> str:
        return self.config["archivos"]["standing_copia"]
