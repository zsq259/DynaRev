import logging
from pathlib import Path


def configure_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("dyna_agent")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(run_dir / "runner.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
