from pathlib import Path

PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"

def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    if data_dir is None:
        return PACKAGE_DATA_DIR
    return Path(data_dir).expanduser().resolve()