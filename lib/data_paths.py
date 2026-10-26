"""Default dataset paths rooted two directories above this project."""

from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parents[3] / "data"


def project_data_path(*parts):
    """Return an absolute path below <project>/../../data."""
    return str(DATA_ROOT.joinpath(*parts))
