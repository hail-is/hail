import json
import subprocess
import sys
from typing import List, Optional, Tuple


def run(command: List[str]):
    """Run a gcloud command."""
    return subprocess.check_call(["gcloud", *command])


def get_config(setting: str) -> Optional[str]:
    """Get a gcloud configuration value."""
    try:
        return (
            subprocess.check_output(["gcloud", "config", "get-value", setting], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError as e:
        print(f"Warning: could not run 'gcloud config get-value {setting}': {e.output.decode}", file=sys.stderr)
        return None


def get_version() -> Tuple[int, int, int]:
    """Get gcloud version as a tuple."""
    version_output = (
        subprocess.check_output(["gcloud", "version", "--format=json"], stderr=subprocess.DEVNULL).decode().strip()
    )
    version_info = json.loads(version_output)
    v = version_info["Google Cloud SDK"].split(".")
    version = (int(v[0]), int(v[1]), int(v[2]))
    return version
