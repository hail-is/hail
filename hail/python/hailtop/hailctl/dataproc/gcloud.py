import subprocess
import sys
import typing


def run(command: typing.List[str]):
    """Run a gcloud command."""
    return subprocess.check_call(["gcloud"] + command)


def get_config(setting: str) -> typing.Optional[str]:
    """Get a gcloud configuration value."""
    try:
        return subprocess.check_output(["gcloud", "config", "get-value", setting], stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError as e:
        print(f"Warning: could not run 'gcloud config get-value {setting}': {e.output.decode}", file=sys.stderr)
        return None
