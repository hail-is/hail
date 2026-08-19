import json
import re
import subprocess
import sys
from typing import List, Optional, Tuple

CUSTOM_MACHINE_TYPE = re.compile(r"(?:[a-z0-9]+-)?custom-(\d+)-(\d+)(?:-ext)?")


def run(command: List[str]):
    """Run a gcloud command."""
    return subprocess.check_call(["gcloud", *command])


def output(command: List[str]) -> str:
    """Run a gcloud command and return its stdout."""
    return subprocess.check_output(["gcloud", *command]).decode()


def get_machine_type_info(machine_type: str) -> Tuple[int, float]:
    """Get the vCPU count and advertised memory (GiB) of a machine type."""
    custom = CUSTOM_MACHINE_TYPE.fullmatch(machine_type)
    if custom:
        return int(custom.group(1)), int(custom.group(2)) / 1024
    output = subprocess.check_output(
        [
            "gcloud",
            "compute",
            "machine-types",
            "list",
            f"--filter=name={machine_type}",
            "--limit=1",
            "--format=json",
        ],
        stderr=subprocess.DEVNULL,
    )
    machine_types = json.loads(output)
    if not machine_types:
        raise RuntimeError(f"machine type '{machine_type}' not found")
    return machine_types[0]["guestCpus"], machine_types[0]["memoryMb"] / 1024


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
