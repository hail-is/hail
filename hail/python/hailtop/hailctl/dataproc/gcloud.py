import json
import subprocess
import sys
import typing


def get_config(setting: str) -> typing.Optional[str]:
    """Get a gcloud configuration value."""
    try:
        return subprocess.check_output(["gcloud", "config", "get-value", setting], stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError as e:
        print(f"Warning: could not run 'gcloud config get-value {setting}': {e.output.decode}", file=sys.stderr)
        return None


class GCloudRunner:
    def __init__(self, project: typing.Optional[str], zone: typing.Optional[str], dry_run: bool):
        if not project:
            project = get_config('project')
        if not project:
            raise RuntimeError("Unable to determine the GCP project.  Specify the --project option to hailctl, or use `gcloud config set project <my-project>` to set a default.")
        self._project = project
        if not zone:
            zone = get_config("compute/zone")
        if not zone:
            raise RuntimeError("Unable to determine the compute zone.  Specify the --zone option to hailctl, or use `gcloud config set compute/zone <my-zone>` to set a default.")
        self._zone = zone
        self._dry_run = dry_run

    def run(self, command: typing.List[str]):
        gcloud_cmd = ['gcloud', f'--project={self._project}', f'--zone={self._zone}', *command]
        print(' '.join(gcloud_cmd))
        if not self._dry_run:
            subprocess.check_call(gcloud_cmd)


def get_version() -> typing.Tuple[int, int, int]:
    """Get gcloud version as a tuple."""
    version_output = subprocess.check_output(["gcloud", "version", "--format=json"], stderr=subprocess.DEVNULL).decode().strip()
    version_info = json.loads(version_output)
    version = tuple(int(n) for n in version_info["Google Cloud SDK"].split("."))
    return version
