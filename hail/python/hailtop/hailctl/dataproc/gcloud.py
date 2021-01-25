import json
import subprocess
import sys
import typing


def get_config(gcloud_configuration: typing.Optional[str], setting: str) -> typing.Optional[str]:
    """Get a gcloud configuration value."""
    try:
        cmd = ['gcloud']
        if gcloud_configuration:
            cmd.append(f'--configuration={gcloud_configuration}')
        cmd.extend(["config", "get-value", setting])
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError as e:
        print(f"Warning: could not run 'gcloud config get-value {setting}': {e.output.decode}", file=sys.stderr)
        return None


def run(command: typing.List[str]):
    """Run a gcloud command."""
    return subprocess.check_call(command)


class GCloudRunner:
    def __init__(self, beta: bool, dry_run: bool, gcloud_configuration: typing.Optional[str], project: typing.Optional[str], zone: typing.Optional[str]):
        self._beta = beta
        self._dry_run = dry_run
        self._gcloud_configuration = gcloud_configuration

        if not project:
            project = self.get_config('project')
        if not project:
            raise RuntimeError("Unable to determine the GCP project.  Specify the --project option to hailctl, or use `gcloud config set project <my-project>` to set a default.")
        self._project = project

        if not zone:
            zone = self.get_config("compute/zone")
        if not zone:
            raise RuntimeError("Unable to determine the compute zone.  Specify the --zone option to hailctl, or use `gcloud config set compute/zone <my-zone>` to set a default.")
        self._zone = zone

        dataproc_region = self.get_config("dataproc/region")
        if not dataproc_region:
            dataproc_region = zone.split('-')
            dataproc_region = dataproc_region[:2]
            dataproc_region = '-'.join(dataproc_region)
        else:
            if not zone.startswith(dataproc_region):
                raise RuntimeError("Compute zone and Dataproc region are incompatible: zone: {zone} vs region: {dataproc_region}.")

        self._dataproc_region = dataproc_region

    def get_config(self, setting: str) -> typing.Optional[str]:
        return get_config(self._gcloud_configuration, setting)

    def run_gcloud_command(self, command: typing.List[str]):
        gcloud_cmd = ['gcloud', f'--project={self._project}']
        if self._gcloud_configuration:
            gcloud_cmd.append(f'--configuration={self._gcloud_configuration}')
        if self._beta:
            gcloud_cmd.append('beta')
        gcloud_cmd.extend(command)
        print(' '.join(gcloud_cmd))
        if not self._dry_run:
            run(gcloud_cmd)

    def run_dataproc_command(self, command: typing.List[str]):
        dataproc_cmd = ['dataproc', f'--region={self._dataproc_region}', *command]
        self.run_gcloud_command(dataproc_cmd)

    def run_compute_command(self, command: typing.List[str]):
        compute_cmd = ['compute', f'--zone={self._zone}', *command]
        self.run_gcloud_command(compute_cmd)


def get_version() -> typing.Tuple[int, int, int]:
    """Get gcloud version as a tuple."""
    version_output = subprocess.check_output(["gcloud", "version", "--format=json"], stderr=subprocess.DEVNULL).decode().strip()
    version_info = json.loads(version_output)
    version = tuple(int(n) for n in version_info["Google Cloud SDK"].split("."))
    return version
