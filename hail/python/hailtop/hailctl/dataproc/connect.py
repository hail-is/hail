import json
import webbrowser
from enum import Enum
from typing import List, Optional

from . import gcloud


class DataprocConnectService(str, Enum):
    NOTEBOOK = 'notebook'
    NB = 'nb'
    SPARK_UI = 'spark-ui'
    UI = 'ui'
    SPARK_HISTORY = 'spark-history'
    HIST = 'hist'

    def shortcut(self):
        if self == self.UI:
            return self.SPARK_UI
        if self == self.HIST:
            return self.SPARK_HISTORY
        if self == self.NB:
            return self.NOTEBOOK

        return self


# component gateway endpoint and query string for each service; the spark ui
# is the history server listing incomplete (running) applications
GATEWAY_ENDPOINTS = {
    DataprocConnectService.NOTEBOOK: ('JupyterLab', ''),
    DataprocConnectService.SPARK_UI: ('Spark History Server', '?showIncomplete=true'),
    DataprocConnectService.SPARK_HISTORY: ('Spark History Server', ''),
}


def connect(
    name: str,
    service: DataprocConnectService,
    project: Optional[str],
    zone: Optional[str],
    dry_run: bool,
    pass_through_args: List[str],
):
    endpoint, query = GATEWAY_ENDPOINTS[service.shortcut()]

    # component gateway urls are authenticated https; no ssh tunnel or
    # browser proxy configuration is required
    region = gcloud.get_config('dataproc/region')
    if not region:
        zone = zone if zone else gcloud.get_config('compute/zone')
        if not zone:
            raise RuntimeError(
                "Could not determine dataproc region. Use `gcloud config set dataproc/region <my-region>` to set a default."
            )
        region = zone.rsplit('-', 1)[0]

    cmd = [
        'dataproc',
        'clusters',
        'describe',
        name,
        f'--region={region}',
        '--format=json(config.endpointConfig.httpPorts)',
        *pass_through_args,
    ]
    if project:
        cmd.append(f'--project={project}')

    print('gcloud ' + ' '.join(cmd))
    if dry_run:
        return

    described = json.loads(gcloud.output(cmd)) or {}
    http_ports = described.get('config', {}).get('endpointConfig', {}).get('httpPorts', {})
    url = http_ports.get(endpoint)
    if not url:
        raise RuntimeError(
            f"Could not find the '{endpoint}' component gateway url for cluster '{name}'. "
            "Was the cluster created with --enable-component-gateway?"
        )
    webbrowser.open(url + query)
