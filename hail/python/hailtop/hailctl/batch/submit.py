import os
import shlex
import sys
from pathlib import Path, PurePath
from typing import Any, Dict, List, Tuple

import orjson
import yaml

import hailtop.batch as hb
from hailtop.config import (
    ConfigVariable,
    configuration_of,
    get_deploy_config,
)

from .batch_cli_utils import StructuredFormatPlusTextOption


class HailctlBatchSubmitError(Exception):
    def __init__(self, message: str, exit_code: int):
        self.message = message
        self.exit_code = exit_code


def submit(
    image: str,
    entrypoint: List[str],
    name: str | None,
    cpu: str,
    memory: str,
    storage: str,
    machine_type: str | None,
    spot: bool,
    workdir: str | None,
    cloudfuse: List[Tuple[str, str, bool]] | None,
    env: Dict[str, str] | None,
    billing_project: str | None,
    remote_tmpdir: str | None,
    regions: List[str] | None,
    requester_pays_project: str | None,
    attributes: Dict[str, str] | None,
    volume_mounts: List[Tuple[str, str]] | None,
    shell: str | None,
    output: StructuredFormatPlusTextOption,
    wait: bool,
    quiet: bool,
    # FIXME: requester pays config
):
    with hb.ServiceBackend(
        billing_project=billing_project,
        remote_tmpdir=remote_tmpdir,
        regions=regions,
        gcs_requester_pays_configuration=requester_pays_project,
    ) as backend:
        entrypoint_str = ' '.join(shq(x) for x in entrypoint)
        b = hb.Batch(
            backend=backend,
            name=name or entrypoint_str,
            attributes=attributes,
        )

        volume_mount_inputs = []
        mkdirs = set()

        for str_src, dst_str in volume_mounts or []:
            src = Path(str_src).expanduser()
            if src.is_file():
                local_dst = PurePath(dst_str)
                if dst_str.endswith('/'):
                    local_dst /= src.name

                mkdirs.add(local_dst.parent)
                volume_mount_inputs.append((local_dst, b.read_input(src)))
            else:
                if not src.is_dir():
                    raise ValueError(f'"{src}" does not exist or is not a valid directory.')

                if str_src.endswith('/') and not dst_str.endswith('/'):
                    raise ValueError('copy and renaming a directory is not supported.')

                dst = PurePath(dst_str)
                if not str_src.endswith('/') and dst_str.endswith('/'):
                    dst /= src.name

                for curdir, _, filenames in os.walk(src, followlinks=True):
                    for file in filenames:
                        path = Path(curdir) / file
                        local_dst = dst / path.relative_to(src)
                        mkdirs.add(local_dst.parent)
                        volume_mount_inputs.append((local_dst, b.read_input(path)))

        volume_mount_str = "\n".join(f'mv {src} {dst}' for dst, src in volume_mount_inputs)
        mkdirs_str = "\n".join(f'mkdir -p {folder}' for folder in mkdirs)

        j = b.new_job(name=name or 'submit', attributes=attributes, shell=shell)
        j.image(image)

        for var in ConfigVariable:
            if (value := configuration_of(var, None, None)) is not None:
                j.env(var.envvar, value)

        for varname, value in (env or {}).items():
            j.env(varname, value)

        j.cpu(cpu)
        j.memory(memory)
        j.storage(storage)
        j.spot(spot)
        j._machine_type = machine_type

        set_workdir = f'cd {workdir}' if workdir is not None else ''

        j.command(f"""
{mkdirs_str}
{volume_mount_str}
{set_workdir}
{entrypoint_str}
""")

        if cloudfuse is not None:
            for bucket, mount, read_only in cloudfuse:
                j.cloudfuse(bucket, mount, read_only=read_only)

        bc_b = b.run(wait=False, disable_progress_bar=True)
        assert bc_b is not None  # needed for typechecking

        def print_job_info():
            if output == StructuredFormatPlusTextOption.TEXT:  # pyright: ignore
                deploy_config = get_deploy_config()
                url = deploy_config.external_url('batch', f'/batches/{bc_b.id}/jobs/1')
                print(f'Submitted batch {bc_b.id}, see {url}')
            else:
                status = bc_b.get_job(1).status()
                match output:
                    case StructuredFormatPlusTextOption.JSON:  # pyright: ignore
                        print(orjson.dumps(status).decode('utf-8').strip())
                    case StructuredFormatPlusTextOption.YAML:  # pyright: ignore
                        yaml.dump(status, sys.stdout)

        if wait:
            bc_b.wait(disable_progress_bar=quiet)

        print_job_info()


def shq(p: Any) -> str:
    return shlex.quote(str(p))
