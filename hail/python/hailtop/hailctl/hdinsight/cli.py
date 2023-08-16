import subprocess

from typing import Optional, List, Annotated as Ann

import typer
from typer import Option as Opt, Argument as Arg

from .start import start as hdinsight_start, VepVersion
from .submit import submit as hdinsight_submit


app = typer.Typer(
    name='hdinsight',
    no_args_is_help=True,
    help='Manage and monitor Hail HDInsight clusters.',
    pretty_exceptions_show_locals=False,
)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def start(
    ctx: typer.Context,
    cluster_name: str,
    storage_account: Ann[str, Arg(help='Storage account in which to create a container for ephemeral cluster data.')],
    resource_group: Ann[str, Arg(help='Resource group in which to place cluster.')],
    http_password: Ann[
        Optional[str], Opt(help='Password for web access. If unspecified one will be generated.')
    ] = None,
    sshuser_password: Ann[
        Optional[str], Opt(help='Password for ssh access. If unspecified one will be generated.')
    ] = None,
    location: Ann[str, Opt(help='Azure location in which to place the cluster.')] = 'eastus',
    num_workers: Ann[int, Opt(help='Initial number of workers.')] = 2,
    install_hail_uri: Ann[
        Optional[str],
        Opt(
            help="A custom install hail bash script to use. Must be accessible by the cluster's head nodes. http(s) and wasb(s) protocols are both acceptable"
        ),
    ] = None,
    install_native_deps_uri: Ann[
        Optional[str],
        Opt(
            help="A custom native deps bash script to use. Must be accessible by the cluster's nodes. http(s) and wasb(s) protocols are both acceptable"
        ),
    ] = None,
    wheel_uri: Ann[
        Optional[str],
        Opt(
            help="A custom wheel file to use. Must be accessible by the cluster's head nodes. only http(s) protocol is acceptable"
        ),
    ] = None,
    vep: Ann[Optional[VepVersion], Opt(help='Install VEP for the specified reference genome.')] = None,
    vep_loftee_uri: Ann[
        Optional[str],
        Opt(
            help="(REQUIRED FOR VEP) A folder file containing the VEP loftee data files. There are tarred, requester-pays copies available at gs://hail-REGION-vep/loftee-beta/GRCh38.tar and gs://hail-REGION-vep/loftee-beta/GRCh37.tar where REGION is one of us, eu, uk, and aus-sydney. Must be accessible by the cluster's head nodes. Must be an Azure blob storage URI like https://account.blob.core.windows.net/container/foo. See the Azure-specific VEP instructions in the Hail documentation."
        ),
    ] = None,
    vep_homo_sapiens_uri: Ann[
        Optional[str],
        Opt(
            help="(REQUIRED FOR VEP) A folder file containing the VEP homo sapiens data files. There are tarred, requester-pays copies available at gs://hail-REGION-vep/homo-sapiens/95_GRCh38.tar and gs://hail-REGION-vep/homo-sapiens/85_GRCh37.tar where REGION is one of us, eu, uk, and aus-sydney. Must be accessible by the cluster's head nodes. Must be an Azure blob storage URI like https://account.blob.core.windows.net/container/foo. See the Azure-specific VEP instructions in the Hail documentation."
        ),
    ] = None,
    vep_config_uri: Ann[
        Optional[str],
        Opt(
            help="A VEP config to use. Must be accessible by the cluster's head nodes. Only http(s) protocol is acceptable."
        ),
    ] = None,
    install_vep_uri: Ann[
        Optional[str],
        Opt(
            help="A custom VEP install script to use. Must be accessible by the cluster's nodes. http(s) and wasb(s) protocols are both acceptable"
        ),
    ] = None,
):
    '''
    Start an HDInsight cluster configured for Hail.
    '''
    from ... import pip_version  # pylint: disable=import-outside-toplevel

    hail_version = pip_version()

    def default_artifact(filename: str) -> str:
        return f'https://raw.githubusercontent.com/hail-is/hail/{hail_version}/hail/python/hailtop/hailctl/hdinsight/resources/{filename}'

    hdinsight_start(
        cluster_name,
        storage_account,
        resource_group,
        http_password,
        sshuser_password,
        location,
        num_workers,
        install_hail_uri or default_artifact('install-hail.sh'),
        install_native_deps_uri or default_artifact('install-native-deps.sh'),
        wheel_uri
        or f'https://storage.googleapis.com/hail-common/azure-hdinsight-wheels/hail-{hail_version}-py3-none-any.whl',
        vep,
        vep_loftee_uri,
        vep_homo_sapiens_uri,
        vep_config_uri,
        install_vep_uri or default_artifact('install-vep.sh'),
        ctx.args,
    )


@app.command()
def stop(
    name: str,
    storage_account: Ann[str, Arg(help="Storage account in which the cluster's container exists.")],
    resource_group: Ann[str, Arg(help='Resource group in which the cluster exists.')],
    extra_hdinsight_delete_args: Optional[List[str]] = None,
    extra_storage_delete_args: Optional[List[str]] = None,
):
    '''
    Stop an HDInsight cluster configured for Hail.
    '''
    print(f"Stopping cluster '{name}'...")

    subprocess.check_call(
        [
            'az',
            'hdinsight',
            'delete',
            '--name',
            name,
            '--resource-group',
            resource_group,
            *(extra_hdinsight_delete_args or []),
        ]
    )
    subprocess.check_call(
        [
            'az',
            'storage',
            'container',
            'delete',
            '--name',
            name,
            '--account-name',
            storage_account,
            *(extra_storage_delete_args or []),
        ]
    )


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def submit(
    ctx: typer.Context,
    name: str,
    storage_account: Ann[str, Arg(help="Storage account in which the cluster's container exists.")],
    http_password: Ann[str, Arg(help='Web password for the cluster')],
    script: Ann[str, Arg(help='Path to script.')],
    arguments: Ann[Optional[List[str]], Arg(help='You should use -- if you want to pass option-like arguments through.')] = None,
):
    '''
    Submit a job to an HDInsight cluster configured for Hail.

    If you wish to pass option-like arguments you should use "--". For example:



    $ hailctl hdinsight submit name account password script.py --image-name docker.io/image my_script.py -- some-argument --animal dog
    '''
    raise ValueError((name, storage_account, http_password, script, [*(arguments or []), *ctx.args]))
    hdinsight_submit(name, storage_account, http_password, script, [*(arguments or []), *ctx.args])


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def list(ctx: typer.Context):
    '''
    List HDInsight clusters configured for Hail.
    '''
    subprocess.check_call(['az', 'hdinsight', 'list', *ctx.args])
