import os
import tempfile
import zipfile
import click

from .dataproc import dataproc


@dataproc.command(
    help="Submit a Python script to a running Dataproc cluster.")
@click.argument('cluster_name')
@click.argument('script')
@click.option('--files',
              help="Comma-separated list of files to add to the working directory of the Hail application.")
@click.option('--pyfiles',
              help="Comma-separated list of files (or directories with python files) to add to the PYTHONPATH.")
@click.option('--properties', '-p',
              help="Extra Spark properties to set.")
@click.option('--extra-gcloud-submit-args',
              default='',
              help="Extra arguments to pass to 'gcloud dataproc clusters submit'")
@click.argument('script_args', nargs=-1)
@click.pass_context
def submit(
        ctx,
        cluster_name, script,
        files, pyfiles, properties, extra_gcloud_submit_args, script_args):
    runner = ctx.parent.obj

    print("Submitting to cluster '{}'...".format(cluster_name))

    # create files argument
    if not files:
        files = ''
    pyfiles = []
    if pyfiles:
        pyfiles.extend(pyfiles.split(','))
    pyfiles.extend(os.environ.get('HAIL_SCRIPTS', '').split(':'))
    if pyfiles:
        tfile = tempfile.mkstemp(suffix='.zip', prefix='pyscripts_')[1]
        zipf = zipfile.ZipFile(tfile, 'w', zipfile.ZIP_DEFLATED)
        for hail_script_entry in pyfiles:
            if hail_script_entry.endswith('.py'):
                zipf.write(hail_script_entry, arcname=os.path.basename(hail_script_entry))
            else:
                for root, _, pyfiles_walk in os.walk(hail_script_entry):
                    for pyfile in pyfiles_walk:
                        if pyfile.endswith('.py'):
                            zipf.write(os.path.join(root, pyfile),
                                       os.path.relpath(os.path.join(root, pyfile),
                                                       os.path.join(hail_script_entry, '..')))
        zipf.close()
        pyfiles = tfile
    else:
        pyfiles = ''

    if not properties:
        properties = ''

    # pyspark submit command
    cmd = [
        'jobs',
        'submit',
        'pyspark',
        script,
        '--cluster={}'.format(cluster_name),
        '--files={}'.format(files),
        '--py-files={}'.format(pyfiles),
        '--properties={}'.format(properties)
    ]

    if script_args:
        cmd.extend(['--', *script_args])

    cmd.extend(extra_gcloud_submit_args.split())

    runner.run_dataproc_command(cmd)
