import os
import tempfile
import zipfile

from typing import Optional, List

from . import gcloud


def submit(
    name: str,
    script: str,
    files: str,
    pyfiles: str,
    properties: Optional[str],
    gcloud_configuration: Optional[str],
    dry_run: bool,
    region: Optional[str],
    pass_through_args: List[str],
):
    print("Submitting to cluster '{}'...".format(name))

    if ',' in files:
        files_list = files.split(',')
        files_list = [os.path.expanduser(file) for file in files_list]
        files = ','.join(files_list)

    # If you only provide one (comma-sep) argument, and it's a zip file, use that file directly
    if not (pyfiles and pyfiles.endswith('.zip') and ',' not in pyfiles):
        pyfiles_list = []
        if pyfiles:
            pyfiles_list.extend(pyfiles.split(','))
        pyfiles_list.extend(os.environ.get('HAIL_SCRIPTS', '').split(':'))
        if pyfiles_list:
            tfile = tempfile.mkstemp(suffix='.zip', prefix='pyscripts_')[1]
            with zipfile.ZipFile(tfile, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for _hail_script_entry in pyfiles_list:
                    hail_script_entry = os.path.expanduser(_hail_script_entry)
                    if hail_script_entry.endswith('.py'):
                        zipf.write(hail_script_entry, arcname=os.path.basename(hail_script_entry))
                    else:
                        for root, _, pyfiles_walk in os.walk(hail_script_entry):
                            for pyfile in pyfiles_walk:
                                if pyfile.endswith('.py'):
                                    zipf.write(
                                        os.path.join(root, pyfile),
                                        os.path.relpath(
                                            os.path.join(root, pyfile), os.path.join(hail_script_entry, '..')
                                        ),
                                    )
            pyfiles = tfile

    # pyspark submit command
    cmd = [
        'dataproc',
        'jobs',
        'submit',
        'pyspark',
        script,
        '--cluster={}'.format(name),
        '--files={}'.format(files),
        '--py-files={}'.format(pyfiles),
        '--properties={}'.format(properties or ''),
    ]
    if gcloud_configuration:
        cmd.append('--configuration={}'.format(gcloud_configuration))

    if region:
        cmd.append('--region={}'.format(region))

    # append arguments to pass to the Hail script
    if pass_through_args:
        cmd.append('--')
        cmd.extend(pass_through_args)

    # print underlying gcloud command
    print('gcloud command:')
    print('gcloud ' + ' '.join(cmd[:5]) + ' \\\n    ' + ' \\\n    '.join(cmd[6:]))

    # submit job
    if not dry_run:
        gcloud.run(cmd)
