from subprocess import check_call
import os
import tempfile
import zipfile


def init_parser(parser):
    parser.add_argument('name', type=str, help='Cluster name.')
    parser.add_argument('script', type=str, help="Path to script.")
    parser.add_argument('--files', required=False, type=str, help='Comma-separated list of files to add to the working directory of the Hail application.')
    parser.add_argument('--pyfiles', required=False, type=str, help='Comma-separated list of files (or directories with python files) to add to the PYTHONPATH.')
    parser.add_argument('--properties', '-p', required=False, type=str, help='Extra Spark properties to set.')
    parser.add_argument('--dry-run', action='store_true', help="Print gcloud dataproc command, but don't run it.")


def main(args, pass_through_args):  # pylint: disable=unused-argument
    print("Submitting to cluster '{}'...".format(args.name))

    # create files argument
    files = ''
    if args.files:
        files = args.files
    pyfiles = []
    if args.pyfiles:
        pyfiles.extend(args.pyfiles.split(','))
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

    # create properties argument
    properties = ''
    if args.properties:
        properties = args.properties

    # pyspark submit command
    cmd = [
        'gcloud',
        'dataproc',
        'jobs',
        'submit',
        'pyspark',
        args.script,
        '--cluster={}'.format(args.name),
        '--files={}'.format(files),
        '--py-files={}'.format(pyfiles),
        '--properties={}'.format(properties)
    ]

    # append arguments to pass to the Hail script
    if pass_through_args:
        cmd.append('--')
        cmd.extend(pass_through_args)

    # print underlying gcloud command
    print('gcloud command:')
    print(' '.join(cmd[:6]) + ' \\\n    ' + ' \\\n    '.join(cmd[6:]))

    # submit job
    if not args.dry_run:
        check_call(cmd)
