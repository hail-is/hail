from subprocess import check_call

def init_parser(parser):
    parser.add_argument('name', type=str, help='Cluster name.')
    parser.add_argument('script', type=str)
    parser.add_argument('--files', required=False, type=str, help='Comma-separated list of files to add to the working directory of the Hail application.')
    parser.add_argument('--properties', '-p', required=False, type=str, help='Extra Spark properties to set.')
    parser.add_argument('--args', type=str, help='Quoted string of arguments to pass to the Hail script being submitted.')


def main(args):
    print("Submitting to cluster '{}'...".format(args.name))

    # create files argument
    files = ''
    if args.files:
        files = args.files

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
        # '--py-files={}'.format(zip_path),
        '--properties={}'.format(properties)
    ]

    # append arguments to pass to the Hail script
    if args.args:
        cmd.append('--')
        for x in args.args.split():
            cmd.append(x)

    # print underlying gcloud command
    print('gcloud command:')
    print(' '.join(cmd[:6]) + ' \\\n    ' + ' \\\n    '.join(cmd[6:]))

    # submit job
    check_call(cmd)
