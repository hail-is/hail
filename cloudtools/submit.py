from subprocess import check_output, check_call


def init_parser(parser):
    parser.add_argument('name', type=str, help='Cluster name.')
    parser.add_argument('script', type=str)
    parser.add_argument('--hash', default='latest', type=str,
                        help='Hail build to use for notebook initialization (default: %(default)s).')
    parser.add_argument('--spark', default='2.0.2', type=str, choices=['2.0.2', '2.1.0'],
                        help='Spark version used to build Hail (default: %(default)s).')
    parser.add_argument('--version', default='0.1', type=str, choices=['0.1', 'devel'],
                        help='Hail version to use (default: %(default)s).')
    parser.add_argument('--jar', required=False, type=str, help='Custom Hail jar to use.')
    parser.add_argument('--zip', required=False, type=str, help='Custom Hail zip to use.')
    parser.add_argument('--properties', '-p', required=False, type=str, help='Extra Spark properties to set.')


def main(args):
    print("Submitting to cluster '{}'...".format(args.name))

    # get Hail hash using either most recent, or an older version if specified
    if args.hash != 'latest':
        hash_name = args.hash
    else:
        hash_name = check_output(['gsutil', 'cat', 'gs://hail-common/builds/{0}/latest-hash-spark-{1}.txt'.format(args.version, args.spark)]).strip()

    # Hail jar
    if args.jar:
        hail_jar = args.jar.rsplit('/')[-1]
        jar_path = args.jar
    else:
        hail_jar = 'hail-{0}-{1}-Spark-{2}.jar'.format(args.version, hash_name, args.spark)
        jar_path = 'gs://hail-common/builds/{0}/jars/{1}'.format(args.version, hail_jar)

    # Hail zip
    if args.zip:
        zip_path = args.zip
    else:
        hail_zip = 'hail-{0}-{1}.zip'.format(args.version, hash_name)
        zip_path = 'gs://hail-common/builds/{0}/python/{1}'.format(args.version, hail_zip)

    # create properties argument
    properties = 'spark.driver.extraClassPath=./{0},spark.executor.extraClassPath=./{0}'.format(hail_jar)
    if args.properties:
        properties = properties + ',' + args.properties

    # pyspark submit command
    cmd = [
        'gcloud',
        'dataproc',
        'jobs',
        'submit',
        'pyspark',
        args.script,
        '--cluster={}'.format(args.name),
        '--files={}'.format(jar_path),
        '--py-files={}'.format(zip_path),
        '--properties={}'.format(properties)
    ]
    check_call(cmd)
