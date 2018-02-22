#!/usr/bin/python
import os
import json
from subprocess import check_output, call


def mkdir_if_not_exists(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != os.errno.EEXIST:
            raise

# get role of machine (master or worker)
role = check_output(['/usr/share/google/get_metadata_value', 'attributes/dataproc-role'])

# initialization actions to perform on master machine only
if role == 'Master':

    # make local directory for miniconda3 and Hail jar and zip
    mkdir_if_not_exists('/home/hail/')

    call('wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh', shell=True)
    call('bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/hail/miniconda3', shell=True)

    # additional packages to install
    pkgs = [
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'decorator',
        'jupyter',
        'lxml',
        'jupyter-spark',
        'jgscm'
    ]

    # add user-requested packages
    try:
        user_pkgs = check_output(['/usr/share/google/get_metadata_value', 'attributes/PKGS'])
    except:
        pass
    else:
        pkgs.extend(user_pkgs.split(','))

    # get Hail hash and Spark version to use for Jupyter notebook, if set through cluster startup metadata
    spark = check_output(['/usr/share/google/get_metadata_value', 'attributes/SPARK'])
    hail_version = check_output(['/usr/share/google/get_metadata_value', 'attributes/HAIL_VERSION'])
    hash_name = check_output(['/usr/share/google/get_metadata_value', 'attributes/HASH'])

    py4j_versions = {
            "2.0": "0.10.3",
            "2.1": "0.10.3",
            "2.2": "0.10.4"}
    py4j = py4j_versions[spark[:3]]

    # Hail jar
    try:
        custom_jar = check_output(['/usr/share/google/get_metadata_value', 'attributes/JAR'])
    except:
        hail_jar = 'hail-{0}-{1}-Spark-{2}.jar'.format(hail_version, hash_name, spark)
        jar_path = 'gs://hail-common/builds/{0}/jars/{1}'.format(hail_version, hail_jar)
    else:
        hail_jar = custom_jar.rsplit('/')[-1]
        jar_path = custom_jar

    # Hail zip
    try:
        custom_zip = check_output(['/usr/share/google/get_metadata_value', 'attributes/ZIP'])
    except:
        hail_zip = 'hail-{0}-{1}.zip'.format(hail_version, hash_name)
        zip_path = 'gs://hail-common/builds/{0}/python/{1}'.format(hail_version, hail_zip)
    else:
        hail_zip = custom_zip.rsplit('/')[-1]
        zip_path = custom_zip


    call(['gsutil', 'cp', jar_path, '/home/hail/'])
    call(['gsutil', 'cp', zip_path, '/home/hail/'])

    for pkg in pkgs:
        call('/home/hail/miniconda3/bin/pip install {}'.format(pkg), shell=True)

    # copy conf files to custom directory
    mkdir_if_not_exists('/home/hail/conf')
    call(['cp', '/etc/spark/conf/spark-defaults.conf', '/home/hail/conf/spark-defaults.conf'])
    call(['cp', '/etc/spark/conf/spark-env.sh', '/home/hail/conf/spark-env.sh'])

    # modify custom Spark conf file to reference Hail jar and zip
    with open('/home/hail/conf/spark-defaults.conf', 'a') as f:
        opts = [
            'spark.files=/home/hail/{}'.format(hail_jar),
            'spark.submit.pyFiles=/home/hail/{}'.format(hail_zip),
            'spark.driver.extraClassPath=./{}'.format(hail_jar),
            'spark.executor.extraClassPath=./{}'.format(hail_jar)
        ]
        f.write('\n'.join(opts))

    # create Jupyter kernel spec file
    kernel = {
        'argv': [
            '/home/hail/miniconda3/bin/python',
            '-m',
            'ipykernel',
            '-f',
            '{connection_file}'
        ],
        'display_name': 'Hail',
        'language': 'python',
        'env': {
            'PYSPARK_PYTHON': '/home/hail/miniconda3/bin/python',
            'PYSPARK_DRIVER_PYTHON': '/home/hail/miniconda3/bin/python',
            'PYTHONHASHSEED': '0',
            'SPARK_HOME': '/usr/lib/spark/',
            'SPARK_CONF_DIR': '/home/hail/conf/',
            'PYTHONPATH': '/usr/lib/spark/python/:/usr/lib/spark/python/lib/py4j-{}-src.zip:/home/hail/{}'.format(py4j, hail_zip)
        }
    }

    # write kernel spec file to default Jupyter kernel directory
    mkdir_if_not_exists('/home/hail/miniconda3/share/jupyter/kernels/hail/')
    with open('/home/hail/miniconda3/share/jupyter/kernels/hail/kernel.json', 'w') as f:
        json.dump(kernel, f)

    # create Jupyter configuration file
    mkdir_if_not_exists('/home/hail/miniconda3/etc/jupyter/')
    with open('/home/hail/miniconda3/etc/jupyter/jupyter_notebook_config.py', 'w') as f:
        opts = [
            'c.Application.log_level = "DEBUG"',
            'c.NotebookApp.ip = "127.0.0.1"',
            'c.NotebookApp.open_browser = False',
            'c.NotebookApp.port = 8123',
            'c.NotebookApp.token = ""',
            'c.NotebookApp.contents_manager_class = "jgscm.GoogleStorageContentManager"'
        ]
        f.write('\n'.join(opts) + '\n')

    # setup jupyter-spark extension
    call(['/home/hail/miniconda3/bin/jupyter', 'serverextension', 'enable', '--user', '--py', 'jupyter_spark'])
    call(['/home/hail/miniconda3/bin/jupyter', 'nbextension', 'install', '--user', '--py', 'jupyter_spark'])
    call(['/home/hail/miniconda3/bin/jupyter', 'nbextension', 'enable', '--user', '--py', 'jupyter_spark'])
    call(['/home/hail/miniconda3/bin/jupyter', 'nbextension', 'enable', '--user', '--py', 'widgetsnbextension'])
    
    # create systemd service file for Jupyter notebook server process
    with open('/lib/systemd/system/jupyter.service', 'w') as f:
        opts = [
            '[Unit]',
            'Description=Jupyter Notebook',
            'After=hadoop-yarn-resourcemanager.service',
            '[Service]',
            'Type=simple',
            'User=root',
            'Group=root',
            'WorkingDirectory=/home/hail/',
            'ExecStart=/home/hail/miniconda3/bin/python /home/hail/miniconda3/bin/jupyter notebook --allow-root',
            'Restart=always',
            'RestartSec=1',
            '[Install]',
            'WantedBy=multi-user.target'
        ]
        f.write('\n'.join(opts) + '\n')

    # add Jupyter service to autorun and start it
    call(['systemctl', 'daemon-reload'])
    call(['systemctl', 'enable', 'jupyter'])
    call(['service', 'jupyter', 'start'])
