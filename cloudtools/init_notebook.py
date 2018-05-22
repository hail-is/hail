#!/usr/bin/python
import os
import json
from subprocess import check_output, call

import sys
if sys.version_info >= (3,0):
    # Python 3 check_output returns a byte string
    decode_f = lambda x: x.decode()
else:
    # In Python 2, bytes and str are the same
    decode_f = lambda x: x

def get_metadata(key):
    return decode_f(check_output(['/usr/share/google/get_metadata_value', 'attributes/{}'.format(key)]))

def mkdir_if_not_exists(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != os.errno.EEXIST:
            raise

# get role of machine (master or worker)
role = get_metadata('dataproc-role')

if role == 'Master':
    # additional packages to install
    pkgs = [
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'decorator==4.2.1',
        'parsimonious',
        'jupyter',
        'lxml',
        'jupyter-spark',
        'bokeh',
        'ipywidgets',
        'google-cloud=0.32.0',
        'jgscm'
    ]

    # add user-requested packages
    try:
        user_pkgs = get_metadata('PKGS')
    except:
        pass
    else:
        pkgs.extend(user_pkgs.split(','))

    py4j = decode_f(check_output('ls /usr/lib/spark/python/lib/py4j*', shell=True).strip())

    jar_path = get_metadata('JAR')
    zip_path = get_metadata('ZIP')

    call(['gsutil', 'cp', jar_path, '/home/hail/hail.jar'])
    call(['gsutil', 'cp', zip_path, '/home/hail/hail.zip'])

    call('/opt/conda/bin/conda update setuptools', shell=True)
    for pkg in pkgs:
        call('/opt/conda/bin/pip install {}'.format(pkg), shell=True)

    env_to_set = {
        'PYTHONHASHSEED': '0',
        'PYTHONPATH':
         '/usr/lib/spark/python/:{}:/home/hail/hail.zip'.format(py4j),
        'SPARK_HOME': '/usr/lib/spark/',
        'PYSPARK_PYTHON': '/opt/conda/bin/python',
        'PYSPARK_DRIVER_PYTHON': '/opt/conda/bin/python'
    }

    for e, value in env_to_set.items():
        call('echo "export {}={}" | tee -a /etc/environment /usr/lib/spark/conf/spark-env.sh'.format(e, value), shell=True)

    conf_to_set = [
        'spark.jars=/home/hail/hail.jar',
        'spark.executorEnv.PYTHONHASHSEED=0',
        'spark.submit.pyFiles=/home/hail/hail.zip',
        'spark.driver.extraClassPath=/home/hail/hail.jar',
        'spark.executor.extraClassPath=./hail.jar'
    ]

    for c in conf_to_set:
        call('echo "{}" >> /etc/spark/conf/spark-defaults.conf'.format(c), shell=True)

    # modify custom Spark conf file to reference Hail jar and zip

    # create Jupyter kernel spec file
    kernel = {
        'argv': [
            '/opt/conda/bin/python',
            '-m',
            'ipykernel',
            '-f',
            '{connection_file}'
        ],
        'display_name': 'Hail',
        'language': 'python',
        'env': env_to_set
    }

    # write kernel spec file to default Jupyter kernel directory
    mkdir_if_not_exists('/opt/conda/share/jupyter/kernels/hail/')
    with open('/opt/conda/share/jupyter/kernels/hail/kernel.json', 'w') as f:
        json.dump(kernel, f)

    # create Jupyter configuration file
    mkdir_if_not_exists('/opt/conda/etc/jupyter/')
    with open('/opt/conda/etc/jupyter/jupyter_notebook_config.py', 'w') as f:
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
    call(['/opt/conda/bin/jupyter', 'serverextension', 'enable', '--user', '--py', 'jupyter_spark'])
    call(['/opt/conda/bin/jupyter', 'nbextension', 'install', '--user', '--py', 'jupyter_spark'])
    call(['/opt/conda/bin/jupyter', 'nbextension', 'enable', '--user', '--py', 'jupyter_spark'])
    call(['/opt/conda/bin/jupyter', 'nbextension', 'enable', '--user', '--py', 'widgetsnbextension'])
    
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
            'ExecStart=/opt/conda/bin/python /opt/conda/bin/jupyter notebook --allow-root',
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
