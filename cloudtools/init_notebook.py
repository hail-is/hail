#!/usr/bin/python
import json
import os
import subprocess as sp
import sys
from subprocess import check_output


if sys.version_info >= (3,0):
    decode = lambda s: s.decode()
    # Python 3 check_output returns a byte string
else:
    # In Python 2, bytes and str are the same
    decode = lambda s: s

if sys.version_info >= (3,7):
    def safe_call(*args):
        sp.run(args, capture_output=True, check=True)
else:
    def safe_call(*args):
        try:
            sp.check_output(args, stderr=sp.STDOUT)
        except sp.CalledProcessError as e:
            print(decode(e.output))
            raise e

def get_metadata(key):
    return decode(check_output(['/usr/share/google/get_metadata_value', 'attributes/{}'.format(key)]))

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
    conda_pkgs = [
        'mkl<2020',
        'numpy<2',
        'scipy<2',
        # pandas uses minor version for backwards incompatible changes
        # https://pandas.pydata.org/pandas-docs/version/0.22/whatsnew.html
        'pandas<0.24'
    ]
    pip_pkgs = [
        'seaborn<0.10',
        'decorator==4.2.1',
        'parsimonious<0.9',
        'ipywidgets<8',
        'jupyter_console<5',
        'nbconvert<6',
        'notebook<6',
        'qtconsole<5',
        'jupyter',
        'lxml<5',
        'jupyter-spark<0.5',
        'bokeh<0.14',
        'google-cloud==0.32.0',
        'jgscm<0.2'
    ]
    if sys.version_info < (3,5):
        pip_pkgs.extend([
            'matplotlib<3',
            # ipython 6 requires python>=3.3
            'ipython<6',
            # the jupyter metapackage has no version dependencies, so it always
            # pulls latest ipykernel, ipykernel >=5 requires python>=3.4
            'ipykernel<5',
        ])
    else:
        pip_pkgs.extend([
            'matplotlib<4',
            'ipython<7',
            'ipykernel<6',
        ])

    # add user-requested packages
    try:
        user_pkgs = get_metadata('PKGS')
    except:
        pass
    else:
        pip_pkgs.extend(user_pkgs.split('|'))

    safe_call('/opt/conda/bin/conda', 'update', 'setuptools')

    print('conda packages are {}'.format(conda_pkgs))
    command = ['/opt/conda/bin/conda', 'install']
    command.extend(conda_pkgs)
    safe_call(*command)

    print('pip packages are {}'.format(pip_pkgs))
    command = ['/opt/conda/bin/pip', 'install']
    command.extend(pip_pkgs)
    safe_call(*command)

    py4j = decode(check_output('ls /usr/lib/spark/python/lib/py4j*', shell=True).strip())

    print('getting metadata')

    jar_path = get_metadata('JAR')
    zip_path = get_metadata('ZIP')

    print('copying jar and zip')
    safe_call('gsutil', 'cp', jar_path, '/home/hail/hail.jar')
    safe_call('gsutil', 'cp', zip_path, '/home/hail/hail.zip')

    env_to_set = {
        'PYTHONHASHSEED': '0',
        'PYTHONPATH':
         '/usr/lib/spark/python/:{}:/home/hail/hail.zip'.format(py4j),
        'SPARK_HOME': '/usr/lib/spark/',
        'PYSPARK_PYTHON': '/opt/conda/bin/python',
        'PYSPARK_DRIVER_PYTHON': '/opt/conda/bin/python'
    }

    print('setting environment')

    for e, value in env_to_set.items():
        safe_call('/bin/sh', '-c',
                   'echo "export {}={}" | tee -a /etc/environment /usr/lib/spark/conf/spark-env.sh'.format(e, value))

    conf_to_set = [
        'spark.jars=/home/hail/hail.jar',
        'spark.executorEnv.PYTHONHASHSEED=0',
        'spark.submit.pyFiles=/home/hail/hail.zip',
        'spark.driver.extraClassPath=/home/hail/hail.jar',
        'spark.executor.extraClassPath=./hail.jar'
    ]

    print('setting spark-defaults.conf')

    for c in conf_to_set:
        safe_call('/bin/sh', '-c', 'echo "{}" >> /etc/spark/conf/spark-defaults.conf'.format(c))

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
    safe_call('/opt/conda/bin/jupyter', 'serverextension', 'enable', '--user', '--py', 'jupyter_spark')
    safe_call('/opt/conda/bin/jupyter', 'nbextension', 'install', '--user', '--py', 'jupyter_spark')
    safe_call('/opt/conda/bin/jupyter', 'nbextension', 'enable', '--user', '--py', 'jupyter_spark')
    safe_call('/opt/conda/bin/jupyter', 'nbextension', 'enable', '--user', '--py', 'widgetsnbextension')

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
    safe_call('systemctl', 'daemon-reload')
    safe_call('systemctl', 'enable', 'jupyter')
    safe_call('service', 'jupyter', 'start')
