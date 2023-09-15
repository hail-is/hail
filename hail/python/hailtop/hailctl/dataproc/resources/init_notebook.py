#!/opt/conda/default/bin/python3
import json
import os
import subprocess as sp
import sys
import errno
from subprocess import check_output

assert sys.version_info > (3, 0), sys.version_info


def safe_call(*args, **kwargs):
    try:
        sp.check_output(args, stderr=sp.STDOUT, **kwargs)
    except sp.CalledProcessError as e:
        print(e.output.decode())
        raise e


def get_metadata(key):
    return check_output(['/usr/share/google/get_metadata_value', 'attributes/{}'.format(key)]).decode()


def mkdir_if_not_exists(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


# get role of machine (master or worker)
role = get_metadata('dataproc-role')

if role == 'Master':
    # additional packages to install
    pip_pkgs = [
        'setuptools',
        'mkl<2020',
        'lxml<5',
        'https://github.com/hail-is/jgscm/archive/v0.1.13+hail.zip',
        'ipykernel==6.22.0',
        'ipywidgets==8.0.6',
        'jupyter-console==6.6.3',
        'nbconvert==7.3.1',
        'notebook==6.5.4',
        'qtconsole==5.4.2',
    ]

    # add user-requested packages
    try:
        user_pkgs = get_metadata('PKGS')
    except Exception:
        pass
    else:
        pip_pkgs.extend(user_pkgs.split('|'))

    print('pip packages are {}'.format(pip_pkgs))
    command = ['pip', 'install']
    command.extend(pip_pkgs)
    safe_call(*command)

    print('getting metadata')

    wheel_path = get_metadata('WHEEL')
    wheel_name = wheel_path.split('/')[-1]

    print('copying wheel')
    safe_call('gcloud', 'storage', 'cp', wheel_path, f'/home/hail/{wheel_name}')

    safe_call('pip', 'install', '--no-dependencies', f'/home/hail/{wheel_name}')

    print('setting environment')

    spark_lib_base = '/usr/lib/spark/python/lib/'
    files_to_add = [os.path.join(spark_lib_base, x) for x in os.listdir(spark_lib_base) if x.endswith('.zip')]

    env_to_set = {
        'PYTHONHASHSEED': '0',
        'PYTHONPATH': ':'.join(files_to_add),
        'SPARK_HOME': '/usr/lib/spark/',
        'PYSPARK_PYTHON': '/opt/conda/default/bin/python',
        'PYSPARK_DRIVER_PYTHON': '/opt/conda/default/bin/python',
        'HAIL_LOG_DIR': '/home/hail',
        'HAIL_DATAPROC': '1',
    }

    # VEP ENV
    try:
        vep_config_uri = get_metadata('VEP_CONFIG_URI')
    except Exception:
        pass
    else:
        env_to_set["VEP_CONFIG_URI"] = vep_config_uri

    print('setting environment')

    for e, value in env_to_set.items():
        safe_call(
            '/bin/sh',
            '-c',
            'set -ex; echo "export {}={}" | tee -a /etc/environment /usr/lib/spark/conf/spark-env.sh'.format(e, value),
        )

    hail_jar = (
        sp.check_output(['/bin/sh', '-c', 'set -ex; python3 -m pip show hail | grep Location | sed "s/Location: //"'])
        .decode('ascii')
        .strip()
        + '/hail/backend/hail-all-spark.jar'
    )

    if not os.path.exists(hail_jar):
        raise ValueError(f'{hail_jar} must exist')

    conf_to_set = [
        'spark.executorEnv.PYTHONHASHSEED=0',
        'spark.app.name=Hail',
        # the below are necessary to make 'submit' work
        'spark.jars={}'.format(hail_jar),
        'spark.driver.extraClassPath={}'.format(hail_jar),
        'spark.executor.extraClassPath=./hail-all-spark.jar',
    ]

    print('setting spark-defaults.conf')

    with open('/etc/spark/conf/spark-defaults.conf', 'a') as out:
        out.write('\n')
        for c in conf_to_set:
            out.write(c)
            out.write('\n')

    # Update python3 kernel spec with the environment variables and the hail
    # spark monitor
    try:
        with open('/opt/conda/default/share/jupyter/kernels/python3/kernel.json', 'r') as f:
            python3_kernel = json.load(f)
    except:
        python3_kernel = {
            'argv': ['/opt/conda/default/bin/python', '-m', 'ipykernel', '-f', '{connection_file}'],
            'display_name': 'Python 3',
            'language': 'python',
        }
    python3_kernel['env'] = {
        **python3_kernel.get('env', dict()),
        **env_to_set,
        'HAIL_SPARK_MONITOR': '1',
        'SPARK_MONITOR_UI': 'http://localhost:8088/proxy/%APP_ID%',
    }

    # write python3 kernel spec file to default Jupyter kernel directory
    mkdir_if_not_exists('/opt/conda/default/share/jupyter/kernels/python3/')
    with open('/opt/conda/default/share/jupyter/kernels/python3/kernel.json', 'w') as f:
        json.dump(python3_kernel, f)

    # some old notebooks use the "Hail" kernel, so create that too
    hail_kernel = {**python3_kernel, 'display_name': 'Hail'}
    mkdir_if_not_exists('/opt/conda/default/share/jupyter/kernels/hail/')
    with open('/opt/conda/default/share/jupyter/kernels/hail/kernel.json', 'w') as f:
        json.dump(hail_kernel, f)

    # create Jupyter configuration file
    mkdir_if_not_exists('/opt/conda/default/etc/jupyter/')
    with open('/opt/conda/default/etc/jupyter/jupyter_notebook_config.py', 'w') as f:
        opts = [
            'c.Application.log_level = "DEBUG"',
            'c.NotebookApp.ip = "127.0.0.1"',
            'c.NotebookApp.open_browser = False',
            'c.NotebookApp.port = 8123',
            'c.NotebookApp.token = ""',
            'c.NotebookApp.contents_manager_class = "jgscm.GoogleStorageContentManager"',
        ]
        f.write('\n'.join(opts) + '\n')

    print('copying spark monitor')
    spark_monitor_gs = (
        'gs://hail-common/sparkmonitor-c1289a19ac117336fec31ec08a2b13afe7e420cf/sparkmonitor-0.0.12-py3-none-any.whl'
    )
    spark_monitor_wheel = '/home/hail/' + spark_monitor_gs.split('/')[-1]
    safe_call('gcloud', 'storage', 'cp', spark_monitor_gs, spark_monitor_wheel)
    safe_call('pip', 'install', spark_monitor_wheel)

    # setup jupyter-spark extension
    safe_call('/opt/conda/default/bin/jupyter', 'serverextension', 'enable', '--user', '--py', 'sparkmonitor')
    safe_call('/opt/conda/default/bin/jupyter', 'nbextension', 'install', '--user', '--py', 'sparkmonitor')
    safe_call('/opt/conda/default/bin/jupyter', 'nbextension', 'enable', '--user', '--py', 'sparkmonitor')
    safe_call('/opt/conda/default/bin/jupyter', 'nbextension', 'enable', '--user', '--py', 'widgetsnbextension')
    safe_call(
        """ipython profile create && echo "c.InteractiveShellApp.extensions.append('sparkmonitor.kernelextension')" >> $(ipython profile locate default)/ipython_kernel_config.py""",
        shell=True,
    )

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
            'ExecStart=/opt/conda/default/bin/python /opt/conda/default/bin/jupyter notebook --allow-root',
            'Restart=always',
            'RestartSec=1',
            '[Install]',
            'WantedBy=multi-user.target',
        ]
        f.write('\n'.join(opts) + '\n')

    # add Jupyter service to autorun and start it
    safe_call('systemctl', 'daemon-reload')
    safe_call('systemctl', 'enable', 'jupyter')
    safe_call('service', 'jupyter', 'start')
