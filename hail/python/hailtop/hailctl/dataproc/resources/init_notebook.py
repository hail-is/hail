#!/usr/bin/env python3
import errno
import json
import os
import subprocess as sp
import sys
import sysconfig
import urllib.request

assert sys.version_info > (3, 12), sys.version_info

# Dataproc 3.0 images manage the default Python environment with pixi rather
# than conda, and neither layout is a documented interface. Derive every path
# from the interpreter running this init action instead of hard-coding either.
python_exe = os.path.realpath(sys.executable)
env_prefix = sys.prefix


def safe_call(*args, **kwargs):
    try:
        sp.check_output(args, stderr=sp.STDOUT, **kwargs)
    except sp.CalledProcessError as e:
        print(e.output.decode())
        raise e


def pip_install(*args):
    safe_call(python_exe, '-m', 'pip', 'install', *args)


def get_metadata(key):
    return sp.check_output(['/usr/share/google/get_metadata_value', f'attributes/{key}']).decode()


def mkdir_if_not_exists(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


# netlib's bundled jni shims link the system blas, lapack and arpack; the
# image ships openblas but not arpack, so jvm eigensolvers fall back to java
safe_call('apt-get', 'update', '-qq')
safe_call('apt-get', 'install', '-qq', '-y', 'libarpack2')

# get role of machine (master or worker)
role = get_metadata('dataproc-role')

if role == 'Master':
    # install hail's dependencies and user-requested packages
    pkgs = get_metadata('PKGS').split('|')
    print(f'pip packages are {pkgs}')
    pip_install(*pkgs)

    # sparkmonitor renders spark job progress in jupyter notebooks. Its kernel
    # extension crashes if spark events arrive before the jupyterlab frontend
    # opens its comm channel; apply the proposed patch until it is released.
    pip_install('sparkmonitor==3.3.0')
    urllib.request.urlretrieve(
        'https://github.com/swan-cern/sparkmonitor/commit/28e39fc0d6ee910dc40c791528f9e3a23ea543ad.patch',
        '/tmp/sparkmonitor.patch',
    )
    safe_call(
        'git',
        'apply',
        '--include=sparkmonitor/*',
        '/tmp/sparkmonitor.patch',
        cwd=sysconfig.get_paths()['purelib'],
    )

    # gcs-jupyter-plugin adds a cloud storage browser to jupyterlab, listing
    # the project's buckets and writing edits back to gcs. its stale pins
    # (aiohttp~=3.9.5, pydantic~=1.10, ...) conflict with hail's requirements
    # but newer versions satisfy it in practice, so skip dependency resolution
    pip_install('google-cloud-jupyter-config')
    pip_install('--no-deps', 'gcs-jupyter-plugin')

    # the cloud storage browser requires gcloud be configured with an account,
    # project and region; dataproc images configure all but the region
    zone = sp.check_output(['/usr/share/google/get_metadata_value', 'zone']).decode().rsplit('/', 1)[-1]
    region = zone.rsplit('-', 1)[0]
    safe_call('gcloud', 'config', 'set', 'compute/region', region)
    safe_call('gcloud', 'config', 'set', 'dataproc/region', region)

    print('getting metadata')

    wheel_path = get_metadata('WHEEL')
    wheel_name = wheel_path.split('/')[-1]

    print('copying wheel')
    safe_call('gcloud', 'storage', 'cp', wheel_path, f'/home/hail/{wheel_name}')

    pip_install('--no-dependencies', f'/home/hail/{wheel_name}')

    print('setting environment')

    # dataproc images install spark at /usr/lib/spark; prefer SPARK_HOME when
    # the environment provides it
    spark_home = os.environ.get('SPARK_HOME', '/usr/lib/spark').rstrip('/')

    env_to_set = {
        'PYTHONHASHSEED': '0',
        'PYSPARK_PYTHON': python_exe,
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

    # spark-env.sh is sourced by spark-submit, delivering these to jobs;
    # /etc/environment is read by pam for ssh sessions
    with open(os.path.join(spark_home, 'conf', 'spark-env.sh'), 'a') as f:
        f.writelines(f'export {e}={value}\n' for e, value in env_to_set.items())

    # /etc/environment entries must be KEY=VALUE
    with open('/etc/environment', 'a') as f:
        f.writelines(f'{e}={value}\n' for e, value in env_to_set.items())

    hail_location = next(
        line.removeprefix('Location: ').strip()
        for line in sp.check_output([python_exe, '-m', 'pip', 'show', 'hail']).decode().splitlines()
        if line.startswith('Location: ')
    )

    hail_jar = hail_location + '/hail/backend/hail-all-spark.jar'

    if not os.path.exists(hail_jar):
        raise ValueError(f'{hail_jar} must exist')

    conf_to_set = [
        'spark.executorEnv.PYTHONHASHSEED=0',
        # the below are necessary to make 'submit' work
        f'spark.jars={hail_jar}',
        f'spark.driver.extraClassPath={hail_jar}',
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
    kernels_dir = os.path.join(env_prefix, 'share', 'jupyter', 'kernels')
    try:
        with open(os.path.join(kernels_dir, 'python3', 'kernel.json'), 'r') as f:
            python3_kernel = json.load(f)
            assert isinstance(python3_kernel, dict)
    except:
        python3_kernel = {
            'argv': [python_exe, '-m', 'ipykernel', '-f', '{connection_file}'],
            'display_name': 'Python 3',
            'language': 'python',
        }

    spark_lib_base = os.path.join(spark_home, 'python', 'lib')
    files_to_add = [os.path.join(spark_lib_base, x) for x in os.listdir(spark_lib_base) if x.endswith('.zip')]
    python3_kernel['env'] = {
        **python3_kernel.get('env', dict()),
        **env_to_set,
        # spark constructs PYTHONPATH for the python processes it launches;
        # the kernel is a plain python process, so needs pyspark on its path
        'PYTHONPATH': ':'.join(files_to_add),
        'HAIL_SPARK_MONITOR': '1',
    }

    # write python3 kernel spec file to default Jupyter kernel directory
    mkdir_if_not_exists(os.path.join(kernels_dir, 'python3'))
    with open(os.path.join(kernels_dir, 'python3', 'kernel.json'), 'w') as f:
        json.dump(python3_kernel, f)

    # sparkmonitor's frontend is a prebuilt jupyterlab extension, active on
    # install; the kernel side must be enabled in the ipython kernel config.
    # /etc/ipython applies whatever user the kernel runs as
    mkdir_if_not_exists('/etc/ipython')
    with open('/etc/ipython/ipython_kernel_config.py', 'a') as f:
        f.write("c.InteractiveShellApp.extensions.append('sparkmonitor.kernelextension')\n")

    # the jupyter optional component's server starts before init actions run;
    # restart it to pick up the hail kernel specs and sparkmonitor labextension
    try:
        safe_call('systemctl', 'restart', 'jupyter')
    except sp.CalledProcessError:
        print('warning: failed to restart the jupyter component service')
