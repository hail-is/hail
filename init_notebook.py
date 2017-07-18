#!/usr/bin/env python

import os
import json
import time
from subprocess import call, Popen, PIPE

# get role of machine (master or worker)
role = Popen('/usr/share/google/get_metadata_value attributes/dataproc-role', shell=True, stdout=PIPE).communicate()[0]

# initialization actions to perform on master machine only
if role == 'Master':

    # download Anaconda Python 2.7 installation script
    call(['wget', '-P', '/home/anaconda2/', 'https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh'])

    # install Anaconda in /home/anaconda2/
    call(['bash', '/home/anaconda2/Anaconda2-4.3.1-Linux-x86_64.sh', '-b', '-f', '-p', '/home/anaconda2/'])
    os.chmod('/home/anaconda2/', 0777)

    # additional packages to install
    pkgs = [
        'lxml',
        'jupyter-spark',
        'jgscm'
    ]

    # use pip to install packages
    for pkg in pkgs:
        call(['/home/anaconda2/bin/pip', 'install', pkg])

    # get Hail hash and Spark version to use for Jupyter notebook, if set through cluster startup metadata
    spark = Popen('/usr/share/google/get_metadata_value attributes/SPARK', shell=True, stdout=PIPE).communicate()[0].strip()
    hash = Popen('/usr/share/google/get_metadata_value attributes/HASH', shell=True, stdout=PIPE).communicate()[0].strip() 

    # default to Spark 2.0.2 if not otherwise specified through metadata
    if not spark:
        spark = '2.0.2'

    # default to latest Hail build if none specified through metadata
    if not hash:
        hash = Popen(['gsutil', 'cat', 'gs://hail-common/latest-hash-spark{}.txt'.format(spark)], stdout=PIPE, stderr=PIPE).communicate()[0].strip()

    # Hail jar
    jar = Popen('/usr/share/google/get_metadata_value attributes/JAR', shell=True, stdout=PIPE).communicate()[0].strip()
    if jar:
        hail_jar = jar.rsplit('/')[-1]
        jar_path = jar
    else:
        hail_jar = 'hail-hail-is-master-all-spark{0}-{1}.jar'.format(spark, hash)
        jar_path = 'gs://hail-common/' + hail_jar

    # Hail zip
    zip = Popen('/usr/share/google/get_metadata_value attributes/ZIP', shell=True, stdout=PIPE).communicate()[0].strip()
    if zip:
        hail_zip = zip.rsplit('/')[-1]
        zip_path = zip
    else:    
        hail_zip = 'pyhail-hail-is-master-{}.zip'.format(hash)
        zip_path = 'gs://hail-common/' + hail_zip

    # make directory for Hail and Jupyter notebook related files
    if not os.path.isdir('/home/hail/'):
        os.mkdir('/home/hail/')
    os.chmod('/home/hail/', 0777)

    # copy Hail jar and zip to local directory on master node
    call(['gsutil', 'cp', jar_path, '/home/hail/'])
    call(['gsutil', 'cp', zip_path, '/home/hail/'])

    # create Jupyter kernel spec file
    kernel = {
        'argv': [
            '/home/anaconda2/bin/python',
            '-m',
            'ipykernel',
            '-f',
            '{connection_file}'
        ],
        'display_name': 'Hail',
        'language': 'python',
        'env': {
            'PYTHONHASHSEED': '0',
            'SPARK_HOME': '/usr/lib/spark/',
            'SPARK_CONF_DIR': '/home/hail/conf/',
            'PYTHONPATH': '/usr/lib/spark/python/:/usr/lib/spark/python/lib/py4j-0.10.3-src.zip:/home/hail/{}'.format(hail_zip)
        }
    }    

    # write kernel spec file to default Jupyter kernel directory
    os.makedirs('/home/anaconda2/share/jupyter/kernels/hail/')
    with open('/home/anaconda2/share/jupyter/kernels/hail/kernel.json', 'wb') as f:
        json.dump(kernel, f)

    # make directory for custom Spark conf
    os.mkdir('/home/hail/conf')

    # copy conf files to custom directory
    call(['cp', '/etc/spark/conf/spark-defaults.conf', '/home/hail/conf/spark-defaults.conf'])
    call(['cp', '/etc/spark/conf/spark-env.sh', '/home/hail/conf/spark-env.sh'])

    # modify custom Spark conf file to reference Hail jar and zip
    with open('/home/hail/conf/spark-defaults.conf', 'ab') as f:
        opts = [
            'spark.files=/home/hail/{}'.format(hail_jar),
            'spark.submit.pyFiles=/home/hail/{}'.format(hail_zip),
            'spark.driver.extraClassPath=./{}'.format(hail_jar),
            'spark.executor.extraClassPath=./{}'.format(hail_jar)
        ]
        f.write('\n'.join(opts))

    # add Spark variable designating Anaconda Python executable as the default on driver, in both custom and default conf files
    with open('/home/hail/conf/spark-env.sh', 'ab') as f_custom, open('/etc/spark/conf/spark-env.sh', 'ab') as f_default:
        f_custom.write('PYSPARK_DRIVER_PYTHON=/home/anaconda2/bin/python' + '\n')
        f_default.write('PYSPARK_DRIVER_PYTHON=/home/anaconda2/bin/python' + '\n')

    # create Jupyter configuration file
    call(['mkdir', '-p', '/home/anaconda2/etc/jupyter/'])
    with open('/home/anaconda2/etc/jupyter/jupyter_notebook_config.py', 'wb') as f:
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
    call(['/home/anaconda2/bin/jupyter', 'serverextension', 'enable', '--user', '--py', 'jupyter_spark'])
    call(['/home/anaconda2/bin/jupyter', 'nbextension', 'install', '--user', '--py', 'jupyter_spark'])
    call(['/home/anaconda2/bin/jupyter', 'nbextension', 'enable', '--user', '--py', 'jupyter_spark'])
    call(['/home/anaconda2/bin/jupyter', 'nbextension', 'enable', '--user', '--py', 'widgetsnbextension'])

    # create systemd service file for Jupyter notebook server process
    with open('/lib/systemd/system/jupyter.service', 'wb') as f:
    	opts = [
    		'[Unit]',
    		'Description=Jupyter Notebook',
    		'After=hadoop-yarn-resourcemanager.service',
    		'[Service]',
    		'Type=simple',
    		'User=root',
    		'Group=root',
    		'WorkingDirectory=/home/hail/',
            'ExecStart=/home/anaconda2/bin/python /home/anaconda2/bin/jupyter notebook',
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

    # sleep for 30 seconds to allow Jupyter notebook server to start
    time.sleep(30)
