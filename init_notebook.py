#!/usr/bin/env python

import os
import json
from subprocess import call, Popen, PIPE

# get role of machine (master or worker)
role = Popen('/usr/share/google/get_metadata_value attributes/dataproc-role', shell=True, stdout=PIPE).communicate()[0]

# initialization actions to perform on master machine only
if role == 'Master':

    # additional packages to install
    pkgs = [
        'lxml',
        'jupyter-spark',
        'jgscm'
    ]

    # use pip to install packages
    for pkg in pkgs:
        call('/home/anaconda2/bin/pip install {}'.format(pkg), shell=True)

    # get latest Hail hash
    hash = Popen('gsutil cat gs://hail-common/latest-hash.txt', shell=True, stdout=PIPE, stderr=PIPE).communicate()[0].strip()

    # Hail jar and zip names
    hail_jar = 'hail-hail-is-master-all-spark2.0.2-{}.jar'.format(hash)
    hail_zip = 'pyhail-hail-is-master-{}.zip'.format(hash)

    # make directory for Hail and Jupyter notebook related files
    call('mkdir /home/hail/', shell=True)

    # copy Hail jar and zip to local directory on master node
    call('gsutil cp gs://hail-common/{} /home/hail/'.format(hail_jar), shell=True)
    call('gsutil cp gs://hail-common/{} /home/hail/'.format(hail_zip), shell=True)

    # modify default Spark config file to reference Hail jar and zip
    with open('/etc/spark/conf/spark-defaults.conf', 'ab') as f:
        opts = [
            'spark.jars=/home/hail/{}'.format(hail_jar),
            'spark.submit.pyFiles=/home/hail/{}'.format(hail_zip)
        ]
        f.write('\n'.join(opts))

    # create Jupyter configuration file
    call('mkdir -p /home/anaconda2/etc/jupyter/', shell=True)
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

    # create kernel spec file
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
            'PYTHONPATH': '/usr/lib/spark/python/:/usr/lib/spark/python/lib/py4j-0.10.3-src.zip:/home/hail/pyhail-hail-is-master-{}.zip'.format(hash)
        }
    }

    call('mkdir -p /home/anaconda2/share/jupyter/kernels/hail/', shell=True)
    with open('/home/anaconda2/share/jupyter/kernels/hail/kernel.json', 'wb') as f:
    	json.dump(kernel, f)

    # setup jupyter-spark extension
    call('/home/anaconda2/bin/jupyter serverextension enable --user --py jupyter_spark', shell=True)
    call('/home/anaconda2/bin/jupyter nbextension install --user --py jupyter_spark', shell=True)
    call('/home/anaconda2/bin/jupyter nbextension enable --user --py jupyter_spark', shell=True)
    call('/home/anaconda2/bin/jupyter nbextension enable --user --py widgetsnbextension', shell=True)

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
    		'ExecStart=/bin/sh -c "/home/anaconda2/bin/python /home/anaconda2/bin/jupyter notebook > /home/hail/jupyter.log 2>&1"',
    		'Restart=always',
    		'RestartSec=1',
    		'[Install]',
    		'WantedBy=multi-user.target'
    	]
    	f.write('\n'.join(opts) + '\n')

    # add Jupyter service to autorun and start it
    call('systemctl daemon-reload', shell=True)
    call('systemctl enable jupyter', shell=True)
    call('service jupyter start', shell=True)

	# give all permissions to Hail and Anaconda directories
    call('chmod -R 777 /home/hail/', shell=True)
    call('chmod -R 777 /home/anaconda2/', shell=True)
