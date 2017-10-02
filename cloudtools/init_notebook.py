#!/usr/bin/python
import os
import json
from subprocess import check_output, call

# get role of machine (master or worker)
role = check_output(['/usr/share/google/get_metadata_value', 'attributes/dataproc-role'])

# initialization actions to perform on master machine only
if role == 'Master':

	# install pip
	call(['apt-get', 'update'])
	call(['apt-get', 'install', '-y', 'python-dev'])
	call(['apt-get', 'install', '-y', 'python-pip'])
	call(['pip', 'install', '--upgrade', 'pip'])

	# additional packages to install
	pkgs = [
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

	# use pip to install packages
	for pkg in pkgs:
		call(['pip', 'install', '--upgrade', pkg])

	# get Hail hash and Spark version to use for Jupyter notebook, if set through cluster startup metadata
	spark = check_output(['/usr/share/google/get_metadata_value', 'attributes/SPARK'])
	hail_version = check_output(['/usr/share/google/get_metadata_value', 'attributes/HAIL_VERSION'])
	hash_name = check_output(['/usr/share/google/get_metadata_value', 'attributes/HASH'])

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

	# copy Hail jar and zip to local directory on master node
	call(['gsutil', 'cp', jar_path, '/usr/lib/spark/jars/'])
	call(['gsutil', 'cp', zip_path, '/usr/lib/spark/python/'])

	# modify custom Spark conf file to reference Hail jar and zip
	with open('/etc/spark/conf/spark-defaults.conf', 'a') as f:
		opts = [
			'spark.files=/usr/lib/spark/jars/{}'.format(hail_jar),
			'spark.submit.pyFiles=/usr/lib/spark/python/{}'.format(hail_zip),
			'spark.driver.extraClassPath=./{}'.format(hail_jar),
			'spark.executor.extraClassPath=./{}'.format(hail_jar)
		]
		f.write('\n'.join(opts))

	# create Jupyter kernel spec file
	kernel = {
		'argv': [
			'/usr/bin/python',
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
			'SPARK_CONF_DIR': '/etc/spark/conf/',
			'PYTHONPATH': '/usr/lib/spark/python/:/usr/lib/spark/python/lib/py4j-0.10.3-src.zip:/usr/lib/spark/python/{}'.format(hail_zip)
		}
	}

	# write kernel spec file to default Jupyter kernel directory
	os.mkdir('/usr/local/share/jupyter/kernels/hail/')
	with open('/usr/local/share/jupyter/kernels/hail/kernel.json', 'w') as f:
		json.dump(kernel, f)

	# create Jupyter configuration file
	os.mkdir('/usr/local/etc/jupyter/')
	with open('/usr/local/etc/jupyter/jupyter_notebook_config.py', 'w') as f:
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
	call(['/usr/local/bin/jupyter', 'serverextension', 'enable', '--user', '--py', 'jupyter_spark'])
	call(['/usr/local/bin/jupyter', 'nbextension', 'install', '--user', '--py', 'jupyter_spark'])
	call(['/usr/local/bin/jupyter', 'nbextension', 'enable', '--user', '--py', 'jupyter_spark'])
	call(['/usr/local/bin/jupyter', 'nbextension', 'enable', '--user', '--py', 'widgetsnbextension'])
	
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
			'WorkingDirectory=/usr/local/',
			'ExecStart=/usr/bin/python /usr/local/bin/jupyter notebook --allow-root',
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
