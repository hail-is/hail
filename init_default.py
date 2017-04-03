#!/usr/bin/env python

from subprocess import call, Popen, PIPE

# get role of machine (master or worker)
role = Popen('/usr/share/google/get_metadata_value attributes/dataproc-role', shell=True, stdout=PIPE).communicate()[0]

# install Anaconda Python distribution on master machine only
if role == 'Master':

	# download Anaconda Python 2.7 installation script
    call('wget -P /home/anaconda2/ https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh', shell=True)

    # install Anaconda in /home/anaconda2/
    call('bash /home/anaconda2/Anaconda2-4.3.1-Linux-x86_64.sh -b -f -p /home/anaconda2/', shell=True)

    # add Spark variable designating Anaconda Python executable as the default on driver
    with open('/etc/spark/conf/spark-env.sh', 'ab') as f:
        f.write('PYSPARK_DRIVER_PYTHON=/home/anaconda2/bin/python' + '\n')
