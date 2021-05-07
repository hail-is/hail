import os
import urllib.request
import shutil

# Regarding explicitly selecting 2.0.1: https://github.com/hail-is/hail/issues/8343
with urllib.request.urlopen("https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop2-2.0.1.jar") as input:
    with open(os.environ['SPARK_HOME'] + "/jars/gcs-connector-hadoop2-2.0.1.jar", 'wb') as output:
        shutil.copyfileobj(input, output)
