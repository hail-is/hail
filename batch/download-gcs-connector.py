import os
import shutil
import urllib.request

with urllib.request.urlopen("https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop2-2.2.7.jar") as input:
    with open(os.environ['SPARK_HOME'] + "/jars/gcs-connector-hadoop2-2.2.7.jar", 'wb') as output:
        shutil.copyfileobj(input, output)
