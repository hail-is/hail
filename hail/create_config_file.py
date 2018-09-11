import json
import sys

both = {
    'vars': {
        'hash': None,
        'driver_memory': None,
        'spark': None,
        'image': None
    },
    'flags': {
        'metadata': {
            'JAR': 'gs://hail-common/builds/{version}/jars/hail-{version}-{hash}-Spark-{spark}.jar',
            'ZIP': 'gs://hail-common/builds/{version}/python/hail-{version}-{hash}.zip',
        },
        'properties': {
            'spark:spark.driver.memory': '{driver_memory}g',
            'spark:spark.driver.maxResultSize': '0',
            'spark:spark.task.maxFailures': '20',
            'spark:spark.kryoserializer.buffer.max': '1g',
            'spark:spark.driver.extraJavaOptions': '-Xss4M',
            'spark:spark.executor.extraJavaOptions': '-Xss4M',
            'hdfs:dfs.replication': '1'
        }
    }
}

devel = {
    'vars': {
        'version': 'devel',
        'supported_spark': {'2.2.0': '1.2-deb9'}
    },
    'flags': {
        'image-version': '{image}',
        'metadata': {'MINICONDA_VERSION': '4.4.10'}
    }
}

v0_1 = {
    'vars': {
        'version': '0.1',
        'supported_spark': {'2.0.2': '1.1'}
    },
    'flags': {
        'image-version': '{image}',
        'metadata': {'MINICONDA_VARIANT': '2'}
    }
}

def merge(target, source):
    target['vars'].update(source['vars'])
    for k, v in source['flags'].items():
        if k not in target['flags']:
            target['flags'][k] = v
        elif isinstance(v, list):
            target['flags'][k].extend(v)
        elif isinstance(v, dict):
            target['flags'][k].update(v)
    return target

if __name__ == '__main__':
    if sys.argv[1] == 'devel':
        configs = merge(devel, both)
    else:
        assert sys.argv[1] == '0.1'
        configs = merge(v0_1, both)
    local_file = sys.argv[2]
    with open(local_file, 'w') as f:
        f.write(json.dumps(configs))
