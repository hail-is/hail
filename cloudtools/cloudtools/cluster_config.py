import json

class ClusterConfig:
    def __init__(self, json_str):
        params = json.loads(json_str)
        self.vars = params['vars']
        self.flags = params['flags']

    def extend_flag(self, flag, values):
        if flag not in self.flags:
            self.flags[flag] = values
        elif isinstance(self.flags[flag], list):
            assert isinstance(values, list)
            self.flags[flag].extend(values)
        else:
            assert isinstance(self.flags[flag], dict)
            assert isinstance(values, dict)
            self.flags[flag].update(values)

    def parse_and_extend(self, flag, values):
        values = dict(tuple(pair.split('=')) for pair in values.split(',') if '=' in pair)
        self.extend_flag(flag, values)

    def format(self, obj):
        if isinstance(obj, dict):
            return self.format(['{}={}'.format(k, v) for k, v in obj.items()])
        if isinstance(obj, list):
            return self.format(','.join(obj))
        else:
            return str(obj).format(**self.vars)

    def jar(self):
        return self.flags['metadata']['JAR'].format(**self.vars)

    def zip(self):
        return self.flags['metadata']['ZIP'].format(**self.vars)

    def configure(self, sha, spark):
        self.vars['spark'] = spark
        image = self.vars['supported_spark'].get(spark)
        if image is None:
            raise ValueError(
                'Incompatible spark version {spark}, compatible versions are: {compat}'.format(
                    spark=spark, compat=list(self.vars['supported_spark'])))
        self.vars['image'] = image
        self.vars['hash'] = sha

    def get_command(self, name):
        flags = ['--{}={}'.format(f, self.format(v)) for f, v in self.flags.items()]
        return ['gcloud',
                'dataproc',
                'clusters',
                'create',
                name] + flags
