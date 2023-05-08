class ClusterConfig:
    def __init__(self):
        self.vars = {}
        self.flags = {}

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
        values = dict(tuple(pair.split('=')) for pair in values.split(',') if '=' in pair)  # type: ignore
        self.extend_flag(flag, values)

    def format(self, obj):
        if isinstance(obj, dict):
            return self.format(['{}={}'.format(k, v) for k, v in obj.items()])
        if isinstance(obj, list):
            return self.format(','.join(obj))
        return str(obj).format(**self.vars)

    def get_command(self, name):
        flags = ['--{}={}'.format(f, self.format(v)) for f, v in self.flags.items()]
        return ['gcloud',
                'dataproc',
                'clusters',
                'create',
                name,
                *flags]
