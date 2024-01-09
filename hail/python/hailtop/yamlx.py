import yaml


def yaml_dump_multiline_str_as_literal_block(dumper, data):
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data.lstrip().rstrip(), style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


class HailDumper(yaml.SafeDumper):
    @property
    def yaml_representers(self):
        return {**super().yaml_representers, str: yaml_dump_multiline_str_as_literal_block}


def dump(data) -> str:
    return yaml.dump(data, Dumper=HailDumper)
