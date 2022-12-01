import yaml as _yaml


class yaml_literally_shown_str(str):
    pass


def yaml_literally_shown_str_representer(dumper, data):
    return dumper.represent_scalar(u'tag:yaml.org,2002:str', data, style='|')


_yaml.add_representer(yaml_literally_shown_str, yaml_literally_shown_str_representer)
