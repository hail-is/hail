import os
import configparser

user_config = None


def get_user_config_path():
    config_file = os.environ.get('XDG_CONFIG_HOME', os.path.expanduser('~/.config')) + '/hail/config.yaml'
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    return config_file


def get_user_config():
    global user_config
    if user_config is None:
        user_config = configparser.ConfigParser()
        config_file = get_user_config_path()
        if os.path.exists(config_file):
            user_config.read(config_file)
    return user_config
