import os
import configparser

user_config = None


def get_user_config_path():
    xdg_config_home = os.environ.get('XDG_CONFIG_HOME', os.path.expanduser('~/.config'))
    return xdg_config_home + '/hail/config.yaml'


def get_user_config():
    global user_config
    if user_config is None:
        user_config = configparser.ConfigParser()
        config_file = get_user_config_path()
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        open(config_file, 'a').close()
        user_config.read(config_file)
    return user_config
