import os
import configparser

from pathlib import Path

user_config = None


def get_user_config_path():
    xdg_config_home = os.environ.get('XDG_CONFIG_HOME', os.path.expanduser('~/.config'))
    return Path(xdg_config_home, 'hail', 'config.ini')


def get_user_config():
    global user_config
    if user_config is None:
        user_config = configparser.ConfigParser()
        config_file = get_user_config_path()
        os.makedirs(config_file.parent, exist_ok=True)
        # in older versions, the config file was accidentally named
        # config.yaml, if the new config does not exist, and the old
        # one does, silently rename it
        old_path = config_file.with_name('config.yaml')
        if old_path.exists() and not config_file.exists():
            old_path.rename(config_file)
        else:
            config_file.touch(exist_ok=True)
        user_config.read(config_file)
    return user_config
