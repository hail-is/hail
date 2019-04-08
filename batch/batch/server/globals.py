import datetime
import collections


def _log_path(id, task_name):
    return f'logs/job-{id}-{task_name}.log'


def _read_file(fname):
    with open(fname, 'r') as f:
        return f.read()


_recent_events = collections.deque(maxlen=50)


def get_recent_events():
    return _recent_events


def add_event(event):
    global _recent_events

    event['time'] = str(datetime.datetime.now())
    _recent_events.append(event)
