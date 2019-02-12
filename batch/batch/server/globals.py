import datetime
import collections

pod_name_job = {}
job_id_job = {}
batch_id_batch = {}


def _log_path(id):
    return f'logs/job-{id}.log'


def _read_file(fname):
    with open(fname, 'r') as f:
        return f.read()


_counter = 0


def max_id():
    return _counter


def next_id():
    global _counter

    _counter = _counter + 1
    return _counter


_recent_events = collections.deque(maxlen=50)


def get_recent_events():
    return _recent_events


def add_event(event):
    global _recent_events

    event['time'] = str(datetime.datetime.now())
    _recent_events.append(event)
