import time

def time_msecs():
    return int(time.time() * 1000 + 0.5)

def time_msecs_str(t):
    return datetime.datetime.utcfromtimestamp(t / 1000).strftime(
        '%Y-%m-%dT%H:%M:%SZ')
