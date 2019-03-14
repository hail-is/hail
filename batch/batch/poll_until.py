import math
import random
import time


def poll_until(p, max_polls=None):
    i = 0
    while True and (max_polls is None or i < max_polls):
        x = p()
        if x:
            return x
        j = random.randrange(math.floor(1.1 ** i))
        time.sleep(0.100 * j)
        # max 4.45s
        if i < 64:
            i = i + 1
    raise ValueError(f'poll_until: exceeded max polls: {i} {max_polls}')
