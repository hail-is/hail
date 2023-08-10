import os
import random

import hailtop.batch as hb

DOCKER_ROOT_IMAGE = os.environ['DOCKER_ROOT_IMAGE']


def flip(p):
    return random.random() <= p


def stress():
    b = hb.Batch(name='stress', backend=hb.ServiceBackend(billing_project='hail'), default_image=DOCKER_ROOT_IMAGE)

    for i in range(100):
        j = b.new_job(name=f'parent_{i}')
        d = random.choice(range(4))
        if flip(0.2):
            j.command(f'sleep {d}; exit 1')
        else:
            j.command(f'sleep {d}; echo parent {i}')

        for k in range(10):
            d = random.choice(range(4))
            c = b.new_job(name=f'child_{i}_{k}').command(f'sleep {d}; echo child {i} {k}')
            c.depends_on(j)
            if flip(0.1):
                c._always_run = True
            if flip(0.01):
                c._machine_type = 'n1-standard-1'
                if flip(0.5):
                    c.spot(False)

    b.run(open=False, wait=False)


if __name__ == "__main__":
    stress()
