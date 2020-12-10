import hailtop.batch as hb
import random


DOCKER_ROOT_IMAGE = 'gcr.io/hail-vdc/ubuntu:18.04'


worker_types = ['standard', 'highcpu', 'highmem']


def flip(p):
    return random.random() <= p


def random_worker_type():
    return random.choice(worker_types)


def stress():
    b = hb.Batch(
        name='stress',
        backend=hb.ServiceBackend(billing_project='hail'),
        default_image=DOCKER_ROOT_IMAGE)

    for i in range(100):
        j = (b
             .new_job(name=f'parent_{i}')
             .worker_type(random_worker_type()))
        d = random.choice(range(4))
        if flip(0.2):
            j.command(f'sleep {d}; exit 1')
        else:
            j.command(f'sleep {d}; echo parent {i}')

        for k in range(10):
            d = random.choice(range(4))
            c = (b
                 .new_job(name=f'child_{i}_{k}')
                 .worker_type(random_worker_type())
                 .command(f'sleep {d}; echo child {i} {k}'))
            c.depends_on(j)
            if flip(0.2):
                c._always_run = True

    b.run(open=False, wait=False)


if __name__ == "__main__":
    stress()
