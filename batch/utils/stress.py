from hailtop.batch import hb
import random


def flip(p):
    return random.random() <= p


def stress():
    b = hb.Batch(
        name='stress',
        backend=hb.ServiceBackend(billing_project='hail'),
        default_image='ubuntu:18.04')

    for i in range(100):
        j = (b
             .new_job(name=f'parent_{i}'))
        d = random.choice(range(4))
        if flip(0.2):
            j.command(f'sleep {d}; exit 1')
        else:
            j.command(f'sleep {d}; echo parent {i}')
        for j in range(10):
            d = random.choice(range(4))
            c = (b
                 .new_job(name=f'child_{i}_{j}')
                 .command(f'sleep {d}; echo child {i} {j}'))
            c.depends_on(j)
            if flip(0.2):
                c._always_run = True

    b.run(open=False, wait=False)


if __name__ == "__main__":
    stress()
