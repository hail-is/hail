from hailtop import pipeline
import random

def flip(p):
    return random.random() <= p

def stress():
    p = pipeline.Pipeline(
        name='stress',
        backend=pipeline.BatchBackend(billing_project='hail'),
        default_image='ubuntu:18.04')

    for i in range(100):
        t = (p
             .new_task(name=f'parent_{i}'))
        d = random.choice(range(4))
        if flip(0.2):
            t.command(f'sleep {d}; exit 1')
        else:
            t.command(f'sleep {d}; echo parent {i}')
        for j in range(10):
            d = random.choice(range(4))
            c = (p
                 .new_task(name=f'child_{i}_{j}')
                 .command(f'sleep {d}; echo child {i} {j}'))
            c.depends_on(t)
            if flip(0.2):
                c._always_run = True

    p.run(open=False, wait=False)

if __name__ == "__main__":
    stress()
