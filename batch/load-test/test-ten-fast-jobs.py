import hailtop.batch as hb
import math
import argparse


parser = argparse.ArgumentParser(description='Load test the Batch service using for a given max scheduling rate')
parser.add_argument('msr', type=int, nargs=1, help='the maximum scheduling rate for the Batch Driver')
parser.add_argument('n', type=int, nargs=1, help='the max number of machines that can be provisioned for this batch')

args = parser.parse_args()
msr = args.msr
n = args.n

# Need to set hailctl remote_tmpdir
backend = hb.ServiceBackend('test')
b = hb.Batch(backend=backend, name='load-test')

# the number of quarter-core jobs that can be running on n machines (assuming each machines has 16 cores)
max_concurrent_quarter_core_jobs = 64 * n

# TODO: should the factor of 10 be user-specificable?
# this load test will require the driver to schedule 10 times as many jobs as can possibly be running at once
n_jobs = 10 * max_concurrent_quarter_core_jobs
for idx in range(n_jobs):
    sleep_time = math.ceil(max_concurrent_quarter_core_jobs / msr)
    j = b.new_job(name=f'job_{idx}')
    j.cpu('250m')
    j.command(f'sleep {sleep_time}')

b.run()
