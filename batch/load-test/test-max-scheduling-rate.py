import hailtop.batch as hb
import math
import argparse


parser = argparse.ArgumentParser(description='Load test the Batch service using for a given max scheduling rate')
parser.add_argument('msr', type=int, nargs=1, help='the maximum scheduling rate for the Batch Driver (in jobs/sec)')
parser.add_argument('n', type=int, nargs=1, help='the max number of machines that can be provisioned for this batch')
parser.add_argument('duration', type=int, nargs=1, help='the duration of the load test (in sec)')

args = parser.parse_args()
msr = args.msr
n = args.n
duration = args.duration

# Need to set hailctl remote_tmpdir
backend = hb.ServiceBackend('test')
b = hb.Batch(backend=backend, name='load-test')

# the number of quarter-core jobs that can be running on n machines (assuming each machines has 16 cores)
max_concurrent_quarter_core_jobs = 64 * n
# the amount of time for which each job sleeps
sleep_time = math.floor(max_concurrent_quarter_core_jobs / msr)
# the number of jobs is the max scheduling rate times the scheduling duration (which is sleep_time less
# than the total duration since the batch completes sleep_time seconds after the last job is scheduled)
n_jobs = msr * (duration - sleep_time)

for idx in range(n_jobs):
    j = b.new_job(name=f'job_{idx}')
    j.cpu('250m')
    j.command(f'sleep {sleep_time}')

b.run()
