import argparse
import uuid
import subprocess as sp
from shlex import quote as shq


def run(args_):
    parser = argparse.ArgumentParser()

    parser.add_argument('name')

    parser.add_argument('--cleanup-tables',
                        type=str,
                        required=False,
                        default='../batch2/delete-batch-tables.sql')

    parser.add_argument('--setup-tables',
                        type=str,
                        required=False,
                        default='../batch2/create-batch-tables.sql')

    parser.add_argument('--timeout',
                        type=int,
                        required=False,
                        default=1200)

    parser.add_argument('--log-path',
                        type=str,
                        required=False,
                        default='benchmark.log')

    parser.add_argument('--dry-run',
                        action='store_true')

    parser.add_argument('--parallelism',
                        type=int,
                        required=False,
                        default=1,
                        help='Number of chunks to insert in parallel.')

    parser.add_argument('--n-replicates',
                        type=int,
                        required=False,
                        default=10,
                        help='Number of replicates to run of each test.')

    parser.add_argument('--batch-sizes',
                        type=str,
                        required=False,
                        default='1000,10000,100000',
                        help='Number of jobs per batch to test.')

    parser.add_argument('--chunk-size',
                        type=int,
                        required=False,
                        default=1000,
                        help='Number of jobs inserted at a time.')

    args = parser.parse_args(args_)

    token = uuid.uuid4().hex[:8]

    pod_name = f'db-benchmark-{token}'

    image = f'gcr.io/hail-vdc/db-benchmark:{token}'

    script = f"""
set -ex

# copy sql scripts
cp {args.setup_tables} create-tables.sql
cp {args.cleanup_tables} delete-tables.sql

# make docker image with benchmark.py and sql scripts
make -C ../docker build
docker build -t {shq(image)} -f Dockerfile --cache-from service-base .
docker push {image}

python3 ../ci/jinja2_render.py '{{ \
"pod_name": "{pod_name}", \
"image": "{image}", \
"db_secret_name": "{args.name}-sql-config", \
"n_replicates": {args.n_replicates}, \
"batch_sizes": "{args.batch_sizes}", \
"parallelism": {args.parallelism}, \
"chunk_size": {args.chunk_size} \
}}' pod.yaml pod.yaml.out

kubectl -n db-benchmark apply -f pod.yaml.out
set +e
while [[ $(kubectl get pods -n db-benchmark {pod_name} -o 'jsonpath={{..status.conditions[?(@.type=="PodScheduled")].status}}') != "True" ]]; do echo "waiting for pod" && sleep 10; done  && \
  USE_KUBE_CONFIG=1 python3 ../ci/wait-for.py {args.timeout} db-benchmark Pod {pod_name}
kubectl -n db-benchmark --tail=999999999 logs {pod_name} > {args.log_path}
"""

    cleanup = f"""
kubectl -n db-benchmark delete pod {pod_name}
rm create-tables.sql delete-tables.sql pod.yaml.out
gcloud -q container images untag {shq(image)}
"""

    if args.dry_run:
        print(script)
    else:
        try:
            print(f'running benchmark; view pod logs with "kubectl -n db-benchmark --tail=999999999 logs {pod_name}"')
            sp.check_output(script, shell=True, stderr=sp.STDOUT)
            print(f'finished running benchmark')
            print(f'results available at {args.log_path}')
        except sp.CalledProcessError as e:
            print(e)
            print(e.output)
            raise
        finally:
            sp.check_output(cleanup, shell=True)
