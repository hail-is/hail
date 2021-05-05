import importlib
import os
import sys
import re
import random
import argparse
import logging

from typing import Set
from shlex import quote as shq

from hailtop.utils import sync_check_shell
from hailtop import batch as hb

BENCHMARK_IMAGE = 'gcr.io/hail-vdc/base:latest'

log = logging.getLogger('submit')

GITHUB_COMMIT_REGEX = re.compile('(?P<owner>[^/]+)/(?P<name>[^:]+):(?P<sha>.+)')


class Repo:
    def __init__(self, owner, name):
        assert isinstance(owner, str)
        assert isinstance(name, str)
        self.owner = owner
        self.name = name
        self.url = f'https://github.com/{owner}/{name}.git'

    def __str__(self):
        return self.short_str()

    @staticmethod
    def from_short_str(s):
        pieces = s.split("/")
        assert len(pieces) == 2, f'{pieces} {s}'
        return Repo(pieces[0], pieces[1])

    def short_str(self):
        return f'{self.owner}/{self.name}'


class Commit:
    def __init__(self, repo: Repo, sha: str):
        self.repo = repo
        self.sha = sha

    @staticmethod
    def from_str(s):
        commit = GITHUB_COMMIT_REGEX.fullmatch(s).groupdict()
        assert commit
        repo = Repo(commit['owner'], commit['name'])
        sha = commit['sha']
        return Commit(repo, sha)

    def short_str(self):
        return f'{self.repo.owner}/{self.repo.name}:{self.sha}'

    def __str__(self):
        return self.short_str()

    def repo_dir(self):
        return f'/tmp/repo/{self.repo.owner}/{self.repo.name}'

    def checkout_script(self):
        return f'''
set -ex
mkdir -p { shq(self.repo_dir()) }
if [ ! -d { shq(self.repo_dir()) }/.git ]
then
    cd /
    rm -rf { shq(self.repo_dir()) }
    git clone { shq(self.repo.url) } { shq(self.repo_dir()) }
    cd { shq(self.repo_dir()) }
else
    cd { shq(self.repo_dir()) }
    git reset --hard
    git fetch -q origin
fi
pwd
git checkout {self.sha}
'''


def submit(hail_code: Commit, benchmark_code: Commit, test_names: Set[str], n_replicates: int, n_iters: int):

    sync_check_shell(benchmark_code.checkout_script())

    sys.path.insert(0, f'{benchmark_code.repo_dir()}/benchmark/python/benchmark_hail')

    importlib.invalidate_caches  # pylint: disable=pointless-statement
    from benchmark_hail.run.resources import all_resources  # pylint: disable=import-error, import-outside-toplevel
    from benchmark_hail.run.utils import list_benchmarks  # pylint: disable=import-error, import-outside-toplevel

    output_file = f'gs://hail-benchmarks-2/benchmark/{hail_code.sha}-{benchmark_code.sha}.json'

    b = hb.Batch(
        name=f'benchmark-{hail_code.sha}',
        backend=hb.ServiceBackend(billing_project='hail'),
        default_image=BENCHMARK_IMAGE,
        default_cpu='2',
        attributes={
            'output_file': output_file,
            'n_replicates': str(n_replicates),
            'n_iters': str(n_iters),
            'image': str(BENCHMARK_IMAGE),
            'hail_code': str(hail_code),
            'benchmark_code': str(benchmark_code),
        },
    )

    build_hail = b.new_job('build_hail_wheel')
    build_hail.command(
        f'''
 set -ex
 { hail_code.checkout_script() }
 cd hail
 time ./gradlew --version
 time make wheel
 time (cd python && zip -r hail.zip hail hailtop)
 (cd build/deploy/dist/ && tar -cvf wheel-container.tar hail-*-py3-none-any.whl)
 cp build/deploy/dist/hail-*-py3-none-any.whl {build_hail.wheel}
'''
    )

    build_benchmark = b.new_job('build_benchmark_wheel')
    build_benchmark.command(
        f'''
 set -ex
 {benchmark_code.checkout_script()}
 make -C hail python/hail/hail_pip_version
 export HAIL_VERSION=$(cat hail/python/hail/hail_pip_version)
 export HAIL_BENCHMARK_VERSION=$HAIL_VERSION
 cd benchmark/python/ && python3 setup.py -q bdist_wheel
 python3 -m pip -q install dist/benchmark_hail-$HAIL_VERSION-py3-none-any.whl
 cp dist/benchmark_hail-$HAIL_VERSION-py3-none-any.whl {build_benchmark.wheel}
'''
    )
    resource_jobs = {}
    for r in all_resources:
        j = b.new_job(f'create_resource_{r.name()}').cpu(4)
        j.command(f'mv {build_hail.wheel} hail--py3-none-any.whl')
        j.command('pip install hail--py3-none-any.whl')
        j.command(f'mv {build_benchmark.wheel} benchmark_hail-$HAIL_VERSION-py3-none-any.whl')
        j.command('pip install benchmark_hail-$HAIL_VERSION-py3-none-any.whl')
        j.command(f'hail-bench create-resources --data-dir benchmark-resources --group {r.name()}')
        j.command(f"time tar -cf {r.name()}.tar benchmark-resources/{r.name()} --exclude='*.crc'")
        j.command(f'ls -lh {r.name()}.tar')
        j.command(f'mv {r.name()}.tar {j.ofile}')
        resource_jobs[r] = j

    all_benchmarks = list_benchmarks()
    assert len(all_benchmarks) > 0

    all_output = []

    n_passed_filter = 0
    job_fs = []
    for benchmark in all_benchmarks:
        if benchmark.name in test_names:
            n_passed_filter += 1
            for replicate in range(n_replicates):
                job_fs.append((benchmark.name, replicate, benchmark.groups))

    log.info(
        f'generating {n_passed_filter} * {n_replicates} = {n_passed_filter * n_replicates} individual benchmark jobs'
    )

    random.shuffle(job_fs)
    for name, replicate, groups in job_fs:
        j = b.new_job(name=f'{name}_{replicate}')
        j.command(f'mv {build_hail.wheel} hail--py3-none-any.whl')
        j.command('pip install hail--py3-none-any.whl')
        j.command(f'mv {build_benchmark.wheel} benchmark_hail--py3-none-any.whl')
        j.command('pip install benchmark_hail--py3-none-any.whl')
        j.command('mkdir -p benchmark-resources')
        for resource_group in groups:
            resource_job = resource_jobs[resource_group]
            j.command(f'mv {resource_job.ofile} benchmark-resources/{resource_group.name()}.tar')
            j.command(f'time tar -xf benchmark-resources/{resource_group.name()}.tar')
        j.command(
            f'MKL_NUM_THREADS=1'
            f'OPENBLAS_NUM_THREADS=1'
            f'OMP_NUM_THREADS=1'
            f'VECLIB_MAXIMUM_THREADS=1'
            f'PYSPARK_SUBMIT_ARGS="--driver-memory 6G pyspark-shell" '
            f'hail-bench run -o {j.ofile} -n {n_iters} --data-dir benchmark-resources -t {name}'
        )
        all_output.append(j.ofile)

    combine_branch_factor = int(os.environ.get('BENCHMARK_BRANCH_FACTOR', 32))
    phase_i = 1
    while len(all_output) > combine_branch_factor:
        new_output = []

        job_i = 1
        i = 0
        while i < len(all_output):
            combine = b.new_job(f'combine_output_phase{phase_i}_job{job_i}')
            combine.command(f'mv {build_hail.wheel} hail--py3-none-any.whl')
            combine.command('pip install hail--py3-none-any.whl')
            combine.command(f'mv {build_benchmark.wheel} benchmark_hail--py3-none-any.whl')
            combine.command('pip install benchmark_hail--py3-none-any.whl')
            combine.command(
                f'hail-bench combine -o {combine.ofile} ' + ' '.join(all_output[i : i + combine_branch_factor])
            )
            new_output.append(combine.ofile)
            i += combine_branch_factor
            job_i += 1

        phase_i += 1
        all_output = new_output
    combine = b.new_job('final_combine_output')
    combine.command(f'mv {build_hail.wheel} hail--py3-none-any.whl')
    combine.command('pip install hail--py3-none-any.whl')
    combine.command(f'mv {build_benchmark.wheel} benchmark_hail--py3-none-any.whl')
    combine.command('pip install benchmark_hail--py3-none-any.whl')
    combine.command(f'hail-bench combine -o {combine.ofile} ' + ' '.join(all_output))
    combine.command(f'cat {combine.ofile}')

    log.info(f'writing output to {output_file}')

    b.write_output(combine.ofile, output_file)
    b.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--hail-source',
        type=str,
        help='owner/name:sha for where the Hail code should be checked out from.',
        required=True,
    )
    parser.add_argument(
        '--benchmark-source',
        type=str,
        help='owner/name:sha for where the Benchmark code should be checked out from.',
        required=True,
    )
    parser.add_argument('--replicates', type=int, default=1, help='Number of replicates.')
    parser.add_argument('--iters', type=int, default=1, help='Number of iterations.')
    parser.add_argument('--tests', type=str, help='Benchmark tests to run.', required=True)

    args = parser.parse_args()

    hail_code = Commit.from_str(args.hail_source)
    benchmark_code = Commit.from_str(args.benchmark_source)
    tests = set(args.tests.split(','))

    submit(
        hail_code=hail_code,
        benchmark_code=benchmark_code,
        test_names=tests,
        n_replicates=args.replicates,
        n_iters=args.iters,
    )
