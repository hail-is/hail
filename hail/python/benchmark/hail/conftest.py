import json
import logging
import os
import platform
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal

import pytest
from _pytest.runner import pytest_runtest_protocol as runner_runtest_protocol

import hail as hl
from benchmark.hail.fixtures import (
    HG00187,
    HG00190,
    HG00308,
    HG00313,
    HG00320,
    HG00323,
    HG00339,
    HG00373,
    HG00524,
    HG00553,
    HG00590,
    HG00592,
    HG00637,
    HG01058,
    HG01356,
    HG01357,
    HG01377,
    HG01378,
    HG01383,
    HG01384,
    HG02188,
    HG02223,
    HG02224,
    HG02230,
    HG03088,
    HG03363,
    HG03578,
    HG03805,
    HG03833,
    HG04001,
    HG04158,
    NA11830,
    NA12249,
    NA18507,
    NA18530,
    NA18534,
    NA18609,
    NA18613,
    NA18618,
    NA18969,
    NA19456,
    NA19747,
    NA20317,
    NA20760,
    NA20769,
    NA20796,
    NA20802,
    NA21099,
    NA21122,
    NA21123,
    balding_nichols_5k_5k,
    chr22_gvcfs,
    empty_gvcf,
    gnomad_dp_sim,
    many_ints_ht,
    many_ints_tsv,
    many_partitions_ht,
    many_strings_ht,
    many_strings_tsv,
    onekg_chr22,
    profile25_mt,
    profile25_vcf,
    random_doubles_mt,
    random_doubles_tsv,
    resource_dir,
    sim_ukb_bgen,
    sim_ukb_sample,
    single_gvcf,
)
from benchmark.hail.utils import (
    get_peak_task_memory,
    init_hail_for_benchmarks,
    run_with_timeout,
    select,
)
from hail.utils.java import Env


@contextmanager
def init_hail(run_config):
    init_hail_for_benchmarks(run_config)
    try:
        yield
    finally:
        hl.stop()


results = pytest.StashKey[Dict[str, Dict[str, Any]]]()

# item stash
start = pytest.StashKey[datetime]()
end = pytest.StashKey[datetime]()
iteration = pytest.StashKey[int]()
runs = pytest.StashKey[List[Dict[str, Any]]]()

# used internally
context = pytest.StashKey[Literal['burn_in', 'benchmark']]()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    run_config = item.session.config.run_config
    # Initialise hail before running every benchmark for two reasons:
    # - each benchmark runs in a clean hail session
    # - our means of getting max task memory is quite crude (regex on logs)
    #   and a fresh session provides a new log
    with init_hail(run_config):
        if run_config.iterations is not None:
            burn_in_iterations = 0
            iterations = run_config.iterations
            logging.info(
                msg=(
                    f'Picked up iterations override. Config: '
                    f'burn_in_iterations: {burn_in_iterations}, '
                    f'iterations: {iterations}.'
                )
            )

        else:
            burn_in_iterations, iterations = select(
                ['burn_in_iterations', 'iterations'], **(item.get_closest_marker('benchmark').kwargs)
            )

        s = item.stash
        s[start] = datetime.now(timezone.utc).isoformat()
        s[runs] = []
        s[end] = None

        logging.info(
            msg=(
                f'Executing "{item.nodeid}" with '
                f'{burn_in_iterations} burn in iterations and '
                f'{iterations} timed iterations.'
            )
        )

        s[context] = 'burn_in'
        for k in range(burn_in_iterations):
            s[iteration] = k
            # `nextitem` is used to determine which fixtures need to be torn-down
            # after the test finishes. For example, if `nextitem` is `None`, then
            # all fixtures (including session fixtures) will be finalised.
            # Since we're invoking this benchmark repeatedly, we want to tear-down
            # function/method level fixtures only, leaving module and session
            # fixtures in place; `item.parent` is one such `Item` that represents this.
            runner_runtest_protocol(item, nextitem=item.parent)

        s[context] = 'benchmark'
        total_iterations = burn_in_iterations + iterations
        for k in range(burn_in_iterations, total_iterations):
            s[iteration] = k
            # on the final iteration, perform the required teardown for the test
            is_final_iteration = k == total_iterations - 1
            runner_runtest_protocol(item, nextitem=nextitem if is_final_iteration else item.parent)

        s[end] = datetime.now(timezone.utc).isoformat()

    # prevent other plugins running that might invoke the benchmark again
    return True


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    with run_with_timeout(
        pyfuncitem.config.run_config,
        pyfuncitem.obj,
        **{arg: pyfuncitem.funcargs[arg] for arg in pyfuncitem._fixtureinfo.argnames},
    ) as (time, timed_out, traceback):
        s = pyfuncitem.stash

        is_burn_in = s[context] == 'burn_in'

        s[runs].append({
            'iteration': s[iteration],
            'time': time,
            'is_burn_in': is_burn_in,
            'timed_out': timed_out,
            'failure': traceback,
            'task_memory': get_peak_task_memory(Env.hc()._log),
        })

        logging.info(f'{"(burn in) " if is_burn_in else ""}iteration {s[iteration]}, time: {time}s')

    # prevent other plugins running that might invoke the benchmark again
    return True


@contextmanager
def open_file_or_stdout(file):
    if file is None:
        yield sys.stdout
    else:
        with open(file, 'w') as f:
            yield f


@pytest.hookimpl
def pytest_sessionfinish(session):
    if not session.config.option.collectonly:
        run_config = session.config.run_config

        meta = {
            'uname': platform.uname()._asdict(),
            'version': hl.__version__,
            **({'batch_id': batch} if (batch := os.getenv('HAIL_BATCH_ID')) else {}),
            **({'job_id': job} if (job := os.getenv('HAIL_JOB_ID')) else {}),
            **({'trial': trial} if (trial := os.getenv('BENCHMARK_TRIAL_ID')) else {}),
        }

        now = datetime.now(timezone.utc).isoformat()
        with open_file_or_stdout(run_config.output) as out:
            for item in session.items:
                path, _, name = item.location
                json.dump(
                    {
                        'path': path,
                        'name': name,
                        **meta,
                        'start': item.stash[start],
                        'end': item.stash.get(end, now),
                        'runs': item.stash[runs],
                    },
                    out,
                )
                out.write('\n')


# make fixtures discoverable to `pytest --fixtures` as well as all tests
# within benchmark/hail without explict import.
__all__ = [
    'HG00187',
    'HG00190',
    'HG00308',
    'HG00313',
    'HG00320',
    'HG00323',
    'HG00339',
    'HG00373',
    'HG00524',
    'HG00553',
    'HG00590',
    'HG00592',
    'HG00637',
    'HG01058',
    'HG01356',
    'HG01357',
    'HG01377',
    'HG01378',
    'HG01383',
    'HG01384',
    'HG02188',
    'HG02223',
    'HG02224',
    'HG02230',
    'HG03088',
    'HG03363',
    'HG03578',
    'HG03805',
    'HG03833',
    'HG04001',
    'HG04158',
    'NA11830',
    'NA12249',
    'NA18507',
    'NA18530',
    'NA18534',
    'NA18609',
    'NA18613',
    'NA18618',
    'NA18969',
    'NA19456',
    'NA19747',
    'NA20317',
    'NA20760',
    'NA20769',
    'NA20796',
    'NA20802',
    'NA21099',
    'NA21122',
    'NA21123',
    'balding_nichols_5k_5k',
    'chr22_gvcfs',
    'empty_gvcf',
    'gnomad_dp_sim',
    'many_ints_ht',
    'many_ints_tsv',
    'many_partitions_ht',
    'many_strings_ht',
    'many_strings_tsv',
    'onekg_chr22',
    'profile25_mt',
    'profile25_vcf',
    'random_doubles_mt',
    'random_doubles_tsv',
    'resource_dir',
    'sim_ukb_bgen',
    'sim_ukb_sample',
    'single_gvcf',
]
