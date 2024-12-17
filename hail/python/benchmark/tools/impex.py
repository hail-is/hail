import json
from pathlib import Path
from typing import Any, Generator, List, Sequence

import hail as hl
from benchmark.tools import maybe, prune


class __types:
    trun = hl.tstruct(
        iteration=hl.tint,  # 0-based
        is_burn_in=hl.tbool,  # ignore for a/b testing
        time=hl.tfloat,  # seconds
        failure=hl.tstr,  # exception message dumped to a string, optional
        timed_out=hl.tbool,  # whether or not the failure was caused by a timeout
        task_memory=hl.tfloat,  # don't think this works yet sadly.
    )

    ttrial = hl.tstruct(
        path=hl.tstr,
        name=hl.tstr,
        version=hl.tstr,
        uname=hl.tdict(hl.tstr, hl.tstr),
        batch_id=hl.tint,
        job_id=hl.tint,
        trial=hl.tint,
        attempt_id=hl.tstr,
        start=hl.tstr,
        end=hl.tstr,
        **trun,
    )


def __write_tsv_row(os, row: Sequence[str]) -> None:
    if len(row) > 0:
        os.write('\t'.join(row))
        os.write('\n')


def dump_tsv(jsonl: Path, tsv: Path) -> None:
    def explode(trial: dict) -> Generator[List[Any], Any, None]:
        trial['uname'] = json.dumps(trial['uname'])
        for run in trial['runs']:
            flattened = prune({**trial, **run, 'failure': maybe(json.dumps, run.get('failure')), 'runs': None})
            yield [maybe(str, flattened.get(f), 'NA') for f in __types.ttrial]

    with (
        jsonl.open(encoding='utf-8') as in_,
        tsv.open('w', encoding='utf-8') as out,
    ):
        __write_tsv_row(out, [n for n in __types.ttrial])
        for line in in_:
            trial = json.loads(line)
            for row in explode(trial):
                __write_tsv_row(out, row)


def import_timings(timings_tsv: Path) -> hl.Table:
    ht = hl.import_table(str(timings_tsv), types=__types.ttrial)
    trial_key = [t for t in __types.ttrial.fields if t not in set(('uname', *__types.trun.fields))]
    ht = ht.group_by(*trial_key).aggregate(
        runs=hl.sorted(
            hl.agg.collect(ht.row_value.select(*__types.trun)),
            lambda t: t.iteration,
        ),
    )

    # Rename terms to be consistent with that of Laaber et al.:
    # - "trial" (ie batch job) -> "instance"
    # - "run"   (benchmark invocation) -> "trial"
    #
    # Note we don't run benchmarks multiple times per trial as these are
    # "macro"-benchmarks. This is one area where we differ from Laaber at al.
    ht = ht.select(
        instance=hl.struct(
            instance=ht.trial,
            batch_id=ht.batch_id,
            job_id=ht.job_id,
            attempt_id=ht.attempt_id,
            start=ht.start,
            end=ht.end,
            trials=hl.filter(
                lambda t: (
                    hl.is_missing(t.failure)
                    | (hl.is_defined(t.failure) & (hl.len(t.failure) == 0))
                    | ~t.timed_out
                    | ~t.is_burn_in
                ),
                ht.runs,
            ),
        ),
    )

    return ht.group_by(ht.path, ht.name, ht.version).aggregate(
        instances=hl.sorted(
            hl.agg.collect(ht.instance),
            key=lambda i: (i.instance, i.attempt_id),
        )
    )
