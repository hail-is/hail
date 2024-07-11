import json
from pathlib import Path
from typing import Any, Generator, List, Sequence

import hail as hl
from benchmark.tools import maybe, prune

class __types:
    titeration = hl.tstruct(
        is_burn_in=hl.tbool,  # ignore for a/b testing
        time=hl.tfloat,  # seconds
        failure=hl.tstr,  # exception message dumped to a string, optional
        timed_out=hl.tbool,  # whether or not the failure was caused by a timeout
        task_memory=hl.tfloat,  # don't think this works yet sadly.
    )

    tinstance = hl.tstruct(
        path=hl.tstr,
        name=hl.tstr,
        version=hl.tstr,
        platform=hl.tstr,
        python=hl.tstr,
        batch_id=hl.tint,
        job_id=hl.tint,
        attempt_id=hl.tstr,
        start=hl.tstr,
        end=hl.tstr,
        **titeration,
    )


def __write_tsv_row(os, row: Sequence[str]) -> None:
    if len(row) > 0:
        os.write('\t'.join(row))
        os.write('\n')


def dump_tsv(jsonl: Path, tsv: Path) -> None:
    def explode(instance: dict) -> Generator[List[Any], Any, None]:
        for iteration in instance['iterations']:
            flattened = prune({**instance, **iteration, 'failure': maybe(json.dumps, iteration.get('failure'))})
            yield [maybe(str, flattened.get(f), 'NA') for f in __types.tinstance]

    with (
        jsonl.open(encoding='utf-8') as in_,
        tsv.open('w', encoding='utf-8') as out,
    ):
        __write_tsv_row(out, [n for n in __types.tinstance])
        for line in in_:
            trial = json.loads(line)
            for row in explode(trial):
                __write_tsv_row(out, row)


def import_timings(timings_tsv: Path) -> hl.Table:
    ht = hl.import_table(str(timings_tsv), types=__types.tinstance)
    instance_key = [t for t in __types.tinstance.fields if t not in set(__types.titeration.fields)]

    ht = ht.group_by(*instance_key).aggregate(
        iterations=hl.agg.collect(ht.row_value.select(*__types.titeration)),
    )

    ht = ht.select(
        instance=hl.struct(
            batch_id=ht.batch_id,
            job_id=ht.job_id,
            attempt_id=ht.attempt_id,
            start=ht.start,
            end=ht.end,
            iterations=hl.filter(
                lambda t: (
                    hl.is_missing(t.failure)
                    | (hl.is_defined(t.failure) & (hl.len(t.failure) == 0))
                    | ~t.timed_out
                    | ~t.is_burn_in
                ),
                ht.iterations,
            ),
        ),
    )

    return ht.group_by(ht.path, ht.name, ht.version).aggregate(
        instances=hl.sorted(
            hl.agg.collect(ht.instance),
            key=lambda i: (i.job_id, i.start),
        )
    )
