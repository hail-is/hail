import json
import tempfile
from os import PathLike
from pathlib import Path
from typing import Any, Generator, List, Sequence

import hail as hl
from benchmark.tools import maybe, prune


class __tflattened:
    titeration = hl.tstruct(
        is_burn_in=hl.tbool,  # ignore for a/b testing
        time=hl.tfloat,  # seconds
        failure=hl.tstr,  # exception message dumped to a string, optional
        timed_out=hl.tbool,  # whether the failure was caused by a timeout
        task_memory=hl.tfloat,  # doesn't work yet sadly.
    )

    tinstance = hl.tstruct(
        batch_id=hl.tint,
        job_id=hl.tint,
        attempt_id=hl.tstr,
        start=hl.tstr,
        end=hl.tstr,
        **titeration,
    )

    trow = hl.tstruct(
        path=hl.tstr,
        name=hl.tstr,
        version=hl.tstr,
        platform=hl.tstr,
        python=hl.tstr,
        **tinstance,
    )


def __flatten_to_tsv(jsonl: PathLike | str, tsv: PathLike | str) -> None:
    def __write_tsv_row(os, row: Sequence[str]) -> None:
        if len(row) > 0:
            os.write('\t'.join(row))
            os.write('\n')

    def flatten(instance: dict) -> Generator[List[Any], Any, None]:
        for iteration in instance['iterations']:
            flattened = prune({**instance, **iteration, 'failure': maybe(json.dumps, iteration.get('failure'))})
            yield [maybe(str, flattened.get(f), 'NA') for f in __tflattened.trow]

    with (
        Path(jsonl).open(encoding='utf-8') as in_,
        Path(tsv).open('w', encoding='utf-8') as out,
    ):
        __write_tsv_row(out, list(__tflattened.trow))
        for line in in_:
            instance = json.loads(line)
            for row in flatten(instance):
                __write_tsv_row(out, row)


def import_benchmarks(jsonl: PathLike | str, tmpdir: PathLike | str = tempfile.gettempdir()) -> hl.Table:
    with hl.TemporaryFilename(dir=str(tmpdir)) as tsvfile:
        # Parallelizing json is really, really slow - flatten and dump jsonl to tsv then read.
        __flatten_to_tsv(jsonl, tsvfile)

        ht = hl.import_table(str(tsvfile), types=__tflattened.trow)

        # collect iterations
        key = [t for t in __tflattened.trow.fields if t not in set(__tflattened.titeration.fields)]
        ht = ht.group_by(*key).aggregate(
            iterations=hl.agg.collect(ht.row_value.select(*__tflattened.titeration)),
        )

        # collect instances
        key = [t for t in ht.key.dtype.fields if t not in set(__tflattened.tinstance.fields)]
        ht = ht.group_by(*key).aggregate(
            instances=hl.agg.collect(
                hl.struct(
                    **ht.key.drop(*key),
                    **ht.row_value,
                ),
            ),
        )

        return ht.checkpoint(str(Path(tmpdir) / f'{Path(jsonl).stem}.ht'))
