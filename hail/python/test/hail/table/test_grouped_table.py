import unittest

import hail as hl
from ..helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class GroupedTableTests(unittest.TestCase):
    @skip_when_service_backend('''intermittent worker failure:
>       self.assertTrue(result._same(expected))

Caused by: is.hail.utils.HailException: Premature end of file: expected 4 bytes, found 0
	at is.hail.utils.ErrorHandling.fatal(ErrorHandling.scala:11)
	at is.hail.utils.ErrorHandling.fatal$(ErrorHandling.scala:11)
	at is.hail.utils.package$.fatal(package.scala:77)
	at is.hail.utils.richUtils.RichInputStream$.readFully$extension1(RichInputStream.scala:13)
	at is.hail.io.StreamBlockInputBuffer.readBlock(InputBuffers.scala:546)
	at is.hail.io.BlockingInputBuffer.readBlock(InputBuffers.scala:382)
	at is.hail.io.BlockingInputBuffer.ensure(InputBuffers.scala:388)
	at is.hail.io.BlockingInputBuffer.readInt(InputBuffers.scala:412)
	at __C8457collect_distributed_array.__m8461INPLACE_DECODE_r_int32_TO_r_int32(Unknown Source)
	at __C8457collect_distributed_array.__m8460INPLACE_DECODE_r_struct_of_r_int32ANDr_int32END_TO_r_struct_of_r_int32ANDr_int32END(Unknown Source)
	at __C8457collect_distributed_array.__m8459INPLACE_DECODE_r_struct_of_r_struct_of_r_int32ANDr_int32ENDANDr_struct_of_r_binaryENDEND_TO_r_struct_of_r_struct_of_r_int32ANDr_int32ENDANDr_tuple_of_r_binaryENDEND(Unknown Source)
	at __C8457collect_distributed_array.__m8458DECODE_r_struct_of_r_struct_of_r_struct_of_r_int32ANDr_int32ENDANDr_struct_of_r_binaryENDENDEND_TO_SBaseStructPointer(Unknown Source)
	at __C8457collect_distributed_array.apply(Unknown Source)
	at __C8457collect_distributed_array.apply(Unknown Source)
	at is.hail.backend.BackendUtils.$anonfun$collectDArray$2(BackendUtils.scala:31)
	at is.hail.utils.package$.using(package.scala:627)
	at is.hail.annotations.RegionPool.scopedRegion(RegionPool.scala:144)
	at is.hail.backend.BackendUtils.$anonfun$collectDArray$1(BackendUtils.scala:30)
	at is.hail.backend.service.Worker$.main(Worker.scala:120)
	at is.hail.backend.service.Worker.main(Worker.scala)
	... 11 more''')
    def test_aggregate_by(self):
        ht = hl.utils.range_table(4)
        ht = ht.annotate(foo=0, group=ht.idx < 2, bar='hello').annotate_globals(glob=5)
        grouped = ht.group_by(ht.group)
        result = grouped.aggregate(sum=hl.agg.sum(ht.idx + ht.glob) + ht.glob - 15, max=hl.agg.max(ht.idx))

        expected = (
            hl.Table.parallelize(
                [{'group': True, 'sum': 1, 'max': 1},
                 {'group': False, 'sum': 5, 'max': 3}],
                hl.tstruct(group=hl.tbool, sum=hl.tint64, max=hl.tint32)
            ).annotate_globals(glob=5).key_by('group')
        )

        self.assertTrue(result._same(expected))

        with self.assertRaises(ValueError):
            grouped.aggregate(group=hl.agg.sum(ht.idx))

    def test_aggregate_by_with_joins(self):
        ht = hl.utils.range_table(4)
        ht2 = hl.utils.range_table(4)
        ht2 = ht2.annotate(idx2=ht2.idx)

        ht = ht.annotate_globals(glob=5)
        grouped = ht.group_by(group=ht2[ht.idx].idx2 < 2)
        result = grouped.aggregate(sum=hl.agg.sum(ht2[ht.idx].idx2 + ht.glob) + ht.glob - 15,
                                   max=hl.agg.max(ht2[ht.idx].idx2))

        expected = (
            hl.Table.parallelize(
                [{'group': True, 'sum': 1, 'max': 1},
                 {'group': False, 'sum': 5, 'max': 3}],
                hl.tstruct(group=hl.tbool, sum=hl.tint64, max=hl.tint32)
            ).annotate_globals(glob=5).key_by('group')
        )

        self.assertTrue(result._same(expected))

    def test_issue_2446_takeby(self):
        t = hl.utils.range_table(10)
        result = t.group_by(foo=5).aggregate(x=hl.agg.take(t.idx, 3, ordering=t.idx))
        self.assertTrue(result.collect()[0].x == [0, 1, 2])
