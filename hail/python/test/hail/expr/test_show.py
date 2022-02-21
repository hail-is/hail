from ..helpers import startTestHailContext, stopTestHailContext, fails_service_backend, skip_when_service_backend
import unittest

import hail as hl

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):
    @skip_when_service_backend('''intermittent worker failure:
>       mt.sample_idx.show()

Caused by: java.lang.AssertionError: assertion failed
	at scala.Predef$.assert(Predef.scala:208)
	at is.hail.io.BlockingInputBuffer.ensure(InputBuffers.scala:389)
	at is.hail.io.BlockingInputBuffer.readInt(InputBuffers.scala:412)
	at __C1436collect_distributed_array.__m1444INPLACE_DECODE_r_binary_TO_r_binary(Unknown Source)
	at __C1436collect_distributed_array.__m1443INPLACE_DECODE_r_struct_of_r_binaryEND_TO_r_tuple_of_r_binaryEND(Unknown Source)
	at __C1436collect_distributed_array.__m1442INPLACE_DECODE_r_struct_of_r_struct_of_r_binaryENDEND_TO_r_struct_of_r_tuple_of_r_binaryENDEND(Unknown Source)
	at __C1436collect_distributed_array.__m1441DECODE_r_struct_of_r_struct_of_r_struct_of_r_binaryENDENDEND_TO_SBaseStructPointer(Unknown Source)
	at __C1436collect_distributed_array.apply(Unknown Source)
	at __C1436collect_distributed_array.apply(Unknown Source)
	at is.hail.backend.BackendUtils.$anonfun$collectDArray$2(BackendUtils.scala:31)
	at is.hail.utils.package$.using(package.scala:627)
	at is.hail.annotations.RegionPool.scopedRegion(RegionPool.scala:144)
	at is.hail.backend.BackendUtils.$anonfun$collectDArray$1(BackendUtils.scala:30)
	at is.hail.backend.service.Worker$.main(Worker.scala:120)
	at is.hail.backend.service.Worker.main(Worker.scala)
	... 12 more''')
    def test(self):
        mt = hl.balding_nichols_model(3, 10, 10)
        t = mt.rows()
        mt.GT.show()
        mt.locus.show()
        mt.af.show()
        mt.pop.show()
        mt.sample_idx.show()
        mt.bn.show()
        mt.bn.fst.show()
        mt.GT.n_alt_alleles().show()
        (mt.GT.n_alt_alleles() * mt.GT.n_alt_alleles()).show()
        (mt.af * mt.GT.n_alt_alleles()).show()
        t.af.show()
        (t.af * 3).show()

    def test_show_negative(self):
        hl.utils.range_table(5).show(-1)
