package is.hail.expr.ir

import is.hail.SparkSuite
import is.hail.TestUtils.assertEvalsTo
import is.hail.expr.ir.TestUtils.IRAggCount
import is.hail.table.{Ascending, SortField}
import is.hail.utils.FastIndexedSeq
import org.testng.annotations.Test

class SimplifySuite extends SparkSuite {

  @Test def testRepartitionableMapUpdatesForUpstreamOptimizations() {
    val range = TableUnkey(TableRange(10, 3))
    val simplifiableIR =
      If(True(),
        GetField(Ref("row", range.typ.rowType), "idx").ceq(0),
        False())
    val checksRepartitioningIR =
      TableFilter(
        TableOrderBy(range, FastIndexedSeq(SortField("idx", Ascending))),
        simplifiableIR)

    assertEvalsTo(TableAggregate(checksRepartitioningIR, IRAggCount), 1L)
  }

}
