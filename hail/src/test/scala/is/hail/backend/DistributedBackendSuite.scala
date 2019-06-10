package is.hail.backend

import is.hail.{DistributedSuite, SparkSuite}
import is.hail.expr.ir._
import is.hail.utils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class DistributedBackendSuite extends DistributedSuite {
  @Test def testPipeline() {
    val tr = TableRange(5000, 50)
    val mapped = TableExplode(TableMapRows(tr,
      InsertFields(Ref("row", tr.typ.rowType),
        FastIndexedSeq("foo" -> Str("foo"),
          "range" -> ArrayRange(0, GetField(Ref("row", tr.typ.rowType), "idx"), 1)))),
      Array("range"))
    val testIR = TableFilter(mapped, GetField(Ref("row", mapped.typ.rowType), "range").ceq(3))

    val (v, t) = hc.backend.execute(TableCollect(testIR), optimize = false)
    assert(v == Row(Array.tabulate(5000)(i => Row(i, "foo", 3)).filter(_.getAs[Int](0) > 3).toFastIndexedSeq, Row()))
    t.value.foreach { case (stage, timing) =>
      println(s"Time taken for $stage: ${ timing("readable") }")
    }
  }
}
