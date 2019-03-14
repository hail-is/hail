package is.hail.expr.ir

import is.hail.SparkSuite
import is.hail.TestUtils.assertEvalsTo
import is.hail.expr.ir.TestUtils.IRAggCount
import is.hail.expr.types.virtual.{TArray, TFloat64, TInt32, TStruct}
import is.hail.table.{Ascending, SortField}
import is.hail.utils.FastIndexedSeq
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class SimplifySuite extends SparkSuite {
  @Test def testTableMultiWayZipJoinGlobalsRewrite() {
    hc
    val tmwzj = TableGetGlobals(TableMultiWayZipJoin(
      Array(TableRange(10, 10),
        TableRange(10, 10),
        TableRange(10, 10)),
      "rowField",
      "globalField"))
    assertEvalsTo(tmwzj, Row(FastIndexedSeq(Row(), Row(), Row())))
  }

  @Test def testRepartitionableMapUpdatesForUpstreamOptimizations() {
    hc
    val range = TableKeyBy(TableRange(10, 3), FastIndexedSeq())
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

  lazy val base = Literal(TStruct("1" -> TInt32(), "2" -> TInt32()), Row(1,2))
  @Test def testInsertFieldsRewriteRules() {
    val ir1 = InsertFields(InsertFields(base, Seq("1" -> I32(2)), None), Seq("1" -> I32(3)), None)
    assert(Simplify(ir1) == InsertFields(base, Seq("1" -> I32(3)), None))

    val ir2 = InsertFields(InsertFields(base, Seq("3" -> I32(2)), Some(FastIndexedSeq("3", "1", "2"))), Seq("3" -> I32(3)), None)
    assert(Simplify(ir2) == InsertFields(base, Seq("3" -> I32(3)), Some(FastIndexedSeq("3", "1", "2"))))

    val ir3 = InsertFields(InsertFields(base, Seq("3" -> I32(2)), Some(FastIndexedSeq("3", "1", "2"))), Seq("4" -> I32(3)), Some(FastIndexedSeq("3", "1", "2", "4")))
    assert(Simplify(ir3) == InsertFields(base, Seq("3" -> I32(2), "4" -> I32(3)), Some(FastIndexedSeq("3", "1", "2", "4"))))
  }

  @Test def testInsertSelectRewriteRules() {
    val ir1 = SelectFields(InsertFields(base, FastIndexedSeq("3" -> I32(1)), None), FastIndexedSeq("1"))
    assert(Simplify(ir1) == SelectFields(base, FastIndexedSeq("1")))

    val ir2 = SelectFields(InsertFields(base, FastIndexedSeq("3" -> I32(1)), None), FastIndexedSeq("3", "1"))
    assert(Simplify(ir2) == InsertFields(SelectFields(base, FastIndexedSeq("1")), FastIndexedSeq("3" -> I32(1)), Some(FastIndexedSeq("3", "1"))))
  }

  @Test def testBlockMatrixRewriteRules() {
    val bmir = ValueToBlockMatrix(MakeArray(IndexedSeq(F64(1), F64(2), F64(3), F64(4)), TArray(TFloat64())),
      IndexedSeq(2, 2), 10)
    val identityBroadcast = BlockMatrixBroadcast(bmir, IndexedSeq(0, 1), IndexedSeq(2, 2), 10)

    assert(Simplify(identityBroadcast) == bmir)
  }
}
