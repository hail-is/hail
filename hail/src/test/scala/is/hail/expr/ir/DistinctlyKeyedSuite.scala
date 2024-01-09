package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.types.virtual.TInt32
import is.hail.utils.FastSeq

import org.testng.annotations.Test

class DistinctlyKeyedSuite extends HailSuite {
  @Test def distinctlyKeyedRangeTableBase(): Unit = {
    val tableRange = TableRange(10, 2)
    val tableFilter = TableFilter(
      tableRange,
      ApplyComparisonOp(LT(TInt32), GetField(Ref("row", tableRange.typ.rowType), "idx"), I32(5)),
    )
    val tableDistinct = TableDistinct(tableFilter)
    val tableIRSeq = IndexedSeq(tableRange, tableFilter, tableDistinct)
    val distinctlyKeyedAnalysis = DistinctlyKeyed.apply(tableDistinct)
    assert(tableIRSeq.forall(tableIR => distinctlyKeyedAnalysis.contains(tableIR)))
  }

  @Test def readTableKeyByDistinctlyKeyedAnalysis(): Unit = {
    val rt = TableRange(40, 4)
    val idxRef = GetField(Ref("row", rt.typ.rowType), "idx")
    val at = TableMapRows(
      rt,
      MakeStruct(FastSeq(
        "idx" -> idxRef,
        "const" -> 5,
        "half" -> idxRef.floorDiv(2),
        "oneRepeat" -> If(idxRef ceq I32(10), I32(9), idxRef),
      )),
    )
    val keyedByConst = TableKeyBy(at, IndexedSeq("const"))
    val pathConst = ctx.createTmpPath("test-table-distinctly-keyed", "ht")
    Interpret[Unit](ctx, TableWrite(keyedByConst, TableNativeWriter(pathConst)))
    val readConst = TableIR.read(fs, pathConst)
    val distinctlyKeyedAnalysis1 = DistinctlyKeyed.apply(readConst)
    assert(!distinctlyKeyedAnalysis1.contains(readConst))

    val keyedByIdxAndHalf = TableKeyBy(at, IndexedSeq("idx", "half"))
    val pathIdxAndHalf = ctx.createTmpPath("test-table-write-distinctness", "ht")
    Interpret[Unit](ctx, TableWrite(keyedByIdxAndHalf, TableNativeWriter(pathIdxAndHalf)))
    val readIdxAndHalf = TableIR.read(fs, pathIdxAndHalf)
    val distinctlyKeyedAnalysis2 = DistinctlyKeyed.apply(readIdxAndHalf)
    assert(distinctlyKeyedAnalysis2.contains(readIdxAndHalf))

    val disruptedKeysTable = TableKeyBy(readIdxAndHalf, IndexedSeq("idx", "oneRepeat"))
    val distinctlyKeyedAnalysis3 = DistinctlyKeyed.apply(disruptedKeysTable)
    assert(!distinctlyKeyedAnalysis3.contains(disruptedKeysTable))

    val intactKeysTable = TableKeyBy(readIdxAndHalf, IndexedSeq("idx", "half", "oneRepeat"))
    val distinctlyKeyedAnalysis4 = DistinctlyKeyed.apply(intactKeysTable)
    assert(distinctlyKeyedAnalysis4.contains(intactKeysTable))
  }

  @Test def nonDistinctlyKeyedParent(): Unit = {
    val tableRange1 = TableRange(10, 2)
    val tableRange2 = TableRange(10, 2)
    val row = Ref("row", tableRange2.typ.rowType)
    val tableRange1Mapped = TableMapRows(
      tableRange1,
      InsertFields(row, FastSeq("x" -> ToArray(StreamRange(0, GetField(row, "idx"), 1)))),
    )
    val tableRange2Mapped = TableMapRows(
      tableRange2,
      InsertFields(row, FastSeq("x" -> ToArray(StreamRange(0, GetField(row, "idx"), 1)))),
    )
    val tableUnion = TableUnion(IndexedSeq(tableRange1Mapped, tableRange2Mapped))
    val tableExplode = TableExplode(tableUnion, FastSeq("x"))
    val notDistinctlyKeyedSeq = IndexedSeq(tableUnion, tableExplode)
    val distinctlyKeyedAnalysis = DistinctlyKeyed.apply(tableExplode)
    assert(notDistinctlyKeyedSeq.forall(tableIR => !distinctlyKeyedAnalysis.contains(tableIR)))

    val distinctlyKeyedSeq = IndexedSeq(tableRange2Mapped, tableRange1)
    assert(distinctlyKeyedSeq.forall(tableIR => distinctlyKeyedAnalysis.contains(tableIR)))
  }

  @Test def distinctlyKeyedParent(): Unit = {
    val tableRange1 = TableRange(10, 2)
    val tableRange2 = TableRange(10, 2)
    val row = Ref("row", tableRange2.typ.rowType)
    val tableRange1Mapped = TableMapRows(
      tableRange1,
      InsertFields(row, FastSeq("x" -> ToArray(StreamRange(0, GetField(row, "idx"), 1)))),
    )
    val tableRange2Mapped = TableMapRows(
      tableRange2,
      InsertFields(row, FastSeq("x" -> ToArray(StreamRange(0, GetField(row, "idx"), 1)))),
    )
    val tableUnion = TableUnion(IndexedSeq(tableRange1Mapped, tableRange2Mapped))
    val tableExplode = TableExplode(tableUnion, FastSeq("x"))
    val tableDistinct = TableDistinct(tableExplode)
    val distinctlyKeyedAnalysis = DistinctlyKeyed.apply(tableDistinct)
    assert(distinctlyKeyedAnalysis.contains(tableDistinct))
  }

  @Test def iRparent(): Unit = {
    val tableRange = TableRange(10, 2)
    val tableFilter = TableFilter(
      tableRange,
      ApplyComparisonOp(LT(TInt32), GetField(Ref("row", tableRange.typ.rowType), "idx"), I32(5)),
    )
    val tableDistinct = TableDistinct(tableFilter)
    val tableCollect = TableCollect(tableDistinct)
    val distinctlyKeyedAnalysis = DistinctlyKeyed.apply(tableCollect)
    assert(distinctlyKeyedAnalysis.contains(tableDistinct))
    assert(!distinctlyKeyedAnalysis.contains(tableCollect))
  }
}
