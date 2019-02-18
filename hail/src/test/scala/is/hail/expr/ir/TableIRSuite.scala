package is.hail.expr.ir

import is.hail.SparkSuite
import is.hail.expr.ir.TestUtils._
import is.hail.expr.types._
import is.hail.expr.types.virtual.{TArray, TFloat64, TInt32, TString, TStruct}
import is.hail.rvd.{RVD, RVDContext, RVDPartitioner}
import is.hail.sparkextras.ContextRDD
import is.hail.table.Table
import is.hail.utils._
import is.hail.TestUtils._
import is.hail.io.CodecSpec
import org.apache.spark.sql.Row
import org.testng.annotations.{BeforeClass, DataProvider, Test}
import is.hail.TestUtils._

class TableIRSuite extends SparkSuite {
  def rangeKT: TableIR = TableKeyBy(Table.range(hc, 20, Some(4)), FastIndexedSeq())

  @BeforeClass def ensureHCDefined() { initializeHailContext() }

  @Test def testRangeCount() {
    val node1 = TableCount(TableRange(10, 2))
    val node2 = TableCount(TableRange(15, 5))
    val node = ApplyBinaryPrimOp(Add(), node1, node2)
    assertEvalsTo(node1, 10L)
    assertEvalsTo(node2, 15L)
    assertEvalsTo(node, 25L)
  }

  @Test def testRangeCollect() {
    val t = TableRange(10, 2)
    val row = Ref("row", t.typ.rowType)
    val node = TableCollect(TableMapRows(t, InsertFields(row, FastIndexedSeq("x" -> GetField(row, "idx")))))
    assertEvalsTo(TableCollect(t), Row(Array.tabulate(10)(Row(_)).toFastIndexedSeq, Row()))
    assertEvalsTo(node, Row(Array.tabulate(10)(i => Row(i, i)).toFastIndexedSeq, Row()))
  }

  @Test def testGetGlobals() {
    val t = TableRange(10, 2)
    val newGlobals = InsertFields(Ref("global", t.typ.globalType), FastSeq("x" -> TableCollect(t)))
    val node = TableGetGlobals(TableMapGlobals(t, newGlobals))
    assertEvalsTo(node, Row(Row(Array.tabulate(10)(i => Row(i)).toFastIndexedSeq, Row())))
  }

  @Test def testCollectGlobals() {
    val t = TableRange(10, 2)
    val newGlobals = InsertFields(Ref("global", t.typ.globalType), FastSeq("x" -> TableCollect(t)))
    val node =TableMapRows(
      TableMapGlobals(t, newGlobals),
      InsertFields(Ref("row", t.typ.rowType), FastSeq("x" -> GetField(Ref("global", newGlobals.typ), "x"))))

    val collectedT = Row(Array.tabulate(10)(i => Row(i)).toFastIndexedSeq, Row())
    val expected = Array.tabulate(10)(i => Row(i, collectedT)).toFastIndexedSeq

    assertEvalsTo(TableCollect(node), Row(expected, Row(collectedT)))
  }

  @Test def testScanCountBehavesLikeIndex() {
    val t = rangeKT
    val oldRow = Ref("row", t.typ.rowType)

    val newRow = InsertFields(oldRow, Seq("idx2" -> IRScanCount))
    val newTable = TableMapRows(t, newRow)
    val expected = Array.tabulate(20)(i => Row(i, i.toLong)).toFastIndexedSeq
    assertEvalsTo(ArraySort(TableAggregate(newTable, IRAggCollect(Ref("row", newRow.typ))), True()), expected)
  }

  @Test def testScanCollectBehavesLikeRange() {
    val t = rangeKT
    val oldRow = Ref("row", t.typ.rowType)

    val newRow = InsertFields(oldRow, Seq("range" -> IRScanCollect(GetField(oldRow, "idx"))))
    val newTable = TableMapRows(t, newRow)

    val expected = Array.tabulate(20)(i => Row(i, Array.range(0, i).toFastIndexedSeq)).toFastIndexedSeq
    assertEvalsTo(ArraySort(TableAggregate(newTable, IRAggCollect(Ref("row", newRow.typ))), True()), expected)
  }

  val rowType = TStruct(("A", TInt32()), ("B", TInt32()), ("C", TInt32()))
  val joinedType = TStruct(("A", TInt32()), ("B", TInt32()), ("C", TInt32()), ("B_1", TInt32()), ("C_1", TInt32()))
  val kType = TStruct(("A", TInt32()), ("B", TInt32()))

  val leftData = IndexedSeq(
    ( 3, 1, -1),
    ( 3, 2, -1),
    (11, 1, -1),
    (11, 2, -1),
    (16, 1, -1),
    (16, 2, -1),
    (17, 1, -1),
    (17, 2, -1),
    (22, 1, -1),
    (22, 2, -1),
    (23, 1, -1),
    (23, 2, -1),
    (26, 1, -1),
    (26, 2, -1),
    (27, 1, -1),
    (27, 2, -1),
    (32, 1, -1),
    (32, 2, -1),
    (33, 1, -1),
    (33, 2, -1),
    (36, 1, -1),
    (36, 2, -1),
    (37, 1, -1),
    (37, 2, -1)
  ).map(Row.fromTuple)

  val rightData = IndexedSeq(
    ( 6, 1, 1),
    ( 6, 2, 1),
    (17, 1, 1),
    (17, 2, 1),
    (18, 1, 1),
    (18, 2, 1),
    (21, 1, 1),
    (21, 2, 1),
    (22, 1, 1),
    (22, 2, 1),
    (27, 1, 1),
    (27, 2, 1),
    (28, 1, 1),
    (28, 2, 1),
    (31, 1, 1),
    (31, 2, 1),
    (32, 1, 1),
    (32, 2, 1),
    (37, 1, 1),
    (37, 2, 1),
    (38, 1, 1),
    (38, 2, 1),
    (41, 1, 1),
    (41, 2, 1)
  ).map(Row.fromTuple)

  val expected = Array(
    ( 3,    1,  -1, null, null),
    ( 3,    2,  -1, null, null),
    ( 6, null,null,    1,    1),
    ( 6, null,null,    2,    1),
    (11,    1,  -1, null, null),
    (11,    2,  -1, null, null),
    (16,    1,  -1, null, null),
    (16,    2,  -1, null, null),
    (17,    1,  -1,    1,    1),
    (17,    1,  -1,    2,    1),
    (17,    2,  -1,    1,    1),
    (17,    2,  -1,    2,    1),
    (18, null,null,    1,    1),
    (18, null,null,    2,    1),
    (21, null,null,    1,    1),
    (21, null,null,    2,    1),
    (22,    1,  -1,    1,    1),
    (22,    1,  -1,    2,    1),
    (22,    2,  -1,    1,    1),
    (22,    2,  -1,    2,    1),
    (23,    1,  -1, null, null),
    (23,    2,  -1, null, null),
    (26,    1,  -1, null, null),
    (26,    2,  -1, null, null),
    (27,    1,  -1,    1,    1),
    (27,    1,  -1,    2,    1),
    (27,    2,  -1,    1,    1),
    (27,    2,  -1,    2,    1),
    (28, null,null,    1,    1),
    (28, null,null,    2,    1),
    (31, null,null,    1,    1),
    (31, null,null,    2,    1),
    (32,    1,  -1,    1,    1),
    (32,    1,  -1,    2,    1),
    (32,    2,  -1,    1,    1),
    (32,    2,  -1,    2,    1),
    (33,    1,  -1, null, null),
    (33,    2,  -1, null, null),
    (36,    1,  -1, null, null),
    (36,    2,  -1, null, null),
    (37,    1,  -1,    1,    1),
    (37,    1,  -1,    2,    1),
    (37,    2,  -1,    1,    1),
    (37,    2,  -1,    2,    1),
    (38, null,null,    1,    1),
    (38, null,null,    2,    1),
    (41, null,null,    1,    1),
    (41, null,null,    2,    1)
  ).map(Row.fromTuple)

  val leftPartitioners = Array(
    IndexedSeq(
      Interval(Row(0, 0), Row(4, 1), true, false),
      Interval(Row(10, -1), Row(19, 1), true, false),
      Interval(Row(20, 0), Row(24, 0), true, true),
      Interval(Row(25, 0), Row(39, 0), true, true))
//    IndexedSeq(
//      Interval(Row(0, 0), Row(10), true, false),
//      Interval(Row(10), Row(44, 0), true, true)),
//    IndexedSeq(Interval(Row(), Row(), true, true))
  ).map(new RVDPartitioner(kType, _))

  val rightPartitioners = Array(
    IndexedSeq(
      Interval(Row(5, 0), Row(9, 1), true, false),
      Interval(Row(15, -1), Row(29, 1), true, false),
      Interval(Row(30, 0), Row(34, 0), true, true),
      Interval(Row(35, 0), Row(44, 0), true, true))
//    IndexedSeq(
//      Interval(Row(0, 0), Row(10), true, false),
//      Interval(Row(10), Row(44, 0), true, true)),
//    IndexedSeq(Interval(Row(), Row(), true, true))
  ).map(new RVDPartitioner(kType, _))

  val joinTypes = Array(
    ("outer", (row: Row) => true),
    ("left",  (row: Row) => !row.isNullAt(1)),
    ("right", (row: Row) => !row.isNullAt(3)),
    ("inner", (row: Row) => !row.isNullAt(1) && !row.isNullAt(3))
  )

  @DataProvider(name = "join")
  def joinData(): Array[Array[Any]] =
    for {
      l <- leftPartitioners
      r <- rightPartitioners
      (j, p) <- joinTypes
      leftProject <- Seq[Set[Int]](Set(), Set(1), Set(2), Set(1, 2))
      rightProject <- Seq[Set[Int]](Set(), Set(1), Set(2), Set(1, 2))
      if !leftProject.contains(1) || rightProject.contains(1)
    } yield Array[Any](l, r, j, p, leftProject, rightProject)

  @Test def testTableParallelize() {
    val t = TStruct("rows" -> TArray(TStruct("a" -> TInt32(), "b" -> TString())), "global" -> TStruct("x" -> TString()))
    val value = Row(IndexedSeq(Row(0, "row1"), Row(1, "row2")), Row("glob"))

    assertEvalsTo(
      TableCollect(
        TableParallelize(
          Literal(
            t,
            value
          ))), value)
  }

  @Test def testShuffleAndJoinDoesntMemoryLeak() {
    val row = Ref("row", TStruct("idx" -> TInt32()))
    val t1 = TableRename(TableRange(1, 1), Map("idx" -> "idx_"), Map.empty)
    val t2 =
      TableKeyBy(
        TableMapRows(
          TableRange(50000, 1),
          InsertFields(row,
            FastIndexedSeq("k" -> (I32(49999)-GetField(row, "idx"))))),
        FastIndexedSeq("k"))

    Interpret(TableJoin(t1, t2, "left")).rvd.count()
  }

  @Test def testTableRename() {
    val before = TableMapGlobals(TableRange(10, 1), MakeStruct(Seq("foo" -> I32(0))))
    val t = TableRename(before, Map("idx" -> "idx_"), Map("foo" -> "foo_"))
    assert(t.typ == TableType(rowType = TStruct("idx_" -> TInt32()), key = FastIndexedSeq("idx_"), globalType = TStruct("foo_" -> TInt32())))
    val beforeValue = Interpret(before)
    val after = Interpret(t)
    assert(beforeValue.globals.safeValue == after.globals.safeValue)
    assert(beforeValue.rdd.collect().toFastIndexedSeq == after.rdd.collect().toFastIndexedSeq)
  }
}
