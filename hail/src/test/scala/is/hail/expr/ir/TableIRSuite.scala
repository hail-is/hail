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
  def getKT: Table = {
    val data = Array(Array("Sample1", 9, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5), Array("Sample4", 1, 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString()), ("field1", TInt32()), ("field2", TInt32()))
    val keyNames = IndexedSeq("Sample")

    val kt = Table(hc, rdd, signature, keyNames)
    kt.typeCheck()
    kt
  }

  def rangeKT: TableIR = Table.range(hc, 20, Some(4)).unkey().tir

  @BeforeClass def initializeHailContext() {
    assert(hc != null)
  }

  @Test def testRangeCount() {
    val node = ApplyBinaryPrimOp(Add(), TableCount(TableRange(10, 2)), TableCount(TableRange(15, 5)))
    assertEvalsTo(node, 25L)
  }

  @Test def testRangeCollect() {
    val t = TableRange(10, 2)
    val row = Ref("row", t.typ.rowType)
    val node = TableCollect(TableMapRows(t, InsertFields(row, FastIndexedSeq("x" -> GetField(row, "idx")))))
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

  @Test def testFilter() {
    val kt = getKT

    assertEvalsTo(TableCount(TableFilter(kt.tir,
      GetField(Ref("row", kt.typ.rowType), "field1").ceq(3))), 1L)
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

  @Test(dataProvider = "join")
  def testTableJoin(
    leftPart: RVDPartitioner,
    rightPart: RVDPartitioner,
    joinType: String,
    pred: Row => Boolean,
    leftProject: Set[Int],
    rightProject: Set[Int]
  ) {
    val (leftType, leftProjectF) = rowType.filter(f => !leftProject.contains(f.index))
    val left = new Table(hc, TableKeyBy(
      TableParallelize(
        Literal(
          TStruct("rows" -> TArray(leftType), "global" -> TStruct()),
          Row(leftData.map(leftProjectF.asInstanceOf[Row => Row]), Row())),
        Some(1)),
      if (!leftProject.contains(1)) IndexedSeq("A", "B") else IndexedSeq("A")))
    val partitionedLeft = left.copy2(
      rvd = left.value.rvd
        .repartition(if (!leftProject.contains(1)) leftPart else leftPart.coarsen(1)))

    val (rightType, rightProjectF) = rowType.filter(f => !rightProject.contains(f.index))
    val right = new Table(hc, TableKeyBy(
      TableParallelize(
        Literal(
          TStruct("rows" -> TArray(rightType), "global" -> TStruct()),
          Row(rightData.map(rightProjectF.asInstanceOf[Row => Row]), Row())),
        Some(1)),
      if (!rightProject.contains(1)) IndexedSeq("A", "B") else IndexedSeq("A")))
    val partitionedRight = right.copy2(
      rvd = right.value.rvd
        .repartition(if (!rightProject.contains(1)) rightPart else rightPart.coarsen(1)))

    val (_, joinProjectF) = joinedType.filter(f => !leftProject.contains(f.index) && !rightProject.contains(f.index - 2))
    val joined = TableCollect(
      TableJoin(
        partitionedLeft.tir,
        TableRename(
          partitionedRight.tir,
          Array("A","B","C")
            .filter(partitionedRight.typ.rowType.hasField)
            .map(a => a -> (a + "_"))
            .toMap,
          Map.empty),
        joinType, 1))
      assertEvalsTo(joined, Row(expected.filter(pred).map(joinProjectF).toFastIndexedSeq, Row()))
  }

  @Test def testTableKeyBy() {
    val data = Array(Array("A", 1), Array("A", 2), Array("B", 1))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("field1", TString()), ("field2", TInt32()))
    val keyNames = IndexedSeq("field1", "field2")
    val kt = Table(hc, rdd, signature, keyNames)
    val distinctCount = TableCount(TableDistinct(TableLiteral(
      kt.value.copy(typ = kt.typ.copy(key = IndexedSeq("field1")))
    )))
    assertEvalsTo(distinctCount, 2L)
  }

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

  @Test def testTableWrite() {
    val table = TableRange(5, 4)
    val path = tmpDir.createLocalTempFile(extension = "ht")
    Interpret(TableWrite(table, path))
    val before = table.execute(hc)
    val after = Table.read(hc, path)
    assert(before.globals.safeValue == after.globals.safeValue)
    assert(before.rdd.collect().toFastIndexedSeq == after.rdd.collect().toFastIndexedSeq)
  }

  @Test def testTableMultiWayZipJoin() {
    val rowSig = TStruct(
      "id" -> TInt32(),
      "name" -> TString(),
      "data" -> TFloat64()
    )
    val key = IndexedSeq("id")
    val d1 = sc.parallelize(Array(
      Array(0, "a", 0.0),
      Array(1, "b", 3.14),
      Array(2, "c", 2.78)).map(Row.fromSeq(_)))
    val d2 = sc.parallelize(Array(
      Array(0, "d", 1.1),
      Array(0, "x", 2.2),
      Array(2, "v", 7.89)).map(Row.fromSeq(_)))
    val d3 = sc.parallelize(Array(
      Array(1, "f", 9.99),
      Array(2, "g", -1.0),
      Array(3, "z", 0.01)).map(Row.fromSeq(_)))
    val t1 = Table(hc, d1, rowSig, key)
    val t2 = Table(hc, d2, rowSig, key)
    val t3 = Table(hc, d3, rowSig, key)

    val testIr = TableMultiWayZipJoin(IndexedSeq(t1, t2, t3).map(_.tir), "__data", "__globals")
    val testTable = new Table(hc, testIr)

    val expectedSchema = TStruct(
      "id" -> TInt32(),
      "__data" -> TArray(TStruct(
        "name" -> TString(),
        "data" -> TFloat64()))
    )
    val globalSig = TStruct("__globals" -> TArray(TStruct()))
    val globalData = Row.fromSeq(Array(IndexedSeq(Array[Any](), Array[Any](), Array[Any]()).map(Row.fromSeq(_))))
    val expectedData = sc.parallelize(Array(
        Array(0, IndexedSeq(Row.fromSeq(Array("a", 0.0)),  Row.fromSeq(Array("d", 1.1)),  null)),
        Array(0, IndexedSeq(null,                          Row.fromSeq(Array("x", 2.2)),  null)),
        Array(1, IndexedSeq(Row.fromSeq(Array("b", 3.14)), null,                          Row.fromSeq(Array("f",  9.99)))),
        Array(2, IndexedSeq(Row.fromSeq(Array("c", 2.78)), Row.fromSeq(Array("v", 7.89)), Row.fromSeq(Array("g", -1.0)))),
        Array(3, IndexedSeq(null,                          null,                          Row.fromSeq(Array("z",  0.01))))
      ).map(Row.fromSeq(_)))
    val expectedTable = Table(hc, expectedData, expectedSchema, key, globalSig, globalData)
    assert(testTable.same(expectedTable))
  }

  @Test def testTableMultiWayZipJoinGlobals() {
    val t1 = TableMapGlobals(TableRange(10, 1), MakeStruct(Seq("x" -> I32(5))))
    val t2 = TableMapGlobals(TableRange(10, 1), MakeStruct(Seq("x" -> I32(0))))
    val t3 = TableMapGlobals(TableRange(10, 1), MakeStruct(Seq("x" -> NA(TInt32()))))
    val testIr = TableMultiWayZipJoin(IndexedSeq(t1, t2, t3), "__data", "__globals")
    val testTable = new Table(hc, testIr)
    val texp = new Table(hc, TableMapGlobals(
      TableRange(10, 1),
      MakeStruct(Seq("__globals" -> MakeArray(
        Seq(
          MakeStruct(Seq("x" -> I32(5))),
          MakeStruct(Seq("x" -> I32(0))),
          MakeStruct(Seq("x" -> NA(TInt32())))),
        TArray(TStruct("x" -> TInt32()))
      )
    ))))

    assert(testTable.globals.safeValue == texp.globals.safeValue)
  }
}
