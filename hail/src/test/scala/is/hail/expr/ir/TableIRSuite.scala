package is.hail.expr.ir

import is.hail.SparkSuite
import is.hail.expr.ir.TestUtils._
import is.hail.expr.types._
import is.hail.rvd.{RVD, RVDPartitioner}
import is.hail.table.Table
import is.hail.utils._
import org.apache.spark.sql.Row
import org.testng.annotations.{DataProvider, Test}

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

  @Test def testFilter() {
    val kt = getKT
    val kt2 = new Table(hc, TableFilter(kt.tir,
      GetField(Ref("row", kt.typ.rowType), "field1").ceq(3)))
    assert(kt2.count() == 1)
  }

  @Test def testScanCountBehavesLikeIndex() {
    val t = rangeKT
    val oldRow = Ref("row", t.typ.rowType)

    val newRow = InsertFields(oldRow, Seq("idx2" -> IRScanCount))
    val newTable = TableMapRows(t, newRow)
    val rows = Interpret[IndexedSeq[Row]](TableAggregate(newTable, IRAggCollect(Ref("row", newRow.typ))), optimize = false)
    assert(rows.forall { case Row(row_idx, idx) => row_idx == idx})
  }

  @Test def testScanCollectBehavesLikeRange() {
    val t = rangeKT
    val oldRow = Ref("row", t.typ.rowType)

    val newRow = InsertFields(oldRow, Seq("range" -> IRScanCollect(GetField(oldRow, "idx"))))
    val newTable = TableMapRows(t, newRow)
    val rows = Interpret[IndexedSeq[Row]](TableAggregate(newTable, IRAggCollect(Ref("row", newRow.typ))), optimize = false)
    assert(rows.forall { case Row(row_idx: Int, range: IndexedSeq[_]) => range sameElements Array.range(0, row_idx)})
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
        Literal(TArray(leftType), leftData.map(leftProjectF.asInstanceOf[Row => Row])),
        Some(1)),
      if (!leftProject.contains(1)) IndexedSeq("A", "B") else IndexedSeq("A")))
    val partitionedLeft = left.copy2(
      rvd = left.value.rvd
        .repartition(if (!leftProject.contains(1)) leftPart else leftPart.coarsen(1)))

    val (rightType, rightProjectF) = rowType.filter(f => !rightProject.contains(f.index))
    val right = new Table(hc, TableKeyBy(
      TableParallelize(
        Literal(TArray(rightType), rightData.map(rightProjectF.asInstanceOf[Row => Row])),
        Some(1)),
      if (!rightProject.contains(1)) IndexedSeq("A", "B") else IndexedSeq("A")))
    val partitionedRight = right.copy2(
      rvd = right.value.rvd
        .repartition(if (!rightProject.contains(1)) rightPart else rightPart.coarsen(1)))

    val (_, joinProjectF) = joinedType.filter(f => !leftProject.contains(f.index) && !rightProject.contains(f.index - 2))
    val joined = TableJoin(
      partitionedLeft.tir,
      TableRename(
        partitionedRight.tir,
        Array("A","B","C")
          .filter(partitionedRight.typ.rowType.hasField)
          .map(a => a -> (a + "_"))
          .toMap,
        Map.empty),
      joinType, 1)
      .execute(hc).rdd.collect()
    val thisExpected = expected.filter(pred).map(joinProjectF)
    assert(joined sameElements expected.filter(pred).map(joinProjectF))
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

    TableJoin(t1, t2, "left").execute(hc).rvd.count()
  }

  @Test def testTableRename() {
    val before = TableMapGlobals(TableRange(10, 1), MakeStruct(Seq("foo" -> I32(0))))
    val t = TableRename(before, Map("idx" -> "idx_"), Map("foo" -> "foo_"))
    assert(t.typ == TableType(rowType = TStruct("idx_" -> TInt32()), key = FastIndexedSeq("idx_"), globalType = TStruct("foo_" -> TInt32())))
    val beforeValue = before.execute(hc)
    val after = t.execute(hc)
    assert(beforeValue.globals.safeValue == after.globals.safeValue)
    assert(beforeValue.rdd.collect().toFastIndexedSeq == after.rdd.collect().toFastIndexedSeq)
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
}
