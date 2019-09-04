package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.expr.ir.TestUtils._
import is.hail.expr.types._
import is.hail.expr.types.virtual._
import is.hail.rvd.RVDPartitioner
import is.hail.table.Table
import is.hail.utils._
import org.apache.spark.sql.Row
import org.testng.annotations.{DataProvider, Test}
import is.hail.TestUtils._
import is.hail.annotations.BroadcastRow
import is.hail.io.CodecSpec

class TableIRSuite extends HailSuite {
  def getKT: Table = {
    val data = Array(Array("Sample1", 9, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5), Array("Sample4", 1, 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString()), ("field1", TInt32()), ("field2", TInt32()))
    val keyNames = FastIndexedSeq("Sample")

    val kt = Table(hc, rdd, signature, keyNames)
    kt.typeCheck()
    kt
  }

  def rangeKT: TableIR = Table.range(hc, 20, Some(4)).unkey().tir

  implicit val execStrats = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized, ExecStrategy.CxxCompile, ExecStrategy.LoweredJVMCompile)

  @Test def testRangeCount() {
    val node1 = TableCount(TableRange(10, 2))
    val node2 = TableCount(TableRange(15, 5))
    val node = ApplyBinaryPrimOp(Add(), node1, node2)
    assertEvalsTo(node1, 10L)
    assertEvalsTo(node2, 15L)
    assertEvalsTo(node, 25L)
  }

  @Test def testRangeRead() {
    implicit val execStrats = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized, ExecStrategy.LoweredJVMCompile)
    val original = TableMapGlobals(TableRange(10, 3), MakeStruct(FastIndexedSeq("foo" -> I32(57))))

    val path = tmpDir.createTempFile()
    new Table(hc, original).write(path, overwrite = true)

    val read = TableIR.read(hc, path, false, None)
    val droppedRows = TableIR.read(hc, path, true, None)

    val expectedRows = Array.tabulate(10)(i => Row(i)).toFastIndexedSeq
    val expectedGlobals = Row(57)

    assertEvalsTo(TableCollect(read), Row(expectedRows, expectedGlobals))
    assertEvalsTo(TableCollect(droppedRows), Row(FastIndexedSeq(), expectedGlobals))
  }

  @Test def testRangeCollect() {
    val t = TableRange(10, 2)
    val row = Ref("row", t.typ.rowType)
    val node = TableCollect(TableMapRows(t, InsertFields(row, FastIndexedSeq("x" -> GetField(row, "idx")))))
    assertEvalsTo(TableCollect(t), Row(Array.tabulate(10)(Row(_)).toFastIndexedSeq, Row()))
    assertEvalsTo(node, Row(Array.tabulate(10)(i => Row(i, i)).toFastIndexedSeq, Row()))
  }

  @Test def testRangeSum() {
    implicit val execStrats = ExecStrategy.interpretOnly
    val t = TableRange(10, 2)
    val row = Ref("row", t.typ.rowType)
    val sum = AggSignature(Sum(), FastSeq(), None, FastSeq(TInt64()))
    val node = TableCollect(TableMapRows(t, InsertFields(row, FastIndexedSeq("sum" -> ApplyScanOp(FastSeq(), None, FastSeq(Cast(GetField(row, "idx"), TInt64())), sum)))))
    assertEvalsTo(node, Row(Array.tabulate(10)(i => Row(i, Array.range(0, i).sum.toLong)).toFastIndexedSeq, Row()))
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
    val node = TableMapRows(
      TableMapGlobals(t, newGlobals),
      InsertFields(Ref("row", t.typ.rowType), FastSeq("x" -> GetField(Ref("global", newGlobals.typ), "x"))))

    val collectedT = Row(Array.tabulate(10)(i => Row(i)).toFastIndexedSeq, Row())
    val expected = Array.tabulate(10)(i => Row(i, collectedT)).toFastIndexedSeq

    assertEvalsTo(TableCollect(node), Row(expected, Row(collectedT)))
  }

  @Test def testRangeExplode() {
    val t = TableRange(10, 2)
    val row = Ref("row", t.typ.rowType)

    val t2 = TableMapRows(t, InsertFields(row, FastIndexedSeq("x" -> ArrayRange(0, GetField(row, "idx"), 1))))
    val node = TableExplode(t2, FastIndexedSeq("x"))
    val expected = Array.range(0, 10).flatMap(i => Array.range(0, i).map(Row(i, _))).toFastIndexedSeq
    assertEvalsTo(TableCollect(node), Row(expected, Row()))

    val t3 = TableMapRows(t, InsertFields(row,
      FastIndexedSeq("x" ->
        MakeStruct(FastSeq("y" -> ArrayRange(0, GetField(row, "idx"), 1))))))
    val node2 = TableExplode(t3, FastIndexedSeq("x", "y"))
    val expected2 = Array.range(0, 10).flatMap(i => Array.range(0, i).map(j => Row(i, Row(j)))).toFastIndexedSeq
    assertEvalsTo(TableCollect(node2), Row(expected2, Row()))
  }

  @Test def testFilter() {
    val t = TableRange(10, 2)
    val node = TableFilter(
      TableMapGlobals(t, MakeStruct(FastSeq("x" -> GetField(ArrayRef(GetField(TableCollect(t), "rows"), 4), "idx")))),
       ApplyComparisonOp(EQ(TInt32()), GetField(Ref("row", t.typ.rowType), "idx"), GetField(Ref("global", TStruct("x" -> TInt32())), "x")))

    val expected = Array.tabulate(10)(Row(_)).filter(_.get(0) == 4).toFastIndexedSeq

    assertEvalsTo(TableCollect(node), Row(expected, Row(4)))
  }

  @Test def testTableMapWithLiterals() {
    val t = TableRange(10, 2)
    val node = TableMapRows(t,
      InsertFields(Ref("row", t.typ.rowType),
        FastIndexedSeq(
          "a" -> Str("foo"),
          "b" -> Literal(TTuple(TInt32(), TString()), Row(1, "hello")))))

    val expected = Array.tabulate(10)(Row(_, "foo", Row(1, "hello"))).toFastIndexedSeq
    assertEvalsTo(TableCollect(node), Row(expected, Row()))
  }

  @Test def testScanCountBehavesLikeIndex() {
    implicit val execStrats = ExecStrategy.interpretOnly
    val t = rangeKT
    val oldRow = Ref("row", t.typ.rowType)

    val newRow = InsertFields(oldRow, Seq("idx2" -> IRScanCount))
    val newTable = TableMapRows(t, newRow)
    val expected = Array.tabulate(20)(i => Row(i, i.toLong)).toFastIndexedSeq
    assertEvalsTo(ArraySort(TableAggregate(newTable, IRAggCollect(Ref("row", newRow.typ))), True()), expected)
  }

  @Test def testScanCollectBehavesLikeRange() {
    implicit val execStrats = ExecStrategy.interpretOnly
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

  val leftData = FastIndexedSeq(
    (3, 1, -1),
    (3, 2, -1),
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

  val rightData = FastIndexedSeq(
    (6, 1, 1),
    (6, 2, 1),
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
    (3, 1, -1, null, null),
    (3, 2, -1, null, null),
    (6, null, null, 1, 1),
    (6, null, null, 2, 1),
    (11, 1, -1, null, null),
    (11, 2, -1, null, null),
    (16, 1, -1, null, null),
    (16, 2, -1, null, null),
    (17, 1, -1, 1, 1),
    (17, 1, -1, 2, 1),
    (17, 2, -1, 1, 1),
    (17, 2, -1, 2, 1),
    (18, null, null, 1, 1),
    (18, null, null, 2, 1),
    (21, null, null, 1, 1),
    (21, null, null, 2, 1),
    (22, 1, -1, 1, 1),
    (22, 1, -1, 2, 1),
    (22, 2, -1, 1, 1),
    (22, 2, -1, 2, 1),
    (23, 1, -1, null, null),
    (23, 2, -1, null, null),
    (26, 1, -1, null, null),
    (26, 2, -1, null, null),
    (27, 1, -1, 1, 1),
    (27, 1, -1, 2, 1),
    (27, 2, -1, 1, 1),
    (27, 2, -1, 2, 1),
    (28, null, null, 1, 1),
    (28, null, null, 2, 1),
    (31, null, null, 1, 1),
    (31, null, null, 2, 1),
    (32, 1, -1, 1, 1),
    (32, 1, -1, 2, 1),
    (32, 2, -1, 1, 1),
    (32, 2, -1, 2, 1),
    (33, 1, -1, null, null),
    (33, 2, -1, null, null),
    (36, 1, -1, null, null),
    (36, 2, -1, null, null),
    (37, 1, -1, 1, 1),
    (37, 1, -1, 2, 1),
    (37, 2, -1, 1, 1),
    (37, 2, -1, 2, 1),
    (38, null, null, 1, 1),
    (38, null, null, 2, 1),
    (41, null, null, 1, 1),
    (41, null, null, 2, 1)
  ).map(Row.fromTuple)

  val leftPartitioners = Array(
    FastIndexedSeq(
      Interval(Row(0, 0), Row(4, 1), true, false),
      Interval(Row(10, -1), Row(19, 1), true, false),
      Interval(Row(20, 0), Row(24, 0), true, true),
      Interval(Row(25, 0), Row(39, 0), true, true))
    //    FastIndexedSeq(
    //      Interval(Row(0, 0), Row(10), true, false),
    //      Interval(Row(10), Row(44, 0), true, true)),
    //    FastIndexedSeq(Interval(Row(), Row(), true, true))
  ).map(new RVDPartitioner(kType, _))

  val rightPartitioners = Array(
    FastIndexedSeq(
      Interval(Row(5, 0), Row(9, 1), true, false),
      Interval(Row(15, -1), Row(29, 1), true, false),
      Interval(Row(30, 0), Row(34, 0), true, true),
      Interval(Row(35, 0), Row(44, 0), true, true))
    //    FastIndexedSeq(
    //      Interval(Row(0, 0), Row(10), true, false),
    //      Interval(Row(10), Row(44, 0), true, true)),
    //    FastIndexedSeq(Interval(Row(), Row(), true, true))
  ).map(new RVDPartitioner(kType, _))

  val joinTypes = Array(
    ("outer", (row: Row) => true),
    ("left", (row: Row) => !row.isNullAt(1)),
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
    implicit val execStrats = ExecStrategy.interpretOnly
    val (leftType, leftProjectF) = rowType.filter(f => !leftProject.contains(f.index))
    val left = new Table(hc, TableKeyBy(
      TableParallelize(
        Literal(
          TStruct("rows" -> TArray(leftType), "global" -> TStruct()),
          Row(leftData.map(leftProjectF.asInstanceOf[Row => Row]), Row())),
        Some(1)),
      if (!leftProject.contains(1)) FastIndexedSeq("A", "B") else FastIndexedSeq("A")))
    val partitionedLeft = left.copy2(
      rvd = left.rvd
        .repartition(if (!leftProject.contains(1)) leftPart else leftPart.coarsen(1)))

    val (rightType, rightProjectF) = rowType.filter(f => !rightProject.contains(f.index))
    val right = new Table(hc, TableKeyBy(
      TableParallelize(
        Literal(
          TStruct("rows" -> TArray(rightType), "global" -> TStruct()),
          Row(rightData.map(rightProjectF.asInstanceOf[Row => Row]), Row())),
        Some(1)),
      if (!rightProject.contains(1)) FastIndexedSeq("A", "B") else FastIndexedSeq("A")))
    val partitionedRight = right.copy2(
      rvd = right.rvd
        .repartition(if (!rightProject.contains(1)) rightPart else rightPart.coarsen(1)))

    val (_, joinProjectF) = joinedType.filter(f => !leftProject.contains(f.index) && !rightProject.contains(f.index - 2))
    val joined = TableCollect(
      TableJoin(
        partitionedLeft.tir,
        TableRename(
          partitionedRight.tir,
          Array("A", "B", "C")
            .filter(partitionedRight.typ.rowType.hasField)
            .map(a => a -> (a + "_"))
            .toMap,
          Map.empty),
        joinType, 1))
    assertEvalsTo(joined, Row(expected.filter(pred).map(joinProjectF).toFastIndexedSeq, Row()))
  }

  @Test def testTableKeyBy() {
    implicit val execStrats = ExecStrategy.interpretOnly
    val data = Array(Array("A", 1), Array("A", 2), Array("B", 1))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("field1", TString()), ("field2", TInt32()))
    val keyNames = FastIndexedSeq("field1", "field2")
    val kt = Table(hc, rdd, signature, keyNames)
    val tt = TableType(rowType = signature, key = keyNames, globalType = TStruct())
    val base = TableLiteral(
      TableValue(tt,
        BroadcastRow.empty(ctx),
        rdd),
      ctx)

    // construct the table with a longer key, then copy the table to shorten the key in type, but not rvd
    val distinctCount = TableCount(TableDistinct(TableLiteral(tt.copy(key = FastIndexedSeq("field1")), base.rvd, base.enc, base.encodedGlobals)))
    assertEvalsTo(distinctCount, 2L)
  }

  @Test def testTableParallelize() {
    implicit val execStrats = ExecStrategy.interpretOnly
    val t = TStruct("rows" -> TArray(TStruct("a" -> TInt32(), "b" -> TString())), "global" -> TStruct("x" -> TString()))
    val value = Row(FastIndexedSeq(Row(0, "row1"), Row(1, "row2")), Row("glob"))

    assertEvalsTo(
      TableCollect(
        TableParallelize(
          Literal(
            t,
            value
          ))), value)
  }

  @Test def testShuffleAndJoinDoesntMemoryLeak() {
    implicit val execStrats = ExecStrategy.interpretOnly
    val row = Ref("row", TStruct("idx" -> TInt32()))
    val t1 = TableRename(TableRange(1, 1), Map("idx" -> "idx_"), Map.empty)
    val t2 =
      TableKeyBy(
        TableMapRows(
          TableRange(50000, 1),
          InsertFields(row,
            FastIndexedSeq("k" -> (I32(49999) - GetField(row, "idx"))))),
        FastIndexedSeq("k"))

    Interpret(TableJoin(t1, t2, "left"), ctx).rvd.count()
  }

  @Test def testTableRename() {
    implicit val execStrats = ExecStrategy.interpretOnly
    val before = TableMapGlobals(TableRange(10, 1), MakeStruct(Seq("foo" -> I32(0))))
    val t = TableRename(before, Map("idx" -> "idx_"), Map("foo" -> "foo_"))
    assert(t.typ == TableType(rowType = TStruct("idx_" -> TInt32()), key = FastIndexedSeq("idx_"), globalType = TStruct("foo_" -> TInt32())))
    val beforeValue = Interpret(before, ctx)
    val after = Interpret(t, ctx)
    assert(beforeValue.globals.javaValue == after.globals.javaValue)
    assert(beforeValue.rdd.collect().toFastIndexedSeq == after.rdd.collect().toFastIndexedSeq)
  }

  @Test def testTableWrite() {
    implicit val execStrats = ExecStrategy.interpretOnly
    val table = TableRange(5, 4)
    val path = tmpDir.createLocalTempFile(extension = "ht")
    Interpret[Unit](ctx, TableWrite(table, TableNativeWriter(path)))
    val before = table.execute(ctx)
    val after = Table.read(hc, path)
    assert(before.globals.javaValue == after.globals)
    assert(before.rdd.collect().toFastIndexedSeq == after.rdd.collect().toFastIndexedSeq)
  }

  @Test def testTableMultiWayZipJoin() {
    implicit val execStrats = ExecStrategy.interpretOnly
    val rowSig = TStruct(
      "id" -> TInt32(),
      "name" -> TString(),
      "data" -> TFloat64()
    )
    val key = FastIndexedSeq("id")
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

    val testIr = TableMultiWayZipJoin(FastIndexedSeq(t1, t2, t3).map(_.tir), "__data", "__globals")
    val testTable = new Table(hc, testIr)

    val expectedSchema = TStruct(
      "id" -> TInt32(),
      "__data" -> TArray(TStruct(
        "name" -> TString(),
        "data" -> TFloat64()))
    )
    val globalSig = TStruct("__globals" -> TArray(TStruct()))
    val globalData = Row.fromSeq(Array(FastIndexedSeq(Array[Any](), Array[Any](), Array[Any]()).map(Row.fromSeq(_))))
    val expectedData = sc.parallelize(Array(
      Array(0, FastIndexedSeq(Row.fromSeq(Array("a", 0.0)), Row.fromSeq(Array("d", 1.1)), null)),
      Array(0, FastIndexedSeq(null, Row.fromSeq(Array("x", 2.2)), null)),
      Array(1, FastIndexedSeq(Row.fromSeq(Array("b", 3.14)), null, Row.fromSeq(Array("f", 9.99)))),
      Array(2, FastIndexedSeq(Row.fromSeq(Array("c", 2.78)), Row.fromSeq(Array("v", 7.89)), Row.fromSeq(Array("g", -1.0)))),
      Array(3, FastIndexedSeq(null, null, Row.fromSeq(Array("z", 0.01))))
    ).map(Row.fromSeq(_)))
    val expectedTable = Table(hc, expectedData, expectedSchema, key, globalSig, globalData)
    assert(testTable.same(expectedTable))
  }

  @Test def testTableMultiWayZipJoinGlobals() {
    implicit val execStrats = ExecStrategy.interpretOnly
    val t1 = TableMapGlobals(TableRange(10, 1), MakeStruct(Seq("x" -> I32(5))))
    val t2 = TableMapGlobals(TableRange(10, 1), MakeStruct(Seq("x" -> I32(0))))
    val t3 = TableMapGlobals(TableRange(10, 1), MakeStruct(Seq("x" -> NA(TInt32()))))
    val testIr = TableMultiWayZipJoin(FastIndexedSeq(t1, t2, t3), "__data", "__globals")
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

    assert(testTable.globals == texp.globals)
  }
}
