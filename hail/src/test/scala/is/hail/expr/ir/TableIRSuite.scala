package is.hail.expr.ir

import is.hail.ExecStrategy.ExecStrategy
import is.hail.TestUtils._
import is.hail.expr.ir.TestUtils._
import is.hail.expr.types._
import is.hail.expr.types.virtual._
import is.hail.rvd.RVDPartitioner
import is.hail.utils._
import is.hail.{ExecStrategy, HailSuite}
import org.apache.spark.sql.Row
import org.testng.annotations.{DataProvider, Test}

class TableIRSuite extends HailSuite {
  def rangeKT: TableIR = TableKeyBy(TableRange(20, 4), FastIndexedSeq())

  def collect(tir: TableIR): TableCollect = TableCollect(TableKeyBy(tir, FastIndexedSeq()))

  implicit val execStrats: Set[ExecStrategy] = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized, ExecStrategy.LoweredJVMCompile)

  @Test def testRangeCount() {
    val node1 = TableCount(TableRange(10, 2))
    val node2 = TableCount(TableRange(15, 5))
    val node = ApplyBinaryPrimOp(Add(), node1, node2)
    assertEvalsTo(node1, 10L)
    assertEvalsTo(node2, 15L)
    assertEvalsTo(node, 25L)
  }

  @Test def testRangeRead() {
    val original = TableKeyBy(TableMapGlobals(TableRange(10, 3), MakeStruct(FastIndexedSeq("foo" -> I32(57)))), FastIndexedSeq())

    val path = tmpDir.createTempFile()
    CompileAndEvaluate[Unit](ctx, TableWrite(original, TableNativeWriter(path, overwrite = true)), false)

    val read = TableIR.read(hc, path, false, None)
    val droppedRows = TableIR.read(hc, path, true, None)

    val expectedRows = Array.tabulate(10)(i => Row(i)).toFastIndexedSeq
    val expectedGlobals = Row(57)

    assertEvalsTo(TableCollect(read), Row(expectedRows, expectedGlobals))
    assertEvalsTo(TableCollect(droppedRows), Row(FastIndexedSeq(), expectedGlobals))
  }

  @Test def testRangeCollect() {
    implicit val execStrats = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized)
    val t = TableRange(10, 2)
    val row = Ref("row", t.typ.rowType)
    val node = collect(TableMapRows(t, InsertFields(row, FastIndexedSeq("x" -> GetField(row, "idx")))))
    assertEvalsTo(collect(t), Row(Array.tabulate(10)(Row(_)).toFastIndexedSeq, Row()))
    assertEvalsTo(node, Row(Array.tabulate(10)(i => Row(i, i)).toFastIndexedSeq, Row()))
  }

  @Test def testRangeSum() {
    implicit val execStrats = ExecStrategy.interpretOnly
    val t = TableRange(10, 2)
    val row = Ref("row", t.typ.rowType)
    val sum = AggSignature(Sum(), FastSeq(), FastSeq(TInt64))
    val node = collect(TableMapRows(t, InsertFields(row, FastIndexedSeq("sum" -> ApplyScanOp(FastSeq(), FastSeq(Cast(GetField(row, "idx"), TInt64)), sum)))))
    assertEvalsTo(node, Row(Array.tabulate(10)(i => Row(i, Array.range(0, i).sum.toLong)).toFastIndexedSeq, Row()))
  }

  @Test def testGetGlobals() {
    implicit val execStrats = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized)
    val t = TableRange(10, 2)
    val newGlobals = InsertFields(Ref("global", t.typ.globalType), FastSeq("x" -> collect(t)))
    val node = TableGetGlobals(TableMapGlobals(t, newGlobals))
    assertEvalsTo(node, Row(Row(Array.tabulate(10)(i => Row(i)).toFastIndexedSeq, Row())))
  }

  @Test def testCollectGlobals() {
    implicit val execStrats = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized)
    val t = TableRange(10, 2)
    val newGlobals = InsertFields(Ref("global", t.typ.globalType), FastSeq("x" -> collect(t)))
    val node = TableMapRows(
      TableMapGlobals(t, newGlobals),
      InsertFields(Ref("row", t.typ.rowType), FastSeq("x" -> GetField(Ref("global", newGlobals.typ), "x"))))

    val collectedT = Row(Array.tabulate(10)(i => Row(i)).toFastIndexedSeq, Row())
    val expected = Array.tabulate(10)(i => Row(i, collectedT)).toFastIndexedSeq

    assertEvalsTo(collect(node), Row(expected, Row(collectedT)))
  }

  @Test def testRangeExplode() {
    implicit val execStrats = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized)
    val t = TableRange(10, 2)
    val row = Ref("row", t.typ.rowType)

    val t2 = TableMapRows(t, InsertFields(row, FastIndexedSeq("x" -> ToArray(StreamRange(0, GetField(row, "idx"), 1)))))
    val node = TableExplode(t2, FastIndexedSeq("x"))
    val expected = Array.range(0, 10).flatMap(i => Array.range(0, i).map(Row(i, _))).toFastIndexedSeq
    assertEvalsTo(collect(node), Row(expected, Row()))

    val t3 = TableMapRows(t, InsertFields(row,
      FastIndexedSeq("x" ->
        MakeStruct(FastSeq("y" -> ToArray(StreamRange(0, GetField(row, "idx"), 1)))))))
    val node2 = TableExplode(t3, FastIndexedSeq("x", "y"))
    val expected2 = Array.range(0, 10).flatMap(i => Array.range(0, i).map(j => Row(i, Row(j)))).toFastIndexedSeq
    assertEvalsTo(collect(node2), Row(expected2, Row()))
  }

  @Test def testFilter() {
    implicit val execStrats = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized)
    val t = TableRange(10, 2)
    val node = TableFilter(
      TableMapGlobals(t, MakeStruct(FastSeq("x" -> GetField(ArrayRef(GetField(collect(t), "rows"), 4), "idx")))),
       ApplyComparisonOp(EQ(TInt32), GetField(Ref("row", t.typ.rowType), "idx"), GetField(Ref("global", TStruct("x" -> TInt32)), "x")))

    val expected = Array.tabulate(10)(Row(_)).filter(_.get(0) == 4).toFastIndexedSeq

    assertEvalsTo(collect(node), Row(expected, Row(4)))
  }

  @Test def testTableMapWithLiterals() {
    implicit val execStrats = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized)
    val t = TableRange(10, 2)
    val node = TableMapRows(t,
      InsertFields(Ref("row", t.typ.rowType),
        FastIndexedSeq(
          "a" -> Str("foo"),
          "b" -> Literal(TTuple(TInt32, TString), Row(1, "hello")))))

    val expected = Array.tabulate(10)(Row(_, "foo", Row(1, "hello"))).toFastIndexedSeq
    assertEvalsTo(collect(node), Row(expected, Row()))
  }

  @Test def testScanCountBehavesLikeIndex() {
    implicit val execStrats = ExecStrategy.interpretOnly
    val t = rangeKT
    val oldRow = Ref("row", t.typ.rowType)

    val newRow = InsertFields(oldRow, Seq("idx2" -> IRScanCount))
    val newTable = TableMapRows(t, newRow)
    val expected = Array.tabulate(20)(i => Row(i, i.toLong)).toFastIndexedSeq
    assertEvalsTo(ArraySort(ToStream(TableAggregate(newTable, IRAggCollect(Ref("row", newRow.typ)))), True()), expected)
  }

  @Test def testScanCollectBehavesLikeRange() {
    implicit val execStrats = ExecStrategy.interpretOnly
    val t = rangeKT
    val oldRow = Ref("row", t.typ.rowType)

    val newRow = InsertFields(oldRow, Seq("range" -> IRScanCollect(GetField(oldRow, "idx"))))
    val newTable = TableMapRows(t, newRow)

    val expected = Array.tabulate(20)(i => Row(i, Array.range(0, i).toFastIndexedSeq)).toFastIndexedSeq
    assertEvalsTo(ArraySort(ToStream(TableAggregate(newTable, IRAggCollect(Ref("row", newRow.typ)))), True()), expected)
  }

  val rowType = TStruct(("A", TInt32), ("B", TInt32), ("C", TInt32))
  val joinedType = TStruct(("A", TInt32), ("B", TInt32), ("C", TInt32), ("B_1", TInt32), ("C_1", TInt32))
  val kType = TStruct(("A", TInt32), ("B", TInt32))

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
    val left = Interpret(TableKeyBy(
      TableParallelize(
        Literal(
          TStruct("rows" -> TArray(leftType), "global" -> TStruct.empty),
          Row(leftData.map(leftProjectF.asInstanceOf[Row => Row]), Row())),
        Some(1)),
      if (!leftProject.contains(1)) FastIndexedSeq("A", "B") else FastIndexedSeq("A")),
      ctx,
      optimize = false)
    val partitionedLeft = left.copy(rvd = left.rvd
      .repartition(if (!leftProject.contains(1)) leftPart else leftPart.coarsen(1), ctx)
    )

    val (rightType, rightProjectF) = rowType.filter(f => !rightProject.contains(f.index))
    val right = Interpret(TableKeyBy(
      TableParallelize(
        Literal(
          TStruct("rows" -> TArray(rightType), "global" -> TStruct.empty),
          Row(rightData.map(rightProjectF.asInstanceOf[Row => Row]), Row())),
        Some(1)),
      if (!rightProject.contains(1)) FastIndexedSeq("A", "B") else FastIndexedSeq("A")),
      ctx,
      optimize = false)
    val partitionedRight = right.copy(
      rvd = right.rvd
        .repartition(if (!rightProject.contains(1)) rightPart else rightPart.coarsen(1), ctx))

    val (_, joinProjectF) = joinedType.filter(f => !leftProject.contains(f.index) && !rightProject.contains(f.index - 2))
    val joined = collect(
      TableJoin(
        TableLiteral(partitionedLeft, ctx),
        TableRename(
          TableLiteral(partitionedRight, ctx),
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
    val signature = TStruct(("field1", TString), ("field2", TInt32))
    val keyNames = FastIndexedSeq("field1", "field2")
    val tt = TableType(rowType = signature, key = keyNames, globalType = TStruct.empty)
    val base = TableLiteral(
      TableValue(ctx, tt.rowType, tt.key, rdd),
      ctx)

    // construct the table with a longer key, then copy the table to shorten the key in type, but not rvd
    val distinctCount = TableCount(TableDistinct(TableLiteral(tt.copy(key = FastIndexedSeq("field1")), base.rvd, base.enc, base.encodedGlobals)))
    assertEvalsTo(distinctCount, 2L)
  }

  @Test def testTableParallelize() {
    implicit val execStrats = ExecStrategy.interpretOnly
    val t = TStruct("rows" -> TArray(TStruct("a" -> TInt32, "b" -> TString)), "global" -> TStruct("x" -> TString))
    val value = Row(FastIndexedSeq(Row(0, "row1"), Row(1, "row2")), Row("glob"))

    assertEvalsTo(
      collect(
        TableParallelize(
          Literal(
            t,
            value
          ))), value)
  }

  @Test def testShuffleAndJoinDoesntMemoryLeak() {
    implicit val execStrats = ExecStrategy.interpretOnly
    val row = Ref("row", TStruct("idx" -> TInt32))
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
    assert(t.typ == TableType(rowType = TStruct("idx_" -> TInt32), key = FastIndexedSeq("idx_"), globalType = TStruct("foo_" -> TInt32)))
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
    val after = Interpret(TableIR.read(hc, path), ctx, false)
    assert(before.globals.javaValue == after.globals.javaValue)
    assert(before.rdd.collect().toFastIndexedSeq == after.rdd.collect().toFastIndexedSeq)
  }

  @Test def testPartitionCountsWithDropRows() {
    val tr = new TableReader {
      override def apply(tr: TableRead, ctx: ExecuteContext): TableValue = ???

      override def partitionCounts: Option[IndexedSeq[Long]] = Some(FastIndexedSeq(1, 2, 3, 4))

      override def fullType: TableType = TableType(TStruct.empty, FastIndexedSeq(), TStruct.empty)
    }
    val tir = TableRead(tr.fullType, true, tr)
    assert(tir.partitionCounts.forall(_.sum == 0))
  }
}
