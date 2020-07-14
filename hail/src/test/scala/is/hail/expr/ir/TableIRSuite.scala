package is.hail.expr.ir

import is.hail.ExecStrategy.ExecStrategy
import is.hail.TestUtils._
import is.hail.expr.ir.TestUtils._
import is.hail.methods.ForceCountTable
import is.hail.types._
import is.hail.types.physical.PStruct
import is.hail.types.virtual._
import is.hail.rvd.RVDPartitioner
import is.hail.utils._
import is.hail.{ExecStrategy, HailSuite}
import org.apache.spark.sql.Row
import org.testng.annotations.{DataProvider, Test}

class TableIRSuite extends HailSuite {
  def rangeKT: TableIR = TableKeyBy(TableRange(20, 4), FastIndexedSeq())

  def collect(tir: TableIR): IR =
    TableAggregate(tir, MakeStruct(FastSeq(
      "rows" -> IRAggCollect(Ref("row", tir.typ.rowType)),
      "global" -> Ref("global", tir.typ.globalType))))
  def collectNoKey(tir: TableIR): IR = TableCollect(tir)

  implicit val execStrats: Set[ExecStrategy] = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized, ExecStrategy.LoweredJVMCompile)

  @Test def testRangeCount() {
    val node1 = TableCount(TableRange(10, 2))
    val node2 = TableCount(TableRange(15, 5))
    val node = ApplyBinaryPrimOp(Add(), node1, node2)
    assertEvalsTo(node1, 10L)
    assertEvalsTo(node2, 15L)
    assertEvalsTo(node, 25L)
  }

  @Test def testForceCount(): Unit = {
    implicit val execStrats = ExecStrategy.interpretOnly
    val tableRangeSize = Int.MaxValue / 20
    val forceCountRange = TableToValueApply(TableRange(tableRangeSize, 2), ForceCountTable())
    assertEvalsTo(forceCountRange, tableRangeSize.toLong)
  }

  @Test def testRangeRead() {
    implicit val execStrats = ExecStrategy.lowering
    val original = TableKeyBy(TableMapGlobals(TableRange(10, 3), MakeStruct(FastIndexedSeq("foo" -> I32(57)))), FastIndexedSeq())

    val path = ctx.createTmpPath("test-range-read", "ht")
    val write = TableWrite(original, TableNativeWriter(path, overwrite = true))
    assertEvalsTo(write, ())
    val read = TableIR.read(fs, path, false, None)
    val droppedRows = TableIR.read(fs, path, true, None)

    val expectedRows = Array.tabulate(10)(i => Row(i)).toFastIndexedSeq
    val expectedGlobals = Row(57)

    assertEvalsTo(TableCollect(read), Row(expectedRows, expectedGlobals))
    assertEvalsTo(TableCollect(droppedRows), Row(FastIndexedSeq(), expectedGlobals))
  }

  @Test def testCountRead(): Unit = {
    implicit val execStrats = ExecStrategy.lowering
    val tir: TableIR = TableRead.native(fs, "src/test/resources/three_key.ht")
    assertEvalsTo(TableCount(tir), 120L)
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

  val expectedUnion = Array(
    (3, 1, -1),
    (3, 2, -1),
    (6, 1, 1),
    (6, 2, 1),
    (11, 1, -1),
    (11, 2, -1),
    (16, 1, -1),
    (16, 2, -1),
    (17, 1, -1),
    (17, 1, 1),
    (17, 2, -1),
    (17, 2, 1),
    (18, 1, 1),
    (18, 2, 1),
    (21, 1, 1),
    (21, 2, 1),
    (22, 1, -1),
    (22, 1, 1),
    (22, 2, -1),
    (22, 2, 1),
    (23, 1, -1),
    (23, 2, -1),
    (26, 1, -1),
    (26, 2, -1),
    (27, 1, -1),
    (27, 1, 1),
    (27, 2, -1),
    (27, 2, 1),
    (28, 1, 1),
    (28, 2, 1),
    (31, 1, 1),
    (31, 2, 1),
    (32, 1, -1),
    (32, 1, 1),
    (32, 2, -1),
    (32, 2, 1),
    (33, 1, -1),
    (33, 2, -1),
    (36, 1, -1),
    (36, 2, -1),
    (37, 1, -1),
    (37, 1, 1),
    (37, 2, -1),
    (37, 2, 1),
    (38, 1, 1),
    (38, 2, 1),
    (41, 1, 1),
    (41, 2, 1)
  ).map(Row.fromTuple)

  val expectedZipJoin = Array(
    (3, 1, FastIndexedSeq(Row(-1), null)),
    (3, 2, FastIndexedSeq(Row(-1), null)),
    (6, 1, FastIndexedSeq(null, Row(1))),
    (6, 2, FastIndexedSeq(null, Row(1))),
    (11, 1, FastIndexedSeq(Row(-1), null)),
    (11, 2, FastIndexedSeq(Row(-1), null)),
    (16, 1, FastIndexedSeq(Row(-1), null)),
    (16, 2, FastIndexedSeq(Row(-1), null)),
    (17, 1, FastIndexedSeq(Row(-1), Row(1))),
    (17, 2, FastIndexedSeq(Row(-1), Row(1))),
    (18, 1, FastIndexedSeq(null, Row(1))),
    (18, 2, FastIndexedSeq(null, Row(1))),
    (21, 1, FastIndexedSeq(null, Row(1))),
    (21, 2, FastIndexedSeq(null, Row(1))),
    (22, 1, FastIndexedSeq(Row(-1), Row(1))),
    (22, 2, FastIndexedSeq(Row(-1), Row(1))),
    (23, 1, FastIndexedSeq(Row(-1), null)),
    (23, 2, FastIndexedSeq(Row(-1), null)),
    (26, 1, FastIndexedSeq(Row(-1), null)),
    (26, 2, FastIndexedSeq(Row(-1), null)),
    (27, 1, FastIndexedSeq(Row(-1), Row(1))),
    (27, 2, FastIndexedSeq(Row(-1), Row(1))),
    (28, 1, FastIndexedSeq(null, Row(1))),
    (28, 2, FastIndexedSeq(null, Row(1))),
    (31, 1, FastIndexedSeq(null, Row(1))),
    (31, 2, FastIndexedSeq(null, Row(1))),
    (32, 1, FastIndexedSeq(Row(-1), Row(1))),
    (32, 2, FastIndexedSeq(Row(-1), Row(1))),
    (33, 1, FastIndexedSeq(Row(-1), null)),
    (33, 2, FastIndexedSeq(Row(-1), null)),
    (36, 1, FastIndexedSeq(Row(-1), null)),
    (36, 2, FastIndexedSeq(Row(-1), null)),
    (37, 1, FastIndexedSeq(Row(-1), Row(1))),
    (37, 2, FastIndexedSeq(Row(-1), Row(1))),
    (38, 1, FastIndexedSeq(null, Row(1))),
    (38, 2, FastIndexedSeq(null, Row(1))),
    (41, 1, FastIndexedSeq(null, Row(1))),
    (41, 2, FastIndexedSeq(null, Row(1)))
    ).map(Row.fromTuple)

  val expectedOuterJoin = Array(
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

  val joinTypes = Array(
    ("outer", (row: Row) => true),
    ("left", (row: Row) => !row.isNullAt(1)),
    ("right", (row: Row) => !row.isNullAt(3)),
    ("inner", (row: Row) => !row.isNullAt(1) && !row.isNullAt(3))
  )

  @DataProvider(name = "join")
  def joinData(): Array[Array[Any]] =
    for {
      lParts <- Array[Integer](1, 2, 3)
      rParts <- Array[Integer](1, 2, 3)
      (j, p) <- joinTypes
      leftProject <- Seq[Set[Int]](Set(), Set(1), Set(2), Set(1, 2))
      rightProject <- Seq[Set[Int]](Set(), Set(1), Set(2), Set(1, 2))
      if !leftProject.contains(1) || rightProject.contains(1)
    } yield Array[Any](lParts, rParts, j, p, leftProject, rightProject)

  @Test(dataProvider = "join")
  def testTableJoin(
    lParts: Int,
    rParts: Int,
    joinType: String,
    pred: Row => Boolean,
    leftProject: Set[Int],
    rightProject: Set[Int]
  ) {
    val (leftType, leftProjectF) = rowType.filter(f => !leftProject.contains(f.index))
    val left = TableKeyBy(
      TableParallelize(
        Literal(
          TStruct("rows" -> TArray(leftType), "global" -> TStruct.empty),
          Row(leftData.map(leftProjectF.asInstanceOf[Row => Row]), Row())),
        Some(lParts)),
      if (!leftProject.contains(1)) FastIndexedSeq("A", "B") else FastIndexedSeq("A"))

    val (rightType, rightProjectF) = rowType.filter(f => !rightProject.contains(f.index))
    val right = TableKeyBy(
      TableParallelize(
        Literal(
          TStruct("rows" -> TArray(rightType), "global" -> TStruct.empty),
          Row(rightData.map(rightProjectF.asInstanceOf[Row => Row]), Row())),
        Some(rParts)),
      if (!rightProject.contains(1)) FastIndexedSeq("A", "B") else FastIndexedSeq("A"))

    val (_, joinProjectF) = joinedType.filter(f => !leftProject.contains(f.index) && !rightProject.contains(f.index - 2))
    val joined = collect(
      TableJoin(
        left,
        TableRename(
          right,
          Array("A", "B", "C")
            .filter(right.typ.rowType.hasField)
            .map(a => a -> (a + "_"))
            .toMap,
          Map.empty),
        joinType, 1))

    assertEvalsTo(joined, Row(expectedOuterJoin.filter(pred).map(joinProjectF).toFastIndexedSeq, Row()))
  }

  @DataProvider(name = "union")
  def unionData(): Array[Array[Any]] =
    for {
      lParts <- Array[Integer](1, 2, 3)
      rParts <- Array[Integer](1, 2, 3)
    } yield Array[Any](lParts, rParts)

  @Test(dataProvider = "union")
  def testTableUnion(lParts: Int, rParts: Int) {
    val left = TableKeyBy(
      TableParallelize(
        Literal(
          TStruct("rows" -> TArray(rowType), "global" -> TStruct.empty),
          Row(leftData, Row())),
        Some(lParts)),
      FastIndexedSeq("A", "B"))

    val right = TableKeyBy(
      TableParallelize(
        Literal(
          TStruct("rows" -> TArray(rowType), "global" -> TStruct.empty),
          Row(rightData, Row())),
        Some(rParts)),
      FastIndexedSeq("A", "B"))

    val merged = collect(TableUnion(FastIndexedSeq(left, right)))

    assertEvalsTo(merged, Row(expectedUnion.toFastIndexedSeq, Row()))
  }

  @Test(dataProvider = "union")
  def testTableMultiWayZipJoin(lParts: Int, rParts: Int) {
    implicit val execStrats = Set(ExecStrategy.LoweredJVMCompile)
    val left = TableKeyBy(
      TableParallelize(
        Literal(
          TStruct("rows" -> TArray(rowType), "global" -> TStruct.empty),
          Row(leftData, Row())),
        Some(lParts)),
      FastIndexedSeq("A", "B"))

    val right = TableKeyBy(
      TableParallelize(
        Literal(
          TStruct("rows" -> TArray(rowType), "global" -> TStruct.empty),
          Row(rightData, Row())),
        Some(rParts)),
      FastIndexedSeq("A", "B"))

    val merged = collect(TableMultiWayZipJoin(FastIndexedSeq(left, right), "row", "global"))

    assertEvalsTo(merged, Row(expectedZipJoin.toFastIndexedSeq, Row(FastIndexedSeq(Row(), Row()))))
  }

  // Catches a bug in the partitioner created by the importer.
  @Test def testTableJoinOfImport() {
    val mt = importVCF(ctx, "src/test/resources/sample.vcf")
    var t: TableIR = MatrixRowsTable(mt)
    t = TableMapRows(t, SelectFields(Ref("row", t.typ.rowType), Seq("locus", "alleles")))
    val join: TableIR = TableJoin(t, t, "inner", 2)
    assertEvalsTo(TableCount(join), 346L)
  }

  @Test def testTableKeyBy() {
    implicit val execStrats = ExecStrategy.interpretOnly
    val data = Array(Array("A", 1), Array("A", 2), Array("B", 1))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("field1", TString), ("field2", TInt32))
    val keyNames = FastIndexedSeq("field1", "field2")
    val tt = TableType(rowType = signature, key = keyNames, globalType = TStruct.empty)
    val base = TableLiteral(
      TableValue(ctx, tt.rowType, tt.key, rdd))

    // construct the table with a longer key, then copy the table to shorten the key in type, but not rvd
    val distinctCount = TableCount(TableDistinct(TableLiteral(tt.copy(key = FastIndexedSeq("field1")), base.rvd, base.enc, base.encodedGlobals)))
    assertEvalsTo(distinctCount, 2L)
  }

  @Test def testTableKeyByLowering() {
    implicit val execStrats = ExecStrategy.lowering
    val t = TStruct("rows" -> TArray(TStruct("a" -> TInt32, "b" -> TString)), "global" -> TStruct("x" -> TString))
    val length = 10
    val value = Row(FastIndexedSeq(0 until length: _*).map(i => Row(0, "row" + i)), Row("global"))

    val par = TableParallelize(Literal(t, value))

    val keyed = TableKeyBy(par, IndexedSeq("a"), false)
    assertEvalsTo(TableCount(keyed), length.toLong)
  }

  @Test def testTableParallelize() {
    implicit val execStrats = ExecStrategy.allRelational
    val t = TStruct("rows" -> TArray(TStruct("a" -> TInt32, "b" -> TString)), "global" -> TStruct("x" -> TString))
    Array(1, 10, 17, 34, 103).foreach { length =>
      val value = Row(FastIndexedSeq(0 until length: _*).map(i => Row(i, "row" + i)), Row("global"))
      assertEvalsTo(
        collectNoKey(
          TableParallelize(
            Literal(
              t,
              value
            ))), value)
    }
  }

  @Test def testTableParallelizeCount() {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.allRelational
    val t = TStruct("rows" -> TArray(TStruct("a" -> TInt32, "b" -> TString)), "global" -> TStruct("x" -> TString))
    val value = Row(FastIndexedSeq(Row(0, "row1"), Row(1, "row2")), Row("glob"))

    assertEvalsTo(
      TableCount(
        TableParallelize(
          Literal(
            t,
            value
          ))),
      2L
    )
  }

  @Test def testTableHead(): Unit = {
    val t = TStruct("rows" -> TArray(TStruct("a" -> TInt32, "b" -> TString)), "global" -> TStruct("x" -> TString))
    def makeData(length: Int): Row = {
      Row(FastIndexedSeq(0 until length: _*).map(i => Row(i, "row" + i)), Row("global"))
    }
    val numRowsToTakeArray = Array(0, 4, 7, 12)
    val numInitialPartitionsArray = Array(1, 2, 6, 10, 13)
    val initialDataLength = 10
    val initialData = makeData(initialDataLength)

    numRowsToTakeArray.foreach { howManyRowsToTake =>
      val headData = makeData(Math.min(howManyRowsToTake, initialDataLength))
      numInitialPartitionsArray.foreach { howManyInitialPartitions =>
        assertEvalsTo(
          collectNoKey(
            TableHead(
              TableParallelize(
                Literal(t, initialData),
                Some(howManyInitialPartitions)
              ),
              howManyRowsToTake
            )
          ),
          headData)
      }
    }
  }

  @Test def testTableTail(): Unit = {
    val t = TStruct("rows" -> TArray(TStruct("a" -> TInt32, "b" -> TString)), "global" -> TStruct("x" -> TString))
    val numRowsToTakeArray = Array(0, 2, 7, 10, 12)
    val numInitialPartitionsArray = Array(1, 3, 6, 10, 13)
    val initialDataLength = 10
    def makeData(length: Int): Row = {
      Row(FastIndexedSeq((initialDataLength - length) until initialDataLength: _*).map(i => Row(i, "row" + i)), Row("global"))
    }
    val initialData = makeData(initialDataLength)


    numRowsToTakeArray.foreach { howManyRowsToTake =>
      val headData = makeData(Math.min(howManyRowsToTake, initialDataLength))
      numInitialPartitionsArray.foreach { howManyInitialPartitions =>
        assertEvalsTo(
          collectNoKey(
            TableTail(
              TableParallelize(
                Literal(t, initialData),
                Some(howManyInitialPartitions)
              ),
              howManyRowsToTake
            )
          ),
          headData)
      }
    }
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

  @Test def testTableRename(): Unit = {
    implicit val execStrats = ExecStrategy.lowering
    val t = TStruct("rows" -> TArray(TStruct("a" -> TInt32, "b" -> TString)), "global" -> TStruct(("x", TString), ("y", TInt32)))
    val value = Row(FastIndexedSeq(0 until 10: _*).map(i => Row(i, "row" + i)), Row("globalVal", 3))
    val adjustedValue = Row(FastIndexedSeq(0 until 10: _*).map(i => Row(i + 3, "row" + i)), Row("globalVal", 3))

    val renameIR =
      TableRename(
        TableParallelize(
          Literal(
            t,
            value
          )),
        Map[String, String]("a" -> "c"),
        Map[String, String]("y" -> "z")
      )

    val newRow = MakeStruct(Seq(
      ("foo", GetField(Ref("row", renameIR.typ.rowType), "c") + GetField(Ref("global", TStruct(("x", TString), ("z", TInt32))), "z")),
      ("bar", GetField(Ref("row", renameIR.typ.rowType), "b")))
    )
    val mapped = TableMapRows(renameIR, newRow)

    assertEvalsTo(
      collectNoKey(
        mapped
      ), adjustedValue)
  }

  @Test def testTableMapGlobals(): Unit = {
    val t = TStruct("rows" -> TArray(TStruct("a" -> TInt32, "b" -> TString)), "global" -> TStruct("x" -> TString))
    val innerRowRef = Ref("row", t.field("rows").typ.asInstanceOf[TArray].elementType)
    val innerGlobalRef = Ref("global", t.field("global").typ)
    val length = 10
    val value = Row(FastIndexedSeq(0 until length: _*).map(i => Row(i, "row" + i)), Row("global"))
    val modifedValue = Row(FastIndexedSeq(0 until length: _*).map(i => Row(i, "global")), Row("newGlobals"))
    assertEvalsTo(
      collectNoKey(
        TableMapGlobals(
          TableMapRows(
            TableParallelize(
              Literal(
                t,
                value
              )),
            MakeStruct(FastIndexedSeq("a" -> GetField(innerRowRef, "a"), "b" -> GetField(innerGlobalRef, "x")))
          ),
          MakeStruct(FastIndexedSeq("x" -> Str("newGlobals")))
        )
      ),
      modifedValue)
  }

  @Test def testTableWrite() {
    implicit val execStrats = ExecStrategy.interpretOnly
    val table = TableRange(5, 4)
    val path = ctx.createTmpPath("test-table-write", "ht")
    Interpret[Unit](ctx, TableWrite(table, TableNativeWriter(path)))
    val before = table.execute(ctx)
    val after = Interpret(TableIR.read(fs, path), ctx, false)
    assert(before.globals.javaValue == after.globals.javaValue)
    assert(before.rdd.collect().toFastIndexedSeq == after.rdd.collect().toFastIndexedSeq)
  }

  @Test def testPartitionCountsWithDropRows() {
    val tr = new TableReader {
      def pathsUsed: Seq[String] = FastSeq()

      override def apply(tr: TableRead, ctx: ExecuteContext): TableValue = ???

      override def partitionCounts: Option[IndexedSeq[Long]] = Some(FastIndexedSeq(1, 2, 3, 4))

      def rowAndGlobalPTypes(ctx: ExecuteContext, requestedType: TableType): (PStruct, PStruct) = ???

      override def fullType: TableType = TableType(TStruct.empty, FastIndexedSeq(), TStruct.empty)
    }
    val tir = TableRead(tr.fullType, true, tr)
    assert(tir.partitionCounts.forall(_.sum == 0))
  }

  @Test def testScanInAggInMapRows() {
    implicit val execStrats = ExecStrategy.interpretOnly
    val sumSig = AggSignature(Sum(), FastSeq(), FastSeq(TInt64))
    var tr: TableIR = TableRange(10, 3)
    tr = TableKeyBy(tr, FastIndexedSeq(), false)
    tr = TableMapRows(tr, InsertFields(Ref("row", tr.typ.rowType),
      FastSeq(("result",
        StreamAgg(
          StreamAggScan(
            StreamRange(0, GetField(Ref("row", tr.typ.rowType), "idx"), 1),
            "streamx",
            ApplyScanOp(FastIndexedSeq(), FastIndexedSeq(Ref("streamx", TInt32).toL), sumSig)),
          "aggx",
          ApplyAggOp(FastIndexedSeq(), FastIndexedSeq(Ref("aggx", TInt64)), sumSig))))))
    assertEvalsTo(TableCollect(tr), Row(IndexedSeq.tabulate(10) { i =>
      val r = (0 until i).map(_.toLong).scanLeft(0L)(_ + _).init.sum
      Row(i, r)
    }, Row()
    ))
  }

  @Test def testScanInAggInScanInMapRows() {
    implicit val execStrats = ExecStrategy.interpretOnly
    val sumSig = AggSignature(Sum(), FastSeq(), FastSeq(TInt64))
    var tr: TableIR = TableRange(10, 3)
    tr = TableKeyBy(tr, FastIndexedSeq(), false)
    tr = TableMapRows(tr, InsertFields(Ref("row", tr.typ.rowType),
      FastSeq(("result",
        ApplyScanOp(FastIndexedSeq(),
          FastIndexedSeq(StreamAgg(
            StreamAggScan(
              StreamRange(0, GetField(Ref("row", tr.typ.rowType), "idx"), 1),
              "streamx",
              ApplyScanOp(FastIndexedSeq(), FastIndexedSeq(Ref("streamx", TInt32).toL), sumSig)),
            "aggx",
            ApplyAggOp(FastIndexedSeq(), FastIndexedSeq(Ref("aggx", TInt64)), sumSig))),
          sumSig)))))
    assertEvalsTo(TableCollect(tr), Row(Array.tabulate(10) { i =>
      (0 until i).map(_.toLong).scanLeft(0L)(_ + _).init.sum
    }.scanLeft(0L)(_ + _)
      .zipWithIndex
      .map { case (x, idx) => Row(idx, x) }.init.toFastIndexedSeq,
      Row()
    ))
  }

  @Test def testTableGroupWithinPartitions(): Unit = {
    val t = TStruct("rows" -> TArray(TStruct("a" -> TInt32, "b" -> TString)), "global" -> TStruct("x" -> TString))
    val length = 6
    val value = Row(FastIndexedSeq(0 until length: _*).map(i => Row(i, "row" + i)), Row("global"))
    val rowsData = FastIndexedSeq(
      Row(FastIndexedSeq(
        Row(0, "row0"),
        Row(1, "row1")
      )),
      Row(FastIndexedSeq(
        Row(2, "row2")
      )),
      Row(FastIndexedSeq(
        Row(3, "row3"),
        Row(4, "row4")
      )),
      Row(FastIndexedSeq(
        Row(5, "row5")
      ))
    )
    val ans = Row(rowsData, Row("global"))
    assertEvalsTo(
      collectNoKey(
        TableGroupWithinPartitions(
          TableParallelize(
            Literal(t, value),
            Some(2)
          ),
          "grouped_fields",
          2
        )
      ),
      ans)
  }

  @Test def testTableAggregateByKey(): Unit = {
    implicit val execStrats = ExecStrategy.interpretOnly  // FIXME: requires method splitting resolution to make allRelational
    var tir: TableIR = TableRead.native(fs, "src/test/resources/three_key.ht")
    tir = TableKeyBy(tir, FastIndexedSeq("x", "y"), true)
    tir = TableAggregateByKey(tir, MakeStruct(FastSeq(
      ("sum", ApplyAggOp(FastIndexedSeq(), FastIndexedSeq(GetField(Ref("row", tir.typ.rowType), "z").toL), AggSignature(Sum(), FastIndexedSeq(), FastIndexedSeq(TInt64)))),
      ("n", ApplyAggOp(FastIndexedSeq(), FastIndexedSeq(), AggSignature(Count(), FastIndexedSeq(), FastIndexedSeq())))
    )))
    val ir = GetField(TableCollect(TableKeyBy(tir, FastIndexedSeq())), "rows")
    assertEvalsTo(ir, (0 until 10).flatMap(i => (0 until i).map(j => Row(i, j, (0 until j).sum.toLong, j.toLong))).filter(_.getAs[Long](3) > 0))
  }

  @Test def testTableDistinct(): Unit = {
    val tir: TableIR = TableRead.native(fs, "src/test/resources/three_key.ht")
    val keyedByX = TableKeyBy(tir, FastIndexedSeq("x"), true)
    val distinctByX = TableDistinct(keyedByX)
    assertEvalsTo(TableCount(distinctByX), 8L)
    assertEvalsTo(collect(distinctByX), Row(FastIndexedSeq(2 to 9: _*).map(i => Row(i, 1, 0)), Row()))

    val keyedByXAndY = TableKeyBy(tir, FastIndexedSeq("x", "y"), true)
    val distinctByXAndY = TableDistinct(keyedByXAndY)
    assertEvalsTo(TableCount(distinctByXAndY), 36L)

    val distinctByAll = TableDistinct(tir)
    assertEvalsTo(TableCount(distinctByAll), 120L)
  }

  @Test def testRangeOrderByDescending() {
    var tir: TableIR = TableRange(10, 3)
    tir = TableOrderBy(tir, FastIndexedSeq(SortField("idx", Descending)))
    val x = GetField(TableCollect(tir), "rows")

    assertEvalsTo(x, (0 until 10).reverse.map(i => Row(i)))(ExecStrategy.allRelational)
  }

  @Test def testTableLeftJoinRightDistinctRangeTables(): Unit = {
    IndexedSeq((1, 1), (3, 2), (10, 5), (5, 10)).foreach { case(nParts1, nParts2) =>
      val rangeTable1 = TableRange(10, nParts1)
      var rangeTable2: TableIR = TableRange(5, nParts2)
      val row = Ref("row", rangeTable2.typ.rowType)
      rangeTable2 = TableMapRows(rangeTable2, InsertFields(row, FastIndexedSeq("x" -> GetField(row, "idx"))))
      val joinedRanges = TableLeftJoinRightDistinct(rangeTable1, rangeTable2, "foo")
      assertEvalsTo(TableCount(joinedRanges), 10L)

      val expectedJoinCollectResult = Row(
        (0 until 5).map(i => Row(FastIndexedSeq(i, Row(i)): _*)) ++ (5 until 10).map(i => Row(FastIndexedSeq(i, null): _*)),
        Row())
      assertEvalsTo(collect(joinedRanges), expectedJoinCollectResult)
    }
  }

  val parTable1Length = 7
  val parTable1Type = TStruct("rows" -> TArray(TStruct("a1" -> TString, "b1" -> TInt32, "c1" -> TString)), "global" -> TStruct("x" -> TString))
  val value1 = Row(FastIndexedSeq(0 until parTable1Length: _*).map(i => Row("row" + i, i * i, s"t1_${i}")), Row("global"))
  val table1 = TableParallelize(Literal(parTable1Type, value1), Some(2))

  val parTable2Length = 9
  val parTable2Type = TStruct("rows" -> TArray(TStruct("a2" -> TString, "b2" -> TInt32, "c2" -> TString)), "global" -> TStruct("y"-> TInt32))
  val value2 = Row(FastIndexedSeq(0 until parTable2Length: _*).map(i => Row("row" + i, -2 * i, s"t2_${i}")), Row(15))
  val table2 = TableParallelize(Literal(parTable2Type, value2), Some(3))

  val table1KeyedByA = TableKeyBy(table1, IndexedSeq("a1"))
  val table2KeyedByA = TableKeyBy(table2, IndexedSeq("a2"))
  val joinedParKeyedByA = TableLeftJoinRightDistinct(table1KeyedByA, table2KeyedByA, "joinRoot")

  @Test def testTableLeftJoinRightDistinctParallelizeSameKey(): Unit = {
    assertEvalsTo(TableCount(table1KeyedByA), parTable1Length.toLong)
    assertEvalsTo(TableCount(table2KeyedByA), parTable2Length.toLong)

    assertEvalsTo(TableCount(joinedParKeyedByA), parTable1Length.toLong)
    assertEvalsTo(collect(joinedParKeyedByA), Row(FastIndexedSeq(0 until parTable1Length: _*).map(i =>
      Row("row" + i, i * i, s"t1_${i}", Row(-2 * i, s"t2_${i}"))), Row("global"))
    )
  }

  @Test def testTableLeftJoinRightDistinctParallelizePrefixKey(): Unit = {
    val table1KeyedByAAndB = TableKeyBy(table1, IndexedSeq("a1", "b1"))
    val joinedParKeyedByAAndB = TableLeftJoinRightDistinct(table1KeyedByAAndB, table2KeyedByA, "joinRoot")

    assertEvalsTo(TableCount(joinedParKeyedByAAndB), parTable1Length.toLong)
    assertEvalsTo(collect(joinedParKeyedByAAndB), Row(FastIndexedSeq(0 until parTable1Length: _*).map(i =>
      Row("row" + i, i * i, s"t1_${i}", Row(-2 * i, s"t2_${i}"))), Row("global"))
    )
  }

  @Test def testTableKeyByAndAggregate(): Unit = {
    implicit val execStrats = ExecStrategy.interpretOnly //FIXME: Lowering is implemented, will work when method splitting is fixed.
    val tir: TableIR = TableRead.native(fs, "src/test/resources/three_key.ht")
    val unkeyed = TableKeyBy(tir, IndexedSeq[String]())
    val rowRef = Ref("row", unkeyed.typ.rowType)
    val aggSignature = AggSignature(Sum(), FastIndexedSeq(), FastIndexedSeq(TInt64))
    val aggExpression = MakeStruct(FastSeq("y_sum" -> ApplyAggOp(FastIndexedSeq(), FastIndexedSeq(Cast(GetField(rowRef, "y"), TInt64)), aggSignature)))
    val keyByXAndAggregateSum = TableKeyByAndAggregate(unkeyed, aggExpression, MakeStruct(FastSeq("x" -> GetField(rowRef, "x"))))

    assertEvalsTo(
      collect(keyByXAndAggregateSum),
      Row(FastIndexedSeq(Row(2, 1L), Row(3,5L), Row(4, 14L), Row(5, 30L), Row(6, 55L), Row(7, 91L), Row(8, 140L), Row(9, 204L)), Row())
    )

    // Keying by a newly computed field.
    val keyByXPlusTwoAndAggregateSum = TableKeyByAndAggregate(unkeyed, aggExpression, MakeStruct(FastSeq("xPlusTwo" -> (GetField(rowRef, "x") + 2))))
    assertEvalsTo(
      collect(keyByXPlusTwoAndAggregateSum),
      Row(FastIndexedSeq(Row(4, 1L), Row(5,5L), Row(6, 14L), Row(7, 30L), Row(8, 55L), Row(9, 91L), Row(10, 140L), Row(11, 204L)), Row())
    )

    // Keying by just Z when original is keyed by x,y,z, naming it x anyway.
    val keyByZAndAggregateSum =  TableKeyByAndAggregate(tir, aggExpression, MakeStruct(FastSeq("x" -> GetField(rowRef, "z"))))
    assertEvalsTo(
      collect(keyByZAndAggregateSum),
      Row(FastIndexedSeq(Row(0, 120L), Row(1, 112L), Row(2, 98L), Row(3, 80L), Row(4, 60L), Row(5, 40L), Row(6, 22L), Row(7, 8L)), Row())
    )
  }

  @Test def testTableAggregateCollectAndTake(): Unit = {
    implicit val execStrats = ExecStrategy.allRelational
    var tir: TableIR = TableRange(10, 3)
    tir = TableMapRows(tir, InsertFields(Ref("row", tir.typ.rowType), FastSeq("aStr" -> Str("foo"))))
    val x = TableAggregate(tir,
      MakeTuple.ordered(FastSeq(
        ApplyAggOp(Collect())(Ref("row", tir.typ.rowType)),
        ApplyAggOp(Take(), I32(5))(GetField(Ref("row", tir.typ.rowType), "idx"))
      )))

    assertEvalsTo(x, Row(
      (0 until 10).map(i => Row(i, "foo")),
      0 until 5))
  }

  @Test def testIssue9016() {
    val rows = mapIR(ToStream(MakeArray(makestruct("a" -> MakeTuple.ordered(FastSeq(I32(0), I32(1))))))) { row =>
      If(IsNA(row),
        NA(TStruct("a" -> TTuple(FastSeq(TupleField(1, TInt32))))),
        makestruct("a" -> bindIR(GetField(row, "a")) { a =>
          If(IsNA(a), NA(TTuple(FastSeq(TupleField(1, TInt32)))), MakeTuple(FastSeq(1 -> GetTupleElement(a, 1))))
        }))
    }
    val table = TableParallelize(makestruct("rows" -> ToArray(rows), "global" -> makestruct()), None)
    assertEvalsTo(TableCollect(table), Row(FastIndexedSeq(Row(Row(1))), Row()))

  }
}
