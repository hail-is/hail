package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.annotations.SafeNDArray
import is.hail.expr.Nat
import is.hail.expr.ir.TestUtils._
import is.hail.expr.ir.analyses.PartitionCounts
import is.hail.expr.ir.defs._
import is.hail.expr.ir.lowering.{DArrayLowering, ExecuteRelational, LowerTableIR}
import is.hail.methods.{ForceCountTable, NPartitionsTable}
import is.hail.rvd.RVDPartitioner
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.Locus

import org.apache.spark.sql.Row
import org.scalatest
import org.scalatest.{Failed, Succeeded}
import org.scalatest.Inspectors.forAll
import org.scalatest.enablers.InspectorAsserting.assertingNatureOfAssertion
import org.testng.annotations.{DataProvider, Test}

class TableIRSuite extends HailSuite {

  implicit val execStrats: Set[ExecStrategy] =
    Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized, ExecStrategy.LoweredJVMCompile)

  @Test def testRangeCount(): scalatest.Assertion = {
    val node1 = TableCount(TableRange(10, 2))
    val node2 = TableCount(TableRange(15, 5))
    val node = ApplyBinaryPrimOp(Add(), node1, node2)
    assertEvalsTo(node1, 10L)
    assertEvalsTo(node2, 15L)
    assertEvalsTo(node, 25L)
  }

  @Test def testForceCount(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.interpretOnly
    val tableRangeSize = Int.MaxValue / 20
    val forceCountRange = TableToValueApply(TableRange(tableRangeSize, 2), ForceCountTable())
    assertEvalsTo(forceCountRange, tableRangeSize.toLong)
  }

  @Test def testRangeRead(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.lowering
    val original = TableKeyBy(
      TableMapGlobals(TableRange(10, 3), MakeStruct(FastSeq("foo" -> I32(57)))),
      FastSeq(),
    )

    val path = ctx.createTmpPath("test-range-read", "ht")
    val write = TableWrite(original, TableNativeWriter(path, overwrite = true))
    assertEvalsTo(write, ())
    val read = TableIR.read(fs, path, false, None)
    val droppedRows = TableIR.read(fs, path, true, None)

    val uids = for {
      (partSize, partIndex) <- partition(10, 3).zipWithIndex
      i <- 0 until partSize
    } yield Row(partIndex.toLong, i.toLong)
    val expectedRows = (0 until 10, uids).zipped.map((i, uid) => Row(i, uid))
    val expectedGlobals = Row(57)

    assertEvalsTo(TableCollect(read), Row(expectedRows, expectedGlobals))
    assertEvalsTo(TableCollect(droppedRows), Row(FastSeq(), expectedGlobals))
  }

  @Test def testCountRead(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.lowering
    val tir: TableIR = TableRead.native(fs, getTestResource("three_key.ht"))
    assertEvalsTo(TableCount(tir), 120L)
  }

  @Test def testRangeCollect(): scalatest.Assertion = {
    implicit val execStrats = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized)
    val t = TableRange(10, 2)
    val row = Ref(TableIR.rowName, t.typ.rowType)
    val node = collect(TableMapRows(t, InsertFields(row, FastSeq("x" -> GetField(row, "idx")))))
    assertEvalsTo(collect(t), Row(Array.tabulate(10)(Row(_)).toFastSeq, Row()))
    assertEvalsTo(node, Row(Array.tabulate(10)(i => Row(i, i)).toFastSeq, Row()))
  }

  @Test def testNestedRangeCollect(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.allRelational

    val r = TableRange(2, 2)

    val tc = GetField(collect(TableRange(2, 2)), "rows")
    val m = TableMapRows(
      TableRange(2, 2),
      InsertFields(Ref(TableIR.rowName, r.typ.rowType), FastSeq("collected" -> tc)),
    )
    assertEvalsTo(
      collect(m),
      Row(
        FastSeq(
          Row(0, FastSeq(Row(0), Row(1))),
          Row(1, FastSeq(Row(0), Row(1))),
        ),
        Row(),
      ),
    )
  }

  @Test def testRangeSum(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.interpretOnly
    val t = TableRange(10, 2)
    val row = Ref(TableIR.rowName, t.typ.rowType)
    val sum = AggSignature(Sum(), FastSeq(), FastSeq(TInt64))
    val node = collect(TableMapRows(
      t,
      InsertFields(
        row,
        FastSeq("sum" -> ApplyScanOp(FastSeq(), FastSeq(Cast(GetField(row, "idx"), TInt64)), sum)),
      ),
    ))
    assertEvalsTo(
      node,
      Row(Array.tabulate(10)(i => Row(i, Array.range(0, i).sum.toLong)).toFastSeq, Row()),
    )
  }

  @Test def testGetGlobals(): scalatest.Assertion = {
    implicit val execStrats = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized)
    val t = TableRange(10, 2)
    val newGlobals =
      InsertFields(Ref(TableIR.globalName, t.typ.globalType), FastSeq("x" -> collect(t)))
    val node = TableGetGlobals(TableMapGlobals(t, newGlobals))
    assertEvalsTo(node, Row(Row(Array.tabulate(10)(i => Row(i)).toFastSeq, Row())))
  }

  @Test def testCollectGlobals(): scalatest.Assertion = {
    implicit val execStrats = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized)
    val t = TableRange(10, 2)
    val newGlobals =
      InsertFields(Ref(TableIR.globalName, t.typ.globalType), FastSeq("x" -> collect(t)))
    val node = TableMapRows(
      TableMapGlobals(t, newGlobals),
      InsertFields(
        Ref(TableIR.rowName, t.typ.rowType),
        FastSeq("x" -> GetField(Ref(TableIR.globalName, newGlobals.typ), "x")),
      ),
    )

    val collectedT = Row(Array.tabulate(10)(i => Row(i)).toFastSeq, Row())
    val expected = Array.tabulate(10)(i => Row(i, collectedT)).toFastSeq

    assertEvalsTo(collect(node), Row(expected, Row(collectedT)))
  }

  @Test def testRangeExplode(): scalatest.Assertion = {
    implicit val execStrats = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized)
    val t = TableRange(10, 2)
    val row = Ref(TableIR.rowName, t.typ.rowType)

    val t2 = TableMapRows(
      t,
      InsertFields(row, FastSeq("x" -> ToArray(StreamRange(0, GetField(row, "idx"), 1)))),
    )
    val node = TableExplode(t2, FastSeq("x"))
    val expected = Array.range(0, 10).flatMap(i => Array.range(0, i).map(Row(i, _))).toFastSeq
    assertEvalsTo(collect(node), Row(expected, Row()))

    val t3 = TableMapRows(
      t,
      InsertFields(
        row,
        FastSeq("x" ->
          MakeStruct(FastSeq("y" -> ToArray(StreamRange(0, GetField(row, "idx"), 1))))),
      ),
    )
    val node2 = TableExplode(t3, FastSeq("x", "y"))
    val expected2 =
      Array.range(0, 10).flatMap(i => Array.range(0, i).map(j => Row(i, Row(j)))).toFastSeq
    assertEvalsTo(collect(node2), Row(expected2, Row()))
  }

  @Test def testFilter(): scalatest.Assertion = {
    implicit val execStrats = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized)
    val t = TableRange(10, 2)
    val node = TableFilter(
      TableMapGlobals(
        t,
        MakeStruct(FastSeq("x" -> GetField(ArrayRef(GetField(collect(t), "rows"), 4), "idx"))),
      ),
      ApplyComparisonOp(
        EQ(TInt32),
        GetField(Ref(TableIR.rowName, t.typ.rowType), "idx"),
        GetField(Ref(TableIR.globalName, TStruct("x" -> TInt32)), "x"),
      ),
    )

    val expected = Array.tabulate(10)(Row(_)).filter(_.get(0) == 4).toFastSeq

    assertEvalsTo(collect(node), Row(expected, Row(4)))
  }

  @Test def testFilterIntervals(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.allRelational

    def assertFilterIntervals(
      intervals: IndexedSeq[Interval],
      keep: Boolean,
      expected: IndexedSeq[Int],
    ): scalatest.Assertion = {
      var t: TableIR = TableRange(10, 5)
      t = TableFilterIntervals(
        t,
        intervals.map(i => Interval(Row(i.start), Row(i.end), i.includesStart, i.includesEnd)),
        keep,
      )
      assertEvalsTo(GetField(collect(t), "rows"), expected.map(Row(_)))
    }

    assertFilterIntervals(
      FastSeq(Interval(0, 5, true, false)),
      true,
      FastSeq(0, 1, 2, 3, 4),
    )

    assertFilterIntervals(
      FastSeq(Interval(0, 5, true, false)),
      false,
      FastSeq(5, 6, 7, 8, 9),
    )

    assertFilterIntervals(
      FastSeq(),
      true,
      FastSeq(),
    )

    assertFilterIntervals(
      FastSeq(),
      false,
      FastSeq(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    )

    assertFilterIntervals(
      FastSeq(),
      true,
      FastSeq(),
    )

    assertFilterIntervals(
      FastSeq(
        Interval(0, 5, true, false),
        Interval(1, 6, false, true),
        Interval(8, 9, true, false),
      ),
      false,
      FastSeq(7, 9),
    )

    assertFilterIntervals(
      FastSeq(
        Interval(0, 5, true, false),
        Interval(1, 6, false, true),
        Interval(8, 9, true, false),
      ),
      true,
      FastSeq(0, 1, 2, 3, 4, 5, 6, 8),
    )
  }

  @Test def testTableMapWithLiterals(): scalatest.Assertion = {
    implicit val execStrats = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized)
    val t = TableRange(10, 2)
    val node = TableMapRows(
      t,
      InsertFields(
        Ref(TableIR.rowName, t.typ.rowType),
        FastSeq(
          "a" -> Str("foo"),
          "b" -> Literal(TTuple(TInt32, TString), Row(1, "hello")),
        ),
      ),
    )

    val expected = Array.tabulate(10)(Row(_, "foo", Row(1, "hello"))).toFastSeq
    assertEvalsTo(collect(node), Row(expected, Row()))
  }

  @Test def testScanCountBehavesLikeIndex(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.interpretOnly
    val t = rangeKT
    val oldRow = Ref(TableIR.rowName, t.typ.rowType)

    val newRow = InsertFields(oldRow, FastSeq("idx2" -> IRScanCount))
    val newTable = TableMapRows(t, newRow)
    val expected = Array.tabulate(20)(i => Row(i, i.toLong)).toFastSeq
    assertEvalsTo(
      ArraySort(
        ToStream(TableAggregate(newTable, IRAggCollect(Ref(TableIR.rowName, newRow.typ)))),
        True(),
      ),
      expected,
    )
  }

  @Test def testScanCollectBehavesLikeRange(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.interpretOnly
    val t = rangeKT
    val oldRow = Ref(TableIR.rowName, t.typ.rowType)

    val newRow = InsertFields(oldRow, FastSeq("range" -> IRScanCollect(GetField(oldRow, "idx"))))
    val newTable = TableMapRows(t, newRow)

    val expected = Array.tabulate(20)(i => Row(i, Array.range(0, i).toFastSeq)).toFastSeq
    assertEvalsTo(
      ArraySort(
        ToStream(TableAggregate(newTable, IRAggCollect(Ref(TableIR.rowName, newRow.typ)))),
        True(),
      ),
      expected,
    )
  }

  val rowType = TStruct(("A", TInt32), ("B", TInt32), ("C", TInt32))

  val joinedType =
    TStruct(("A", TInt32), ("B", TInt32), ("C", TInt32), ("B_1", TInt32), ("C_1", TInt32))

  val kType = TStruct(("A", TInt32), ("B", TInt32))

  val leftData = FastSeq(
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
    (37, 2, -1),
  ).map(Row.fromTuple)

  val rightData = FastSeq(
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
    (41, 2, 1),
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
    (41, 2, 1),
  ).map(Row.fromTuple)

  val expectedZipJoin = Array(
    (3, 1, FastSeq(Row(-1), null)),
    (3, 2, FastSeq(Row(-1), null)),
    (6, 1, FastSeq(null, Row(1))),
    (6, 2, FastSeq(null, Row(1))),
    (11, 1, FastSeq(Row(-1), null)),
    (11, 2, FastSeq(Row(-1), null)),
    (16, 1, FastSeq(Row(-1), null)),
    (16, 2, FastSeq(Row(-1), null)),
    (17, 1, FastSeq(Row(-1), Row(1))),
    (17, 2, FastSeq(Row(-1), Row(1))),
    (18, 1, FastSeq(null, Row(1))),
    (18, 2, FastSeq(null, Row(1))),
    (21, 1, FastSeq(null, Row(1))),
    (21, 2, FastSeq(null, Row(1))),
    (22, 1, FastSeq(Row(-1), Row(1))),
    (22, 2, FastSeq(Row(-1), Row(1))),
    (23, 1, FastSeq(Row(-1), null)),
    (23, 2, FastSeq(Row(-1), null)),
    (26, 1, FastSeq(Row(-1), null)),
    (26, 2, FastSeq(Row(-1), null)),
    (27, 1, FastSeq(Row(-1), Row(1))),
    (27, 2, FastSeq(Row(-1), Row(1))),
    (28, 1, FastSeq(null, Row(1))),
    (28, 2, FastSeq(null, Row(1))),
    (31, 1, FastSeq(null, Row(1))),
    (31, 2, FastSeq(null, Row(1))),
    (32, 1, FastSeq(Row(-1), Row(1))),
    (32, 2, FastSeq(Row(-1), Row(1))),
    (33, 1, FastSeq(Row(-1), null)),
    (33, 2, FastSeq(Row(-1), null)),
    (36, 1, FastSeq(Row(-1), null)),
    (36, 2, FastSeq(Row(-1), null)),
    (37, 1, FastSeq(Row(-1), Row(1))),
    (37, 2, FastSeq(Row(-1), Row(1))),
    (38, 1, FastSeq(null, Row(1))),
    (38, 2, FastSeq(null, Row(1))),
    (41, 1, FastSeq(null, Row(1))),
    (41, 2, FastSeq(null, Row(1))),
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
    (41, null, null, 2, 1),
  ).map(Row.fromTuple)

  val joinTypes = Array(
    ("outer", (row: Row) => true),
    ("left", (row: Row) => !row.isNullAt(1)),
    ("right", (row: Row) => !row.isNullAt(3)),
    ("inner", (row: Row) => !row.isNullAt(1) && !row.isNullAt(3)),
  )

  @DataProvider(name = "join")
  def joinData(): Array[Array[Any]] = {
    val defaultLParts = 2
    val defaultRParts = 2
    val defaultLeftProject = Set(1, 2)
    val defaultRightProject = Set(1, 2)

    val ab = new BoxedArrayBuilder[Array[Any]]()
    for ((j, p) <- joinTypes) {
      for {
        lParts <- Array[Integer](1, 2, 3)
        rParts <- Array[Integer](1, 2, 3)
      }
        ab += Array[Any](lParts, rParts, j, p, defaultLeftProject, defaultRightProject)

      for {
        leftProject <- Seq[Set[Int]](Set(), Set(1), Set(2), Set(1, 2))
        rightProject <- Seq[Set[Int]](Set(), Set(1), Set(2), Set(1, 2))
        if !leftProject.contains(1) || rightProject.contains(1)
      }
        ab += Array[Any](defaultLParts, defaultRParts, j, p, leftProject, rightProject)
    }
    ab.result()
  }

  @Test(dataProvider = "join")
  def testTableJoin(
    lParts: Int,
    rParts: Int,
    joinType: String,
    pred: Row => Boolean,
    leftProject: Set[Int],
    rightProject: Set[Int],
  ): scalatest.Assertion = {
    val (leftType, leftProjectF) = rowType.filter(f => !leftProject.contains(f.index))
    val left = TableKeyBy(
      TableParallelize(
        Literal(
          TStruct("rows" -> TArray(leftType), "global" -> TStruct.empty),
          Row(leftData.map(leftProjectF.asInstanceOf[Row => Row]), Row()),
        ),
        Some(lParts),
      ),
      if (!leftProject.contains(1)) FastSeq("A", "B") else FastSeq("A"),
    )

    val (rightType, rightProjectF) = rowType.filter(f => !rightProject.contains(f.index))
    val right = TableKeyBy(
      TableParallelize(
        Literal(
          TStruct("rows" -> TArray(rightType), "global" -> TStruct.empty),
          Row(rightData.map(rightProjectF.asInstanceOf[Row => Row]), Row()),
        ),
        Some(rParts),
      ),
      if (!rightProject.contains(1)) FastSeq("A", "B") else FastSeq("A"),
    )

    val (_, joinProjectF) =
      joinedType.filter(f => !leftProject.contains(f.index) && !rightProject.contains(f.index - 2))
    val joined = collect(
      TableJoin(
        left,
        TableRename(
          right,
          Array("A", "B", "C")
            .filter(right.typ.rowType.hasField)
            .map(a => a -> (a + "_"))
            .toMap,
          Map.empty,
        ),
        joinType,
        1,
      )
    )

    assertEvalsTo(joined, Row(expectedOuterJoin.filter(pred).map(joinProjectF).toFastSeq, Row()))
  }

  @DataProvider(name = "union")
  def unionData(): Array[Array[Any]] =
    for {
      lParts <- Array[Integer](1, 2, 3)
      rParts <- Array[Integer](1, 2, 3)
    } yield Array[Any](lParts, rParts)

  @Test(dataProvider = "union")
  def testTableUnion(lParts: Int, rParts: Int): scalatest.Assertion = {
    val left = TableKeyBy(
      TableParallelize(
        Literal(
          TStruct("rows" -> TArray(rowType), "global" -> TStruct.empty),
          Row(leftData, Row()),
        ),
        Some(lParts),
      ),
      FastSeq("A", "B"),
    )

    val right = TableKeyBy(
      TableParallelize(
        Literal(
          TStruct("rows" -> TArray(rowType), "global" -> TStruct.empty),
          Row(rightData, Row()),
        ),
        Some(rParts),
      ),
      FastSeq("A", "B"),
    )

    val merged = collect(TableUnion(FastSeq(left, right)))

    assertEvalsTo(merged, Row(expectedUnion.toFastSeq, Row()))
  }

  @Test(dataProvider = "union")
  def testTableMultiWayZipJoin(lParts: Int, rParts: Int): scalatest.Assertion = {
    implicit val execStrats = Set(ExecStrategy.LoweredJVMCompile)
    val left = TableKeyBy(
      TableParallelize(
        Literal(
          TStruct("rows" -> TArray(rowType), "global" -> TStruct.empty),
          Row(leftData, Row()),
        ),
        Some(lParts),
      ),
      FastSeq("A", "B"),
    )

    val right = TableKeyBy(
      TableParallelize(
        Literal(
          TStruct("rows" -> TArray(rowType), "global" -> TStruct.empty),
          Row(rightData, Row()),
        ),
        Some(rParts),
      ),
      FastSeq("A", "B"),
    )

    val merged = collect(TableMultiWayZipJoin(FastSeq(left, right), "row", "global"))

    assertEvalsTo(merged, Row(expectedZipJoin.toFastSeq, Row(FastSeq(Row(), Row()))))
  }

  // Catches a bug in the partitioner created by the importer.
  @Test def testTableJoinOfImport(): scalatest.Assertion = {
    val mnr = MatrixNativeReader(fs, getTestResource("sample.vcf.mt"))
    val mt2 = MatrixRead(mnr.fullMatrixType, false, false, mnr)
    val t2 = MatrixRowsTable(mt2)
    val mt = importVCF(ctx, getTestResource("sample.vcf"))
    var t: TableIR = MatrixRowsTable(mt)
    t = TableMapRows(
      t,
      SelectFields(Ref(TableIR.rowName, t.typ.rowType), FastSeq("locus", "alleles")),
    )
    val join: TableIR = TableJoin(t, t2, "inner", 2)
    assertEvalsTo(TableCount(join), 346L)
  }

  @Test def testNativeReaderWithOverlappingPartitions(): scalatest.Assertion = {
    val path = getTestResource("sample.vcf-20-partitions-with-overlap.mt/rows")
    // i1 overlaps the first two partitions
    val i1 = Interval(Row(Locus("20", 10200000)), Row(Locus("20", 10500000)), true, true)

    def test(filterIntervals: Boolean, expectedNParts: Int): scalatest.Assertion = {
      val opts = NativeReaderOptions(FastSeq(i1), TLocus("GRCh37"), filterIntervals)
      val tr = TableNativeReader(fs, TableNativeReaderParameters(path, Some(opts)))
      val tir = TableRead(tr.fullTypeWithoutUIDs, false, tr)
      val nParts = TableToValueApply(tir, NPartitionsTable())
      val count = TableToValueApply(tir, ForceCountTable())
      assertEvalsTo(nParts, expectedNParts)
      assertEvalsTo(count, 20L)
    }

    test(false, 1)
    test(true, 2)
  }

  @Test def testTableKeyBy(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.interpretOnly
    val data = Array(Array("A", 1), Array("A", 2), Array("B", 1))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("field1", TString), ("field2", TInt32))
    val keyNames = FastSeq("field1", "field2")
    val tt = TableType(rowType = signature, key = keyNames, globalType = TStruct.empty)
    val base = TableLiteral(
      TableValue(ctx, tt.rowType, tt.key, rdd),
      theHailClassLoader,
    )

    /* construct the table with a longer key, then copy the table to shorten the key in type, but
     * not rvd */
    val distinctCount = TableCount(TableDistinct(TableLiteral(
      tt.copy(key = FastSeq("field1")),
      base.rvd,
      base.enc,
      base.encodedGlobals,
    )))
    assertEvalsTo(distinctCount, 2L)
  }

  @Test def testTableKeyByLowering(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.lowering
    val t = TStruct(
      "rows" -> TArray(TStruct("a" -> TInt32, "b" -> TString)),
      "global" -> TStruct("x" -> TString),
    )
    val length = 10
    val value = Row(FastSeq(0 until length: _*).map(i => Row(0, "row" + i)), Row("global"))

    val par = TableParallelize(Literal(t, value))

    val keyed = TableKeyBy(par, IndexedSeq("a"), false)
    assertEvalsTo(TableCount(keyed), length.toLong)
  }

  @Test def testTableParallelize(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.allRelational
    val t = TStruct(
      "rows" -> TArray(TStruct("a" -> TInt32, "b" -> TString)),
      "global" -> TStruct("x" -> TString),
    )
    forAll(Array(1, 10, 17, 34, 103)) { length =>
      val value = Row(FastSeq(0 until length: _*).map(i => Row(i, "row" + i)), Row("global"))
      assertEvalsTo(
        collectNoKey(
          TableParallelize(
            Literal(
              t,
              value,
            )
          )
        ),
        value,
      )
    }
  }

  @Test def testTableParallelizeCount(): scalatest.Assertion = {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.allRelational
    val t = TStruct(
      "rows" -> TArray(TStruct("a" -> TInt32, "b" -> TString)),
      "global" -> TStruct("x" -> TString),
    )
    val value = Row(FastSeq(Row(0, "row1"), Row(1, "row2")), Row("glob"))

    assertEvalsTo(
      TableCount(
        TableParallelize(
          Literal(
            t,
            value,
          )
        )
      ),
      2L,
    )
  }

  @Test def testTableHead(): scalatest.Assertion = {
    val t = TStruct(
      "rows" -> TArray(TStruct("a" -> TInt32, "b" -> TString)),
      "global" -> TStruct("x" -> TString),
    )
    def makeData(length: Int): Row =
      Row(FastSeq(0 until length: _*).map(i => Row(i, "row" + i)), Row("global"))
    val numRowsToTakeArray = Array(0, 4, 7, 12)
    val numInitialPartitionsArray = Array(1, 2, 6, 10, 13)
    val initialDataLength = 10
    val initialData = makeData(initialDataLength)

    forAll(numRowsToTakeArray) { howManyRowsToTake =>
      val headData = makeData(Math.min(howManyRowsToTake, initialDataLength))
      forAll(numInitialPartitionsArray) { howManyInitialPartitions =>
        assertEvalsTo(
          collectNoKey(
            TableHead(
              TableParallelize(
                Literal(t, initialData),
                Some(howManyInitialPartitions),
              ),
              howManyRowsToTake,
            )
          ),
          headData,
        )
      }
    }
  }

  @Test def testTableTail(): scalatest.Assertion = {
    val t = TStruct(
      "rows" -> TArray(TStruct("a" -> TInt32, "b" -> TString)),
      "global" -> TStruct("x" -> TString),
    )
    val numRowsToTakeArray = Array(0, 2, 7, 10, 12)
    val numInitialPartitionsArray = Array(1, 3, 6, 10, 13)
    val initialDataLength = 10
    def makeData(length: Int): Row =
      Row(
        FastSeq((initialDataLength - length) until initialDataLength: _*).map(i =>
          Row(i, "row" + i)
        ),
        Row("global"),
      )
    val initialData = makeData(initialDataLength)

    forAll(numRowsToTakeArray) { howManyRowsToTake =>
      val headData = makeData(Math.min(howManyRowsToTake, initialDataLength))
      forAll(numInitialPartitionsArray) { howManyInitialPartitions =>
        assertEvalsTo(
          collectNoKey(
            TableTail(
              TableParallelize(
                Literal(t, initialData),
                Some(howManyInitialPartitions),
              ),
              howManyRowsToTake,
            )
          ),
          headData,
        )
      }
    }
  }

  @Test def testShuffleAndJoinDoesntMemoryLeak(): scalatest.Assertion = {
    implicit val execStrats = Set(ExecStrategy.LoweredJVMCompile, ExecStrategy.Interpret)
    val row = Ref(TableIR.rowName, TStruct("idx" -> TInt32))
    val t1 = TableRename(TableRange(1, 1), Map("idx" -> "idx_"), Map.empty)
    val t2 =
      TableKeyBy(
        TableMapRows(
          TableRange(50000, 1),
          InsertFields(row, FastSeq("k" -> (I32(49999) - GetField(row, "idx")))),
        ),
        FastSeq("k"),
      )

    assertEvalsTo(TableCount(TableJoin(t1, t2, "left")), 1L)
  }

  @Test def testTableRename(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.lowering
    val t = TStruct(
      "rows" -> TArray(TStruct("a" -> TInt32, "b" -> TString)),
      "global" -> TStruct(("x", TString), ("y", TInt32)),
    )
    val value = Row(FastSeq(0 until 10: _*).map(i => Row(i, "row" + i)), Row("globalVal", 3))
    val adjustedValue =
      Row(FastSeq(0 until 10: _*).map(i => Row(i + 3, "row" + i)), Row("globalVal", 3))

    val renameIR =
      TableRename(
        TableParallelize(
          Literal(
            t,
            value,
          )
        ),
        Map[String, String]("a" -> "c"),
        Map[String, String]("y" -> "z"),
      )

    val newRow = MakeStruct(FastSeq(
      (
        "foo",
        GetField(Ref(TableIR.rowName, renameIR.typ.rowType), "c") + GetField(
          Ref(TableIR.globalName, TStruct(("x", TString), ("z", TInt32))),
          "z",
        ),
      ),
      ("bar", GetField(Ref(TableIR.rowName, renameIR.typ.rowType), "b")),
    ))
    val mapped = TableMapRows(renameIR, newRow)

    assertEvalsTo(
      collectNoKey(
        mapped
      ),
      adjustedValue,
    )
  }

  @Test def testTableMapGlobals(): scalatest.Assertion = {
    val t = TStruct(
      "rows" -> TArray(TStruct("a" -> TInt32, "b" -> TString)),
      "global" -> TStruct("x" -> TString),
    )
    val innerRowRef = Ref(TableIR.rowName, t.field("rows").typ.asInstanceOf[TArray].elementType)
    val innerGlobalRef = Ref(TableIR.globalName, t.field("global").typ)
    val length = 10
    val value = Row(FastSeq(0 until length: _*).map(i => Row(i, "row" + i)), Row("global"))
    val modifedValue =
      Row(FastSeq(0 until length: _*).map(i => Row(i, "global")), Row("newGlobals"))
    assertEvalsTo(
      collectNoKey(
        TableMapGlobals(
          TableMapRows(
            TableParallelize(
              Literal(
                t,
                value,
              )
            ),
            MakeStruct(FastSeq(
              "a" -> GetField(innerRowRef, "a"),
              "b" -> GetField(innerGlobalRef, "x"),
            )),
          ),
          MakeStruct(FastSeq("x" -> Str("newGlobals"))),
        )
      ),
      modifedValue,
    )
  }

  @Test def testTableWrite(): scalatest.Assertion = {
    val table = TableRange(5, 4)
    val path = ctx.createTmpPath("test-table-write", "ht")
    Interpret[Unit](ctx, TableWrite(table, TableNativeWriter(path)))
    val before = ExecuteRelational(ctx, table).asTableValue(ctx)
    val read = TableIR.read(fs, path, requestedType = Some(table.typ))
    assert(read.isDistinctlyKeyed)
    val after = Interpret(read, ctx, false)
    assert(before.globals.javaValue == after.globals.javaValue)
    assert(before.rdd.collect().toFastSeq == after.rdd.collect().toFastSeq)
  }

  @Test def testWriteKeyDistinctness(): scalatest.Assertion = {
    val rt = TableRange(40, 4)
    val idxRef = GetField(Ref(TableIR.rowName, rt.typ.rowType), "idx")
    val at = TableMapRows(
      rt,
      MakeStruct(FastSeq(
        "idx" -> idxRef,
        "const" -> 5,
        "half" -> idxRef.floorDiv(2),
        "oneRepeat" -> If(idxRef ceq I32(10), I32(9), idxRef),
        "oneMissing" -> If(idxRef ceq I32(4), NA(TInt32), idxRef),
        "twoMissing" -> If((idxRef ceq 10) || (idxRef ceq 2), NA(TInt32), idxRef),
      )),
    )
    val keyedByConst = TableKeyBy(at, IndexedSeq("const"))
    val pathConst = ctx.createTmpPath("test-table-write-distinctness", "ht")
    Interpret[Unit](ctx, TableWrite(keyedByConst, TableNativeWriter(pathConst)))
    val readConst = TableIR.read(fs, pathConst)
    assert(!readConst.isDistinctlyKeyed)

    val keyedByHalf = TableKeyBy(at, IndexedSeq("half"))
    val pathHalf = ctx.createTmpPath("test-table-write-distinctness", "ht")
    Interpret[Unit](ctx, TableWrite(keyedByHalf, TableNativeWriter(pathHalf)))
    val readHalf = TableIR.read(fs, pathHalf)
    assert(!readHalf.isDistinctlyKeyed)

    val keyedByIdxAndHalf = TableKeyBy(at, IndexedSeq("idx", "half"))
    val pathIdxAndHalf = ctx.createTmpPath("test-table-write-distinctness", "ht")
    Interpret[Unit](ctx, TableWrite(keyedByIdxAndHalf, TableNativeWriter(pathIdxAndHalf)))
    val readIdxAndHalf = TableIR.read(fs, pathIdxAndHalf)
    assert(readIdxAndHalf.isDistinctlyKeyed)

    val keyedByOneRepeat = TableKeyBy(at, IndexedSeq("oneRepeat"))
    val pathOneRepeat = ctx.createTmpPath("test-table-write-distinctness", "ht")
    Interpret[Unit](ctx, TableWrite(keyedByOneRepeat, TableNativeWriter(pathOneRepeat)))
    val readOneRepeat = TableIR.read(fs, pathOneRepeat)
    assert(!readOneRepeat.isDistinctlyKeyed)

    val keyedByOneMissing = TableKeyBy(at, IndexedSeq("oneMissing"))
    val pathOneMissing = ctx.createTmpPath("test-table-write-distinctness", "ht")
    Interpret[Unit](ctx, TableWrite(keyedByOneMissing, TableNativeWriter(pathOneMissing)))
    val readOneMissing = TableIR.read(fs, pathOneMissing)
    assert(readOneMissing.isDistinctlyKeyed)

    val keyedByTwoMissing = TableKeyBy(at, IndexedSeq("twoMissing"))
    val pathTwoMissing = ctx.createTmpPath("test-table-write-distinctness", "ht")
    Interpret[Unit](ctx, TableWrite(keyedByTwoMissing, TableNativeWriter(pathTwoMissing)))
    val readTwoMissing = TableIR.read(fs, pathTwoMissing)
    assert(!readTwoMissing.isDistinctlyKeyed)
  }

  @Test def testPartitionCountsWithDropRows(): scalatest.Assertion = {
    val tr = new FakeTableReader {
      override def pathsUsed: Seq[String] = Seq.empty
      override def partitionCounts: Option[IndexedSeq[Long]] = Some(FastSeq(1, 2, 3, 4))
      override def fullType: TableType = TableType(TStruct(), FastSeq(), TStruct.empty)
    }
    val tir = TableRead(tr.fullType, true, tr)
    assert(PartitionCounts(tir).forall(_.sum == 0))
  }

  @Test def testScanInAggInMapRows(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.interpretOnly
    var tr: TableIR = TableRange(10, 3)
    tr = TableKeyBy(tr, FastSeq(), false)
    tr = TableMapRows(
      tr,
      InsertFields(
        Ref(TableIR.rowName, tr.typ.rowType),
        FastSeq((
          "result",
          streamAggIR(
            streamAggScanIR(
              StreamRange(0, GetField(Ref(TableIR.rowName, tr.typ.rowType), "idx"), 1)
            )(elt => ApplyScanOp(Sum())(elt.toL))
          )(ApplyAggOp(Sum())(_)),
        )),
      ),
    )
    assertEvalsTo(
      TableCollect(tr),
      Row(
        IndexedSeq.tabulate(10) { i =>
          val r = (0 until i).map(_.toLong).scanLeft(0L)(_ + _).init.sum
          Row(i, r)
        },
        Row(),
      ),
    )
  }

  @Test def testScanInAggInScanInMapRows(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.interpretOnly
    var tr: TableIR = TableRange(10, 3)
    tr = TableKeyBy(tr, FastSeq(), false)
    tr = TableMapRows(
      tr,
      InsertFields(
        Ref(TableIR.rowName, tr.typ.rowType),
        FastSeq((
          "result",
          ApplyScanOp(Sum())(
            streamAggIR(
              streamAggScanIR(
                StreamRange(0, GetField(Ref(TableIR.rowName, tr.typ.rowType), "idx"), 1)
              )(elt => ApplyScanOp(Sum())(elt.toL))
            )(ApplyAggOp(Sum())(_))
          ),
        )),
      ),
    )
    assertEvalsTo(
      TableCollect(tr),
      Row(
        Array.tabulate(10)(i => (0 until i).map(_.toLong).scanLeft(0L)(_ + _).init.sum).scanLeft(
          0L
        )(_ + _)
          .zipWithIndex
          .map { case (x, idx) => Row(idx, x) }.init.toFastSeq,
        Row(),
      ),
    )
  }

  @Test def testTableAggregateByKey(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.allRelational
    var tir: TableIR = TableRead.native(fs, getTestResource("three_key.ht"))
    tir = TableKeyBy(tir, FastSeq("x", "y"), true)
    tir = TableAggregateByKey(
      tir,
      MakeStruct(FastSeq(
        (
          "sum",
          ApplyAggOp(
            FastSeq(),
            FastSeq(GetField(Ref(TableIR.rowName, tir.typ.rowType), "z").toL),
            AggSignature(Sum(), FastSeq(), FastSeq(TInt64)),
          ),
        ),
        ("n", ApplyAggOp(FastSeq(), FastSeq(), AggSignature(Count(), FastSeq(), FastSeq()))),
      )),
    )
    val ir = GetField(TableCollect(TableKeyBy(tir, FastSeq())), "rows")
    assertEvalsTo(
      ir,
      (0 until 10).flatMap(i =>
        (0 until i).map(j => Row(i, j, (0 until j).sum.toLong, j.toLong))
      ).filter(_.getAs[Long](3) > 0),
    )
  }

  @Test def testTableDistinct(): scalatest.Assertion = {
    val tir: TableIR = TableRead.native(fs, getTestResource("three_key.ht"))
    val keyedByX = TableKeyBy(tir, FastSeq("x"), true)
    val distinctByX = TableDistinct(keyedByX)
    assertEvalsTo(TableCount(distinctByX), 8L)
    assertEvalsTo(collect(distinctByX), Row(FastSeq(2 to 9: _*).map(i => Row(i, 1, 0)), Row()))

    val keyedByXAndY = TableKeyBy(tir, FastSeq("x", "y"), true)
    val distinctByXAndY = TableDistinct(keyedByXAndY)
    assertEvalsTo(TableCount(distinctByXAndY), 36L)

    val distinctByAll = TableDistinct(tir)
    assertEvalsTo(TableCount(distinctByAll), 120L)
  }

  @Test def testRangeOrderByDescending(): scalatest.Assertion = {
    var tir: TableIR = TableRange(10, 3)
    tir = TableOrderBy(tir, FastSeq(SortField("idx", Descending)))
    val x = GetField(TableCollect(tir), "rows")

    assertEvalsTo(x, (0 until 10).reverse.map(i => Row(i)))(ExecStrategy.allRelational)
  }

  @Test def testTableLeftJoinRightDistinctRangeTables(): scalatest.Assertion = {
    forAll(IndexedSeq((1, 1), (3, 2), (10, 5), (5, 10))) { case (nParts1, nParts2) =>
      val rangeTable1 = TableRange(10, nParts1)
      var rangeTable2: TableIR = TableRange(5, nParts2)
      val row = Ref(TableIR.rowName, rangeTable2.typ.rowType)
      rangeTable2 =
        TableMapRows(rangeTable2, InsertFields(row, FastSeq("x" -> GetField(row, "idx"))))
      val joinedRanges = TableLeftJoinRightDistinct(rangeTable1, rangeTable2, "foo")
      assertEvalsTo(TableCount(joinedRanges), 10L)

      val expectedJoinCollectResult = Row(
        (0 until 5).map(i => Row(FastSeq(i, Row(i)): _*)) ++ (5 until 10).map(i =>
          Row(FastSeq(i, null): _*)
        ),
        Row(),
      )
      assertEvalsTo(collect(joinedRanges), expectedJoinCollectResult)
    }
  }

  @Test def testNestedStreamInTable(): scalatest.Assertion = {
    var tir: TableIR = TableRange(1, 1)
    var ir: IR = rangeIR(5)
    ir = StreamGrouped(ir, 2)
    ir = ToArray(mapIR(ir)(ToArray))
    ir = InsertFields(Ref(TableIR.rowName, tir.typ.rowType), FastSeq("foo" -> ir))
    tir = TableMapRows(tir, ir)
    assertEvalsTo(
      collect(tir),
      Row(FastSeq(Row(0, FastSeq(FastSeq(0, 1), FastSeq(2, 3), FastSeq(4)))), Row()),
    )
  }

  val parTable1Length = 7

  val parTable1Type = TStruct(
    "rows" -> TArray(TStruct("a1" -> TString, "b1" -> TInt32, "c1" -> TString)),
    "global" -> TStruct("x" -> TString),
  )

  val value1 = Row(
    FastSeq(0 until parTable1Length: _*).map(i => Row("row" + i, i * i, s"t1_$i")),
    Row("global"),
  )

  val table1 = TableParallelize(Literal(parTable1Type, value1), Some(2))

  val parTable2Length = 9

  val parTable2Type = TStruct(
    "rows" -> TArray(TStruct("a2" -> TString, "b2" -> TInt32, "c2" -> TString)),
    "global" -> TStruct("y" -> TInt32),
  )

  val value2 =
    Row(FastSeq(0 until parTable2Length: _*).map(i => Row("row" + i, -2 * i, s"t2_$i")), Row(15))

  val table2 = TableParallelize(Literal(parTable2Type, value2), Some(3))

  val table1KeyedByA = TableKeyBy(table1, IndexedSeq("a1"))
  val table2KeyedByA = TableKeyBy(table2, IndexedSeq("a2"))
  val joinedParKeyedByA = TableLeftJoinRightDistinct(table1KeyedByA, table2KeyedByA, "joinRoot")

  @Test def testTableLeftJoinRightDistinctParallelizeSameKey(): scalatest.Assertion = {
    assertEvalsTo(TableCount(table1KeyedByA), parTable1Length.toLong)
    assertEvalsTo(TableCount(table2KeyedByA), parTable2Length.toLong)

    assertEvalsTo(TableCount(joinedParKeyedByA), parTable1Length.toLong)
    assertEvalsTo(
      collect(joinedParKeyedByA),
      Row(
        FastSeq(0 until parTable1Length: _*).map(i =>
          Row("row" + i, i * i, s"t1_$i", Row(-2 * i, s"t2_$i"))
        ),
        Row("global"),
      ),
    )
  }

  @Test def testTableLeftJoinRightDistinctParallelizePrefixKey(): scalatest.Assertion = {
    val table1KeyedByAAndB = TableKeyBy(table1, IndexedSeq("a1", "b1"))
    val joinedParKeyedByAAndB =
      TableLeftJoinRightDistinct(table1KeyedByAAndB, table2KeyedByA, "joinRoot")

    assertEvalsTo(TableCount(joinedParKeyedByAAndB), parTable1Length.toLong)
    assertEvalsTo(
      collect(joinedParKeyedByAAndB),
      Row(
        FastSeq(0 until parTable1Length: _*).map(i =>
          Row("row" + i, i * i, s"t1_$i", Row(-2 * i, s"t2_$i"))
        ),
        Row("global"),
      ),
    )
  }

  @Test def testTableIntervalJoin(): scalatest.Assertion = {
    val intervals: IndexedSeq[Interval] =
      for {
        (start, end, includesStart, includesEnd) <- FastSeq(
          (1, 6, true, false),
          (2, 2, false, false),
          (3, 5, true, true),
          (4, 6, true, true),
          (6, 7, false, true),
        )
      } yield Interval(
        IntervalEndpoint(start, if (includesStart) -1 else 1),
        IntervalEndpoint(end, if (includesEnd) 1 else -1),
      )

    val left =
      TableKeyBy(
        TableParallelize(MakeStruct(FastSeq(
          "rows" -> Literal(TArray(TStruct("a" -> TInt32)), (0 until 9).map(Row(_))),
          "global" -> MakeStruct(FastSeq("left" -> Str("globals"))),
        ))),
        FastSeq("a"),
        isSorted = true,
      )

    val right =
      TableKeyBy(
        TableParallelize(MakeStruct(FastSeq(
          "rows" -> Literal(
            TArray(TStruct("interval" -> TInterval(TInt32), "b" -> TInt32)),
            intervals.zipWithIndex.map { case (i, idx) => Row(i, idx) },
          ),
          "global" -> MakeStruct(FastSeq("bye" -> I32(-1))),
        ))),
        FastSeq("interval"),
        isSorted = true,
      )

    val join = TableIntervalJoin(left, right, "rights", product = true)

    assertEvalsTo(
      collect(join),
      Row(
        FastSeq(
          Row(0, FastSeq()),
          Row(1, FastSeq(Row(0))),
          Row(2, FastSeq(Row(0))),
          Row(3, FastSeq(Row(2), Row(0))),
          Row(4, FastSeq(Row(2), Row(0), Row(3))),
          Row(5, FastSeq(Row(2), Row(0), Row(3))),
          Row(6, FastSeq(Row(3))),
          Row(7, FastSeq(Row(4))),
          Row(8, FastSeq()),
        ),
        Row("globals"),
      ),
    )
  }

  @Test def testTableKeyByAndAggregate(): scalatest.Assertion = {
    val tir: TableIR = TableRead.native(fs, getTestResource("three_key.ht"))
    val unkeyed = TableKeyBy(tir, IndexedSeq[String]())
    val rowRef = Ref(TableIR.rowName, unkeyed.typ.rowType)
    val aggSignature = AggSignature(Sum(), FastSeq(), FastSeq(TInt64))
    val aggExpression = MakeStruct(FastSeq("y_sum" -> ApplyAggOp(
      FastSeq(),
      FastSeq(Cast(GetField(rowRef, "y"), TInt64)),
      aggSignature,
    )))
    val keyByXAndAggregateSum = TableKeyByAndAggregate(
      unkeyed,
      aggExpression,
      MakeStruct(FastSeq("x" -> GetField(rowRef, "x"))),
      bufferSize = 50,
    )

    assertEvalsTo(
      collect(keyByXAndAggregateSum),
      Row(
        FastSeq(
          Row(2, 1L),
          Row(3, 5L),
          Row(4, 14L),
          Row(5, 30L),
          Row(6, 55L),
          Row(7, 91L),
          Row(8, 140L),
          Row(9, 204L),
        ),
        Row(),
      ),
    )

    // Keying by a newly computed field.
    val keyByXPlusTwoAndAggregateSum = TableKeyByAndAggregate(
      unkeyed,
      aggExpression,
      MakeStruct(FastSeq("xPlusTwo" -> (GetField(rowRef, "x") + 2))),
      bufferSize = 50,
    )
    assertEvalsTo(
      collect(keyByXPlusTwoAndAggregateSum),
      Row(
        FastSeq(
          Row(4, 1L),
          Row(5, 5L),
          Row(6, 14L),
          Row(7, 30L),
          Row(8, 55L),
          Row(9, 91L),
          Row(10, 140L),
          Row(11, 204L),
        ),
        Row(),
      ),
    )

    // Keying by just Z when original is keyed by x,y,z, naming it x anyway.
    val keyByZAndAggregateSum = TableKeyByAndAggregate(
      tir,
      aggExpression,
      MakeStruct(FastSeq("x" -> GetField(rowRef, "z"))),
      bufferSize = 50,
    )
    assertEvalsTo(
      collect(keyByZAndAggregateSum),
      Row(
        FastSeq(
          Row(0, 120L),
          Row(1, 112L),
          Row(2, 98L),
          Row(3, 80L),
          Row(4, 60L),
          Row(5, 40L),
          Row(6, 22L),
          Row(7, 8L),
        ),
        Row(),
      ),
    )
  }

  @Test def testTableAggregateCollectAndTake(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.allRelational
    var tir: TableIR = TableRange(10, 3)
    tir =
      TableMapRows(
        tir,
        InsertFields(Ref(TableIR.rowName, tir.typ.rowType), FastSeq("aStr" -> Str("foo"))),
      )
    val x = TableAggregate(
      tir,
      MakeTuple.ordered(FastSeq(
        ApplyAggOp(Collect())(Ref(TableIR.rowName, tir.typ.rowType)),
        ApplyAggOp(Take(), I32(5))(GetField(Ref(TableIR.rowName, tir.typ.rowType), "idx")),
      )),
    )

    assertEvalsTo(
      x,
      Row(
        (0 until 10).map(i => Row(i, "foo")),
        0 until 5,
      ),
    )
  }

  @Test def testNDArrayMultiplyAddAggregator(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.allRelational
    var tir: TableIR = TableRange(6, 3)
    val nDArray1 = Literal(
      TNDArray(TFloat64, Nat(2)),
      SafeNDArray(IndexedSeq(2L, 2L), IndexedSeq(1.0, 1.0, 1.0, 1.0)),
    )
    val nDArray2 = Literal(
      TNDArray(TFloat64, Nat(2)),
      SafeNDArray(IndexedSeq(2L, 2L), IndexedSeq(2.0, 2.0, 2.0, 2.0)),
    )
    tir = TableMapRows(
      tir,
      InsertFields(
        Ref(TableIR.rowName, tir.typ.rowType),
        FastSeq("nDArrayA" -> nDArray1, "nDArrayB" -> nDArray2),
      ),
    )
    val x = TableAggregate(
      tir,
      ApplyAggOp(NDArrayMultiplyAdd())(
        GetField(Ref(TableIR.rowName, tir.typ.rowType), "nDArrayA"),
        GetField(Ref(TableIR.rowName, tir.typ.rowType), "nDArrayB"),
      ),
    )
    assertEvalsTo(x, SafeNDArray(Vector(2, 2), IndexedSeq(24.0, 24.0, 24.0, 24.0)))
  }

  @Test def testTableScanCollect(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.allRelational
    var tir: TableIR = TableRange(5, 3)
    tir = TableMapRows(
      tir,
      InsertFields(
        Ref(TableIR.rowName, tir.typ.rowType),
        FastSeq("scans" -> MakeTuple.ordered(FastSeq(
          ApplyScanOp(Count())(),
          ApplyScanOp(Collect())(GetField(Ref(TableIR.rowName, tir.typ.rowType), "idx")),
        ))),
      ),
    )
    val x = TableAggregate(tir, ApplyAggOp(Collect())(Ref(TableIR.rowName, tir.typ.rowType)))

    assertEvalsTo(
      x,
      FastSeq(
        Row(0, Row(0L, FastSeq())),
        Row(1, Row(1L, FastSeq(0))),
        Row(2, Row(2L, FastSeq(0, 1))),
        Row(3, Row(3L, FastSeq(0, 1, 2))),
        Row(4, Row(4L, FastSeq(0, 1, 2, 3))),
      ),
    )
  }

  @Test def testIssue9016(): scalatest.Assertion = {
    val rows =
      mapIR(ToStream(MakeArray(makestruct("a" -> MakeTuple.ordered(FastSeq(I32(0), I32(1))))))) {
        row =>
          If(
            IsNA(row),
            NA(TStruct("a" -> TTuple(FastSeq(TupleField(1, TInt32))))),
            makestruct("a" -> bindIR(GetField(row, "a")) { a =>
              If(
                IsNA(a),
                NA(TTuple(FastSeq(TupleField(1, TInt32)))),
                MakeTuple(FastSeq(1 -> GetTupleElement(a, 1))),
              )
            }),
          )
      }
    val table =
      TableParallelize(makestruct("rows" -> ToArray(rows), "global" -> makestruct()), None)
    assertEvalsTo(TableCollect(table), Row(FastSeq(Row(Row(1))), Row()))
  }

  @Test def testTableNativeZippedReaderWithPrefixKey(): scalatest.Assertion = {
    /* This test is important because it tests that we can handle lowering with a
     * TableNativeZippedReader when elements of the original key get pruned away (so I copy key to
     * only be "locus" instead of "locus", "alleles") */
    val rowsPath = getTestResource("sample.vcf.mt/rows")
    val entriesPath = getTestResource("sample.vcf.mt/entries")

    val mnr = MatrixNativeReader(fs, getTestResource("sample.vcf.mt"))
    val mnrSpec = mnr.getSpec()

    val reader =
      TableNativeZippedReader(rowsPath, entriesPath, None, mnrSpec.rowsSpec, mnrSpec.entriesSpec)
    val tableType = mnr.matrixToTableType(mnr.fullMatrixType).copy(
      globalType = TStruct(),
      key = IndexedSeq("locus"),
    )
    val irToLower = TableAggregate(
      TableRead(tableType, false, reader),
      MakeTuple.ordered(FastSeq(
        ApplyAggOp(Collect())(GetField(Ref(TableIR.rowName, tableType.rowType), "rsid"))
      )),
    )
    val optimized = Optimize(ctx, irToLower)
    val analyses = LoweringAnalyses.apply(optimized, ctx)
    LowerTableIR(optimized, DArrayLowering.All, ctx, analyses)
    scalatest.Succeeded
  }

  @Test def testTableMapPartitions(): scalatest.Assertion = {

    val table =
      TableKeyBy(
        TableMapGlobals(
          TableRange(20, nPartitions = 4),
          MakeStruct(FastSeq("greeting" -> Str("Hello"))),
        ),
        IndexedSeq(),
      )

    val rowType = TStruct("idx" -> TInt32)

    assertEvalsTo(
      collect(
        mapPartitions(table) { (_, part) =>
          mapIR(part)(InsertFields(_, FastSeq("str" -> Str("foo"))))
        }
      ),
      Row(IndexedSeq.tabulate(20)(i => Row(i, "foo")), Row("Hello")),
    )

    assertEvalsTo(
      collect(
        mapPartitions(table)((_, part) => filterIR(part)(GetField(_, "idx") > 0))
      ),
      Row(IndexedSeq.tabulate(20)(i => Row(i)).filter(_.getAs[Int](0) > 0), Row("Hello")),
    )

    assertEvalsTo(
      collect(
        mapPartitions(table) { (_, part) =>
          flatMapIR(part) { _ =>
            mapIR(rangeIR(3))(i => MakeStruct(FastSeq("str" -> Str("Hello"), "i" -> i)))
          }
        }
      ),
      Row((0 until 20).flatMap(i => (0 until 3).map(j => Row("Hello", j))), Row("Hello")),
    )

    assertEvalsTo(
      collect(
        mapPartitions(table) { (_, part) =>
          // replace every row in partition with the first row
          filterIR(
            streamScanIR(part, NA(rowType))((acc, row) => If(IsNA(acc), row, acc))
          )(x => !IsNA(x))
        }
      ),
      Row(
        IndexedSeq.tabulate(20) { i =>
          // 0,1,2,3,4,5,6,7,8,9,... ==>
          // 0,0,0,0,0,5,5,5,5,5,...
          Row((i / 5) * 5)
        },
        Row("Hello"),
      ),
    )

    val e = intercept[HailException](TypeCheck(
      ctx,
      collect(mapPartitions(table)((_, part) => flatMapIR(StreamRange(0, 2, 1))(_ => part))),
    ))
    assert(
      "must iterate over the partition exactly once".r.findFirstIn(e.getCause.getMessage).isDefined
    )
  }

  @Test def testRepartitionCostEstimate(): scalatest.Assertion = {
    val empty = RVDPartitioner.empty(ctx.stateManager, TStruct(Array.empty[Field]))
    val some = RVDPartitioner.unkeyed(ctx.stateManager, _)

    val data = IndexedSeq(
      (empty, empty, Succeeded, Failed("Repartitioning from an empty partitioner should be free")),
      (
        empty,
        some(1),
        Succeeded,
        Failed("Repartitioning from an empty partitioner should be free"),
      ),
      (some(1), empty, Succeeded, Failed("Repartitioning to an empty partitioner should be free")),
      (
        some(5),
        some(1),
        Succeeded,
        Failed("Combining multiple partitions into one should not incur a reload"),
      ),
      (
        some(1),
        some(60),
        Failed("Recomputing the same partition multiple times should be replaced with a reload"),
        Succeeded,
      ),
    )

    forAll(data) { case (a, b, t, f) =>
      (if (LowerTableIR.isRepartitioningCheap(a, b)) t else f).toSucceeded.asInstanceOf[
        scalatest.Assertion
      ]
    }
  }

}
