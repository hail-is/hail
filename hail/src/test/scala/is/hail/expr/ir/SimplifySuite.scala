package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.TestUtils.assertEvalsTo
import is.hail.expr.ir.TestUtils.IRAggCount
import is.hail.types.virtual._
import is.hail.utils.{FastIndexedSeq, FastSeq, Interval}
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class SimplifySuite extends HailSuite {
  implicit val execStrats = ExecStrategy.interpretOnly

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

  lazy val base = Literal(TStruct("1" -> TInt32, "2" -> TInt32), Row(1,2))

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
    val bmir = ValueToBlockMatrix(MakeArray(FastIndexedSeq(F64(1), F64(2), F64(3), F64(4)), TArray(TFloat64)),
      FastIndexedSeq(2, 2), 10)
    val identityBroadcast = BlockMatrixBroadcast(bmir, FastIndexedSeq(0, 1), FastIndexedSeq(2, 2), 10)

    assert(Simplify(identityBroadcast) == bmir)
  }

  @Test def testContainsRewrites() {
    assertEvalsTo(invoke("contains", TBoolean, Literal(TArray(TString), FastIndexedSeq("a")), In(0, TString)),
      FastIndexedSeq("a" -> TString),
      true)

    assertEvalsTo(invoke("contains", TBoolean, ToSet(ToStream(In(0, TArray(TString)))), Str("a")),
      FastIndexedSeq(FastIndexedSeq("a") -> TArray(TString)),
      true)


    assertEvalsTo(invoke("contains", TBoolean, ToArray(ToStream(In(0, TSet(TString)))), Str("a")),
      FastIndexedSeq(Set("a") -> TSet(TString)),
      true)
  }

  @Test def testTableCountExplodeSetRewrite() {
    var ir: TableIR = TableRange(1, 1)
    ir = TableMapRows(ir, InsertFields(Ref("row", ir.typ.rowType), Seq("foo" -> Literal(TSet(TInt32), Set(1)))))
    ir = TableExplode(ir, FastIndexedSeq("foo"))
    assertEvalsTo(TableCount(ir), 1L)
  }

  @Test def testNestedInsertsSimplify() {
    val r = Ref("row", TStruct(("x", TInt32)))
    val r2 = Ref("row2", TStruct(("x", TInt32), ("y", TFloat64)))

    val ir1 = Let("row2", InsertFields(r, FastSeq(("y", F64(0.0)))), InsertFields(r2, FastSeq(("z", GetField(r2, "x").toD))))
    val ir2 = Let("row2", InsertFields(r, FastSeq(("y", F64(0.0)))), InsertFields(r2, FastSeq(("z", GetField(r2, "x").toD + GetField(r2, "y")))))
    val ir3 = Let("row2", InsertFields(r, FastSeq(("y", F64(0.0)))), InsertFields(Ref("something_else", TStruct.empty), FastSeq(("z", GetField(r2, "y").toI))))

    assert(Simplify(ir1) == InsertFields(r, FastSeq(("y", F64(0)), ("z", GetField(r, "x").toD)), Some(FastIndexedSeq("x", "y", "z"))))
    assert(Simplify(ir2) == InsertFields(r, FastSeq(("y", F64(0.0)), ("z", GetField(r, "x").toD + F64(0.0))), Some(FastIndexedSeq("x", "y", "z"))))

    assert(Optimize[IR](ctx, ir3) == InsertFields(Ref("something_else", TStruct.empty), FastSeq(("z", I32(0)))))

    val shouldNotRewrite = Let("row2", InsertFields(r, FastSeq(("y", Ref("other", TFloat64)))), InsertFields(r2, FastSeq(("z", invoke("str", TString, r2)))))

    assert(Simplify(shouldNotRewrite) == shouldNotRewrite)
  }

  @Test def testNestedInsertsSimplifyAcrossLets() {
    val l = Let("a",
      Let("b",
        I32(1) + Ref("OTHER_1", TInt32),
        InsertFields(
          Ref("TOP", TStruct("foo" -> TInt32)),
          FastSeq(
            ("field0", Ref("b", TInt32)),
            ("field1", I32(1) + Ref("b", TInt32))))),
      InsertFields(
        Ref("a", TStruct("foo" -> TInt32, "field0" -> TInt32, "field1" -> TInt32)),
        FastSeq(
          ("field2", I32(1) + GetField(Ref("a", TStruct("foo" -> TInt32, "field0" -> TInt32, "field1" -> TInt32)), "field1"))
        )
      )
    )
    val simplified = new NormalizeNames(_.toString, true).apply(Simplify(l))
    val expected = Let("1",
      I32(1) + Ref("OTHER_1", TInt32),
      Let("2", I32(1) + Ref("1", TInt32),
        InsertFields(Ref("TOP", TStruct("foo" -> TInt32)),
          FastSeq(
            ("field0", Ref("1", TInt32)),
            ("field1", Ref("2", TInt32)),
            ("field2", I32(1) + Ref("2", TInt32))
          ),
          Some(FastSeq("foo", "field0", "field1", "field2")))))

    assert(simplified == expected)
  }

  @Test def testArrayAggNoAggRewrites(): Unit = {
    val doesRewrite: Array[StreamAgg] = Array(
      StreamAgg(In(0, TArray(TInt32)), "foo", Ref("x", TInt32)),
      StreamAgg(In(0, TArray(TInt32)), "foo",
        AggLet("bar", In(1, TInt32) * In(1, TInt32), Ref("x", TInt32), true)))

    doesRewrite.foreach { a =>
      assert(Simplify(a) == a.query)
    }

    val doesNotRewrite: Array[StreamAgg] = Array(
      StreamAgg(In(0, TArray(TInt32)), "foo",
        ApplyAggOp(FastIndexedSeq(), FastIndexedSeq(Ref("foo", TInt32)),
          AggSignature(Sum(), FastSeq(), FastSeq(TInt32)))),
      StreamAgg(In(0, TArray(TInt32)), "foo",
        AggLet("bar", In(1, TInt32) * In(1, TInt32), Ref("x", TInt32), false))
    )

    doesNotRewrite.foreach { a =>
      assert(Simplify(a) == a)
    }
  }

  @Test def testArrayAggScanNoAggRewrites(): Unit = {
    val doesRewrite: Array[StreamAggScan] = Array(
      StreamAggScan(In(0, TArray(TInt32)), "foo", Ref("x", TInt32)),
      StreamAggScan(In(0, TArray(TInt32)), "foo",
        AggLet("bar", In(1, TInt32) * In(1, TInt32), Ref("x", TInt32), false)))

    doesRewrite.foreach { a =>
      assert(Simplify(a) == a.query)
    }

    val doesNotRewrite: Array[StreamAggScan] = Array(
      StreamAggScan(In(0, TArray(TInt32)), "foo",
        ApplyScanOp(FastIndexedSeq(), FastIndexedSeq(Ref("foo", TInt32)),
          AggSignature(Sum(), FastSeq(), FastSeq(TInt64)))),
      StreamAggScan(In(0, TArray(TInt32)), "foo",
        AggLet("bar", In(1, TInt32) * In(1, TInt32), Ref("x", TInt32), true))
    )

    doesNotRewrite.foreach { a =>
      assert(Simplify(a) == a)
    }
  }

  @Test def testArrayLenCollectToTableCount(): Unit = {
    val tr = TableRange(10, 10)
    val a = ArrayLen(GetField(TableCollect(tr), "rows"))
    assert(a.typ == TInt32)
    val s = Simplify(a).asInstanceOf[IR]
    assertEvalsTo(s, 10)
    assert(s.typ == TInt32)
  }

  @Test def testMatrixColsTableMatrixMapColsWithAggLetDoesNotSimplify(): Unit = {
    val reader = MatrixRangeReader(1, 1, None)
    var mir: MatrixIR = MatrixRead(reader.fullMatrixType, false, false, reader)
    val colType = reader.fullMatrixType.colType
    mir = MatrixMapCols(mir, AggLet("foo", I32(1), InsertFields(Ref("sa", colType), FastSeq(("bar", I32(2)))), false), None)
    val tir = MatrixColsTable(mir)

    assert(Simplify(tir) == tir)
  }

  @Test def testFilterParallelize() {
    for (rowsAndGlobals <- Array(
      MakeStruct(FastSeq(
      ("rows", In(0, TArray(TStruct("x" -> TInt32)))),
      ("global", In(1, TStruct.empty)))),
      In(0, TStruct("rows" -> TArray(TStruct("x" -> TInt32)), "global" -> TStruct.empty)))
    ) {
      val tp = TableParallelize(rowsAndGlobals, None)
      val tf = TableFilter(tp, GetField(Ref("row", tp.typ.rowType), "x") < 100)

      val rw = Simplify(tf)
      TypeCheck(rw)
      assert(!Exists(rw, _.isInstanceOf[TableFilter]))
    }
  }

  @Test def testNestedFilterIntervals() {
    var tir: TableIR = TableRange(10, 5)
    def r = Ref("row", tir.typ.rowType)
    tir = TableMapRows(tir, InsertFields(r,  FastSeq("idx2" -> GetField(r, "idx"))))
    tir = TableKeyBy(tir, FastIndexedSeq("idx", "idx2"))
    tir = TableFilterIntervals(tir, FastIndexedSeq(Interval(Row(0), Row(1), true, false)), false)
    tir = TableFilterIntervals(tir, FastIndexedSeq(Interval(Row(8), Row(10), true, false)), false)
    assert(Simplify(tir).asInstanceOf[TableFilterIntervals].intervals == FastIndexedSeq(Interval(Row(0), Row(1), true, false), Interval(Row(8), Row(10), true, false)))
  }

  @Test(enabled = false) def testFilterIntervalsKeyByToFilter() {
    var t: TableIR = TableRange(100, 10)
    t = TableMapRows(t, InsertFields(Ref("row", t.typ.rowType), FastSeq(("x", I32(1) - GetField(Ref("row", t.typ.rowType), "idx")))))
    t = TableKeyBy(t, FastIndexedSeq("x"))
    t = TableFilterIntervals(t, FastIndexedSeq(Interval(Row(-10), Row(10), includesStart = true, includesEnd = false)), keep = true)

    val t2 = Simplify(t)
    assert(t2 match {
      case TableKeyBy(TableFilter(child, _), _, _) => !Exists(child, _.isInstanceOf[TableFilterIntervals])
      case _ => false
    })
  }
}
