package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.TestUtils._
import is.hail.expr.ir.defs._
import is.hail.types.virtual._
import is.hail.utils.Interval
import is.hail.variant.Locus

import org.apache.spark.sql.Row

class SimplifySuite extends HailSuite {

  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.interpretOnly

  def assertSimplifiesTo(input: BaseIR, expected: BaseIR)(implicit loc: munit.Location): Unit = {
    val simplified = Simplify(ctx, input)
    assert(
      simplified.isAlphaEquiv(ctx, expected),
      s"""The simplified IR was not alpha-equivalent:
         |  before = ${Pretty(ctx, input)}
         |   after = ${Pretty(ctx, simplified)}
         |expected = ${Pretty(ctx, expected)}
         | """.stripMargin,
    )
  }

  test("TableMultiWayZipJoinGlobalsRewrite") {
    val tmwzj = TableGetGlobals(TableMultiWayZipJoin(
      ArraySeq(TableRange(10, 10), TableRange(10, 10), TableRange(10, 10)),
      "rowField",
      "globalField",
    ))
    assertEvalsTo(tmwzj, Row(FastSeq(Row(), Row(), Row())))
  }

  test("RepartitionableMapUpdatesForUpstreamOptimizations") {
    val range = TableKeyBy(TableRange(10, 3), FastSeq())
    val simplifiableIR =
      If(True(), GetField(Ref(TableIR.rowName, range.typ.rowType), "idx").ceq(0), False())
    val checksRepartitioningIR =
      TableFilter(
        TableOrderBy(range, FastSeq(SortField("idx", Ascending))),
        simplifiableIR,
      )

    assertEvalsTo(TableAggregate(checksRepartitioningIR, IRAggCount), 1L)
  }

  lazy val base = Literal(TStruct("1" -> TInt32, "2" -> TInt32), Row(1, 2))

  test("InsertFieldsRewriteRules") {
    val ir1 =
      InsertFields(InsertFields(base, FastSeq("1" -> I32(2)), None), FastSeq("1" -> I32(3)), None)

    assertSimplifiesTo(
      ir1,
      InsertFields(base, FastSeq("1" -> I32(3)), Some(FastSeq("1", "2"))),
    )

    val ir2 =
      InsertFields(
        InsertFields(base, FastSeq("3" -> I32(2)), Some(FastSeq("3", "1", "2"))),
        FastSeq("3" -> I32(3)),
        None,
      )

    assertSimplifiesTo(
      ir2,
      InsertFields(base, FastSeq("3" -> I32(3)), Some(FastSeq("3", "1", "2"))),
    )

    val ir3 =
      InsertFields(
        InsertFields(base, FastSeq("3" -> I32(2)), Some(FastSeq("3", "1", "2"))),
        FastSeq("4" -> I32(3)),
        Some(FastSeq("3", "1", "2", "4")),
      )

    assertSimplifiesTo(
      ir3,
      InsertFields(
        base,
        FastSeq("3" -> I32(2), "4" -> I32(3)),
        Some(FastSeq("3", "1", "2", "4")),
      ),
    )

    val ir4 =
      InsertFields(
        InsertFields(base, FastSeq("3" -> I32(0), "4" -> I32(1))),
        FastSeq("3" -> I32(5)),
      )

    assertSimplifiesTo(
      ir4,
      InsertFields(
        base,
        FastSeq("4" -> I32(1), "3" -> I32(5)),
        Some(FastSeq("1", "2", "3", "4")),
      ),
    )
  }

  lazy val base2 =
    Literal(TStruct("A" -> TInt32, "B" -> TInt32, "C" -> TInt32, "D" -> TInt32), Row(1, 2, 3, 4))

  test("InsertFieldsWhereFieldBeingInsertedCouldBeSelected") {
    val ir1 =
      InsertFields(
        SelectFields(base2, IndexedSeq("A", "B", "C")),
        IndexedSeq("B" -> GetField(base2, "B")),
        None,
      )
    val simplify1 = Simplify(ctx, ir1)
    assert(simplify1.typ == ir1.typ)
  }

  test("InsertSelectRewriteRules") {
    assertSimplifiesTo(
      SelectFields(InsertFields(base, FastSeq("3" -> I32(1)), None), FastSeq("1")),
      SelectFields(base, FastSeq("1")),
    )

    assertSimplifiesTo(
      SelectFields(InsertFields(base, FastSeq("3" -> I32(1)), None), FastSeq("3", "1")),
      InsertFields(
        SelectFields(base, FastSeq("1")),
        FastSeq("3" -> I32(1)),
        Some(FastSeq("3", "1")),
      ),
    )
  }

  test("ContainsRewrites") {
    assertEvalsTo(
      invoke("contains", TBoolean, Literal(TArray(TString), FastSeq("a")), In(0, TString)),
      FastSeq("a" -> TString),
      true,
    )

    assertEvalsTo(
      invoke("contains", TBoolean, ToSet(ToStream(In(0, TArray(TString)))), Str("a")),
      FastSeq(FastSeq("a") -> TArray(TString)),
      true,
    )

    assertEvalsTo(
      invoke("contains", TBoolean, ToArray(ToStream(In(0, TSet(TString)))), Str("a")),
      FastSeq(Set("a") -> TSet(TString)),
      true,
    )
  }

  test("TableCountExplodeSetRewrite") {
    var ir: TableIR = TableRange(1, 1)
    ir = TableMapRows(
      ir,
      InsertFields(
        Ref(TableIR.rowName, ir.typ.rowType),
        FastSeq("foo" -> Literal(TSet(TInt32), Set(1))),
      ),
    )
    ir = TableExplode(ir, FastSeq("foo"))
    assertEvalsTo(TableCount(ir), 1L)
  }

  object checkNestedInsertsSimplify extends TestCases {
    def apply(input: IR, expected: IR)(implicit loc: munit.Location): Unit =
      test("nested inserts simplify")(assertSimplifiesTo(input, expected))
  }

  {
    val unbound = Name("do-not-touch")
    val r = Ref(Name("unbound-struct"), TStruct("x" -> TInt32))

    checkNestedInsertsSimplify(
      bindIR(InsertFields(r, FastSeq("y" -> F64(0)))) { r2 =>
        InsertFields(r2, FastSeq("z" -> GetField(r2, "x").toD))
      },
      bindIRs(F64(0), r) { case Seq(x0, r2) =>
        InsertFields(
          r2,
          FastSeq("y" -> x0, "z" -> GetField(r2, "x").toD),
          Some(FastSeq("x", "y", "z")),
        )
      },
    )

    checkNestedInsertsSimplify(
      bindIR(InsertFields(r, FastSeq("y" -> F64(0)))) { r2 =>
        InsertFields(r2, FastSeq("z" -> (GetField(r2, "x").toD + GetField(r2, "y"))))
      },
      bindIRs(F64(0), r) { case Seq(x0, r2) =>
        InsertFields(
          r2,
          FastSeq("y" -> x0, "z" -> (GetField(r2, "x").toD + x0)),
          Some(FastSeq("x", "y", "z")),
        )
      },
    )

    checkNestedInsertsSimplify(
      bindIR(InsertFields(r, FastSeq("y" -> F64(0)))) { r2 =>
        InsertFields(Ref(unbound, TStruct.empty), FastSeq("z" -> GetField(r2, "y").toI))
      },
      bindIRs(F64(0), r) { case Seq(x0, _) =>
        InsertFields(Ref(unbound, TStruct.empty), FastSeq("z" -> x0.toI))
      },
    )

    {
      val unrewriteable =
        bindIR(InsertFields(r, FastSeq("y" -> Ref(unbound, TFloat64)))) { r2 =>
          InsertFields(r2, FastSeq(("z", invoke("str", TString, r2))))
        }
      checkNestedInsertsSimplify(unrewriteable, unrewriteable)
    }

    checkNestedInsertsSimplify(
      IRBuilder.scoped { b =>
        val a = b.strictMemoize(I32(32))
        val r2 = b.strictMemoize(InsertFields(r, FastSeq("y" -> F64(0))))
        val r3 = b.strictMemoize(InsertFields(r2, FastSeq("w" -> a)))
        InsertFields(r3, FastSeq("z" -> (GetField(r3, "x").toD + GetField(r3, "y"))))
      },
      IRBuilder.scoped { b =>
        val a = b.strictMemoize(I32(32))
        val x0 = b.strictMemoize(F64(0))
        val r2 = b.strictMemoize(r)
        val x1 = b.strictMemoize(x0)
        val x2 = b.strictMemoize(a)
        val r3 = b.strictMemoize(r2)
        InsertFields(
          r3,
          FastSeq(
            "y" -> x1,
            "w" -> x2,
            "z" -> (GetField(r3, "x").toD + x1),
          ),
          Some(FastSeq("x", "y", "w", "z")),
        )
      },
    )

    checkNestedInsertsSimplify(
      IRBuilder.scoped { outer =>
        val ins =
          outer.strictMemoize {
            IRBuilder.scoped { in =>
              val a = in.strictMemoize(I32(1) + Ref(unbound, TInt32))
              InsertFields(r, FastSeq("field0" -> a, "field1" -> (I32(1) + a)))
            }
          }

        InsertFields(ins, FastSeq("field2" -> (I32(1) + GetField(ins, "field1"))))
      },
      IRBuilder.scoped { ib =>
        val a = ib.strictMemoize(I32(1) + Ref(unbound, TInt32))
        val x0 = ib.strictMemoize(a)
        val x1 = ib.strictMemoize(I32(1) + a)
        val s = ib.strictMemoize(r)
        InsertFields(
          s,
          FastSeq("field0" -> x0, "field1" -> x1, "field2" -> (I32(1) + x1)),
          Some(FastSeq("x", "field0", "field1", "field2")),
        )
      },
    )
  }

  test("ArrayAggNoAggRewrites") {
    val doesRewrite: Array[StreamAgg] = {
      val x = Ref(freshName(), TInt32)
      Array(
        streamAggIR(ToStream(In(0, TArray(TInt32))))(_ => x),
        streamAggIR(ToStream(In(0, TArray(TInt32)))) { _ =>
          aggBindIR(In(1, TInt32) * In(1, TInt32), true)(_ => x)
        },
      )
    }

    doesRewrite.foreach(a => assertSimplifiesTo(a, a.query))

    val doesNotRewrite: Array[StreamAgg] = Array(
      streamAggIR(ToStream(In(0, TArray(TInt32))))(ApplyAggOp(Sum())(_)),
      streamAggIR(ToStream(In(0, TArray(TInt32)))) { _ =>
        aggBindIR(In(1, TInt32) * In(1, TInt32))(_ => Ref(freshName(), TInt32))
      },
    )

    doesNotRewrite.foreach(a => assertSimplifiesTo(a, a))
  }

  test("ArrayAggScanNoAggRewrites") {
    val doesRewrite: Array[StreamAggScan] = Array(
      streamAggScanIR(ToStream(In(0, TArray(TInt32))))(_ => Ref(freshName(), TInt32)),
      streamAggScanIR(ToStream(In(0, TArray(TInt32)))) { _ =>
        aggBindIR(In(1, TInt32) * In(1, TInt32))(_ => Ref(freshName(), TInt32))
      },
    )

    doesRewrite.foreach(ir => assert(!Simplify(ctx, ir).isInstanceOf[StreamAggScan]))

    val doesNotRewrite: Array[StreamAggScan] = Array(
      streamAggScanIR(ToStream(In(0, TArray(TInt32))))(foo => ApplyScanOp(Sum())(foo)),
      streamAggScanIR(ToStream(In(0, TArray(TInt32)))) { _ =>
        aggBindIR(In(1, TInt32) * In(1, TInt32), true)(_ => Ref(freshName(), TInt32))
      },
    )

    doesNotRewrite.foreach(a => assertSimplifiesTo(a, a))
  }

  test("ArrayLenCollectToTableCount") {
    val tr = TableRange(10, 10)
    val a = ArrayLen(GetField(TableCollect(tr), "rows"))
    assert(a.typ == TInt32)
    val s = Simplify(ctx, a).asInstanceOf[IR]
    assertEvalsTo(s, 10)
    assert(s.typ == TInt32)
  }

  test("MatrixColsTableMatrixMapColsWithAggLetDoesNotSimplify") {
    val reader = MatrixRangeReader(ctx, 1, 1, None)
    var mir: MatrixIR = MatrixRead(reader.fullMatrixType, false, false, reader)
    val colType = reader.fullMatrixType.colType
    mir = MatrixMapCols(
      mir,
      aggBindIR(I32(1)) { _ =>
        InsertFields(Ref(MatrixIR.colName, colType), FastSeq(("bar", I32(2))))
      },
      None,
    )
    val tir = MatrixColsTable(mir)

    assertSimplifiesTo(tir, tir)
  }

  test("FilterParallelize") {
    Array(
      MakeStruct(FastSeq(
        ("rows", In(0, TArray(TStruct("x" -> TInt32)))),
        ("global", In(1, TStruct.empty)),
      )),
      In(0, TStruct("rows" -> TArray(TStruct("x" -> TInt32)), "global" -> TStruct.empty)),
    ).foreach { rowsAndGlobals =>
      val tp = TableParallelize(rowsAndGlobals, None)
      val tf = TableFilter(tp, GetField(Ref(TableIR.rowName, tp.typ.rowType), "x") < 100)

      val rw = Simplify(ctx, tf)
      TypeCheck(ctx, rw)
      assert(!Exists(rw, _.isInstanceOf[TableFilter]))
    }
  }

  test("StreamLenSimplifications") {
    val rangeIR = StreamRange(I32(0), I32(10), I32(1))
    val mapOfRange = mapIR(rangeIR)(range_element => range_element + 5)
    val mapBlockedByLet =
      bindIR(I32(5))(ref => mapIR(rangeIR)(range_element => range_element + ref))

    assert(Simplify(ctx, StreamLen(rangeIR)) == Simplify(ctx, StreamLen(mapOfRange)))
    assert(Simplify(ctx, StreamLen(mapBlockedByLet)) match {
      case Block(_, body) => body == Simplify(ctx, StreamLen(mapOfRange))
    })
  }

  test("NestedFilterIntervals") {
    var tir: TableIR = TableRange(10, 5)
    def r = Ref(TableIR.rowName, tir.typ.rowType)
    tir = TableMapRows(tir, InsertFields(r, FastSeq("idx2" -> GetField(r, "idx"))))
    tir = TableKeyBy(tir, FastSeq("idx", "idx2"))
    tir = TableFilterIntervals(tir, FastSeq(Interval(Row(0), Row(1), true, false)), false)
    tir = TableFilterIntervals(tir, FastSeq(Interval(Row(8), Row(10), true, false)), false)
    assert(Simplify(ctx, tir).asInstanceOf[TableFilterIntervals].intervals == FastSeq(
      Interval(Row(0), Row(1), true, false),
      Interval(Row(8), Row(10), true, false),
    ))
  }

  test("SimplifyReadFilterIntervals") {
    val src = getTestResource("sample-indexed-0.2.52.mt")

    val mnr = MatrixNativeReader(fs, src, None)
    val mr = MatrixRead(mnr.fullMatrixType, false, false, mnr)

    val tnr = TableNativeReader(fs, TableNativeReaderParameters(src + "/rows", None))
    val tr = TableRead(tnr.fullType, false, tnr)

    val tzr = mr.lower(ctx).asInstanceOf[TableMapGlobals].child.asInstanceOf[TableRead]
    val tzrr = tzr.tr.asInstanceOf[TableNativeZippedReader]

    val intervals1 = FastSeq(
      Interval(Row(Locus("1", 100000)), Row(Locus("1", 200000)), true, false),
      Interval(Row(Locus("2", 100000)), Row(Locus("2", 200000)), true, false),
    )
    val intervals2 = FastSeq(
      Interval(Row(Locus("1", 150000)), Row(Locus("1", 250000)), true, false),
      Interval(Row(Locus("2", 150000)), Row(Locus("2", 250000)), true, false),
    )
    val intersection = FastSeq(
      Interval(Row(Locus("1", 150000)), Row(Locus("1", 200000)), true, false),
      Interval(Row(Locus("2", 150000)), Row(Locus("2", 200000)), true, false),
    )

    val exp1 =
      TableRead(
        tnr.fullType,
        false,
        TableNativeReader(
          fs,
          TableNativeReaderParameters(
            src + "/rows",
            Some(NativeReaderOptions(intervals1, tnr.fullType.keyType, true)),
          ),
        ),
      )

    assertSimplifiesTo(TableFilterIntervals(tr, intervals1, true), exp1)

    assertSimplifiesTo(
      TableFilterIntervals(exp1, intervals2, true),
      TableRead(
        tnr.fullType,
        false,
        TableNativeReader(
          fs,
          TableNativeReaderParameters(
            src + "/rows",
            Some(NativeReaderOptions(intersection, tnr.fullType.keyType, true)),
          ),
        ),
      ),
    )

    val ztfi1 = TableFilterIntervals(tzr, intervals1, true)

    assertSimplifiesTo(
      ztfi1,
      TableRead(
        tzr.typ,
        false,
        tzrr.copy(options = Some(NativeReaderOptions(intervals1, tnr.fullType.keyType, true))),
      ),
    )

    assertSimplifiesTo(
      TableFilterIntervals(ztfi1, intervals2, true),
      TableRead(
        tzr.typ,
        false,
        tzrr.copy(options = Some(NativeReaderOptions(intersection, tnr.fullType.keyType, true))),
      ),
    )
  }

  test("FilterIntervalsKeyByToFilter".ignore) {
    var t: TableIR = TableRange(100, 10)
    t = TableMapRows(
      t,
      InsertFields(
        Ref(TableIR.rowName, t.typ.rowType),
        FastSeq(("x", I32(1) - GetField(Ref(TableIR.rowName, t.typ.rowType), "idx"))),
      ),
    )
    t = TableKeyBy(t, FastSeq("x"))
    t = TableFilterIntervals(
      t,
      FastSeq(Interval(Row(-10), Row(10), includesStart = true, includesEnd = false)),
      keep = true,
    )

    val t2 = Simplify(ctx, t)
    assert(t2 match {
      case TableKeyBy(TableFilter(child, _), _, _, _) =>
        !Exists(child, _.isInstanceOf[TableFilterIntervals])
      case _ => false
    })
  }

  test("SimplifyArraySlice") {
    val stream = StreamRange(I32(0), I32(10), I32(1))
    val streamSlice1 = Simplify(ctx, ArraySlice(ToArray(stream), I32(0), Some(I32(7))))
    assert(streamSlice1 match {
      case ToArray(StreamTake(_, _)) => true
      case _ => false
    })
    assertEvalsTo(streamSlice1.asInstanceOf[IR], FastSeq(0, 1, 2, 3, 4, 5, 6))

    val streamSlice2 = Simplify(ctx, ArraySlice(ToArray(stream), I32(3), Some(I32(5))))
    assert(streamSlice2 match {
      case ToArray(StreamTake(StreamDrop(_, _), _)) => true
      case _ => false
    })
    assertEvalsTo(streamSlice2.asInstanceOf[IR], FastSeq(3, 4))

    val streamSlice3 = Simplify(ctx, ArraySlice(ToArray(stream), I32(6), Some(I32(2))))
    assert(streamSlice3 match {
      case MakeArray(_, _) => true
      case _ => false
    })
    assertEvalsTo(streamSlice3.asInstanceOf[IR], FastSeq())

    val streamSlice4 = Simplify(ctx, ArraySlice(ToArray(stream), I32(0), None))
    assert(streamSlice4 match {
      case ToArray(StreamDrop(_, _)) => true
      case _ => false
    })
    assertEvalsTo(streamSlice4.asInstanceOf[IR], FastSeq(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
  }

  def ref(typ: Type) = Ref(Name("#undefined"), typ)

  object checkUnaryBooleanSimplification extends TestCases {
    def apply(input: IR, expected: IR)(implicit loc: munit.Location): Unit =
      test("unary boolean simplification")(assertSimplifiesTo(input, expected))
  }

  checkUnaryBooleanSimplification(
    ApplyUnaryPrimOp(Bang, ApplyUnaryPrimOp(Bang, ref(TBoolean))),
    ref(TBoolean),
  )

  object checkUnaryIntegralSimplification extends TestCases {
    def apply(input: IR, expected: IR)(implicit loc: munit.Location): Unit =
      test("unary integral simplification")(assertSimplifiesTo(input, expected))
  }

  Array(TInt32, TInt64).foreach { typ =>
    checkUnaryIntegralSimplification(
      ApplyUnaryPrimOp(Negate, ApplyUnaryPrimOp(Negate, ref(typ))),
      ref(typ),
    )
    checkUnaryIntegralSimplification(
      ApplyUnaryPrimOp(BitNot, ApplyUnaryPrimOp(BitNot, ref(typ))),
      ref(typ),
    )
    checkUnaryIntegralSimplification(
      ApplyUnaryPrimOp(Negate, ApplyUnaryPrimOp(BitNot, ref(typ))),
      ApplyUnaryPrimOp(Negate, ApplyUnaryPrimOp(BitNot, ref(typ))),
    )
    checkUnaryIntegralSimplification(
      ApplyUnaryPrimOp(BitNot, ApplyUnaryPrimOp(Negate, ref(typ))),
      ApplyUnaryPrimOp(BitNot, ApplyUnaryPrimOp(Negate, ref(typ))),
    )
  }

  object checkBinaryIntegralSimplification extends TestCases {
    def apply(input: IR, expected: IR)(implicit loc: munit.Location): Unit =
      test("binary integral simplification")(assertSimplifiesTo(input, expected))
  }

  Array[(Any => IR, Type)](
    (Literal.coerce(TInt32, _), TInt32),
    (Literal.coerce(TInt64, _), TInt64),
  ).foreach { case (pure, typ) =>
    // Addition
    checkBinaryIntegralSimplification(
      ApplyBinaryPrimOp(Add(), ref(typ), ref(typ)),
      ApplyBinaryPrimOp(Multiply(), pure(2), ref(typ)),
    )
    checkBinaryIntegralSimplification(ApplyBinaryPrimOp(Add(), pure(0), ref(typ)), ref(typ))
    checkBinaryIntegralSimplification(ApplyBinaryPrimOp(Add(), ref(typ), pure(0)), ref(typ))

    // Subtraction
    checkBinaryIntegralSimplification(ApplyBinaryPrimOp(Subtract(), ref(typ), ref(typ)), pure(0))
    checkBinaryIntegralSimplification(
      ApplyBinaryPrimOp(Subtract(), pure(0), ref(typ)),
      ApplyUnaryPrimOp(Negate, ref(typ)),
    )
    checkBinaryIntegralSimplification(ApplyBinaryPrimOp(Subtract(), ref(typ), pure(0)), ref(typ))

    // Multiplication
    checkBinaryIntegralSimplification(ApplyBinaryPrimOp(Multiply(), pure(0), ref(typ)), pure(0))
    checkBinaryIntegralSimplification(ApplyBinaryPrimOp(Multiply(), ref(typ), pure(0)), pure(0))
    checkBinaryIntegralSimplification(ApplyBinaryPrimOp(Multiply(), pure(1), ref(typ)), ref(typ))
    checkBinaryIntegralSimplification(ApplyBinaryPrimOp(Multiply(), ref(typ), pure(1)), ref(typ))
    checkBinaryIntegralSimplification(
      ApplyBinaryPrimOp(Multiply(), pure(-1), ref(typ)),
      ApplyUnaryPrimOp(Negate, ref(typ)),
    )
    checkBinaryIntegralSimplification(
      ApplyBinaryPrimOp(Multiply(), ref(typ), pure(-1)),
      ApplyUnaryPrimOp(Negate, ref(typ)),
    )

    // Div (truncated to -Inf)
    checkBinaryIntegralSimplification(
      ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), ref(typ)),
      pure(1),
    )
    checkBinaryIntegralSimplification(
      ApplyBinaryPrimOp(RoundToNegInfDivide(), pure(0), ref(typ)),
      pure(0),
    )
    checkBinaryIntegralSimplification(
      ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), pure(0)),
      Die("division by zero", typ),
    )
    checkBinaryIntegralSimplification(
      ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), pure(1)),
      ref(typ),
    )
    checkBinaryIntegralSimplification(
      ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), pure(-1)),
      ApplyUnaryPrimOp(Negate, ref(typ)),
    )

    // Bitwise And
    checkBinaryIntegralSimplification(ApplyBinaryPrimOp(BitAnd(), pure(0), ref(typ)), pure(0))
    checkBinaryIntegralSimplification(ApplyBinaryPrimOp(BitAnd(), ref(typ), pure(0)), pure(0))
    checkBinaryIntegralSimplification(ApplyBinaryPrimOp(BitAnd(), pure(-1), ref(typ)), ref(typ))
    checkBinaryIntegralSimplification(ApplyBinaryPrimOp(BitAnd(), ref(typ), pure(-1)), ref(typ))

    // Bitwise Or
    checkBinaryIntegralSimplification(ApplyBinaryPrimOp(BitOr(), pure(0), ref(typ)), ref(typ))
    checkBinaryIntegralSimplification(ApplyBinaryPrimOp(BitOr(), ref(typ), pure(0)), ref(typ))
    checkBinaryIntegralSimplification(ApplyBinaryPrimOp(BitOr(), pure(-1), ref(typ)), pure(-1))
    checkBinaryIntegralSimplification(ApplyBinaryPrimOp(BitOr(), ref(typ), pure(-1)), pure(-1))

    // Bitwise Xor
    checkBinaryIntegralSimplification(ApplyBinaryPrimOp(BitXOr(), ref(typ), ref(typ)), pure(0))
    checkBinaryIntegralSimplification(ApplyBinaryPrimOp(BitXOr(), ref(typ), pure(0)), ref(typ))
    checkBinaryIntegralSimplification(ApplyBinaryPrimOp(BitXOr(), pure(0), ref(typ)), ref(typ))

    // Shifts
    Array(LeftShift(), RightShift(), LogicalRightShift()).foreach { shift =>
      checkBinaryIntegralSimplification(
        ApplyBinaryPrimOp(shift, pure(0), ref(TInt32)),
        pure(0),
      )
      checkBinaryIntegralSimplification(
        ApplyBinaryPrimOp(shift, ref(typ), I32(0)),
        ref(typ),
      )
    }
  }

  object checkBinaryFloatingSimplification extends TestCases {
    def apply(input: IR, expected: IR)(implicit loc: munit.Location): Unit =
      test("binary floating simplification")(assertSimplifiesTo(input, expected))
  }

  Array[(Any => IR, Type)](
    (Literal.coerce(TFloat32, _), TFloat32),
    (Literal.coerce(TFloat64, _), TFloat64),
  ).foreach { case (pure, typ) =>
    // Addition
    checkBinaryFloatingSimplification(ApplyBinaryPrimOp(Add(), pure(0), ref(typ)), ref(typ))
    checkBinaryFloatingSimplification(ApplyBinaryPrimOp(Add(), ref(typ), pure(0)), ref(typ))

    // Subtraction
    checkBinaryFloatingSimplification(
      ApplyBinaryPrimOp(Subtract(), pure(0), ref(typ)),
      ApplyUnaryPrimOp(Negate, ref(typ)),
    )
    checkBinaryFloatingSimplification(ApplyBinaryPrimOp(Subtract(), ref(typ), pure(0)), ref(typ))

    // Multiplication
    checkBinaryFloatingSimplification(ApplyBinaryPrimOp(Multiply(), pure(1), ref(typ)), ref(typ))
    checkBinaryFloatingSimplification(ApplyBinaryPrimOp(Multiply(), ref(typ), pure(1)), ref(typ))
    checkBinaryFloatingSimplification(
      ApplyBinaryPrimOp(Multiply(), pure(-1), ref(typ)),
      ApplyUnaryPrimOp(Negate, ref(typ)),
    )
    checkBinaryFloatingSimplification(
      ApplyBinaryPrimOp(Multiply(), ref(typ), pure(-1)),
      ApplyUnaryPrimOp(Negate, ref(typ)),
    )

    // Div (truncated to -Inf)
    checkBinaryFloatingSimplification(
      ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), pure(1)),
      ref(typ),
    )
    checkBinaryFloatingSimplification(
      ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), pure(-1)),
      ApplyUnaryPrimOp(Negate, ref(typ)),
    )
  }

  object checkBlockMatrixSimplification extends TestCases {
    def apply(input: BlockMatrixIR, expected: BlockMatrixIR)(implicit loc: munit.Location): Unit =
      test("block matrix simplification")(assertSimplifiesTo(input, expected))
  }

  {
    val matrix =
      ValueToBlockMatrix(
        MakeArray((1 to 4).map(i => F64(i.toDouble)), TArray(TFloat64)),
        FastSeq(2L, 2L),
        10,
      )

    checkBlockMatrixSimplification(
      BlockMatrixBroadcast(matrix, 0 to 1, matrix.shape, matrix.blockSize),
      matrix,
    )
    checkBlockMatrixSimplification(bmMap(matrix, true)(x => x), matrix)
    checkBlockMatrixSimplification(
      bmMap(matrix, true)(_ => F64(2356)),
      BlockMatrixBroadcast(
        ValueToBlockMatrix(F64(2356), FastSeq(1, 1), matrix.blockSize),
        FastSeq(),
        matrix.shape,
        matrix.blockSize,
      ),
    )
  }

  object checkSwitchSimplification extends TestCases {
    def apply(
      x: IR,
      default: IR,
      cases: IndexedSeq[IR],
      expected: BaseIR,
    )(implicit loc: munit.Location
    ): Unit =
      test("switch simplification")(assertSimplifiesTo(Switch(x, default, cases), expected))
  }

  checkSwitchSimplification(I32(-1), I32(-1), IndexedSeq.tabulate(5)(I32), I32(-1))
  checkSwitchSimplification(I32(1), I32(-1), IndexedSeq.tabulate(5)(I32), I32(1))

  checkSwitchSimplification(
    ref(TInt32),
    I32(-1),
    IndexedSeq.tabulate(5)(I32),
    Switch(ref(TInt32), I32(-1), IndexedSeq.tabulate(5)(I32)),
  )

  checkSwitchSimplification(I32(256), I32(-1), IndexedSeq.empty[IR], I32(-1))

  checkSwitchSimplification(
    ref(TInt32),
    I32(-1),
    IndexedSeq.empty[IR],
    Switch(ref(TInt32), I32(-1), IndexedSeq.empty[IR]),
  )

  object checkIfSimplification extends TestCases {
    def apply(
      pred: IR,
      cnsq: IR,
      altr: IR,
      expected: BaseIR,
    )(implicit loc: munit.Location
    ): Unit =
      test("if simplification")(assertSimplifiesTo(If(pred, cnsq, altr), expected))
  }

  {
    val x = Ref(freshName(), TInt32)
    val y = Ref(freshName(), TInt32)
    val c = Ref(freshName(), TBoolean)

    checkIfSimplification(True(), x, Die("Failure", x.typ), x)
    checkIfSimplification(False(), Die("Failure", x.typ), x, x)
    checkIfSimplification(IsNA(x), NA(x.typ), x, x)
    checkIfSimplification(ApplyUnaryPrimOp(Bang, c), x, y, If(c, y, x))
    checkIfSimplification(c, If(c, x, y), y, If(c, x, y))
    checkIfSimplification(c, x, If(c, x, y), If(c, x, y))
    checkIfSimplification(c, x, x, If(IsNA(c), NA(x.typ), x))
  }

  object checkMakeStruct extends TestCases {
    def apply(
      fields: IndexedSeq[(String, IR)],
      expected: IR,
    )(implicit loc: munit.Location
    ): Unit =
      test("make struct")(assertSimplifiesTo(MakeStruct(fields), expected))
  }

  {
    val s = ref(TStruct(
      "a" -> TInt32,
      "b" -> TInt64,
      "c" -> TFloat32,
    ))

    def get(name: String) = GetField(s, name)

    checkMakeStruct(
      FastSeq("x" -> get("a")),
      CastRename(SelectFields(s, FastSeq("a")), TStruct("x" -> TInt32)),
    )
    checkMakeStruct(
      FastSeq("x" -> get("a"), "y" -> get("b")),
      CastRename(SelectFields(s, FastSeq("a", "b")), TStruct("x" -> TInt32, "y" -> TInt64)),
    )
    checkMakeStruct(
      FastSeq("a" -> get("a"), "b" -> get("b")),
      SelectFields(s, FastSeq("a", "b")),
    )
    checkMakeStruct(
      FastSeq("a" -> get("a"), "b" -> get("b"), "c" -> get("c")),
      s,
    )
  }

  object checkCastSimplify extends TestCases {
    def apply(
      t1: Type,
      t2: Type,
      simplifies: Boolean,
    )(implicit loc: munit.Location
    ): Unit =
      test("cast simplify") {
        val x = ref(t1)
        val ir = Cast(Cast(x, t2), t1)
        assertSimplifiesTo(ir, if (simplifies) x else ir)
      }
  }

  checkCastSimplify(TInt32, TFloat32, false)
  checkCastSimplify(TInt32, TInt64, true)
  checkCastSimplify(TInt64, TInt32, false)
  checkCastSimplify(TInt32, TFloat64, true)
  checkCastSimplify(TFloat32, TFloat64, true)
  checkCastSimplify(TFloat64, TFloat32, false)
}
