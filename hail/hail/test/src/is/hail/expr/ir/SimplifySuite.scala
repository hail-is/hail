package is.hail.expr.ir

import is.hail.{ExecStrategy, ParameterizedTest}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.TestUtils._
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.TestUtils._
import is.hail.expr.ir.defs._
import is.hail.expr.ir.implicits.forTesting._
import is.hail.io.fs.FS
import is.hail.types.virtual._
import is.hail.utils.Interval
import is.hail.variant.Locus

import org.apache.spark.sql.Row
import org.junit.jupiter.api.{Disabled, Test}
import org.scalactic.{Equivalence, Prettifier}
import org.scalatest.matchers.{MatchResult, Matcher}
import org.scalatest.matchers.dsl.MatcherFactory1
import org.scalatest.matchers.must.Matchers.not
import org.scalatest.matchers.should.Matchers.{a, convertToAnyShouldWrapper}

class SimplifySuite {

  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.interpretOnly

  def simplifyTo(expected: BaseIR)(implicit ctx: ExecuteContext)
    : MatcherFactory1[BaseIR, Equivalence] =
    new MatcherFactory1[BaseIR, Equivalence] {
      override def matcher[T <: BaseIR: Equivalence]: Matcher[T] =
        Matcher[BaseIR] { input =>
          val simplified =
            Simplify(ctx, input)

          val prettyDiff =
            s"""  before = ${Pretty(ctx, input)}
               |   after = ${Pretty(ctx, simplified)}
               |expected = ${Pretty(ctx, expected)}
               | """.stripMargin

          MatchResult(
            matches =
              simplified.isAlphaEquiv(ctx, expected),
            rawFailureMessage =
              s"The simplified IR was not alpha-equivalent:\n$prettyDiff",
            rawNegatedFailureMessage =
              s"The simplified IR was alpha-equivalent:\n$prettyDiff",
          )
        }
    }

  implicit def irPrettifier(implicit ctx: ExecuteContext): Prettifier = {
    case ir: BaseIR => Pretty(ctx, ir)
    case x => Prettifier.default(x)
  }

  @Test def testTableMultiWayZipJoinGlobalsRewrite(implicit ctx: ExecuteContext): Unit = {
    val tmwzj = TableGetGlobals(TableMultiWayZipJoin(
      ArraySeq(TableRange(10, 10), TableRange(10, 10), TableRange(10, 10)),
      "rowField",
      "globalField",
    ))
    assertEvalsTo(tmwzj, Row(FastSeq(Row(), Row(), Row())))
  }

  @Test def testRepartitionableMapUpdatesForUpstreamOptimizations(implicit ctx: ExecuteContext)
    : Unit = {
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

  @Test def testInsertFieldsRewriteRules(implicit ctx: ExecuteContext): Unit = {
    val ir1 =
      InsertFields(InsertFields(base, FastSeq("1" -> I32(2)), None), FastSeq("1" -> I32(3)), None)

    ir1 should simplifyTo(InsertFields(
      base,
      FastSeq("1" -> I32(3)),
      Some(FastSeq("1", "2")),
    ))

    val ir2 =
      InsertFields(
        InsertFields(base, FastSeq("3" -> I32(2)), Some(FastSeq("3", "1", "2"))),
        FastSeq("3" -> I32(3)),
        None,
      )

    ir2 should simplifyTo(InsertFields(
      base,
      FastSeq("3" -> I32(3)),
      Some(FastSeq("3", "1", "2")),
    ))

    val ir3 =
      InsertFields(
        InsertFields(base, FastSeq("3" -> I32(2)), Some(FastSeq("3", "1", "2"))),
        FastSeq("4" -> I32(3)),
        Some(FastSeq("3", "1", "2", "4")),
      )

    ir3 should simplifyTo(InsertFields(
      base,
      FastSeq("3" -> I32(2), "4" -> I32(3)),
      Some(FastSeq("3", "1", "2", "4")),
    ))

    val ir4 =
      InsertFields(
        InsertFields(base, FastSeq("3" -> I32(0), "4" -> I32(1))),
        FastSeq("3" -> I32(5)),
      )

    ir4 should simplifyTo(InsertFields(
      base,
      FastSeq("4" -> I32(1), "3" -> I32(5)),
      Some(FastSeq("1", "2", "3", "4")),
    ))
  }

  lazy val base2 =
    Literal(TStruct("A" -> TInt32, "B" -> TInt32, "C" -> TInt32, "D" -> TInt32), Row(1, 2, 3, 4))

  @Test def testInsertFieldsWhereFieldBeingInsertedCouldBeSelected(implicit ctx: ExecuteContext)
    : Unit = {
    val ir1 =
      InsertFields(
        SelectFields(base2, IndexedSeq("A", "B", "C")),
        IndexedSeq("B" -> GetField(base2, "B")),
        None,
      )
    val simplify1 = Simplify(ctx, ir1)
    assert(simplify1.typ == ir1.typ)
  }

  @Test def testInsertSelectRewriteRules(implicit ctx: ExecuteContext): Unit = {
    SelectFields(InsertFields(base, FastSeq("3" -> I32(1)), None), FastSeq("1")) should
      simplifyTo(SelectFields(base, FastSeq("1")))

    SelectFields(InsertFields(base, FastSeq("3" -> I32(1)), None), FastSeq("3", "1")) should
      simplifyTo {
        InsertFields(
          SelectFields(base, FastSeq("1")),
          FastSeq("3" -> I32(1)),
          Some(FastSeq("3", "1")),
        )
      }
  }

  @Test def testContainsRewrites(implicit ctx: ExecuteContext): Unit = {
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

  @Test def testTableCountExplodeSetRewrite(implicit ctx: ExecuteContext): Unit = {
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

  def testNestedInsertsSimplify() = {
    val unbound = Name("do-not-touch")
    val r = Ref(Name("unbound-struct"), TStruct("x" -> TInt32))

    Seq(
      (
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
      ),
      (
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
      ),
      (
        bindIR(InsertFields(r, FastSeq("y" -> F64(0)))) { r2 =>
          InsertFields(Ref(unbound, TStruct.empty), FastSeq("z" -> GetField(r2, "y").toI))
        },
        bindIRs(F64(0), r) { case Seq(x0, _) =>
          InsertFields(Ref(unbound, TStruct.empty), FastSeq("z" -> x0.toI))
        },
      ), {
        val u = bindIR(InsertFields(r, FastSeq("y" -> Ref(unbound, TFloat64)))) { r2 =>
          InsertFields(r2, FastSeq(("z", invoke("str", TString, r2))))
        }
        (u, u)
      },
      (
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
      ),
      (
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
      ),
    )
  }

  @ParameterizedTest
  def testNestedInsertsSimplify(input: IR, expected: IR)(implicit ctx: ExecuteContext): Unit =
    input should simplifyTo(expected)

  @Test def testArrayAggNoAggRewrites(implicit ctx: ExecuteContext): Unit = {
    val doesRewrite: Array[StreamAgg] = {
      val x = Ref(freshName(), TInt32)
      Array(
        streamAggIR(ToStream(In(0, TArray(TInt32))))(_ => x),
        streamAggIR(ToStream(In(0, TArray(TInt32)))) { _ =>
          aggBindIR(In(1, TInt32) * In(1, TInt32), true)(_ => x)
        },
      )
    }

    doesRewrite.foreach(a => a should simplifyTo(a.query))

    val doesNotRewrite: Array[StreamAgg] = Array(
      streamAggIR(ToStream(In(0, TArray(TInt32))))(ApplyAggOp(Sum())(_)),
      streamAggIR(ToStream(In(0, TArray(TInt32)))) { _ =>
        aggBindIR(In(1, TInt32) * In(1, TInt32))(_ => Ref(freshName(), TInt32))
      },
    )

    doesNotRewrite.foreach(a => a should simplifyTo(a))
  }

  @Test def testArrayAggScanNoAggRewrites(implicit ctx: ExecuteContext): Unit = {
    val doesRewrite: Array[StreamAggScan] = Array(
      streamAggScanIR(ToStream(In(0, TArray(TInt32))))(_ => Ref(freshName(), TInt32)),
      streamAggScanIR(ToStream(In(0, TArray(TInt32)))) { _ =>
        aggBindIR(In(1, TInt32) * In(1, TInt32))(_ => Ref(freshName(), TInt32))
      },
    )

    doesRewrite.foreach(ir => Simplify(ctx, ir) should not be a[StreamAggScan])

    val doesNotRewrite: Array[StreamAggScan] = Array(
      streamAggScanIR(ToStream(In(0, TArray(TInt32))))(foo => ApplyScanOp(Sum())(foo)),
      streamAggScanIR(ToStream(In(0, TArray(TInt32)))) { _ =>
        aggBindIR(In(1, TInt32) * In(1, TInt32), true)(_ => Ref(freshName(), TInt32))
      },
    )

    doesNotRewrite.foreach(a => Simplify(ctx, a) shouldBe a)
  }

  @Test def testArrayLenCollectToTableCount(implicit ctx: ExecuteContext): Unit = {
    val tr = TableRange(10, 10)
    val a = ArrayLen(GetField(TableCollect(tr), "rows"))
    assert(a.typ == TInt32)
    val s = Simplify(ctx, a).asInstanceOf[IR]
    assertEvalsTo(s, 10)
    assert(s.typ == TInt32)
  }

  @Test def testMatrixColsTableMatrixMapColsWithAggLetDoesNotSimplify(implicit ctx: ExecuteContext)
    : Unit = {
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

    tir should simplifyTo(tir)
  }

  def testFilterParallelize(): ArraySeq[IR] = ArraySeq(
    makestruct(
      "rows" -> In(0, TArray(TStruct("x" -> TInt32))),
      "global" -> In(1, TStruct.empty),
    ),
    In(0, TStruct("rows" -> TArray(TStruct("x" -> TInt32)), "global" -> TStruct.empty)),
  )

  @ParameterizedTest
  def testFilterParallelize(rowsAndGlobals: IR)(implicit ctx: ExecuteContext): Unit = {
    val tp = TableParallelize(rowsAndGlobals, None)
    val tf = TableFilter(tp, GetField(Ref(TableIR.rowName, tp.typ.rowType), "x") < 100)

    val rw = Simplify(ctx, tf)
    TypeCheck(ctx, rw)
    assert(!Exists(rw, _.isInstanceOf[TableFilter]))
  }

  @Test def testStreamLenSimplifications(implicit ctx: ExecuteContext): Unit = {
    val rangeIR = StreamRange(I32(0), I32(10), I32(1))
    val mapOfRange = mapIR(rangeIR)(_ + 5)
    val mapBlockedByLet =
      bindIR(I32(5))(ref => mapIR(rangeIR)(_ + ref))

    assert(Simplify(ctx, StreamLen(rangeIR)) == Simplify(ctx, StreamLen(mapOfRange)))
    assert(Simplify(ctx, StreamLen(mapBlockedByLet)) match {
      case Block(_, body) => body == Simplify(ctx, StreamLen(mapOfRange))
    })
  }

  @Test def testNestedFilterIntervals(implicit ctx: ExecuteContext): Unit = {
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

  @Test def testSimplifyReadFilterIntervals(fs: FS)(implicit ctx: ExecuteContext): Unit = {
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

    TableFilterIntervals(tr, intervals1, true) should simplifyTo(exp1)

    TableFilterIntervals(exp1, intervals2, true) should simplifyTo {
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
      )
    }

    val ztfi1 = TableFilterIntervals(tzr, intervals1, true)

    ztfi1 should simplifyTo {
      TableRead(
        tzr.typ,
        false,
        tzrr.copy(options = Some(NativeReaderOptions(intervals1, tnr.fullType.keyType, true))),
      )
    }

    TableFilterIntervals(ztfi1, intervals2, true) should simplifyTo {
      TableRead(
        tzr.typ,
        false,
        tzrr.copy(options = Some(NativeReaderOptions(intersection, tnr.fullType.keyType, true))),
      )
    }
  }

  @Disabled
  @Test def testFilterIntervalsKeyByToFilter(implicit ctx: ExecuteContext): Unit = {
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

  @Test def testSimplifyArraySlice(implicit ctx: ExecuteContext): Unit = {
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

  def testUnaryBooleanSimplification() =
    Seq(
      (ApplyUnaryPrimOp(Bang, ApplyUnaryPrimOp(Bang, ref(TBoolean))), ref(TBoolean))
    )

  @ParameterizedTest
  def testUnaryBooleanSimplification(input: IR, expected: IR)(implicit ctx: ExecuteContext): Unit =
    input should simplifyTo(expected)

  def testUnaryIntegralSimplification() =
    Array(TInt32, TInt64).toSeq.flatMap { typ =>
      Seq(
        (ApplyUnaryPrimOp(Negate, ApplyUnaryPrimOp(Negate, ref(typ))), ref(typ)),
        (ApplyUnaryPrimOp(BitNot, ApplyUnaryPrimOp(BitNot, ref(typ))), ref(typ)),
        (
          ApplyUnaryPrimOp(Negate, ApplyUnaryPrimOp(BitNot, ref(typ))),
          ApplyUnaryPrimOp(Negate, ApplyUnaryPrimOp(BitNot, ref(typ))),
        ),
        (
          ApplyUnaryPrimOp(BitNot, ApplyUnaryPrimOp(Negate, ref(typ))),
          ApplyUnaryPrimOp(BitNot, ApplyUnaryPrimOp(Negate, ref(typ))),
        ),
      )
    }

  @ParameterizedTest
  def testUnaryIntegralSimplification(input: IR, expected: IR)(implicit ctx: ExecuteContext): Unit =
    input should simplifyTo(expected)

  def testBinaryIntegralSimplification() =
    Array((Literal.coerce(TInt32, _)) -> TInt32, (Literal.coerce(TInt64, _)) -> TInt64).toSeq
      .flatMap {
        case (pure, typ) =>
          Seq(
            // Addition
            (
              ApplyBinaryPrimOp(Add(), ref(typ), ref(typ)),
              ApplyBinaryPrimOp(Multiply(), pure(2), ref(typ)),
            ),
            (ApplyBinaryPrimOp(Add(), pure(0), ref(typ)), ref(typ)),
            (ApplyBinaryPrimOp(Add(), ref(typ), pure(0)), ref(typ)),

            // Subtraction
            (ApplyBinaryPrimOp(Subtract(), ref(typ), ref(typ)), pure(0)),
            (
              ApplyBinaryPrimOp(Subtract(), pure(0), ref(typ)),
              ApplyUnaryPrimOp(Negate, ref(typ)),
            ),
            (ApplyBinaryPrimOp(Subtract(), ref(typ), pure(0)), ref(typ)),

            // Multiplication
            (ApplyBinaryPrimOp(Multiply(), pure(0), ref(typ)), pure(0)),
            (ApplyBinaryPrimOp(Multiply(), ref(typ), pure(0)), pure(0)),
            (ApplyBinaryPrimOp(Multiply(), pure(1), ref(typ)), ref(typ)),
            (ApplyBinaryPrimOp(Multiply(), ref(typ), pure(1)), ref(typ)),
            (
              ApplyBinaryPrimOp(Multiply(), pure(-1), ref(typ)),
              ApplyUnaryPrimOp(Negate, ref(typ)),
            ),
            (
              ApplyBinaryPrimOp(Multiply(), ref(typ), pure(-1)),
              ApplyUnaryPrimOp(Negate, ref(typ)),
            ),

            // Div (truncated to -Inf)
            (ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), ref(typ)), pure(1)),
            (ApplyBinaryPrimOp(RoundToNegInfDivide(), pure(0), ref(typ)), pure(0)),
            (
              ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), pure(0)),
              Die("division by zero", typ),
            ),
            (ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), pure(1)), ref(typ)),
            (
              ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), pure(-1)),
              ApplyUnaryPrimOp(Negate, ref(typ)),
            ),

            // Bitwise And
            (ApplyBinaryPrimOp(BitAnd(), pure(0), ref(typ)), pure(0)),
            (ApplyBinaryPrimOp(BitAnd(), ref(typ), pure(0)), pure(0)),
            (ApplyBinaryPrimOp(BitAnd(), pure(-1), ref(typ)), ref(typ)),
            (ApplyBinaryPrimOp(BitAnd(), ref(typ), pure(-1)), ref(typ)),

            // Bitwise Or
            (ApplyBinaryPrimOp(BitOr(), pure(0), ref(typ)), ref(typ)),
            (ApplyBinaryPrimOp(BitOr(), ref(typ), pure(0)), ref(typ)),
            (ApplyBinaryPrimOp(BitOr(), pure(-1), ref(typ)), pure(-1)),
            (ApplyBinaryPrimOp(BitOr(), ref(typ), pure(-1)), pure(-1)),

            // Bitwise Xor
            (ApplyBinaryPrimOp(BitXOr(), ref(typ), ref(typ)), pure(0)),
            (ApplyBinaryPrimOp(BitXOr(), ref(typ), pure(0)), ref(typ)),
            (ApplyBinaryPrimOp(BitXOr(), pure(0), ref(typ)), ref(typ)),
          ) ++
            // Shifts
            Array(LeftShift(), RightShift(), LogicalRightShift()).toSeq.flatMap { shift =>
              Seq(
                (ApplyBinaryPrimOp(shift, pure(0), ref(TInt32)), pure(0)),
                (ApplyBinaryPrimOp(shift, ref(typ), I32(0)), ref(typ)),
              )
            }
      }

  @ParameterizedTest
  def testBinaryIntegralSimplification(input: IR, expected: IR)(implicit ctx: ExecuteContext)
    : Unit =
    input should simplifyTo(expected)

  def testBinaryFloatingSimplification() =
    Array(
      (Literal.coerce(TFloat32, _)) -> TFloat32,
      (Literal.coerce(TFloat64, _)) -> TFloat64,
    ).toSeq.flatMap { case (pure, typ) =>
      Seq(
        // Addition
        (ApplyBinaryPrimOp(Add(), pure(0), ref(typ)), ref(typ)),
        (ApplyBinaryPrimOp(Add(), ref(typ), pure(0)), ref(typ)),

        // Subtraction
        (ApplyBinaryPrimOp(Subtract(), pure(0), ref(typ)), ApplyUnaryPrimOp(Negate, ref(typ))),
        (ApplyBinaryPrimOp(Subtract(), ref(typ), pure(0)), ref(typ)),

        // Multiplication
        (ApplyBinaryPrimOp(Multiply(), pure(1), ref(typ)), ref(typ)),
        (ApplyBinaryPrimOp(Multiply(), ref(typ), pure(1)), ref(typ)),
        (
          ApplyBinaryPrimOp(Multiply(), pure(-1), ref(typ)),
          ApplyUnaryPrimOp(Negate, ref(typ)),
        ),
        (
          ApplyBinaryPrimOp(Multiply(), ref(typ), pure(-1)),
          ApplyUnaryPrimOp(Negate, ref(typ)),
        ),

        // Div (truncated to -Inf)
        (ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), pure(1)), ref(typ)),
        (
          ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), pure(-1)),
          ApplyUnaryPrimOp(Negate, ref(typ)),
        ),
      )
    }

  @ParameterizedTest
  def testBinaryFloatingSimplification(input: IR, expected: IR)(implicit ctx: ExecuteContext)
    : Unit =
    input should simplifyTo(expected)

  def testBlockMatrixSimplification() = {
    val matrix =
      ValueToBlockMatrix(
        MakeArray((1 to 4).map(i => F64(i.toDouble)), TArray(TFloat64)),
        FastSeq(2L, 2L),
        10,
      )

    Seq(
      (BlockMatrixBroadcast(matrix, 0 to 1, matrix.shape, matrix.blockSize), matrix),
      (bmMap(matrix, true)(x => x), matrix),
      (
        bmMap(matrix, true)(_ => F64(2356)),
        BlockMatrixBroadcast(
          ValueToBlockMatrix(F64(2356), FastSeq(1, 1), matrix.blockSize),
          FastSeq(),
          matrix.shape,
          matrix.blockSize,
        ),
      ),
    )
  }

  @ParameterizedTest
  def testBlockMatrixSimplification(
    input: BlockMatrixIR,
    expected: BlockMatrixIR,
  )(implicit ctx: ExecuteContext
  ): Unit =
    input should simplifyTo(expected)

  def testTestSwitchSimplification() =
    Seq(
      (I32(-1), I32(-1), IndexedSeq.tabulate(5)(I32), I32(-1)),
      (I32(1), I32(-1), IndexedSeq.tabulate(5)(I32), I32(1)),
      (
        ref(TInt32),
        I32(-1),
        IndexedSeq.tabulate(5)(I32),
        Switch(ref(TInt32), I32(-1), IndexedSeq.tabulate(5)(I32)),
      ),
      (I32(256), I32(-1), IndexedSeq.empty[IR], I32(-1)),
      (
        ref(TInt32),
        I32(-1),
        IndexedSeq.empty[IR],
        Switch(ref(TInt32), I32(-1), IndexedSeq.empty[IR]),
      ),
    )

  @ParameterizedTest
  def testTestSwitchSimplification(
    x: IR,
    default: IR,
    cases: IndexedSeq[IR],
    expected: BaseIR,
  )(implicit ctx: ExecuteContext
  ): Unit =
    Switch(x, default, cases) should simplifyTo(expected)

  def testIfSimplification() = {
    val x = Ref(freshName(), TInt32)
    val y = Ref(freshName(), TInt32)
    val c = Ref(freshName(), TBoolean)

    Seq(
      (True(), x, Die("Failure", x.typ), x),
      (False(), Die("Failure", x.typ), x, x),
      (IsNA(x), NA(x.typ), x, x),
      (ApplyUnaryPrimOp(Bang, c), x, y, If(c, y, x)),
      (c, If(c, x, y), y, If(c, x, y)),
      (c, x, If(c, x, y), If(c, x, y)),
      (c, x, x, If(IsNA(c), NA(x.typ), x)),
    )
  }

  @ParameterizedTest
  def testIfSimplification(
    pred: IR,
    cnsq: IR,
    altr: IR,
    expected: BaseIR,
  )(implicit ctx: ExecuteContext
  ): Unit =
    If(pred, cnsq, altr) should simplifyTo(expected)

  def testMakeStruct() = {
    val s = ref(TStruct(
      "a" -> TInt32,
      "b" -> TInt64,
      "c" -> TFloat32,
    ))

    def get(name: String) = GetField(s, name)

    Seq(
      (
        FastSeq("x" -> get("a")),
        CastRename(SelectFields(s, FastSeq("a")), TStruct("x" -> TInt32)),
      ),
      (
        FastSeq("x" -> get("a"), "y" -> get("b")),
        CastRename(SelectFields(s, FastSeq("a", "b")), TStruct("x" -> TInt32, "y" -> TInt64)),
      ),
      (
        FastSeq("a" -> get("a"), "b" -> get("b")),
        SelectFields(s, FastSeq("a", "b")),
      ),
      (
        FastSeq("a" -> get("a"), "b" -> get("b"), "c" -> get("c")),
        s,
      ),
    )
  }

  @ParameterizedTest
  def testMakeStruct(fields: IndexedSeq[(String, IR)], expected: IR)(implicit ctx: ExecuteContext)
    : Unit =
    MakeStruct(fields) should simplifyTo(expected)

  def testCastSimplify() =
    Seq(
      (TInt32, TFloat32, false),
      (TInt32, TInt64, true),
      (TInt64, TInt32, false),
      (TInt32, TFloat64, true),
      (TFloat32, TFloat64, true),
      (TFloat64, TFloat32, false),
    )

  @ParameterizedTest
  def testCastSimplify(t1: Type, t2: Type, simplifies: Boolean)(implicit ctx: ExecuteContext)
    : Unit = {
    val x = ref(t1)
    val ir = Cast(Cast(x, t2), t1)
    ir should simplifyTo(if (simplifies) x else ir)
  }
}
