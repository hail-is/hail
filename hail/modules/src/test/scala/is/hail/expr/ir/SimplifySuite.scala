package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.expr.ir.TestUtils.IRAggCount
import is.hail.types.virtual._
import is.hail.utils.{FastSeq, Interval}
import is.hail.variant.Locus

import org.apache.spark.sql.Row
import org.testng.annotations.{DataProvider, Test}

class SimplifySuite extends HailSuite {
  implicit val execStrats = ExecStrategy.interpretOnly

  @Test def testTableMultiWayZipJoinGlobalsRewrite(): Unit = {
    hc
    val tmwzj = TableGetGlobals(TableMultiWayZipJoin(
      Array(TableRange(10, 10), TableRange(10, 10), TableRange(10, 10)),
      "rowField",
      "globalField",
    ))
    assertEvalsTo(tmwzj, Row(FastSeq(Row(), Row(), Row())))
  }

  @Test def testRepartitionableMapUpdatesForUpstreamOptimizations(): Unit = {
    hc
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

  @Test def testInsertFieldsRewriteRules(): Unit = {
    val ir1 =
      InsertFields(InsertFields(base, FastSeq("1" -> I32(2)), None), FastSeq("1" -> I32(3)), None)
    assert(Simplify(ctx, ir1) == InsertFields(
      base,
      FastSeq("1" -> I32(3)),
      Some(FastSeq("1", "2")),
    ))

    val ir2 = InsertFields(
      InsertFields(base, FastSeq("3" -> I32(2)), Some(FastSeq("3", "1", "2"))),
      FastSeq("3" -> I32(3)),
      None,
    )
    assert(Simplify(ctx, ir2) == InsertFields(
      base,
      FastSeq("3" -> I32(3)),
      Some(FastSeq("3", "1", "2")),
    ))

    val ir3 = InsertFields(
      InsertFields(base, FastSeq("3" -> I32(2)), Some(FastSeq("3", "1", "2"))),
      FastSeq("4" -> I32(3)),
      Some(FastSeq("3", "1", "2", "4")),
    )
    assert(Simplify(ctx, ir3) == InsertFields(
      base,
      FastSeq("3" -> I32(2), "4" -> I32(3)),
      Some(FastSeq("3", "1", "2", "4")),
    ))

    val ir4 =
      InsertFields(
        InsertFields(base, FastSeq("3" -> I32(0), "4" -> I32(1))),
        FastSeq("3" -> I32(5)),
      )
    assert(Simplify(ctx, ir4) == InsertFields(
      base,
      FastSeq("4" -> I32(1), "3" -> I32(5)),
      Some(FastSeq("1", "2", "3", "4")),
    ))
  }

  lazy val base2 =
    Literal(TStruct("A" -> TInt32, "B" -> TInt32, "C" -> TInt32, "D" -> TInt32), Row(1, 2, 3, 4))

  @Test def testInsertFieldsWhereFieldBeingInsertedCouldBeSelected(): Unit = {
    val ir1 =
      InsertFields(
        SelectFields(base2, IndexedSeq("A", "B", "C")),
        IndexedSeq("B" -> GetField(base2, "B")),
        None,
      )
    val simplify1 = Simplify(ctx, ir1)
    assert(simplify1.typ == ir1.typ)
  }

  @Test def testInsertSelectRewriteRules(): Unit = {
    val ir1 = SelectFields(InsertFields(base, FastSeq("3" -> I32(1)), None), FastSeq("1"))
    assert(Simplify(ctx, ir1) == SelectFields(base, FastSeq("1")))

    val ir2 = SelectFields(InsertFields(base, FastSeq("3" -> I32(1)), None), FastSeq("3", "1"))
    assert(Simplify(ctx, ir2) == InsertFields(
      SelectFields(base, FastSeq("1")),
      FastSeq("3" -> I32(1)),
      Some(FastSeq("3", "1")),
    ))
  }

  @Test def testContainsRewrites(): Unit = {
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

  @Test def testTableCountExplodeSetRewrite(): Unit = {
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

  @DataProvider(name = "NestedInserts")
  def nestedInserts: Array[Array[Any]] = {
    val r = Ref(freshName(), TStruct("x" -> TInt32))

    Array[Array[Any]](
      Array(
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
      Array(
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
      Array(
        bindIR(InsertFields(r, FastSeq("y" -> F64(0)))) { r2 =>
          InsertFields(Ref(freshName(), TStruct.empty), FastSeq("z" -> GetField(r2, "y").toI))
        },
        bindIRs(F64(0), r) { case Seq(x0, _) =>
          InsertFields(Ref(freshName(), TStruct.empty), FastSeq("z" -> x0.toI))
        },
      ),
      Array.fill(2) { // unrewriteable
        bindIR(InsertFields(r, FastSeq("y" -> Ref(freshName(), TFloat64)))) { r2 =>
          InsertFields(r2, FastSeq(("z", invoke("str", TString, r2))))
        }
      },
      Array(
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
      Array(
        bindIR(
          bindIR(
            I32(1) + Ref(freshName(), TInt32)
          ) { b =>
            InsertFields(
              Ref(freshName(), TStruct("foo" -> TInt32)),
              FastSeq("field0" -> b, "field1" -> (I32(1) + b)),
            )
          }
        )(a => InsertFields(a, FastSeq("field2" -> (I32(1) + GetField(a, "field1"))))),
        IRBuilder.scoped { x =>
          val b = x.strictMemoize(I32(1) + Ref(freshName(), TInt32))
          val x0 = x.strictMemoize(b)
          val x1 = x.strictMemoize(I32(1) + b)
          val a = Ref(freshName(), TStruct("foo" -> TInt32))
          InsertFields(
            a,
            FastSeq("field0" -> x0, "field1" -> x1, "field2" -> (I32(1) + x1)),
            Some(FastSeq("foo", "field0", "field1", "field2")),
          )
        },
      ),
    )
  }

  @Test(dataProvider = "NestedInserts")
  def testNestedInsertsSimplify(input: IR, expected: IR): Unit = {
    val actual = Simplify(ctx, input)
    actual.isAlphaEquiv(ctx, expected)
  }

  @Test def testArrayAggNoAggRewrites(): Unit = {
    val doesRewrite: Array[StreamAgg] = {
      val x = Ref(freshName(), TInt32)
      Array(
        streamAggIR(ToStream(In(0, TArray(TInt32))))(_ => x),
        streamAggIR(ToStream(In(0, TArray(TInt32)))) { _ =>
          aggBindIR(In(1, TInt32) * In(1, TInt32), true)(_ => x)
        },
      )
    }

    doesRewrite.foreach(a => assert(Simplify(ctx, a) == a.query))

    val doesNotRewrite: Array[StreamAgg] = Array(
      streamAggIR(
        ToStream(In(0, TArray(TInt32)))
      ) { foo =>
        ApplyAggOp(
          FastSeq(),
          FastSeq(foo),
          AggSignature(Sum(), FastSeq(), FastSeq(TInt32)),
        )
      },
      streamAggIR(ToStream(In(0, TArray(TInt32)))) { _ =>
        aggBindIR(In(1, TInt32) * In(1, TInt32))(_ => Ref(freshName(), TInt32))
      },
    )

    doesNotRewrite.foreach(a => assert(Simplify(ctx, a) == a))
  }

  @Test def testArrayAggScanNoAggRewrites(): Unit = {
    val doesRewrite: Array[StreamAggScan] = Array(
      streamAggScanIR(ToStream(In(0, TArray(TInt32))))(_ => Ref(freshName(), TInt32)),
      streamAggScanIR(ToStream(In(0, TArray(TInt32)))) { _ =>
        aggBindIR(In(1, TInt32) * In(1, TInt32))(_ => Ref(freshName(), TInt32))
      },
    )

    doesRewrite.foreach(a => assert(Simplify(ctx, a) == a.query))

    val doesNotRewrite: Array[StreamAggScan] = Array(
      streamAggScanIR(ToStream(In(0, TArray(TInt32))))(foo => ApplyScanOp(Sum())(foo)),
      streamAggScanIR(ToStream(In(0, TArray(TInt32)))) { _ =>
        aggBindIR(In(1, TInt32) * In(1, TInt32), true)(_ => Ref(freshName(), TInt32))
      },
    )

    doesNotRewrite.foreach(a => assert(Simplify(ctx, a) == a))
  }

  @Test def testArrayLenCollectToTableCount(): Unit = {
    val tr = TableRange(10, 10)
    val a = ArrayLen(GetField(TableCollect(tr), "rows"))
    assert(a.typ == TInt32)
    val s = Simplify(ctx, a).asInstanceOf[IR]
    assertEvalsTo(s, 10)
    assert(s.typ == TInt32)
  }

  @Test def testMatrixColsTableMatrixMapColsWithAggLetDoesNotSimplify(): Unit = {
    val reader = MatrixRangeReader(1, 1, None)
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

    assert(Simplify(ctx, tir) == tir)
  }

  @Test def testFilterParallelize(): Unit = {
    for (
      rowsAndGlobals <- Array(
        MakeStruct(FastSeq(
          ("rows", In(0, TArray(TStruct("x" -> TInt32)))),
          ("global", In(1, TStruct.empty)),
        )),
        In(0, TStruct("rows" -> TArray(TStruct("x" -> TInt32)), "global" -> TStruct.empty)),
      )
    ) {
      val tp = TableParallelize(rowsAndGlobals, None)
      val tf = TableFilter(tp, GetField(Ref(TableIR.rowName, tp.typ.rowType), "x") < 100)

      val rw = Simplify(ctx, tf)
      TypeCheck(ctx, rw)
      assert(!Exists(rw, _.isInstanceOf[TableFilter]))
    }
  }

  @Test def testStreamLenSimplifications(): Unit = {
    val rangeIR = StreamRange(I32(0), I32(10), I32(1))
    val mapOfRange = mapIR(rangeIR)(range_element => range_element + 5)
    val mapBlockedByLet =
      bindIR(I32(5))(ref => mapIR(rangeIR)(range_element => range_element + ref))

    assert(Simplify(ctx, StreamLen(rangeIR)) == Simplify(ctx, StreamLen(mapOfRange)))
    assert(Simplify(ctx, StreamLen(mapBlockedByLet)) match {
      case Block(_, body) => body == Simplify(ctx, StreamLen(mapOfRange))
    })
  }

  @Test def testNestedFilterIntervals(): Unit = {
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

  @Test def testSimplifyReadFilterIntervals(): Unit = {
    val src = "src/test/resources/sample-indexed-0.2.52.mt"

    val mnr = MatrixNativeReader(fs, src, None)
    val mr = MatrixRead(mnr.fullMatrixType, false, false, mnr)

    val tnr = TableNativeReader(fs, TableNativeReaderParameters(src + "/rows", None))
    val tr = TableRead(tnr.fullType, false, tnr)

    val tzr = mr.lower().asInstanceOf[TableMapGlobals].child.asInstanceOf[TableRead]
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
    val tfi1 = TableFilterIntervals(tr, intervals1, true)
    val exp1 = TableRead(
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

    assert(Simplify(ctx, tfi1) == exp1)

    val tfi2 = TableFilterIntervals(exp1, intervals2, true)
    val exp2 = TableRead(
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

    assert(Simplify(ctx, tfi2) == exp2)

    val ztfi1 = TableFilterIntervals(tzr, intervals1, true)
    val zexp1 = TableRead(
      tzr.typ,
      false,
      tzrr.copy(options = Some(NativeReaderOptions(intervals1, tnr.fullType.keyType, true))),
    )
    assert(Simplify(ctx, ztfi1) == zexp1)

    val ztfi2 = TableFilterIntervals(ztfi1, intervals2, true)
    val zexp2 = TableRead(
      tzr.typ,
      false,
      tzrr.copy(options = Some(NativeReaderOptions(intersection, tnr.fullType.keyType, true))),
    )

    assert(Simplify(ctx, ztfi2) == zexp2)
  }

  @Test(enabled = false) def testFilterIntervalsKeyByToFilter(): Unit = {
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
      case TableKeyBy(TableFilter(child, _), _, _) =>
        !Exists(child, _.isInstanceOf[TableFilterIntervals])
      case _ => false
    })
  }

  @Test def testSimplifyArraySlice(): Unit = {
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

  def ref(typ: Type) = Ref(Name(""), typ)

  @DataProvider(name = "unaryBooleanArithmetic")
  def unaryBooleanArithmetic: Array[Array[Any]] =
    Array(
      Array(ApplyUnaryPrimOp(Bang, ApplyUnaryPrimOp(Bang, ref(TBoolean))), ref(TBoolean))
    ).asInstanceOf[Array[Array[Any]]]

  @Test(dataProvider = "unaryBooleanArithmetic")
  def testUnaryBooleanSimplification(input: IR, expected: IR): Unit =
    assert(Simplify(ctx, input) == expected)

  @DataProvider(name = "unaryIntegralArithmetic")
  def unaryIntegralArithmetic: Array[Array[Any]] =
    Array(TInt32, TInt64).flatMap { typ =>
      Array(
        Array(ApplyUnaryPrimOp(Negate, ApplyUnaryPrimOp(Negate, ref(typ))), ref(typ)),
        Array(ApplyUnaryPrimOp(BitNot, ApplyUnaryPrimOp(BitNot, ref(typ))), ref(typ)),
        Array(
          ApplyUnaryPrimOp(Negate, ApplyUnaryPrimOp(BitNot, ref(typ))),
          ApplyUnaryPrimOp(Negate, ApplyUnaryPrimOp(BitNot, ref(typ))),
        ),
        Array(
          ApplyUnaryPrimOp(BitNot, ApplyUnaryPrimOp(Negate, ref(typ))),
          ApplyUnaryPrimOp(BitNot, ApplyUnaryPrimOp(Negate, ref(typ))),
        ),
      ).asInstanceOf[Array[Array[Any]]]
    }

  @Test(dataProvider = "unaryIntegralArithmetic")
  def testUnaryIntegralSimplification(input: IR, expected: IR): Unit =
    assert(Simplify(ctx, input) == expected)

  @DataProvider(name = "binaryIntegralArithmetic")
  def binaryIntegralArithmetic: Array[Array[Any]] =
    Array((Literal.coerce(TInt32, _)) -> TInt32, (Literal.coerce(TInt64, _)) -> TInt64).flatMap {
      case (pure, typ) =>
        Array.concat(
          Array(
            // Addition
            Array(
              ApplyBinaryPrimOp(Add(), ref(typ), ref(typ)),
              ApplyBinaryPrimOp(Multiply(), pure(2), ref(typ)),
            ),
            Array(ApplyBinaryPrimOp(Add(), pure(0), ref(typ)), ref(typ)),
            Array(ApplyBinaryPrimOp(Add(), ref(typ), pure(0)), ref(typ)),

            // Subtraction
            Array(ApplyBinaryPrimOp(Subtract(), ref(typ), ref(typ)), pure(0)),
            Array(
              ApplyBinaryPrimOp(Subtract(), pure(0), ref(typ)),
              ApplyUnaryPrimOp(Negate, ref(typ)),
            ),
            Array(ApplyBinaryPrimOp(Subtract(), ref(typ), pure(0)), ref(typ)),

            // Multiplication
            Array(ApplyBinaryPrimOp(Multiply(), pure(0), ref(typ)), pure(0)),
            Array(ApplyBinaryPrimOp(Multiply(), ref(typ), pure(0)), pure(0)),
            Array(ApplyBinaryPrimOp(Multiply(), pure(1), ref(typ)), ref(typ)),
            Array(ApplyBinaryPrimOp(Multiply(), ref(typ), pure(1)), ref(typ)),
            Array(
              ApplyBinaryPrimOp(Multiply(), pure(-1), ref(typ)),
              ApplyUnaryPrimOp(Negate, ref(typ)),
            ),
            Array(
              ApplyBinaryPrimOp(Multiply(), ref(typ), pure(-1)),
              ApplyUnaryPrimOp(Negate, ref(typ)),
            ),

            // Div (truncated to -Inf)
            Array(ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), ref(typ)), pure(1)),
            Array(ApplyBinaryPrimOp(RoundToNegInfDivide(), pure(0), ref(typ)), pure(0)),
            Array(
              ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), pure(0)),
              Die("division by zero", typ),
            ),
            Array(ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), pure(1)), ref(typ)),
            Array(
              ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), pure(-1)),
              ApplyUnaryPrimOp(Negate, ref(typ)),
            ),

            // Bitwise And
            Array(ApplyBinaryPrimOp(BitAnd(), pure(0), ref(typ)), pure(0)),
            Array(ApplyBinaryPrimOp(BitAnd(), ref(typ), pure(0)), pure(0)),
            Array(ApplyBinaryPrimOp(BitAnd(), pure(-1), ref(typ)), ref(typ)),
            Array(ApplyBinaryPrimOp(BitAnd(), ref(typ), pure(-1)), ref(typ)),

            // Bitwise Or
            Array(ApplyBinaryPrimOp(BitOr(), pure(0), ref(typ)), ref(typ)),
            Array(ApplyBinaryPrimOp(BitOr(), ref(typ), pure(0)), ref(typ)),
            Array(ApplyBinaryPrimOp(BitOr(), pure(-1), ref(typ)), pure(-1)),
            Array(ApplyBinaryPrimOp(BitOr(), ref(typ), pure(-1)), pure(-1)),

            // Bitwise Xor
            Array(ApplyBinaryPrimOp(BitXOr(), ref(typ), ref(typ)), pure(0)),
            Array(ApplyBinaryPrimOp(BitXOr(), ref(typ), pure(0)), ref(typ)),
            Array(ApplyBinaryPrimOp(BitXOr(), pure(0), ref(typ)), ref(typ)),
          ).asInstanceOf[Array[Array[Any]]],
          // Shifts
          Array(LeftShift(), RightShift(), LogicalRightShift()).flatMap { shift =>
            Array(
              Array(ApplyBinaryPrimOp(shift, pure(0), ref(TInt32)), pure(0)),
              Array(ApplyBinaryPrimOp(shift, ref(typ), I32(0)), ref(typ)),
            )
          }.asInstanceOf[Array[Array[Any]]],
        )
    }

  @Test(dataProvider = "binaryIntegralArithmetic")
  def testBinaryIntegralSimplification(input: IR, expected: IR): Unit =
    assert(Simplify(ctx, input) == expected)

  @DataProvider(name = "floatingIntegralArithmetic")
  def binaryFloatingArithmetic: Array[Array[Any]] =
    Array(
      (Literal.coerce(TFloat32, _)) -> TFloat32,
      (Literal.coerce(TFloat64, _)) -> TFloat64,
    ).flatMap { case (pure, typ) =>
      Array(
        // Addition
        Array(ApplyBinaryPrimOp(Add(), pure(0), ref(typ)), ref(typ)),
        Array(ApplyBinaryPrimOp(Add(), ref(typ), pure(0)), ref(typ)),

        // Subtraction
        Array(ApplyBinaryPrimOp(Subtract(), pure(0), ref(typ)), ApplyUnaryPrimOp(Negate, ref(typ))),
        Array(ApplyBinaryPrimOp(Subtract(), ref(typ), pure(0)), ref(typ)),

        // Multiplication
        Array(ApplyBinaryPrimOp(Multiply(), pure(1), ref(typ)), ref(typ)),
        Array(ApplyBinaryPrimOp(Multiply(), ref(typ), pure(1)), ref(typ)),
        Array(
          ApplyBinaryPrimOp(Multiply(), pure(-1), ref(typ)),
          ApplyUnaryPrimOp(Negate, ref(typ)),
        ),
        Array(
          ApplyBinaryPrimOp(Multiply(), ref(typ), pure(-1)),
          ApplyUnaryPrimOp(Negate, ref(typ)),
        ),

        // Div (truncated to -Inf)
        Array(ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), pure(1)), ref(typ)),
        Array(
          ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), pure(-1)),
          ApplyUnaryPrimOp(Negate, ref(typ)),
        ),
      ).asInstanceOf[Array[Array[Any]]]
    }

  @Test(dataProvider = "binaryIntegralArithmetic")
  def testBinaryFloatingSimplification(input: IR, expected: IR): Unit =
    assert(Simplify(ctx, input) == expected)

  @DataProvider(name = "blockMatrixRules")
  def blockMatrixRules: Array[Array[Any]] = {
    val matrix =
      ValueToBlockMatrix(
        MakeArray((1 to 4).map(F64(_)), TArray(TFloat64)),
        FastSeq(2, 2),
        10,
      )

    Array(
      Array(BlockMatrixBroadcast(matrix, 0 to 1, matrix.shape, matrix.blockSize), matrix),
      Array(bmMap(matrix, true)(x => x), matrix),
      Array(
        bmMap(matrix, true)(_ => ref(TFloat64)),
        BlockMatrixBroadcast(
          ValueToBlockMatrix(ref(TFloat64), FastSeq(1, 1), matrix.blockSize),
          FastSeq(),
          matrix.shape,
          matrix.blockSize,
        ),
      ),
      Array(
        bmMap(matrix, true)(_ => F64(2356)),
        BlockMatrixBroadcast(
          ValueToBlockMatrix(F64(2356), FastSeq(1, 1), matrix.blockSize),
          FastSeq(),
          matrix.shape,
          matrix.blockSize,
        ),
      ),
    ).asInstanceOf[Array[Array[Any]]]
  }

  @Test(dataProvider = "blockMatrixRules")
  def testBlockMatrixSimplification(input: BlockMatrixIR, expected: BlockMatrixIR): Unit =
    assert(Simplify(ctx, input) == expected)

  @DataProvider(name = "SwitchRules")
  def switchRules: Array[Array[Any]] =
    Array(
      Array(I32(-1), I32(-1), IndexedSeq.tabulate(5)(I32), I32(-1)),
      Array(I32(1), I32(-1), IndexedSeq.tabulate(5)(I32), I32(1)),
      Array(
        ref(TInt32),
        I32(-1),
        IndexedSeq.tabulate(5)(I32),
        Switch(ref(TInt32), I32(-1), IndexedSeq.tabulate(5)(I32)),
      ),
      Array(I32(256), I32(-1), IndexedSeq.empty[IR], I32(-1)),
      Array(
        ref(TInt32),
        I32(-1),
        IndexedSeq.empty[IR],
        Switch(ref(TInt32), I32(-1), IndexedSeq.empty[IR]),
      ), // missingness
    )

  @Test(dataProvider = "SwitchRules")
  def testTestSwitchSimplification(x: IR, default: IR, cases: IndexedSeq[IR], expected: Any): Unit =
    assert(Simplify(ctx, Switch(x, default, cases)) == expected)

  @DataProvider(name = "IfRules")
  def ifRules: Array[Array[Any]] = {
    val x = Ref(freshName(), TInt32)
    val y = Ref(freshName(), TInt32)
    val c = Ref(freshName(), TBoolean)

    Array(
      Array(True(), x, Die("Failure", x.typ), x),
      Array(False(), Die("Failure", x.typ), x, x),
      Array(IsNA(x), NA(x.typ), x, x),
      Array(ApplyUnaryPrimOp(Bang, c), x, y, If(c, y, x)),
      Array(c, If(c, x, y), y, If(c, x, y)),
      Array(c, x, If(c, x, y), If(c, x, y)),
      Array(c, x, x, If(IsNA(c), NA(x.typ), x)),
    )
  }

  @Test(dataProvider = "IfRules")
  def testIfSimplification(pred: IR, cnsq: IR, altr: IR, expected: Any): Unit =
    assert(Simplify(ctx, If(pred, cnsq, altr)) == expected)

  @DataProvider(name = "MakeStructRules")
  def makeStructRules: Array[Array[Any]] = {
    val s = ref(TStruct(
      "a" -> TInt32,
      "b" -> TInt64,
      "c" -> TFloat32,
    ))

    def get(name: String) = GetField(s, name)

    Array(
      Array(
        FastSeq("x" -> get("a")),
        CastRename(SelectFields(s, FastSeq("a")), TStruct("x" -> TInt32)),
      ),
      Array(
        FastSeq("x" -> get("a"), "y" -> get("b")),
        CastRename(SelectFields(s, FastSeq("a", "b")), TStruct("x" -> TInt32, "y" -> TInt64)),
      ),
      Array(
        FastSeq("a" -> get("a"), "b" -> get("b")),
        SelectFields(s, FastSeq("a", "b")),
      ),
      Array(
        FastSeq("a" -> get("a"), "b" -> get("b"), "c" -> get("c")),
        s,
      ),
    )
  }

  @Test(dataProvider = "MakeStructRules")
  def testMakeStruct(fields: IndexedSeq[(String, IR)], expected: IR): Unit = {
    val x = Simplify(ctx, MakeStruct(fields))
    assert(x == expected)
  }

}
