package is.hail.expr.ir

import is.hail.expr.ir.TestUtils.IRAggCount
import is.hail.types.virtual._
import is.hail.utils.{FastIndexedSeq, FastSeq, Interval}
import is.hail.variant.Locus
import is.hail.{ExecStrategy, HailSuite}
import org.apache.spark.sql.Row
import org.testng.annotations.{DataProvider, Test}

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
    assert(Simplify(ctx, ir1) == InsertFields(base, Seq("1" -> I32(3)), Some(FastIndexedSeq("1", "2"))))

    val ir2 = InsertFields(InsertFields(base, Seq("3" -> I32(2)), Some(FastIndexedSeq("3", "1", "2"))), Seq("3" -> I32(3)), None)
    assert(Simplify(ctx, ir2) == InsertFields(base, Seq("3" -> I32(3)), Some(FastIndexedSeq("3", "1", "2"))))

    val ir3 = InsertFields(InsertFields(base, Seq("3" -> I32(2)), Some(FastIndexedSeq("3", "1", "2"))), Seq("4" -> I32(3)), Some(FastIndexedSeq("3", "1", "2", "4")))
    assert(Simplify(ctx, ir3) == InsertFields(base, Seq("3" -> I32(2), "4" -> I32(3)), Some(FastIndexedSeq("3", "1", "2", "4"))))

    val ir4 = InsertFields(InsertFields(base, Seq("3" -> I32(0), "4" -> I32(1))), Seq("3" -> I32(5)))
    assert(Simplify(ctx, ir4) == InsertFields(base, Seq("4" -> I32(1), "3" -> I32(5)), Some(FastIndexedSeq("1", "2", "3", "4"))))
  }

  lazy val base2 = Literal(TStruct("A" -> TInt32, "B" -> TInt32, "C" -> TInt32, "D" -> TInt32), Row(1, 2, 3, 4))
  @Test def testInsertFieldsWhereFieldBeingInsertedCouldBeSelected(): Unit = {
    val ir1 =
        InsertFields(
          SelectFields(base2, IndexedSeq("A", "B", "C")),
          IndexedSeq("B" -> GetField(base2, "B")),
          None
        )
    val simplify1 = Simplify(ctx, ir1)
    assert(simplify1.typ == ir1.typ)
  }

  @Test def testInsertSelectRewriteRules() {
    val ir1 = SelectFields(InsertFields(base, FastIndexedSeq("3" -> I32(1)), None), FastIndexedSeq("1"))
    assert(Simplify(ctx, ir1) == SelectFields(base, FastIndexedSeq("1")))

    val ir2 = SelectFields(InsertFields(base, FastIndexedSeq("3" -> I32(1)), None), FastIndexedSeq("3", "1"))
    assert(Simplify(ctx, ir2) == InsertFields(SelectFields(base, FastIndexedSeq("1")), FastIndexedSeq("3" -> I32(1)), Some(FastIndexedSeq("3", "1"))))
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

    assert(Simplify(ctx, ir1) == InsertFields(r, FastSeq(("y", F64(0)), ("z", GetField(r, "x").toD)), Some(FastIndexedSeq("x", "y", "z"))))
    assert(Simplify(ctx, ir2) == InsertFields(r, FastSeq(("y", F64(0.0)), ("z", GetField(r, "x").toD)), Some(FastIndexedSeq("x", "y", "z"))))

    assert(Optimize[IR](ir3, "direct", ctx) == InsertFields(Ref("something_else", TStruct.empty), FastSeq(("z", I32(0)))))

    val shouldNotRewrite = Let("row2", InsertFields(r, FastSeq(("y", Ref("other", TFloat64)))), InsertFields(r2, FastSeq(("z", invoke("str", TString, r2)))))

    assert(Simplify(ctx, shouldNotRewrite) == shouldNotRewrite)
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
    val simplified = new NormalizeNames(_.toString, true).apply(Simplify(ctx, l))
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
      assert(Simplify(ctx, a) == a.query)
    }

    val doesNotRewrite: Array[StreamAgg] = Array(
      StreamAgg(In(0, TArray(TInt32)), "foo",
        ApplyAggOp(FastIndexedSeq(), FastIndexedSeq(Ref("foo", TInt32)),
          AggSignature(Sum(), FastSeq(), FastSeq(TInt32)))),
      StreamAgg(In(0, TArray(TInt32)), "foo",
        AggLet("bar", In(1, TInt32) * In(1, TInt32), Ref("x", TInt32), false))
    )

    doesNotRewrite.foreach { a =>
      assert(Simplify(ctx, a) == a)
    }
  }

  @Test def testArrayAggScanNoAggRewrites(): Unit = {
    val doesRewrite: Array[StreamAggScan] = Array(
      StreamAggScan(In(0, TArray(TInt32)), "foo", Ref("x", TInt32)),
      StreamAggScan(In(0, TArray(TInt32)), "foo",
        AggLet("bar", In(1, TInt32) * In(1, TInt32), Ref("x", TInt32), false)))

    doesRewrite.foreach { a =>
      assert(Simplify(ctx, a) == a.query)
    }

    val doesNotRewrite: Array[StreamAggScan] = Array(
      StreamAggScan(In(0, TArray(TInt32)), "foo",
        ApplyScanOp(FastIndexedSeq(), FastIndexedSeq(Ref("foo", TInt32)),
          AggSignature(Sum(), FastSeq(), FastSeq(TInt64)))),
      StreamAggScan(In(0, TArray(TInt32)), "foo",
        AggLet("bar", In(1, TInt32) * In(1, TInt32), Ref("x", TInt32), true))
    )

    doesNotRewrite.foreach { a =>
      assert(Simplify(ctx, a) == a)
    }
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
    mir = MatrixMapCols(mir, AggLet("foo", I32(1), InsertFields(Ref("sa", colType), FastSeq(("bar", I32(2)))), false), None)
    val tir = MatrixColsTable(mir)

    assert(Simplify(ctx, tir) == tir)
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

      val rw = Simplify(ctx, tf)
      TypeCheck(ctx, rw)
      assert(!Exists(rw, _.isInstanceOf[TableFilter]))
    }
  }

  @Test def testStreamLenSimplifications(): Unit = {
    val rangeIR = StreamRange(I32(0), I32(10), I32(1))
    val mapOfRange = mapIR(rangeIR)(range_element => range_element + 5)
    val mapBlockedByLet = bindIR(I32(5))(ref => mapIR(rangeIR)(range_element => range_element + ref))

    assert(Simplify(ctx, StreamLen(rangeIR)) == Simplify(ctx, StreamLen(mapOfRange)))
    assert(Simplify(ctx, StreamLen(mapBlockedByLet)) match {
      case Let(name, value, body) => body == Simplify(ctx, StreamLen(mapOfRange))
    })
  }

  @Test def testNestedFilterIntervals() {
    var tir: TableIR = TableRange(10, 5)
    def r = Ref("row", tir.typ.rowType)
    tir = TableMapRows(tir, InsertFields(r,  FastSeq("idx2" -> GetField(r, "idx"))))
    tir = TableKeyBy(tir, FastIndexedSeq("idx", "idx2"))
    tir = TableFilterIntervals(tir, FastIndexedSeq(Interval(Row(0), Row(1), true, false)), false)
    tir = TableFilterIntervals(tir, FastIndexedSeq(Interval(Row(8), Row(10), true, false)), false)
    assert(Simplify(ctx, tir).asInstanceOf[TableFilterIntervals].intervals == FastIndexedSeq(Interval(Row(0), Row(1), true, false), Interval(Row(8), Row(10), true, false)))
  }

  @Test def testSimplifyReadFilterIntervals() {
    val src = "src/test/resources/sample-indexed-0.2.52.mt"

    val mnr = MatrixNativeReader(fs, src, None)
    val mr = MatrixRead(mnr.fullMatrixType, false, false, mnr)

    val tnr = TableNativeReader(fs, TableNativeReaderParameters(src + "/rows", None))
    val tr = TableRead(tnr.fullType, false, tnr)

    val tzr = mr.lower().asInstanceOf[TableMapGlobals].child.asInstanceOf[TableRead]
    val tzrr = tzr.tr.asInstanceOf[TableNativeZippedReader]

    val intervals1 = FastIndexedSeq(Interval(Row(Locus("1", 100000)), Row(Locus("1", 200000)), true, false), Interval(Row(Locus("2", 100000)), Row(Locus("2", 200000)), true, false))
    val intervals2 = FastIndexedSeq(Interval(Row(Locus("1", 150000)), Row(Locus("1", 250000)), true, false), Interval(Row(Locus("2", 150000)), Row(Locus("2", 250000)), true, false))
    val intersection = FastIndexedSeq(Interval(Row(Locus("1", 150000)), Row(Locus("1", 200000)), true, false), Interval(Row(Locus("2", 150000)), Row(Locus("2", 200000)), true, false))
    val tfi1 = TableFilterIntervals(tr, intervals1, true)
    val exp1 = TableRead(tnr.fullType, false, TableNativeReader(fs, TableNativeReaderParameters(src + "/rows", Some(NativeReaderOptions(intervals1, tnr.fullType.keyType, true)))))

    assert(Simplify(ctx, tfi1) == exp1)

    val tfi2 = TableFilterIntervals(exp1, intervals2, true)
    val exp2 = TableRead(tnr.fullType, false, TableNativeReader(fs, TableNativeReaderParameters(src + "/rows", Some(NativeReaderOptions(intersection, tnr.fullType.keyType, true)))))

    assert(Simplify(ctx, tfi2) == exp2)

    val ztfi1 = TableFilterIntervals(tzr, intervals1, true)
    val zexp1 = TableRead(tzr.typ, false, tzrr.copy(options = Some(NativeReaderOptions(intervals1, tnr.fullType.keyType, true))))
    assert(Simplify(ctx, ztfi1) == zexp1)

    val ztfi2 = TableFilterIntervals(ztfi1, intervals2, true)
    val zexp2 = TableRead(tzr.typ, false, tzrr.copy(options = Some(NativeReaderOptions(intersection, tnr.fullType.keyType, true))))

    assert(Simplify(ctx, ztfi2) == zexp2)
  }

  @Test(enabled = false) def testFilterIntervalsKeyByToFilter() {
    var t: TableIR = TableRange(100, 10)
    t = TableMapRows(t, InsertFields(Ref("row", t.typ.rowType), FastSeq(("x", I32(1) - GetField(Ref("row", t.typ.rowType), "idx")))))
    t = TableKeyBy(t, FastIndexedSeq("x"))
    t = TableFilterIntervals(t, FastIndexedSeq(Interval(Row(-10), Row(10), includesStart = true, includesEnd = false)), keep = true)

    val t2 = Simplify(ctx, t)
    assert(t2 match {
      case TableKeyBy(TableFilter(child, _), _, _) => !Exists(child, _.isInstanceOf[TableFilterIntervals])
      case _ => false
    })
  }

  @Test def testSimplifyArraySlice(): Unit = {
    val stream = StreamRange(I32(0), I32(10), I32(1))
    val streamSlice1 = Simplify(ctx, ArraySlice(ToArray(stream), I32(0), Some(I32(7))))
    assert(streamSlice1 match {
      case ToArray(StreamTake(_,_)) => true
      case _ => false
    } )
    assertEvalsTo(streamSlice1.asInstanceOf[IR], FastSeq(0, 1, 2, 3, 4, 5, 6))

    val streamSlice2 = Simplify(ctx, ArraySlice(ToArray(stream), I32(3), Some(I32(5))))
    assert(streamSlice2 match {
      case ToArray(StreamTake(StreamDrop(_,_), _)) => true
      case _ => false
    } )
    assertEvalsTo(streamSlice2.asInstanceOf[IR], FastSeq(3, 4))

    val streamSlice3 = Simplify(ctx, ArraySlice(ToArray(stream), I32(6), Some(I32(2))))
    assert(streamSlice3 match {
      case MakeArray(_, _) => true
      case _ => false
    } )
    assertEvalsTo(streamSlice3.asInstanceOf[IR], FastSeq())

    val streamSlice4 = Simplify(ctx, ArraySlice(ToArray(stream), I32(0), None))
    assert(streamSlice4 match {
      case ToArray(StreamDrop(_, _)) => true
      case _ => false
    } )
    assertEvalsTo(streamSlice4.asInstanceOf[IR], FastSeq(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
  }

  def ref(typ: Type) = Ref("", typ)

  @DataProvider(name = "unaryBooleanArithmetic")
  def unaryBooleanArithmetic: Array[Array[Any]] =
    Array(
      Array(ApplyUnaryPrimOp(Bang(), ApplyUnaryPrimOp(Bang(), ref(TBoolean))), ref(TBoolean))
    ).asInstanceOf[Array[Array[Any]]]

  @Test(dataProvider = "unaryBooleanArithmetic")
  def testUnaryBooleanSimplification(input: IR, expected: IR): Unit =
    assert(Simplify(ctx, input) == expected)

  @DataProvider(name = "unaryIntegralArithmetic")
  def unaryIntegralArithmetic: Array[Array[Any]] =
    Array(TInt32, TInt64).flatMap { typ =>
      Array(
        Array(ApplyUnaryPrimOp(Negate(), ApplyUnaryPrimOp(Negate(), ref(typ))), ref(typ)),
        Array(ApplyUnaryPrimOp(BitNot(), ApplyUnaryPrimOp(BitNot(), ref(typ))), ref(typ)),
        Array(ApplyUnaryPrimOp(Negate(), ApplyUnaryPrimOp(BitNot(), ref(typ))), ApplyUnaryPrimOp(Negate(), ApplyUnaryPrimOp(BitNot(), ref(typ)))),
        Array(ApplyUnaryPrimOp(BitNot(), ApplyUnaryPrimOp(Negate(), ref(typ))), ApplyUnaryPrimOp(BitNot(), ApplyUnaryPrimOp(Negate(), ref(typ))))
      ).asInstanceOf[Array[Array[Any]]]
    }

  @Test(dataProvider = "unaryIntegralArithmetic")
  def testUnaryIntegralSimplification(input: IR, expected: IR): Unit =
    assert(Simplify(ctx, input) == expected)

  @DataProvider(name="binaryIntegralArithmetic")
  def binaryIntegralArithmetic: Array[Array[Any]] =
    Array((Literal.coerce(TInt32, _)) -> TInt32, (Literal.coerce(TInt64, _)) -> TInt64).flatMap { case (pure, typ) =>
      Array.concat(
        Array(
          // Addition
          Array(ApplyBinaryPrimOp(Add(), ref(typ), ref(typ)), ApplyBinaryPrimOp(Multiply(), pure(2), ref(typ))),
          Array(ApplyBinaryPrimOp(Add(), pure(0), ref(typ)), ref(typ)),
          Array(ApplyBinaryPrimOp(Add(), ref(typ), pure(0)), ref(typ)),

          // Subtraction
          Array(ApplyBinaryPrimOp(Subtract(), ref(typ), ref(typ)), pure(0)),
          Array(ApplyBinaryPrimOp(Subtract(), pure(0), ref(typ)), ApplyUnaryPrimOp(Negate(), ref(typ))),
          Array(ApplyBinaryPrimOp(Subtract(), ref(typ), pure(0)), ref(typ)),

          // Multiplication
          Array(ApplyBinaryPrimOp(Multiply(), pure(0), ref(typ)), pure(0)),
          Array(ApplyBinaryPrimOp(Multiply(), ref(typ), pure(0)), pure(0)),
          Array(ApplyBinaryPrimOp(Multiply(), pure(1), ref(typ)), ref(typ)),
          Array(ApplyBinaryPrimOp(Multiply(), ref(typ), pure(1)), ref(typ)),
          Array(ApplyBinaryPrimOp(Multiply(), pure(-1), ref(typ)), ApplyUnaryPrimOp(Negate(), ref(typ))),
          Array(ApplyBinaryPrimOp(Multiply(), ref(typ), pure(-1)), ApplyUnaryPrimOp(Negate(), ref(typ))),

          // Div (truncated to -Inf)
          Array(ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), ref(typ)), pure(1)),
          Array(ApplyBinaryPrimOp(RoundToNegInfDivide(), pure(0), ref(typ)), pure(0)),
          Array(ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), pure(0)), Die("division by zero", typ)),
          Array(ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), pure(1)), ref(typ)),
          Array(ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), pure(-1)), ApplyUnaryPrimOp(Negate(), ref(typ))),

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
            Array(ApplyBinaryPrimOp(shift, ref(typ), I32(0)), ref(typ))
          )
        }.asInstanceOf[Array[Array[Any]]]
      )
    }

  @Test(dataProvider = "binaryIntegralArithmetic")
  def testBinaryIntegralSimplification(input: IR, expected: IR): Unit =
    assert(Simplify(ctx, input) == expected)

  @DataProvider(name = "floatingIntegralArithmetic")
  def binaryFloatingArithmetic: Array[Array[Any]] =
    Array((Literal.coerce(TFloat32, _)) -> TFloat32, (Literal.coerce(TFloat64, _)) -> TFloat64).flatMap { case (pure, typ) =>
      Array(
        // Addition
        Array(ApplyBinaryPrimOp(Add(), pure(0), ref(typ)), ref(typ)),
        Array(ApplyBinaryPrimOp(Add(), ref(typ), pure(0)), ref(typ)),

        // Subtraction
        Array(ApplyBinaryPrimOp(Subtract(), pure(0), ref(typ)), ApplyUnaryPrimOp(Negate(), ref(typ))),
        Array(ApplyBinaryPrimOp(Subtract(), ref(typ), pure(0)), ref(typ)),

        // Multiplication
        Array(ApplyBinaryPrimOp(Multiply(), pure(1), ref(typ)), ref(typ)),
        Array(ApplyBinaryPrimOp(Multiply(), ref(typ), pure(1)), ref(typ)),
        Array(ApplyBinaryPrimOp(Multiply(), pure(-1), ref(typ)), ApplyUnaryPrimOp(Negate(), ref(typ))),
        Array(ApplyBinaryPrimOp(Multiply(), ref(typ), pure(-1)), ApplyUnaryPrimOp(Negate(), ref(typ))),

        // Div (truncated to -Inf)
        Array(ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), pure(1)), ref(typ)),
        Array(ApplyBinaryPrimOp(RoundToNegInfDivide(), ref(typ), pure(-1)), ApplyUnaryPrimOp(Negate(), ref(typ))),
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
        FastIndexedSeq(2, 2),
        10
      )

    Array(
      Array(BlockMatrixBroadcast(matrix, 0 to 1, matrix.shape, matrix.blockSize), matrix),
      Array(BlockMatrixMap(matrix, "x", Ref("x", TFloat64), true), matrix),
      Array(BlockMatrixMap(matrix, "x", ref(TFloat64), true), BlockMatrixBroadcast(
        ValueToBlockMatrix(ref(TFloat64), FastIndexedSeq(1, 1), matrix.blockSize),
        FastIndexedSeq(),
        matrix.shape,
        matrix.blockSize
      )),
      Array(BlockMatrixMap(matrix, "x", F64(2356), true), BlockMatrixBroadcast(
        ValueToBlockMatrix(F64(2356), FastIndexedSeq(1, 1), matrix.blockSize),
        FastIndexedSeq(),
        matrix.shape,
        matrix.blockSize
      )),
    ).asInstanceOf[Array[Array[Any]]]
  }

  @Test(dataProvider = "blockMatrixRules")
  def testBlockMatrixSimplification(input: BlockMatrixIR, expected: BlockMatrixIR): Unit =
    assert(Simplify(ctx, input) == expected)
}


