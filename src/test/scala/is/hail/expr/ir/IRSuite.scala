package is.hail.expr.ir

import is.hail.SparkSuite
import is.hail.expr.types._
import is.hail.TestUtils._
import is.hail.annotations.BroadcastRow
import is.hail.expr.Parser
import is.hail.table.{Ascending, Descending, SortField, Table}
import is.hail.utils._
import is.hail.variant.MatrixTable
import org.apache.spark.sql.Row
import org.testng.annotations.{DataProvider, Test}

class IRSuite extends SparkSuite {
  @Test def testI32() {
    assertEvalsTo(I32(5), 5)
  }

  @Test def testI64() {
    assertEvalsTo(I64(5), 5L)
  }

  @Test def testF32() {
    assertEvalsTo(F32(3.14f), 3.14f)
  }

  @Test def testF64() {
    assertEvalsTo(F64(3.14), 3.14)
  }

  @Test def testStr() {
    assertEvalsTo(Str("Hail"), "Hail")
  }

  @Test def testTrue() {
    assertEvalsTo(True(), true)
  }

  @Test def testFalse() {
    assertEvalsTo(False(), false)
  }

  // FIXME Void() doesn't work becuase we can't handle a void type in a tuple

  @Test def testCast() {
    assertEvalsTo(Cast(I32(5), TInt32()), 5)
    assertEvalsTo(Cast(I32(5), TInt64()), 5L)
    assertEvalsTo(Cast(I32(5), TFloat32()), 5.0f)
    assertEvalsTo(Cast(I32(5), TFloat64()), 5.0)

    assertEvalsTo(Cast(I64(5), TInt32()), 5)
    assertEvalsTo(Cast(I64(0xf29fb5c9af12107dL), TInt32()), 0xaf12107d) // truncate
    assertEvalsTo(Cast(I64(5), TInt64()), 5L)
    assertEvalsTo(Cast(I64(5), TFloat32()), 5.0f)
    assertEvalsTo(Cast(I64(5), TFloat64()), 5.0)

    assertEvalsTo(Cast(F32(3.14f), TInt32()), 3)
    assertEvalsTo(Cast(F32(3.99f), TInt32()), 3) // truncate
    assertEvalsTo(Cast(F32(3.14f), TInt64()), 3L)
    assertEvalsTo(Cast(F32(3.14f), TFloat32()), 3.14f)
    assertEvalsTo(Cast(F32(3.14f), TFloat64()), 3.14)

    assertEvalsTo(Cast(F64(3.14), TInt32()), 3)
    assertEvalsTo(Cast(F64(3.99), TInt32()), 3) // truncate
    assertEvalsTo(Cast(F64(3.14), TInt64()), 3L)
    assertEvalsTo(Cast(F64(3.14), TFloat32()), 3.14f)
    assertEvalsTo(Cast(F64(3.14), TFloat64()), 3.14)
  }

  @Test def testNA() {
    assertEvalsTo(NA(TInt32()), null)
  }

  @Test def testIsNA() {
    assertEvalsTo(IsNA(NA(TInt32())), true)
    assertEvalsTo(IsNA(I32(5)), false)
  }

  @Test def testIf() {
    assertEvalsTo(If(True(), I32(5), I32(7)), 5)
    assertEvalsTo(If(False(), I32(5), I32(7)), 7)
    assertEvalsTo(If(NA(TBoolean()), I32(5), I32(7)), null)
    assertEvalsTo(If(True(), NA(TInt32()), I32(7)), null)
  }

  @Test def testLet() {
    assertEvalsTo(Let("v", I32(5), Ref("v", TInt32())), 5)
    assertEvalsTo(Let("v", NA(TInt32()), Ref("v", TInt32())), null)
    assertEvalsTo(Let("v", I32(5), NA(TInt32())), null)
  }

  @Test def testMakeArray() {
    assertEvalsTo(MakeArray(FastSeq(I32(5), NA(TInt32()), I32(-3)), TArray(TInt32())), FastIndexedSeq(5, null, -3))
    assertEvalsTo(MakeArray(FastSeq(), TArray(TInt32())), FastIndexedSeq())
  }

  @Test def testArrayRef() {
    assertEvalsTo(ArrayRef(MakeArray(FastIndexedSeq(I32(5), NA(TInt32())), TArray(TInt32())), I32(0)), 5)
    assertEvalsTo(ArrayRef(MakeArray(FastIndexedSeq(I32(5), NA(TInt32())), TArray(TInt32())), I32(1)), null)
    assertEvalsTo(ArrayRef(MakeArray(FastIndexedSeq(I32(5), NA(TInt32())), TArray(TInt32())), NA(TInt32())), null)

    assertFatal(ArrayRef(MakeArray(FastIndexedSeq(I32(5)), TArray(TInt32())), I32(2)), "array index out of bounds")
  }

  @Test def testArrayLen() {
    assertEvalsTo(ArrayLen(NA(TArray(TInt32()))), null)
    assertEvalsTo(ArrayLen(MakeArray(FastIndexedSeq(), TArray(TInt32()))), 0)
    assertEvalsTo(ArrayLen(MakeArray(FastIndexedSeq(I32(5), NA(TInt32())), TArray(TInt32()))), 2)
  }

  @Test def testArraySort() {
    assertEvalsTo(ArraySort(NA(TArray(TInt32())), True()), null)

    val a = MakeArray(FastIndexedSeq(I32(-7), I32(2), NA(TInt32()), I32(2)), TArray(TInt32()))
    assertEvalsTo(ArraySort(a, True()),
      FastIndexedSeq(-7, 2, 2, null))
    assertEvalsTo(ArraySort(a, False()),
      FastIndexedSeq(2, 2, -7, null))
  }

  @Test def testToSet() {
    assertEvalsTo(ToSet(NA(TArray(TInt32()))), null)

    val a = MakeArray(FastIndexedSeq(I32(-7), I32(2), NA(TInt32()), I32(2)), TArray(TInt32()))
    assertEvalsTo(ToSet(a), Set(-7, 2, null))
  }

  @Test def testToArrayFromSet() {
    val t = TSet(TInt32())
    assertEvalsTo(ToArray(NA(t)), null)
    assertEvalsTo(ToArray(In(0, t)),
      FastIndexedSeq((Set(-7, 2, null), t)),
      FastIndexedSeq(-7, 2, null))
  }

  @Test def testToArrayFromDict() {
    val t = TDict(TInt32(), TString())
    assertEvalsTo(ToArray(NA(t)), null)

    val d = Map(1 -> "a", 2 -> null, (null, "c"))
    assertEvalsTo(ToArray(In(0, t)),
      // wtf you can't do null -> ...
      FastIndexedSeq((d, t)),
      FastIndexedSeq(Row(1, "a"), Row(2, null), Row(null, "c")))
  }

  @Test def testToArrayFromArray() {
    val t = TArray(TInt32())
    assertEvalsTo(ToArray(NA(t)), null)
    assertEvalsTo(ToArray(In(0, t)),
      FastIndexedSeq((FastIndexedSeq(-7, 2, null, 2), t)),
      FastIndexedSeq(-7, 2, null, 2))
  }

  @Test def testSetContains() {
    val t = TSet(TInt32())
    assertEvalsTo(invoke("contains", NA(t), I32(2)), null)

    assertEvalsTo(invoke("contains", In(0, t), NA(TInt32())),
      FastIndexedSeq((Set(-7, 2, null), t)),
      true)
    assertEvalsTo(invoke("contains", In(0, t), I32(2)),
      FastIndexedSeq((Set(-7, 2, null), t)),
      true)
    assertEvalsTo(invoke("contains", In(0, t), I32(0)),
      FastIndexedSeq((Set(-7, 2, null), t)),
      false)
    assertEvalsTo(invoke("contains", In(0, t), I32(7)),
      FastIndexedSeq((Set(-7, 2), t)),
      false)
  }

  @Test def testDictContains() {
    val t = TDict(TInt32(), TString())
    assertEvalsTo(invoke("contains", NA(t), I32(2)), null)

    val d = Map(1 -> "a", 2 -> null, (null, "c"))
    assertEvalsTo(invoke("contains", In(0, t), NA(TInt32())),
      FastIndexedSeq((d, t)),
      true)
    assertEvalsTo(invoke("contains", In(0, t), I32(2)),
      FastIndexedSeq((d, t)),
      true)
    assertEvalsTo(invoke("contains", In(0, t), I32(0)),
      FastIndexedSeq((d, t)),
      false)
    assertEvalsTo(invoke("contains", In(0, t), I32(3)),
      FastIndexedSeq((Map(1 -> "a", 2 -> null), t)),
      false)
  }

  @Test def testArrayMap() {
    val naa = NA(TArray(TInt32()))
    val a = MakeArray(Seq(I32(3), NA(TInt32()), I32(7)), TArray(TInt32()))

    assertEvalsTo(ArrayMap(naa, "a", I32(5)), null)

    assertEvalsTo(ArrayMap(a, "a", ApplyBinaryPrimOp(Add(), Ref("a", TInt32()), I32(1))), FastIndexedSeq(4, null, 8))

    assertEvalsTo(Let("a", I32(5),
      ArrayMap(a, "a", Ref("a", TInt32()))),
      FastIndexedSeq(3, null, 7))
  }

  @Test def testArrayFilter() {
    val naa = NA(TArray(TInt32()))
    val a = MakeArray(Seq(I32(3), NA(TInt32()), I32(7)), TArray(TInt32()))

    assertEvalsTo(ArrayFilter(naa, "x", True()), null)

    assertEvalsTo(ArrayFilter(a, "x", NA(TBoolean())), FastIndexedSeq())
    assertEvalsTo(ArrayFilter(a, "x", False()), FastIndexedSeq())
    assertEvalsTo(ArrayFilter(a, "x", True()), FastIndexedSeq(3, null, 7))

    assertEvalsTo(ArrayFilter(a, "x",
      IsNA(Ref("x", TInt32()))), FastIndexedSeq(null))
    assertEvalsTo(ArrayFilter(a, "x",
      ApplyUnaryPrimOp(Bang(), IsNA(Ref("x", TInt32())))), FastIndexedSeq(3, 7))

    assertEvalsTo(ArrayFilter(a, "x",
      ApplyComparisonOp(LT(TInt32()), Ref("x", TInt32()), I32(6))), FastIndexedSeq(3))
  }

  @Test def testArrayFlatMap() {
    val ta = TArray(TInt32())
    val taa = TArray(ta)
    val naa = NA(taa)
    val naaa = MakeArray(FastIndexedSeq(NA(ta), NA(ta)), taa)
    val a = MakeArray(FastIndexedSeq(
      MakeArray(FastIndexedSeq(I32(7), NA(TInt32())), ta),
      NA(ta),
      MakeArray(FastIndexedSeq(I32(2)), ta)),
      taa)

    assertEvalsTo(ArrayFlatMap(naa, "a", MakeArray(FastIndexedSeq(I32(5)), ta)), null)

    assertEvalsTo(ArrayFlatMap(naaa, "a", Ref("a", ta)), FastIndexedSeq())

    assertEvalsTo(ArrayFlatMap(a, "a", Ref("a", ta)), FastIndexedSeq(7, null, 2))

    assertEvalsTo(ArrayFlatMap(ArrayRange(I32(0), I32(3), I32(1)), "i", ArrayRef(a, Ref("i", TInt32()))), FastIndexedSeq(7, null, 2))

    assertEvalsTo(Let("a", I32(5),
      ArrayFlatMap(a, "a", Ref("a", ta))),
      FastIndexedSeq(7, null, 2))
  }

  @Test def testDie() {
    assertFatal(Die("mumblefoo", TFloat64()), "mble")
  }

  @Test def testArrayRange() {
    assertEvalsTo(ArrayRange(I32(0), I32(5), NA(TInt32())), null)
    assertEvalsTo(ArrayRange(I32(0), NA(TInt32()), I32(1)), null)
    assertEvalsTo(ArrayRange(NA(TInt32()), I32(5), I32(1)), null)

    assertFatal(ArrayRange(I32(0), I32(5), I32(0)), "step size")
  }

  @Test def testTableCount() {
    assertEvalsTo(TableCount(TableRange(0, 4)), 0L)
    assertEvalsTo(TableCount(TableRange(7, 4)), 7L)
  }

  @Test def testGroupByKey() {
    def tuple(k: String, v: Int): IR = MakeTuple(Seq(Str(k), I32(v)))
    def groupby(tuples: IR*): IR = GroupByKey(MakeArray(tuples, TArray(TTuple(TString(), TInt32()))))
    val collection1 = groupby(tuple("foo", 0), tuple("bar", 4), tuple("foo", -1), tuple("bar", 0), tuple("foo", 10), tuple("", 0))

    assertEvalsTo(collection1, Map("" -> FastIndexedSeq(0), "bar" -> FastIndexedSeq(4, 0), "foo" -> FastIndexedSeq(0, -1, 10)))
  }

  @DataProvider(name="compareDifferentTypes")
  def compareDifferentTypesData(): Array[Array[Any]] = Array(
    Array(FastIndexedSeq(0.0, 0.0), TArray(+TFloat64()), TArray(TFloat64())),
    Array(Set(0, 1), TSet(+TInt32()), TSet(TInt32())),
    Array(Map(0L -> 5, 3L -> 20), TDict(+TInt64(), TInt32()), TDict(TInt64(), +TInt32())),
    Array(Interval(1, 2, includesStart = false, includesEnd = true), TInterval(+TInt32()), TInterval(TInt32())),
    Array(Row("foo", 0.0), TStruct("a" -> +TString(), "b" -> +TFloat64()), TStruct("a" -> TString(), "b" -> TFloat64())),
    Array(Row("foo", 0.0), TTuple(TString(), +TFloat64()), TTuple(+TString(), +TFloat64())),
    Array(Row(FastIndexedSeq("foo"), 0.0), TTuple(+TArray(TString()), +TFloat64()), TTuple(TArray(+TString()), +TFloat64()))
  )

  @Test(dataProvider="compareDifferentTypes")
  def testComparisonOpDifferentTypes(a: Any, t1: Type, t2: Type) {
    assertEvalsTo(ApplyComparisonOp(EQ(t1, t2), In(0, t1), In(1, t2)), IndexedSeq(a -> t1, a -> t2), true)
    assertEvalsTo(ApplyComparisonOp(LT(t1, t2), In(0, t1), In(1, t2)), IndexedSeq(a -> t1, a -> t2), false)
    assertEvalsTo(ApplyComparisonOp(GT(t1, t2), In(0, t1), In(1, t2)), IndexedSeq(a -> t1, a -> t2), false)
    assertEvalsTo(ApplyComparisonOp(LTEQ(t1, t2), In(0, t1), In(1, t2)), IndexedSeq(a -> t1, a -> t2), true)
    assertEvalsTo(ApplyComparisonOp(GTEQ(t1, t2), In(0, t1), In(1, t2)), IndexedSeq(a -> t1, a -> t2), true)
    assertEvalsTo(ApplyComparisonOp(NEQ(t1, t2), In(0, t1), In(1, t2)), IndexedSeq(a -> t1, a -> t2), false)
    assertEvalsTo(ApplyComparisonOp(EQWithNA(t1, t2), In(0, t1), In(1, t2)), IndexedSeq(a -> t1, a -> t2), true)
    assertEvalsTo(ApplyComparisonOp(NEQWithNA(t1, t2), In(0, t1), In(1, t2)), IndexedSeq(a -> t1, a -> t2), false)
  }

  @DataProvider(name = "valueIRs")
  def valueIRs(): Array[Array[IR]] = {
    val b = True()
    val c = Ref("c", TBoolean())
    val i = I32(5)
    val j = I32(7)
    val str = Str("Hail")
    val a = Ref("a", TArray(TInt32()))
    val aa = Ref("aa", TArray(TArray(TInt32())))
    val da = Ref("da", TArray(TTuple(TInt32(), TString())))
    val v = Ref("v", TInt32())
    val s = Ref("s", TStruct("x" -> TInt32(), "y" -> TInt64(), "z" -> TFloat64()))
    val t = Ref("t", TTuple(TInt32(), TInt64(), TFloat64()))

    val call = Ref("call", TCall())

    val collectSig = AggSignature(Collect(), Seq(), None, Seq(TInt32()))

    val callStatsSig = AggSignature(CallStats(), Seq(), Some(Seq(TInt32())), Seq(TCall()))

    val histSig = AggSignature(Histogram(), Seq(TFloat64(), TFloat64(), TInt32()), None, Seq(TFloat64()))

    val takeBySig = AggSignature(TakeBy(), Seq(TInt32()), None, Seq(TFloat64(), TInt32()))

    val irs = Array(
      i, I64(5), F32(3.14f), F64(3.14), str, True(), False(), Void(),
      Cast(i, TFloat64()),
      NA(TInt32()), IsNA(i),
      If(b, i, j),
      Let("v", i, v),
      Ref("x", TInt32()),
      ApplyBinaryPrimOp(Add(), i, j),
      ApplyUnaryPrimOp(Negate(), i),
      ApplyComparisonOp(EQ(TInt32()), i, j),
      MakeArray(FastSeq(i, NA(TInt32()), I32(-3)), TArray(TInt32())),
      ArrayRef(a, i),
      ArrayLen(a),
      ArrayRange(I32(0), I32(5), I32(1)),
      ArraySort(a, b),
      ToSet(a),
      ToDict(da),
      ToArray(a),
      LowerBoundOnOrderedCollection(a, i, onKey = true),
      GroupByKey(da),
      ArrayMap(a, "v", v),
      ArrayFilter(a, "v", b),
      ArrayFlatMap(aa, "v", v),
      ArrayFold(a, I32(0), "x", "v", v),
      ArrayFor(a, "v", Void()),
      ApplyAggOp(I32(0), FastIndexedSeq.empty, None, collectSig),
      ApplyAggOp(F64(-2.11), FastIndexedSeq(F64(-5.0), F64(5.0), I32(100)), None, histSig),
      ApplyAggOp(call, FastIndexedSeq.empty, Some(FastIndexedSeq(I32(2))), callStatsSig),
      ApplyAggOp(F64(-2.11), FastIndexedSeq(I32(10)), None, takeBySig),
      InitOp(I32(0), FastIndexedSeq(I32(2)), callStatsSig),
      SeqOp(I32(0), FastIndexedSeq(i), collectSig),
      SeqOp(I32(0), FastIndexedSeq(F64(-2.11), I32(17)), takeBySig),
      Begin(IndexedSeq(Void())),
      MakeStruct(Seq("x" -> i)),
      SelectFields(s, Seq("x", "z")),
      InsertFields(s, Seq("x" -> i)),
      GetField(s, "x"),
      MakeTuple(Seq(i, b)),
      GetTupleElement(t, 1),
      StringSlice(str, I32(1), I32(2)),
      StringLength(str),
      In(2, TFloat64()),
      Die("mumblefoo", TFloat64()),
      invoke("&&", b, c), // ApplySpecial
      invoke("toFloat64", i), // Apply
      invoke("isDefined", s), // ApplyIR
      Uniroot("x", F64(3.14), F64(-5.0), F64(5.0)))
    irs.map(x => Array(x))
  }

  @DataProvider(name = "tableIRs")
  def tableIRs(): Array[Array[TableIR]] = {
    try {
      val ht = Table.read(hc, "src/test/resources/backward_compatability/1.0.0/table/0.ht")
      val mt = MatrixTable.read(hc, "src/test/resources/backward_compatability/1.0.0/matrix_table/0.hmt")

      val read = ht.tir.asInstanceOf[TableRead]
      val mtRead = mt.ast.asInstanceOf[MatrixRead]
      val b = True()

      val xs: Array[TableIR] = Array(
        TableUnkey(read),
        TableDistinct(read),
        TableKeyBy(read, Array("m", "d"), Some(1)),
        TableFilter(read, b),
        read,
        MatrixColsTable(mtRead),
        TableAggregateByKey(read,
          MakeStruct(FastIndexedSeq(
            "a" -> I32(5)))),
        TableJoin(read,
          TableRange(100, 10), "inner"),
        MatrixEntriesTable(mtRead),
        MatrixRowsTable(mtRead),
        TableParallelize(
          TableType(
            TStruct("a" -> TInt32()),
            None,
            TStruct.empty()),
          FastIndexedSeq(Row(null), Row(5), Row(-3)),
          None),
        TableMapRows(read,
          MakeStruct(FastIndexedSeq(
            "a" -> GetField(Ref("row", read.typ.rowType), "f32"),
            "b" -> F64(-2.11))),
          None, None),
        TableMapGlobals(read,
          MakeStruct(FastIndexedSeq(
            "foo" -> NA(TArray(TInt32())))),
          BroadcastRow(Row(), TStruct.empty(), hc.sc)),
        TableRange(100, 10),
        TableUnion(
          FastIndexedSeq(TableRange(100, 10), TableRange(50, 10))),
        TableExplode(read, "mset"),
        TableUnkey(read),
        TableOrderBy(TableUnkey(read), FastIndexedSeq(SortField("m", Ascending), SortField("m", Descending)))
      )
      xs.map(x => Array(x))
    } catch {
      case t: Throwable =>
        println(t)
        println(t.printStackTrace())
        throw t
    }
  }

  @DataProvider(name = "matrixIRs")
  def matrixIRs(): Array[Array[MatrixIR]] = {
    try {
      val tableRead = Table.read(hc, "src/test/resources/backward_compatability/1.0.0/table/0.ht")
        .tir.asInstanceOf[TableRead]
      val read = MatrixTable.read(hc, "src/test/resources/backward_compatability/1.0.0/matrix_table/0.hmt")
        .ast.asInstanceOf[MatrixRead]
      val range = MatrixTable.range(hc, 3, 7, None)
        .ast.asInstanceOf[MatrixRead]

      val b = True()

      val newCol = MakeStruct(FastIndexedSeq(
        "col_idx" -> GetField(Ref("sa", read.typ.colType), "col_idx"),
        "new_f32" -> ApplyBinaryPrimOp(Add(),
          GetField(Ref("sa", read.typ.colType), "col_f32"),
          F32(-5.2f))))
      val newRow = MakeStruct(FastIndexedSeq(
        "row_idx" -> GetField(Ref("va", read.typ.rowType), "row_idx"),
        "new_f32" -> ApplyBinaryPrimOp(Add(),
          GetField(Ref("va", read.typ.rowType), "row_f32"),
          F32(-5.2f))))

      val xs = Array[MatrixIR](
        read,
        MatrixFilterRows(read, b),
        MatrixFilterCols(read, b),
        MatrixFilterEntries(read, b),
        MatrixChooseCols(read, Array(0, 0, 0)),
        MatrixMapCols(read, newCol, None),
        MatrixMapRows(read, newRow, None),
        MatrixMapEntries(read, MakeStruct(FastIndexedSeq(
          "global_f32" -> ApplyBinaryPrimOp(Add(),
            GetField(Ref("global", read.typ.globalType), "global_f32"),
            F32(-5.2f))))),
        MatrixCollectColsByKey(read),
        MatrixAggregateColsByKey(read, newCol),
        MatrixAggregateRowsByKey(read, newRow),
        range,
      MatrixExplodeRows(read, FastIndexedSeq("row_mset")))

      xs.map(x => Array(x))
    } catch {
      case t: Throwable =>
        println(t)
        println(t.printStackTrace())
        throw t
    }
  }

  @Test(dataProvider = "valueIRs")
  def testValueIRParser(x: IR) {
    val s = Pretty(x)
    val x2 = Parser.parse(Parser.ir_value_expr, s)
    assert(x2 == x)
  }

  @Test(dataProvider = "tableIRs")
  def testTableIRParser(x: TableIR) {
    val s = Pretty(x)
    val x2 = Parser.parse(Parser.table_ir, s)
    assert(x2 == x)
  }

  @Test(dataProvider = "matrixIRs")
  def testMatrixIRParser(x: MatrixIR) {
    val s = Pretty(x)
    val x2 = Parser.parse(Parser.matrix_ir, s)
    assert(x2 == x)
  }
}
