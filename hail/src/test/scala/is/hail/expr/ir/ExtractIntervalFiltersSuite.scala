package is.hail.expr.ir

import is.hail.rvd.RVDPartitioner
import is.hail.types.virtual._
import is.hail.utils.{FastIndexedSeq, FastSeq, Interval, IntervalEndpoint}
import is.hail.variant.{Locus, ReferenceGenome}
import is.hail.{ExecStrategy, HailSuite}
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class ExtractIntervalFiltersSuite extends HailSuite { outer =>

  val ref1 = Ref("foo", TStruct("w" -> TInt32, "x" -> TInt32, "y" -> TBoolean))
  val unknownBool = GetField(ref1, "y")
  val k1 = GetField(ref1, "x")
  val k1Full = SelectFields(ref1, FastSeq("x"))
  val ref1Key = FastIndexedSeq("x")

  val structRef = Ref("foo", TStruct("x" -> TInt32, "y" -> TInt32, "z" -> TInt32))
  val structRefKey = FastIndexedSeq("y", "z")

  val structT1 = TStruct("y" -> TInt32, "z" -> TInt32)
  val structT2 = TStruct("y" -> TInt32)

  val fullKeyRefs = Array(
    SelectFields(structRef, structRefKey),
    MakeStruct(FastSeq("y" -> GetField(structRef, "y"), "z" -> GetField(structRef, "z"))))

  val prefixKeyRefs = Array(
    SelectFields(structRef, FastSeq("y")),
    MakeStruct(FastSeq("y" -> GetField(structRef, "y"))))

  def wrappedIntervalEndpoint(x: Any, sign: Int) = IntervalEndpoint(Row(x), sign)

  def grch38: ReferenceGenome = ctx.getReference(ReferenceGenome.GRCh38)

  def check(filter: IR, expectedIntervals: Seq[Interval], residualFilter: IR, probes: IndexedSeq[Row], rowRef: Ref, key: IR): Unit = {
    val (rw, intervals) = ExtractIntervalFilters.extractPartitionFilters(ctx, filter, rowRef, key.typ.asInstanceOf[TStruct].fieldNames).get
    assert(rw == residualFilter)
    assert(intervals == expectedIntervals)

    val keyType = key.typ.asInstanceOf[TStruct]

    val irIntervals: IR = Literal(
      TArray(RVDPartitioner.intervalIRRepresentation(keyType)),
      expectedIntervals.map { i =>
        RVDPartitioner.intervalToIRRepresentation(i, keyType.size)
      })

    val filterIsTrue = Coalesce(FastSeq(filter, False()))
    val residualIsTrue = Coalesce(FastSeq(residualFilter, False()))
    val keyInIntervals = invoke("partitionerContains", TBoolean, irIntervals, key)
    val accRef = Ref(genUID(), TBoolean)

    val testIR = StreamFold(
      ToStream(Literal(TArray(rowRef.typ), probes)),
      True(),
      accRef.name,
      rowRef.name,
      ApplyComparisonOp(EQ(TBoolean),
        filterIsTrue,
        invoke("land", TBoolean, keyInIntervals, residualIsTrue)))

    assertEvalsTo(testIR, true)(ExecStrategy.compileOnly)
  }

  @Test def testKeyComparison() {
    def check(
      op: ComparisonOp[Boolean],
      point: IR,
      trueIntervals: IndexedSeq[Interval],
      falseIntervals: IndexedSeq[Interval],
    ) {
      val testRows = FastIndexedSeq(
        Row(0, -1, true),
        Row(0,  0, true),
        Row(0,  1, true),
        Row(0, null, true))

      outer.check(ApplyComparisonOp(op, k1, point), trueIntervals, True(), testRows, ref1, k1Full)
      outer.check(ApplyComparisonOp(ComparisonOp.swap(op), point, k1), trueIntervals, True(), testRows, ref1, k1Full)
      outer.check(ApplyComparisonOp(ComparisonOp.negate(op), k1, point), falseIntervals, True(), testRows, ref1, k1Full)
      outer.check(ApplyComparisonOp(ComparisonOp.swap(ComparisonOp.negate(op)), point, k1), falseIntervals, True(), testRows, ref1, k1Full)
    }

    check(LT(TInt32), I32(0),
      FastIndexedSeq(Interval(Row(), Row(0), true, false)),
      FastIndexedSeq(Interval(Row(0), Row(null), true, false)))
    check(GT(TInt32), I32(0),
      FastIndexedSeq(Interval(Row(0), Row(null), false, false)),
      FastIndexedSeq(Interval(Row(), Row(0), true, true)))
    check(EQ(TInt32), I32(0),
      FastIndexedSeq(Interval(Row(0), Row(0), true, true)),
      FastIndexedSeq(
        Interval(Row(), Row(0), true, false),
        Interval(Row(0), Row(null), false, false)))

    // These are never true (always missing), extracts the empty set of intervals
    check(LT(TInt32), NA(TInt32),
      FastIndexedSeq(),
      FastIndexedSeq())
    check(GT(TInt32), NA(TInt32),
      FastIndexedSeq(),
      FastIndexedSeq())
    check(EQ(TInt32), NA(TInt32),
      FastIndexedSeq(),
      FastIndexedSeq())

    check(EQWithNA(TInt32), I32(0),
      FastIndexedSeq(Interval(Row(0), Row(0), true, true)),
      FastIndexedSeq(
        Interval(Row(), Row(0), true, false),
        Interval(Row(0), Row(), false, true)))
    check(EQWithNA(TInt32), NA(TInt32),
      FastIndexedSeq(Interval(Row(null), Row(), true, true)),
      FastIndexedSeq(Interval(Row(), Row(null), true, false)))

//    check(IsNA(k1),
//      Interval(Row(null), Row(), true, true))

    assert(ExtractIntervalFilters.extractPartitionFilters(ctx, ApplyComparisonOp(Compare(TInt32), I32(0), k1), ref1, ref1Key).isEmpty)
  }

  @Test def testLiteralContains() {
    def check(node: IR, expectedIntervals: Interval*) {
      val testRows = FastIndexedSeq(
        Row(0, 1, true),
        Row(0, 5, true),
        Row(0, 10, true),
        Row(0, null, true))
      outer.check(node, expectedIntervals, True(), testRows, ref1, k1Full)
    }

    for  {
      lit <- Array(
        Literal(TSet(TInt32), Set(null, 10, 1)),
        Literal(TArray(TInt32), FastIndexedSeq(10, 1, null)),
        Literal(TDict(TInt32, TString), Map(1 -> "foo", (null, "bar"), 10 -> "baz")))
    } {
      val ir = invoke("contains", TBoolean, lit, k1)
      check(ir,
        Interval(Row(1), Row(1), true, true),
        Interval(Row(10), Row(10), true, true),
        Interval(Row(null), Row(), true, true))

      val ir2 = ApplyUnaryPrimOp(Bang, ir)
      check(ir2,
        Interval(Row(), Row(1), true, false),
        Interval(Row(1), Row(10), false, false),
        Interval(Row(10), Row(null), false, false))
    }

    for  {
      lit <- Array(
        Literal(TSet(TInt32), Set(10, 1)),
        Literal(TArray(TInt32), FastIndexedSeq(10, 1)),
        Literal(TDict(TInt32, TString), Map(1 -> "foo", 10 -> "baz")))
    } {
      val ir = invoke("contains", TBoolean, lit, k1)
      check(ir,
        Interval(Row(1), Row(1), true, true),
        Interval(Row(10), Row(10), true, true))

      val ir2 = ApplyUnaryPrimOp(Bang, ir)
      check(ir2,
        Interval(Row(), Row(1), true, false),
        Interval(Row(1), Row(10), false, false),
        Interval(Row(10), Row(), false, true))
    }
  }

  @Test def testLiteralContainsStruct() {
    hc // force initialization

    def check(node: IR, expectedIntervals: Interval*) {
      val testRows = FastIndexedSeq(
        Row(0, 1, 2),
        Row(0, 3, 4),
        Row(0, 3, null),
        Row(0, 5, null),
        Row(0, 5, 5),
        Row(0, null, 1),
        Row(0, null, null))
      outer.check(node, expectedIntervals, True(), testRows, structRef, fullKeyRefs(0))
    }

    for {
      lit <- Array(
        Literal(TSet(structT1), Set(Row(1, 2), Row(3, 4), Row(3, null))),
        Literal(TArray(structT1), FastIndexedSeq(Row(3, 4), Row(1, 2), null, Row(3, null))),
        Literal(TDict(structT1, TString), Map(Row(1, 2) -> "foo", Row(3, 4) -> "bar", Row(3, null) -> "baz")))
    } {
      for (k <- fullKeyRefs) {
        val ir = invoke("contains", TBoolean, lit, k)

        check(ir,
          Interval(Row(1, 2), Row(1, 2), true, true),
          Interval(Row(3, 4), Row(3, 4), true, true),
          Interval(Row(3, null), Row(3, null), true, true))

        val ir2 = ApplyUnaryPrimOp(Bang, ir)
        check(ir2,
          Interval(Row(), Row(1, 2), true, false),
          Interval(Row(1, 2), Row(3, 4), false, false),
          Interval(Row(3, 4), Row(3, null), false, false),
          Interval(Row(3, null), Row(), false, true))
      }
    }

    for {
      lit <- Array(
        Literal(TSet(structT2), Set(Row(1), Row(3), null, Row(null))),
        Literal(TArray(structT2), FastIndexedSeq(null, Row(3), Row(null), Row(1))),
        Literal(TDict(structT2, TString), Map(Row(1) -> "foo", Row(null) -> "baz", Row(3) -> "bar")))
    } {
      for (k <- prefixKeyRefs) {
        val ir = invoke("contains", TBoolean, lit, k)

        val (rw, i) = ExtractIntervalFilters.extractPartitionFilters(ctx, ir, structRef, structRefKey).get
        assert(rw == True())
        assert(i == FastSeq(
          Interval(Row(1), Row(1), true, true),
          Interval(Row(3), Row(3), true, true),
          Interval(Row(null), Row(), true, true)))

        val ir2 = ApplyUnaryPrimOp(Bang, ir)
        val (rw2, i2) = ExtractIntervalFilters.extractPartitionFilters(ctx, ir2, structRef, structRefKey).get
        assert(rw2 == True())
        assert(i2 == FastSeq(
          Interval(Row(), Row(1), true, false),
          Interval(Row(1), Row(3), false, false),
          Interval(Row(3), Row(null), false, false)))
      }
    }
  }

  @Test def testIntervalContains() {
    def check(node: IR, expectedIntervals: Interval*) {
      val testRows = FastIndexedSeq(
        Row(0, 0, true),
        Row(0, 1, true),
        Row(0, 5, true),
        Row(0, 10, true),
        Row(0, null, true))
      outer.check(node, expectedIntervals, True(), testRows, ref1, k1Full)
    }

    val interval = Interval(1, 5, false, true)
    val ir = invoke("contains", TBoolean, Literal(TInterval(TInt32), interval), k1)
    check(ir, Interval(Row(1), Row(5), false, true))

    val ir2 = ApplyUnaryPrimOp(Bang, ir)
    check(ir2,
      Interval(Row(), Row(1), true, true),
      Interval(Row(5), Row(null), false, false))
  }

  @Test def testIntervalContainsStruct() {
    val fullInterval = Interval(Row(1, 1), Row(2, 2), false, true)
    val prefixInterval = Interval(Row(1), Row(2), false, true)

    def check(node: IR, expectedIntervals: Interval*) {
      val testRows = FastIndexedSeq(
        Row(0, null, 0),
        Row(0, 0, 0),
        Row(0, 0, null),
        Row(0, 1, 0),
        Row(0, 1, 1),
        Row(0, 1, null),
        Row(0, 2, 1),
        Row(0, 2, 2),
        Row(0, 2, null),
        Row(0, 3, 0),
        Row(0, null, null))
      outer.check(node, expectedIntervals, True(), testRows, structRef, fullKeyRefs(0))
    }

    for (k <- fullKeyRefs) {
      val ir = invoke("contains", TBoolean, Literal(TInterval(structT1), fullInterval), k)
      check(ir, fullInterval)

      val ir2 = ApplyUnaryPrimOp(Bang, ir)
      check(ir2,
        Interval(Row(), Row(1, 1), true, true),
        Interval(Row(2, 2), Row(), false, true))
    }

    for (k <- prefixKeyRefs) {
      val ir = invoke("contains", TBoolean, Literal(TInterval(structT2), prefixInterval), k)
      check(ir, prefixInterval)

      val ir2 = ApplyUnaryPrimOp(Bang, ir)
      check(ir2,
        Interval(Row(), Row(1), true, true),
        Interval(Row(2), Row(), false, true))
    }
  }

  @Test def testLocusContigComparison() {
    hc // force initialization
    val ref = Ref("foo", TStruct("x" -> TLocus(ReferenceGenome.GRCh38)))
    val k = GetField(ref, "x")

    def check(node: IR, expectedIntervals: Interval*) {
      val testRows = FastIndexedSeq(
        Row(Locus("chr1", 5)),
        Row(Locus("chr2", 1)),
        Row(Locus("chr2", 1000)),
        Row(Locus("chr3", 5)),
        Row(null))
      outer.check(node, expectedIntervals, True(), testRows, ref, ref)
    }

    val ir1 = ApplyComparisonOp(EQ(TString), Str("chr2"), invoke("contig", TString, k))
    val ir2 = ApplyComparisonOp(EQ(TString), invoke("contig", TString, k), Str("chr2"))
    val ir3 = ApplyUnaryPrimOp(Bang, ir2)

    check(ir1,
      Interval(Row(Locus("chr2", 1)), Row(Locus("chr2", grch38.contigLength("chr2"))), true, false))

    check(ir2,
      Interval(Row(Locus("chr2", 1)), Row(Locus("chr2", grch38.contigLength("chr2"))), true, false))

    check(ir3,
      Interval(Row(), Row(Locus("chr2", 1)), true, false),
      Interval(Row(Locus("chr2", grch38.contigLength("chr2"))), Row(null), true, false))
  }

  @Test def testLocusPositionComparison() {
    hc // force initialization
    val ref = Ref("foo", TStruct("x" -> TLocus(ReferenceGenome.GRCh38)))
    val k = GetField(ref, "x")
    val pos = invoke("position", TInt32, k)

    def check(
      op: ComparisonOp[Boolean],
      point: Int,
      truePosIntervals: IndexedSeq[Interval],
      falsePosIntervals: IndexedSeq[Interval],
    ) {
      val trueIntervals = ExtractIntervalFilters.liftPosIntervalsToLocus(truePosIntervals, grch38, ctx)
      val falseIntervals = ExtractIntervalFilters.liftPosIntervalsToLocus(falsePosIntervals, grch38, ctx)

      val testRows = FastIndexedSeq(
        Row(Locus("chr1", 1)),
        Row(Locus("chr1", 5)),
        Row(Locus("chr1", 100)),
        Row(Locus("chr1", 105)),
        Row(Locus("chr2", 1)),
        Row(Locus("chr2", 5)),
        Row(Locus("chr2", 100)),
        Row(Locus("chr3", 105)),
        Row(null))

      outer.check(ApplyComparisonOp(op, pos, I32(point)), trueIntervals, True(), testRows, ref, ref)
      outer.check(ApplyComparisonOp(ComparisonOp.swap(op), I32(point), pos), trueIntervals, True(), testRows, ref, ref)
      outer.check(ApplyComparisonOp(ComparisonOp.negate(op), pos, I32(point)), falseIntervals, True(), testRows, ref, ref)
      outer.check(ApplyComparisonOp(ComparisonOp.swap(ComparisonOp.negate(op)), I32(point), pos), falseIntervals, True(), testRows, ref, ref)
    }

    check(GT(TInt32), 100,
      FastIndexedSeq(Interval(Row(100), Row(null), false, false)),
      FastIndexedSeq(Interval(Row(), Row(100), true, true)))
    check(GT(TInt32), -1000,
        FastIndexedSeq(Interval(Row(1), Row(null), true, false)),
        FastIndexedSeq())

    check(LT(TInt32), 100,
      FastIndexedSeq(Interval(Row(), Row(100), true, false)),
      FastIndexedSeq(Interval(Row(100), Row(null), true, false)))
    check(LT(TInt32), -1000,
      FastIndexedSeq(),
      FastIndexedSeq(Interval(Row(), Row(null), true, false)))

    check(EQ(TInt32), 100,
      FastIndexedSeq(Interval(Row(100), Row(100), true, true)),
      FastIndexedSeq(
        Interval(Row(), Row(100), true, false),
        Interval(Row(100), Row(null), false, false)))
    check(EQ(TInt32), -1000,
      FastIndexedSeq(),
      FastIndexedSeq(Interval(Row(), Row(null), true, false)))

    check(EQWithNA(TInt32), 100,
      FastIndexedSeq(Interval(Row(100), Row(100), true, true)),
      FastIndexedSeq(
        Interval(Row(), Row(100), true, false),
        Interval(Row(100), Row(), false, true)))
    check(EQWithNA(TInt32), -1000,
      FastIndexedSeq(),
      FastIndexedSeq(Interval(Row(), Row(), true, true)))

    assert(ExtractIntervalFilters.extractPartitionFilters(ctx, ApplyComparisonOp(Compare(TInt32), I32(0), pos), ref, ref1Key).isEmpty)
  }

  @Test def testLocusContigContains() {
    hc // force initialization
    val ref = Ref("foo", TStruct("x" -> TLocus(ReferenceGenome.GRCh38)))
    val k = GetField(ref, "x")
    val contig = invoke("contig", TString, k)

    for {
      lit <- Array(
        Literal(TSet(TString), Set("chr10", "chr1", null, "foo")),
        Literal(TArray(TString), FastIndexedSeq("foo", "chr10", null, "chr1")),
        Literal(TDict(TString, TString), Map("chr1" -> "foo", "chr10" -> "bar", "foo" -> "baz", (null, "quux"))))
    } {
      val ir = invoke("contains", TBoolean, lit, contig)

      val (rw, intervals) = ExtractIntervalFilters.extractPartitionFilters(ctx, ir, ref, ref1Key).get
      assert(rw == True())
      assert(intervals == FastSeq(
        Interval(
          Row(Locus("chr1", 1)),
          Row(Locus("chr1", grch38.contigLength("chr1"))),
          true, false),
        Interval(
          Row(Locus("chr10", 1)),
          Row(Locus("chr10", grch38.contigLength("chr10"))),
          true, false),
        Interval(Row(null), Row(), true, true)))

      val ir2 = ApplyUnaryPrimOp(Bang, ir)
      val (rw2, intervals2) = ExtractIntervalFilters.extractPartitionFilters(ctx, ir2, ref, ref1Key).get
      assert(rw2 == True())
      assert(intervals2 == FastSeq(
        Interval(
          Row(),
          Row(Locus("chr1", 1)),
          true, false),
        Interval(
          Row(Locus("chr1", grch38.contigLength("chr1"))),
          Row(Locus("chr10", 1)),
          true, false),
        Interval(
          Row(Locus("chr10", grch38.contigLength("chr10"))),
          Row(null),
          true, false)))
    }
  }

  @Test def testIntervalListFold() {
    val inIntervals = FastIndexedSeq(
      Interval(0, 10, true, false),
      Interval(20, 25, true, false),
      Interval(-10, 5, true, false))
    val inIntervalsWithNull = FastIndexedSeq(
      Interval(0, 10, true, false),
      null,
      Interval(20, 25, true, false),
      Interval(-10, 5, true, false))

    def containsKey(intervals: IndexedSeq[Interval]) = StreamFold(
      ToStream(Literal(TArray(TInterval(TInt32)), intervals)),
      False(),
      "acc",
      "elt",
      invoke("lor", TBoolean,
        Ref("acc", TBoolean),
        invoke("contains", TBoolean, Ref("elt", TInterval(TInt32)), k1)))

    val ir1 = containsKey(inIntervals)
    val ir2 = containsKey(inIntervalsWithNull)
    TypeCheck(ctx, ir1, BindingEnv(Env(ref1.name -> ref1.typ)))
    TypeCheck(ctx, ir2, BindingEnv(Env(ref1.name -> ref1.typ)))

    val (rw, intervals) = ExtractIntervalFilters.extractPartitionFilters(ctx, ir1, ref1, ref1Key).get
    assert(rw == True())
    assert(intervals == FastSeq(
      Interval(Row(-10), Row(10), true, false),
      Interval(Row(20), Row(25), true, false)))

    val notIR1 = ApplyUnaryPrimOp(Bang, ir1)
    val (rw2, intervals2) = ExtractIntervalFilters.extractPartitionFilters(ctx, notIR1, ref1, ref1Key).get
    assert(rw2 == True())
    assert(intervals2 == FastSeq(
      Interval(Row(), Row(-10), true, false),
      Interval(Row(10), Row(20), true, false),
      Interval(Row(25), Row(null), true, false)))

    // Whenever ir1 would be false, ir2 is instead missing, because of the null
    // In particular, it is never false, so notIR2 filters everything
    val (rw3, intervals3) = ExtractIntervalFilters.extractPartitionFilters(ctx, ir2, ref1, ref1Key).get
    assert(rw3 == True())
    assert(intervals3 == FastSeq(
      Interval(Row(-10), Row(10), true, false),
      Interval(Row(20), Row(25), true, false)))

    val notIR2 = ApplyUnaryPrimOp(Bang, ir2)
    val (rw4, intervals4) = ExtractIntervalFilters.extractPartitionFilters(ctx, notIR2, ref1, ref1Key).get
    assert(rw4 == True())
    assert(intervals4 == FastSeq())
  }

  @Test def testDisjunction() {
    val ir1 = ApplyComparisonOp(LT(TInt32), k1, I32(5))
    val ir2 = ApplyComparisonOp(GT(TInt32), k1, I32(10))

    val (rw, intervals) = ExtractIntervalFilters.extractPartitionFilters(ctx, invoke("lor", TBoolean, ir1, ir2), ref1, ref1Key).get
    assert(rw == True())
    assert(intervals == FastSeq(Interval(Row(), Row(5), true, false), Interval(Row(10), Row(null), false, false)))

    assert(ExtractIntervalFilters.extractPartitionFilters(ctx, invoke("lor", TBoolean, ir1, Ref("foo", TBoolean)), ref1, ref1Key).isEmpty)

    val ir3 = invoke("land", TBoolean,
                     ApplyUnaryPrimOp(Bang, invoke("lor", TBoolean, ir1, unknownBool)),
                     ApplyUnaryPrimOp(Bang, invoke("lor", TBoolean, ir2, unknownBool)))
    val (rw3, intervals3) = ExtractIntervalFilters.extractPartitionFilters(ctx, ir3, ref1, ref1Key).get
    assert(rw3 == invoke("land", TBoolean,
                         ApplyUnaryPrimOp(Bang, invoke("lor", TBoolean, False(), unknownBool)),
                         ApplyUnaryPrimOp(Bang, invoke("lor", TBoolean, False(), unknownBool))))
    assert(intervals3 == FastSeq(Interval(Row(5), Row(10), true, true)))
  }

  @Test def testConjunction() {
    val ir1 = ApplyComparisonOp(GT(TInt32), k1, I32(0))
    val ir2 = ApplyComparisonOp(GT(TInt32), k1, I32(10))

    val (rw1, intervals1) = ExtractIntervalFilters.extractPartitionFilters(ctx, invoke("land", TBoolean, ir1, ir2), ref1, ref1Key).get
    assert(rw1 == True())
    assert(intervals1 == FastSeq(
      Interval(Row(10), Row(null), false, false)))

    val (rw2, intervals2) = ExtractIntervalFilters.extractPartitionFilters(ctx, invoke("land", TBoolean, unknownBool, ir2), ref1, ref1Key).get
    assert(rw2 == invoke("land", TBoolean, unknownBool, True()))
    assert(intervals2 == FastSeq(
      Interval(Row(10), Row(null), false, false)))

    assert(ExtractIntervalFilters.extractPartitionFilters(ctx, invoke("land", TBoolean, unknownBool, unknownBool), ref1, ref1Key).isEmpty)
  }

  @Test def testIntegration() {
    hc // force initialization
    val tab1 = TableRange(10, 5)

    def k = GetField(Ref("row", tab1.typ.rowType), "idx")

    val tf = TableFilter(tab1,
      Coalesce(FastSeq(invoke("land", TBoolean,
        ApplyComparisonOp(GT(TInt32), k, I32(3)),
        ApplyComparisonOp(LTEQ(TInt32), k, I32(9))
      ), False())))

    assert(ExtractIntervalFilters(ctx, tf).asInstanceOf[TableFilter].child.isInstanceOf[TableFilterIntervals])
    assertEvalsTo(TableCount(tf), 6L)(ExecStrategy.interpretOnly)
  }
}
