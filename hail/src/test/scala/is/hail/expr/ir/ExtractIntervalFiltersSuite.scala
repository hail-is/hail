package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.rvd.RVDPartitioner
import is.hail.types.virtual._
import is.hail.utils.{FastSeq, Interval, IntervalEndpoint}
import is.hail.variant.{Locus, ReferenceGenome}

import org.apache.spark.sql.Row
import org.testng.annotations.Test

class ExtractIntervalFiltersSuite extends HailSuite { outer =>

  val ref1 = Ref("foo", TStruct("w" -> TInt32, "x" -> TInt32, "y" -> TBoolean))
  val unknownBool = GetField(ref1, "y")
  val k1 = GetField(ref1, "x")
  val k1Full = SelectFields(ref1, FastSeq("x"))
  val ref1Key = FastSeq("x")

  val structRef = Ref("foo", TStruct("x" -> TInt32, "y" -> TInt32, "z" -> TInt32))
  val structRefKey = FastSeq("y", "z")

  val structT1 = TStruct("y" -> TInt32, "z" -> TInt32)
  val structT2 = TStruct("y" -> TInt32)

  val fullKeyRefs = Array(
    SelectFields(structRef, structRefKey),
    MakeStruct(FastSeq("y" -> GetField(structRef, "y"), "z" -> GetField(structRef, "z"))),
  )

  val prefixKeyRefs = Array(
    SelectFields(structRef, FastSeq("y")),
    MakeStruct(FastSeq("y" -> GetField(structRef, "y"))),
  )

  def wrappedIntervalEndpoint(x: Any, sign: Int) = IntervalEndpoint(Row(x), sign)

  def grch38: ReferenceGenome = ctx.getReference(ReferenceGenome.GRCh38)

  def lt(l: IR, r: IR): IR = ApplyComparisonOp(LT(l.typ), l, r)
  def gt(l: IR, r: IR): IR = ApplyComparisonOp(GT(l.typ), l, r)
  def lteq(l: IR, r: IR): IR = ApplyComparisonOp(LTEQ(l.typ), l, r)
  def gteq(l: IR, r: IR): IR = ApplyComparisonOp(GTEQ(l.typ), l, r)
  def eq(l: IR, r: IR): IR = ApplyComparisonOp(EQ(l.typ), l, r)
  def neq(l: IR, r: IR): IR = ApplyComparisonOp(NEQ(l.typ), l, r)
  def eqna(l: IR, r: IR): IR = ApplyComparisonOp(EQWithNA(l.typ), l, r)
  def neqna(l: IR, r: IR): IR = ApplyComparisonOp(NEQWithNA(l.typ), l, r)
  def or(l: IR, r: IR): IR = invoke("lor", TBoolean, l, r)
  def and(l: IR, r: IR): IR = invoke("land", TBoolean, l, r)
  def not(b: IR): IR = ApplyUnaryPrimOp(Bang, b)

  def check(
    filter: IR,
    rowRef: Ref,
    key: IR,
    probes: IndexedSeq[Row],
    residualFilter: IR,
    trueIntervals: Seq[Interval],
  ): Unit = {
    val result = ExtractIntervalFilters.extractPartitionFilters(
      ctx,
      filter,
      rowRef,
      key.typ.asInstanceOf[TStruct].fieldNames,
    )
    if (result.isEmpty) {
      assert(trueIntervals == FastSeq(Interval(Row(), Row(), true, true)))
      return
    }
    val (rw, intervals) = result.get
    assert(rw == residualFilter)
    if (trueIntervals != null) assert(intervals == trueIntervals)

    val keyType = key.typ.asInstanceOf[TStruct]

    val irIntervals: IR = Literal(
      TArray(RVDPartitioner.intervalIRRepresentation(keyType)),
      trueIntervals.map(i => RVDPartitioner.intervalToIRRepresentation(i, keyType.size)),
    )

    val filterIsTrue = Coalesce(FastSeq(filter, False()))
    val residualIsTrue = Coalesce(FastSeq(residualFilter, False()))
    val keyInIntervals = invoke("partitionerContains", TBoolean, irIntervals, key)
    val accRef = Ref(genUID(), TBoolean)

    val testIR = StreamFold(
      ToStream(Literal(TArray(rowRef.typ), probes)),
      True(),
      accRef.name,
      rowRef.name,
      ApplyComparisonOp(
        EQ(TBoolean),
        filterIsTrue,
        invoke("land", TBoolean, keyInIntervals, residualIsTrue),
      ),
    )

    assertEvalsTo(testIR, true)(ExecStrategy.compileOnly)
  }

  def checkAll(
    filter: IR,
    rowRef: Ref,
    key: IR,
    probes: IndexedSeq[Row],
    trueIntervals: Seq[Interval],
    falseIntervals: Seq[Interval],
    naIntervals: Seq[Interval],
    trueResidual: IR = True(),
    falseResidual: IR = True(),
    naResidual: IR = True(),
  ): Unit = {
    check(filter, rowRef, key, probes, trueResidual, trueIntervals)
    check(ApplyUnaryPrimOp(Bang, filter), rowRef, key, probes, falseResidual, falseIntervals)
    check(IsNA(filter), rowRef, key, probes, naResidual, naIntervals)
  }

  @Test def testIsNA(): Unit = {
    val testRows = FastSeq(
      Row(0, 0, true),
      Row(0, null, true),
    )
    checkAll(
      IsNA(k1),
      ref1,
      k1Full,
      testRows,
      FastSeq(Interval(Row(null), Row(), true, true)),
      FastSeq(Interval(Row(), Row(null), true, false)),
      FastSeq(),
    )
  }

  @Test def testKeyComparison() {
    def check(
      op: ComparisonOp[Boolean],
      point: IR,
      trueIntervals: IndexedSeq[Interval],
      falseIntervals: IndexedSeq[Interval],
      naIntervals: IndexedSeq[Interval],
    ) {
      val testRows = FastSeq(
        Row(0, -1, true),
        Row(0, 0, true),
        Row(0, 1, true),
        Row(0, null, true),
      )

      checkAll(
        ApplyComparisonOp(op, k1, point),
        ref1,
        k1Full,
        testRows,
        trueIntervals,
        falseIntervals,
        naIntervals,
      )
      checkAll(
        ApplyComparisonOp(ComparisonOp.swap(op), point, k1),
        ref1,
        k1Full,
        testRows,
        trueIntervals,
        falseIntervals,
        naIntervals,
      )
      checkAll(
        ApplyComparisonOp(ComparisonOp.negate(op), k1, point),
        ref1,
        k1Full,
        testRows,
        falseIntervals,
        trueIntervals,
        naIntervals,
      )
      checkAll(
        ApplyComparisonOp(ComparisonOp.swap(ComparisonOp.negate(op)), point, k1),
        ref1,
        k1Full,
        testRows,
        falseIntervals,
        trueIntervals,
        naIntervals,
      )
    }

    check(
      LT(TInt32),
      I32(0),
      FastSeq(Interval(Row(), Row(0), true, false)),
      FastSeq(Interval(Row(0), Row(null), true, false)),
      FastSeq(Interval(Row(null), Row(), true, true)),
    )
    check(
      GT(TInt32),
      I32(0),
      FastSeq(Interval(Row(0), Row(null), false, false)),
      FastSeq(Interval(Row(), Row(0), true, true)),
      FastSeq(Interval(Row(null), Row(), true, true)),
    )
    check(
      EQ(TInt32),
      I32(0),
      FastSeq(Interval(Row(0), Row(0), true, true)),
      FastSeq(
        Interval(Row(), Row(0), true, false),
        Interval(Row(0), Row(null), false, false),
      ),
      FastSeq(Interval(Row(null), Row(), true, true)),
    )

    // These are never true (always missing), extracts the empty set of intervals
    check(LT(TInt32), NA(TInt32), FastSeq(), FastSeq(), FastSeq(Interval(Row(), Row(), true, true)))
    check(GT(TInt32), NA(TInt32), FastSeq(), FastSeq(), FastSeq(Interval(Row(), Row(), true, true)))
    check(EQ(TInt32), NA(TInt32), FastSeq(), FastSeq(), FastSeq(Interval(Row(), Row(), true, true)))

    check(
      EQWithNA(TInt32),
      I32(0),
      FastSeq(Interval(Row(0), Row(0), true, true)),
      FastSeq(
        Interval(Row(), Row(0), true, false),
        Interval(Row(0), Row(), false, true),
      ),
      FastSeq(),
    )
    check(
      EQWithNA(TInt32),
      NA(TInt32),
      FastSeq(Interval(Row(null), Row(), true, true)),
      FastSeq(Interval(Row(), Row(null), true, false)),
      FastSeq(),
    )

    assert(ExtractIntervalFilters.extractPartitionFilters(
      ctx,
      ApplyComparisonOp(Compare(TInt32), I32(0), k1),
      ref1,
      ref1Key,
    ).isEmpty)
  }

  @Test def testLiteralContains() {
    def check(
      node: IR,
      trueIntervals: IndexedSeq[Interval],
      falseIntervals: IndexedSeq[Interval],
      naIntervals: IndexedSeq[Interval],
    ) {
      val testRows = FastSeq(
        Row(0, 1, true),
        Row(0, 5, true),
        Row(0, 10, true),
        Row(0, null, true),
      )
      checkAll(node, ref1, k1Full, testRows, trueIntervals, falseIntervals, naIntervals)
    }

    for {
      lit <- Array(
        Literal(TSet(TInt32), Set(null, 10, 1)),
        Literal(TArray(TInt32), FastSeq(10, 1, null)),
        Literal(TDict(TInt32, TString), Map(1 -> "foo", (null, "bar"), 10 -> "baz")),
      )
    } {
      check(
        invoke("contains", TBoolean, lit, k1),
        FastSeq(
          Interval(Row(1), Row(1), true, true),
          Interval(Row(10), Row(10), true, true),
          Interval(Row(null), Row(), true, true),
        ),
        FastSeq(
          Interval(Row(), Row(1), true, false),
          Interval(Row(1), Row(10), false, false),
          Interval(Row(10), Row(null), false, false),
        ),
        FastSeq(),
      )
    }

    for {
      lit <- Array(
        Literal(TSet(TInt32), Set(10, 1)),
        Literal(TArray(TInt32), FastSeq(10, 1)),
        Literal(TDict(TInt32, TString), Map(1 -> "foo", 10 -> "baz")),
      )
    } {
      check(
        invoke("contains", TBoolean, lit, k1),
        FastSeq(
          Interval(Row(1), Row(1), true, true),
          Interval(Row(10), Row(10), true, true),
        ),
        FastSeq(
          Interval(Row(), Row(1), true, false),
          Interval(Row(1), Row(10), false, false),
          Interval(Row(10), Row(), false, true),
        ),
        FastSeq(),
      )
    }
  }

  @Test def testLiteralContainsStruct() {
    hc // force initialization

    def check(
      node: IR,
      trueIntervals: IndexedSeq[Interval],
      falseIntervals: IndexedSeq[Interval],
      naIntervals: IndexedSeq[Interval],
    ) {
      val testRows = FastSeq(
        Row(0, 1, 2),
        Row(0, 3, 4),
        Row(0, 3, null),
        Row(0, 5, null),
        Row(0, 5, 5),
        Row(0, null, 1),
        Row(0, null, null),
      )
      checkAll(
        node,
        structRef,
        fullKeyRefs(0),
        testRows,
        trueIntervals,
        falseIntervals,
        naIntervals,
      )
    }

    for {
      lit <- Array(
        Literal(TSet(structT1), Set(Row(1, 2), Row(3, 4), Row(3, null))),
        Literal(TArray(structT1), FastSeq(Row(3, 4), Row(1, 2), Row(3, null))),
        Literal(
          TDict(structT1, TString),
          Map(Row(1, 2) -> "foo", Row(3, 4) -> "bar", Row(3, null) -> "baz"),
        ),
      )
    } {
      for (k <- fullKeyRefs) {
        check(
          invoke("contains", TBoolean, lit, k),
          IndexedSeq(
            Interval(Row(1, 2), Row(1, 2), true, true),
            Interval(Row(3, 4), Row(3, 4), true, true),
            Interval(Row(3, null), Row(3, null), true, true),
          ),
          IndexedSeq(
            Interval(Row(), Row(1, 2), true, false),
            Interval(Row(1, 2), Row(3, 4), false, false),
            Interval(Row(3, 4), Row(3, null), false, false),
            Interval(Row(3, null), Row(), false, true),
          ),
          IndexedSeq(),
        )
      }
    }

    for {
      lit <- Array(
        Literal(TSet(structT2), Set(Row(1), Row(3), Row(null))),
        Literal(TArray(structT2), FastSeq(Row(3), Row(null), Row(1))),
        Literal(TDict(structT2, TString), Map(Row(1) -> "foo", Row(null) -> "baz", Row(3) -> "bar")),
      )
    } {
      for (k <- prefixKeyRefs) {
        check(
          invoke("contains", TBoolean, lit, k),
          IndexedSeq(
            Interval(Row(1), Row(1), true, true),
            Interval(Row(3), Row(3), true, true),
            Interval(Row(null), Row(), true, true),
          ),
          IndexedSeq(
            Interval(Row(), Row(1), true, false),
            Interval(Row(1), Row(3), false, false),
            Interval(Row(3), Row(null), false, false),
          ),
          IndexedSeq(),
        )
      }
    }
  }

  @Test def testIntervalContains() {
    val interval = Interval(1, 5, false, true)
    val testRows = FastSeq(
      Row(0, 0, true),
      Row(0, 1, true),
      Row(0, 5, true),
      Row(0, 10, true),
      Row(0, null, true),
    )

    val ir = invoke("contains", TBoolean, Literal(TInterval(TInt32), interval), k1)
    checkAll(
      ir,
      ref1,
      k1Full,
      testRows,
      FastSeq(Interval(Row(1), Row(5), false, true)),
      FastSeq(
        Interval(Row(), Row(1), true, true),
        Interval(Row(5), Row(null), false, false),
      ),
      FastSeq(Interval(Row(null), Row(), true, true)),
    )
  }

  @Test def testIntervalContainsStruct() {
    val fullInterval = Interval(Row(1, 1), Row(2, 2), false, true)
    val prefixInterval = Interval(Row(1), Row(2), false, true)

    def check(
      node: IR,
      trueIntervals: IndexedSeq[Interval],
      falseIntervals: IndexedSeq[Interval],
      naIntervals: IndexedSeq[Interval],
    ) {
      val testRows = FastSeq(
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
        Row(0, null, null),
      )
      checkAll(
        node,
        structRef,
        fullKeyRefs(0),
        testRows,
        trueIntervals,
        falseIntervals,
        naIntervals,
      )
    }

    for (k <- fullKeyRefs) {
      check(
        invoke("contains", TBoolean, Literal(TInterval(structT1), fullInterval), k),
        FastSeq(fullInterval),
        FastSeq(
          Interval(Row(), Row(1, 1), true, true),
          Interval(Row(2, 2), Row(), false, true),
        ),
        FastSeq(),
      )
    }

    for (k <- prefixKeyRefs) {
      check(
        invoke("contains", TBoolean, Literal(TInterval(structT2), prefixInterval), k),
        FastSeq(prefixInterval),
        FastSeq(
          Interval(Row(), Row(1), true, true),
          Interval(Row(2), Row(), false, true),
        ),
        FastSeq(),
      )
    }
  }

  @Test def testLocusContigComparison() {
    hc // force initialization
    val ref = Ref("foo", TStruct("x" -> TLocus(ReferenceGenome.GRCh38)))
    val k = GetField(ref, "x")

    val ir1 = eq(Str("chr2"), invoke("contig", TString, k))
    val ir2 = eq(invoke("contig", TString, k), Str("chr2"))

    val testRows = FastSeq(
      Row(Locus("chr1", 5)),
      Row(Locus("chr2", 1)),
      Row(Locus("chr2", 1000)),
      Row(Locus("chr3", 5)),
      Row(null),
    )

    val trueIntervals = FastSeq(
      Interval(Row(Locus("chr2", 1)), Row(Locus("chr2", grch38.contigLength("chr2"))), true, false)
    )
    val falseIntervals = FastSeq(
      Interval(Row(), Row(Locus("chr2", 1)), true, false),
      Interval(Row(Locus("chr2", grch38.contigLength("chr2"))), Row(null), true, false),
    )
    val naIntervals = FastSeq(Interval(Row(null), Row(), true, true))

    checkAll(ir1, ref, ref, testRows, trueIntervals, falseIntervals, naIntervals)
    checkAll(ir2, ref, ref, testRows, trueIntervals, falseIntervals, naIntervals)
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
      naPosIntervals: IndexedSeq[Interval],
    ) {
      val trueIntervals =
        ExtractIntervalFilters.liftPosIntervalsToLocus(truePosIntervals, grch38, ctx)
      val falseIntervals =
        ExtractIntervalFilters.liftPosIntervalsToLocus(falsePosIntervals, grch38, ctx)
      val naIntervals = ExtractIntervalFilters.liftPosIntervalsToLocus(naPosIntervals, grch38, ctx)

      val testRows = FastSeq(
        Row(Locus("chr1", 1)),
        Row(Locus("chr1", 5)),
        Row(Locus("chr1", 100)),
        Row(Locus("chr1", 105)),
        Row(Locus("chr2", 1)),
        Row(Locus("chr2", 5)),
        Row(Locus("chr2", 100)),
        Row(Locus("chr3", 105)),
        Row(null),
      )

      checkAll(
        ApplyComparisonOp(op, pos, I32(point)),
        ref,
        ref,
        testRows,
        trueIntervals,
        falseIntervals,
        naIntervals,
      )
      checkAll(
        ApplyComparisonOp(ComparisonOp.swap(op), I32(point), pos),
        ref,
        ref,
        testRows,
        trueIntervals,
        falseIntervals,
        naIntervals,
      )
      checkAll(
        ApplyComparisonOp(ComparisonOp.negate(op), pos, I32(point)),
        ref,
        ref,
        testRows,
        falseIntervals,
        trueIntervals,
        naIntervals,
      )
      checkAll(
        ApplyComparisonOp(ComparisonOp.swap(ComparisonOp.negate(op)), I32(point), pos),
        ref,
        ref,
        testRows,
        falseIntervals,
        trueIntervals,
        naIntervals,
      )
    }

    check(
      GT(TInt32),
      100,
      FastSeq(Interval(Row(100), Row(null), false, false)),
      FastSeq(Interval(Row(), Row(100), true, true)),
      FastSeq(Interval(Row(null), Row(), true, true)),
    )
    check(
      GT(TInt32),
      -1000,
      FastSeq(Interval(Row(1), Row(null), true, false)),
      FastSeq(),
      FastSeq(Interval(Row(null), Row(), true, true)),
    )

    check(
      LT(TInt32),
      100,
      FastSeq(Interval(Row(), Row(100), true, false)),
      FastSeq(Interval(Row(100), Row(null), true, false)),
      FastSeq(Interval(Row(null), Row(), true, true)),
    )
    check(
      LT(TInt32),
      -1000,
      FastSeq(),
      FastSeq(Interval(Row(), Row(null), true, false)),
      FastSeq(Interval(Row(null), Row(), true, true)),
    )

    check(
      EQ(TInt32),
      100,
      FastSeq(Interval(Row(100), Row(100), true, true)),
      FastSeq(
        Interval(Row(), Row(100), true, false),
        Interval(Row(100), Row(null), false, false),
      ),
      FastSeq(Interval(Row(null), Row(), true, true)),
    )
    check(
      EQ(TInt32),
      -1000,
      FastSeq(),
      FastSeq(Interval(Row(), Row(null), true, false)),
      FastSeq(Interval(Row(null), Row(), true, true)),
    )

    check(
      EQWithNA(TInt32),
      100,
      FastSeq(Interval(Row(100), Row(100), true, true)),
      FastSeq(
        Interval(Row(), Row(100), true, false),
        Interval(Row(100), Row(), false, true),
      ),
      FastSeq(),
    )
    check(
      EQWithNA(TInt32),
      -1000,
      FastSeq(),
      FastSeq(Interval(Row(), Row(), true, true)),
      FastSeq(),
    )

    assert(ExtractIntervalFilters.extractPartitionFilters(
      ctx,
      ApplyComparisonOp(Compare(TInt32), I32(0), pos),
      ref,
      ref1Key,
    ).isEmpty)
  }

  @Test def testLocusContigContains() {
    hc // force initialization
    val ref = Ref("foo", TStruct("x" -> TLocus(ReferenceGenome.GRCh38)))
    val k = GetField(ref, "x")
    val contig = invoke("contig", TString, k)

    def check(
      node: IR,
      trueIntervals: IndexedSeq[Interval],
      falseIntervals: IndexedSeq[Interval],
      naIntervals: IndexedSeq[Interval],
    ) {
      val testRows = FastSeq(
        Row(Locus("chr1", 5)),
        Row(Locus("chr2", 1)),
        Row(Locus("chr10", 5)),
        Row(null),
      )
      checkAll(node, ref, ref, testRows, trueIntervals, falseIntervals, naIntervals)
    }

    for {
      lit <- Array(
        Literal(TSet(TString), Set("chr10", "chr1", null, "foo")),
        Literal(TArray(TString), FastSeq("foo", "chr10", null, "chr1")),
        Literal(
          TDict(TString, TString),
          Map("chr1" -> "foo", "chr10" -> "bar", "foo" -> "baz", (null, "quux")),
        ),
      )
    } {
      check(
        invoke("contains", TBoolean, lit, contig),
        FastSeq(
          Interval(
            Row(Locus("chr1", 1)),
            Row(Locus("chr1", grch38.contigLength("chr1"))),
            true,
            false,
          ),
          Interval(
            Row(Locus("chr10", 1)),
            Row(Locus("chr10", grch38.contigLength("chr10"))),
            true,
            false,
          ),
          Interval(Row(null), Row(), true, true),
        ),
        FastSeq(
          Interval(
            Row(),
            Row(Locus("chr1", 1)),
            true,
            false,
          ),
          Interval(
            Row(Locus("chr1", grch38.contigLength("chr1"))),
            Row(Locus("chr10", 1)),
            true,
            false,
          ),
          Interval(
            Row(Locus("chr10", grch38.contigLength("chr10"))),
            Row(null),
            true,
            false,
          ),
        ),
        FastSeq(),
      )
    }

    for {
      lit <- Array(
        Literal(TSet(TString), Set("chr10", "chr1", "foo")),
        Literal(TArray(TString), FastSeq("foo", "chr10", "chr1")),
        Literal(TDict(TString, TString), Map("chr1" -> "foo", "chr10" -> "bar", "foo" -> "baz")),
      )
    } {
      check(
        invoke("contains", TBoolean, lit, contig),
        FastSeq(
          Interval(
            Row(Locus("chr1", 1)),
            Row(Locus("chr1", grch38.contigLength("chr1"))),
            true,
            false,
          ),
          Interval(
            Row(Locus("chr10", 1)),
            Row(Locus("chr10", grch38.contigLength("chr10"))),
            true,
            false,
          ),
        ),
        FastSeq(
          Interval(
            Row(),
            Row(Locus("chr1", 1)),
            true,
            false,
          ),
          Interval(
            Row(Locus("chr1", grch38.contigLength("chr1"))),
            Row(Locus("chr10", 1)),
            true,
            false,
          ),
          Interval(
            Row(Locus("chr10", grch38.contigLength("chr10"))),
            Row(),
            true,
            true,
          ),
        ),
        FastSeq(),
      )
    }
  }

  @Test def testIntervalListFold() {
    val inIntervals = FastSeq(
      Interval(0, 10, true, false),
      Interval(20, 25, true, false),
      Interval(-10, 5, true, false),
    )
    val inIntervalsWithNull = FastSeq(
      Interval(0, 10, true, false),
      null,
      Interval(20, 25, true, false),
      Interval(-10, 5, true, false),
    )

    def check(
      node: IR,
      trueIntervals: IndexedSeq[Interval],
      falseIntervals: IndexedSeq[Interval],
      naIntervals: IndexedSeq[Interval],
    ) {
      val testRows = FastSeq(
        Row(0, -15, true),
        Row(0, -10, true),
        Row(0, -5, true),
        Row(0, 0, true),
        Row(0, 5, true),
        Row(0, 10, true),
        Row(0, 15, true),
        Row(0, 20, true),
        Row(0, 22, true),
        Row(0, 25, true),
        Row(0, 30, true),
        Row(0, null, true),
      )
      checkAll(node, ref1, k1Full, testRows, trueIntervals, falseIntervals, naIntervals)
    }

    def containsKey(intervals: IndexedSeq[Interval]) = StreamFold(
      ToStream(Literal(TArray(TInterval(TInt32)), intervals)),
      False(),
      "acc",
      "elt",
      invoke(
        "lor",
        TBoolean,
        Ref("acc", TBoolean),
        invoke("contains", TBoolean, Ref("elt", TInterval(TInt32)), k1),
      ),
    )

    check(
      containsKey(inIntervals),
      FastSeq(
        Interval(Row(-10), Row(10), true, false),
        Interval(Row(20), Row(25), true, false),
      ),
      FastSeq(
        Interval(Row(), Row(-10), true, false),
        Interval(Row(10), Row(20), true, false),
        Interval(Row(25), Row(null), true, false),
      ),
      FastSeq(Interval(Row(null), Row(), true, true)),
    )

    // Whenever the previous would be false, this is instead missing, because of the null
    // In particular, it is never false, so notIR2 filters everything
    check(
      containsKey(inIntervalsWithNull),
      FastSeq(
        Interval(Row(-10), Row(10), true, false),
        Interval(Row(20), Row(25), true, false),
      ),
      FastSeq(),
      FastSeq(
        Interval(Row(), Row(-10), true, false),
        Interval(Row(10), Row(20), true, false),
        Interval(Row(25), Row(), true, true),
      ),
    )
  }

  @Test def testDisjunction() {
    def check(
      node: IR,
      trueIntervals: IndexedSeq[Interval],
      falseIntervals: IndexedSeq[Interval],
      naIntervals: IndexedSeq[Interval],
      trueResidual: IR = True(),
      falseResidual: IR = True(),
      naResidual: IR = True(),
    ) {
      val testRows = FastSeq(
        Row(0, 0, true),
        Row(0, 0, false),
        Row(0, 5, true),
        Row(0, 5, false),
        Row(0, 7, true),
        Row(0, 7, false),
        Row(0, 10, true),
        Row(0, 10, false),
        Row(0, 15, true),
        Row(0, 15, false),
        Row(0, null, true),
        Row(0, null, false),
      )
      checkAll(node, ref1, k1Full, testRows, trueIntervals, falseIntervals, naIntervals,
        trueResidual, falseResidual, naResidual)
    }

    val lt5 = lt(k1, I32(5))
    val gt10 = gt(k1, I32(10))

    check(
      or(lt5, gt10),
      FastSeq(
        Interval(Row(), Row(5), true, false),
        Interval(Row(10), Row(null), false, false),
      ),
      FastSeq(Interval(Row(5), Row(10), true, true)),
      FastSeq(Interval(Row(null), Row(), true, true)),
    )

    check(
      or(lt5, unknownBool),
      // could be true anywhere, since unknownBool might be true
      FastSeq(Interval(Row(), Row(), true, true)),
      // can only be false if lt5 is false
      FastSeq(Interval(Row(5), Row(null), true, false)),
      // can only be missing if lt5 is missing (and unknown is false or missing),
      // or if lt5 is false (and unknown is missing)
      FastSeq(Interval(Row(5), Row(), true, true)),
      // we've filtered to the rows where lt5 is false
      falseResidual = not(or(False(), unknownBool)),
      // we've filtered to where lt5 is false or missing, so can't simplify
      naResidual = IsNA(or(lt5, unknownBool)),
    )

    check(
      and(not(or(lt5, unknownBool)), not(or(gt10, unknownBool))),
      FastSeq(Interval(Row(5), Row(10), true, true)),
      FastSeq(Interval(Row(), Row(), true, true)),
      FastSeq(Interval(Row(5), Row(10), true, true), Interval(Row(null), Row(), true, true)),
      trueResidual = and(
        not(or(False(), unknownBool)),
        not(or(False(), unknownBool)),
      ),
      naResidual = IsNA(and(
        not(or(lt5, unknownBool)),
        not(or(gt10, unknownBool)),
      )),
    )
  }

  @Test def testConjunction() {
    def check(
      node: IR,
      trueIntervals: IndexedSeq[Interval],
      falseIntervals: IndexedSeq[Interval],
      naIntervals: IndexedSeq[Interval],
      trueResidual: IR = True(),
      falseResidual: IR = True(),
      naResidual: IR = True(),
    ) {
      val testRows = FastSeq(
        Row(0, 0, true),
        Row(0, 0, false),
        Row(0, 5, true),
        Row(0, 5, false),
        Row(0, 7, true),
        Row(0, 7, false),
        Row(0, 10, true),
        Row(0, 10, false),
        Row(0, 15, true),
        Row(0, 15, false),
        Row(0, null, true),
        Row(0, null, false),
      )
      checkAll(node, ref1, k1Full, testRows, trueIntervals, falseIntervals, naIntervals,
        trueResidual, falseResidual, naResidual)
    }

    val gt5 = gt(k1, I32(5))
    val lt10 = lt(k1, I32(10))

    check(
      and(gt5, lt10),
      FastSeq(Interval(Row(5), Row(10), false, false)),
      FastSeq(
        Interval(Row(), Row(5), true, true),
        Interval(Row(10), Row(null), true, false),
      ),
      FastSeq(Interval(Row(null), Row(), true, true)),
    )

    check(
      and(gt5, unknownBool),
      // can only be true if gt5 is true
      FastSeq(Interval(Row(5), Row(null), false, false)),
      // could be false anywhere, since unknownBool might be false
      FastSeq(Interval(Row(), Row(), true, true)),
      // can only be missing if gt5 is missing (and unknown is true or missing),
      // or if gt5 is true (and unknown is missing)
      FastSeq(Interval(Row(5), Row(), false, true)),
      // we've filtered to the rows where gt5 is true
      trueResidual = and(True(), unknownBool),
      // we've filtered to where gt5 is false or missing, so can't simplify
      naResidual = IsNA(and(gt5, unknownBool)),
    )
  }

  @Test def testCoalesce(): Unit = {
    def check(
      node: IR,
      trueIntervals: IndexedSeq[Interval],
      falseIntervals: IndexedSeq[Interval],
      naIntervals: IndexedSeq[Interval],
      trueResidual: IR = True(),
      falseResidual: IR = True(),
      naResidual: IR = True(),
    ) {
      val testRows = FastSeq(
        Row(0, 0, true),
        Row(0, 5, true),
        Row(0, 7, true),
        Row(0, 10, true),
        Row(0, 15, true),
        Row(0, null, true),
      )
      checkAll(node, ref1, k1Full, testRows, trueIntervals, falseIntervals, naIntervals,
        trueResidual, falseResidual, naResidual)
    }

    val gt5 = gt(k1, I32(5))
    val lt10 = lt(k1, I32(10))

    check(
      Coalesce(FastSeq(gt5, lt10, False())),
      FastSeq(Interval(Row(5), Row(null), false, false)),
      FastSeq(
        Interval(Row(), Row(5), true, true),
        Interval(Row(null), Row(), true, true),
      ),
      FastSeq(),
    )
  }

  @Test def testIf(): Unit = {
    def check(
      node: IR,
      trueIntervals: IndexedSeq[Interval],
      falseIntervals: IndexedSeq[Interval],
      naIntervals: IndexedSeq[Interval],
      trueResidual: IR = True(),
      falseResidual: IR = True(),
      naResidual: IR = True(),
    ) {
      val testRows = FastSeq(
        Row(0, 0, true),
        Row(0, 5, true),
        Row(0, 7, true),
        Row(0, 10, true),
        Row(0, 15, true),
        Row(0, null, true),
      )
      checkAll(node, ref1, k1Full, testRows, trueIntervals, falseIntervals, naIntervals,
        trueResidual, falseResidual, naResidual)
    }

    check(
      If(gt(k1, I32(0)), lt(k1, I32(5)), gt(k1, I32(-5))),
      FastSeq(Interval(Row(-5), Row(5), false, false)),
      FastSeq(
        Interval(Row(), Row(-5), true, true),
        Interval(Row(5), Row(null), true, false),
      ),
      FastSeq(Interval(Row(null), Row(), true, true)),
    )
  }

  @Test def testSwitch(): Unit = {
    def check(
      node: IR,
      trueIntervals: IndexedSeq[Interval],
      falseIntervals: IndexedSeq[Interval],
      naIntervals: IndexedSeq[Interval],
      trueResidual: IR = True(),
      falseResidual: IR = True(),
      naResidual: IR = True(),
    ) {
      val testRows = FastSeq(
        Row(0, 0, true),
        Row(0, 5, true),
        Row(0, -5, true),
        Row(0, null, true),
        Row(1, 0, true),
        Row(1, 5, true),
        Row(1, -5, true),
        Row(1, null, true),
        Row(null, null, true),
      )
      checkAll(node, ref1, k1Full, testRows, trueIntervals, falseIntervals, naIntervals,
        trueResidual, falseResidual, naResidual)
    }

    check(
      Switch(I32(0), gt(k1, I32(-5)), FastSeq(lt(k1, I32(5)))),
      FastSeq(Interval(Row(), Row(5), true, false)),
      FastSeq(Interval(Row(5), Row(null), true, false)),
      FastSeq(Interval(Row(null), Row(), true, true)),
    )

    check(
      Switch(I32(-1), gt(k1, I32(-5)), FastSeq(lt(k1, I32(5)))),
      FastSeq(Interval(Row(-5), Row(null), false, false)),
      FastSeq(Interval(Row(), Row(-5), true, true)),
      FastSeq(Interval(Row(null), Row(), true, true)),
    )

    val filter = Switch(GetField(ref1, "w"), gt(k1, I32(-5)), FastSeq(lt(k1, I32(5))))
    check(
      filter,
      FastSeq(Interval(Row(), Row(null), true, false)),
      FastSeq(Interval(Row(), Row(-5), true, true), Interval(Row(5), Row(null), true, false)),
      FastSeq(Interval(Row(), Row(), true, true)),
      trueResidual = filter,
      falseResidual = ApplyUnaryPrimOp(Bang, filter),
      naResidual = IsNA(filter),
    )
  }

  @Test def testRelationalChildren(): Unit = {
    val testRows = FastSeq(
      Row(0, 0, true),
      Row(0, 10, true),
      Row(0, 20, true),
      Row(0, null, true),
    )

    val count = TableAggregate(
      TableRange(10, 1),
      ApplyAggOp(FastSeq(), FastSeq(), AggSignature(Count(), FastSeq(), FastSeq())),
    )
    print(count.typ)
    val filter = gt(count, Cast(k1, TInt64))
    check(filter, ref1, k1Full, testRows, filter, FastSeq(Interval(Row(), Row(), true, true)))
  }

  @Test def testIntegration() {
    hc // force initialization
    val tab1 = TableRange(10, 5)

    def k = GetField(Ref("row", tab1.typ.rowType), "idx")

    val tf = TableFilter(
      tab1,
      Coalesce(FastSeq(
        invoke(
          "land",
          TBoolean,
          ApplyComparisonOp(GT(TInt32), k, I32(3)),
          ApplyComparisonOp(LTEQ(TInt32), k, I32(9)),
        ),
        False(),
      )),
    )

    assert(ExtractIntervalFilters(ctx, tf).asInstanceOf[TableFilter].child.isInstanceOf[
      TableFilterIntervals
    ])
    assertEvalsTo(TableCount(tf), 6L)(ExecStrategy.interpretOnly)
  }
}
