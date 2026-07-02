package is.hail.expr.ir

import is.hail.ExecStrategy
import is.hail.TestUtils._
import is.hail.annotations.RowSeq
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir.defs._
import is.hail.rvd.RVDPartitioner
import is.hail.types.virtual._
import is.hail.utils.{Interval, IntervalEndpoint}
import is.hail.variant.{Locus, ReferenceGenome}

import org.apache.spark.sql.Row
import org.junit.jupiter.api.Test

class ExtractIntervalFiltersSuite {

  val ref1 = Ref(freshName(), TStruct("w" -> TInt32, "x" -> TInt32, "y" -> TBoolean))
  val unknownBool = GetField(ref1, "y")
  val k1 = GetField(ref1, "x")
  val k1Full = SelectFields(ref1, FastSeq("x"))
  val ref1Key = FastSeq("x")

  val structRef = Ref(freshName(), TStruct("x" -> TInt32, "y" -> TInt32, "z" -> TInt32))
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

  def wrappedIntervalEndpoint(x: Any, sign: Int) = IntervalEndpoint(RowSeq(x), sign)

  def grch38(implicit ctx: ExecuteContext): ReferenceGenome = ctx.references(ReferenceGenome.GRCh38)

  def lt(l: IR, r: IR): IR = ApplyComparisonOp(LT, l, r)
  def gt(l: IR, r: IR): IR = ApplyComparisonOp(GT, l, r)
  def lteq(l: IR, r: IR): IR = ApplyComparisonOp(LTEQ, l, r)
  def gteq(l: IR, r: IR): IR = ApplyComparisonOp(GTEQ, l, r)
  def eq(l: IR, r: IR): IR = ApplyComparisonOp(EQ, l, r)
  def neq(l: IR, r: IR): IR = ApplyComparisonOp(NEQ, l, r)
  def eqna(l: IR, r: IR): IR = ApplyComparisonOp(EQWithNA, l, r)
  def neqna(l: IR, r: IR): IR = ApplyComparisonOp(NEQWithNA, l, r)
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
  )(implicit ctx: ExecuteContext
  ): Unit = {
    val result = ExtractIntervalFilters.extractPartitionFilters(
      ctx,
      filter,
      rowRef,
      key.typ.asInstanceOf[TStruct].fieldNames,
    )
    if (result.isEmpty) {
      assert(trueIntervals == FastSeq(Interval(RowSeq(), RowSeq(), true, true)))
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
    val accRef = Ref(freshName(), TBoolean)

    val testIR = StreamFold(
      ToStream(Literal(TArray(rowRef.typ), probes)),
      True(),
      accRef.name,
      rowRef.name,
      ApplyComparisonOp(
        EQ,
        filterIsTrue,
        invoke("land", TBoolean, keyInIntervals, residualIsTrue),
      ),
    )

    assertEvalsTo(testIR, true)(ctx, ExecStrategy.compileOnly)
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
  )(implicit ctx: ExecuteContext
  ): Unit = {
    check(filter, rowRef, key, probes, trueResidual, trueIntervals)
    check(ApplyUnaryPrimOp(Bang, filter), rowRef, key, probes, falseResidual, falseIntervals)
    check(IsNA(filter), rowRef, key, probes, naResidual, naIntervals)
  }

  @Test def testIsNA(implicit ctx: ExecuteContext): Unit = {
    val testRows = FastSeq(
      RowSeq(0, 0, true),
      RowSeq(0, null, true),
    )
    checkAll(
      IsNA(k1),
      ref1,
      k1Full,
      testRows,
      FastSeq(Interval(RowSeq(null), RowSeq(), true, true)),
      FastSeq(Interval(RowSeq(), RowSeq(null), true, false)),
      FastSeq(),
    )
  }

  @Test def testKeyComparison(implicit ctx: ExecuteContext): Unit = {
    def check(
      op: ComparisonOp[Boolean],
      point: IR,
      trueIntervals: IndexedSeq[Interval],
      falseIntervals: IndexedSeq[Interval],
      naIntervals: IndexedSeq[Interval],
    ): Unit = {
      val testRows = FastSeq(
        RowSeq(0, -1, true),
        RowSeq(0, 0, true),
        RowSeq(0, 1, true),
        RowSeq(0, null, true),
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
      LT,
      I32(0),
      FastSeq(Interval(RowSeq(), RowSeq(0), true, false)),
      FastSeq(Interval(RowSeq(0), RowSeq(null), true, false)),
      FastSeq(Interval(RowSeq(null), RowSeq(), true, true)),
    )
    check(
      GT,
      I32(0),
      FastSeq(Interval(RowSeq(0), RowSeq(null), false, false)),
      FastSeq(Interval(RowSeq(), RowSeq(0), true, true)),
      FastSeq(Interval(RowSeq(null), RowSeq(), true, true)),
    )
    check(
      EQ,
      I32(0),
      FastSeq(Interval(RowSeq(0), RowSeq(0), true, true)),
      FastSeq(
        Interval(RowSeq(), RowSeq(0), true, false),
        Interval(RowSeq(0), RowSeq(null), false, false),
      ),
      FastSeq(Interval(RowSeq(null), RowSeq(), true, true)),
    )

    // These are never true (always missing), extracts the empty set of intervals
    check(LT, NA(TInt32), FastSeq(), FastSeq(), FastSeq(Interval(RowSeq(), RowSeq(), true, true)))
    check(GT, NA(TInt32), FastSeq(), FastSeq(), FastSeq(Interval(RowSeq(), RowSeq(), true, true)))
    check(EQ, NA(TInt32), FastSeq(), FastSeq(), FastSeq(Interval(RowSeq(), RowSeq(), true, true)))

    check(
      EQWithNA,
      I32(0),
      FastSeq(Interval(RowSeq(0), RowSeq(0), true, true)),
      FastSeq(
        Interval(RowSeq(), RowSeq(0), true, false),
        Interval(RowSeq(0), RowSeq(), false, true),
      ),
      FastSeq(),
    )
    check(
      EQWithNA,
      NA(TInt32),
      FastSeq(Interval(RowSeq(null), RowSeq(), true, true)),
      FastSeq(Interval(RowSeq(), RowSeq(null), true, false)),
      FastSeq(),
    )

    assert(ExtractIntervalFilters.extractPartitionFilters(
      ctx,
      ApplyComparisonOp(Compare, I32(0), k1),
      ref1,
      ref1Key,
    ).isEmpty)
  }

  @Test def testLiteralContains(implicit ctx: ExecuteContext): Unit = {
    def check(
      node: IR,
      trueIntervals: IndexedSeq[Interval],
      falseIntervals: IndexedSeq[Interval],
      naIntervals: IndexedSeq[Interval],
    ): Unit = {
      val testRows = FastSeq(
        RowSeq(0, 1, true),
        RowSeq(0, 5, true),
        RowSeq(0, 10, true),
        RowSeq(0, null, true),
      )
      checkAll(node, ref1, k1Full, testRows, trueIntervals, falseIntervals, naIntervals)
    }

    Array(
      Literal(TSet(TInt32), Set(null, 10, 1)),
      Literal(TArray(TInt32), FastSeq(10, 1, null)),
      Literal(TDict(TInt32, TString), Map(1 -> "foo", (null, "bar"), 10 -> "baz")),
    ).foreach { lit =>
      check(
        invoke("contains", TBoolean, lit, k1),
        FastSeq(
          Interval(RowSeq(1), RowSeq(1), true, true),
          Interval(RowSeq(10), RowSeq(10), true, true),
          Interval(RowSeq(null), RowSeq(), true, true),
        ),
        FastSeq(
          Interval(RowSeq(), RowSeq(1), true, false),
          Interval(RowSeq(1), RowSeq(10), false, false),
          Interval(RowSeq(10), RowSeq(null), false, false),
        ),
        FastSeq(),
      )
    }

    Array(
      Literal(TSet(TInt32), Set(10, 1)),
      Literal(TArray(TInt32), FastSeq(10, 1)),
      Literal(TDict(TInt32, TString), Map(1 -> "foo", 10 -> "baz")),
    ).foreach { lit =>
      check(
        invoke("contains", TBoolean, lit, k1),
        FastSeq(
          Interval(RowSeq(1), RowSeq(1), true, true),
          Interval(RowSeq(10), RowSeq(10), true, true),
        ),
        FastSeq(
          Interval(RowSeq(), RowSeq(1), true, false),
          Interval(RowSeq(1), RowSeq(10), false, false),
          Interval(RowSeq(10), RowSeq(), false, true),
        ),
        FastSeq(),
      )
    }
  }

  @Test def testLiteralContainsStruct(implicit ctx: ExecuteContext): Unit = {
    def check(
      node: IR,
      trueIntervals: IndexedSeq[Interval],
      falseIntervals: IndexedSeq[Interval],
      naIntervals: IndexedSeq[Interval],
    ): Unit = {
      val testRows = FastSeq(
        RowSeq(0, 1, 2),
        RowSeq(0, 3, 4),
        RowSeq(0, 3, null),
        RowSeq(0, 5, null),
        RowSeq(0, 5, 5),
        RowSeq(0, null, 1),
        RowSeq(0, null, null),
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

    Array(
      Literal(TSet(structT1), Set(RowSeq(1, 2), RowSeq(3, 4), RowSeq(3, null))),
      Literal(TArray(structT1), FastSeq(RowSeq(3, 4), RowSeq(1, 2), RowSeq(3, null))),
      Literal(
        TDict(structT1, TString),
        Map(RowSeq(1, 2) -> "foo", RowSeq(3, 4) -> "bar", RowSeq(3, null) -> "baz"),
      ),
    ).foreach { lit =>
      fullKeyRefs.foreach { k =>
        check(
          invoke("contains", TBoolean, lit, k),
          IndexedSeq(
            Interval(RowSeq(1, 2), RowSeq(1, 2), true, true),
            Interval(RowSeq(3, 4), RowSeq(3, 4), true, true),
            Interval(RowSeq(3, null), RowSeq(3, null), true, true),
          ),
          IndexedSeq(
            Interval(RowSeq(), RowSeq(1, 2), true, false),
            Interval(RowSeq(1, 2), RowSeq(3, 4), false, false),
            Interval(RowSeq(3, 4), RowSeq(3, null), false, false),
            Interval(RowSeq(3, null), RowSeq(), false, true),
          ),
          IndexedSeq(),
        )
      }
    }

    Array(
      Literal(TSet(structT2), Set(RowSeq(1), RowSeq(3), RowSeq(null))),
      Literal(TArray(structT2), FastSeq(RowSeq(3), RowSeq(null), RowSeq(1))),
      Literal(
        TDict(structT2, TString),
        Map(RowSeq(1) -> "foo", RowSeq(null) -> "baz", RowSeq(3) -> "bar"),
      ),
    ).foreach { lit =>
      prefixKeyRefs.foreach { k =>
        check(
          invoke("contains", TBoolean, lit, k),
          IndexedSeq(
            Interval(RowSeq(1), RowSeq(1), true, true),
            Interval(RowSeq(3), RowSeq(3), true, true),
            Interval(RowSeq(null), RowSeq(), true, true),
          ),
          IndexedSeq(
            Interval(RowSeq(), RowSeq(1), true, false),
            Interval(RowSeq(1), RowSeq(3), false, false),
            Interval(RowSeq(3), RowSeq(null), false, false),
          ),
          IndexedSeq(),
        )
      }
    }
  }

  @Test def testIntervalContains(implicit ctx: ExecuteContext): Unit = {
    val interval = Interval(1, 5, false, true)
    val testRows = FastSeq(
      RowSeq(0, 0, true),
      RowSeq(0, 1, true),
      RowSeq(0, 5, true),
      RowSeq(0, 10, true),
      RowSeq(0, null, true),
    )

    val ir = invoke("contains", TBoolean, Literal(TInterval(TInt32), interval), k1)
    checkAll(
      ir,
      ref1,
      k1Full,
      testRows,
      FastSeq(Interval(RowSeq(1), RowSeq(5), false, true)),
      FastSeq(
        Interval(RowSeq(), RowSeq(1), true, true),
        Interval(RowSeq(5), RowSeq(null), false, false),
      ),
      FastSeq(Interval(RowSeq(null), RowSeq(), true, true)),
    )
  }

  @Test def testIntervalContainsStruct(implicit ctx: ExecuteContext): Unit = {
    val fullInterval = Interval(RowSeq(1, 1), RowSeq(2, 2), false, true)
    val prefixInterval = Interval(RowSeq(1), RowSeq(2), false, true)

    def check(
      node: IR,
      trueIntervals: IndexedSeq[Interval],
      falseIntervals: IndexedSeq[Interval],
      naIntervals: IndexedSeq[Interval],
    ): Unit = {
      val testRows = FastSeq(
        RowSeq(0, null, 0),
        RowSeq(0, 0, 0),
        RowSeq(0, 0, null),
        RowSeq(0, 1, 0),
        RowSeq(0, 1, 1),
        RowSeq(0, 1, null),
        RowSeq(0, 2, 1),
        RowSeq(0, 2, 2),
        RowSeq(0, 2, null),
        RowSeq(0, 3, 0),
        RowSeq(0, null, null),
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

    fullKeyRefs.foreach { k =>
      check(
        invoke("contains", TBoolean, Literal(TInterval(structT1), fullInterval), k),
        FastSeq(fullInterval),
        FastSeq(
          Interval(RowSeq(), RowSeq(1, 1), true, true),
          Interval(RowSeq(2, 2), RowSeq(), false, true),
        ),
        FastSeq(),
      )
    }

    prefixKeyRefs.foreach { k =>
      check(
        invoke("contains", TBoolean, Literal(TInterval(structT2), prefixInterval), k),
        FastSeq(prefixInterval),
        FastSeq(
          Interval(RowSeq(), RowSeq(1), true, true),
          Interval(RowSeq(2), RowSeq(), false, true),
        ),
        FastSeq(),
      )
    }
  }

  @Test def testLocusContigComparison(implicit ctx: ExecuteContext): Unit = {
    val ref = Ref(freshName(), TStruct("x" -> TLocus(ReferenceGenome.GRCh38)))
    val k = GetField(ref, "x")

    val ir1 = eq(Str("chr2"), invoke("contig", TString, k))
    val ir2 = eq(invoke("contig", TString, k), Str("chr2"))

    val testRows = FastSeq(
      RowSeq(Locus("chr1", 5)),
      RowSeq(Locus("chr2", 1)),
      RowSeq(Locus("chr2", 1000)),
      RowSeq(Locus("chr3", 5)),
      RowSeq(null),
    )

    val trueIntervals = FastSeq(
      Interval(
        RowSeq(Locus("chr2", 1)),
        RowSeq(Locus("chr2", grch38.contigLength("chr2"))),
        true,
        false,
      )
    )
    val falseIntervals = FastSeq(
      Interval(RowSeq(), RowSeq(Locus("chr2", 1)), true, false),
      Interval(RowSeq(Locus("chr2", grch38.contigLength("chr2"))), RowSeq(null), true, false),
    )
    val naIntervals = FastSeq(Interval(RowSeq(null), RowSeq(), true, true))

    checkAll(ir1, ref, ref, testRows, trueIntervals, falseIntervals, naIntervals)
    checkAll(ir2, ref, ref, testRows, trueIntervals, falseIntervals, naIntervals)

    val ir3 = neq(Str("chr2"), invoke("contig", TString, k))
    checkAll(ir3, ref, ref, testRows, falseIntervals, trueIntervals, naIntervals)
    checkAll(not(ir1), ref, ref, testRows, falseIntervals, trueIntervals, naIntervals)
  }

  @Test def testLocusPositionComparison(implicit ctx: ExecuteContext): Unit = {
    val ref = Ref(freshName(), TStruct("x" -> TLocus(ReferenceGenome.GRCh38)))
    val k = GetField(ref, "x")
    val pos = invoke("position", TInt32, k)

    def check(
      op: ComparisonOp[Boolean],
      point: Int,
      truePosIntervals: IndexedSeq[Interval],
      falsePosIntervals: IndexedSeq[Interval],
      naPosIntervals: IndexedSeq[Interval],
    ): Unit = {
      val trueIntervals =
        ExtractIntervalFilters.liftPosIntervalsToLocus(truePosIntervals, grch38, ctx)
      val falseIntervals =
        ExtractIntervalFilters.liftPosIntervalsToLocus(falsePosIntervals, grch38, ctx)
      val naIntervals = ExtractIntervalFilters.liftPosIntervalsToLocus(naPosIntervals, grch38, ctx)

      val testRows = FastSeq(
        RowSeq(Locus("chr1", 1)),
        RowSeq(Locus("chr1", 5)),
        RowSeq(Locus("chr1", 100)),
        RowSeq(Locus("chr1", 105)),
        RowSeq(Locus("chr2", 1)),
        RowSeq(Locus("chr2", 5)),
        RowSeq(Locus("chr2", 100)),
        RowSeq(Locus("chr3", 105)),
        RowSeq(null),
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
      GT,
      100,
      FastSeq(Interval(RowSeq(100), RowSeq(null), false, false)),
      FastSeq(Interval(RowSeq(), RowSeq(100), true, true)),
      FastSeq(Interval(RowSeq(null), RowSeq(), true, true)),
    )
    check(
      GT,
      -1000,
      FastSeq(Interval(RowSeq(1), RowSeq(null), true, false)),
      FastSeq(),
      FastSeq(Interval(RowSeq(null), RowSeq(), true, true)),
    )

    check(
      LT,
      100,
      FastSeq(Interval(RowSeq(), RowSeq(100), true, false)),
      FastSeq(Interval(RowSeq(100), RowSeq(null), true, false)),
      FastSeq(Interval(RowSeq(null), RowSeq(), true, true)),
    )
    check(
      LT,
      -1000,
      FastSeq(),
      FastSeq(Interval(RowSeq(), RowSeq(null), true, false)),
      FastSeq(Interval(RowSeq(null), RowSeq(), true, true)),
    )

    check(
      EQ,
      100,
      FastSeq(Interval(RowSeq(100), RowSeq(100), true, true)),
      FastSeq(
        Interval(RowSeq(), RowSeq(100), true, false),
        Interval(RowSeq(100), RowSeq(null), false, false),
      ),
      FastSeq(Interval(RowSeq(null), RowSeq(), true, true)),
    )
    check(
      EQ,
      -1000,
      FastSeq(),
      FastSeq(Interval(RowSeq(), RowSeq(null), true, false)),
      FastSeq(Interval(RowSeq(null), RowSeq(), true, true)),
    )

    check(
      EQWithNA,
      100,
      FastSeq(Interval(RowSeq(100), RowSeq(100), true, true)),
      FastSeq(
        Interval(RowSeq(), RowSeq(100), true, false),
        Interval(RowSeq(100), RowSeq(), false, true),
      ),
      FastSeq(),
    )
    check(
      EQWithNA,
      -1000,
      FastSeq(),
      FastSeq(Interval(RowSeq(), RowSeq(), true, true)),
      FastSeq(),
    )

    assert(ExtractIntervalFilters.extractPartitionFilters(
      ctx,
      ApplyComparisonOp(Compare, I32(0), pos),
      ref,
      ref1Key,
    ).isEmpty)
  }

  @Test def testLocusContigContains(implicit ctx: ExecuteContext): Unit = {
    val ref = Ref(freshName(), TStruct("x" -> TLocus(ReferenceGenome.GRCh38)))
    val k = GetField(ref, "x")
    val contig = invoke("contig", TString, k)

    def check(
      node: IR,
      trueIntervals: IndexedSeq[Interval],
      falseIntervals: IndexedSeq[Interval],
      naIntervals: IndexedSeq[Interval],
    ): Unit = {
      val testRows = FastSeq(
        RowSeq(Locus("chr1", 5)),
        RowSeq(Locus("chr2", 1)),
        RowSeq(Locus("chr10", 5)),
        RowSeq(null),
      )
      checkAll(node, ref, ref, testRows, trueIntervals, falseIntervals, naIntervals)
    }

    Array(
      Literal(TSet(TString), Set("chr10", "chr1", null, "foo")),
      Literal(TArray(TString), FastSeq("foo", "chr10", null, "chr1")),
      Literal(
        TDict(TString, TString),
        Map("chr1" -> "foo", "chr10" -> "bar", "foo" -> "baz", (null, "quux")),
      ),
    ).foreach { lit =>
      check(
        invoke("contains", TBoolean, lit, contig),
        FastSeq(
          Interval(
            RowSeq(Locus("chr1", 1)),
            RowSeq(Locus("chr1", grch38.contigLength("chr1"))),
            true,
            false,
          ),
          Interval(
            RowSeq(Locus("chr10", 1)),
            RowSeq(Locus("chr10", grch38.contigLength("chr10"))),
            true,
            false,
          ),
          Interval(RowSeq(null), RowSeq(), true, true),
        ),
        FastSeq(
          Interval(
            RowSeq(),
            RowSeq(Locus("chr1", 1)),
            true,
            false,
          ),
          Interval(
            RowSeq(Locus("chr1", grch38.contigLength("chr1"))),
            RowSeq(Locus("chr10", 1)),
            true,
            false,
          ),
          Interval(
            RowSeq(Locus("chr10", grch38.contigLength("chr10"))),
            RowSeq(null),
            true,
            false,
          ),
        ),
        FastSeq(),
      )
    }

    Array(
      Literal(TSet(TString), Set("chr10", "chr1", "foo")),
      Literal(TArray(TString), FastSeq("foo", "chr10", "chr1")),
      Literal(TDict(TString, TString), Map("chr1" -> "foo", "chr10" -> "bar", "foo" -> "baz")),
    ).foreach { lit =>
      check(
        invoke("contains", TBoolean, lit, contig),
        FastSeq(
          Interval(
            RowSeq(Locus("chr1", 1)),
            RowSeq(Locus("chr1", grch38.contigLength("chr1"))),
            true,
            false,
          ),
          Interval(
            RowSeq(Locus("chr10", 1)),
            RowSeq(Locus("chr10", grch38.contigLength("chr10"))),
            true,
            false,
          ),
        ),
        FastSeq(
          Interval(
            RowSeq(),
            RowSeq(Locus("chr1", 1)),
            true,
            false,
          ),
          Interval(
            RowSeq(Locus("chr1", grch38.contigLength("chr1"))),
            RowSeq(Locus("chr10", 1)),
            true,
            false,
          ),
          Interval(
            RowSeq(Locus("chr10", grch38.contigLength("chr10"))),
            RowSeq(),
            true,
            true,
          ),
        ),
        FastSeq(),
      )
    }
  }

  @Test def testIntervalListFold(implicit ctx: ExecuteContext): Unit = {
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
    ): Unit = {
      val testRows = FastSeq(
        RowSeq(0, -15, true),
        RowSeq(0, -10, true),
        RowSeq(0, -5, true),
        RowSeq(0, 0, true),
        RowSeq(0, 5, true),
        RowSeq(0, 10, true),
        RowSeq(0, 15, true),
        RowSeq(0, 20, true),
        RowSeq(0, 22, true),
        RowSeq(0, 25, true),
        RowSeq(0, 30, true),
        RowSeq(0, null, true),
      )
      checkAll(node, ref1, k1Full, testRows, trueIntervals, falseIntervals, naIntervals)
    }

    def containsKey(intervals: IndexedSeq[Interval]) = foldIR(
      ToStream(Literal(TArray(TInterval(TInt32)), intervals)),
      False(),
    )((acc, elt) => invoke("lor", TBoolean, acc, invoke("contains", TBoolean, elt, k1)))

    check(
      containsKey(inIntervals),
      FastSeq(
        Interval(RowSeq(-10), RowSeq(10), true, false),
        Interval(RowSeq(20), RowSeq(25), true, false),
      ),
      FastSeq(
        Interval(RowSeq(), RowSeq(-10), true, false),
        Interval(RowSeq(10), RowSeq(20), true, false),
        Interval(RowSeq(25), RowSeq(null), true, false),
      ),
      FastSeq(Interval(RowSeq(null), RowSeq(), true, true)),
    )

    // Whenever the previous would be false, this is instead missing, because of the null
    // In particular, it is never false, so notIR2 filters everything
    check(
      containsKey(inIntervalsWithNull),
      FastSeq(
        Interval(RowSeq(-10), RowSeq(10), true, false),
        Interval(RowSeq(20), RowSeq(25), true, false),
      ),
      FastSeq(),
      FastSeq(
        Interval(RowSeq(), RowSeq(-10), true, false),
        Interval(RowSeq(10), RowSeq(20), true, false),
        Interval(RowSeq(25), RowSeq(), true, true),
      ),
    )
  }

  @Test def testDisjunction(implicit ctx: ExecuteContext): Unit = {
    def check(
      node: IR,
      trueIntervals: IndexedSeq[Interval],
      falseIntervals: IndexedSeq[Interval],
      naIntervals: IndexedSeq[Interval],
      trueResidual: IR = True(),
      falseResidual: IR = True(),
      naResidual: IR = True(),
    ): Unit = {
      val testRows = FastSeq(
        RowSeq(0, 0, true),
        RowSeq(0, 0, false),
        RowSeq(0, 5, true),
        RowSeq(0, 5, false),
        RowSeq(0, 7, true),
        RowSeq(0, 7, false),
        RowSeq(0, 10, true),
        RowSeq(0, 10, false),
        RowSeq(0, 15, true),
        RowSeq(0, 15, false),
        RowSeq(0, null, true),
        RowSeq(0, null, false),
      )
      checkAll(node, ref1, k1Full, testRows, trueIntervals, falseIntervals, naIntervals,
        trueResidual, falseResidual, naResidual)
    }

    val lt5 = lt(k1, I32(5))
    val gt10 = gt(k1, I32(10))

    check(
      or(lt5, gt10),
      FastSeq(
        Interval(RowSeq(), RowSeq(5), true, false),
        Interval(RowSeq(10), RowSeq(null), false, false),
      ),
      FastSeq(Interval(RowSeq(5), RowSeq(10), true, true)),
      FastSeq(Interval(RowSeq(null), RowSeq(), true, true)),
    )

    check(
      or(lt5, unknownBool),
      // could be true anywhere, since unknownBool might be true
      FastSeq(Interval(RowSeq(), RowSeq(), true, true)),
      // can only be false if lt5 is false
      FastSeq(Interval(RowSeq(5), RowSeq(null), true, false)),
      // can only be missing if lt5 is missing (and unknown is false or missing),
      // or if lt5 is false (and unknown is missing)
      FastSeq(Interval(RowSeq(5), RowSeq(), true, true)),
      // we've filtered to the rows where lt5 is false
      falseResidual = not(or(False(), unknownBool)),
      // we've filtered to where lt5 is false or missing, so can't simplify
      naResidual = IsNA(or(lt5, unknownBool)),
    )

    check(
      and(not(or(lt5, unknownBool)), not(or(gt10, unknownBool))),
      FastSeq(Interval(RowSeq(5), RowSeq(10), true, true)),
      FastSeq(Interval(RowSeq(), RowSeq(), true, true)),
      FastSeq(
        Interval(RowSeq(5), RowSeq(10), true, true),
        Interval(RowSeq(null), RowSeq(), true, true),
      ),
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

  @Test def testConjunction(implicit ctx: ExecuteContext): Unit = {
    def check(
      node: IR,
      trueIntervals: IndexedSeq[Interval],
      falseIntervals: IndexedSeq[Interval],
      naIntervals: IndexedSeq[Interval],
      trueResidual: IR = True(),
      falseResidual: IR = True(),
      naResidual: IR = True(),
    ): Unit = {
      val testRows = FastSeq(
        RowSeq(0, 0, true),
        RowSeq(0, 0, false),
        RowSeq(0, 5, true),
        RowSeq(0, 5, false),
        RowSeq(0, 7, true),
        RowSeq(0, 7, false),
        RowSeq(0, 10, true),
        RowSeq(0, 10, false),
        RowSeq(0, 15, true),
        RowSeq(0, 15, false),
        RowSeq(0, null, true),
        RowSeq(0, null, false),
      )
      checkAll(node, ref1, k1Full, testRows, trueIntervals, falseIntervals, naIntervals,
        trueResidual, falseResidual, naResidual)
    }

    val gt5 = gt(k1, I32(5))
    val lt10 = lt(k1, I32(10))

    check(
      and(gt5, lt10),
      FastSeq(Interval(RowSeq(5), RowSeq(10), false, false)),
      FastSeq(
        Interval(RowSeq(), RowSeq(5), true, true),
        Interval(RowSeq(10), RowSeq(null), true, false),
      ),
      FastSeq(Interval(RowSeq(null), RowSeq(), true, true)),
    )

    check(
      and(gt5, unknownBool),
      // can only be true if gt5 is true
      FastSeq(Interval(RowSeq(5), RowSeq(null), false, false)),
      // could be false anywhere, since unknownBool might be false
      FastSeq(Interval(RowSeq(), RowSeq(), true, true)),
      // can only be missing if gt5 is missing (and unknown is true or missing),
      // or if gt5 is true (and unknown is missing)
      FastSeq(Interval(RowSeq(5), RowSeq(), false, true)),
      // we've filtered to the rows where gt5 is true
      trueResidual = and(True(), unknownBool),
      // we've filtered to where gt5 is false or missing, so can't simplify
      naResidual = IsNA(and(gt5, unknownBool)),
    )
  }

  @Test def testCoalesce(implicit ctx: ExecuteContext): Unit = {
    def check(
      node: IR,
      trueIntervals: IndexedSeq[Interval],
      falseIntervals: IndexedSeq[Interval],
      naIntervals: IndexedSeq[Interval],
      trueResidual: IR = True(),
      falseResidual: IR = True(),
      naResidual: IR = True(),
    ): Unit = {
      val testRows = FastSeq(
        RowSeq(0, 0, true),
        RowSeq(0, 5, true),
        RowSeq(0, 7, true),
        RowSeq(0, 10, true),
        RowSeq(0, 15, true),
        RowSeq(0, null, true),
      )
      checkAll(node, ref1, k1Full, testRows, trueIntervals, falseIntervals, naIntervals,
        trueResidual, falseResidual, naResidual)
    }

    val gt5 = gt(k1, I32(5))
    val lt10 = lt(k1, I32(10))

    check(
      Coalesce(FastSeq(gt5, lt10, False())),
      FastSeq(Interval(RowSeq(5), RowSeq(null), false, false)),
      FastSeq(
        Interval(RowSeq(), RowSeq(5), true, true),
        Interval(RowSeq(null), RowSeq(), true, true),
      ),
      FastSeq(),
    )
  }

  @Test def testIf(implicit ctx: ExecuteContext): Unit = {
    def check(
      node: IR,
      trueIntervals: IndexedSeq[Interval],
      falseIntervals: IndexedSeq[Interval],
      naIntervals: IndexedSeq[Interval],
      trueResidual: IR = True(),
      falseResidual: IR = True(),
      naResidual: IR = True(),
    ): Unit = {
      val testRows = FastSeq(
        RowSeq(0, 0, true),
        RowSeq(0, 5, true),
        RowSeq(0, 7, true),
        RowSeq(0, 10, true),
        RowSeq(0, 15, true),
        RowSeq(0, null, true),
      )
      checkAll(node, ref1, k1Full, testRows, trueIntervals, falseIntervals, naIntervals,
        trueResidual, falseResidual, naResidual)
    }

    check(
      If(gt(k1, I32(0)), lt(k1, I32(5)), gt(k1, I32(-5))),
      FastSeq(Interval(RowSeq(-5), RowSeq(5), false, false)),
      FastSeq(
        Interval(RowSeq(), RowSeq(-5), true, true),
        Interval(RowSeq(5), RowSeq(null), true, false),
      ),
      FastSeq(Interval(RowSeq(null), RowSeq(), true, true)),
    )
  }

  @Test def testSwitch(implicit ctx: ExecuteContext): Unit = {
    def check(
      node: IR,
      trueIntervals: IndexedSeq[Interval],
      falseIntervals: IndexedSeq[Interval],
      naIntervals: IndexedSeq[Interval],
      trueResidual: IR = True(),
      falseResidual: IR = True(),
      naResidual: IR = True(),
    ): Unit = {
      val testRows = FastSeq(
        RowSeq(0, 0, true),
        RowSeq(0, 5, true),
        RowSeq(0, -5, true),
        RowSeq(0, null, true),
        RowSeq(1, 0, true),
        RowSeq(1, 5, true),
        RowSeq(1, -5, true),
        RowSeq(1, null, true),
        RowSeq(null, null, true),
      )
      checkAll(node, ref1, k1Full, testRows, trueIntervals, falseIntervals, naIntervals,
        trueResidual, falseResidual, naResidual)
    }

    check(
      Switch(I32(0), gt(k1, I32(-5)), FastSeq(lt(k1, I32(5)))),
      FastSeq(Interval(RowSeq(), RowSeq(5), true, false)),
      FastSeq(Interval(RowSeq(5), RowSeq(null), true, false)),
      FastSeq(Interval(RowSeq(null), RowSeq(), true, true)),
    )

    check(
      Switch(I32(-1), gt(k1, I32(-5)), FastSeq(lt(k1, I32(5)))),
      FastSeq(Interval(RowSeq(-5), RowSeq(null), false, false)),
      FastSeq(Interval(RowSeq(), RowSeq(-5), true, true)),
      FastSeq(Interval(RowSeq(null), RowSeq(), true, true)),
    )

    val filter = Switch(GetField(ref1, "w"), gt(k1, I32(-5)), FastSeq(lt(k1, I32(5))))
    check(
      filter,
      FastSeq(Interval(RowSeq(), RowSeq(null), true, false)),
      FastSeq(
        Interval(RowSeq(), RowSeq(-5), true, true),
        Interval(RowSeq(5), RowSeq(null), true, false),
      ),
      FastSeq(Interval(RowSeq(), RowSeq(), true, true)),
      trueResidual = filter,
      falseResidual = ApplyUnaryPrimOp(Bang, filter),
      naResidual = IsNA(filter),
    )
  }

  @Test def testRelationalChildren(implicit ctx: ExecuteContext): Unit = {
    val testRows = FastSeq(
      RowSeq(0, 0, true),
      RowSeq(0, 10, true),
      RowSeq(0, 20, true),
      RowSeq(0, null, true),
    )

    val count = TableAggregate(
      TableRange(10, 1),
      ApplyAggOp(Count())(),
    )
    print(count.typ)
    val filter = gt(count, Cast(k1, TInt64))
    check(filter, ref1, k1Full, testRows, filter, FastSeq(Interval(RowSeq(), RowSeq(), true, true)))
  }

  @Test def testIntegration(implicit ctx: ExecuteContext): Unit = {
    val tab1 = TableRange(10, 5)

    def k = GetField(Ref(TableIR.rowName, tab1.typ.rowType), "idx")

    val tf = TableFilter(
      tab1,
      Coalesce(FastSeq(
        invoke(
          "land",
          TBoolean,
          ApplyComparisonOp(GT, k, I32(3)),
          ApplyComparisonOp(LTEQ, k, I32(9)),
        ),
        False(),
      )),
    )

    assert(ExtractIntervalFilters(ctx, tf).asInstanceOf[TableFilter].child.isInstanceOf[
      TableFilterIntervals
    ])
    assertEvalsTo(TableCount(tf), 6L)(ctx, ExecStrategy.interpretOnly)
  }
}
