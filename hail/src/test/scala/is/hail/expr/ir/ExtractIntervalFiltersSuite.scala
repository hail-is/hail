package is.hail.expr.ir

import is.hail.types.virtual._
import is.hail.utils.{FastSeq, Interval, IntervalEndpoint}
import is.hail.variant.{Locus, ReferenceGenome}
import is.hail.{ExecStrategy, HailSuite}
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class ExtractIntervalFiltersSuite extends HailSuite {

  lazy val ref1 = Ref("foo", TStruct("w" -> TInt32, "x" -> TInt32))
  lazy val k1 = GetField(ref1, "x")
  val ref1Key = FastSeq("x")

  val structRef = Ref("foo", TStruct("x" -> TInt32, "y" -> TInt32, "z" -> TInt32))
  val structRefKey = FastSeq("y", "z")

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

  @Test def testKeyComparison() {
    def check(node: ApplyComparisonOp, expectedInterval: Interval) {
      val (rw, intervals) = ExtractIntervalFilters.extractPartitionFilters(ctx, node, ref1, ref1Key).get
      assert(rw == True())
      assert(intervals.toSeq == FastSeq(expectedInterval))
    }

    check(ApplyComparisonOp(GT(TInt32), k1, I32(0)),
      Interval(wrappedIntervalEndpoint(0, 1), wrappedIntervalEndpoint(Int.MaxValue, 1)))
    check(ApplyComparisonOp(GT(TInt32), I32(0), k1),
      Interval(wrappedIntervalEndpoint(Int.MinValue, -1), wrappedIntervalEndpoint(0, -1)))

    check(ApplyComparisonOp(GTEQ(TInt32), k1, I32(0)),
      Interval(wrappedIntervalEndpoint(0, -1), wrappedIntervalEndpoint(Int.MaxValue, 1)))
    check(ApplyComparisonOp(GTEQ(TInt32), I32(0), k1),
      Interval(wrappedIntervalEndpoint(Int.MinValue, -1), wrappedIntervalEndpoint(0, 1)))

    check(ApplyComparisonOp(LT(TInt32), k1, I32(0)),
      Interval(wrappedIntervalEndpoint(Int.MinValue, -1), wrappedIntervalEndpoint(0, -1)))
    check(ApplyComparisonOp(LT(TInt32), I32(0), k1),
      Interval(wrappedIntervalEndpoint(0, 1), wrappedIntervalEndpoint(Int.MaxValue, 1)))

    check(ApplyComparisonOp(LTEQ(TInt32), k1, I32(0)),
      Interval(wrappedIntervalEndpoint(Int.MinValue, -1), wrappedIntervalEndpoint(0, 1)))
    check(ApplyComparisonOp(LTEQ(TInt32), I32(0), k1),
      Interval(wrappedIntervalEndpoint(0, -1), wrappedIntervalEndpoint(Int.MaxValue, 1)))

    check(ApplyComparisonOp(EQ(TInt32), k1, I32(0)),
      Interval(wrappedIntervalEndpoint(0, -1), wrappedIntervalEndpoint(0, 1)))
    check(ApplyComparisonOp(EQ(TInt32), I32(0), k1),
      Interval(wrappedIntervalEndpoint(0, -1), wrappedIntervalEndpoint(0, 1)))

    assert(ExtractIntervalFilters.extractPartitionFilters(ctx, ApplyComparisonOp(NEQ(TInt32), I32(0), k1), ref1, ref1Key).isEmpty)
    assert(ExtractIntervalFilters.extractPartitionFilters(ctx, ApplyComparisonOp(EQWithNA(TInt32), I32(0), k1), ref1, ref1Key).isEmpty)
    assert(ExtractIntervalFilters.extractPartitionFilters(ctx, ApplyComparisonOp(NEQWithNA(TInt32), I32(0), k1), ref1, ref1Key).isEmpty)
    assert(ExtractIntervalFilters.extractPartitionFilters(ctx, ApplyComparisonOp(Compare(TInt32), I32(0), k1), ref1, ref1Key).isEmpty)
  }

  @Test def testLiteralContains() {
    for (lit <- Array(
      Literal(TSet(TInt32), Set(1, 10)),
      Literal(TArray(TInt32), FastSeq(1, 10)),
      Literal(TDict(TInt32, TString), Map(1 -> "foo", 10 -> "bar")))) {
      val ir = invoke("contains", TBoolean, lit, k1)

      val (rw, i) = ExtractIntervalFilters.extractPartitionFilters(ctx, ir, ref1, ref1Key).get
      assert(rw == True())
      assert(i.toSeq == FastSeq(Interval(wrappedIntervalEndpoint(1, -1), wrappedIntervalEndpoint(1, 1)),
        Interval(wrappedIntervalEndpoint(10, -1), wrappedIntervalEndpoint(10, 1))))
    }
  }

  @Test def testLiteralContainsStruct() {
    hc // force initialization

    for (lit <- Array(
      Literal(TSet(structT1), Set(Row(1, 2), Row(3, 4))),
      Literal(TArray(structT1), FastSeq(Row(1, 2), Row(3, 4))),
      Literal(TDict(structT1, TString), Map(Row(1, 2) -> "foo", Row(3, 4) -> "bar")))) {
      for (k <- fullKeyRefs) {

        val ir = invoke("contains", TBoolean, lit, k)

        val (rw, i) = ExtractIntervalFilters.extractPartitionFilters(ctx, ir, structRef, structRefKey).get
        assert(rw == True())
        assert(i.toSeq == FastSeq(Interval(IntervalEndpoint(Row(1, 2), -1), IntervalEndpoint(Row(1, 2), 1)),
          Interval(IntervalEndpoint(Row(3, 4), -1), IntervalEndpoint(Row(3, 4), 1))))
      }
    }

    for (lit <- Array(
      Literal(TSet(structT2), Set(Row(1), Row(3))),
      Literal(TArray(structT2), FastSeq(Row(1), Row(3))),
      Literal(TDict(structT2, TString), Map(Row(1) -> "foo", Row(3) -> "bar")))) {
      for (k <- prefixKeyRefs) {

        val ir = invoke("contains", TBoolean, lit, k)

        val (rw, i) = ExtractIntervalFilters.extractPartitionFilters(ctx, ir, structRef, structRefKey).get
        assert(rw == True())
        assert(i.toSeq == FastSeq(Interval(IntervalEndpoint(Row(1), -1), IntervalEndpoint(Row(1), 1)),
          Interval(IntervalEndpoint(Row(3), -1), IntervalEndpoint(Row(3), 1))))
      }
    }
  }


  @Test def testIntervalContains() {
    val interval = Interval(IntervalEndpoint(1, 1), IntervalEndpoint(5, 1))
    val ir = invoke("contains", TBoolean, Literal(TInterval(TInt32), interval), k1)
    val (rw, i) = ExtractIntervalFilters.extractPartitionFilters(ctx, ir, ref1, ref1Key).get
    assert(rw == True())
    assert(i.toSeq == FastSeq(Interval(wrappedIntervalEndpoint(1, 1), wrappedIntervalEndpoint(5, 1))))
  }

  @Test def testIntervalContainsStruct() {
    val fullInterval = Interval(IntervalEndpoint(Row(1, 1), 1), IntervalEndpoint(Row(2, 2), 1))
    val prefixInterval = Interval(IntervalEndpoint(Row(1), 1), IntervalEndpoint(Row(2), 1))

    for (k <- fullKeyRefs) {
      val ir = invoke("contains", TBoolean, Literal(TInterval(structT1), fullInterval), k)
      val (rw, i) = ExtractIntervalFilters.extractPartitionFilters(ctx, ir, structRef, structRefKey).get
      assert(rw == True())
      assert(i.toSeq == FastSeq(fullInterval))
    }

    for (k <- prefixKeyRefs) {
      val ir = invoke("contains", TBoolean, Literal(TInterval(structT2), prefixInterval), k)
      val (rw, i) = ExtractIntervalFilters.extractPartitionFilters(ctx, ir, structRef, structRefKey).get
      assert(rw == True())
      assert(i.toSeq == FastSeq(prefixInterval))
    }
  }

  @Test def testLocusContigComparison() {
    hc // force initialization
    val ref = Ref("foo", TStruct("x" -> TLocus(ReferenceGenome.GRCh38)))
    val k = GetField(ref, "x")

    val ir1 = ApplyComparisonOp(EQ(TString), Str("chr2"), invoke("contig", TString, k))
    val ir2 = ApplyComparisonOp(EQ(TString), invoke("contig", TString, k), Str("chr2"))

    val (rw1, i1) = ExtractIntervalFilters.extractPartitionFilters(ctx, ir1, ref, ref1Key).get
    assert(rw1 == True())
    assert(i1.toSeq == FastSeq(Interval(wrappedIntervalEndpoint(Locus("chr2", 1), -1),
      wrappedIntervalEndpoint(Locus("chr2", grch38.contigLength("chr2")), -1))))

    val (rw2, i2) = ExtractIntervalFilters.extractPartitionFilters(ctx, ir2, ref, ref1Key).get
    assert(rw2 == True())
    assert(i2.toSeq == FastSeq(Interval(wrappedIntervalEndpoint(Locus("chr2", 1), -1),
      wrappedIntervalEndpoint(Locus("chr2", grch38.contigLength("chr2")), -1))))
  }

  @Test def testLocusPositionComparison() {
    hc // force initialization
    val ref = Ref("foo", TStruct("x" -> TLocus(ReferenceGenome.GRCh38)))
    val k = GetField(ref, "x")
    val pos = invoke("position", TInt32, k)

    def check(node: ApplyComparisonOp, expectedInterval: (String, Int) => Interval) {
      val (rw, intervals) = ExtractIntervalFilters.extractPartitionFilters(ctx, node, ref, ref1Key).get
      assert(rw == True())
      assert(intervals.toSeq == grch38.contigs
        .map { c => expectedInterval(c, grch38.contigLength(c)) }
        .filter(_ != null)
        .toFastSeq)
    }

    check(ApplyComparisonOp(GT(TInt32), pos, I32(100)),
      (c: String, len: Int) => if (len < 100)
        null
      else
        Interval(wrappedIntervalEndpoint(Locus(c, 100), 1), wrappedIntervalEndpoint(Locus(c, len), -1)))
    check(ApplyComparisonOp(GT(TInt32), pos, I32(-1000)),
      (c: String, len: Int) => Interval(wrappedIntervalEndpoint(Locus(c, 1), -1), wrappedIntervalEndpoint(Locus(c, len), -1)))
    check(ApplyComparisonOp(GT(TInt32), I32(100), pos),
      (c: String, _: Int) => Interval(wrappedIntervalEndpoint(Locus(c, 1), -1), wrappedIntervalEndpoint(Locus(c, 100), -1)))

    check(ApplyComparisonOp(GTEQ(TInt32), pos, I32(100)),
      (c: String, len: Int) => if (len < 100)
        null
      else
        Interval(wrappedIntervalEndpoint(Locus(c, 100), -1), wrappedIntervalEndpoint(Locus(c, len), -1)))
    check(ApplyComparisonOp(GTEQ(TInt32), pos, I32(-1000)),
      (c: String, len: Int) => Interval(wrappedIntervalEndpoint(Locus(c, 1), -1), wrappedIntervalEndpoint(Locus(c, len), -1)))
    check(ApplyComparisonOp(GTEQ(TInt32), I32(100), pos),
      (c: String, _: Int) => Interval(wrappedIntervalEndpoint(Locus(c, 1), -1), wrappedIntervalEndpoint(Locus(c, 100), 1)))

    check(ApplyComparisonOp(LT(TInt32), pos, I32(100)),
      (c: String, _: Int) => Interval(wrappedIntervalEndpoint(Locus(c, 1), -1), wrappedIntervalEndpoint(Locus(c, 100), -1)))
    check(ApplyComparisonOp(LT(TInt32), pos, I32(-1000)),
      (c: String, len: Int) => null)
    check(ApplyComparisonOp(LT(TInt32), I32(100), pos),
      (c: String, len: Int) => if (len < 100)
        null
      else
        Interval(wrappedIntervalEndpoint(Locus(c, 100), 1), wrappedIntervalEndpoint(Locus(c, len), -1)))

    check(ApplyComparisonOp(LTEQ(TInt32), pos, I32(100)),
      (c: String, _: Int) => Interval(wrappedIntervalEndpoint(Locus(c, 1), -1), wrappedIntervalEndpoint(Locus(c, 100), 1)))
    check(ApplyComparisonOp(LTEQ(TInt32), pos, I32(-1000)),
      (c: String, len: Int) => null)
    check(ApplyComparisonOp(LTEQ(TInt32), I32(100), pos),
      (c: String, len: Int) => if (len < 100)
        null
      else
        Interval(wrappedIntervalEndpoint(Locus(c, 100), -1), wrappedIntervalEndpoint(Locus(c, len), -1)))

    check(ApplyComparisonOp(EQ(TInt32), pos, I32(100)),
      (c: String, len: Int) => if (len < 100)
        null
      else
        Interval(wrappedIntervalEndpoint(Locus(c, 100), -1), wrappedIntervalEndpoint(Locus(c, 100), 1)))
    check(ApplyComparisonOp(EQ(TInt32), I32(-1000), pos),
      (c: String, len: Int) => null)
    check(ApplyComparisonOp(EQ(TInt32), I32(100), pos),
      (c: String, len: Int) => if (len < 100)
        null
      else
        Interval(wrappedIntervalEndpoint(Locus(c, 100), -1), wrappedIntervalEndpoint(Locus(c, 100), 1)))

    assert(ExtractIntervalFilters.extractPartitionFilters(ctx, ApplyComparisonOp(NEQ(TInt32), I32(0), pos), ref, ref1Key).isEmpty)
    assert(ExtractIntervalFilters.extractPartitionFilters(ctx, ApplyComparisonOp(EQWithNA(TInt32), I32(0), pos), ref, ref1Key).isEmpty)
    assert(ExtractIntervalFilters.extractPartitionFilters(ctx, ApplyComparisonOp(NEQWithNA(TInt32), I32(0), pos), ref, ref1Key).isEmpty)
    assert(ExtractIntervalFilters.extractPartitionFilters(ctx, ApplyComparisonOp(Compare(TInt32), I32(0), pos), ref, ref1Key).isEmpty)
  }

  @Test def testLocusContigContains() {
    hc // force initialization
    val ref = Ref("foo", TStruct("x" -> TLocus(ReferenceGenome.GRCh38)))
    val k = GetField(ref, "x")
    val contig = invoke("contig", TString, k)

    for (lit <- Array(
      Literal(TSet(TString), Set("chr1", "chr10")),
      Literal(TArray(TString), FastSeq("chr1", "chr10")),
      Literal(TDict(TString, TString), Map("chr1" -> "foo", "chr10" -> "bar")))) {

      val ir = invoke("contains", TBoolean, lit, contig)

      val (rw, intervals) = ExtractIntervalFilters.extractPartitionFilters(ctx, ir, ref, ref1Key).get
      assert(rw == True())
      assert(intervals.toSeq == FastSeq(
        Interval(
          wrappedIntervalEndpoint(Locus("chr1", 1), -1),
          wrappedIntervalEndpoint(Locus("chr1", grch38.contigLength("chr1")), -1)),
        Interval(
          wrappedIntervalEndpoint(Locus("chr10", 1), -1),
          wrappedIntervalEndpoint(Locus("chr10", grch38.contigLength("chr10")), -1))))
    }
  }

  @Test def testIntervalListFold() {
    val inIntervals = FastSeq(
      Interval(IntervalEndpoint(0, -1), IntervalEndpoint(10, -1)),
      null,
      Interval(IntervalEndpoint(20, -1), IntervalEndpoint(25, -1)),
      Interval(IntervalEndpoint(-10, -1), IntervalEndpoint(5, -1))
    )

    val ir = StreamFold(
      ToStream(Literal(TArray(TInterval(TInt32)), inIntervals)),
      False(),
      "acc",
      "elt",
      invoke("lor", TBoolean,
        Ref("acc", TBoolean),
        invoke("contains", TBoolean, Ref("elt", TInterval(TInt32)), k1)))
    TypeCheck(ctx, ir, BindingEnv(Env(ref1.name -> ref1.typ)))

    val (rw, intervals) = ExtractIntervalFilters.extractPartitionFilters(ctx, ir, ref1, ref1Key).get
    assert(rw == True())
    assert(intervals.toSeq == FastSeq(
      Interval(wrappedIntervalEndpoint(-10, -1), wrappedIntervalEndpoint(10, -1)),
      Interval(wrappedIntervalEndpoint(20, -1), wrappedIntervalEndpoint(25, -1))))
  }

  @Test def testDisjunction() {
    val ir1 = ApplyComparisonOp(GT(TInt32), k1, I32(0))
    val ir2 = ApplyComparisonOp(GT(TInt32), k1, I32(10))

    val (rw, intervals) = ExtractIntervalFilters.extractPartitionFilters(ctx, invoke("lor", TBoolean, ir1, ir2), ref1, ref1Key).get
    assert(rw == True())
    assert(intervals.toSeq == FastSeq(Interval(wrappedIntervalEndpoint(0, 1), wrappedIntervalEndpoint(Int.MaxValue, 1))))

    assert(ExtractIntervalFilters.extractPartitionFilters(ctx, invoke("lor", TBoolean, ir1, Ref("foo", TBoolean)), ref1, ref1Key).isEmpty)

    val ir3 = invoke("lor", TBoolean, ir1, invoke("land", TBoolean, ir2, Ref("foo", TBoolean)))
    assert(ExtractIntervalFilters.extractPartitionFilters(ctx, ir3, ref1, ref1Key).isEmpty)
  }

  @Test def testConjunction() {
    val ir1 = ApplyComparisonOp(GT(TInt32), k1, I32(0))
    val ir2 = ApplyComparisonOp(GT(TInt32), k1, I32(10))
    val ir3 = In(0, TBoolean)

    val (rw1, intervals1) = ExtractIntervalFilters.extractPartitionFilters(ctx, invoke("land", TBoolean, ir1, ir2), ref1, ref1Key).get
    assert(rw1 == invoke("land", TBoolean, True(), True()))
    assert(intervals1.toSeq == FastSeq(Interval(wrappedIntervalEndpoint(10, 1), wrappedIntervalEndpoint(Int.MaxValue, 1))))

    val (rw2, intervals2) = ExtractIntervalFilters.extractPartitionFilters(ctx, invoke("land", TBoolean, ir3, ir2), ref1, ref1Key).get
    assert(rw2 == invoke("land", TBoolean, ir3, True()))
    assert(intervals2.toSeq == FastSeq(Interval(wrappedIntervalEndpoint(10, 1), wrappedIntervalEndpoint(Int.MaxValue, 1))))

    assert(ExtractIntervalFilters.extractPartitionFilters(ctx, invoke("land", TBoolean, ir3, ir3), ref1, ref1Key).isEmpty)
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
