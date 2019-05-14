package is.hail.expr.ir

import is.hail.TestUtils._
import is.hail.expr.types.virtual._
import is.hail.utils.{FastIndexedSeq, FastSeq, Interval, IntervalEndpoint}
import is.hail.variant.{Locus, ReferenceGenome}
import is.hail.{ExecStrategy, SparkSuite}
import org.testng.annotations.Test

class ExtractIntervalFiltersSuite extends SparkSuite {

  lazy val ref1 = Ref("foo", TStruct("x" -> TInt32()))
  lazy val k1 = GetField(ref1, "x")

  @Test def testKeyComparison() {
    def check(node: ApplyComparisonOp, expectedInterval: Interval) {
      val (rw, intervals) = ExtractIntervalFilters.extractAndRewrite(node, ref1, k1).get
      assert(rw == True())
      assert(intervals.toSeq == FastSeq(expectedInterval))
    }

    check(ApplyComparisonOp(GT(TInt32()), k1, I32(0)),
      Interval(IntervalEndpoint(0, 1), IntervalEndpoint(Int.MaxValue, 1)))
    check(ApplyComparisonOp(GT(TInt32()), I32(0), k1),
      Interval(IntervalEndpoint(Int.MinValue, -1), IntervalEndpoint(0, -1)))

    check(ApplyComparisonOp(GTEQ(TInt32()), k1, I32(0)),
      Interval(IntervalEndpoint(0, -1), IntervalEndpoint(Int.MaxValue, 1)))
    check(ApplyComparisonOp(GTEQ(TInt32()), I32(0), k1),
      Interval(IntervalEndpoint(Int.MinValue, -1), IntervalEndpoint(0, 1)))

    check(ApplyComparisonOp(LT(TInt32()), k1, I32(0)),
      Interval(IntervalEndpoint(Int.MinValue, -1), IntervalEndpoint(0, -1)))
    check(ApplyComparisonOp(LT(TInt32()), I32(0), k1),
      Interval(IntervalEndpoint(0, 1), IntervalEndpoint(Int.MaxValue, 1)))

    check(ApplyComparisonOp(LTEQ(TInt32()), k1, I32(0)),
      Interval(IntervalEndpoint(Int.MinValue, -1), IntervalEndpoint(0, 1)))
    check(ApplyComparisonOp(LTEQ(TInt32()), I32(0), k1),
      Interval(IntervalEndpoint(0, -1), IntervalEndpoint(Int.MaxValue, 1)))

    check(ApplyComparisonOp(EQ(TInt32()), k1, I32(0)),
      Interval(IntervalEndpoint(0, -1), IntervalEndpoint(0, 1)))
    check(ApplyComparisonOp(EQ(TInt32()), I32(0), k1),
      Interval(IntervalEndpoint(0, -1), IntervalEndpoint(0, 1)))

    assert(ExtractIntervalFilters.extractAndRewrite(ApplyComparisonOp(NEQ(TInt32()), I32(0), k1), ref1, k1).isEmpty)
    assert(ExtractIntervalFilters.extractAndRewrite(ApplyComparisonOp(EQWithNA(TInt32()), I32(0), k1), ref1, k1).isEmpty)
    assert(ExtractIntervalFilters.extractAndRewrite(ApplyComparisonOp(NEQWithNA(TInt32()), I32(0), k1), ref1, k1).isEmpty)
    assert(ExtractIntervalFilters.extractAndRewrite(ApplyComparisonOp(Compare(TInt32()), I32(0), k1), ref1, k1).isEmpty)
  }

  @Test def testLiteralContains() {
    for (lit <- Array(
      Literal(TSet(TInt32()), Set(1, 10)),
      Literal(TArray(TInt32()), FastIndexedSeq(1, 10)),
      Literal(TDict(TInt32(), TString()), Map(1 -> "foo", 10 -> "bar")))) {
      val ir = invoke("contains", lit, k1)

      val (rw, i) = ExtractIntervalFilters.extractAndRewrite(ir, ref1, k1).get
      assert(rw == True())
      assert(i.toSeq == FastSeq(Interval(IntervalEndpoint(1, -1), IntervalEndpoint(1, 1)),
        Interval(IntervalEndpoint(10, -1), IntervalEndpoint(10, 1))))
    }
  }

  @Test def testIntervalContains() {
    val interval = Interval(IntervalEndpoint(1, 1), IntervalEndpoint(5, 1))
    val ir = invoke("contains", Literal(TInterval(TInt32()), interval), k1)
    val (rw, i) = ExtractIntervalFilters.extractAndRewrite(ir, ref1, k1).get
    assert(rw == True())
    assert(i.toSeq == FastSeq(interval))
  }

  @Test def testLocusContigComparison() {
    hc // force initialization
    val ref = Ref("foo", TStruct("x" -> TLocus(ReferenceGenome.GRCh38)))
    val k = GetField(ref, "x")

    val ir1 = ApplyComparisonOp(EQ(TString()), Str("chr2"), invoke("contig", k))
    val ir2 = ApplyComparisonOp(EQ(TString()), invoke("contig", k), Str("chr2"))

    val (rw1, i1) = ExtractIntervalFilters.extractAndRewrite(ir1, ref, k).get
    assert(rw1 == True())
    assert(i1.toSeq == FastSeq(Interval(IntervalEndpoint(Locus("chr2", 1), -1),
      IntervalEndpoint(Locus("chr2", ReferenceGenome.GRCh38.contigLength("chr2")), -1))))

    val (rw2, i2) = ExtractIntervalFilters.extractAndRewrite(ir2, ref, k).get
    assert(rw2 == True())
    assert(i2.toSeq == FastSeq(Interval(IntervalEndpoint(Locus("chr2", 1), -1),
      IntervalEndpoint(Locus("chr2", ReferenceGenome.GRCh38.contigLength("chr2")), -1))))
  }

  @Test def testLocusPositionComparison() {
    hc // force initialization
    val ref = Ref("foo", TStruct("x" -> TLocus(ReferenceGenome.GRCh38)))
    val k = GetField(ref, "x")
    val pos = invoke("position", k)

    def check(node: ApplyComparisonOp, expectedInterval: (String, Int) => Interval) {
      val (rw, intervals) = ExtractIntervalFilters.extractAndRewrite(node, ref, k).get
      assert(rw == True())
      assert(intervals.toSeq == ReferenceGenome.GRCh38.contigs
        .map { c => expectedInterval(c, ReferenceGenome.GRCh38.contigLength(c)) }
        .filter(_ != null)
        .toFastSeq)
    }

    check(ApplyComparisonOp(GT(TInt32()), pos, I32(100)),
      (c: String, len: Int) => if (len < 100)
        null
      else
        Interval(IntervalEndpoint(Locus(c, 100), 1), IntervalEndpoint(Locus(c, len), -1)))
    check(ApplyComparisonOp(GT(TInt32()), pos, I32(-1000)),
      (c: String, len: Int) => Interval(IntervalEndpoint(Locus(c, 1), -1), IntervalEndpoint(Locus(c, len), -1)))
    check(ApplyComparisonOp(GT(TInt32()), I32(100), pos),
      (c: String, _: Int) => Interval(IntervalEndpoint(Locus(c, 1), -1), IntervalEndpoint(Locus(c, 100), -1)))

    check(ApplyComparisonOp(GTEQ(TInt32()), pos, I32(100)),
      (c: String, len: Int) => if (len < 100)
        null
      else
        Interval(IntervalEndpoint(Locus(c, 100), -1), IntervalEndpoint(Locus(c, len), -1)))
    check(ApplyComparisonOp(GTEQ(TInt32()), pos, I32(-1000)),
      (c: String, len: Int) => Interval(IntervalEndpoint(Locus(c, 1), -1), IntervalEndpoint(Locus(c, len), -1)))
    check(ApplyComparisonOp(GTEQ(TInt32()), I32(100), pos),
      (c: String, _: Int) => Interval(IntervalEndpoint(Locus(c, 1), -1), IntervalEndpoint(Locus(c, 100), 1)))

    check(ApplyComparisonOp(LT(TInt32()), pos, I32(100)),
      (c: String, _: Int) => Interval(IntervalEndpoint(Locus(c, 1), -1), IntervalEndpoint(Locus(c, 100), -1)))
    check(ApplyComparisonOp(LT(TInt32()), pos, I32(-1000)),
      (c: String, len: Int) => null)
    check(ApplyComparisonOp(LT(TInt32()), I32(100), pos),
      (c: String, len: Int) => if (len < 100)
        null
      else
        Interval(IntervalEndpoint(Locus(c, 100), 1), IntervalEndpoint(Locus(c, len), -1)))

    check(ApplyComparisonOp(LTEQ(TInt32()), pos, I32(100)),
      (c: String, _: Int) => Interval(IntervalEndpoint(Locus(c, 1), -1), IntervalEndpoint(Locus(c, 100), 1)))
    check(ApplyComparisonOp(LTEQ(TInt32()), pos, I32(-1000)),
      (c: String, len: Int) => null)
    check(ApplyComparisonOp(LTEQ(TInt32()), I32(100), pos),
      (c: String, len: Int) => if (len < 100)
        null
      else
        Interval(IntervalEndpoint(Locus(c, 100), -1), IntervalEndpoint(Locus(c, len), -1)))

    check(ApplyComparisonOp(EQ(TInt32()), pos, I32(100)),
      (c: String, len: Int) => if (len < 100)
        null
      else
        Interval(IntervalEndpoint(Locus(c, 100), -1), IntervalEndpoint(Locus(c, 100), 1)))
    check(ApplyComparisonOp(EQ(TInt32()), I32(-1000), pos),
      (c: String, len: Int) => null)
    check(ApplyComparisonOp(EQ(TInt32()), I32(100), pos),
      (c: String, len: Int) => if (len < 100)
        null
      else
        Interval(IntervalEndpoint(Locus(c, 100), -1), IntervalEndpoint(Locus(c, 100), 1)))

    assert(ExtractIntervalFilters.extractAndRewrite(ApplyComparisonOp(NEQ(TInt32()), I32(0), pos), ref, k).isEmpty)
    assert(ExtractIntervalFilters.extractAndRewrite(ApplyComparisonOp(EQWithNA(TInt32()), I32(0), pos), ref, k).isEmpty)
    assert(ExtractIntervalFilters.extractAndRewrite(ApplyComparisonOp(NEQWithNA(TInt32()), I32(0), pos), ref, k).isEmpty)
    assert(ExtractIntervalFilters.extractAndRewrite(ApplyComparisonOp(Compare(TInt32()), I32(0), pos), ref, k).isEmpty)
  }

  @Test def testLocusContigContains() {
    hc // force initialization
    val ref = Ref("foo", TStruct("x" -> TLocus(ReferenceGenome.GRCh38)))
    val k = GetField(ref, "x")
    val contig = invoke("contig", k)

    for (lit <- Array(
      Literal(TSet(TString()), Set("chr1", "chr10")),
      Literal(TArray(TString()), FastIndexedSeq("chr1", "chr10")),
      Literal(TDict(TString(), TString()), Map("chr1" -> "foo", "chr10" -> "bar")))) {

      val ir = invoke("contains", lit, contig)

      val (rw, intervals) = ExtractIntervalFilters.extractAndRewrite(ir, ref, k).get
      assert(rw == True())
      assert(intervals.toSeq == FastSeq(
        Interval(
          IntervalEndpoint(Locus("chr1", 1), -1),
          IntervalEndpoint(Locus("chr1", ReferenceGenome.GRCh38.contigLength("chr1")), -1)),
        Interval(
          IntervalEndpoint(Locus("chr10", 1), -1),
          IntervalEndpoint(Locus("chr10", ReferenceGenome.GRCh38.contigLength("chr10")), -1))))
    }
  }

  @Test def testIntervalListFold() {
    val inIntervals = FastIndexedSeq(
      Interval(IntervalEndpoint(0, -1), IntervalEndpoint(10, -1)),
      null,
      Interval(IntervalEndpoint(-10, -1), IntervalEndpoint(5, -1)))

    val ir = ArrayFold(
      Literal(TArray(TInterval(TInt32())), inIntervals),
      False(),
      "acc",
      "elt",
      invoke("||", Ref("acc", TBoolean()), invoke("contains", Ref("elt", TInterval(TInt32())), k1))
    )
    val (rw, intervals) = ExtractIntervalFilters.extractAndRewrite(ir, ref1, k1).get
    assert(rw == True())
    assert(intervals.toSeq == FastSeq(Interval(IntervalEndpoint(-10, -1), IntervalEndpoint(10, -1))))
  }

  @Test def testDisjunction() {
    val ir1 = ApplyComparisonOp(GT(TInt32()), k1, I32(0))
    val ir2 = ApplyComparisonOp(GT(TInt32()), k1, I32(10))

    val (rw, intervals) = ExtractIntervalFilters.extractAndRewrite(invoke("||", ir1, ir2), ref1, k1).get
    assert(rw == True())
    assert(intervals.toSeq == FastSeq(Interval(IntervalEndpoint(0, 1), IntervalEndpoint(Int.MaxValue, 1))))

    assert(ExtractIntervalFilters.extractAndRewrite(invoke("||", ir1, Ref("foo", TBoolean())), ref1, k1).isEmpty)
  }

  @Test def testConjunction() {
    val ir1 = ApplyComparisonOp(GT(TInt32()), k1, I32(0))
    val ir2 = ApplyComparisonOp(GT(TInt32()), k1, I32(10))
    val ir3 = In(0, TBoolean())

    val (rw1, intervals1) = ExtractIntervalFilters.extractAndRewrite(invoke("&&", ir1, ir2), ref1, k1).get
    assert(rw1 == invoke("&&", True(), True()))
    assert(intervals1.toSeq == FastSeq(Interval(IntervalEndpoint(10, 1), IntervalEndpoint(Int.MaxValue, 1))))

    val (rw2, intervals2) = ExtractIntervalFilters.extractAndRewrite(invoke("&&", ir3, ir2), ref1, k1).get
    assert(rw2 == invoke("&&", ir3, True()))
    assert(intervals2.toSeq == FastSeq(Interval(IntervalEndpoint(10, 1), IntervalEndpoint(Int.MaxValue, 1))))

    assert(ExtractIntervalFilters.extractAndRewrite(invoke("&&", ir3, ir3), ref1, k1).isEmpty)
  }

  @Test def testIntegration() {
    hc // force initialization
    val tab1 = TableRange(10, 5)

    def k = GetField(Ref("row", tab1.typ.rowType), "idx")

    val tf = TableFilter(tab1,
      Coalesce(FastSeq(invoke("&&",
        ApplyComparisonOp(GT(TInt32()), k, I32(3)),
        ApplyComparisonOp(LTEQ(TInt32()), k, I32(9))
      ), False())))

    assert(ExtractIntervalFilters(tf).asInstanceOf[TableFilter].child.isInstanceOf[TableToTableApply])
    assertEvalsTo(TableCount(tf), 6L)(ExecStrategy.interpretOnly)
  }
}
