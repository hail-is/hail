package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.TestUtils._
import org.testng.annotations.Test
import is.hail.utils.{FastIndexedSeq, FastSeq}
import org.apache.spark.sql.Row

class ExtractAggregatorsSuite {

  def testSum(a: IndexedSeq[Any], expected: Any) {
    val aggSig = AggSignature(Sum(), TFloat64(), FastSeq())
    assertEvalsTo(ApplyAggOp(
      SeqOp(Ref("a", TFloat64()), I32(0), aggSig),
      FastSeq(), aggSig),
      (a.map(i => Row(i)), TStruct("a" -> TFloat64())),
      expected)
  }

  @Test
  def sum() {
    testSum((0 to 100).map(_.toDouble), 5050.0)
  }

  @Test
  def sumEmpty() {
    testSum(FastIndexedSeq(), 0.0)
  }

  @Test
  def sumOne() {
    testSum(FastIndexedSeq(42.0), 42.0)
  }

  @Test
  def sumMissing() {
    testSum(FastIndexedSeq(null, 42.0, null), 42.0)
  }

  @Test
  def sumAllMissing() {
    testSum(FastIndexedSeq(null, null, null), 0.0)
  }

  @Test
  def sumMultivar() {
    val aggSig = AggSignature(Sum(), TFloat64(), FastSeq())
    assertEvalsTo(ApplyAggOp(
      SeqOp(ApplyBinaryPrimOp(Multiply(), Ref("a", TFloat64()), Ref("b", TFloat64())), I32(0), aggSig),
      FastSeq(), aggSig),
      (FastIndexedSeq(Row(1.0, 10.0), Row(10.0, 10.0), Row(null, 10.0)), TStruct("a" -> TFloat64(), "b" -> TFloat64())),
      110.0)
  }

  @Test
  def ifInApplyAggOp() {
    val aggSig = AggSignature(Sum(), TFloat64(), FastSeq())
    assertEvalsTo(
      ApplyAggOp(
        If(
          ApplyBinaryPrimOp(NEQ(), Ref("a", TFloat64()), F64(10.0)),
          SeqOp(ApplyBinaryPrimOp(Multiply(), Ref("a", TFloat64()), Ref("b", TFloat64())),
            I32(0), aggSig),
          Begin(FastSeq())),
        FastSeq(), aggSig),
      (FastIndexedSeq(Row(1.0, 10.0), Row(10.0, 10.0), Row(null, 10.0)), TStruct("a" -> TFloat64(), "b" -> TFloat64())),
      10.0)
  }

  @Test
  def fraction() {
    val aggSig = AggSignature(Fraction(), TBoolean(), FastSeq())
    assertEvalsTo(
      ApplyAggOp(
        SeqOp(Ref("a", TBoolean()), I32(0), aggSig),
        FastSeq(), aggSig),
      (FastIndexedSeq(Row(true), Row(false), Row(null), Row(true), Row(false)), TStruct("a" -> TBoolean())),
      2.0 / 5.0)
  }

  @Test
  def collectBoolean() {
    val aggSig = AggSignature(Collect(), TBoolean(), FastSeq())
    assertEvalsTo(
      ApplyAggOp(
        SeqOp(Ref("a", TBoolean()), I32(0), aggSig),
        FastSeq(), aggSig),
      (FastIndexedSeq(Row(true), Row(false), Row(null), Row(true), Row(false)), TStruct("a" -> TBoolean())),
      FastIndexedSeq(true, false, null, true, false))
  }

  @Test
  def collectInt() {
    val aggSig = AggSignature(Collect(), TInt32(), FastSeq())
    assertEvalsTo(
      ApplyAggOp(
        SeqOp(Ref("a", TInt32()), I32(0), aggSig),
        FastSeq(), aggSig),
      (FastIndexedSeq(Row(10), Row(null), Row(5)), TStruct("a" -> TInt32())),
      FastIndexedSeq(10, null, 5))
  }

  /*

  @Test
  def flatMap() {
    val tAgg = TAggregable(TFloat64(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val ir =
      ApplyAggOp(
        AggFlatMap(
          AggMap(AggIn(tAgg),
            "x",
            ApplyBinaryPrimOp(Multiply(), Ref("scope0", TFloat64()), Ref("x", TFloat64()))),
          "x",
          MakeArray(Seq(F64(100.0), Ref("x", TFloat64())), TArray(TFloat64()))),
        Sum())
    val sum = run[Double, Double, Double](region,
      ir,
      addBoxedArray[java.lang.Double](region, 1.0, 10.0, null),
      _ => 10.0)
    assert(sum === 110.0 + 200.0 + 100.0)
  }

  @Test
  def maxBoolean1() {
    val tAgg = TAggregable(TBoolean(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val input = Array[Boolean](true, false, false)
    val ir =
      ApplyAggOp(AggIn(tAgg), Max())
    val actual = run[Boolean, Double, Boolean](region,
      ir,
      addBoxedArray(region, input: _*),
      _ => 10.0)

    assert(actual === true)
  }

  @Test
  def maxBoolean2() {
    val tAgg = TAggregable(TBoolean(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val input = Array[Boolean](false, false, false)
    val ir =
      ApplyAggOp(AggIn(tAgg), Max())
    val actual = run[Boolean, Double, Boolean](region,
      ir,
      addBoxedArray(region, input: _*),
      _ => 10.0)

    assert(actual === false)
  }

  @Test
  def maxInt() {
    val tAgg = TAggregable(TInt32(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val input = Array[Int](5, 10, -5)
    val ir =
      ApplyAggOp(AggIn(tAgg), Max())
    val actual = run[Int, Double, Int](region,
      ir,
      addBoxedArray(region, input: _*),
      _ => 10.0)

    assert(actual === 10)
  }

  @Test
  def maxLong() {
    val tAgg = TAggregable(TInt64(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val input = Array[Long](5L, 10L, -5L)
    val ir =
      ApplyAggOp(AggIn(tAgg), Max())
    val actual = run[Long, Double, Long](region,
      ir,
      addBoxedArray(region, input: _*),
      _ => 10.0)

    assert(actual === 10L)
  }

  @Test
  def maxFloat() {
    val tAgg = TAggregable(TFloat32(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val input = Array[Float](5.0f, 10.0f, -5.0f)
    val ir =
      ApplyAggOp(AggIn(tAgg), Max())
    val actual = run[Float, Double, Float](region,
      ir,
      addBoxedArray(region, input: _*),
      _ => 10.0)

    assert(actual === 10.0f)
  }

  @Test
  def maxDouble() {
    val tAgg = TAggregable(TFloat64(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val input = Array[Double](5.0, 10.0, -5.0)
    val ir =
      ApplyAggOp(AggIn(tAgg), Max())
    val actual = run[Double, Double, Double](region,
      ir,
      addBoxedArray(region, input: _*),
      _ => 10.0)

    assert(actual === 10.0)
  }

  @Test
  def takeInt() {
    val tAgg = TAggregable(TInt32(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val input = Array[Int](5, 10, -5)
    val expected = input.take(2)
    val ir =
      ApplyAggOp(AggIn(tAgg), Take(), FastSeq(I32(2)))
    val aOff = run[Int, Double, Long](region,
      ir,
      addBoxedArray(region, input: _*),
      _ => 10.0)

    assert(loadArray[Int](region, aOff) === expected)
  }

  @Test
  def takeDouble() {
    val tAgg = TAggregable(TFloat64(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val input = Array[Double](5.0, 10.0, -5.0)
    val expected = input.take(2)
    val ir =
      ApplyAggOp(AggIn(tAgg), Take(), FastSeq(I32(2)))
    val aOff = run[Double, Double, Long](region,
      ir,
      addBoxedArray(region, input: _*),
      _ => 10.0)

    assert(loadArray[Double](region, aOff) === expected)
  }

  @Test
  def takeDoubleWithMissingness() {
    val tAgg = TAggregable(TFloat64(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val input = Array[java.lang.Double](5.0, null, -5.0)
    val expected = input.take(2)
    val ir =
      ApplyAggOp(AggIn(tAgg), Take(), FastSeq(I32(2)))
    val aOff = run[Double, Double, Long](region,
      ir,
      addBoxedArray(region, input: _*),
      _ => 10.0)

    assert(loadArray[java.lang.Double](region, aOff) === expected)
  }

  @Test
  def histogram() {
    val tAgg = TAggregable(TFloat64(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val input = Array[java.lang.Double](5.0, null, -5.0, 1.0, 15.0, 1.0, 17.0, 1.5)
    val expected = input.take(2)
    val ir =
      ApplyAggOp(AggIn(tAgg), Histogram(), FastSeq(F64(0.0), F64(10.0), I32(5)))
    val hOff = run[Double, Double, Long](region,
      ir,
      addBoxedArray(region, input: _*),
      _ => 10.0)

    val t = RegionValueHistogramAggregator.typ
    val binEdges = t.fieldIdx("bin_edges")
    val binFrequencies = t.fieldIdx("bin_freq")
    val nLess = t.fieldIdx("n_smaller")
    val nGreater = t.fieldIdx("n_larger")
    assert(t.isFieldDefined(region, hOff, binEdges))
    assert(t.isFieldDefined(region, hOff, binFrequencies))
    assert(t.isFieldDefined(region, hOff, nLess))
    assert(t.isFieldDefined(region, hOff, nGreater))
    val binEdgeArray = loadArray[Double](
      region, t.loadField(region, hOff, binEdges))
    assert(binEdgeArray === Seq(0.0, 2.0, 4.0, 6.0, 8.0, 10.0))
    val binFrequenciesArray = loadArray[Long](
      region, t.loadField(region, hOff, binFrequencies))
    assert(binFrequenciesArray === Array[Long](3L, 0L, 1L, 0L, 0L))
    assert(region.loadLong(t.loadField(region, hOff, nLess)) === 1L)
    assert(region.loadLong(t.loadField(region, hOff, nGreater)) === 2L)
  }
  */
}
