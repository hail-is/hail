package is.hail.expr.ir

import is.hail.annotations._
import is.hail.annotations.aggregators._
import is.hail.asm4s._
import ScalaToRegionValue._
import RegionValueToScala._
import is.hail.check._
import is.hail.expr.{HailRep, hailType}
import is.hail.expr.types._
import org.testng.annotations.Test
import org.scalatest._
import Matchers._

class ExtractAggregatorsSuite {

  private def defaultValue(t: Type): Any = t match {
    case _: TBoolean => false
    case _: TInt32 => 0
    case _: TInt64 => 0L
    case _: TFloat32 => 0.0f
    case _: TFloat64 => 0.0
    case _ => 0L // reference types
  }

  private def packageResults(region: Region, t: TStruct, a: Array[RegionValueAggregator]): Long = {
    val rvb = new RegionValueBuilder()
    rvb.set(region)
    rvb.start(t)
    rvb.startStruct()
    a.foreach(_.result(rvb))
    rvb.endStruct()
    rvb.end()
  }

  private def loadIRIntermediate(typ: Type): (Region, Long) => Any = typ match {
    case _: TBoolean =>
      _.loadBoolean(_)
    case _: TInt32 =>
      _.loadInt(_)
    case _: TInt64 =>
      _.loadLong(_)
    case _: TFloat32 =>
      _.loadFloat(_)
    case _: TFloat64 =>
      _.loadDouble(_)
    case _ =>
      (_, off) => off
  }

  private def runStage1[IN : HailRep : TypeInfo, SCOPE0: HailRep : TypeInfo](region: Region, ir: IR, aOff: Long, scope0: Int => SCOPE0): (IR, Long) = {
    val tAgg = TAggregable(hailType[IN], Map("scope0" -> (0, hailType[SCOPE0])))

    // special arguments: region, aggregator, element, element missingness
    val (post, aggResultStruct, aggIR, rvAggs) = ExtractAggregators(ir, tAgg)

    val seqOps = {
      val fb = EmitFunctionBuilder[Region, Array[RegionValueAggregator], IN, Boolean, SCOPE0, Boolean, Unit]

      val s = Subst(aggIR, Env.empty[IR].bind("scope0", In(1, hailType[SCOPE0])))
      Emit(s, fb, 2, tAgg)

      fb.result(
        // Some(new java.io.PrintWriter(System.out))
      )()
    }

    val tArray = TArray(hailType[IN])
    val len = tArray.loadLength(region, aOff)
    val loadT = loadIRIntermediate(hailType[IN])
    for (i <- 0 until len) {
      val me = tArray.isElementMissing(region, aOff, i)
      val e =
        if (me) defaultValue(hailType[IN]).asInstanceOf[IN]
        else loadT(region, tArray.loadElement(region, aOff, i)).asInstanceOf[IN]
        seqOps(region, rvAggs, e, me, scope0(i), false)
    }

    val aggResultsOff = packageResults(region, aggResultStruct, rvAggs)

    val env = Env.empty[IR].bind("AGGR", In(0, aggResultStruct))
    val postSubst = Subst(post, env)

    (postSubst, aggResultsOff)
  }

  private def compileStage0[R: TypeInfo](ir: IR): AsmFunction3[Region, Long, Boolean, R] = {
    val fb = EmitFunctionBuilder[Region, Long, Boolean, R]
    // nb: inference is done by stage1
    Emit(ir, fb)
    fb.result()()
  }

  private def run[IN : HailRep : TypeInfo, SCOPE0 : HailRep : TypeInfo, R : TypeInfo](
    region: Region, ir: IR, aOff: Long, scope0: Int => SCOPE0): R = {
    val (post, ret) = runStage1[IN, SCOPE0](region, ir, aOff, scope0)
    val f = compileStage0[R](post)
    f(region, ret, false)
  }

  @Test
  def sum() {
    val tAgg = TAggregable(TFloat64(), Map("scope0" -> (0, TInt32())))
    val region = Region()
    val sum = run[Double, Int, Double](region,
      ApplyAggOp(AggIn(tAgg), Sum(), Seq()),
      addArray(region, (0 to 100).map(_.toDouble):_*),
      _ => 10)

    assert(sum === 5050.0)
  }

  @Test
  def sumEmpty() {
    val tAgg = TAggregable(TFloat64(), Map("scope0" -> (0, TInt32())))
    val region = Region()
    val sum = run[Double, Int, Double](region,
      ApplyAggOp(AggIn(tAgg), Sum(), Seq()),
      addArray[Double](region),
      _ => 10)

    assert(sum === 0.0)
  }

  @Test
  def sumOne() {
    val tAgg = TAggregable(TFloat64(), Map("scope0" -> (0, TInt32())))
    val region = Region()
    val sum = run[Double, Int, Double](region,
      ApplyAggOp(AggIn(tAgg), Sum(), Seq()),
      addArray[Double](region, 42.0),
      _ => 10)

    assert(sum === 42.0)
  }

  @Test
  def sumMissing() {
    val tAgg = TAggregable(TFloat64(), Map("scope0" -> (0, TInt32())))
    val region = Region()
    val sum = run[Double, Int, Double](region,
      ApplyAggOp(AggIn(tAgg), Sum(), Seq()),
      addBoxedArray[java.lang.Double](region, null, 42.0, null),
      _ => 10)
    assert(sum === 42.0)
  }

  @Test
  def sumAllMissing() {
    val tAgg = TAggregable(TFloat64(), Map("scope0" -> (0, TInt32())))
    val region = Region()
    val sum = run[Double, Int, Double](region,
      ApplyAggOp(AggIn(tAgg), Sum(), Seq()),
      addBoxedArray[java.lang.Double](region, null, null, null),
      _ => 10)
    assert(sum === 0.0)
  }

  @Test
  def usingScope1() {
    val tAgg = TAggregable(TFloat64(), Map("scope0" -> (0, TInt32())))
    val region = Region()
    val sum = run[Double, Int, Int](region,
      ApplyAggOp(AggMap(AggIn(tAgg), "x", Ref("scope0", TInt32())),
        Sum(), Seq()),
      addBoxedArray[java.lang.Double](region, null, null, null),
      10 * _)
    assert(sum == 30)
  }

  @Test
  def usingScope2() {
    val tAgg = TAggregable(TFloat64(), Map("scope0" -> (0, TInt32())))
    val region = Region()
    val sum = run[Double, Int, Int](region,
      ApplyAggOp(AggMap(AggIn(tAgg), "x", Ref("scope0", TInt32())),
        Sum(), Seq()),
      addBoxedArray[java.lang.Double](region, 1.0, 2.0, null),
      10 * _)
    assert(sum == 30)
  }

  @Test
  def usingScope3() {
    val tAgg = TAggregable(TFloat64(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val sum = run[Double, Double, Double](region,
      ApplyAggOp(AggMap(AggIn(tAgg), "x", ApplyBinaryPrimOp(Multiply(), Ref("scope0", TFloat64()), Ref("x", TFloat64()))),
        Sum(), Seq()),
      addBoxedArray[java.lang.Double](region, 1.0, 10.0, null),
      _ => 10.0)
    assert(sum === 110.0)
  }

  @Test
  def filter1() {
    val tAgg = TAggregable(TFloat64(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val ir =
      ApplyAggOp(
        AggMap(
          AggFilter(AggIn(tAgg), "x", ApplyBinaryPrimOp
            (NEQ(), Ref("x", TFloat64()), F64(10.0))), "x",
          ApplyBinaryPrimOp(Multiply(), Ref("scope0", TFloat64()), Ref("x", TFloat64()))),
        Sum(), Seq())
    val sum = run[Double, Double, Double](region,
      ir,
      addBoxedArray[java.lang.Double](region, 1.0, 10.0, null),
      _ => 10.0)
    assert(sum === 10.0)
  }

  @Test
  def filter2() {
    val tAgg = TAggregable(TFloat64(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val ir =
      ApplyAggOp(
        AggMap(
          AggFilter(AggIn(tAgg),
            "x",
            ApplyBinaryPrimOp(EQ(), Ref("x", TFloat64()), F64(10.0))),
          "x",
          ApplyBinaryPrimOp(Multiply(), Ref("scope0", TFloat64()), Ref("x", TFloat64()))),
        Sum(), Seq())
    val sum = run[Double, Double, Double](region,
      ir,
      addBoxedArray[java.lang.Double](region, 1.0, 10.0, null),
      _ => 10.0)
    assert(sum === 100.0)
  }

  @Test
  def filter3() {
    val tAgg = TAggregable(TFloat64(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val ir =
      ApplyAggOp(
        AggFilter(
          AggMap(AggIn(tAgg),
            "x",
            ApplyBinaryPrimOp(Multiply(), Ref("scope0", TFloat64()), Ref("x", TFloat64()))),
          "x",
          ApplyBinaryPrimOp(EQ(), Ref("x", TFloat64()), F64(100.0))),
        Sum(), Seq())
    val sum = run[Double, Double, Double](region,
      ir,
      addBoxedArray[java.lang.Double](region, 1.0, 10.0, null),
      _ => 10.0)
    assert(sum === 100.0)
  }

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
        Sum(), Seq())
    val sum = run[Double, Double, Double](region,
      ir,
      addBoxedArray[java.lang.Double](region, 1.0, 10.0, null),
      _ => 10.0)
    assert(sum === 110.0 + 200.0 + 100.0)
  }

  @Test
  def fraction() {
    val tAgg = TAggregable(TFloat64(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val ir =
      ApplyAggOp(
        AggMap(
          AggMap(AggIn(tAgg),
            "x",
            ApplyBinaryPrimOp(Multiply(), Ref("scope0", TFloat64()), Ref("x", TFloat64()))),
          "x",
          ApplyBinaryPrimOp(EQ(), Ref("x", TFloat64()), F64(100.0))),
        Fraction(), Seq())
    val fraction = run[Double, Double, Double](region,
      ir,
      addBoxedArray[java.lang.Double](region, 1.0, 10.0, null),
      _ => 10.0)
    assert(fraction === 1.0/3.0)
  }

  @Test
  def fractionNaDoesNotContribute1() {
    val tAgg = TAggregable(TBoolean(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val ir =
      ApplyAggOp(AggIn(tAgg), Fraction(), Seq())
    val fraction = run[Boolean, Double, Double](region,
      ir,
      addBoxedArray[java.lang.Boolean](region, true, false, null),
      _ => 10.0)
    assert(fraction === 1.0/3.0)
  }

  @Test
  def fractionNaDoesNotContribute2() {
    val tAgg = TAggregable(TBoolean(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val ir =
      ApplyAggOp(AggIn(tAgg), Fraction(), Seq())
    val fraction = run[Boolean, Double, Double](region,
      ir,
      addBoxedArray[java.lang.Boolean](region, false, false, null),
      _ => 10.0)
    assert(fraction === 0.0/3.0)
  }

  @Test
  def fractionNaDoesNotContribute3() {
    val tAgg = TAggregable(TBoolean(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val ir =
      ApplyAggOp(AggIn(tAgg), Fraction(), Seq())
    val fraction = run[Boolean, Double, Double](region,
      ir,
      addBoxedArray[java.lang.Boolean](region, true, true, null),
      _ => 10.0)
    assert(fraction === 2.0/3.0)
  }

  @Test
  def collectBoxedBoolean() {
    val tAgg = TAggregable(TBoolean(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val expected = Array[java.lang.Boolean](true, true, null)
    val ir =
      ApplyAggOp(AggIn(tAgg), Collect(), Seq())
    val aOff = run[Boolean, Double, Long](region,
      ir,
      addBoxedArray(region, expected: _*),
      _ => 10.0)

    val actual = loadArray[java.lang.Boolean](region, aOff)
    assert(actual === expected)
  }

  @Test
  def collectBoxedInt() {
    val tAgg = TAggregable(TInt32(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val expected = Array[java.lang.Integer](5, 10, null)
    val ir =
      ApplyAggOp(AggIn(tAgg), Collect(), Seq())
    val aOff = run[Int, Double, Long](region,
      ir,
      addBoxedArray(region, expected: _*),
      _ => 10.0)

    val actual = loadArray[java.lang.Integer](region, aOff)
    assert(actual === expected)
  }

  @Test
  def collectInt() {
    val tAgg = TAggregable(TInt32(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val expected = Array[Int](5, 10, -5)
    val ir =
      ApplyAggOp(AggIn(tAgg), Collect(), Seq())
    val aOff = run[Int, Double, Long](region,
      ir,
      addBoxedArray(region, expected: _*),
      _ => 10.0)

    val actual = loadArray[Int](region, aOff)
    assert(actual === expected)
  }

  @Test
  def mapCollectInt() {
    val tAgg = TAggregable(TInt32(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val input = Array[Int](5, 10, -5)
    val expected = input.map(x => x * x).toArray
    val ir =
      ApplyAggOp(
        AggMap(
          AggIn(tAgg),
          "x",
          ApplyBinaryPrimOp(
            Multiply(),
            Ref("x", TInt32()),
            Ref("x", TInt32()))),
        Collect(), Seq())
    val aOff = run[Int, Double, Long](region,
      ir,
      addBoxedArray(region, input: _*),
      _ => 10.0)

    val actual = loadArray[Int](region, aOff)
    assert(actual === expected)
  }

  @Test
  def maxBoolean1() {
    val tAgg = TAggregable(TBoolean(), Map("scope0" -> (0, TFloat64())))
    val region = Region()
    val input = Array[Boolean](true, false, false)
    val ir =
      ApplyAggOp(AggIn(tAgg), Max(), Seq())
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
      ApplyAggOp(AggIn(tAgg), Max(), Seq())
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
      ApplyAggOp(AggIn(tAgg), Max(), Seq())
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
      ApplyAggOp(AggIn(tAgg), Max(), Seq())
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
      ApplyAggOp(AggIn(tAgg), Max(), Seq())
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
      ApplyAggOp(AggIn(tAgg), Max(), Seq())
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
      ApplyAggOp(AggIn(tAgg), Take(), Seq(I32(2)))
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
      ApplyAggOp(AggIn(tAgg), Take(), Seq(I32(2)))
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
      ApplyAggOp(AggIn(tAgg), Take(), Seq(I32(2)))
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
      ApplyAggOp(AggIn(tAgg), Histogram(), Seq(F64(0.0), F64(10.0), I32(5)))
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
}
