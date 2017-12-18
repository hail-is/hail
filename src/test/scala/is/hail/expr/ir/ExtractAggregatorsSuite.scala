package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.annotations._
import ScalaToRegionValue._
import is.hail.annotations.aggregators._
import is.hail.check._
import is.hail.expr.{TAggregable, TArray, TFloat64, TInt32, TStruct, RegionValueAggregator, hailType, HailRep, TBoolean, TFloat32, TInt64, Type}
import is.hail.expr.ir._
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
    t.fields.map(_.typ).zip(a).foreach { case (t, agg) =>
      rvb.addRegionValue(t, region, agg.result(region))
    }
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

    val aggFb = FunctionBuilder.functionBuilder[Region, Array[RegionValueAggregator], IN, Boolean, SCOPE0, Boolean, Unit]
    Infer(ir, Some(tAgg))
    val (post, aggResultStruct, aggregators) = ExtractAggregators(ir, tAgg, aggFb)
    val seqOp = aggFb.result(Some(new java.io.PrintWriter(System.out)))()

    val tArray = TArray(hailType[IN])
    val len = tArray.loadLength(region, aOff)
    val loadT = loadIRIntermediate(hailType[IN])
    for (i <- 0 until len) {
      val me = tArray.isElementMissing(region, aOff, i)
      val e =
        if (me) defaultValue(hailType[IN]).asInstanceOf[IN]
        else loadT(region, tArray.loadElement(region, aOff, i)).asInstanceOf[IN]
      seqOp(region, aggregators,
        e,
        me,
        scope0(i),
        false)
    }

    val aggResultsOff = packageResults(region, aggResultStruct, aggregators)
    (post, aggResultsOff)
  }

  private def compileStage0[R: TypeInfo](ir: IR): AsmFunction3[Region, Long, Boolean, R] = {
    val fb = FunctionBuilder.functionBuilder[Region, Long, Boolean, R]
    // nb: inference is done by stage1
    Emit(ir, fb)
    fb.result()()
  }

  private def run[IN : HailRep : TypeInfo, SCOPE0 : HailRep : TypeInfo, R : TypeInfo](region: Region, ir: IR, aOff: Long, scope0: Int => SCOPE0): R = {
    val (post, ret) = runStage1[IN, SCOPE0](region, ir, aOff, scope0)
    val f = compileStage0[R](post)
    f(region, ret, false)
  }

  @Test
  def sum() {
    val region = Region()
    val sum = run[Double, Int, Double](region,
      ApplyAggNullaryOp(AggIn(), Sum()),
      addArray(region, (0 to 100).map(_.toDouble):_*),
      _ => 10)

    assert(sum === 5050.0)
  }

  @Test
  def sumEmpty() {
    val region = Region()
    val sum = run[Double, Int, Double](region,
      ApplyAggNullaryOp(AggIn(), Sum()),
      addArray[Double](region),
      _ => 10)

    assert(sum === 0.0)
  }

  @Test
  def sumOne() {
    val region = Region()
    val sum = run[Double, Int, Double](region,
      ApplyAggNullaryOp(AggIn(), Sum()),
      addArray[Double](region, 42.0),
      _ => 10)

    assert(sum === 42.0)
  }

  @Test
  def sumMissing() {
    val region = Region()
    val sum = run[Double, Int, Double](region,
      ApplyAggNullaryOp(AggIn(), Sum()),
      addBoxedArray[java.lang.Double](region, null, 42.0, null),
      _ => 10)
    assert(sum === 42.0)
  }

  @Test
  def sumAllMissing() {
    val region = Region()
    val sum = run[Double, Int, Double](region,
      ApplyAggNullaryOp(AggIn(), Sum()),
      addBoxedArray[java.lang.Double](region, null, null, null),
      _ => 10)
    assert(sum === 0.0)
  }

  @Test
  def usingScope1() {
    val region = Region()
    val sum = run[Double, Int, Int](region,
      ApplyAggNullaryOp(AggMap(AggIn(), "x", Ref("scope0")),
        Sum()),
      addBoxedArray[java.lang.Double](region, null, null, null),
      10 * _)
    assert(sum == 30)
  }

  @Test
  def usingScope2() {
    val region = Region()
    val sum = run[Double, Int, Int](region,
      ApplyAggNullaryOp(AggMap(AggIn(), "x", Ref("scope0")),
        Sum()),
      addBoxedArray[java.lang.Double](region, 1.0, 2.0, null),
      10 * _)
    assert(sum == 30)
  }

  @Test
  def usingScope3() {
    val region = Region()
    val sum = run[Double, Double, Double](region,
      ApplyAggNullaryOp(AggMap(AggIn(), "x", ApplyBinaryPrimOp(Multiply(), Ref("scope0"), Ref("x"))),
        Sum()),
      addBoxedArray[java.lang.Double](region, 1.0, 10.0, null),
      _ => 10.0)
    assert(sum === 110.0)
  }

  @Test
  def filter1() {
    val region = Region()
    val ir =
      ApplyAggNullaryOp(
        AggMap(
          AggFilter(AggIn(), "x", ApplyBinaryPrimOp
            (NEQ(), Ref("x"), F64(10.0))), "x",
          ApplyBinaryPrimOp(Multiply(), Ref("scope0"), Ref("x"))),
        Sum())
    val sum = run[Double, Double, Double](region,
      ir,
      addBoxedArray[java.lang.Double](region, 1.0, 10.0, null),
      _ => 10.0)
    assert(sum === 10.0)
  }

  @Test
  def filter2() {
    val region = Region()
    val ir =
      ApplyAggNullaryOp(
        AggMap(
          AggFilter(AggIn(),
            "x",
            ApplyBinaryPrimOp(EQ(), Ref("x"), F64(10.0))),
          "x",
          ApplyBinaryPrimOp(Multiply(), Ref("scope0"), Ref("x"))),
        Sum())
    val sum = run[Double, Double, Double](region,
      ir,
      addBoxedArray[java.lang.Double](region, 1.0, 10.0, null),
      _ => 10.0)
    assert(sum === 100.0)
  }

  @Test
  def filter3() {
    val region = Region()
    val ir =
      ApplyAggNullaryOp(
        AggFilter(
          AggMap(AggIn(),
            "x",
            ApplyBinaryPrimOp(Multiply(), Ref("scope0"), Ref("x"))),
          "x",
          ApplyBinaryPrimOp(EQ(), Ref("x"), F64(100.0))),
        Sum())
    val sum = run[Double, Double, Double](region,
      ir,
      addBoxedArray[java.lang.Double](region, 1.0, 10.0, null),
      _ => 10.0)
    assert(sum === 100.0)
  }

  @Test
  def flatMap() {
    val region = Region()
    val ir =
      ApplyAggNullaryOp(
        AggFlatMap(
          AggMap(AggIn(),
            "x",
            ApplyBinaryPrimOp(Multiply(), Ref("scope0"), Ref("x"))),
          "x",
          MakeArray(Array(F64(100.0), Ref("x")))),
        Sum())
    val sum = run[Double, Double, Double](region,
      ir,
      addBoxedArray[java.lang.Double](region, 1.0, 10.0, null),
      _ => 10.0)
    assert(sum === 110.0 + 200.0 + 100.0)
  }

  @Test
  def fraction() {
    val region = Region()
    val ir =
      ApplyAggNullaryOp(
        AggMap(
          AggMap(AggIn(),
            "x",
            ApplyBinaryPrimOp(Multiply(), Ref("scope0"), Ref("x"))),
          "x",
          ApplyBinaryPrimOp(EQ(), Ref("x"), F64(100.0))),
        Fraction())
    val fraction = run[Double, Double, Double](region,
      ir,
      addBoxedArray[java.lang.Double](region, 1.0, 10.0, null),
      _ => 10.0)
    assert(fraction === 1.0/3.0)
  }

  @Test
  def fractionNaDoesNotContribute1() {
    val region = Region()
    val ir =
      ApplyAggNullaryOp(AggIn(), Fraction())
    val fraction = run[Boolean, Double, Double](region,
      ir,
      addBoxedArray[java.lang.Boolean](region, true, false, null),
      _ => 10.0)
    assert(fraction === 1.0/3.0)
  }

  @Test
  def fractionNaDoesNotContribute2() {
    val region = Region()
    val ir =
      ApplyAggNullaryOp(AggIn(), Fraction())
    val fraction = run[Boolean, Double, Double](region,
      ir,
      addBoxedArray[java.lang.Boolean](region, false, false, null),
      _ => 10.0)
    assert(fraction === 0.0/3.0)
  }

  @Test
  def fractionNaDoesNotContribute3() {
    val region = Region()
    val ir =
      ApplyAggNullaryOp(AggIn(), Fraction())
    val fraction = run[Boolean, Double, Double](region,
      ir,
      addBoxedArray[java.lang.Boolean](region, true, true, null),
      _ => 10.0)
    assert(fraction === 2.0/3.0)
  }

  @Test
  def collectBoxedBoolean() {
    val region = Region()
    val expected = Array[java.lang.Boolean](true, true, null)
    val ir =
      ApplyAggNullaryOp(AggIn(), Collect())
    val aOff = run[Boolean, Double, Long](region,
      ir,
      addBoxedArray(region, expected: _*),
      _ => 10.0)

    val actual = RegionValueToScala.loadArray[java.lang.Boolean](region, aOff)
    assert(actual === expected)
  }

  @Test
  def collectBoxedInt() {
    val region = Region()
    val expected = Array[java.lang.Integer](5, 10, null)
    val ir =
      ApplyAggNullaryOp(AggIn(), Collect())
    val aOff = run[Int, Double, Long](region,
      ir,
      addBoxedArray(region, expected: _*),
      _ => 10.0)

    val actual = RegionValueToScala.loadArray[java.lang.Integer](region, aOff)
    assert(actual === expected)
  }

  @Test
  def collectInt() {
    val region = Region()
    val expected = Array[Int](5, 10, -5)
    val ir =
      ApplyAggNullaryOp(AggIn(), Collect())
    val aOff = run[Int, Double, Long](region,
      ir,
      addBoxedArray(region, expected: _*),
      _ => 10.0)

    val actual = RegionValueToScala.loadArray[Int](region, aOff)
    assert(actual === expected)
  }

  @Test
  def mapCollectInt() {
    val region = Region()
    val input = Array[Int](5, 10, -5)
    val expected = input.map(x => x * x).toArray
    val ir =
      ApplyAggNullaryOp(
        AggMap(
          AggIn(),
          "x",
          ApplyBinaryPrimOp(
            Multiply(),
            Ref("x"),
            Ref("x"))),
        Collect())
    val aOff = run[Int, Double, Long](region,
      ir,
      addBoxedArray(region, input: _*),
      _ => 10.0)

    val actual = RegionValueToScala.loadArray[Int](region, aOff)
    assert(actual === expected)
  }

  @Test
  def maxBoolean1() {
    val region = Region()
    val input = Array[Boolean](true, false, false)
    val ir =
      ApplyAggNullaryOp(AggIn(), Max())
    val actual = run[Boolean, Double, Boolean](region,
      ir,
      addBoxedArray(region, input: _*),
      _ => 10.0)

    assert(actual === true)
  }

  @Test
  def maxBoolean2() {
    val region = Region()
    val input = Array[Boolean](false, false, false)
    val ir =
      ApplyAggNullaryOp(AggIn(), Max())
    val actual = run[Boolean, Double, Boolean](region,
      ir,
      addBoxedArray(region, input: _*),
      _ => 10.0)

    assert(actual === false)
  }

  @Test
  def maxInt() {
    val region = Region()
    val input = Array[Int](5, 10, -5)
    val ir =
      ApplyAggNullaryOp(AggIn(), Max())
    val actual = run[Int, Double, Int](region,
      ir,
      addBoxedArray(region, input: _*),
      _ => 10.0)

    assert(actual === 10)
  }

  @Test
  def maxLong() {
    val region = Region()
    val input = Array[Long](5L, 10L, -5L)
    val ir =
      ApplyAggNullaryOp(AggIn(), Max())
    val actual = run[Long, Double, Long](region,
      ir,
      addBoxedArray(region, input: _*),
      _ => 10.0)

    assert(actual === 10L)
  }

  @Test
  def maxFloat() {
    val region = Region()
    val input = Array[Float](5.0f, 10.0f, -5.0f)
    val ir =
      ApplyAggNullaryOp(AggIn(), Max())
    val actual = run[Float, Double, Float](region,
      ir,
      addBoxedArray(region, input: _*),
      _ => 10.0)

    assert(actual === 10.0f)
  }

  @Test
  def maxDouble() {
    val region = Region()
    val input = Array[Double](5.0, 10.0, -5.0)
    val ir =
      ApplyAggNullaryOp(AggIn(), Max())
    val actual = run[Double, Double, Double](region,
      ir,
      addBoxedArray(region, input: _*),
      _ => 10.0)

    assert(actual === 10.0)
  }

  @Test
  def takeInt() {
    val region = Region()
    val input = Array[Int](5, 10, -5)
    val expected = input.take(2)
    val ir =
      ApplyAggUnaryOp(AggIn(), Take(), I32(2))
    val aOff = run[Int, Double, Long](region,
      ir,
      addBoxedArray(region, input: _*),
      _ => 10.0)

    assert(RegionValueToScala.loadArray[Int](region, aOff) === expected)
  }

  @Test
  def takeDouble() {
    val region = Region()
    val input = Array[Double](5.0, 10.0, -5.0)
    val expected = input.take(2)
    val ir =
      ApplyAggUnaryOp(AggIn(), Take(), I32(2))
    val aOff = run[Double, Double, Long](region,
      ir,
      addBoxedArray(region, input: _*),
      _ => 10.0)

    assert(RegionValueToScala.loadArray[Double](region, aOff) === expected)
  }

  @Test
  def takeDoubleWithMissingness() {
    val region = Region()
    val input = Array[java.lang.Double](5.0, null, -5.0)
    val expected = input.take(2)
    val ir =
      ApplyAggUnaryOp(AggIn(), Take(), I32(2))
    val aOff = run[Double, Double, Long](region,
      ir,
      addBoxedArray(region, input: _*),
      _ => 10.0)

    assert(RegionValueToScala.loadArray[java.lang.Double](region, aOff) === expected)
  }

  @Test
  def histogram() {
    val region = Region()
    val input = Array[java.lang.Double](5.0, null, -5.0, 1.0, 15.0, 1.0, 17.0, 1.5)
    val expected = input.take(2)
    val ir =
      ApplyAggTernaryOp(AggIn(), Histogram(), F64(0.0), F64(10.0), I32(5))
    val hOff = run[Double, Double, Long](region,
      ir,
      addBoxedArray(region, input: _*),
      _ => 10.0)

    val t = RegionValueHistogramAggregator.typ
    val binEdges = t.fieldIdx("binEdges")
    val binFrequencies = t.fieldIdx("binFrequencies")
    val nLess = t.fieldIdx("nLess")
    val nGreater = t.fieldIdx("nGreater")
    assert(t.isFieldDefined(region, hOff, binEdges))
    assert(t.isFieldDefined(region, hOff, binFrequencies))
    assert(t.isFieldDefined(region, hOff, nLess))
    assert(t.isFieldDefined(region, hOff, nGreater))
    val binEdgeArray = RegionValueToScala.loadArray[Double](
      region, t.loadField(region, hOff, binEdges))
    assert(binEdgeArray === Array(0.0, 2.0, 4.0, 6.0, 8.0, 10.0))
    val binFrequenciesArray = RegionValueToScala.loadArray[Long](
      region, t.loadField(region, hOff, binFrequencies))
    assert(binFrequenciesArray === Array[Long](3L, 0L, 1L, 0L, 0L))
    assert(region.loadLong(t.loadField(region, hOff, nLess)) === 1L)
    assert(region.loadLong(t.loadField(region, hOff, nGreater)) === 2L)
  }
}
