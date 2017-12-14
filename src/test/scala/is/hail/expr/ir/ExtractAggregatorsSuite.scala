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

  private def runStage1[T : HailRep : TypeInfo, U: HailRep : TypeInfo](region: Region, ir: IR, aOff: Long, scope0: Int => U): (IR, Long) = {
    val tAgg = TAggregable(hailType[T], Map("scope0" -> (0, hailType[U])))

    val aggFb = FunctionBuilder.functionBuilder[Region, Array[RegionValueAggregator], T, Boolean, U, Boolean, Unit]
    Infer(ir, Some(tAgg))
    val (post, aggResultStruct, aggregators) = ExtractAggregators(ir, tAgg, aggFb)
    val seqOp = aggFb.result()()

    val tArray = TArray(hailType[T])
    val len = tArray.loadLength(region, aOff)
    val loadT = loadIRIntermediate(hailType[T])
    for (i <- 0 until len) {
      val me = tArray.isElementMissing(region, aOff, i)
      val e =
        if (me) defaultValue(hailType[T]).asInstanceOf[T]
        else loadT(region, tArray.loadElement(region, aOff, i)).asInstanceOf[T]
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

  private def run[T : HailRep : TypeInfo, U : HailRep : TypeInfo, R : TypeInfo](region: Region, ir: IR, aOff: Long, scope0: Int => U): R = {
    val (post, ret) = runStage1[T, U](region, ir, aOff, scope0)
    val f = compileStage0[R](post)
    f(region, ret, false)
  }

  @Test
  def sum() {
    val region = Region()
    val sum = run[Double, Int, Double](region,
      AggSum(AggIn()),
      addArray(region, (0 to 100).map(_.toDouble):_*),
      _ => 10)

    assert(sum === 5050.0)
  }

  @Test
  def sumEmpty() {
    val region = Region()
    val sum = run[Double, Int, Double](region,
      AggSum(AggIn()),
      addArray[Double](region),
      _ => 10)

    assert(sum === 0.0)
  }

  @Test
  def sumOne() {
    val region = Region()
    val sum = run[Double, Int, Double](region,
      AggSum(AggIn()),
      addArray[Double](region, 42.0),
      _ => 10)

    assert(sum === 42.0)
  }

  @Test
  def sumMissing() {
    val region = Region()
    val sum = run[Double, Int, Double](region,
      AggSum(AggIn()),
      addBoxedArray[java.lang.Double](region, null, 42.0, null),
      _ => 10)
    assert(sum === 42.0)
  }

  @Test
  def sumAllMissing() {
    val region = Region()
    val sum = run[Double, Int, Double](region,
      AggSum(AggIn()),
      addBoxedArray[java.lang.Double](region, null, null, null),
      _ => 10)
    assert(sum === 0.0)
  }

  @Test
  def usingScope1() {
    val region = Region()
    val sum = run[Double, Int, Int](region,
      AggSum(AggMap(AggIn(), "x", Ref("scope0"))),
      addBoxedArray[java.lang.Double](region, null, null, null),
      10 * _)
    assert(sum == 30)
  }

  @Test
  def usingScope2() {
    val region = Region()
    val sum = run[Double, Int, Int](region,
      AggSum(AggMap(AggIn(), "x", Ref("scope0"))),
      addBoxedArray[java.lang.Double](region, 1.0, 2.0, null),
      10 * _)
    assert(sum == 30)
  }

  @Test
  def usingScope3() {
    val region = Region()
    val sum = run[Double, Double, Double](region,
      AggSum(AggMap(AggIn(), "x", ApplyBinaryPrimOp(Multiply(), Ref("scope0"), Ref("x")))),
      addBoxedArray[java.lang.Double](region, 1.0, 10.0, null),
      _ => 10.0)
    assert(sum === 110.0)
  }

  @Test
  def filter1() {
    val region = Region()
    val ir =
      AggSum(
        AggMap(
          AggFilter(AggIn(), "x", ApplyBinaryPrimOp
            (NEQ(), Ref("x"), F64(10.0))), "x",
          ApplyBinaryPrimOp(Multiply(), Ref("scope0"), Ref("x"))))
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
      AggSum(
        AggMap(
          AggFilter(AggIn(),
            "x",
            ApplyBinaryPrimOp(EQ(), Ref("x"), F64(10.0))),
          "x",
          ApplyBinaryPrimOp(Multiply(), Ref("scope0"), Ref("x"))))
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
      AggSum(
        AggFilter(
          AggMap(AggIn(),
            "x",
            ApplyBinaryPrimOp(Multiply(), Ref("scope0"), Ref("x"))),
          "x",
          ApplyBinaryPrimOp(EQ(), Ref("x"), F64(100.0))))
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
      AggSum(
        AggFlatMap(
          AggMap(AggIn(),
            "x",
            ApplyBinaryPrimOp(Multiply(), Ref("scope0"), Ref("x"))),
          "x",
          MakeArray(Array(F64(100.0), Ref("x")))))
    val sum = run[Double, Double, Double](region,
      ir,
      addBoxedArray[java.lang.Double](region, 1.0, 10.0, null),
      _ => 10.0)
    assert(sum === 110.0 + 200.0 + 100.0)
  }
}
