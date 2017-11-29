package is.hail.methods.ir

import is.hail.annotations._
import ScalaToRegionValue._
import is.hail.asm4s._
import is.hail.check.{Gen, Parameters, Prop}
import is.hail.expr.{TAggregable, TArray, TFloat64, TInt32, TStruct}
import is.hail.expr.ir._
import org.testng.annotations.Test
import org.scalatest._
import Matchers._

class ExtractAggregatorsSuite {

  @Test
  def sum() {
    val tAgg = TAggregable(TFloat64(), Map("foo" -> (0, TInt32())))
    val region = MemoryBuffer()
    val aOff = addArray(region, (0 to 100).map(_.toDouble).toArray)

    val ir: IR = AggSum(AggIn(tAgg))
    val (post, runAggregations) = RunAggregators.onArray(ir, tAgg, aOff, region.appendInt(10), false)

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Double]
    Compile(post, fb)

    // out is never missing
    assert(fb.result()()(region, runAggregations(region), false) === 5050.0)
  }


  @Test
  def sumEmpty() {
    val tAgg = TAggregable(TFloat64(), Map("foo" -> (0, TInt32())))
    val region = MemoryBuffer()
    val aOff = addArray(region, Array())

    val ir: IR = AggSum(AggIn(tAgg))
    val (post, runAggregations) = RunAggregators.onArray(ir, tAgg, aOff, region.appendInt(10), false)

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Double]
    Compile(post, fb)

    // out is never missing
    assert(fb.result()()(region, runAggregations(region), false) === 0.0)
  }

  @Test
  def sumOne() {
    val tAgg = TAggregable(TFloat64(), Map("foo" -> (0, TInt32())))
    val region = MemoryBuffer()
    val aOff = addArray(region, Array(42.0))

    val ir: IR = AggSum(AggIn(tAgg))
    val (post, runAggregations) = RunAggregators.onArray(ir, tAgg, aOff, region.appendInt(10), false)

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Double]
    Compile(post, fb)

    // out is never missing
    assert(fb.result()()(region, runAggregations(region), false) === 42.0)
  }

  @Test
  def sumMissing() {
    val tAgg = TAggregable(TFloat64(), Map("foo" -> (0, TInt32())))
    val region = MemoryBuffer()
    val aOff = addBoxedArray(region, Array[java.lang.Double](null, 42.0, null))

    val ir: IR = AggSum(AggIn(tAgg))
    val (post, runAggregations) = RunAggregators.onArray(ir, tAgg, aOff, region.appendInt(10), false)

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Double]
    Compile(post, fb)

    // out is never missing
    assert(fb.result()()(region, runAggregations(region), false) === 42.0)
  }

  @Test
  def sumAllMissing() {
    val tAgg = TAggregable(TFloat64(), Map("foo" -> (0, TInt32())))
    val region = MemoryBuffer()
    val aOff = addBoxedArray(region, Array[java.lang.Double](null, null, null))

    val ir: IR = AggSum(AggIn(tAgg))
    val (post, runAggregations) = RunAggregators.onArray(ir, tAgg, aOff, region.appendInt(10), false)

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Double]
    Compile(post, fb)

    // out is never missing
    assert(fb.result()()(region, runAggregations(region), false) === 0.0)
  }

  @Test
  def usingScope1() {
    val tAgg = TAggregable(TFloat64(), Map("foo" -> (0, TInt32())))
    val region = MemoryBuffer()
    val aOff = addBoxedArray(region, Array[java.lang.Double](null, null, null))

    val ir: IR = AggSum(AggMap(AggIn(tAgg), "x",
      Ref("foo"),
      TAggregable(TInt32(), Map("foo" -> (0, TInt32())))))
    val (post, runAggregations) = RunAggregators.onArray(ir, tAgg, aOff, region.appendInt(10), false)

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Int]
    Compile(post, fb)

    // out is never missing
    assert(fb.result()()(region, runAggregations(region), false) === 30)
  }

  @Test
  def usingScope2() {
    val tAgg = TAggregable(TFloat64(), Map("foo" -> (0, TInt32())))
    val region = MemoryBuffer()
    val aOff = addBoxedArray(region, Array[java.lang.Double](1.0, 2.0, null))

    val ir: IR = AggSum(AggMap(AggIn(tAgg), "x",
      Ref("foo"),
      TAggregable(TInt32(), Map("foo" -> (0, TInt32())))))
    val (post, runAggregations) = RunAggregators.onArray(ir, tAgg, aOff, region.appendInt(10), false)

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Int]
    Compile(post, fb)

    // out is never missing
    assert(fb.result()()(region, runAggregations(region), false) === 30)
  }

  @Test
  def usingScope3() {
    val tAgg = TAggregable(TFloat64(), Map("foo" -> (0, TFloat64())))
    val region = MemoryBuffer()
    val aOff = addBoxedArray(region, Array[java.lang.Double](1.0, 10.0, null))

    val ir: IR = AggSum(AggMap(AggIn(tAgg), "x",
      ApplyBinaryPrimOp(Multiply(), Ref("foo"), Ref("x")),
      TAggregable(TInt32(), Map("foo" -> (0, TInt32())))))
    val (post, runAggregations) = RunAggregators.onArray(ir, tAgg, aOff, region.appendDouble(10.0), false)

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Double]
    Compile(post, fb)

    // out is never missing
    assert(fb.result()()(region, runAggregations(region), false) === 110.0)
  }
}
