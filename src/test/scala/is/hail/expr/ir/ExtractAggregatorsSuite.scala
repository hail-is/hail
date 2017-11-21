package is.hail.methods.ir

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.check.{Gen, Parameters, Prop}
import is.hail.expr.{TAggregable, TArray, TFloat64, TInt32}
import is.hail.expr.ir._
import org.testng.annotations.Test
import org.scalatest._
import Matchers._

class ExtractAggregatorsSuite {

  private def addArray(mb: MemoryBuffer, a: Array[Double]): Long = {
    val rvb = new RegionValueBuilder(mb)
    rvb.start(TArray(TFloat64()))
    rvb.startArray(a.length)
    a.foreach(rvb.addDouble(_))
    rvb.endArray()
    rvb.end()
  }

  private def addBoxedArray(mb: MemoryBuffer, a: Array[java.lang.Double]): Long = {
    val rvb = new RegionValueBuilder(mb)
    rvb.start(TArray(TFloat64()))
    rvb.startArray(a.length)
    a.foreach { e =>
      if (e == null)
        rvb.setMissing()
      else
        rvb.addDouble(e)
    }
    rvb.endArray()
    rvb.end()
  }

  def doit(ir: IR, fb: FunctionBuilder2) {
    Infer(ir)
    println(ir)
    Compile(ir, fb)
  }

  @Test
  def sum() {
    val tAgg = TAggregable(TFloat64(), Map("foo" -> (0, TInt32())))
    val ir: IR = AggSum(AggIn(tAgg))
    val (post, t, f) = ExtractAggregators(ir, tAgg)
    println(post)
    val region = MemoryBuffer()
    val tArray = TArray(TFloat64())
    val rvb = new RegionValueBuilder()
    rvb.set(region)
    rvb.start(tArray)
    rvb.startArray(100)
    (0 to 100).foreach(rvb.addDouble(_))
    rvb.endArray()
    val aoff = rvb.end()
    val agg = new ExtractAggregators.Aggregable {
      def aggregate(
        zero: (MemoryBuffer) => Long,
        seq: (MemoryBuffer, Long, Long) => Long,
        comb: (MemoryBuffer, Long, Long) => Long): Long = {
        var i = 0
        var z = zero(region)
        while (i < 100) {
          z = seq(region, z, tArray.loadElement(region, aoff, i))
        }
        z
      }
    }
    assert(region.loadFloat(f(region, agg)) === 5050)
  }
}
