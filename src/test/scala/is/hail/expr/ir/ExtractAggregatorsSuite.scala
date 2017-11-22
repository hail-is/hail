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

  @Test
  def sum() {
    val tAgg = TAggregable(TFloat64(), Map("foo" -> (0, TInt32())))
    val ir: IR = AggSum(AggIn(tAgg))
    Infer(ir)
    val (post, t, f) = ExtractAggregators(ir, tAgg)
    println(post)
    val region = MemoryBuffer()
    val tArray = TArray(TFloat64())
    val aoff = addArray(region, (0 to 100).map(_.toDouble).toArray)
    val agg = new ExtractAggregators.Aggregable {
      def aggregate(
        zero: (MemoryBuffer) => Long,
        seq: (MemoryBuffer, Long, Boolean, Long, Boolean) => Long,
        comb: (MemoryBuffer, Long, Boolean, Long, Boolean) => Long): Long = {
        var i = 0
        var z = zero(region)
        while (i < 100) {
          // NB: the struct containing the aggregation intermediates is never
          // missing
          z = seq(region, z, false, tArray.loadElement(region, aoff, i), !tArray.isElementDefined(region, aoff, i))
          i += 1
        }
        z
      }
    }
    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Double]
    Compile(post, fb)
    printRegion(region, "before")
    val outOff = f(region, agg)
    printRegion(region, "after")
    assert(fb.result()()(region, outOff, false) === 5050.0)
    assert(t.isFieldDefined(region, outOff, t.fieldIdx("0")))
    assert(region.loadDouble(t.loadField(region, outOff, t.fieldIdx("0"))) === 5050.0)
  }

  def printRegion(region: MemoryBuffer, string: String) {
    println(string)
    val size = region.size
    println("Region size: " + size.toString)
    val bytes = region.loadBytes(0, size.toInt)
    println("Array: ")
    var j = 0
    for (i <- bytes) {
      j += 1
      printf("%02X", i)
      if (j % 32 == 0) {
        print('\n')
      } else {
        print(' ')
      }
    }
    print('\n')
  }
}
