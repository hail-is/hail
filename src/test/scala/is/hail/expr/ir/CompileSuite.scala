package is.hail.methods.ir

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.check.{Gen, Parameters, Prop}
import is.hail.expr.{TArray, TFloat64, TInt32}
import is.hail.expr.ir.IR.seq
import is.hail.expr.ir._
import org.testng.annotations.Test
import org.scalatest._
import Matchers._

class CompileSuite {

  private def addArray(mb: MemoryBuffer, a: Array[Double]): Long = {
    val rvb = new RegionValueBuilder(mb)
    rvb.start(TArray(TFloat64))
    rvb.startArray(a.length)
    a.foreach(rvb.addDouble(_))
    rvb.endArray()
    rvb.end()
  }

  @Test
  def mean() {
    val meanIr = Let("x", In(0, TArray(TFloat64)),
      Let("len", ArrayLen(Ref("x")),
        Let("sum", F64(0),
          seq(
            For("i", I32(0), Ref("len"),
              Set("sum", ApplyPrimitive("+", Array(Ref("sum"), ArrayRef(Ref("x"), Ref("i")))))),
            ApplyPrimitive("/", Array(Ref("sum"), ArrayLen(Ref("x"))))))))

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Double]
    Infer(meanIr)
    println(s"typed:\n$meanIr")
    Compile(meanIr, fb, Map())
    val f = fb.result()()
    def run(a: Array[Double]): Double = {
      val mb = MemoryBuffer()
      val aoff = addArray(mb, a)
      f(mb, aoff)
    }

    assert(run(Array()).isNaN)
    assert(run(Array(1.0)) == 1.0)
    assert(run(Array(1.0,2.0,3.0)) == 2.0)
    assert(run(Array(-1.0,0.0,1.0)) == 0.0)
  }

}
