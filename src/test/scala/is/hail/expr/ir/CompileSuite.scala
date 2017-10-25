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
      Let("sum", F64(0),
        seq(
          For("v", "i", Ref("x"),
            Set("sum", ApplyPrimitive("+", Array(Ref("sum"), Ref("v"))))),
          Out(ApplyPrimitive("/", Array(Ref("sum"), ArrayLen(Ref("x"))))))))

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

  @Test
  def meanImpute() {
    val meanIr =
      Let("in", In(0, TArray(TFloat64)),
        Let("out", MakeArrayN(ArrayLen(Ref("in")), TFloat64),
          Let("sum", F64(0),
            seq(
              For("v", "i", Ref("in"),
                Set("sum", ApplyPrimitive("+", Array(Ref("sum"), Ref("v"))))),
              Let("mean", ApplyPrimitive("/", Array(Ref("sum"), ArrayLen(Ref("in")))),
                For("v", "i", Ref("in"),
                  If(IsNA(Ref("v")),
                    ArraySet(Ref("out"), Ref("i"), Ref("mean")),
                    ArraySet(Ref("out"), Ref("i"), Ref("v")))))))))

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
