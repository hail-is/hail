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

  private def addBoxedArray(mb: MemoryBuffer, a: Array[java.lang.Double]): Long = {
    val rvb = new RegionValueBuilder(mb)
    rvb.start(TArray(TFloat64))
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
  def mean() {
    val meanIr = Let("x", In(0, TArray(TFloat64)),
      Let("sum", F64(0),
        seq(
          For("v", "i", Ref("x"),
            Set("sum", ApplyPrimitive("+", Array(Ref("sum"), Ref("v"))))),
          Out(ApplyPrimitive("/", Array(Ref("sum"), ArrayLen(Ref("x"))))))))

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Double]
    Infer(meanIr)
    println(s"typed:\n$meanIr")
    Compile(meanIr, fb)
    val f = fb.result()()
    def run(a: Array[Double]): Double = {
      val mb = MemoryBuffer()
      val aoff = addArray(mb, a)
      f(mb, aoff, false)
    }

    assert(run(Array()).isNaN)
    assert(run(Array(1.0)) == 1.0)
    assert(run(Array(1.0,2.0,3.0)) == 2.0)
    assert(run(Array(-1.0,0.0,1.0)) == 0.0)
  }

  @Test
  def letAdd() {
    val letAddIr =
      Let("nonMissing", F64(0),
        seq(
          Set("nonMissing", ApplyPrimitive("+", Array(Ref("nonMissing"), I32(1)))),
          Out(Ref("nonMissing"))))

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Double]
    Infer(letAddIr)
    println(s"typed:\n$letAddIr")
    Compile(letAddIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()
    val mb = MemoryBuffer()
    assert(f(mb) === 1.0)
  }

  @Test
  def count() {
    val letAddIr =
      Let("in", In(0, TArray(TFloat64)),
        Let("count", F64(0),
          seq(
            For("v", "i", Ref("in"),
              Set("count", ApplyPrimitive("+", Array(Ref("count"), I32(1))))),
            Out(Ref("count")))))

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Double]
    Infer(letAddIr)
    println(s"typed:\n$letAddIr")
    Compile(letAddIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()

    def run(a: Array[java.lang.Double]): Double = {
      val mb = MemoryBuffer()
      val aoff = addBoxedArray(mb, a)
      val t = TArray(TFloat64)
      f(mb, aoff, false)
    }

    assert(run(Array()) === 0.0)
    assert(run(Array(5.0)) === 1.0)
  }

  @Test
  def countNonMissing() {
    val letAddIr =
      Let("in", In(0, TArray(TFloat64)),
        Let("nonMissing", F64(0),
          seq(
            For("v", "i", Ref("in"),
              If(IsNA(Ref("v")),
                seq(),
                Set("nonMissing", ApplyPrimitive("+", Array(Ref("nonMissing"), I32(1)))))),
            Out(Ref("nonMissing")))))

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Double]
    Infer(letAddIr)
    println(s"typed:\n$letAddIr")
    Compile(letAddIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()

    def run(a: Array[java.lang.Double]): Double = {
      val mb = MemoryBuffer()
      val aoff = addBoxedArray(mb, a)
      val t = TArray(TFloat64)
      f(mb, aoff, false)
    }

    assert(run(Array()) === 0.0)
    assert(run(Array(null)) === 0.0)
    assert(run(Array(0.0)) === 1.0)
    assert(run(Array(null, 0.0)) === 1.0)
    assert(run(Array(0.0, null)) === 1.0)
  }

  @Test
  def nonMissingMean() {
    val letAddIr =
      Let("in", In(0, TArray(TFloat64)),
        Let("sum", F64(0.0),
          Let("nonMissing", F64(0),
            seq(
              For("v", "i", Ref("in"),
                If(IsNA(Ref("v")),
                  seq(),
                  seq(Set("nonMissing", ApplyPrimitive("+", Array(Ref("nonMissing"), I32(1)))),
                    Set("sum", ApplyPrimitive("+", Array(Ref("sum"), Ref("v"))))))),
              Out(ApplyPrimitive("/", Array(Ref("sum"), Ref("nonMissing"))))))))

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Double]
    Infer(letAddIr)
    println(s"typed:\n$letAddIr")
    Compile(letAddIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()

    def run(a: Array[java.lang.Double]): Double = {
      val mb = MemoryBuffer()
      val aoff = addBoxedArray(mb, a)
      val t = TArray(TFloat64)
      f(mb, aoff, false)
    }

    assert(run(Array(1.0)) === 1.0)
    assert(run(Array(null, 1.0)) === 1.0)
    assert(run(Array(1.0, null)) === 1.0)
    assert(run(Array(1.0, null, 3.0)) === 2.0)
  }

  @Test
  def meanImpute() {
    val meanImputeIr =
      Let("in", In(0, TArray(TFloat64)),
        Let("out", MakeArrayN(ArrayLen(Ref("in")), TFloat64),
          seq(
            Let("sum", F64(0),
              Let("nonMissing", F64(0),
                seq(
                  For("v", "i", Ref("in"),
                    If(IsNA(Ref("v")),
                      seq(),
                      seq(Set("nonMissing", ApplyPrimitive("+", Array(Ref("nonMissing"), I32(1)))),
                        Set("sum", ApplyPrimitive("+", Array(Ref("sum"), Ref("v"))))))),
                  Let("mean", ApplyPrimitive("/", Array(Ref("sum"), Ref("nonMissing"))),
                    For("v", "i", Ref("in"),
                      If(IsNA(Ref("v")),
                        ArraySet(Ref("out"), Ref("i"), Ref("mean")),
                        ArraySet(Ref("out"), Ref("i"), Ref("v")))))))),
            Out(Ref("out")))))

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Long]
    Infer(meanImputeIr)
    println(s"typed:\n$meanImputeIr")
    Compile(meanImputeIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()
    def run(a: Array[java.lang.Double]): Array[java.lang.Double] = {
      val mb = MemoryBuffer()
      val aoff = addBoxedArray(mb, a)
      val t = TArray(TFloat64)
      val roff = f(mb, aoff, false)
      Array.tabulate[java.lang.Double](a.length) { i =>
        if (t.isElementDefined(mb, roff, i))
          mb.loadDouble(t.loadElement(mb, roff, i))
        else
          null
      }
    }

    // assert(run(Array()) === Array())
    // assert(run(Array(1.0)) === Array(1.0))
    // assert(run(Array(1.0,2.0,3.0)) === Array(1.0,2.0,3.0))
    // assert(run(Array(-1.0,0.0,1.0)) === Array(-1.0,0.0,1.0))

    assert(run(Array(-1.0,null,1.0)) === Array(-1.0,0.0,1.0))
    assert(run(Array(-1.0,null,null)) === Array(-1.0,-1.0,-1.0))
  }

  @Test
  def mapNA() {
    val mapNaIr = Out(MapNA("foo", In(0, TArray(TFloat64)),
      ArrayRef(Ref("foo"), In(1, TInt32))))

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Int, Boolean, Double]
    Infer(mapNaIr)
    println(s"typed:\n$mapNaIr")
    Compile(mapNaIr, fb)
    val f = fb.result()()
    def run(a: Array[java.lang.Double], i: Int): Double = {
      val mb = MemoryBuffer()
      val aoff = if (a != null) addBoxedArray(mb, a) else -1L
      f(mb, aoff, a == null, i, false)
    }

    assert(run(Array(1.0), 0) === 1.0)
    assert(run(Array(1.0,2.0,3.0), 2) === 3.0)
    assert(run(Array(-1.0,0.0,1.0), 0) === -1.0)

    intercept[RuntimeException](run(null, 5))
    intercept[RuntimeException](run(null, 0))
    intercept[RuntimeException](run(Array(-1.0,null,1.0), 1))
  }

}
