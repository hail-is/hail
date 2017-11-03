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

  def doit(ir: IR, fb: FunctionBuilder[_]) {
    Infer(ir)
    println(ir)
    Compile2(ir, fb)
  }

  @Test
  def mean() {
    val meanIr =
      Let("x", In(0, TArray(TFloat64)),
        Out(ApplyPrimitive("/",
          Array(
            ArrayFold(Ref("x"), F64(0.0),
              Lambda(Array("sum" -> TFloat64, "v" -> TFloat64), ApplyPrimitive("+", Array(Ref("sum"), Ref("v"))))),
            ArrayLen(Ref("x"))))))

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Double]
    doit(meanIr, fb)
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
        Out(ApplyPrimitive("+", Array(Ref("nonMissing"), I32(1)))))

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Double]
    doit(letAddIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()
    val mb = MemoryBuffer()
    assert(f(mb) === 1.0)
  }

  @Test
  def count() {
    val letAddIr =
      Let("in", In(0, TArray(TFloat64)),
        Out(ArrayFold(Ref("in"), I32(0),
          Lambda(Array("count" -> TInt32, "v" -> TFloat64),
            ApplyPrimitive("+", Array(Ref("count"), I32(1)))))))

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Int]
    doit(letAddIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()

    def run(a: Array[java.lang.Double]): Int = {
      val mb = MemoryBuffer()
      val aoff = addBoxedArray(mb, a)
      val t = TArray(TFloat64)
      f(mb, aoff, false)
    }

    assert(run(Array()) === 0)
    assert(run(Array(5.0)) === 1)
    assert(run(Array(5.0, 6.0, 1.0)) === 3)
  }

  @Test
  def sum() {
    val letAddIr =
      Let("in", In(0, TArray(TFloat64)),
        Out(ArrayFold(Ref("in"), F64(0),
          Lambda(Array("sum" -> TFloat64, "v" -> TFloat64),
            ApplyPrimitive("+", Array(Ref("sum"), Ref("v")))))))

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Double]
    doit(letAddIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()

    def run(a: Array[java.lang.Double]): Double = {
      val mb = MemoryBuffer()
      val aoff = addBoxedArray(mb, a)
      val t = TArray(TFloat64)
      f(mb, aoff, false)
    }

    assert(run(Array()) === 0.0)
    assert(run(Array(5.0)) === 5.0)
    assert(run(Array(5.0, 6.0)) === 11.0)
    assert(run(Array(5.0, 6.0, 8.0)) === 19.0)
    intercept[RuntimeException](run(Array(null)))
  }

  @Test
  def countNonMissing() {
    val letAddIr =
      Let("in", In(0, TArray(TFloat64)),
        Out(ArrayFold(Ref("in"), F64(0),
          Lambda(Array("count" -> TFloat64, "v" -> TFloat64),
            ApplyPrimitive("+", Array(Ref("count"), If(IsNA(Ref("v")), I32(0), I32(1))))))))

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Double]
    doit(letAddIr, fb)
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
  def nonMissingSum() {
    val sumIr =
      ArrayFold(In(0, TArray(TFloat64)), F64(0),
        Lambda(Array("sum" -> TFloat64, "v" -> TFloat64),
          ApplyPrimitive("+", Array(Ref("sum"), If(IsNA(Ref("v")), F64(0.0), Ref("v"))))))
    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Double]
    doit(sumIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()

    def run(a: Array[java.lang.Double]): Double = {
      val mb = MemoryBuffer()
      val aoff = addBoxedArray(mb, a)
      val t = TArray(TFloat64)
      f(mb, aoff, false)
    }

    assert(run(Array(1.0)) === 1.0)
    assert(run(Array(1.0, null)) === 1.0)
    assert(run(Array(null, 1.0)) === 1.0)
    assert(run(Array(1.0, null, 3.0)) === 4.0)
  }

  @Test
  def nonMissingMean() {
    val letAddIr =
      Let("in", In(0, TArray(TFloat64)),
        Let("nonMissing",
          ArrayFold(Ref("in"), F64(0),
            Lambda(Array("count" -> TFloat64, "v" -> TFloat64),
              ApplyPrimitive("+", Array(Ref("count"), If(IsNA(Ref("v")), I32(0), I32(1)))))),
          Let("sum",
            ArrayFold(Ref("in"), F64(0),
              Lambda(Array("sum" -> TFloat64, "v" -> TFloat64),
                ApplyPrimitive("+", Array(Ref("sum"), If(IsNA(Ref("v")), F64(0.0), Ref("v")))))),
            Out(ApplyPrimitive("/", Array(Ref("sum"), Ref("nonMissing")))))))

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Double]
    doit(letAddIr, fb)
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


  def printRegion(region: MemoryBuffer, string: String) {
    println(string)
    val size = region.size
    println("Region size: " + size.toString)
    val bytes = region.loadBytes(0, size.toInt)
    println("Array: ")
    var j = 0
    for (i <- bytes) {
      j += 1
      print(i)
      if (j % 32 == 0) {
        print('\n')
      } else {
        print('\t')
      }
    }
    print('\n')
  }
  @Test
  def replaceMissingValues() {
    val replaceMissingIr =
      Let("mean", F64(42.0),
        ArrayMap(In(0, TArray(TFloat64)),
          Lambda(Array("v" -> TFloat64), If(IsNA(Ref("v")), Ref("mean"), Ref("v")))))
    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Long]
    doit(replaceMissingIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()
    def run(a: Array[java.lang.Double]): Array[java.lang.Double] = {
      val mb = MemoryBuffer()
      val aoff = addBoxedArray(mb, a)
      val t = TArray(TFloat64)
      val roff = f(mb, aoff, false)
      println(s"array location $roff")
      printRegion(mb, "hi amanda")
      Array.tabulate[java.lang.Double](a.length) { i =>
        if (t.isElementDefined(mb, roff, i)) {
          println(s" $i")
          mb.loadDouble(t.loadElement(mb, roff, i))
        } else
          null
      }
    }

    assert(run(Array(-1.0,null,1.0)) === Array(-1.0,42.0,1.0))
    assert(run(Array(-1.0,null,null)) === Array(-1.0,42.0,42.0))
  }

  @Test
  def meanImpute() {
    val meanImputeIr =
      Let("in", In(0, TArray(TFloat64)),
        Let("nonMissing",
          ArrayFold(Ref("in"), F64(0),
            Lambda(Array("count" -> TFloat64, "v" -> TFloat64),
              ApplyPrimitive("+", Array(Ref("count"), If(IsNA(Ref("v")), I32(0), I32(1)))))),
          Let("sum",
            ArrayFold(Ref("in"), F64(0), Lambda(Array("sum" -> TFloat64, "v" -> TFloat64),
              ApplyPrimitive("+", Array(Ref("sum"), If(IsNA(Ref("v")), F64(0.0), Ref("v")))))),
            Let("mean",
              ApplyPrimitive("/", Array(Ref("sum"), Ref("nonMissing"))),
              Out(ArrayMap(Ref("in"),
                Lambda(Array("v" -> TFloat64), If(IsNA(Ref("v")), Ref("mean"), Ref("v")))))))))

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Long]
    doit(meanImputeIr, fb)
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

    assert(run(Array()) === Array())
    assert(run(Array(1.0)) === Array(1.0))
    assert(run(Array(1.0,2.0,3.0)) === Array(1.0,2.0,3.0))
    assert(run(Array(-1.0,0.0,1.0)) === Array(-1.0,0.0,1.0))

    assert(run(Array(-1.0,null,1.0)) === Array(-1.0,0.0,1.0))
    assert(run(Array(-1.0,null,null)) === Array(-1.0,-1.0,-1.0))
  }

  @Test
  def mapNA() {
    val mapNaIr = Out(MapNA("foo", In(0, TArray(TFloat64)),
      ArrayRef(Ref("foo"), In(1, TInt32))))

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Int, Boolean, Double]
    doit(mapNaIr, fb)
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
