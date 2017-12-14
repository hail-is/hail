package is.hail.methods.ir

import is.hail.annotations._
import ScalaToRegionValue._
import is.hail.asm4s._
import is.hail.check.{Gen, Parameters, Prop}
import is.hail.expr.{TArray, TFloat64, TInt32, TStruct}
import is.hail.expr.ir._
import org.testng.annotations.Test
import org.scalatest._
import Matchers._

class CompileSuite {
  def doit(ir: IR, fb: FunctionBuilder[_]) {
    Infer(ir)
    println(ir)
    Emit(ir, fb)
  }

  @Test
  def mean() {
    val meanIr =
      Let("x", In(0, TArray(TFloat64())),
        ApplyBinaryPrimOp(Divide(),
          ArrayFold(Ref("x"), F64(0.0), "sum", "v",
            ApplyBinaryPrimOp(Add(), Ref("sum"), Ref("v"))),
          Cast(ArrayLen(Ref("x")), TFloat64())))

    val fb = FunctionBuilder.functionBuilder[Region, Long, Boolean, Double]
    doit(meanIr, fb)
    val f = fb.result()()
    def run(a: Array[Double]): Double = {
      val mb = Region()
      val aoff = addArray(mb, a:_*)
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
      Let("foo", F64(0),
        ApplyBinaryPrimOp(Add(),
          Ref("foo"), Cast(I32(1), TFloat64())))

    val fb = FunctionBuilder.functionBuilder[Region, Double]
    doit(letAddIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()
    val mb = Region()
    assert(f(mb) === 1.0)
  }

  @Test
  def count() {
    val letAddIr =
      Let("in", In(0, TArray(TFloat64())),
        ArrayFold(Ref("in"), I32(0), "count", "v",
          ApplyBinaryPrimOp(Add(), Ref("count"), I32(1))))

    val fb = FunctionBuilder.functionBuilder[Region, Long, Boolean, Int]
    doit(letAddIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()

    def run(a: Array[java.lang.Double]): Int = {
      val mb = Region()
      val aoff = addBoxedArray(mb, a:_*)
      val t = TArray(TFloat64())
      f(mb, aoff, false)
    }

    assert(run(Array()) === 0)
    assert(run(Array(5.0)) === 1)
    assert(run(Array(5.0, 6.0, 1.0)) === 3)
  }

  @Test
  def sum() {
    val letAddIr =
      Let("in", In(0, TArray(TFloat64())),
        ArrayFold(Ref("in"), F64(0), "sum", "v",
          ApplyBinaryPrimOp(Add(), Ref("sum"), Ref("v"))))

    val fb = FunctionBuilder.functionBuilder[Region, Long, Boolean, Double]
    doit(letAddIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()

    def run(a: Array[java.lang.Double]): Double = {
      val mb = Region()
      val aoff = addBoxedArray(mb, a:_*)
      val t = TArray(TFloat64())
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
      Let("in", In(0, TArray(TFloat64())),
        ArrayFold(Ref("in"), I32(0), "count", "v",
          ApplyBinaryPrimOp(Add(), Ref("count"), If(IsNA(Ref("v")), I32(0), I32(1)))))

    val fb = FunctionBuilder.functionBuilder[Region, Long, Boolean, Int]
    doit(letAddIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()

    def run(a: Array[java.lang.Double]): Double = {
      val mb = Region()
      val aoff = addBoxedArray(mb, a:_*)
      val t = TArray(TFloat64())
      f(mb, aoff, false)
    }

    assert(run(Array()) === 0)
    assert(run(Array(null)) === 0)
    assert(run(Array(0.0)) === 1)
    assert(run(Array(null, 0.0)) === 1)
    assert(run(Array(0.0, null)) === 1)
  }

  @Test
  def nonMissingSum() {
    val sumIr =
      ArrayFold(In(0, TArray(TFloat64())), F64(0), "sum", "v",
        ApplyBinaryPrimOp(Add(), Ref("sum"), If(IsNA(Ref("v")), F64(0.0), Ref("v"))))
    val fb = FunctionBuilder.functionBuilder[Region, Long, Boolean, Double]
    doit(sumIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()

    def run(a: Array[java.lang.Double]): Double = {
      val mb = Region()
      val aoff = addBoxedArray(mb, a:_*)
      val t = TArray(TFloat64())
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
      Let("in", In(0, TArray(TFloat64())),
        Let("nonMissing",
          ArrayFold(Ref("in"), I32(0), "count", "v",
            ApplyBinaryPrimOp(Add(), Ref("count"), If(IsNA(Ref("v")), I32(0), I32(1)))),
          Let("sum",
            ArrayFold(Ref("in"), F64(0), "sum", "v",
              ApplyBinaryPrimOp(Add(), Ref("sum"), If(IsNA(Ref("v")), F64(0.0), Ref("v")))),
            ApplyBinaryPrimOp(Divide(), Ref("sum"), Cast(Ref("nonMissing"), TFloat64())))))

    val fb = FunctionBuilder.functionBuilder[Region, Long, Boolean, Double]
    doit(letAddIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()

    def run(a: Array[java.lang.Double]): Double = {
      val mb = Region()
      val aoff = addBoxedArray(mb, a:_*)
      val t = TArray(TFloat64())
      f(mb, aoff, false)
    }

    assert(run(Array(1.0)) === 1.0)
    assert(run(Array(null, 1.0)) === 1.0)
    assert(run(Array(1.0, null)) === 1.0)
    assert(run(Array(1.0, null, 3.0)) === 2.0)
  }


  def printRegion(region: Region, string: String) {
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
        ArrayMap(In(0, TArray(TFloat64())), "v",
          If(IsNA(Ref("v")), Ref("mean"), Ref("v"))))
    val fb = FunctionBuilder.functionBuilder[Region, Long, Boolean, Long]
    doit(replaceMissingIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()
    def run(a: Array[java.lang.Double]): Array[java.lang.Double] = {
      val mb = Region()
      val aoff = addBoxedArray(mb, a:_*)
      val t = TArray(TFloat64())
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
      Let("in", In(0, TArray(TFloat64())),
        Let("nonMissing",
          ArrayFold(Ref("in"), I32(0), "count", "v",
            ApplyBinaryPrimOp(Add(), Ref("count"), If(IsNA(Ref("v")), I32(0), I32(1)))),
          Let("sum",
            ArrayFold(Ref("in"), F64(0), "sum", "v",
              ApplyBinaryPrimOp(Add(), Ref("sum"), If(IsNA(Ref("v")), F64(0.0), Ref("v")))),
            Let("mean",
              ApplyBinaryPrimOp(Divide(), Ref("sum"), Cast(Ref("nonMissing"), TFloat64())),
              ArrayMap(Ref("in"), "v",
                If(IsNA(Ref("v")), Ref("mean"), Ref("v")))))))

    val fb = FunctionBuilder.functionBuilder[Region, Long, Boolean, Long]
    doit(meanImputeIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()
    def run(a: Array[java.lang.Double]): Array[java.lang.Double] = {
      val mb = Region()
      val aoff = addBoxedArray(mb, a:_*)
      val t = TArray(TFloat64())
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
    val mapNaIr = MapNA("foo", In(0, TArray(TFloat64())),
      ArrayRef(Ref("foo"), In(1, TInt32())))

    val fb = FunctionBuilder.functionBuilder[Region, Long, Boolean, Int, Boolean, Double]
    doit(mapNaIr, fb)
    val f = fb.result()()
    def run(a: Array[java.lang.Double], i: Int): Double = {
      val mb = Region()
      val aoff = if (a != null) addBoxedArray(mb, a:_*) else -1L
      f(mb, aoff, a == null, i, false)
    }

    assert(run(Array(1.0), 0) === 1.0)
    assert(run(Array(1.0,2.0,3.0), 2) === 3.0)
    assert(run(Array(-1.0,0.0,1.0), 0) === -1.0)

    intercept[RuntimeException](run(null, 5))
    intercept[RuntimeException](run(null, 0))
    intercept[RuntimeException](run(Array(-1.0,null,1.0), 1))
  }

  @Test
  def getFieldSum() {
    val tL = TStruct("0" -> TFloat64())
    val scopeStruct = TStruct("foo" -> TInt32())
    val tR = TStruct("x" -> TFloat64(), "scope" -> scopeStruct)
    val ir = ApplyBinaryPrimOp(Add(),
      GetField(In(0, tL), "0"),
      GetField(In(1, tR), "x"))
    val region = Region()
    val loff = addStruct(region, "0", 3.0)
    val roff = addStruct(region,
      "x", 5.0,
      "scope", scopeStruct, addStruct(region, "foo", 7))
    val fb = FunctionBuilder.functionBuilder[Region, Long, Boolean, Long, Boolean, Double]
    doit(ir, fb)
    val f = fb.result()()
    assert(f(region, loff, false, roff, false) === 8.0)
  }

  @Test
  def getFieldSumStruct() {
    val tL = TStruct("0" -> TFloat64())
    val scopeStruct = TStruct("foo" -> TInt32())
    val tR = TStruct("x" -> TFloat64(), "scope" -> scopeStruct)
    val tOut = TStruct("0" -> TFloat64())
    val ir = MakeStruct(Array(
      ("0",
        ApplyBinaryPrimOp(Add(),
          GetField(In(0, tL), "0"),
          GetField(In(1, tR), "x")))))
    val region = Region()
    val rvb = new RegionValueBuilder()
    val loff = addStruct(region, "0", 3.0)
    val roff = addStruct(region,
      "0", 5.0,
      "scope", scopeStruct, addStruct(region, "foo", 7))
    val fb = FunctionBuilder.functionBuilder[Region, Long, Boolean, Long, Boolean, Long]
    doit(ir, fb)
    val f = fb.result()()
    val outOff = f(region, loff, false, roff, false)
    assert(tOut.isFieldDefined(region, outOff, tOut.fieldIdx("0")))
    assert(region.loadDouble(tOut.loadField(region, outOff, tOut.fieldIdx("0"))) === 8.0)
  }
}
