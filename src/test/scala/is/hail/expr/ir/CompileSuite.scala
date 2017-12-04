package is.hail.methods.ir

import is.hail.annotations._
import ScalaToRegionValue._
import is.hail.asm4s._
import is.hail.check.{Gen, Parameters, Prop}
import is.hail.expr.{TArray, TFloat64, TInt32, TStruct, TSet}
import is.hail.expr.ir._
import org.testng.annotations.Test
import org.scalatest._
import Matchers._

class CompileSuite {
  def doit(ir: IR, fb: FunctionBuilder[_]) {
    Infer(ir)
    Compile(ir, fb)
  }

  @Test
  def mean() {
    val meanIr =
      Let("x", In(0, TArray(TFloat64())),
        ApplyBinaryPrimOp(Divide(),
          ArrayFold(Ref("x"), F64(0.0), "sum", "v",
            ApplyBinaryPrimOp(Add(), Ref("sum"), Ref("v"))),
          Cast(ArrayLen(Ref("x")), TFloat64())))

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Double]
    doit(meanIr, fb)
    val f = fb.result()()
    def run(a: Array[Double]): Double = {
      val mb = MemoryBuffer()
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

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Double]
    doit(letAddIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()
    val mb = MemoryBuffer()
    assert(f(mb) === 1.0)
  }

  @Test
  def count() {
    val letAddIr =
      Let("in", In(0, TArray(TFloat64())),
        ArrayFold(Ref("in"), I32(0), "count", "v",
          ApplyBinaryPrimOp(Add(), Ref("count"), I32(1))))

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Int]
    doit(letAddIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()

    def run(a: Array[java.lang.Double]): Int = {
      val mb = MemoryBuffer()
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

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Double]
    doit(letAddIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()

    def run(a: Array[java.lang.Double]): Double = {
      val mb = MemoryBuffer()
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

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Int]
    doit(letAddIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()

    def run(a: Array[java.lang.Double]): Double = {
      val mb = MemoryBuffer()
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
    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Double]
    doit(sumIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()

    def run(a: Array[java.lang.Double]): Double = {
      val mb = MemoryBuffer()
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

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Double]
    doit(letAddIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()

    def run(a: Array[java.lang.Double]): Double = {
      val mb = MemoryBuffer()
      val aoff = addBoxedArray(mb, a:_*)
      val t = TArray(TFloat64())
      f(mb, aoff, false)
    }

    assert(run(Array(1.0)) === 1.0)
    assert(run(Array(null, 1.0)) === 1.0)
    assert(run(Array(1.0, null)) === 1.0)
    assert(run(Array(1.0, null, 3.0)) === 2.0)
  }

  @Test
  def replaceMissingValues() {
    val replaceMissingIr =
      Let("mean", F64(42.0),
        ArrayMap(In(0, TArray(TFloat64())), "v",
          If(IsNA(Ref("v")), Ref("mean"), Ref("v"))))
    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Long]
    doit(replaceMissingIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()
    def run(a: Array[java.lang.Double]): Array[java.lang.Double] = {
      val mb = MemoryBuffer()
      val aoff = addBoxedArray(mb, a:_*)
      val t = TArray(TFloat64())
      val roff = f(mb, aoff, false)
      Array.tabulate[java.lang.Double](a.length) { i =>
        if (t.isElementDefined(mb, roff, i)) {
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

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Long]
    doit(meanImputeIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()
    def run(a: Array[java.lang.Double]): Array[java.lang.Double] = {
      val mb = MemoryBuffer()
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

    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Int, Boolean, Double]
    doit(mapNaIr, fb)
    val f = fb.result()()
    def run(a: Array[java.lang.Double], i: Int): Double = {
      val mb = MemoryBuffer()
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
    val region = MemoryBuffer()
    val loff = addStruct(region, "0", 3.0)
    val roff = addStruct(region,
      "x", 5.0,
      "scope", scopeStruct, addStruct(region, "foo", 7))
    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Long, Boolean, Double]
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
      ("0", TFloat64(),
        ApplyBinaryPrimOp(Add(),
          GetField(In(0, tL), "0"),
          GetField(In(1, tR), "x")))))
    val region = MemoryBuffer()
    val rvb = new RegionValueBuilder()
    val loff = addStruct(region, "0", 3.0)
    val roff = addStruct(region,
      "0", 5.0,
      "scope", scopeStruct, addStruct(region, "foo", 7))
    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Long, Boolean, Long]
    doit(ir, fb)
    val f = fb.result()()
    val outOff = f(region, loff, false, roff, false)
    assert(tOut.isFieldDefined(region, outOff, tOut.fieldIdx("0")))
    assert(region.loadDouble(tOut.loadField(region, outOff, tOut.fieldIdx("0"))) === 8.0)
  }

  @Test
  def emptySetContainsNothing() {
    val ir = SetContains(MakeSet(Array(), TInt32()), I32(0))

    val region = MemoryBuffer()
    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Boolean]
    doit(ir, fb)
    assert(!fb.result()()(region))
  }

  @Test
  def singletonSetContainsTheItem() {
    val ir = SetContains(MakeSet(Array(I32(0)), TInt32()), I32(0))

    val region = MemoryBuffer()
    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Boolean]
    doit(ir, fb)
    assert(fb.result()()(region))
  }

  @Test
  def singletonSetDoesNotContainOtherItems() {
    val ir = SetContains(MakeSet(Array(I32(0)), TInt32()), I32(1))

    val region = MemoryBuffer()
    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Boolean]
    doit(ir, fb)
    assert(!fb.result()()(region))
  }

  @Test
  def setContainsAddedElement() {
    val ir = SetContains(SetAdd(MakeSet(Array(), TInt32()), I32(0)), I32(0))

    val region = MemoryBuffer()
    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Boolean]
    doit(ir, fb)
    assert(fb.result()()(region))
  }

  @Test
  def nonEmptySetContainsAddedElement() {
    val ir = SetContains(SetAdd(MakeSet(Array(I32(1)), TInt32()), I32(0)), I32(0))

    val region = MemoryBuffer()
    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Boolean]
    doit(ir, fb)
    assert(fb.result()()(region))
  }

  @Test
  def addIdempotence() {
    val ir = SetContains(SetAdd(MakeSet(Array(I32(0)), TInt32()), I32(0)), I32(0))

    val region = MemoryBuffer()
    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Boolean]
    doit(ir, fb)
    assert(fb.result()()(region))
  }

  @Test
  def bigSet() {
    for {
      (testElement, result) <- Array(
        (0, true), (10, true), (1, true), (3, true), (11, false), (-1, false), (-5, true))
    } {
      val region = MemoryBuffer()
      val fb1 = FunctionBuilder.functionBuilder[MemoryBuffer, Long]
      doit(MakeSet(Array(I32(0), I32(1), I32(10), I32(-5), I32(3))), fb1)
      import java.io.PrintWriter
      val aOff = fb1.result(Some(new PrintWriter(System.out)))()(region)

      val fb2 = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Boolean]
      doit(SetContains(In(0, TSet(TInt32())), I32(testElement)), fb2)
      assert(fb2.result()()(region, aOff, false) == result, s"$testElement")
    }
  }
}
