package is.hail.methods.ir

import java.io.PrintWriter

import is.hail.annotations._
import ScalaToRegionValue._
import is.hail.asm4s._
import is.hail.check.{Gen, Parameters, Prop}
import is.hail.expr.ir._
import is.hail.expr.types._
import org.testng.annotations.Test
import org.scalatest._
import Matchers._
import org.apache.spark.sql.Row

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
        ApplyBinaryPrimOp(FloatingPointDivide(),
          ArrayFold(Ref("x"), F64(0.0), "sum", "v",
            ApplyBinaryPrimOp(Add(), Ref("sum"), Ref("v"))),
          Cast(ArrayLen(Ref("x")), TFloat64())))

    val fb = FunctionBuilder.functionBuilder[Region, Long, Boolean, Double]
    doit(meanIr, fb)
    val f = fb.result(Some(new PrintWriter(System.out)))()
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
            ApplyBinaryPrimOp(FloatingPointDivide(), Ref("sum"), Cast(Ref("nonMissing"), TFloat64())))))

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
      printf("%02X", i)
      if (j % 32 == 0) {
        print('\n')
      } else {
        print(' ')
      }
    }
    print('\n')
  }


  def checkRegion(region: Region, offset: Long, typ: Type, a: Annotation): Boolean = {
    val v = typ match {
      case t: TStruct if a.isInstanceOf[IndexedSeq[Annotation]] =>
        assert(t.size == a.asInstanceOf[IndexedSeq[Annotation]].size, "lengths of struct differ.")
        t.fields.foreach { f =>
          assert(
            checkRegion(region, t.loadField(region, offset, f.index), f.typ, a.asInstanceOf[IndexedSeq[Annotation]](f.index)),
            s"failed for type $t, expected $a")
        }
      case t: TStruct if a.isInstanceOf[Row] =>
        assert(t.size == a.asInstanceOf[Row].size, "lengths of struct differ.")
        t.fields.foreach { f =>
          assert(
            checkRegion(region, t.loadField(region, offset, f.index), f.typ, a.asInstanceOf[Row].get(f.index)),
            s"failed for type $t, expected $a")
        }
      case t: TArray =>
        val length = t.loadLength(region, offset)
        val arr = a.asInstanceOf[IndexedSeq[Annotation]]
        assert(length == arr.size, s"for array of type $t, expected length=${ arr.size } but got length=$length")
        arr.zipWithIndex.foreach { case (a,i) =>
          checkRegion(region, t.loadElement(region, offset, i), t.elementType, a) }
      case _: TBoolean =>
        assert(region.loadBoolean(offset) === a)
      case _: TInt32 =>
        assert(region.loadInt(offset) === a)
      case _: TInt64 =>
        assert(region.loadLong(offset) === a)
      case _: TFloat32 =>
        assert(region.loadFloat(offset) === a)
      case _: TFloat64 =>
        assert(region.loadDouble(offset) === a)
    }
    true
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
              ApplyBinaryPrimOp(FloatingPointDivide(), Ref("sum"), Cast(Ref("nonMissing"), TFloat64())),
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
  def appendField() {
    val a = TArray(TFloat64())
    val t = TStruct("0" -> a)
    val ir = InsertFields(In(0, t), Array(("1", I32(532)), ("2", F64(3.2)), ("3", I64(533))))
    val region = Region()
    val off = addStruct[IndexedSeq[Double]](region, "0", IndexedSeq(3.0, 5.0, 7.0))
    val fb = FunctionBuilder.functionBuilder[Region, Long, Boolean, Long]
    doit(ir, fb)
    val f = fb.result()()
    checkRegion(region, f(region, off, false), ir.typ,
      IndexedSeq(IndexedSeq(3.0, 5.0, 7.0), 532, 3.2, 533L))
  }

  @Test
  def insertField() {
    val a = TArray(TFloat64())
    val t = TStruct("0" -> a, "1" -> TStruct("a" -> TArray(TInt32())))
    val ir = InsertFields(In(0, t), Array(("0", I32(44)),
      ("1", ArrayRef(GetField(GetField(In(0, t), "1"), "a"), I32(0))),
      ("2", F64(3.2)),
      ("3", I64(533))))
    val region = Region()
    val off1 = addStruct(region, "a", IndexedSeq(1, 2, 3))
    val off = addStruct(region, "0", IndexedSeq(3.0, 5.0, 7.0), "1", TStruct("a" -> TArray(TInt32())), off1)

    val fb = FunctionBuilder.functionBuilder[Region, Long, Boolean, Long]
    doit(ir, fb)
    val f = fb.result()()
    val noff = f(region, off, false)
    checkRegion(region, noff, ir.typ,
      IndexedSeq(44, 1, 3.2, 533L))
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
