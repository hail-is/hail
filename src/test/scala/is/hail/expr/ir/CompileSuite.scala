package is.hail.methods.ir

import java.io.PrintWriter

import is.hail.annotations._
import ScalaToRegionValue._
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types._
import org.testng.annotations.Test
import org.scalatest._
import Matchers._
import is.hail.TestUtils
import is.hail.TestUtils.eval
import is.hail.expr.ir.functions.IRFunctionRegistry
import is.hail.utils.FastSeq
import is.hail.variant.ReferenceGenome
import org.apache.spark.sql.Row

class CompileSuite {
  def doit(ir: IR, fb: EmitFunctionBuilder[_]) {
    Emit(ir, fb)
  }

  @Test
  def mean() {
    val tarrf64 = TArray(TFloat64())
    val meanIr =
      Let("x", In(0, tarrf64),
        ApplyBinaryPrimOp(FloatingPointDivide(),
          ArrayFold(Ref("x", tarrf64), F64(0.0), "sum", "v",
            ApplyBinaryPrimOp(Add(), Ref("sum", TFloat64()), Ref("v", TFloat64()))),
          Cast(ArrayLen(Ref("x", tarrf64)), TFloat64())))

    val fb = EmitFunctionBuilder[Region, Long, Boolean, Double]
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
          Ref("foo", TFloat64()), Cast(I32(1), TFloat64())))

    val fb = EmitFunctionBuilder[Region, Double]
    doit(letAddIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()
    val mb = Region()
    assert(f(mb) === 1.0)
  }

  @Test
  def count() {
    val tarrf64 = TArray(TFloat64())
    val letAddIr =
      Let("in", In(0, tarrf64),
        ArrayFold(Ref("in", tarrf64), I32(0), "count", "v",
          ApplyBinaryPrimOp(Add(), Ref("count", TInt32()), I32(1))))

    val fb = EmitFunctionBuilder[Region, Long, Boolean, Int]
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
    val tarrf64 = TArray(TFloat64())
    val letAddIr =
      Let("in", In(0, tarrf64),
        ArrayFold(Ref("in", tarrf64), F64(0), "sum", "v",
          ApplyBinaryPrimOp(Add(), Ref("sum", TFloat64()), Ref("v", TFloat64()))))

    val fb = EmitFunctionBuilder[Region, Long, Boolean, Double]
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
    val tin = TArray(TFloat64())
    val letAddIr =
      Let("in", In(0, tin),
        ArrayFold(Ref("in", tin), I32(0), "count", "v",
          ApplyBinaryPrimOp(Add(), Ref("count", TInt32()), If(IsNA(Ref("v", TFloat64())), I32(0), I32(1)))))

    val fb = EmitFunctionBuilder[Region, Long, Boolean, Int]
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
    val tin = TArray(TFloat64())
    val sumIr =
      ArrayFold(In(0, TArray(TFloat64())), F64(0), "sum", "v",
        ApplyBinaryPrimOp(Add(), Ref("sum", TFloat64()), If(IsNA(Ref("v", TFloat64())), F64(0.0), Ref("v", TFloat64()))))
    val fb = EmitFunctionBuilder[Region, Long, Boolean, Double]
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
    val tin = TArray(TFloat64())
    val letAddIr =
      Let("in", In(0, tin),
        Let("nonMissing",
          ArrayFold(Ref("in", tin), I32(0), "count", "v",
            ApplyBinaryPrimOp(Add(), Ref("count", TInt32()), If(IsNA(Ref("v", TFloat64())), I32(0), I32(1)))),
          Let("sum",
            ArrayFold(Ref("in", tin), F64(0), "sum", "v",
              ApplyBinaryPrimOp(Add(), Ref("sum", TFloat64()), If(IsNA(Ref("v", TFloat64())), F64(0.0), Ref("v", TFloat64())))),
            ApplyBinaryPrimOp(FloatingPointDivide(), Ref("sum", TFloat64()), Cast(Ref("nonMissing", TInt32()), TFloat64())))))

    val fb = EmitFunctionBuilder[Region, Long, Boolean, Double]
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
    println(region.prettyBits())
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
          If(IsNA(Ref("v", TFloat64())), Ref("mean", TFloat64()), Ref("v", TFloat64()))))
    val fb = EmitFunctionBuilder[Region, Long, Boolean, Long]
    doit(replaceMissingIr, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()
    def run(a: Array[java.lang.Double]): Array[java.lang.Double] = {
      val mb = Region()
      val aoff = addBoxedArray(mb, a:_*)
      val t = TArray(TFloat64())
      val roff = f(mb, aoff, false)
      println(s"array location $roff")
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
    val tin = TArray(TFloat64())
    val meanImputeIr =
      Let("in", In(0, tin),
        Let("nonMissing",
          ArrayFold(Ref("in", tin), I32(0), "count", "v",
            ApplyBinaryPrimOp(Add(), Ref("count", TInt32()), If(IsNA(Ref("v", TFloat64())), I32(0), I32(1)))),
          Let("sum",
            ArrayFold(Ref("in", tin), F64(0), "sum", "v",
              ApplyBinaryPrimOp(Add(), Ref("sum", TFloat64()), If(IsNA(Ref("v", TFloat64())), F64(0.0), Ref("v", TFloat64())))),
            Let("mean",
              ApplyBinaryPrimOp(FloatingPointDivide(), Ref("sum", TFloat64()), Cast(Ref("nonMissing", TInt32()), TFloat64())),
              ArrayMap(Ref("in", tin), "v",
                If(IsNA(Ref("v", TFloat64())), Ref("mean", TFloat64()), Ref("v", TFloat64())))))))

    val fb = EmitFunctionBuilder[Region, Long, Boolean, Long]
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
    val fb = EmitFunctionBuilder[Region, Long, Boolean, Long, Boolean, Double]
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
    val fb = EmitFunctionBuilder[Region, Long, Boolean, Long]
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

    val fb = EmitFunctionBuilder[Region, Long, Boolean, Long]
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
    val fb = EmitFunctionBuilder[Region, Long, Boolean, Long, Boolean, Long]
    doit(ir, fb)
    val f = fb.result()()
    val outOff = f(region, loff, false, roff, false)
    assert(tOut.isFieldDefined(region, outOff, tOut.fieldIdx("0")))
    assert(region.loadDouble(tOut.loadField(region, outOff, tOut.fieldIdx("0"))) === 8.0)
  }

  @Test
  def testArrayFilterCutoff() {
    val t = TArray(TInt32())
    val ir = ArrayFilter(ArrayRange(I32(0), In(0, TInt32()), I32(1)), "x", ApplyComparisonOp(LT(TInt32()), Ref("x", TInt32()), In(1, TInt32())))
    val region = Region()
    val fb = EmitFunctionBuilder[Region, Int, Boolean, Int, Boolean, Long]
    doit(ir, fb)
    val f = fb.result()()
    for {
      stop <- 0 to 10
      cutoff <- 0 to stop
    } {
      val aoff = f(region, stop, false, cutoff, false)
      val expected = Array.range(0, cutoff)
      val actual = new UnsafeIndexedSeq(t, region, aoff)
      assert(actual.sameElements(expected))
    }
  }
  
  @Test
  def testArrayFilterElement() {
    val t = TArray(TInt32())
    val ir = ArrayFilter(In(0, t), "x", ApplyComparisonOp(EQ(TInt32()), Ref("x", TInt32()), In(1, TInt32())))
    val region = Region()
    val fb = EmitFunctionBuilder[Region, Long, Boolean, Int, Boolean, Long]
    doit(ir, fb)
    val f = fb.result(Some(new java.io.PrintWriter(System.out)))()
    for {
      stop <- 0 to 5
      elt <- 0 to stop
    } {
      region.clear()
      val base = Array.range(0, stop, 1)
      val rvb = new RegionValueBuilder(region)
      rvb.start(t)
      rvb.addAnnotation(t, base.toIndexedSeq)
      val off = rvb.end()

      val aoff = f(region, off, false, elt, false)
      val expected = base.filter(_ == elt)
      val actual = new UnsafeIndexedSeq(t, region, aoff)
      assert(actual.sameElements(expected))
    }
  }

  @Test
  def testArrayZip() {
    val tout = TArray(TTuple(TInt32(), TString()))
    val a1t = TArray(TInt32())
    val a2t = TArray(TString())
    val a1 = In(0, TArray(TInt32()))
    val a2 = In(1, TArray(TString()))
    val min = IRFunctionRegistry.lookupConversion("min", Seq(TArray(TInt32()))).get
    val range = ArrayRange(I32(0), min(Seq(MakeArray(Seq(ArrayLen(a1), ArrayLen(a2)), TArray(TInt32())))), I32(1))
    val ir = ArrayMap(range, "i", MakeTuple(Seq(ArrayRef(a1, Ref("i", TInt32())), ArrayRef(a2, Ref("i", TInt32())))))
    val region = Region()
    val fb = EmitFunctionBuilder[Region, Long, Boolean, Long, Boolean, Long]
    doit(ir, fb)
    val f = fb.result()()
    region.clear()

    val a1actual = Array(1, 2)
    val a2actual = Array("a", "b")
    val rvb = new RegionValueBuilder(region)

    rvb.start(a1t)
    rvb.addAnnotation(a1t, a1actual.toIndexedSeq)
    val off1 = rvb.end()

    rvb.start(a2t)
    rvb.addAnnotation(a2t, a2actual.toIndexedSeq)
    val off2 = rvb.end()

    val aoff = f(region, off1, false, off2, false)
    val actual = new UnsafeIndexedSeq(tout, region, aoff)

    val expected = IndexedSeq(Row(1, "a"), Row(2, "b"))
    assert(actual.sameElements(expected))
  }

  def testArrayFlatMap() {
    val tRange = TArray(TInt32())
    val ir = ArrayFlatMap(ArrayRange(I32(0), In(0, TInt32()), I32(1)), "i", ArrayRange(I32(0), Ref("i", TInt32()), I32(1)))
    val region = Region()
    val fb = EmitFunctionBuilder[Region, Int, Boolean, Long]
    doit(ir, fb)
    val f = fb.result()()
    for {
      stop <- 0 to 5
    } {
      val aoff = f(region, stop, false)
      val expected = Array.range(0, stop, 1).flatMap(i => Array.range(0, i, 1))
      val actual = new UnsafeIndexedSeq(tRange, region, aoff)
      assert(actual.sameElements(expected))
    }
  }

  @Test
  def testArrayFlatMapVsFilter() {
    val tRange = TArray(TInt32())
    val inputIR = ArrayRange(I32(0), In(0, TInt32()), I32(1))
    val filterCond = { x: IR => ApplyComparisonOp(EQ(TInt32()), x, I32(1)) }
    val filterIR = ArrayFilter(inputIR, "i", filterCond(Ref("i", TInt32())))
    val flatMapIR = ArrayFlatMap(inputIR, "i", If(filterCond(Ref("i", TInt32())), MakeArray(Seq(Ref("i", TInt32())), TArray(TInt32())), MakeArray(Seq(), tRange)))

    val region = Region()
    val fb1 = EmitFunctionBuilder[Region, Int, Boolean, Long]
    doit(flatMapIR, fb1)
    val f1 = fb1.result()()

    val fb2 = EmitFunctionBuilder[Region, Int, Boolean, Long]
    doit(filterIR, fb2)
    val f2 = fb2.result()()

    for {
      stop <- 0 to 5
    } {
      region.clear()
      val aoff1 = f1(region, stop, false)
      val actual = new UnsafeIndexedSeq(tRange, region, aoff1)
      val aoff2 = f2(region, stop, false)
      val expected = new UnsafeIndexedSeq(tRange, region, aoff2)
      assert(actual.sameElements(expected))
    }
  }

  @Test
  def testStruct() {
    val region = Region()
    val rvb = new RegionValueBuilder(region)

    def testStructN(n: Int, hardCoded: Boolean = false, print: Option[PrintWriter] = None) {
      val tin = TStruct((0 until n).map(i => s"field$i" -> TInt32()): _*)
      val in = In(0, tin)
      val ir = if (hardCoded)
        MakeStruct((0 until n).map(i => s"foo$i" -> I32(n)))
      else
        MakeStruct((0 until n).map(i => s"foo$i" -> GetField(in, s"field$i")))

      val fb = EmitFunctionBuilder[Region, Long, Boolean, Long]
      doit(ir, fb)
      val f = fb.result(print)()

      region.clear()
      val input = Row(Array.fill(n)(n): _*)
      rvb.start(tin)
      rvb.addAnnotation(tin, input)
      val inOff = rvb.end()

      val outOff = f(region, inOff, false)

      assert(new UnsafeRow(tin, region, inOff) == new UnsafeRow(tin, region, outOff))
    }

    testStructN(5)
    testStructN(5000)
    testStructN(20000, hardCoded=true)
  }

  @Test def testEmitFunctionBuilderWithReferenceGenome() {
    val grch37 = ReferenceGenome.GRCh37
    val grch38 = ReferenceGenome.GRCh38
    val fb = EmitFunctionBuilder[String, String, Boolean]
    val v1 = fb.newLocal[Boolean]
    val v2 = fb.newLocal[Boolean]
    val v3 = fb.newLocal[Boolean]

    val isValid1 = fb.getReferenceGenome(grch37).invoke[String, Boolean]("isValidContig", fb.getArg[String](1))
    val isValid2 = fb.getReferenceGenome(grch37).invoke[String, Boolean]("isValidContig", fb.getArg[String](2))
    assert(fb.numReferenceGenomes == 1)

    val isValid3 = fb.getReferenceGenome(grch38).invoke[String, Boolean]("isValidContig", fb.getArg[String](1))
    assert(fb.numReferenceGenomes == 2)

    fb.emit(Code(v1 := isValid1, v2 := isValid2, v3 := isValid3, v1 && v2 && v3))
    val expected = grch37.isValidContig("X") && grch37.isValidContig("Y") && grch38.isValidContig("X")
    assert(fb.result()()("X", "Y") == expected)
  }

  @Test def testSelectFields() {
    val ir = SelectFields(
      MakeStruct(FastSeq(
        "foo" -> I32(6),
        "bar" -> F64(0.0))),
      FastSeq("foo"))
    val fb = EmitFunctionBuilder[Region, Long]
    doit(ir, fb)
    val f = fb.result()()
    Region.scoped { region =>
      val off = f(region)
      assert(SafeRow(TStruct("foo" -> TInt32()), region, off).get(0) == 6)
    }
  }
}
