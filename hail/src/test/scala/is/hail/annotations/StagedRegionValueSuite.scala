package is.hail.annotations

import is.hail.HailSuite
import is.hail.asm4s._
import is.hail.check.{Gen, Prop}
import is.hail.expr.ir.{EmitFunctionBuilder, EmitRegion}
import is.hail.expr.types._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.utils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class StagedRegionValueSuite extends HailSuite {

  val showRVInfo = true

  @Test
  def testString() {
    val rt = PString()
    val input = "hello"
    val fb = FunctionBuilder.functionBuilder[Region, String, Long]
    val srvb = new StagedRegionValueBuilder(fb, rt)

    fb.emit(
      Code(
        srvb.start(),
        srvb.addString(fb.getArg[String](2)),
        srvb.end()
      )
    )

    val region = Region()
    val rv = RegionValue(region)
    rv.setOffset(fb.result()()(region, input))

    if (showRVInfo) {
      printRegion(region, "string")
      println(rv.pretty(rt))
    }

    val region2 = Region()
    val rv2 = RegionValue(region2)
    rv2.setOffset(region2.appendString(input))

    if (showRVInfo) {
      printRegion(region2, "string")
      println(rv2.pretty(rt))
    }

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(PString.loadString(rv.region, rv.offset) ==
      PString.loadString(rv2.region, rv2.offset))
  }

  @Test
  def testInt() {
    val rt = PInt32()
    val input = 3
    val fb = FunctionBuilder.functionBuilder[Region, Int, Long]
    val srvb = new StagedRegionValueBuilder(fb, rt)

    fb.emit(
      Code(
        srvb.start(),
        srvb.addInt(fb.getArg[Int](2)),
        srvb.end()
      )
    )

    val region = Region()
    val rv = RegionValue(region)
    rv.setOffset(fb.result()()(region, input))

    if (showRVInfo) {
      printRegion(region, "int")
      println(rv.pretty(rt))
    }

    val region2 = Region()
    val rv2 = RegionValue(region2)
    rv2.setOffset(region2.appendInt(input))

    if (showRVInfo) {
      printRegion(region2, "int")
      println(rv2.pretty(rt))
    }

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(rv.region.loadInt(rv.offset) == rv2.region.loadInt(rv2.offset))
  }

  @Test
  def testArray() {
    val rt = PArray(PInt32())
    val input = 3
    val fb = FunctionBuilder.functionBuilder[Region, Int, Long]
    val srvb = new StagedRegionValueBuilder(fb, rt)

    fb.emit(
      Code(
        srvb.start(1),
        srvb.addInt(fb.getArg[Int](2)),
        srvb.advance(),
        srvb.end()
      )
    )

    val region = Region()
    val rv = RegionValue(region)
    rv.setOffset(fb.result()()(region, input))

    if (showRVInfo) {
      printRegion(region, "array")
      println(rv.pretty(rt))
    }

    val region2 = Region()
    val rv2 = RegionValue(region2)
    rv2.setOffset(ScalaToRegionValue(region2, rt, FastIndexedSeq(input)))

    if (showRVInfo) {
      printRegion(region2, "array")
      println(rv2.pretty(rt))
    }

    assert(rt.loadLength(rv.region, rv.offset) == 1)
    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(rv.region.loadInt(rt.loadElement(rv.region, rv.offset, 0)) ==
      rv2.region.loadInt(rt.loadElement(rv2.region, rv2.offset, 0)))
  }

  @Test
  def testStruct() {
    val rt = PStruct("a" -> PString(), "b" -> PInt32())
    val input = 3
    val fb = FunctionBuilder.functionBuilder[Region, Int, Long]
    val srvb = new StagedRegionValueBuilder(fb, rt)

    fb.emit(
      Code(
        srvb.start(),
        srvb.addString("hello"),
        srvb.advance(),
        srvb.addInt(fb.getArg[Int](2)),
        srvb.end()
      )
    )

    val region = Region()
    val rv = RegionValue(region)
    rv.setOffset(fb.result()()(region, input))

    if (showRVInfo) {
      printRegion(region, "struct")
      println(rv.pretty(rt))
    }

    val region2 = Region()
    val rv2 = RegionValue(region2)
    rv2.setOffset(ScalaToRegionValue(region2, rt, Annotation("hello", input)))

    if (showRVInfo) {
      printRegion(region2, "struct")
      println(rv2.pretty(rt))
    }

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(PString.loadString(rv.region, rt.loadField(rv.region, rv.offset, 0)) ==
      PString.loadString(rv2.region, rt.loadField(rv2.region, rv2.offset, 0)))
    assert(rv.region.loadInt(rt.loadField(rv.region, rv.offset, 1)) ==
      rv2.region.loadInt(rt.loadField(rv2.region, rv2.offset, 1)))
  }

  @Test
  def testArrayOfStruct() {
    val rt = PArray(PStruct("a" -> PInt32(), "b" -> PString()))
    val input = "hello"
    val fb = FunctionBuilder.functionBuilder[Region, String, Long]
    val srvb = new StagedRegionValueBuilder(fb, rt)

    val struct = { ssb: StagedRegionValueBuilder =>
      Code(
        ssb.start(),
        ssb.addInt(srvb.arrayIdx + 1),
        ssb.advance(),
        ssb.addString(fb.getArg[String](2))
      )
    }

    fb.emit(
      Code(
        srvb.start(2),
        Code.whileLoop(srvb.arrayIdx < 2,
          Code(
            srvb.addBaseStruct(rt.elementType.asInstanceOf[PStruct], struct),
            srvb.advance()
          )
        ),
        srvb.end()
      )
    )


    val region = Region()
    val rv = RegionValue(region)
    rv.setOffset(fb.result()()(region, input))

    if (showRVInfo) {
      printRegion(region, "array of struct")
      println(rv.pretty(rt))
    }

    val region2 = Region()
    val rv2 = RegionValue(region2)
    val rvb = new RegionValueBuilder(region2)
    rvb.start(rt)
    rvb.startArray(2)
    for (i <- 1 to 2) {
      rvb.startStruct()
      rvb.addInt(i)
      rvb.addString(input)
      rvb.endStruct()
    }
    rvb.endArray()
    rv2.setOffset(rvb.end())

    if (showRVInfo) {
      printRegion(region2, "array of struct")
      println(rv2.pretty(rt))
    }

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(new UnsafeIndexedSeq(rt, rv.region, rv.offset).sameElements(
      new UnsafeIndexedSeq(rt, rv2.region, rv2.offset)))
  }

  @Test
  def testMissingRandomAccessArray() {
    val rt = PArray(PStruct("a" -> PInt32(), "b" -> PString()))
    val intVal = 20
    val strVal = "a string with a partner of 20"
    val region = Region()
    val region2 = Region()
    val rvb = new RegionValueBuilder(region)
    val rvb2 = new RegionValueBuilder(region2)
    val rv = RegionValue(region)
    val rv2 = RegionValue(region2)
    rvb.start(rt)
    rvb.startMissingArray(4)
    rvb.setArrayIndex(2)
    rvb.setPresent()
    rvb.startStruct()
    rvb.addInt(intVal)
    rvb.addString(strVal)
    rvb.endStruct()
    rvb.endArrayUnchecked()
    rv.setOffset(rvb.end())

    rvb2.start(rt)
    rvb2.startArray(4)
    for (i <- 0 to 3) {
      if (i == 2) {
        rvb2.startStruct()
        rvb2.addInt(intVal)
        rvb2.addString(strVal)
        rvb2.endStruct()
      } else {
        rvb2.setMissing()
      }
    }
    rvb2.endArray()
    rv2.setOffset(rvb2.end())
    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(new UnsafeIndexedSeq(rt, rv.region, rv.offset).sameElements(
      new UnsafeIndexedSeq(rt, rv2.region, rv2.offset)))
  }

  @Test
  def testSetFieldPresent() {
    val rt = PStruct("a" -> PInt32(), "b" -> PString(), "c" -> PFloat64())
    val intVal = 30
    val floatVal = 39.273d
    val r = Region()
    val r2 = Region()
    val rv = RegionValue(r)
    val rv2 = RegionValue(r2)
    val rvb = new RegionValueBuilder(r)
    val rvb2 = new RegionValueBuilder(r2)
    rvb.start(rt)
    rvb.startStruct()
    rvb.setMissing()
    rvb.setMissing()
    rvb.addDouble(floatVal)
    rvb.setFieldIndex(0)
    rvb.setPresent()
    rvb.addInt(intVal)
    rvb.setFieldIndex(3)
    rvb.endStruct()
    rv.setOffset(rvb.end())

    rvb2.start(rt)
    rvb2.startStruct()
    rvb2.addInt(intVal)
    rvb2.setMissing()
    rvb2.addDouble(floatVal)
    rvb2.endStruct()
    rv2.setOffset(rvb2.end())

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(rv.region.loadInt(rt.loadField(rv.region, rv.offset, 0)) ==
      rv2.region.loadInt(rt.loadField(rv2.region, rv2.offset, 0)))
    assert(rv.region.loadDouble(rt.loadField(rv.region, rv.offset, 2)) ==
      rv2.region.loadDouble(rt.loadField(rv2.region, rv2.offset, 2)))
  }

  @Test
  def testStructWithArray() {
    val rt = PStruct("a" -> PString(), "b" -> PArray(PInt32()))
    val input = "hello"
    val fb = FunctionBuilder.functionBuilder[Region, String, Long]
    val codeInput = fb.getArg[String](2)
    val srvb = new StagedRegionValueBuilder(fb, rt)

    val array = { sab: StagedRegionValueBuilder =>
      Code(
        sab.start(2),
        Code.whileLoop(sab.arrayIdx < 2,
          Code(
            sab.addInt(sab.arrayIdx + 1),
            sab.advance()
          )
        )
      )
    }

    fb.emit(
      Code(
        srvb.start(),
        srvb.addString(codeInput),
        srvb.advance(),
        srvb.addArray(rt.types(1).asInstanceOf[PArray], array),
        srvb.end()
      )
    )

    val region = Region()
    val rv = RegionValue(region)
    rv.setOffset(fb.result()()(region, input))

    if (showRVInfo) {
      printRegion(region, "struct with array")
      println(rv.pretty(rt))
    }

    val region2 = Region()
    val rv2 = RegionValue(region2)
    val rvb = new RegionValueBuilder(region2)

    rvb.start(rt)
    rvb.startStruct()
    rvb.addString(input)
    rvb.startArray(2)
    for (i <- 1 to 2) {
      rvb.addInt(i)
    }
    rvb.endArray()
    rvb.endStruct()

    rv2.setOffset(rvb.end())

    if (showRVInfo) {
      printRegion(region2, "struct with array")
      println(rv2.pretty(rt))
    }

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(new UnsafeRow(rt, rv.region, rv.offset) ==
      new UnsafeRow(rt, rv2.region, rv2.offset))
  }

  @Test
  def testMissingArray() {
    val rt = PArray(PInt32())
    val input = 3
    val fb = FunctionBuilder.functionBuilder[Region, Int, Long]
    val codeInput = fb.getArg[Int](2)
    val srvb = new StagedRegionValueBuilder(fb, rt)

    fb.emit(
      Code(
        srvb.start(2),
        srvb.addInt(codeInput),
        srvb.advance(),
        srvb.setMissing(),
        srvb.advance(),
        srvb.end()
      )
    )

    val region = Region()
    val rv = RegionValue(region)
    rv.setOffset(fb.result()()(region, input))

    if (showRVInfo) {
      printRegion(region, "missing array")
      println(rv.pretty(rt))
    }

    val region2 = Region()
    val rv2 = RegionValue(region2)
    rv2.setOffset(ScalaToRegionValue(region2, rt, FastIndexedSeq(input, null)))

    if (showRVInfo) {
      printRegion(region2, "missing array")
      println(rv2.pretty(rt))
    }

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(new UnsafeIndexedSeq(rt, rv.region, rv.offset).sameElements(
      new UnsafeIndexedSeq(rt, rv2.region, rv2.offset)))
  }

  def printRegion(region: Region, string: String) {
    println(region.prettyBits())
  }

  @Test
  def testAddPrimitive() {
    val t = PStruct("a" -> PInt32(), "b" -> PBoolean(), "c" -> PFloat64())
    val fb = FunctionBuilder.functionBuilder[Region, Int, Boolean, Double, Long]
    val srvb = new StagedRegionValueBuilder(fb, t)

    fb.emit(
      Code(
        srvb.start(),
        srvb.addIRIntermediate(PInt32())(fb.getArg[Int](2)),
        srvb.advance(),
        srvb.addIRIntermediate(PBoolean())(fb.getArg[Boolean](3)),
        srvb.advance(),
        srvb.addIRIntermediate(PFloat64())(fb.getArg[Double](4)),
        srvb.advance(),
        srvb.end()
      )
    )

    val region = Region()
    val f = fb.result()()
    def run(i: Int, b: Boolean, d: Double): (Int, Boolean, Double) = {
      val off = f(region, i, b, d)
      (region.loadInt(t.loadField(region, off, 0)),
        region.loadBoolean(t.loadField(region, off, 1)),
        region.loadDouble(t.loadField(region, off, 2)))
    }

    assert(run(3, true, 42.0) == ((3, true, 42.0)))
    assert(run(42, false, -1.0) == ((42, false, -1.0)))
  }

  @Test def testDeepCopy() {
    val g = Type.genStruct
      .flatMap(t => Gen.zip(Gen.const(t), t.genValue))
      .filter { case (t, a) => a != null }
      .map { case (t, a) => (PType.canonical(t).asInstanceOf[PStruct], a) }

    val p = Prop.forAll(g) { case (t, a) =>
      assert(t.virtualType.typeCheck(a))
      val copy = Region.scoped { region =>
        val copyOff = Region.scoped { srcRegion =>
          val src = ScalaToRegionValue(srcRegion, t, a)

          val fb = EmitFunctionBuilder[Region, Long, Long]
          fb.emit(
            StagedRegionValueBuilder.deepCopy(
              EmitRegion.default(fb.apply_method),
              t,
              fb.getArg[Long](2).load()))
          val copyF = fb.resultWithIndex()(0, region)
          val newOff = copyF(region, src)


          //clear old stuff
          val len = srcRegion.allocate(0) - src
          srcRegion.storeBytes(src, Array.fill(len.toInt)(0.toByte))
          newOff
        }
        SafeRow(t, region, copyOff)
      }
      copy == a
    }
    p.check()
  }
}
