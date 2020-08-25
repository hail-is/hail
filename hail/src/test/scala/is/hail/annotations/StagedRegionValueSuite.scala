package is.hail.annotations

import is.hail.HailSuite
import is.hail.asm4s._
import is.hail.check.{Gen, Prop}
import is.hail.expr.ir.{EmitFunctionBuilder, EmitRegion}
import is.hail.types._
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class StagedRegionValueSuite extends HailSuite {

  val showRVInfo = true

  @Test
  def testCanonicalString() {
    val rt = PCanonicalString()
    val input = "hello"
    val fb = EmitFunctionBuilder[Region, String, Long](ctx, "fb")
    val srvb = new StagedRegionValueBuilder(fb.emb, rt, fb.emb.getCodeParam[Region](1))

    fb.emit(
      Code(
        srvb.start(),
        srvb.addString(fb.getCodeParam[String](2)),
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
    val bytes = input.getBytes()
    val bt = PCanonicalBinary()
    val boff = bt.allocate(region2, bytes.length)
    bt.store(boff, bytes)
    rv2.setOffset(boff)

    if (showRVInfo) {
      printRegion(region2, "string")
      println(rv2.pretty(rt))
    }

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(rt.loadString(rv.offset) ==
      rt.loadString(rv2.offset))
  }

  @Test
  def testInt() {
    val rt = PInt32()
    val input = 3
    val fb = EmitFunctionBuilder[Region, Int, Long](ctx, "fb")
    val srvb = new StagedRegionValueBuilder(fb.emb, rt, fb.emb.getCodeParam[Region](1))

    fb.emit(
      Code(
        srvb.start(),
        srvb.addInt(fb.getCodeParam[Int](2)),
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
    rv2.setOffset(region2.allocate(4, 4))
    Region.storeInt(rv2.offset, input)

    if (showRVInfo) {
      printRegion(region2, "int")
      println(rv2.pretty(rt))
    }

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(Region.loadInt(rv.offset) == Region.loadInt(rv2.offset))
  }

  @Test
  def testArray() {
    val rt = PCanonicalArray(PInt32())
    val input = 3
    val fb = EmitFunctionBuilder[Region, Int, Long](ctx, "fb")
    val srvb = new StagedRegionValueBuilder(fb.emb, rt, fb.emb.getCodeParam[Region](1))

    fb.emit(
      Code(
        srvb.start(1),
        srvb.addInt(fb.getCodeParam[Int](2)),
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

    assert(rt.loadLength(rv.offset) == 1)
    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(Region.loadInt(rt.loadElement(rv.offset, 0)) ==
      Region.loadInt(rt.loadElement(rv2.offset, 0)))
  }

  @Test
  def testStruct() {
    val rt = PCanonicalStruct("a" -> PCanonicalString(), "b" -> PInt32())
    val input = 3
    val fb = EmitFunctionBuilder[Region, Int, Long](ctx, "fb")
    val srvb = new StagedRegionValueBuilder(fb.emb, rt, fb.emb.getCodeParam[Region](1))

    fb.emit(
      Code(
        srvb.start(),
        srvb.addString("hello"),
        srvb.advance(),
        srvb.addInt(fb.getCodeParam[Int](2)),
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
    assert(rt.types(0).asInstanceOf[PString].loadString(rt.loadField(rv.offset, 0)) ==
      rt.types(0).asInstanceOf[PString].loadString(rt.loadField(rv2.offset, 0)))
    assert(Region.loadInt(rt.loadField(rv.offset, 1)) ==
      Region.loadInt(rt.loadField(rv2.offset, 1)))
  }

  @Test
  def testArrayOfStruct() {
    val rt = PCanonicalArray(PCanonicalStruct("a" -> PInt32(), "b" -> PCanonicalString()))
    val input = "hello"
    val fb = EmitFunctionBuilder[Region, String, Long](ctx, "fb")
    val srvb = new StagedRegionValueBuilder(fb.emb, rt, fb.emb.getCodeParam[Region](1))

    val struct = { ssb: StagedRegionValueBuilder =>
      Code(
        ssb.start(),
        ssb.addInt(srvb.arrayIdx + 1),
        ssb.advance(),
        ssb.addString(fb.getCodeParam[String](2))
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
    val rt = PCanonicalArray(PCanonicalStruct("a" -> PInt32(), "b" -> PCanonicalString()))
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
    val rt = PCanonicalStruct("a" -> PInt32(), "b" -> PCanonicalString(), "c" -> PFloat64())
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
    assert(Region.loadInt(rt.loadField(rv.offset, 0)) ==
      Region.loadInt(rt.loadField(rv2.offset, 0)))
    assert(Region.loadDouble(rt.loadField(rv.offset, 2)) ==
      Region.loadDouble(rt.loadField(rv2.offset, 2)))
  }

  @Test
  def testStructWithArray() {
    val rt = PCanonicalStruct("a" -> PCanonicalString(), "b" -> PCanonicalArray(PInt32()))
    val input = "hello"
    val fb = EmitFunctionBuilder[Region, String, Long](ctx, "fb")
    val codeInput = fb.getCodeParam[String](2)
    val srvb = new StagedRegionValueBuilder(fb.emb, rt, fb.emb.getCodeParam[Region](1))

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
    val rt = PCanonicalArray(PInt32())
    val input = 3
    val fb = EmitFunctionBuilder[Region, Int, Long](ctx, "fb")
    val codeInput = fb.getCodeParam[Int](2)
    val srvb = new StagedRegionValueBuilder(fb.emb, rt, fb.emb.getCodeParam[Region](1))

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
    val t = PCanonicalStruct("a" -> PInt32(), "b" -> PBoolean(), "c" -> PFloat64())
    val fb = EmitFunctionBuilder[Region, Int, Boolean, Double, Long](ctx, "fb")
    val srvb = new StagedRegionValueBuilder(fb.emb, t, fb.emb.getCodeParam[Region](1))

    fb.emit(
      Code(
        srvb.start(),
        srvb.addIRIntermediate(PInt32())(fb.getCodeParam[Int](2)),
        srvb.advance(),
        srvb.addIRIntermediate(PBoolean())(fb.getCodeParam[Boolean](3)),
        srvb.advance(),
        srvb.addIRIntermediate(PFloat64())(fb.getCodeParam[Double](4)),
        srvb.advance(),
        srvb.end()
      )
    )

    val region = Region()
    val f = fb.result()()
    def run(i: Int, b: Boolean, d: Double): (Int, Boolean, Double) = {
      val off = f(region, i, b, d)
      (Region.loadInt(t.loadField(off, 0)),
        Region.loadBoolean(t.loadField(off, 1)),
        Region.loadDouble(t.loadField(off, 2)))
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

          val fb = EmitFunctionBuilder[Region, Long, Long](ctx, "deep_copy")
          fb.emit(
            StagedRegionValueBuilder.deepCopyFromOffset(
              EmitRegion.default(fb.apply_method),
              t,
              fb.getCodeParam[Long](2)))
          val copyF = fb.resultWithIndex()(0, region)
          val newOff = copyF(region, src)


          //clear old stuff
          val len = srcRegion.allocate(0) - src
          Region.storeBytes(src, Array.fill(len.toInt)(0.toByte))
          newOff
        }
        SafeRow(t, copyOff)
      }
      copy == a
    }
    p.check()
  }

  @Test def testUnstagedCopy() {
    val t1 = PCanonicalArray(PCanonicalStruct(
      true,
      "x1" -> PInt32(),
      "x2" -> PCanonicalArray(PInt32(), required = true),
      "x3" -> PCanonicalArray(PInt32(true), required = true),
      "x4" -> PCanonicalSet(PCanonicalStruct(true, "y" -> PCanonicalString(true)), required = false)
    ), required = false)
    val t2 = t1.deepInnerRequired(false)

    val value = IndexedSeq(
      Row(1, IndexedSeq(1,2,3), IndexedSeq(0, -1), Set(Row("asdasdasd"), Row(""))),
      Row(1, IndexedSeq(), IndexedSeq(-1), Set(Row("aa")))
    )

    Region.scoped { r =>
      val rvb = new RegionValueBuilder(r)
      rvb.start(t2)
      rvb.addAnnotation(t2.virtualType, value)
      val v1 = rvb.end()
      assert(SafeRow.read(t2, v1) == value)

      rvb.clear()
      rvb.start(t1)
      rvb.addRegionValue(t2, r, v1)
      val v2 = rvb.end()
      assert(SafeRow.read(t1, v2) == value)
    }
  }

  @Test def testStagedCopy() {
    val t1 = PCanonicalStruct(false, "a" -> PCanonicalArray(PCanonicalStruct(
      true,
      "x1" -> PInt32(),
      "x2" -> PCanonicalArray(PInt32(), required = true),
      "x3" -> PCanonicalArray(PInt32(true), required = true),
      "x4" -> PCanonicalSet(PCanonicalStruct(true, "y" -> PCanonicalString(true)), required = false)
    ), required = false))
    val t2 = t1.deepInnerRequired(false).asInstanceOf[PStruct]

    val value = IndexedSeq(
      Row(1, IndexedSeq(1,2,3), IndexedSeq(0, -1), Set(Row("asdasdasd"), Row(""))),
      Row(1, IndexedSeq(), IndexedSeq(-1), Set(Row("aa")))
    )

    val valueT2 = t2.types(0)
    Region.scoped { r =>
      val rvb = new RegionValueBuilder(r)
      rvb.start(valueT2)
      rvb.addAnnotation(valueT2.virtualType, value)
      val v1 = rvb.end()
      assert(SafeRow.read(valueT2, v1) == value)

      val f1 = EmitFunctionBuilder[Long](ctx, "stagedCopy1")
      val srvb = new StagedRegionValueBuilder(f1.apply_method, t2, f1.partitionRegion)
      f1.emit(Code(
        srvb.start(),
        srvb.addIRIntermediate(t2.types(0))(v1),
        srvb.end()
      ))
      val cp1 = f1.resultWithIndex()(0, r)()
      assert(SafeRow.read(t2, cp1) == Row(value))

      val f2 = EmitFunctionBuilder[Long](ctx, "stagedCopy2")
      val srvb2 = new StagedRegionValueBuilder(f2.apply_method, t1, f2.partitionRegion)
      f2.emit(Code(
        srvb2.start(),
        srvb2.addIRIntermediate(t2.types(0))(v1),
        srvb2.end()
      ))
      val cp2 = f2.resultWithIndex()(0, r)()
      assert(SafeRow.read(t1, cp2) == Row(value))
    }
  }
}
