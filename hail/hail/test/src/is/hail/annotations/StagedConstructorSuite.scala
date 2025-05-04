package is.hail.annotations

import is.hail.HailSuite
import is.hail.asm4s._
import is.hail.backend.{ExecuteContext, HailTaskContext}
import is.hail.expr.ir.{EmitCode, EmitFunctionBuilder, IEmitCode, RequirednessSuite}
import is.hail.io.fs.FS
import is.hail.scalacheck._
import is.hail.types.physical._
import is.hail.types.physical.stypes.concrete.SStringPointer
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.SInt32Value
import is.hail.utils._

import org.apache.spark.sql.Row
import org.scalatest.matchers.must.Matchers.{be, include}
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper
import org.scalatestplus.scalacheck.ScalaCheckDrivenPropertyChecks
import org.testng.annotations.Test

class StagedConstructorSuite extends HailSuite with ScalaCheckDrivenPropertyChecks {

  val showRVInfo = true

  def sm = ctx.stateManager

  @Test
  def testCanonicalString(): Unit = {
    val rt = PCanonicalString()
    val input = "hello"
    val fb = EmitFunctionBuilder[Region, String, Long](ctx, "fb")

    fb.emitWithBuilder { cb =>
      val st = SStringPointer(rt)
      val region = fb.emb.getCodeParam[Region](1)
      rt.store(
        cb,
        region,
        st.constructFromString(cb, region, fb.getCodeParam[String](2)),
        deepCopy = false,
      )
    }

    val region = Region(pool = pool)
    val rv = RegionValue(region)
    rv.setOffset(fb.result()(theHailClassLoader)(region, input))

    if (showRVInfo) {
      printRegion(region, "string")
      println(rv.pretty(rt))
    }

    val region2 = Region(pool = pool)
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
  def testInt(): Unit = {
    val rt = PInt32()
    val input = 3
    val fb = EmitFunctionBuilder[Region, Int, Long](ctx, "fb")

    fb.emitWithBuilder { cb =>
      rt.store(
        cb,
        fb.emb.getCodeParam[Region](1),
        primitive(fb.getCodeParam[Int](2)),
        deepCopy = false,
      )
    }

    val region = Region(pool = pool)
    val rv = RegionValue(region)
    rv.setOffset(fb.result()(theHailClassLoader)(region, input))

    if (showRVInfo) {
      printRegion(region, "int")
      println(rv.pretty(rt))
    }

    val region2 = Region(pool = pool)
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
  def testArray(): Unit = {
    val rt = PCanonicalArray(PInt32())
    val input = 3
    val fb = EmitFunctionBuilder[Region, Int, Long](ctx, "fb")

    fb.emitWithBuilder { cb =>
      val region = fb.emb.getCodeParam[Region](1)
      val elt = fb.getCodeParam[Int](2)
      rt.constructFromElements(cb, region, const(1), false) { (cb, idx) =>
        IEmitCode.present(cb, primitive(elt))
      }.a
    }

    val region = Region(pool = pool)
    val rv = RegionValue(region)
    rv.setOffset(fb.result()(theHailClassLoader)(region, input))

    if (showRVInfo) {
      printRegion(region, "array")
      println(rv.pretty(rt))
    }

    val region2 = Region(pool = pool)
    val rv2 = RegionValue(region2)
    rv2.setOffset(ScalaToRegionValue(sm, region2, rt, FastSeq(input)))

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
  def testStruct(): Unit = {
    val pstring = PCanonicalString()
    val rt = PCanonicalStruct("a" -> pstring, "b" -> PInt32())
    val input = 3
    val fb = EmitFunctionBuilder[Region, Int, Long](ctx, "fb")

    fb.emitWithBuilder { cb =>
      val region = fb.emb.getCodeParam[Region](1)
      rt.constructFromFields(
        cb,
        region,
        FastSeq(
          EmitCode.fromI(cb.emb) { cb =>
            val st = SStringPointer(pstring)
            IEmitCode.present(cb, st.constructFromString(cb, region, const("hello")))
          },
          EmitCode.fromI(cb.emb)(cb => IEmitCode.present(cb, primitive(fb.getCodeParam[Int](2)))),
        ),
        deepCopy = false,
      ).a
    }

    val region = Region(pool = pool)
    val rv = RegionValue(region)
    rv.setOffset(fb.result()(theHailClassLoader)(region, input))

    if (showRVInfo) {
      printRegion(region, "struct")
      println(rv.pretty(rt))
    }

    val region2 = Region(pool = pool)
    val rv2 = RegionValue(region2)
    rv2.setOffset(ScalaToRegionValue(sm, region2, rt, Annotation("hello", input)))

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
  def testArrayOfStruct(): Unit = {
    val structType = PCanonicalStruct("a" -> PInt32(), "b" -> PCanonicalString())
    val arrayType = PCanonicalArray(structType)
    val input = "hello"
    val fb = EmitFunctionBuilder[Region, String, Long](ctx, "fb")

    fb.emitWithBuilder { cb =>
      val region = fb.emb.getCodeParam[Region](1)

      arrayType.constructFromElements(cb, region, const(2), false) { (cb, idx) =>
        val st = SStringPointer(PCanonicalString())
        IEmitCode.present(
          cb,
          structType.constructFromFields(
            cb,
            region,
            FastSeq(
              EmitCode.fromI(cb.emb)(cb => IEmitCode.present(cb, primitive(cb.memoize(idx + 1)))),
              EmitCode.fromI(cb.emb)(cb =>
                IEmitCode.present(
                  cb,
                  st.constructFromString(cb, region, fb.getCodeParam[String](2)),
                )
              ),
            ),
            deepCopy = false,
          ),
        )
      }.a
    }

    val region = Region(pool = pool)
    val rv = RegionValue(region)
    rv.setOffset(fb.result()(theHailClassLoader)(region, input))

    if (showRVInfo) {
      printRegion(region, "array of struct")
      println(rv.pretty(arrayType))
    }

    val region2 = Region(pool = pool)
    val rv2 = RegionValue(region2)
    val rvb = new RegionValueBuilder(sm, region2)
    rvb.start(arrayType)
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
      println(rv2.pretty(arrayType))
    }

    assert(rv.pretty(arrayType) == rv2.pretty(arrayType))
    assert(new UnsafeIndexedSeq(arrayType, rv.region, rv.offset).sameElements(
      new UnsafeIndexedSeq(arrayType, rv2.region, rv2.offset)
    ))
  }

  @Test
  def testMissingRandomAccessArray(): Unit = {
    val rt = PCanonicalArray(PCanonicalStruct("a" -> PInt32(), "b" -> PCanonicalString()))
    val intVal = 20
    val strVal = "a string with a partner of 20"
    val region = Region(pool = pool)
    val region2 = Region(pool = pool)
    val rvb = new RegionValueBuilder(sm, region)
    val rvb2 = new RegionValueBuilder(sm, region2)
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
      new UnsafeIndexedSeq(rt, rv2.region, rv2.offset)
    ))
  }

  @Test
  def testSetFieldPresent(): Unit = {
    val rt = PCanonicalStruct("a" -> PInt32(), "b" -> PCanonicalString(), "c" -> PFloat64())
    val intVal = 30
    val floatVal = 39.273d
    val r = Region(pool = pool)
    val r2 = Region(pool = pool)
    val rv = RegionValue(r)
    val rv2 = RegionValue(r2)
    val rvb = new RegionValueBuilder(sm, r)
    val rvb2 = new RegionValueBuilder(sm, r2)
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
  def testStructWithArray(): Unit = {
    val tArray = PCanonicalArray(PInt32())
    val rt = PCanonicalStruct("a" -> PCanonicalString(), "b" -> tArray)
    val input = "hello"
    val fb = EmitFunctionBuilder[Region, String, Long](ctx, "fb")

    fb.emitWithBuilder { cb =>
      val region = fb.emb.getCodeParam[Region](1)
      rt.constructFromFields(
        cb,
        region,
        FastSeq(
          EmitCode.fromI(cb.emb)(cb =>
            IEmitCode.present(
              cb,
              SStringPointer(PCanonicalString()).constructFromString(
                cb,
                region,
                fb.getCodeParam[String](2),
              ),
            )
          ),
          EmitCode.fromI(cb.emb)(cb =>
            IEmitCode.present(
              cb,
              tArray.constructFromElements(cb, region, const(2), deepCopy = false) { (cb, idx) =>
                IEmitCode.present(cb, primitive(cb.memoize(idx + 1)))
              },
            )
          ),
        ),
        deepCopy = false,
      ).a
    }

    val region = Region(pool = pool)
    val rv = RegionValue(region)
    rv.setOffset(fb.result()(theHailClassLoader)(region, input))

    if (showRVInfo) {
      printRegion(region, "struct with array")
      println(rv.pretty(rt))
    }

    val region2 = Region(pool = pool)
    val rv2 = RegionValue(region2)
    val rvb = new RegionValueBuilder(sm, region2)

    rvb.start(rt)
    rvb.startStruct()
    rvb.addString(input)
    rvb.startArray(2)
    for (i <- 1 to 2)
      rvb.addInt(i)
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
  def testMissingArray(): Unit = {
    val rt = PCanonicalArray(PInt32())
    val input = 3
    val fb = EmitFunctionBuilder[Region, Int, Long](ctx, "fb")

    fb.emitWithBuilder { cb =>
      val region = fb.emb.getCodeParam[Region](1)
      rt.constructFromElements(cb, region, const(2), deepCopy = false) { (cb, idx) =>
        IEmitCode(cb, idx > 0, new SInt32Value(fb.getCodeParam[Int](2)))
      }.a
    }

    val region = Region(pool = pool)
    val rv = RegionValue(region)
    rv.setOffset(fb.result()(theHailClassLoader)(region, input))

    if (showRVInfo) {
      printRegion(region, "missing array")
      println(rv.pretty(rt))
    }

    val region2 = Region(pool = pool)
    val rv2 = RegionValue(region2)
    rv2.setOffset(ScalaToRegionValue(sm, region2, rt, FastSeq(input, null)))

    if (showRVInfo) {
      printRegion(region2, "missing array")
      println(rv2.pretty(rt))
    }

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(new UnsafeIndexedSeq(rt, rv.region, rv.offset).sameElements(
      new UnsafeIndexedSeq(rt, rv2.region, rv2.offset)
    ))
  }

  def printRegion(region: Region, string: String): Unit =
    println(region.prettyBits())

  @Test
  def testAddPrimitive(): Unit = {
    val t = PCanonicalStruct("a" -> PInt32(), "b" -> PBoolean(), "c" -> PFloat64())
    val fb = EmitFunctionBuilder[Region, Int, Boolean, Double, Long](ctx, "fb")

    fb.emitWithBuilder { cb =>
      val region = fb.emb.getCodeParam[Region](1)

      t.constructFromFields(
        cb,
        region,
        FastSeq(
          EmitCode.fromI(cb.emb)(cb => IEmitCode.present(cb, primitive(fb.getCodeParam[Int](2)))),
          EmitCode.fromI(cb.emb)(cb =>
            IEmitCode.present(cb, primitive(fb.getCodeParam[Boolean](3)))
          ),
          EmitCode.fromI(cb.emb)(cb => IEmitCode.present(cb, primitive(fb.getCodeParam[Double](4)))),
        ),
        deepCopy = false,
      ).a
    }

    val region = Region(pool = pool)
    val f = fb.result()(theHailClassLoader)
    def run(i: Int, b: Boolean, d: Double): (Int, Boolean, Double) = {
      val off = f(region, i, b, d)
      (
        Region.loadInt(t.loadField(off, 0)),
        Region.loadBoolean(t.loadField(off, 1)),
        Region.loadDouble(t.loadField(off, 2)),
      )
    }

    assert(run(3, true, 42.0) == ((3, true, 42.0)))
    assert(run(42, false, -1.0) == ((42, false, -1.0)))
  }

  def emitCopy(ctx: ExecuteContext, ptype: PType, deepCopy: Boolean)
    : (HailClassLoader, FS, HailTaskContext, Region) => AsmFunction2[Region, Long, Long] = {
    val fb = EmitFunctionBuilder[Region, Long, Long](ctx, "copy")
    fb.emitWithBuilder { cb =>
      val region = fb.getCodeParam[Region](1)
      val offset = fb.getCodeParam[Long](2)
      val value = ptype.loadCheapSCode(cb, offset)
      ptype.store(cb, region, value, deepCopy)
    }
    fb.resultWithIndex()
  }

  @Test def testShallowCopyOfPointersFailsAcrossRegions(): Unit = {
    val ptype = PCanonicalStruct(required = true, "a" -> PCanonicalArray(PInt32()))
    val value = genVal(ctx, ptype).sample.get
    val ShallowCopy = emitCopy(ctx, ptype, deepCopy = false)

    val ex: RuntimeException =
      ctx.scopedExecution { (hcl, fs, htc, r1) =>
        val invalidPtr: Long =
          using(RegionPool(strictMemoryCheck = true)) { p2 =>
            using(p2.getRegion()) { r2 =>
              val offset = ScalaToRegionValue(sm, r2, ptype, value)
              ShallowCopy(hcl, fs, htc, r1)(r1, offset)
            }
          }

        intercept[RuntimeException] {
          SafeRow(ptype, invalidPtr)
        }
      }

    ex.getMessage should include("invalid memory access")
  }

  @Test def testDeepCopy(): Unit =
    forAll(genPTypeVal[PCanonicalStruct](ctx)) { case (t, a: Row) =>
      val DeepCopy = emitCopy(ctx, t, deepCopy = true)

      val copy: Row =
        ctx.scopedExecution { (hcl, fs, htc, r1) =>
          val validPtr: Long =
            using(RegionPool(strictMemoryCheck = true)) { p2 =>
              using(p2.getRegion()) { r2 =>
                val offset = ScalaToRegionValue(sm, r2, t, a)
                DeepCopy(hcl, fs, htc, r1)(r1, offset)
              }
            }

          SafeRow(t, validPtr)
        }

      copy should be(a)
    }

  @Test def testUnstagedCopy(): Unit = {
    val t1 = PCanonicalArray(
      PCanonicalStruct(
        true,
        "x1" -> PInt32(),
        "x2" -> PCanonicalArray(PInt32(), required = true),
        "x3" -> PCanonicalArray(PInt32(true), required = true),
        "x4" -> PCanonicalSet(
          PCanonicalStruct(true, "y" -> PCanonicalString(true)),
          required = false,
        ),
      ),
      required = false,
    )
    val t2 = RequirednessSuite.deepInnerRequired(t1, false)

    val value = IndexedSeq(
      Row(1, IndexedSeq(1, 2, 3), IndexedSeq(0, -1), Set(Row("asdasdasd"), Row(""))),
      Row(1, IndexedSeq(), IndexedSeq(-1), Set(Row("aa"))),
    )

    pool.scopedRegion { r =>
      val rvb = new RegionValueBuilder(sm, r)
      val v1 = t2.unstagedStoreJavaObject(sm, value, r)
      assert(SafeRow.read(t2, v1) == value)

      rvb.clear()
      rvb.start(t1)
      rvb.addRegionValue(t2, r, v1)
      val v2 = rvb.end()
      assert(SafeRow.read(t1, v2) == value)
    }
  }

  @Test def testStagedCopy(): Unit = {
    val t1 = PCanonicalStruct(
      false,
      "a" -> PCanonicalArray(
        PCanonicalStruct(
          true,
          "x1" -> PInt32(),
          "x2" -> PCanonicalArray(PInt32(), required = true),
          "x3" -> PCanonicalArray(PInt32(true), required = true),
          "x4" -> PCanonicalSet(
            PCanonicalStruct(true, "y" -> PCanonicalString(true)),
            required = false,
          ),
        ),
        required = false,
      ),
    )
    val t2 = RequirednessSuite.deepInnerRequired(t1, false).asInstanceOf[PCanonicalStruct]

    val value = IndexedSeq(
      Row(1, IndexedSeq(1, 2, 3), IndexedSeq(0, -1), Set(Row("asdasdasd"), Row(""))),
      Row(1, IndexedSeq(), IndexedSeq(-1), Set(Row("aa"))),
    )

    val valueT2 = t2.types(0)
    pool.scopedRegion { r =>
      val v1 = valueT2.unstagedStoreJavaObject(sm, value, r)
      assert(SafeRow.read(valueT2, v1) == value)

      val f1 = EmitFunctionBuilder[Long](ctx, "stagedCopy1")
      f1.emitWithBuilder { cb =>
        val region = f1.partitionRegion
        t2.constructFromFields(
          cb,
          region,
          FastSeq(EmitCode.present(cb.emb, t2.types(0).loadCheapSCode(cb, v1))),
          deepCopy = false,
        ).a
      }
      val cp1 = f1.resultWithIndex()(theHailClassLoader, ctx.fs, ctx.taskContext, r)()
      assert(SafeRow.read(t2, cp1) == Row(value))

      val f2 = EmitFunctionBuilder[Long](ctx, "stagedCopy2")
      f2.emitWithBuilder { cb =>
        val region = f2.partitionRegion
        t1.constructFromFields(
          cb,
          region,
          FastSeq(EmitCode.present(cb.emb, t2.types(0).loadCheapSCode(cb, v1))),
          deepCopy = false,
        ).a
      }
      val cp2 = f2.resultWithIndex()(theHailClassLoader, ctx.fs, ctx.taskContext, r)()
      assert(SafeRow.read(t1, cp2) == Row(value))
    }
  }
}
