package is.hail.asm4s

import is.hail.TestUtils._
import is.hail.annotations.Region
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{EmitCodeBuilder, EmitFunctionBuilder, EmitValue, IEmitCode}
import is.hail.types.physical._
import is.hail.types.physical.stypes.{EmitType, SValue}
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.primitives.{
  SFloat32Value, SFloat64Value, SInt32, SInt32Value, SInt64, SInt64Value,
}
import is.hail.types.virtual.{TInt32, TInt64, TStruct}

import org.apache.spark.sql.Row
import org.junit.jupiter.api.Test

class CodeSuite {

  @Test def testForLoop(implicit ctx: ExecuteContext): Unit = {
    val fb = EmitFunctionBuilder[Int](ctx, "foo")
    fb.emitWithBuilder[Int] { cb =>
      val i = cb.newLocal[Int]("i")
      val sum = cb.newLocal[Int]("sum", 0)
      cb.assign(sum, 0)
      cb.for_(cb.assign(i, 0), i < 5, cb.assign(i, i + 1), cb.assign(sum, sum + i))
      sum
    }

    val result = fb.resultWithIndex()(ctx.theHailClassLoader, ctx.fs, ctx.taskContext, ctx.r)()
    assertEq(result, 10)
  }

  @Test def testSizeBasic(implicit ctx: ExecuteContext): Unit = {
    val int64 = new SInt64Value(5L)
    val int32 = new SInt32Value(2)
    val struct = new SStackStructValue(
      SStackStruct(
        TStruct("x" -> TInt64, "y" -> TInt32),
        IndexedSeq(EmitType(SInt64, true), EmitType(SInt32, false)),
      ),
      IndexedSeq(EmitValue(None, int64), EmitValue(Some(false), int32)),
    )
    val str = new SJavaStringValue(const("cat"))

    def testSizeHelper(v: SValue): Long = {
      val fb = EmitFunctionBuilder[Long](ctx, "test_size_in_bytes")
      val mb = fb.apply_method
      mb.emit(EmitCodeBuilder.scopedCode(mb)(cb => v.sizeToStoreInBytes(cb).value))
      fb.result()(ctx.theHailClassLoader)()
    }

    assertEq(testSizeHelper(int64), 8L)
    assertEq(testSizeHelper(int32), 4L)
    assertEq(
      testSizeHelper(struct),
      16L,
    ) // 1 missing byte that gets 4 byte aligned, 8 bytes for long, 4 bytes for missing int
    assertEq(testSizeHelper(str), 7L) // 4 byte header, 3 bytes for the 3 letters.
  }

  @Test def testArraySizeInBytes(implicit ctx: ExecuteContext): Unit = {
    val fb = EmitFunctionBuilder[Region, Long](ctx, "test_size_in_bytes")
    val mb = fb.apply_method
    val ptype = PCanonicalArray(PInt32())
    val stype = SIndexablePointer(ptype)
    mb.emit(EmitCodeBuilder.scopedCode(mb) { cb =>
      val region = fb.emb.getCodeParam[Region](1)
      val sarray = ptype.constructFromElements(cb, region, 5, true) { (cb, idx) =>
        cb.if_(
          idx ceq 2,
          IEmitCode.missing(cb, stype.elementType.defaultValue),
          IEmitCode.present(cb, new SInt32Value(idx)),
        )
      }
      sarray.sizeToStoreInBytes(cb).value
    })
    assertEq(
      fb.result()(ctx.theHailClassLoader)(ctx.r),
      36L,
    ) // 2 missing bytes 8 byte aligned + 8 header bytes + 5 elements * 4 bytes for ints.
  }

  @Test def testIntervalSizeInBytes(implicit ctx: ExecuteContext): Unit = {
    val fb = EmitFunctionBuilder[Region, Long](ctx, "test_size_in_bytes")
    val mb = fb.apply_method

    val structL = new SStackStructValue(
      SStackStruct(
        TStruct("x" -> TInt64, "y" -> TInt32),
        IndexedSeq(EmitType(SInt64, true), EmitType(SInt32, false)),
      ),
      IndexedSeq(EmitValue(None, new SInt64Value(5L)), EmitValue(Some(false), new SInt32Value(2))),
    )
    val structR = new SStackStructValue(
      SStackStruct(
        TStruct("x" -> TInt64, "y" -> TInt32),
        IndexedSeq(EmitType(SInt64, true), EmitType(SInt32, false)),
      ),
      IndexedSeq(EmitValue(None, new SInt64Value(8L)), EmitValue(Some(false), new SInt32Value(5))),
    )

    val pType = PCanonicalInterval(structL.st.storageType())

    mb.emit(EmitCodeBuilder.scopedCode(mb) { cb =>
      val region = fb.emb.getCodeParam[Region](1)
      val sval: SValue = pType.constructFromCodes(
        cb,
        region,
        EmitValue(Some(false), structL),
        EmitValue(Some(false), structR),
        true,
        true,
      )
      sval.sizeToStoreInBytes(cb).value
    })
    assertEq(
      fb.result()(ctx.theHailClassLoader)(ctx.r),
      72L,
    ) // 2 28 byte structs, plus 2 1 byte booleans that get 8 byte for an extra 8 bytes, plus missing bytes.
  }

  @Test def testHash(implicit ctx: ExecuteContext): Unit = {
    val fields = IndexedSeq(
      PField("a", PCanonicalString(), 0),
      PField("b", PInt32(), 1),
      PField("c", PFloat32(), 2),
    )
    assertEq(hashTestNumHelper(new SInt32Value(6)), hashTestNumHelper(new SInt32Value(6)))
    assertEq(
      hashTestNumHelper(new SInt64Value(5000000000L)),
      hashTestNumHelper(new SInt64Value(5000000000L)),
    )
    assertEq(
      hashTestNumHelper(new SFloat32Value(3.14f)),
      hashTestNumHelper(new SFloat32Value(3.14f)),
    )
    assertEq(
      hashTestNumHelper(new SFloat64Value(5000000000.89d)),
      hashTestNumHelper(new SFloat64Value(5000000000.89d)),
    )
    assertEq(hashTestStringHelper("dog"), hashTestStringHelper("dog"))
    assertEq(
      hashTestArrayHelper(IndexedSeq(1, 2, 3, 4, 5, 6)),
      hashTestArrayHelper(IndexedSeq(1, 2, 3, 4, 5, 6)),
    )
    assert(hashTestArrayHelper(IndexedSeq(1, 2)) != hashTestArrayHelper(IndexedSeq(3, 4, 5, 6, 7)))
    assertEq(
      hashTestStructHelper(Row("wolf", 8, .009f), fields),
      hashTestStructHelper(Row("wolf", 8, .009f), fields),
    )
    assert(
      hashTestStructHelper(Row("w", 8, .009f), fields) != hashTestStructHelper(
        Row("opaque", 8, .009f),
        fields,
      )
    )
  }

  def hashTestNumHelper(v: SValue)(implicit ctx: ExecuteContext): Int = {
    val fb = EmitFunctionBuilder[Int](ctx, "test_hash")
    val mb = fb.apply_method
    mb.emit(EmitCodeBuilder.scopedCode(mb) { cb =>
      val hash = v.hash(cb)
      hash.value
    })
    fb.result()(ctx.theHailClassLoader)()
  }

  def hashTestStringHelper(toHash: String)(implicit ctx: ExecuteContext): Int = {
    val pstring = PCanonicalString()
    val fb = EmitFunctionBuilder[Region, Int](ctx, "test_hash")
    val mb = fb.apply_method
    mb.emit(EmitCodeBuilder.scopedCode(mb) { cb =>
      val region = fb.emb.getCodeParam[Region](1)
      val st = SStringPointer(pstring)
      val stringToHash = st.constructFromString(cb, region, toHash)
      val i = stringToHash
      val hash = i.hash(cb)
      hash.value
    })
    val region = Region(pool = ctx.r.pool)
    fb.result()(ctx.theHailClassLoader)(region)
  }

  def hashTestArrayHelper(toHash: IndexedSeq[Int])(implicit ctx: ExecuteContext): Int = {
    val pArray = PCanonicalArray(PInt32(true))
    val fb = EmitFunctionBuilder[Long, Int](ctx, "test_hash")
    val mb = fb.apply_method
    mb.emit(EmitCodeBuilder.scopedCode(mb) { cb =>
      val arrayPointer = fb.emb.getCodeParam[Long](1)
      val arrayToHash = pArray.loadCheapSCode(cb, arrayPointer)
      val hash = arrayToHash.hash(cb)
      hash.value
    })
    val region = Region(pool = ctx.r.pool)
    val arrayPointer = pArray.unstagedStoreJavaObject(ctx.stateManager, toHash, region)
    fb.result()(ctx.theHailClassLoader)(arrayPointer)
  }

  def hashTestStructHelper(toHash: Row, fields: IndexedSeq[PField])(implicit ctx: ExecuteContext)
    : Int = {
    val pStruct = PCanonicalStruct(fields)
    val fb = EmitFunctionBuilder[Long, Int](ctx, "test_hash")
    val mb = fb.apply_method
    mb.emit(EmitCodeBuilder.scopedCode(mb) { cb =>
      val structPointer = fb.emb.getCodeParam[Long](1)
      val structToHash = pStruct.loadCheapSCode(cb, structPointer)
      val hash = structToHash.hash(cb)
      hash.value
    })
    val region = Region(pool = ctx.r.pool)
    val structPointer = pStruct.unstagedStoreJavaObject(ctx.stateManager, toHash, region)
    fb.result()(ctx.theHailClassLoader)(structPointer)
  }
}
