package is.hail.asm4s

import is.hail.HailSuite
import is.hail.annotations.Region
import is.hail.expr.ir.{EmitCodeBuilder, EmitFunctionBuilder, IEmitCode}
import is.hail.types.physical.{PCanonicalArray, PCanonicalBaseStruct, PCanonicalString, PCanonicalStruct, PField, PFloat32, PInt32, PType}
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.concrete.{SBaseStructPointer, SBaseStructPointerCode, SIndexablePointer, SIndexablePointerCode, SStringPointer}
import is.hail.types.physical.stypes.interfaces.{SString, SStringCode}
import is.hail.types.physical.stypes.primitives.{SFloat32Code, SFloat64Code, SInt32Code, SInt64Code}
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class CodeSuite extends HailSuite {

  @Test def testForLoop() {
    val fb = EmitFunctionBuilder[Int](ctx, "foo")
    val mb = fb.apply_method
    val i = mb.newLocal[Int]()
    val sum = mb.newLocal[Int]()
    val code = Code(
      sum := 0,
      Code.forLoop(i := 0, i < 5, i := i + 1, sum :=  sum + i),
      sum.load()
    )
    fb.emit(code)
    val result = fb.resultWithIndex()(ctx.fs, 0, ctx.r)()
    assert(result == 10)
  }
  @Test def testHash() {
    val fields = IndexedSeq(PField("a", PCanonicalString(), 0), PField("b", PInt32(), 1), PField("c", PFloat32(), 2))
    assert(hashTestNumHelper(new SInt32Code(6)) == hashTestNumHelper(new SInt32Code(6)))
    assert(hashTestNumHelper(new SInt64Code(5000000000l)) == hashTestNumHelper(new SInt64Code(5000000000l)))
    assert(hashTestNumHelper(new SFloat32Code(3.14f)) == hashTestNumHelper(new SFloat32Code(3.14f)))
    assert(hashTestNumHelper(new SFloat64Code(5000000000.89d)) == hashTestNumHelper(new SFloat64Code(5000000000.89d)))
    assert(hashTestStringHelper("dog")== hashTestStringHelper("dog"))
    assert(hashTestArrayHelper(IndexedSeq(1,2,3,4,5,6)) == hashTestArrayHelper(IndexedSeq(1,2,3,4,5,6)))
    assert(hashTestArrayHelper(IndexedSeq(1,2)) != hashTestArrayHelper(IndexedSeq(3,4,5,6,7)))
    assert(hashTestStructHelper(Row("wolf", 8, .009f), fields) == hashTestStructHelper(Row("wolf", 8, .009f), fields))
    assert(hashTestStructHelper(Row("w", 8, .009f), fields) != hashTestStructHelper(Row("opaque", 8, .009f), fields))
  }

  def hashTestNumHelper(toHash: SCode): Int = {
    val fb = EmitFunctionBuilder[Int](ctx, "test_hash")
    val mb = fb.apply_method
    mb.emit(EmitCodeBuilder.scopedCode(mb) { cb =>
      val i = toHash.memoize(cb, "value_to_hash")
      val hash = i.hash(cb)
      hash.intCode(cb)
    })
    fb.result()()()
  }

  def hashTestStringHelper(toHash: String): Int = {
    val pstring = PCanonicalString()
    val fb = EmitFunctionBuilder[Region, Int](ctx, "test_hash")
    val mb = fb.apply_method
    mb.emit(EmitCodeBuilder.scopedCode(mb) { cb =>
      val region = fb.emb.getCodeParam[Region](1)
      val st = SStringPointer(pstring)
      val stringToHash = st.constructFromString(cb, region, toHash)
      val i = stringToHash.memoize(cb, "value_to_hash")
      val hash = i.hash(cb)
      hash.intCode(cb)
    })
    val region = Region(pool=pool)
    fb.result()()(region)
  }

  def hashTestArrayHelper(toHash: IndexedSeq[Int]): Int = {
    val pArray = PCanonicalArray(PInt32(true))
    val fb = EmitFunctionBuilder[Long, Int](ctx, "test_hash")
    val mb = fb.apply_method
    mb.emit(EmitCodeBuilder.scopedCode(mb) { cb =>
      val arrayPointer = fb.emb.getCodeParam[Long](1)
      val sIndexPointer = SIndexablePointer(pArray)
      val arrayToHash = new SIndexablePointerCode(sIndexPointer, arrayPointer)
      val i = arrayToHash.memoize(cb, "value_to_hash")
      val hash = i.hash(cb)
      hash.intCode(cb)
    })
    val region = Region(pool=pool)
    val arrayPointer = pArray.unstagedStoreJavaObject(toHash, region)
    fb.result()()(arrayPointer)
  }

  def hashTestStructHelper(toHash: Row, fields : IndexedSeq[PField]): Int = {
    val pStruct = PCanonicalStruct(fields)
    val fb = EmitFunctionBuilder[Long, Int](ctx, "test_hash")
    val mb = fb.apply_method
    mb.emit(EmitCodeBuilder.scopedCode(mb) { cb =>
      val structPointer = fb.emb.getCodeParam[Long](1)
      val sIndexPointer = SBaseStructPointer(pStruct)
      val structToHash = new SBaseStructPointerCode(sIndexPointer, structPointer)
      val i = structToHash.memoize(cb, "value_to_hash")
      val hash = i.hash(cb)
      hash.intCode(cb)
    })
    val region = Region(pool=pool)
    val structPointer = pStruct.unstagedStoreJavaObject(toHash, region)
    fb.result()()(structPointer)
  }
}
