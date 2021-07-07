package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.annotations.{Annotation, Region, RegionValueBuilder, SafeNDArray, SafeRow}
import is.hail.types.encoded._
import is.hail.types.physical.{PCanonicalArray, PCanonicalNDArray, PCanonicalStringOptional, PCanonicalStringRequired, PCanonicalStruct, PFloat32Required, PFloat64Required, PInt32Optional, PInt32Required, PInt64Optional, PInt64Required, PType}
import is.hail.io.{InputBuffer, MemoryBuffer, MemoryInputBuffer, MemoryOutputBuffer, OutputBuffer}
import is.hail.rvd.AbstractRVDSpec
import is.hail.utils._
import org.apache.spark.sql.Row
import org.json4s.jackson.Serialization
import org.testng.annotations.{DataProvider, Test}

class ETypeSuite extends HailSuite {

  @DataProvider(name="etypes")
  def etypes(): Array[Array[Any]] = {
    Array[EType](
      EInt32Required,
      EInt32Optional,
      EInt64Required,
      EFloat32Optional,
      EFloat32Required,
      EFloat64Optional,
      EFloat64Required,
      EBooleanOptional,
      EBinaryRequired,
      EBinaryOptional,
      EBinaryRequired,
      EArray(EInt32Required, required = false),
      EArray(EArray(EInt32Optional, required = true), required = true),
      EBaseStruct(FastIndexedSeq(), required = true),
      EBaseStruct(FastIndexedSeq(EField("x", EBinaryRequired, 0), EField("y", EFloat64Optional, 1)), required = true),
      ENDArrayColumnMajor(EFloat64Required , 3)
    ).map(t => Array(t: Any))
  }

  @Test def testDataProvider(): Unit = {
    etypes()
  }

  @Test(dataProvider="etypes")
  def testSerialization(etype: EType): Unit = {
    implicit val formats = AbstractRVDSpec.formats
    val s = Serialization.write(etype)
    assert(Serialization.read[EType](s) == etype)
  }

  def encodeDecode(inPType: PType, eType: EType, outPType: PType, data: Annotation): Annotation = {
    val fb = EmitFunctionBuilder[Long, OutputBuffer, Unit](ctx, "fb")
    val arg1 = fb.apply_method.getCodeParam[Long](1)
    val arg2 = fb.apply_method.getCodeParam[OutputBuffer](2)
    val enc = eType.buildEncoderMethod(inPType.sType, fb.apply_method.ecb)
    fb.emit(enc.invokeCode(arg1, arg2))

    val x = inPType.unstagedStoreJavaObject(data, ctx.r)

    val buffer = new MemoryBuffer
    val ob = new MemoryOutputBuffer(buffer)

    fb.resultWithIndex()(ctx.fs, 0, ctx.r).apply(x, ob)
    ob.flush()
    buffer.clearPos()

    val fb2 = EmitFunctionBuilder[Region, InputBuffer, Long](ctx, "fb2")
    val regArg = fb2.apply_method.getCodeParam[Region](1)
    val ibArg = fb2.apply_method.getCodeParam[InputBuffer](2)
    val dec = eType.buildDecoderMethod(outPType.virtualType, fb2.apply_method.ecb)
    fb2.emitWithBuilder[Long] { cb =>
      val decoded = cb.invokeSCode(dec, regArg, ibArg)
      outPType.store(cb, regArg, decoded, deepCopy = false)
    }

    val result = fb2.resultWithIndex()(ctx.fs, 0, ctx.r).apply(ctx.r, new MemoryInputBuffer(buffer))
    SafeRow.read(outPType, result)
  }

  def assertEqualEncodeDecode(inPType: PType, eType: EType, outPType: PType, data: Annotation): Unit = {
    val encodeDecodeResult = encodeDecode(inPType, eType, outPType, data)
    assert(encodeDecodeResult == data)
  }

  @Test def testDifferentRequirednessEncodeDecode() {

    val inPType = PCanonicalArray(
      PCanonicalStruct(true,
        "a" -> PInt32Required,
        "b" -> PInt32Optional,
        "c" -> PCanonicalStringRequired,
        "d" -> PCanonicalArray(PCanonicalStruct(true, "x" -> PInt64Required), true)),
      false)
    val etype = EArray(
      EBaseStruct(FastIndexedSeq(
      EField("a", EInt32Required, 0),
        EField("b", EInt32Optional, 1),
        EField("c", EBinaryOptional, 2),
        EField("d", EArray(EBaseStruct(FastIndexedSeq(EField("x", EInt64Optional, 0)), false), true), 3)),
        true),
      false)
    val outPType = PCanonicalArray(
      PCanonicalStruct(false,
        "a" -> PInt32Optional,
        "b" -> PInt32Optional,
        "c" -> PCanonicalStringOptional,
        "d" -> PCanonicalArray(PCanonicalStruct(false, "x" -> PInt64Optional), false)),
      false)

    val data = FastIndexedSeq(Row(1, null, "abc", FastIndexedSeq(Row(7L), Row(8L))))

    assertEqualEncodeDecode(inPType, etype, outPType, data)
  }

  @Test def testNDArrayEncodeDecode(): Unit = {
    val pTypeInt0 = PCanonicalNDArray(PInt32Required, 0, true)
    val eTypeInt0 = ENDArrayColumnMajor(EInt32Required, 0, true)
    val dataInt0 = new SafeNDArray(IndexedSeq[Long](), FastIndexedSeq(0))

    assertEqualEncodeDecode(pTypeInt0, eTypeInt0, pTypeInt0, dataInt0)

    val pTypeFloat1 = PCanonicalNDArray(PFloat32Required, 1, true)
    val eTypeFloat1 = ENDArrayColumnMajor(EFloat32Required, 1, true)
    val dataFloat1 = new SafeNDArray(IndexedSeq(5L), (0 until 5).map(_.toFloat))

    assertEqualEncodeDecode(pTypeFloat1, eTypeFloat1, pTypeFloat1, dataFloat1)

    val pTypeInt2 = PCanonicalNDArray(PInt32Required, 2, true)
    val eTypeInt2 = ENDArrayColumnMajor(EInt32Required, 2, true)
    val dataInt2 = new SafeNDArray(IndexedSeq(2L, 2L), FastIndexedSeq(10, 20, 30, 40))

    assertEqualEncodeDecode(pTypeInt2, eTypeInt2, pTypeInt2, dataInt2)

    val pTypeDouble3 = PCanonicalNDArray(PFloat64Required, 3, false)
    val eTypeDouble3 = ENDArrayColumnMajor(EFloat64Required, 3, false)
    val dataDouble3 = new SafeNDArray(IndexedSeq(3L, 2L, 1L), FastIndexedSeq(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

    assert(encodeDecode(pTypeDouble3, eTypeDouble3, pTypeDouble3, dataDouble3) ==
      new SafeNDArray(IndexedSeq(3L, 2L, 1L), FastIndexedSeq(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))

    // Test for skipping
    val pStructContainingNDArray = PCanonicalStruct(true,
      "a" -> pTypeInt2,
      "b" -> PInt32Optional
    )
    val pOnlyReadB = PCanonicalStruct(true,
      "b" -> PInt32Optional
    )
    val eStructContainingNDArray = EBaseStruct(
      FastIndexedSeq(
        EField("a", ENDArrayColumnMajor(EInt32Required, 2, true), 0),
        EField("b", EInt32Required, 1)
      ),
      true)

    val dataStruct = Row(dataInt2, 3)

    assert(encodeDecode(pStructContainingNDArray, eStructContainingNDArray, pOnlyReadB, dataStruct) ==
      Row(3))
  }
}
