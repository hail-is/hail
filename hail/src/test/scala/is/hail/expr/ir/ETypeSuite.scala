package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.annotations.{Annotation, Region, RegionValueBuilder, SafeRow}
import is.hail.types.encoded._
import is.hail.types.physical.{PCanonicalArray, PCanonicalNDArray, PCanonicalStringOptional, PCanonicalStringRequired, PCanonicalStruct, PFloat64Required, PInt32Optional, PInt32Required, PInt64Optional, PInt64Required, PType}
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
      ENDArray(EFloat64Required , 3)
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

  def assertEqualEncodeDecode(inPType: PType, eType: EType, outPType: PType, data: Annotation): Unit = {
    assert(inPType.virtualType == outPType.virtualType)

    val fb = EmitFunctionBuilder[Long, OutputBuffer, Unit](ctx, "fb")
    val arg1 = fb.apply_method.getCodeParam[Long](1)
    val arg2 = fb.apply_method.getCodeParam[OutputBuffer](2)
    val enc = eType.buildEncoderMethod(inPType, fb.apply_method.ecb)
    fb.emit(enc.invokeCode(arg1, arg2))

    val rvb = new RegionValueBuilder(ctx.r)
    rvb.start(inPType)
    rvb.addAnnotation(inPType.virtualType, data)
    val x = rvb.end()

    val buffer = new MemoryBuffer
    val ob = new MemoryOutputBuffer(buffer)

    fb.resultWithIndex()(0, ctx.r).apply(x, ob)
    ob.flush()
    buffer.clearPos()
    println("Got encode fb.result")

    val fb2 = EmitFunctionBuilder[Region, InputBuffer, Long](ctx, "fb2")
    val regArg = fb2.apply_method.getCodeParam[Region](1)
    val ibArg = fb2.apply_method.getCodeParam[InputBuffer](2)
    val dec = eType.buildDecoderMethod(outPType, fb2.apply_method.ecb)
    fb2.emit(dec.invokeCode(regArg, ibArg))
    println("Emitted decoder")

    val result = fb2.resultWithIndex()(0, ctx.r).apply(ctx.r, new MemoryInputBuffer(buffer))
    println("Got result")
    assert(SafeRow.read(outPType, result) == data)
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
    val pTypeInt2 = PCanonicalNDArray(PInt32Required, 2, true)
    val eTypeInt2 = ENDArray(EInt32Required, 2, true)
    val dataInt2 = Row(Row(2L, 2L), Row(16L, 8L), FastIndexedSeq(1, 2, 3, 4))

    assertEqualEncodeDecode(pTypeInt2, eTypeInt2, pTypeInt2, dataInt2)

    val pTypeFloat3 = PCanonicalNDArray(PFloat64Required, 3, false)
    val eTypeFloat3 = ENDArray(EFloat64Required, 3, false)
    val dataFloat3 = Row(Row(3L, 2L, 1L), Row(32L, 16L, 16L), FastIndexedSeq(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

    assertEqualEncodeDecode(pTypeFloat3, eTypeFloat3, pTypeFloat3, dataFloat3)
  }
}
