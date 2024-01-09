package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.annotations.{Annotation, Region, SafeNDArray, SafeRow}
import is.hail.asm4s.Code
import is.hail.io._
import is.hail.rvd.AbstractRVDSpec
import is.hail.types.encoded._
import is.hail.types.physical._
import is.hail.utils._

import org.json4s.jackson.Serialization

import org.apache.spark.sql.Row
import org.testng.annotations.{DataProvider, Test}

class ETypeSuite extends HailSuite {

  @DataProvider(name = "etypes")
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
      EBaseStruct(FastSeq(), required = true),
      EBaseStruct(
        FastSeq(EField("x", EBinaryRequired, 0), EField("y", EFloat64Optional, 1)),
        required = true,
      ),
      ENDArrayColumnMajor(EFloat64Required, 3),
    ).map(t => Array(t: Any))
  }

  @Test def testDataProvider(): Unit =
    etypes()

  @Test(dataProvider = "etypes")
  def testSerialization(etype: EType): Unit = {
    implicit val formats = AbstractRVDSpec.formats
    val s = Serialization.write(etype)
    assert(Serialization.read[EType](s) == etype)
  }

  def encodeDecode(inPType: PType, eType: EType, outPType: PType, data: Annotation): Annotation = {
    val fb = EmitFunctionBuilder[Long, OutputBuffer, Unit](ctx, "fb")
    val enc = eType.buildEncoderMethod(inPType.sType, fb.apply_method.ecb)
    fb.emitWithBuilder { cb =>
      val arg1 = inPType.loadCheapSCode(cb, fb.apply_method.getCodeParam[Long](1))
      val arg2 = fb.apply_method.getCodeParam[OutputBuffer](2)
      cb.invokeVoid(enc, cb.this_, arg1, arg2)
      Code._empty
    }

    val x = inPType.unstagedStoreJavaObject(ctx.stateManager, data, ctx.r)

    val buffer = new MemoryBuffer
    val ob = new MemoryOutputBuffer(buffer)

    fb.resultWithIndex()(theHailClassLoader, ctx.fs, ctx.taskContext, ctx.r).apply(x, ob)
    ob.flush()
    buffer.clearPos()

    val fb2 = EmitFunctionBuilder[Region, InputBuffer, Long](ctx, "fb2")
    val regArg = fb2.apply_method.getCodeParam[Region](1)
    val ibArg = fb2.apply_method.getCodeParam[InputBuffer](2)
    val dec = eType.buildDecoderMethod(outPType.virtualType, fb2.apply_method.ecb)
    fb2.emitWithBuilder[Long] { cb =>
      val decoded = cb.invokeSCode(dec, cb.this_, regArg, ibArg)
      outPType.store(cb, regArg, decoded, deepCopy = false)
    }

    val result = fb2.resultWithIndex()(theHailClassLoader, ctx.fs, ctx.taskContext, ctx.r).apply(
      ctx.r,
      new MemoryInputBuffer(buffer),
    )
    SafeRow.read(outPType, result)
  }

  def assertEqualEncodeDecode(inPType: PType, eType: EType, outPType: PType, data: Annotation)
    : Unit = {
    val encodeDecodeResult = encodeDecode(inPType, eType, outPType, data)
    assert(encodeDecodeResult == data)
  }

  @Test def testDifferentRequirednessEncodeDecode() {

    val inPType = PCanonicalArray(
      PCanonicalStruct(
        true,
        "a" -> PInt32Required,
        "b" -> PInt32Optional,
        "c" -> PCanonicalStringRequired,
        "d" -> PCanonicalArray(PCanonicalStruct(true, "x" -> PInt64Required), true),
      ),
      false,
    )
    val etype = EArray(
      EBaseStruct(
        FastSeq(
          EField("a", EInt32Required, 0),
          EField("b", EInt32Optional, 1),
          EField("c", EBinaryOptional, 2),
          EField("d", EArray(EBaseStruct(FastSeq(EField("x", EInt64Optional, 0)), false), true), 3),
        ),
        true,
      ),
      false,
    )
    val outPType = PCanonicalArray(
      PCanonicalStruct(
        false,
        "a" -> PInt32Optional,
        "b" -> PInt32Optional,
        "c" -> PCanonicalStringOptional,
        "d" -> PCanonicalArray(PCanonicalStruct(false, "x" -> PInt64Optional), false),
      ),
      false,
    )

    val data = FastSeq(Row(1, null, "abc", FastSeq(Row(7L), Row(8L))))

    assertEqualEncodeDecode(inPType, etype, outPType, data)
  }

  @Test def testNDArrayEncodeDecode(): Unit = {
    val pTypeInt0 = PCanonicalNDArray(PInt32Required, 0, true)
    val eTypeInt0 = ENDArrayColumnMajor(EInt32Required, 0, true)
    val dataInt0 = new SafeNDArray(IndexedSeq[Long](), FastSeq(0))

    assertEqualEncodeDecode(pTypeInt0, eTypeInt0, pTypeInt0, dataInt0)

    val pTypeFloat1 = PCanonicalNDArray(PFloat32Required, 1, true)
    val eTypeFloat1 = ENDArrayColumnMajor(EFloat32Required, 1, true)
    val dataFloat1 = new SafeNDArray(IndexedSeq(5L), (0 until 5).map(_.toFloat))

    assertEqualEncodeDecode(pTypeFloat1, eTypeFloat1, pTypeFloat1, dataFloat1)

    val pTypeInt2 = PCanonicalNDArray(PInt32Required, 2, true)
    val eTypeInt2 = ENDArrayColumnMajor(EInt32Required, 2, true)
    val dataInt2 = new SafeNDArray(IndexedSeq(2L, 2L), FastSeq(10, 20, 30, 40))

    assertEqualEncodeDecode(pTypeInt2, eTypeInt2, pTypeInt2, dataInt2)

    val pTypeDouble3 = PCanonicalNDArray(PFloat64Required, 3, false)
    val eTypeDouble3 = ENDArrayColumnMajor(EFloat64Required, 3, false)
    val dataDouble3 = new SafeNDArray(IndexedSeq(3L, 2L, 1L), FastSeq(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

    assert(encodeDecode(pTypeDouble3, eTypeDouble3, pTypeDouble3, dataDouble3) ==
      new SafeNDArray(IndexedSeq(3L, 2L, 1L), FastSeq(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))

    // Test for skipping
    val pStructContainingNDArray = PCanonicalStruct(true, "a" -> pTypeInt2, "b" -> PInt32Optional)
    val pOnlyReadB = PCanonicalStruct(true, "b" -> PInt32Optional)
    val eStructContainingNDArray = EBaseStruct(
      FastSeq(
        EField("a", ENDArrayColumnMajor(EInt32Required, 2, true), 0),
        EField("b", EInt32Required, 1),
      ),
      true,
    )

    val dataStruct = Row(dataInt2, 3)

    assert(encodeDecode(
      pStructContainingNDArray,
      eStructContainingNDArray,
      pOnlyReadB,
      dataStruct,
    ) ==
      Row(3))
  }

  @Test def testArrayOfString(): Unit = {
    val etype = EArray(EBinary(false), false)
    val toEncode = PCanonicalArray(PCanonicalStringRequired, false)
    val toDecode = PCanonicalArray(PCanonicalStringOptional, false)
    val longListOfStrings = (0 until 36).map(idx => s"foo_name_sample_$idx")
    val data = longListOfStrings

    assert(encodeDecode(toEncode, etype, toDecode, data) == data)
  }
}
