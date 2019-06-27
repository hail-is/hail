package is.hail.expr.ir

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import is.hail.HailSuite
import is.hail.annotations.{Memory, Region, SafeRow, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.agg._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.{CodecSpec, InputBuffer, OutputBuffer}
import is.hail.utils.FastIndexedSeq
import org.apache.spark.sql.Row
import org.testng.annotations.Test

object TUtils {
  def printBytes(region: Region, off: Long, n: Int): String =
    region.loadBytes(off, n).map("%02x".format(_)).mkString(" ")

  def printBytes(region: Code[Region], off: Code[Long], n: Int): Code[String] =
    Code.invokeScalaObject[Region, Long, Int, String](TUtils.getClass, "printBytes", region, off, n)

  def printRow(region: Region, off: Long): String =
    SafeRow(TStruct("a" -> TString(), "b" -> TInt32()).physicalType, region, off).toString

  def printRow(region: Code[Region], off: Code[Long]): Code[String] =
    Code.invokeScalaObject[Region, Long, String](TUtils.getClass, "printRow", region, off)
}

class AggregatorsSuite2 extends HailSuite {

  @Test def simpleTest() {
    val elt = Ref("x", TInt32())
    val row = MakeStruct(FastIndexedSeq("a" -> Str("foo"), "b" -> elt))
    val arrayToAgg: IR = ArrayMap(
      ArrayRange(0, 15, 1), elt.name,
      If(invoke("%", elt, 2).ceq(0), NA(row.typ), row))

//    val arrayToAgg = MakeArray(FastIndexedSeq(), TArray(row.typ))

    val (pt: PArray, f) = Compile[Long](arrayToAgg)

    val fb = EmitFunctionBuilder[Region, Long, Long]
    val r = fb.apply_method.getArg[Region](1)
    val off = fb.apply_method.getArg[Long](2)
    val agg = new PrevNonNullAggregator(row.typ.physicalType)
    val s = agg.createState(fb.apply_method)

    val len = fb.newField[Int]
    val idx = fb.newField[Int]

    val srvb = new StagedRegionValueBuilder(fb.apply_method, PTuple(FastIndexedSeq(row.typ.physicalType)))

    fb.emit(
      Code(
        s.r := r,
        agg.initOp(s, Array()),
        idx := 0,
        len := pt.loadLength(r, off),
        Code.whileLoop(idx < len,
          agg.seqOp(s, Array(RVAVariable(EmitTriplet(Code._empty, pt.isElementMissing(r, off, idx), pt.loadElement(r, off, idx)), row.typ.physicalType))),
          idx := idx + 1),
        srvb.start(),
        agg.result(s, srvb),
        srvb.offset))

    val aggf = fb.resultWithIndex()


    Region.scoped { region =>
      val offset = f(0)(region)
      println(SafeRow(PTuple(FastIndexedSeq(row.typ.physicalType)), region, aggf(0)(region, offset)))
    }
  }

  @Test def simpleTest2() {
    val rowType = TStruct("a" -> TString(), "b" -> TInt32())
    val arrayType = TArray(rowType)

    val value = Literal(TArray(arrayType),
      FastIndexedSeq(
        FastIndexedSeq(Row("a", 0), Row("b", 0), Row("c", 0), Row("f", 0)),
        FastIndexedSeq(Row("a", 1), null, Row("c", 1), null),
        FastIndexedSeq(Row("a", 2), Row("b", 2), Row("c", 2), Row("f", 2)),
        FastIndexedSeq(Row("a", 3), Row("b", 3), Row("c", 3), Row("f", 3)),
        FastIndexedSeq(Row("a", 4), Row("b", 4), Row("c", 4), Row("f", 4)),
        FastIndexedSeq(null, null, null, Row("f", 5))))

    val (pt: PArray, f) = Compile[Long](value)

    val fb = EmitFunctionBuilder[Region, Long, Long]
    val r = fb.apply_method.getArg[Region](1)
    val off = fb.apply_method.getArg[Long](2)
    val pnn = new PrevNonNullAggregator(rowType.physicalType)
    val agg = new ArrayElementRegionValueAggregator(Array(pnn), false)
    val agg2 = new ArrayElementRegionValueAggregator2(Array(pnn))
    val s = agg.createState(fb.apply_method)

    val len = fb.newField[Int]
    val idx = fb.newField[Int]
    val idx2 = fb.newField[Int]

    val resType = PTuple(FastIndexedSeq(PArray(PTuple(FastIndexedSeq(rowType.physicalType)))))
    val srvb = new StagedRegionValueBuilder(fb.apply_method, resType)

    val a = fb.newField[Long]
    val lenVar = RVAVariable(EmitTriplet(Code._empty, pt.isElementMissing(r, off, idx), arrayType.physicalType.loadLength(r, a)), PInt32())
    val idxVar = RVAVariable(EmitTriplet(Code._empty, false, idx2), PInt32())

    val eltVar = RVAVariable(EmitTriplet(Code._empty, arrayType.physicalType.isElementMissing(r, a, idx2), arrayType.physicalType.loadElement(r, a, idx2)), rowType.physicalType)
    val seq = pnn.seqOp(s.nested(0), Array(eltVar))

    fb.emit(
      Code(
        s.r := r,
        agg.initOp(s, Array()),
        idx := 0,
        len := pt.loadLength(r, off),
        Code.whileLoop(idx < len,
          a := pt.loadElement(r, off, idx),
          agg.seqOp(s, Array(lenVar)),
          idx2 := 0,
          Code.whileLoop(idx2 < s.lenRef,
            agg2.seqOp(s, Array(idxVar, RVAVariable(EmitTriplet(seq, false, Code._empty), PVoid))),
            idx2 := idx2 + 1),
          idx := idx + 1),
        srvb.start(),
        agg.result(s, srvb),
        srvb.offset))

    val aggf = fb.resultWithIndex()


    Region.scoped { region =>
      val offset = f(0)(region)
      val res = aggf(0)(region, offset)
      println(SafeRow(resType, region, res))
    }
  }

  @Test def serializeDeserializeAndCombOp() {
    val rowType = TStruct("a" -> TString(), "b" -> TInt32())
    val arrayType = TArray(rowType)

    val value = Literal(TArray(arrayType),
      FastIndexedSeq(
        FastIndexedSeq(Row("a", 0), Row("b", 0), Row("c", 0), Row("f", 0)),
        FastIndexedSeq(Row("a", 1), null, Row("c", 1), null),
        FastIndexedSeq(Row("a", 2), Row("b", 2), Row("c", 2), Row("f", 2))))
    val value2 = Literal(TArray(arrayType),
      FastIndexedSeq(
        FastIndexedSeq(Row("a", 3), Row("b", 3), Row("c", 3), Row("f", 3)),
        FastIndexedSeq(Row("a", 4), Row("b", 4), Row("c", 4), Row("f", 4)),
        FastIndexedSeq(null, null, null, Row("f", 5))))

    val nestedType = PArray(arrayType.physicalType)
    val (pt: PArray, f) = Compile[Long](MakeArray(FastIndexedSeq(value, value2), TArray(value.typ)))

    val fb = EmitFunctionBuilder[Region, Long, Long]
    val r = fb.apply_method.getArg[Region](1)
    val off = fb.apply_method.getArg[Long](2)
    val pnn = new PrevNonNullAggregator(rowType.physicalType)
    val agg = new ArrayElementRegionValueAggregator(Array(pnn), false)
    val agg2 = new ArrayElementRegionValueAggregator2(Array(pnn))
    val s1 = agg.createState(fb.apply_method)
    val s2 = agg.createState(fb.apply_method)

    val len2 = fb.newField[Int]
    val idx2 = fb.newField[Int]

    val off2 = fb.newField[Long]
    val len = fb.newField[Int]
    val idx = fb.newField[Int]

    val serialized = fb.newField[Array[Array[Byte]]]
    val baos = fb.newField[ByteArrayOutputStream]
    val bais = fb.newField[ByteArrayInputStream]
    val ob = fb.newField[OutputBuffer]
    val ib = fb.newField[InputBuffer]

    val resType = PTuple(FastIndexedSeq(PArray(PTuple(FastIndexedSeq(rowType.physicalType)))))
    val srvb = new StagedRegionValueBuilder(fb.apply_method, resType)
    val spec = CodecSpec.defaultUncompressed

    val idx3 = fb.newField[Int]

    val a = fb.newField[Long]
    val lenVar = RVAVariable(EmitTriplet(Code._empty, pt.isElementMissing(r, off2, idx2), arrayType.physicalType.loadLength(r, a)), PInt32())
    val idxVar = RVAVariable(EmitTriplet(Code._empty, false, idx3), PInt32())
    val eltVar = RVAVariable(EmitTriplet(Code._empty, arrayType.physicalType.isElementMissing(r, a, idx3), arrayType.physicalType.loadElement(r, a, idx3)), rowType.physicalType)
    val seq = pnn.seqOp(s1.nested(0), Array(eltVar))

    fb.emit(
      Code(
        idx := 0,
        len := pt.loadLength(r, off),
        serialized := Code.newArray[Array[Byte]](len),
        Code.whileLoop(idx < len,
          baos := Code.newInstance[ByteArrayOutputStream](),
          ob := spec.buildCodeOutputBuffer(baos),
          s1.r := Code.newInstance[Region](),
          off2 := nestedType.loadElement(r, off, idx),
          agg.initOp(s1, Array()),
          idx2 := 0,
          len2 := nestedType.loadLength(r, off2),
          Code.whileLoop(idx2 < len2,
            a := pt.loadElement(r, off2, idx2),
            agg.seqOp(s1, Array(lenVar)),
            idx3 := 0,
            Code.whileLoop(idx3 < s1.lenRef,
              agg2.seqOp(s1, Array(idxVar, RVAVariable(EmitTriplet(seq, false, Code._empty), PVoid))),
              idx3 := idx3 + 1),
            idx2 := idx2 + 1),
          s1.serialize(spec)(ob),
          ob.invoke[Unit]("flush"),
          serialized.load().update(idx, baos.invoke[Array[Byte]]("toByteArray")),
          idx := idx + 1),
        bais := Code.newInstance[ByteArrayInputStream, Array[Byte]](serialized.load()(0)),
        ib := spec.buildCodeInputBuffer(bais),
        s1.r := Code.newInstance[Region](),
        s1.unserialize(spec)(ib),
        idx := 1,
        Code.whileLoop(idx < len,
          bais := Code.newInstance[ByteArrayInputStream, Array[Byte]](serialized.load()(idx)),
          ib := spec.buildCodeInputBuffer(bais),
          s2.r := Code.newInstance[Region](),
          s2.unserialize(spec)(ib),
          agg.combOp(s1, s2),
          idx := idx + 1),
        srvb.start(),
        agg.result(s1, srvb),
        srvb.offset))

    val aggf = fb.resultWithIndex()

    Region.scoped { region =>
      val offset = f(0)(region)
      val res = aggf(0)(region, offset)
      println(SafeRow(resType, region, res))
    }
  }

}
