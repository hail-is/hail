package is.hail.expr.ir

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import is.hail.HailSuite
import is.hail.annotations.{Region, SafeRow, ScalaToRegionValue, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.agg._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.{CodecSpec, InputBuffer, OutputBuffer}
import is.hail.utils._
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

class Aggregators2Suite extends HailSuite {

  val rowType = PStruct("a" -> PString(), "b" -> PInt64())
  val arrayType = PArray(rowType)
  val streamType = PArray(arrayType)

  val pnnAgg = new PrevNonNullAggregator(rowType)
  val countAgg = CountAggregator
  val sumAgg = new SumAggregator(PInt64())

  val aggs: Array[StagedRegionValueAggregator] = Array(pnnAgg, countAgg, sumAgg)

  val lcAgg = new ArrayElementLengthCheckAggregator(aggs, false)
  val eltAgg = new ArrayElementwiseOpAggregator(aggs)

  val value = FastIndexedSeq(
    FastIndexedSeq(Row("a", 0L), Row("b", 0L), Row("c", 0L), Row("f", 0L)),
    FastIndexedSeq(Row("a", 1L), null, Row("c", 1L), null),
    FastIndexedSeq(Row("a", 2L), Row("b", 2L), null, Row("f", 2L)),
    FastIndexedSeq(Row("a", 3L), Row("b", 3L), Row("c", 3L), Row("f", 3L)),
    FastIndexedSeq(Row("a", 4L), Row("b", 4L), Row("c", 4L), null),
    FastIndexedSeq(null, null, null, Row("f", 5L)))

  val expected =
    FastIndexedSeq(
      Row(Row("a", 4), 6L, 10L),
      Row(Row("b", 4), 6L, 9L),
      Row(Row("c", 4), 6L, 8L),
      Row(Row("f", 5), 6L, 10L))

  def rowVar(r: Code[Region], a: Code[Long], i: Code[Int]): RVAVariable =
    RVAVariable(EmitTriplet(Code._empty,
      arrayType.isElementMissing(r, a, i),
      arrayType.loadElement(r, a, i)), rowType)

  def bVar(r: Code[Region], a: Code[Long], i: Code[Int]): RVAVariable = {
    val RVAVariable(row, _) = rowVar(r, a, i)
    RVAVariable(EmitTriplet(row.setup,
      row.m || rowType.isFieldMissing(r, row.value[Long], 1),
      r.loadLong(rowType.loadField(r, row.value[Long], 1))), PInt64())
  }

  def seqOne(s: Array[RVAState], a: Code[Long], i: Code[Int]): Code[Unit] = {
    val r = s(0).r
    Code(
      pnnAgg.seqOp(s(0), Array(rowVar(r, a, i))),
      countAgg.seqOp(s(1), Array()),
      sumAgg.seqOp(s(2), Array(bVar(r, a, i))))
  }

  def initAndSeq(s: ArrayElementState, off: Code[Long]): Code[Unit] = {
    val streamLen = s.mb.newField[Int]
    val streamIdx = s.mb.newField[Int]

    val aidx = s.mb.newField[Int]
    val alen = s.mb.newField[Int]

    val a = s.mb.newField[Long]
    val r = s.region

    val lenVar = RVAVariable(EmitTriplet(Code._empty, false, alen), PInt32())
    val idxVar = RVAVariable(EmitTriplet(Code._empty, false, aidx), PInt32())

    val eltSeqOp = RVAVariable(EmitTriplet(seqOne(s.nested, a, aidx), false, Code._empty), PVoid)

    Code(
      lcAgg.initOp(s, Array()),
      streamIdx := 0,
      streamLen := streamType.loadLength(r, off),
      Code.whileLoop(streamIdx < streamLen,
        a := streamType.loadElement(r, off, streamIdx),
        alen := arrayType.loadLength(r, a),
        aidx := 0,
        lcAgg.seqOp(s, Array(lenVar)),
        Code.whileLoop(aidx < alen,
          eltAgg.seqOp(s, Array(idxVar, eltSeqOp)),
          aidx := aidx + 1),
        streamIdx := streamIdx + 1))
  }

  @Test def testInitSeqResult() {
    val firstCol = value.map(_(0))

    val fb = EmitFunctionBuilder[Region, Long, Long]
    val r = fb.apply_method.getArg[Region](1)
    val off = fb.apply_method.getArg[Long](2)

    val resType = PTuple(aggs.map(_.resultType))
    val states: Array[RVAState] = aggs.map(_.createState(fb.apply_method))
    val srvb = new StagedRegionValueBuilder(EmitRegion.default(fb.apply_method), resType)

    val aidx = fb.newField[Int]
    val alen = fb.newField[Int]

    fb.emit(
      Code(r.load().setNumParents(aggs.length),
        Code(Array.tabulate(aggs.length) { i =>
          Code(states(i).r := r.load().getParentReference(i),
            aggs(i).initOp(states(i), Array()))
        }: _*),
        aidx := 0,
        alen := arrayType.loadLength(r, off),
        Code.whileLoop(aidx < alen,
          seqOne(states, off, aidx),
          aidx := aidx + 1),
        srvb.start(),
        Code(aggs.zip(states).map{ case (agg, s) => Code(agg.result(s, srvb), srvb.advance()) }: _*),
        srvb.offset))

    val aggf = fb.resultWithIndex()

    Region.scoped { region =>
      val offset = ScalaToRegionValue(region, arrayType.virtualType, firstCol)
      val res = aggf(0)(region, offset)
      assert(SafeRow(resType, region, res) == expected(0))
    }
  }

  @Test def testInitSeqResultArray() {
    val fb = EmitFunctionBuilder[Region, Long, Long]
    val r = fb.apply_method.getArg[Region](1)
    val off = fb.apply_method.getArg[Long](2)

    val resType = PTuple(FastSeq(lcAgg.resultType))
    val s = lcAgg.createState(fb.apply_method)
    val srvb = new StagedRegionValueBuilder(EmitRegion.default(fb.apply_method), resType)

    fb.emit(
      Code(
        s.r := r,
        initAndSeq(s, off),
        srvb.start(),
        lcAgg.result(s, srvb),
        srvb.offset))

    val aggf = fb.resultWithIndex()

    Region.scoped { region =>
      val offset = ScalaToRegionValue(region, streamType.virtualType, value)
      val res = aggf(0)(region, offset)
      assert(SafeRow(resType, region, res) == Row(expected))
    }
  }

  @Test def serializeDeserializeAndCombOp() {
    val partitioned = value.grouped(3).toFastIndexedSeq

    val fb = EmitFunctionBuilder[Region, Long, Long]
    val r = fb.apply_method.getArg[Region](1)
    val off = fb.apply_method.getArg[Long](2)

    val resType = PTuple(FastSeq(lcAgg.resultType))
    val s = lcAgg.createState(fb.apply_method)
    val s2 = lcAgg.createState(fb.apply_method)
    val srvb = new StagedRegionValueBuilder(EmitRegion.default(fb.apply_method), resType)

    val partitionIdx = fb.newField[Int]
    val nPart = fb.newField[Int]
    val soff = fb.newField[Long]
    val spec = CodecSpec.defaultUncompressed

    val serialized = fb.newField[Array[Array[Byte]]]
    val baos = fb.newField[ByteArrayOutputStream]
    val bais = fb.newField[ByteArrayInputStream]
    val ob = fb.newField[OutputBuffer]
    val ib = fb.newField[InputBuffer]

    fb.emit(
      Code(
        nPart := PArray(streamType).loadLength(r, off),
        partitionIdx := 0,
        serialized := Code.newArray[Array[Byte]](nPart),
        Code.whileLoop(partitionIdx < nPart,
          baos := Code.newInstance[ByteArrayOutputStream](),
          ob := spec.buildCodeOutputBuffer(baos),
          s.r := Code.newInstance[Region](),
          soff := PArray(streamType).loadElement(s.r, off, partitionIdx),
          initAndSeq(s, soff),
          s.serialize(spec)(ob),
          ob.invoke[Unit]("flush"),
          serialized.load().update(partitionIdx, baos.invoke[Array[Byte]]("toByteArray")),
          partitionIdx := partitionIdx + 1),
        bais := Code.newInstance[ByteArrayInputStream, Array[Byte]](serialized.load()(0)),
        ib := spec.buildCodeInputBuffer(bais),
        s.r := Code.newInstance[Region](),
        s.unserialize(spec)(ib),
        partitionIdx := 1,
        Code.whileLoop(partitionIdx < nPart,
          bais := Code.newInstance[ByteArrayInputStream, Array[Byte]](serialized.load()(partitionIdx)),
          ib := spec.buildCodeInputBuffer(bais),
          s2.r := Code.newInstance[Region](),
          s2.unserialize(spec)(ib),
          lcAgg.combOp(s, s2),
          partitionIdx := partitionIdx + 1),
        srvb.start(),
        lcAgg.result(s, srvb),
        srvb.offset))

    val aggf = fb.resultWithIndex()

    Region.scoped { region =>
      val offset = ScalaToRegionValue(region, TArray(streamType.virtualType), partitioned)
      val res = aggf(0)(region, offset)
      assert(SafeRow(resType, region, res) == Row(expected))
    }
  }
}
