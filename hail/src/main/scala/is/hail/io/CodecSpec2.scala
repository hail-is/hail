package is.hail.io

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, InputStream, OutputStream}

import is.hail.annotations.{Region, RegionValue}
import is.hail.asm4s.Code
import is.hail.{HailContext, cxx}
import is.hail.expr.ir.EmitFunctionBuilder
import is.hail.expr.types.physical.PType
import is.hail.expr.types.virtual.Type
import is.hail.rvd.{AbstractRVDSpec, RVDContext}
import is.hail.sparkextras.ContextRDD
import is.hail.utils.using
import org.apache.spark.rdd.RDD
import org.json4s.Extraction
import org.json4s.jackson.JsonMethods

trait CodecSpec2 extends Spec {
  def encodedType: Type

  type StagedEncoderF[T] = (Code[Region], Code[T], Code[OutputBuffer]) => Code[Unit]
  type StagedDecoderF[T] = (Code[Region], Code[InputBuffer]) => Code[T]

  def buildEncoder(t: PType): (OutputStream) => Encoder = buildEncoder(t, t)
  def buildEncoder(t: PType, requestedType: PType): (OutputStream) => Encoder

  def buildDecoder(requestedType: Type): (PType, (InputStream) => Decoder)

  def encode(t: PType, region: Region, offset: Long): Array[Byte] = {
    val baos = new ByteArrayOutputStream()
    using(buildEncoder(t, t)(baos))(_.writeRegionValue(region, offset))
    baos.toByteArray
  }

  def decode(requestedType: Type, bytes: Array[Byte], region: Region): (PType, Long) = {
    val bais = new ByteArrayInputStream(bytes)
    val (pt, dec) = buildDecoder(requestedType)
    (pt, dec(bais).readRegionValue(region))
  }

  def buildCodeInputBuffer(is: Code[InputStream]): Code[InputBuffer]

  def buildCodeOutputBuffer(os: Code[OutputStream]): Code[OutputBuffer]

  def buildEmitDecoderF[T](requestedType: Type, fb: EmitFunctionBuilder[_]): (PType, StagedDecoderF[T])

  def buildEmitEncoderF[T](t: PType, fb: EmitFunctionBuilder[_]): StagedEncoderF[T]

  def buildNativeDecoderClass(
    requestedType: Type,
    inputStreamType: String,
    tub: cxx.TranslationUnitBuilder
  ): (PType, cxx.Class)

  def buildNativeEncoderClass(t: PType, tub: cxx.TranslationUnitBuilder): cxx.Class

  // FIXME: is there a better place for this to live?
  def decodeRDD(requestedType: Type, bytes: RDD[Array[Byte]]): (PType, ContextRDD[RVDContext, RegionValue]) = {
    val (pt, dec) = buildDecoder(requestedType)
    (pt, ContextRDD.weaken[RVDContext](bytes).cmapPartitions { (ctx, it) =>
      val rv = RegionValue(ctx.region)
      it.map(RegionValue.fromBytes(dec, ctx.region, rv))
    })
  }

  override def toString: String = super[Spec].toString
}

