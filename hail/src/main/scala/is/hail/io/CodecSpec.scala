package is.hail.io

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, InputStream, OutputStream}

import is.hail.annotations.{Region, RegionValue}
import is.hail.asm4s.{Code, TypeInfo, Value}
import is.hail.expr.ir.{EmitClassBuilder, EmitFunctionBuilder, typeToTypeInfo}
import is.hail.expr.types.encoded.EType
import is.hail.expr.types.physical.PType
import is.hail.expr.types.virtual.Type
import is.hail.rvd.RVDContext
import is.hail.sparkextras.ContextRDD
import is.hail.utils.using
import org.apache.spark.rdd.RDD

trait AbstractTypedCodecSpec extends Spec {
  def encodedType: EType
  def encodedVirtualType: Type

  type StagedEncoderF[T] = (Value[Region], Value[T], Value[OutputBuffer]) => Code[Unit]
  type StagedDecoderF[T] = (Value[Region], Value[InputBuffer]) => Code[T]

  def buildEncoder(t: PType): (OutputStream) => Encoder

  def buildDecoder(requestedType: Type): (PType, (InputStream) => Decoder)

  def encode(t: PType, region: Region, offset: Long): Array[Byte] = {
    val baos = new ByteArrayOutputStream()
    using(buildEncoder(t)(baos))(_.writeRegionValue(region, offset))
    baos.toByteArray
  }

  def decode(requestedType: Type, bytes: Array[Byte], region: Region): (PType, Long) = {
    val bais = new ByteArrayInputStream(bytes)
    val (pt, dec) = buildDecoder(requestedType)
    (pt, dec(bais).readRegionValue(region))
  }

  def buildCodeInputBuffer(is: Code[InputStream]): Code[InputBuffer]

  def buildCodeOutputBuffer(os: Code[OutputStream]): Code[OutputBuffer]

  def buildEmitDecoderF[T](requestedType: Type, cb: EmitClassBuilder[_]): (PType, StagedDecoderF[T])

  def buildEmitEncoderF[T](t: PType, cb: EmitClassBuilder[_]): StagedEncoderF[T]

  def buildEmitDecoderF[T](requestedType: Type, cb: EmitClassBuilder[_], ti: TypeInfo[T]): (PType, StagedDecoderF[T]) = {
    val (ptype, dec) = buildEmitDecoderF[T](requestedType, cb)
    assert(ti == typeToTypeInfo(requestedType))
    ptype -> dec
  }

  def buildEmitEncoderF[T](t: PType, cb: EmitClassBuilder[_], ti: TypeInfo[T]): StagedEncoderF[T] = {
    assert(ti == typeToTypeInfo(t))
    buildEmitEncoderF[T](t, cb)
  }

  // FIXME: is there a better place for this to live?
  def decodeRDD(requestedType: Type, bytes: RDD[Array[Byte]]): (PType, ContextRDD[RegionValue]) = {
    val (pt, dec) = buildDecoder(requestedType)
    (pt, ContextRDD.weaken(bytes).cmapPartitions { (ctx, it) =>
      RegionValue.fromBytes(dec, ctx.region, it)
    })
  }

  override def toString: String = super[Spec].toString
}
