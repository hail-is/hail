package is.hail.io

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, InputStream, OutputStream}

import is.hail.annotations.{Region, RegionValue}
import is.hail.asm4s.Code
import is.hail.expr.ir.ExecuteContext
import is.hail.types.encoded.EType
import is.hail.types.physical.PType
import is.hail.types.virtual.Type
import is.hail.sparkextras.ContextRDD
import is.hail.utils.using
import org.apache.spark.rdd.RDD

trait AbstractTypedCodecSpec extends Spec {
  def encodedType: EType
  def encodedVirtualType: Type

  def buildEncoder(ctx: ExecuteContext, t: PType): (OutputStream) => Encoder

  def encodeValue(ctx: ExecuteContext, t: PType, valueAddr: Long): Array[Byte] = {
    val makeEnc = buildEncoder(ctx, t)
    val baos = new ByteArrayOutputStream()
    val enc = makeEnc(baos)
    enc.writeRegionValue(valueAddr)
    enc.flush()
    baos.toByteArray
  }

  def decodedPType(requestedType: Type): PType

  def decodedPType(): PType = encodedType.decodedPType(encodedVirtualType)

  def buildDecoder(ctx: ExecuteContext, requestedType: Type): (PType, (InputStream) => Decoder)

  def encode(ctx: ExecuteContext, t: PType, offset: Long): Array[Byte] = {
    val baos = new ByteArrayOutputStream()
    using(buildEncoder(ctx, t)(baos))(_.writeRegionValue(offset))
    baos.toByteArray
  }

  def decode(ctx: ExecuteContext, requestedType: Type, bytes: Array[Byte], region: Region): (PType, Long) = {
    val bais = new ByteArrayInputStream(bytes)
    val (pt, dec) = buildDecoder(ctx, requestedType)
    (pt, dec(bais).readRegionValue(region))
  }

  def buildCodeInputBuffer(is: Code[InputStream]): Code[InputBuffer]

  def buildCodeOutputBuffer(os: Code[OutputStream]): Code[OutputBuffer]

  // FIXME: is there a better place for this to live?
  def decodeRDD(ctx: ExecuteContext, requestedType: Type, bytes: RDD[Array[Byte]]): (PType, ContextRDD[Long]) = {
    val (pt, dec) = buildDecoder(ctx, requestedType)
    (pt, ContextRDD.weaken(bytes).cmapPartitions { (ctx, it) =>
      RegionValue.fromBytes(dec, ctx.region, it)
    })
  }

  override def toString: String = super[Spec].toString
}
