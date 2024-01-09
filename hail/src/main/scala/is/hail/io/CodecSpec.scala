package is.hail.io

import is.hail.annotations.{Region, RegionValue}
import is.hail.asm4s.{theHailClassLoaderForSparkWorkers, Code, HailClassLoader}
import is.hail.backend.ExecuteContext
import is.hail.sparkextras.ContextRDD
import is.hail.types.encoded.EType
import is.hail.types.physical.PType
import is.hail.types.virtual.Type
import is.hail.utils.{using, ArrayOfByteArrayOutputStream}
import is.hail.utils.prettyPrint.ArrayOfByteArrayInputStream

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, InputStream, OutputStream}

import org.apache.spark.rdd.RDD

trait AbstractTypedCodecSpec extends Spec {
  def encodedType: EType
  def encodedVirtualType: Type

  def buildEncoder(ctx: ExecuteContext, t: PType): (OutputStream, HailClassLoader) => Encoder

  def encodeValue(ctx: ExecuteContext, t: PType, valueAddr: Long): Array[Byte] = {
    val makeEnc = buildEncoder(ctx, t)
    val baos = new ByteArrayOutputStream()
    val enc = makeEnc(baos, ctx.theHailClassLoader)
    enc.writeRegionValue(valueAddr)
    enc.flush()
    baos.toByteArray
  }

  def decodedPType(requestedType: Type): PType

  def decodedPType(): PType = encodedType.decodedPType(encodedVirtualType)

  def buildDecoder(ctx: ExecuteContext, requestedType: Type)
    : (PType, (InputStream, HailClassLoader) => Decoder)

  def encode(ctx: ExecuteContext, t: PType, offset: Long): Array[Byte] = {
    val baos = new ByteArrayOutputStream()
    encode(ctx, t, offset, baos)
    baos.toByteArray
  }

  def encode(ctx: ExecuteContext, t: PType, offset: Long, os: OutputStream): Unit =
    using(buildEncoder(ctx, t)(os, ctx.theHailClassLoader))(_.writeRegionValue(offset))

  def encodeArrays(ctx: ExecuteContext, t: PType, offset: Long): Array[Array[Byte]] = {
    val baos = new ArrayOfByteArrayOutputStream()
    encode(ctx, t, offset, baos)
    baos.toByteArrays()
  }

  def decode(ctx: ExecuteContext, requestedType: Type, bytes: Array[Byte], region: Region)
    : (PType, Long) = {
    val bais = new ByteArrayInputStream(bytes)
    val (pt, dec) = buildDecoder(ctx, requestedType)
    (pt, dec(bais, ctx.theHailClassLoader).readRegionValue(region))
  }

  def decodeArrays(
    ctx: ExecuteContext,
    requestedType: Type,
    bytes: Array[Array[Byte]],
    region: Region,
  ): (PType, Long) = {
    val bais = new ArrayOfByteArrayInputStream(bytes)
    val (pt, dec) = buildDecoder(ctx, requestedType)
    (pt, dec(bais, ctx.theHailClassLoader).readRegionValue(region))
  }

  def buildCodeInputBuffer(is: Code[InputStream]): Code[InputBuffer]

  def buildCodeOutputBuffer(os: Code[OutputStream]): Code[OutputBuffer]

  // FIXME: is there a better place for this to live?
  def decodeRDD(ctx: ExecuteContext, requestedType: Type, bytes: RDD[Array[Byte]])
    : (PType, ContextRDD[Long]) = {
    val (pt, dec) = buildDecoder(ctx, requestedType)
    (
      pt,
      ContextRDD.weaken(bytes).cmapPartitions { (ctx, it) =>
        RegionValue.fromBytes(theHailClassLoaderForSparkWorkers, dec, ctx.region, it)
      },
    )
  }

  override def toString: String = super[Spec].toString
}
