package is.hail.io

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, InputStream, OutputStream}

import is.hail.annotations.{Region, RegionValue}
import is.hail.asm4s.Code
import is.hail.expr.ir.EmitFunctionBuilder
import is.hail.expr.types.physical.PType
import is.hail.expr.types.virtual.Type
import is.hail.rvd.RVDContext
import is.hail.sparkextras.ContextRDD
import is.hail.utils.using
import org.apache.spark.rdd.RDD

trait CodecSpec extends Spec {
  def makeCodecSpec2(pType: PType): CodecSpec2
}

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

object CodecSpec {
  val defaultBufferSpec: BufferSpec = LEB128BufferSpec(
    BlockingBufferSpec(32 * 1024,
      LZ4BlockBufferSpec(32 * 1024,
        new StreamBlockBufferSpec)))

  val default: CodecSpec = PackCodecSpec(defaultBufferSpec)

  val defaultUncompressedBuffer: BufferSpec = BlockingBufferSpec(32 * 1024,
    new StreamBlockBufferSpec)

  val defaultUncompressed: CodecSpec = PackCodecSpec(defaultUncompressedBuffer)

  val unblockedUncompressed: CodecSpec = PackCodecSpec(new StreamBufferSpec)

  def fromShortString(s: String): CodecSpec = s match {
    case "default" => CodecSpec.default
    case "defaultUncompressed" => CodecSpec.defaultUncompressed
    case "unblockedUncompressed" => CodecSpec.unblockedUncompressed
  }

  val baseBufferSpecs: Array[BufferSpec] = Array(
    BlockingBufferSpec(64 * 1024,
      new StreamBlockBufferSpec),
    BlockingBufferSpec(32 * 1024,
      LZ4BlockBufferSpec(32 * 1024,
        new StreamBlockBufferSpec)),
    new StreamBufferSpec)

  val bufferSpecs: Array[BufferSpec] = baseBufferSpecs.flatMap { blockSpec =>
    Array(blockSpec, LEB128BufferSpec(blockSpec))
  }

  val codecSpecs: Array[CodecSpec] = bufferSpecs.flatMap { bufferSpec =>
    Array(PackCodecSpec(bufferSpec))
  }

  val supportedCodecSpecs: Array[CodecSpec] = bufferSpecs.flatMap { bufferSpec =>
    Array(PackCodecSpec(bufferSpec))
  }
}
