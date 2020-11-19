package is.hail.io

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, InputStream, OutputStream}

import is.hail.annotations.{Region, RegionValue}
import is.hail.asm4s.{Code, LineNumber, TypeInfo, Value}
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder, EmitFunctionBuilder, ExecuteContext}
import is.hail.types.encoded.EType
import is.hail.types.physical.{PCode, PType, PValue, typeToTypeInfo}
import is.hail.types.virtual.Type
import is.hail.sparkextras.ContextRDD
import is.hail.utils.using
import org.apache.spark.rdd.RDD

abstract class StagedEncoder[T] {
  def apply(r: Value[Region], v: Value[T], ob: Value[OutputBuffer])(implicit line: LineNumber): Code[Unit]
}

abstract class StagedDecoder[T] {
  def apply(r: Value[Region], ib: Value[InputBuffer])(implicit line: LineNumber): Code[T]
}

abstract class EmitEncoder {
  def apply(r: Value[Region], v: PValue, ob: Value[OutputBuffer])(implicit line: LineNumber): Code[Unit]
}

abstract class EmitDecoder {
  def apply(r: Value[Region], ib: Value[InputBuffer])(implicit line: LineNumber): PCode
}

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

  def buildEmitDecoder(requestedType: Type, cb: EmitClassBuilder[_])(implicit line: LineNumber): EmitDecoder = {
    typeToTypeInfo(decodedPType(requestedType)) match {
      case ti: TypeInfo[t] =>
        val (ptype, dec) = buildTypedEmitDecoderF[t](requestedType, cb)
        new EmitDecoder {
          def apply(r: Value[Region], ib: Value[InputBuffer])(implicit line: LineNumber): PCode =
            PCode(ptype, dec(r, ib))
        }
    }
  }

  def buildEmitEncoder(pt: PType, cb: EmitClassBuilder[_])(implicit line: LineNumber): EmitEncoder = {
    typeToTypeInfo(pt) match {
      case ti: TypeInfo[t] =>
        val enc: StagedEncoder[t] = buildTypedEmitEncoderF[t](pt, cb)
        new EmitEncoder {
          def apply(r: Value[Region], v: PValue, ob: Value[OutputBuffer])(implicit line: LineNumber): Code[Unit] =
            enc(r, v.value.asInstanceOf[Value[t]], ob)
        }
    }
  }

  def buildTypedEmitDecoderF[T](requestedType: Type, cb: EmitClassBuilder[_])(implicit line: LineNumber): (PType, StagedDecoder[T]) = {
    val rt = encodedType.decodedPType(requestedType)
    val mb = encodedType.buildDecoderMethod(rt, cb)
    val dec = new StagedDecoder[T] {
      def apply(region: Value[Region], buf: Value[InputBuffer])(implicit line: LineNumber): Code[T] =
        mb.invokeCode[T](region, buf)
    }
    (rt, dec)
  }

  def buildEmitDecoderF[T](cb: EmitClassBuilder[_])(implicit line: LineNumber): (PType, StagedDecoder[T]) =
    buildTypedEmitDecoderF(encodedVirtualType, cb)

  def buildTypedEmitEncoderF[T](t: PType, cb: EmitClassBuilder[_])(implicit line: LineNumber): StagedEncoder[T] = {
    val mb = encodedType.buildEncoderMethod(t, cb)
    new StagedEncoder[T] {
      def apply(region: Value[Region], off: Value[T], buf: Value[OutputBuffer])(implicit line: LineNumber): Code[Unit] =
        mb.invokeCode[Unit](off, buf)
    }
  }

  // FIXME: is there a better place for this to live?
  def decodeRDD(ctx: ExecuteContext, requestedType: Type, bytes: RDD[Array[Byte]]): (PType, ContextRDD[Long]) = {
    val (pt, dec) = buildDecoder(ctx, requestedType)
    (pt, ContextRDD.weaken(bytes).cmapPartitions { (ctx, it) =>
      RegionValue.fromBytes(dec, ctx.region, it)
    })
  }

  override def toString: String = super[Spec].toString
}
