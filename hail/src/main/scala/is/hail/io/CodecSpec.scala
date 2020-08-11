package is.hail.io

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, InputStream, OutputStream}

import is.hail.annotations.{Region, RegionValue}
import is.hail.asm4s.{Code, TypeInfo, Value}
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder, EmitFunctionBuilder, ExecuteContext}
import is.hail.types.encoded.EType
import is.hail.types.physical.{PCode, PType, PValue, typeToTypeInfo}
import is.hail.types.virtual.Type
import is.hail.rvd.RVDContext
import is.hail.sparkextras.ContextRDD
import is.hail.utils.using
import org.apache.spark.rdd.RDD

trait AbstractTypedCodecSpec extends Spec {
  def encodedType: EType
  def encodedVirtualType: Type

  type StagedEncoderF[T] = (Value[Region], Value[T], Value[OutputBuffer]) => Code[Unit]
  type StagedDecoderF[T] = (Value[Region], Value[InputBuffer]) => Code[T]

  def buildEncoder(ctx: ExecuteContext, t: PType): (OutputStream) => Encoder

  def decodedPType(requestedType: Type): PType

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

  def buildTypedEmitDecoderF[T](requestedType: Type, cb: EmitClassBuilder[_]): (PType, StagedDecoderF[T])

  def buildTypedEmitEncoderF[T](t: PType, cb: EmitClassBuilder[_]): StagedEncoderF[T]

  def buildEmitDecoder(requestedType: Type, cb: EmitClassBuilder[_]): (Value[Region], Value[InputBuffer]) => PCode = {
    def typedBuilder[T](ti: TypeInfo[T]): (Value[Region], Value[InputBuffer]) => PCode = {
      val (ptype, dec) = buildTypedEmitDecoderF[T](requestedType, cb);
      { (r: Value[Region], ib: Value[InputBuffer]) => PCode(ptype, dec(r, ib)) }
    }
    typedBuilder(typeToTypeInfo(decodedPType(requestedType)))
  }

  def buildEmitEncoder(t: PType, cb: EmitClassBuilder[_]): (Value[Region], PValue, Value[OutputBuffer]) => Code[Unit] = {
    def typedBuilder[T](ti: TypeInfo[T]): (Value[Region], PValue, Value[OutputBuffer]) => Code[Unit] = {
      val enc = buildTypedEmitEncoderF[T](t, cb);
      { (r: Value[Region], v: PValue, ob: Value[OutputBuffer]) =>
        enc(r, v.value.asInstanceOf[Value[T]], ob)
      }
    }
    typedBuilder(typeToTypeInfo(t))
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
