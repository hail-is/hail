package is.hail.io

import java.io._

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitFunctionBuilder, ExecuteContext}
import is.hail.types.encoded._
import is.hail.types.physical._
import is.hail.types.virtual._

object TypedCodecSpec {
  def apply(pt: PType, bufferSpec: BufferSpec): TypedCodecSpec = {
    val eType = EType.defaultFromPType(pt)
    TypedCodecSpec(eType, pt.virtualType, bufferSpec)
  }
}

final case class TypedCodecSpec(_eType: EType, _vType: Type, _bufferSpec: BufferSpec) extends AbstractTypedCodecSpec {
  def encodedType: EType = _eType
  def encodedVirtualType: Type = _vType

  def computeSubsetPType(requestedType: Type): PType = {
    _eType._decodedPType(requestedType)
  }

  def buildEncoder(ctx: ExecuteContext, t: PType): (OutputStream) => Encoder = {
    val bufferToEncoder = encodedType.buildEncoder(ctx, t)
    out: OutputStream => bufferToEncoder(_bufferSpec.buildOutputBuffer(out))
  }

  def decodedPType(requestedType: Type): PType = {
    encodedType.decodedPType(requestedType)
  }

  def decodedPType(): PType = {
    encodedType.decodedPType(_vType)
  }

  def buildDecoder(ctx: ExecuteContext, requestedType: Type): (PType, (InputStream) => Decoder) = {
    val (rt, bufferToDecoder) = encodedType.buildDecoder(ctx, requestedType)
    (rt, (in: InputStream) => bufferToDecoder(_bufferSpec.buildInputBuffer(in)))
  }

  def buildStructDecoder(ctx: ExecuteContext, requestedType: TStruct): (PStruct, (InputStream) => Decoder) = {
    val (pType: PStruct, makeDec) = buildDecoder(ctx, requestedType)
    pType -> makeDec
  }

  def buildCodeInputBuffer(is: Code[InputStream]): Code[InputBuffer] = _bufferSpec.buildCodeInputBuffer(is)

  def buildCodeOutputBuffer(os: Code[OutputStream]): Code[OutputBuffer] = _bufferSpec.buildCodeOutputBuffer(os)

  def buildTypedEmitDecoderF[T](requestedType: Type, cb: EmitClassBuilder[_]): (PType, StagedDecoderF[T]) = {
    val rt = encodedType.decodedPType(requestedType)
    val mb = encodedType.buildDecoderMethod(rt, cb)
    (rt, (region: Value[Region], buf: Value[InputBuffer]) => mb.invokeCode[T](region, buf))
  }

  def buildEmitDecoderF[T](cb: EmitClassBuilder[_]): (PType, StagedDecoderF[T]) =
    buildTypedEmitDecoderF(_vType, cb)

  def buildTypedEmitEncoderF[T](t: PType, cb: EmitClassBuilder[_]): StagedEncoderF[T] = {
    val mb = encodedType.buildEncoderMethod(t, cb)
    (region: Value[Region], off: Value[T], buf: Value[OutputBuffer]) => mb.invokeCode[Unit](off, buf)
  }
}
