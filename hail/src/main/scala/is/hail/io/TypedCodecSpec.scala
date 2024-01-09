package is.hail.io

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{EmitClassBuilder, EmitFunctionBuilder}
import is.hail.types.encoded._
import is.hail.types.physical._
import is.hail.types.virtual._

import java.io._

object TypedCodecSpec {
  def apply(pt: PType, bufferSpec: BufferSpec): TypedCodecSpec = {
    val eType = EType.defaultFromPType(pt)
    TypedCodecSpec(eType, pt.virtualType, bufferSpec)
  }
}

final case class TypedCodecSpec(_eType: EType, _vType: Type, _bufferSpec: BufferSpec)
    extends AbstractTypedCodecSpec {
  def encodedType: EType = _eType
  def encodedVirtualType: Type = _vType

  def buildEncoder(ctx: ExecuteContext, t: PType): (OutputStream, HailClassLoader) => Encoder = {
    val bufferToEncoder = encodedType.buildEncoder(ctx, t)
    (out: OutputStream, theHailClassLoader: HailClassLoader) =>
      bufferToEncoder(_bufferSpec.buildOutputBuffer(out), theHailClassLoader)
  }

  def decodedPType(requestedType: Type): PType =
    encodedType.decodedPType(requestedType)

  def buildDecoder(ctx: ExecuteContext, requestedType: Type)
    : (PType, (InputStream, HailClassLoader) => Decoder) = {
    val (rt, bufferToDecoder) = encodedType.buildDecoder(ctx, requestedType)
    (
      rt,
      (in: InputStream, theHailClassLoader: HailClassLoader) =>
        bufferToDecoder(_bufferSpec.buildInputBuffer(in), theHailClassLoader),
    )
  }

  def buildStructDecoder(ctx: ExecuteContext, requestedType: TStruct)
    : (PStruct, (InputStream, HailClassLoader) => Decoder) = {
    val (pType: PStruct, makeDec) = buildDecoder(ctx, requestedType)
    pType -> makeDec
  }

  def buildCodeInputBuffer(is: Code[InputStream]): Code[InputBuffer] =
    _bufferSpec.buildCodeInputBuffer(is)

  def buildCodeOutputBuffer(os: Code[OutputStream]): Code[OutputBuffer] =
    _bufferSpec.buildCodeOutputBuffer(os)
}
