package is.hail.io

import java.io._

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitFunctionBuilder}
import is.hail.expr.types.encoded._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._

object TypedCodecSpec {
  def apply(pt: PType, bufferSpec: BufferSpec): TypedCodecSpec = {
    val eType = EType.defaultFromPType(pt)
    TypedCodecSpec(eType, pt.virtualType, bufferSpec)
  }
}

final case class TypedCodecSpec(_eType: EType, _vType: Type, _bufferSpec: BufferSpec) extends AbstractTypedCodecSpec {
  val encodedType: EType = _eType
  val encodedVirtualType: Type = _vType

  def computeSubsetPType(requestedType: Type): PType = {
    _eType._decodedPType(requestedType)
  }

  def buildEncoder(t: PType): (OutputStream) => Encoder = {
    val f = EType.buildEncoder(encodedType, t)
    out: OutputStream => new CompiledEncoder(_bufferSpec.buildOutputBuffer(out), f)
  }

  def buildDecoder(requestedType: Type): (PType, (InputStream) => Decoder) = {
    val (rt, f) = EType.buildDecoder(encodedType, requestedType)
    (rt, (in: InputStream) => new CompiledDecoder(_bufferSpec.buildInputBuffer(in), f))
  }

  def buildCodeInputBuffer(is: Code[InputStream]): Code[InputBuffer] = _bufferSpec.buildCodeInputBuffer(is)

  def buildCodeOutputBuffer(os: Code[OutputStream]): Code[OutputBuffer] = _bufferSpec.buildCodeOutputBuffer(os)

  def buildEmitDecoderF[T](requestedType: Type, cb: EmitClassBuilder[_]): (PType, StagedDecoderF[T]) = {
    val rt = encodedType.decodedPType(requestedType)
    val mb = encodedType.buildDecoderMethod(rt, cb)
    (rt, (region: Value[Region], buf: Value[InputBuffer]) => mb.invoke[T](region, buf))
  }

  def buildEmitEncoderF[T](t: PType, cb: EmitClassBuilder[_]): StagedEncoderF[T] = {
    val mb = encodedType.buildEncoderMethod(t, cb)
    (region: Value[Region], off: Value[T], buf: Value[OutputBuffer]) => mb.invoke[Unit](off, buf)
  }
}
