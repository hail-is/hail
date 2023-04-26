package is.hail.expr.ir

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.io.{AbstractTypedCodecSpec, BufferSpec, TypedCodecSpec}
import is.hail.types.encoded._
import is.hail.types.physical._
import is.hail.types.physical.stypes.{SCode, SType, SValue}
import is.hail.types.physical.stypes.concrete.SStackStruct
import is.hail.types.virtual._
import is.hail.utils._

import org.json4s.{DefaultFormats, Extraction, Formats, JValue, ShortTypeHints}

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, InputStream, OutputStream}

object ValueWriter {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(List(
      classOf[ETypeFileValueWriter],
      classOf[AbstractTypedCodecSpec],
      classOf[TypedCodecSpec]),
      typeHintFieldName = "name"
    ) + BufferSpec.shortTypeHints
  }  +
    new TStructSerializer +
    new TypeSerializer +
    new PTypeSerializer +
    new ETypeSerializer
}

abstract class ValueWriter {
  final def writeValue(cb: EmitCodeBuilder, value: SValue, _args: EmitCode*): SValue = {
    val args = _args.toIndexedSeq
    val argsTypes = args.map(_.st.virtualType)
    assert(argsTypes == argumentTypes, s"argument mismatch, required argument types $argumentTypes, got $argsTypes")
    _writeValue(cb, value, args)
  }
  protected def _writeValue(cb: EmitCodeBuilder, value: SValue, args: IndexedSeq[EmitCode]): SValue
  def argumentTypes: IndexedSeq[Type]
  // one of void, binary, or string, checked in TypeCheck
  def returnType: Type

  def toJValue: JValue = Extraction.decompose(this)(ValueWriter.formats)
}

abstract class ETypeValueWriter extends ValueWriter {
  def spec: AbstractTypedCodecSpec

  final def serialize(cb: EmitCodeBuilder, value: SValue, os: Value[OutputStream]) = {
    val encoder = spec.encodedType.buildEncoder(value.st, cb.emb.ecb)
    val ob = cb.memoize(spec.buildCodeOutputBuffer(os))
    encoder.apply(cb, value, ob)
    cb += ob.invoke[Unit]("close")
  }
}

final case class ETypeFileValueWriter(spec: AbstractTypedCodecSpec) extends ETypeValueWriter {
  protected def _writeValue(cb: EmitCodeBuilder, value: SValue, args: IndexedSeq[EmitCode]): SValue = {
    val IndexedSeq(path_) = args
    val path = path_.toI(cb).get(cb).asString
    val os = cb.memoize(cb.emb.createUnbuffered(path.loadString(cb)))
    serialize(cb, value, os) // takes ownership and closes the stream
    path
  }

  val argumentTypes: IndexedSeq[Type] = FastIndexedSeq(/*path=*/TString)
  val returnType: Type = TString
}
