package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.io.{AbstractTypedCodecSpec, BufferSpec, TypedCodecSpec}
import is.hail.types.encoded._
import is.hail.types.physical.stypes.SValue
import org.json4s.{DefaultFormats, Extraction, Formats, JValue, ShortTypeHints}

import java.io.OutputStream

object ValueWriter {
  lazy val formats: Formats =
    DefaultFormats.withHints {
      ShortTypeHints(
        List(
          classOf[ETypeValueWriter],
          classOf[AbstractTypedCodecSpec],
          classOf[TypedCodecSpec],
        ),
        typeHintFieldName = "name",
      ) + BufferSpec.shortTypeHints
    } +
      ETypeSerializer
}

abstract class ValueWriter {
  def writeValue(cb: EmitCodeBuilder, value: SValue, os: Value[OutputStream]): Unit

  def toJValue: JValue = Extraction.decompose(this)(ValueWriter.formats)
}

final case class ETypeValueWriter(spec: AbstractTypedCodecSpec) extends ValueWriter {
  def writeValue(cb: EmitCodeBuilder, value: SValue, os: Value[OutputStream]): Unit = {
    val encoder = spec.encodedType.buildEncoder(value.st, cb.emb.ecb)
    val ob = cb.memoize(spec.buildCodeOutputBuffer(os))
    encoder.apply(cb, value, ob)
    cb += ob.invoke[Unit]("flush")
  }
}
