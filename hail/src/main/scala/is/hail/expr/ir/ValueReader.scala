package is.hail.expr.ir

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.io.{AbstractTypedCodecSpec, BufferSpec, TypedCodecSpec}
import is.hail.types.TypeWithRequiredness
import is.hail.types.encoded._
import is.hail.types.physical._
import is.hail.types.physical.stypes.{SCode, SType, SValue}
import is.hail.types.physical.stypes.concrete.SStackStruct
import is.hail.types.virtual._
import is.hail.utils._

import org.json4s.{DefaultFormats, Extraction, Formats, JValue, ShortTypeHints}

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, InputStream, OutputStream}

object ValueReader {
  implicit val formats: Formats =
    new DefaultFormats() {
      override val typeHints = ShortTypeHints(
        List(
          classOf[ETypeValueReader],
          classOf[AbstractTypedCodecSpec],
          classOf[TypedCodecSpec],
        ),
        typeHintFieldName = "name",
      ) + BufferSpec.shortTypeHints
    } +
      new TStructSerializer +
      new TypeSerializer +
      new PTypeSerializer +
      new ETypeSerializer
}

abstract class ValueReader {
  def unionRequiredness(requestedType: Type, requiredness: TypeWithRequiredness): Unit

  def readValue(cb: EmitCodeBuilder, t: Type, region: Value[Region], is: Value[InputStream]): SValue

  def toJValue: JValue = Extraction.decompose(this)(ValueReader.formats)
}

final case class ETypeValueReader(spec: AbstractTypedCodecSpec) extends ValueReader {
  def unionRequiredness(requestedType: Type, requiredness: TypeWithRequiredness): Unit =
    requiredness.fromPType(spec.encodedType.decodedPType(requestedType))

  def readValue(cb: EmitCodeBuilder, t: Type, region: Value[Region], is: Value[InputStream])
    : SValue = {
    val decoder = spec.encodedType.buildDecoder(t, cb.emb.ecb)
    val ib = cb.memoize(spec.buildCodeInputBuffer(is))
    val ret = decoder.apply(cb, region, ib)
    ret
  }
}
