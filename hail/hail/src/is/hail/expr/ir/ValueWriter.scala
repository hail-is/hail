package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.asm4s.implicits._
import is.hail.collection.FastSeq
import is.hail.io.{AbstractTypedCodecSpec, BufferSpec, StreamBufferSpec, TypedCodecSpec}
import is.hail.types.encoded._
import is.hail.types.physical._
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.primitives.SFloat64
import is.hail.types.virtual._
import is.hail.variant.ReferenceGenomeEnv

import java.io.OutputStream

import org.json4s.{DefaultFormats, Extraction, Formats, JValue, ShortTypeHints}

object ValueWriter {

  /** Formats that read types, parsing them against `rgs`. */
  def formats(rgs: ReferenceGenomeEnv): Formats = typeFormats(Some(rgs))

  /** Formats that write types but cannot read them. */
  def formats: Formats = writeOnlyFormats

  private[this] lazy val writeOnlyFormats: Formats = typeFormats(None)

  private def typeFormats(rgs: Option[ReferenceGenomeEnv]): Formats =
    new DefaultFormats() {
      override val typeHints = ShortTypeHints(
        List(
          classOf[ETypeValueWriter],
          classOf[NumpyBinaryValueWriter],
          classOf[AbstractTypedCodecSpec],
          classOf[TypedCodecSpec],
        ),
        typeHintFieldName = "name",
      ) + BufferSpec.shortTypeHints
    } +
      new TStructSerializer(rgs) +
      new TypeSerializer(rgs) +
      new PTypeSerializer(rgs) +
      new ETypeSerializer
}

abstract class ValueWriter {
  def writeValue(cb: EmitCodeBuilder, value: SValue, os: Value[OutputStream]): Unit

  def toJValue: JValue = Extraction.decompose(this)(ValueWriter.formats)
}

final case class ETypeValueWriter(spec: AbstractTypedCodecSpec) extends ValueWriter {
  override def writeValue(cb: EmitCodeBuilder, value: SValue, os: Value[OutputStream]): Unit = {
    val encoder = spec.encodedType.buildEncoder(value.st, cb.emb.ecb)
    val ob = cb.memoize(spec.buildCodeOutputBuffer(os))
    encoder.apply(cb, value, ob)
    cb += ob.invoke[Unit]("flush")
  }
}

final case class NumpyBinaryValueWriter(nRows: Long, nCols: Long) extends ValueWriter {
  override def writeValue(cb: EmitCodeBuilder, value: SValue, os: Value[OutputStream]): Unit = {
    val ob = cb.memoize(new StreamBufferSpec().buildCodeOutputBuffer(os))
    val ndarray = value.asNDArray
    assert(ndarray.st.elementType == SFloat64)
    val i = cb.newLocal[Long]("i")
    val j = cb.newLocal[Long]("j")

    cb.for_(
      cb.assign(i, 0L),
      i < nRows,
      cb.assign(i, i + 1L),
      cb.for_(
        cb.assign(j, 0L),
        j < nCols,
        cb.assign(j, j + 1L),
        cb += ob.writeDouble(ndarray.loadElement(FastSeq(i, j), cb).asFloat64.value),
      ),
    )

    cb += ob.invoke[Unit]("flush")
  }
}
