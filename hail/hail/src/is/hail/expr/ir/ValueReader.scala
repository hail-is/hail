package is.hail.expr.ir

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.asm4s.implicits._
import is.hail.io.{AbstractTypedCodecSpec, BufferSpec, StreamBufferSpec, TypedCodecSpec}
import is.hail.types.TypeWithRequiredness
import is.hail.types.encoded._
import is.hail.types.physical._
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.concrete.SNDArrayPointer
import is.hail.types.physical.stypes.primitives.SFloat64Value
import is.hail.types.virtual._

import java.io.InputStream

import org.json4s.{DefaultFormats, Extraction, Formats, JValue, ShortTypeHints}

object ValueReader {
  implicit val formats: Formats =
    new DefaultFormats() {
      override val typeHints = ShortTypeHints(
        List(
          classOf[ETypeValueReader],
          classOf[NumpyBinaryValueReader],
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
  override def unionRequiredness(requestedType: Type, requiredness: TypeWithRequiredness): Unit =
    requiredness.fromPType(spec.encodedType.decodedPType(requestedType))

  override def readValue(
    cb: EmitCodeBuilder,
    t: Type,
    region: Value[Region],
    is: Value[InputStream],
  ): SValue = {
    val decoder = spec.encodedType.buildDecoder(t, cb.emb.ecb)
    val ib = cb.memoize(spec.buildCodeInputBuffer(is))
    val ret = decoder.apply(cb, region, ib)
    ret
  }
}

final case class NumpyBinaryValueReader(nRows: Long, nCols: Long) extends ValueReader {
  final private val st = SNDArrayPointer(PCanonicalNDArray(PFloat64(true), 2, false))

  override def unionRequiredness(requestedType: Type, requiredness: TypeWithRequiredness): Unit =
    requiredness.fromPType(st.pType.setRequired(true))

  override def readValue(
    cb: EmitCodeBuilder,
    t: Type,
    region: Value[Region],
    is: Value[InputStream],
  ): SValue = {
    val pt = st.pType

    val stride0 = const(nCols * pt.elementType.byteSize)
    val stride1 = const(pt.elementType.byteSize)

    val in = cb.memoize(new StreamBufferSpec().buildCodeInputBuffer(is))
    val (tFirstElementAddress, tFinisher) =
      pt.constructDataFunction(IndexedSeq(nRows, nCols), IndexedSeq(stride0, stride1), cb, region)
    val currElementAddress =
      cb.newLocal[Long]("eblockmatrix_ndarray_currElementAddress", tFirstElementAddress)

    val i = cb.newLocal[Long]("i")
    cb.for_(
      cb.assign(i, 0L),
      i < nRows * nCols,
      cb.assign(i, i + 1L), {
        val elem = SFloat64Value(cb.memoize(in.readDouble()))
        pt.elementType.storeAtAddress(cb, currElementAddress, region, elem, false)
        cb.assign(currElementAddress, currElementAddress + pt.elementType.byteSize)
      },
    )

    tFinisher(cb)
  }
}
