package is.hail.expr.types.physical

import is.hail.annotations.{CodeOrdering, UnsafeUtils}
import is.hail.expr.ir.{EmitMethodBuilder, SortOrder}
import is.hail.expr.types.BaseStruct
import is.hail.utils._

final case class PCanonicalTuple(_types: IndexedSeq[PTupleField], override val required: Boolean = false) extends PTuple with PCanonicalBaseStruct {
  val types = _types.map(_.typ).toArray

  val fields: IndexedSeq[PField] = types.zipWithIndex.map { case (t, i) => PField(s"$i", t, i) }
  val nFields: Int = fields.size

  lazy val fieldIndex: Map[Int, Int] = _types.zipWithIndex.map { case (tf, idx) => tf.index -> idx }.toMap

  def copy(required: Boolean = this.required): PTuple = PCanonicalTuple(_types, required)

  override def truncate(newSize: Int): PTuple =
    PCanonicalTuple(_types.take(newSize), required)

  val missingIdx = new Array[Int](size)
  val nMissing: Int = BaseStruct.getMissingness[PType](types, missingIdx)
  val nMissingBytes = UnsafeUtils.packBitsToBytes(nMissing)
  val byteOffsets = new Array[Long](size)
  override val byteSize: Long = PCanonicalBaseStruct.getByteSizeAndOffsets(types, nMissingBytes, byteOffsets)
  override val alignment: Long = PCanonicalBaseStruct.alignment(types)

  def identBase: String = "tuple"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean) {
    sb.append("Tuple[")
    types.foreachBetween(_.pretty(sb, indent, compact))(sb += ',')
    sb += ']'
  }

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("tuple(")
    fields.foreachBetween({ field =>
      field.typ.pyString(sb)
    }) { sb.append(", ")}
    sb.append(')')
  }


  override val fundamentalType: PTuple = {
    val fundamentalFieldTypes = _types.map(tf => tf.copy(typ = tf.typ.fundamentalType))
    if ((_types, fundamentalFieldTypes).zipped
      .forall { case (t, ft) => t == ft })
      this
    else
      PCanonicalTuple(fundamentalFieldTypes, required)
  }
}