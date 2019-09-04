package is.hail.expr.types.physical

import is.hail.annotations.CodeOrdering
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.{TTuple, TupleField}
import is.hail.utils._

case class PTupleField(index: Int, typ: PType)

object PTuple {
  def apply(required: Boolean, args: PType*): PTuple = PTuple(args.iterator.zipWithIndex.map { case (t, i) => PTupleField(i, t)}.toArray, required)

  def apply(args: PType*): PTuple = PTuple(args.zipWithIndex.map { case (a, i) => PTupleField(i, a)}.toFastIndexedSeq, required = false)
}

final case class PTuple(_types: IndexedSeq[PTupleField], override val required: Boolean = false) extends PBaseStruct {
  lazy val virtualType: TTuple = TTuple(_types.map(tf => TupleField(tf.index, tf.typ.virtualType)), required)

  val types = _types.map(_.typ).toArray
  val fieldRequired: Array[Boolean] = types.map(_.required)

  val fields: IndexedSeq[PField] = types.zipWithIndex.map { case (t, i) => PField(s"$i", t, i) }

  lazy val fieldIndex: Map[Int, Int] = _types.zipWithIndex.map { case (tf, idx) => tf.index -> idx }.toMap

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(other isOfType this)
    CodeOrdering.rowOrdering(this, other.asInstanceOf[PTuple], mb)
  }

  val size: Int = types.length

  override def truncate(newSize: Int): PTuple =
    PTuple(_types.take(newSize), required)

  val missingIdx = new Array[Int](size)
  val nMissing: Int = PBaseStruct.getMissingness(types, missingIdx)
  val nMissingBytes = (nMissing + 7) >>> 3
  val byteOffsets = new Array[Long](size)
  override val byteSize: Long = PBaseStruct.getByteSizeAndOffsets(types, nMissingBytes, byteOffsets)
  override val alignment: Long = PBaseStruct.alignment(types)

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
      PTuple(fundamentalFieldTypes, required)
  }
}
