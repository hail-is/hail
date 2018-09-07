package is.hail.expr.types.physical

import is.hail.annotations.{CodeOrdering, ExtendedOrdering}
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.TTuple
import is.hail.utils._


final case class PTuple(_types: IndexedSeq[PType], override val required: Boolean = false) extends PBaseStruct {
  lazy val virtualType: TTuple = TTuple(types.map(_.virtualType), required)

  val types = _types.toArray
  val fieldRequired: Array[Boolean] = types.map(_.required)

  val fields: IndexedSeq[PField] = types.zipWithIndex.map { case (t, i) => PField(s"$i", t, i) }

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(other isOfType this)
    CodeOrdering.rowOrdering(this, other.asInstanceOf[PTuple], mb)
  }

  val size: Int = types.length

  override def truncate(newSize: Int): PTuple =
    PTuple(types.take(newSize), required)

  val missingIdx = new Array[Int](size)
  val nMissing: Int = PBaseStruct.getMissingness(types, missingIdx)
  val nMissingBytes = (nMissing + 7) >>> 3
  val byteOffsets = new Array[Long](size)
  override val byteSize: Long = PBaseStruct.getByteSizeAndOffsets(types, nMissingBytes, byteOffsets)
  override val alignment: Long = PBaseStruct.alignment(types)

  def ++(that: PTuple): PTuple = PTuple(types ++ that.types, required = false)

  override def canCompare(other: PType): Boolean = other match {
    case t: PTuple => size == t.size && types.zip(t.types).forall { case (t1, t2) => t1.canCompare(t2) }
    case _ => false
  }

  override def unify(concrete: PType): Boolean = concrete match {
    case PTuple(ctypes, _) =>
      size == ctypes.length &&
        (types, ctypes).zipped.forall { case (t, ct) =>
          t.unify(ct)
        }
    case _ => false
  }

  override def subst() = PTuple(types.map(t => t.subst()))

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
    val fundamentalFieldTypes = types.map(t => t.fundamentalType)
    if ((types, fundamentalFieldTypes).zipped
      .forall { case (t, ft) => t == ft })
      this
    else {
      val t = PTuple(fundamentalFieldTypes)
      t.setRequired(required).asInstanceOf[PTuple]
    }
  }
}
