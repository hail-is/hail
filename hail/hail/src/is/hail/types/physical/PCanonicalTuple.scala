package is.hail.types.physical

import is.hail.types.virtual.{TTuple, Type}
import is.hail.utils._

object PCanonicalTuple {
  def apply(required: Boolean, args: PType*): PCanonicalTuple = PCanonicalTuple(
    args.iterator.zipWithIndex.map { case (t, i) => PTupleField(i, t) }.toIndexedSeq,
    required,
  )
}

final case class PCanonicalTuple(
  _types: IndexedSeq[PTupleField],
  override val required: Boolean = false,
) extends PCanonicalBaseStruct(_types.map(_.typ).toArray) with PTuple {
  lazy val fieldIndex: Map[Int, Int] = _types.zipWithIndex.map { case (tf, idx) =>
    tf.index -> idx
  }.toMap

  def setRequired(required: Boolean) =
    if (required == this.required) this else PCanonicalTuple(_types, required)

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = {
    sb.append("PCTuple[")
    _types.foreachBetween { fd =>
      sb.append(fd.index)
      sb.append(':')
      fd.typ.pretty(sb, indent, compact)
    }(sb += ',')
    sb += ']'
  }

  override def deepRename(t: Type) = deepTupleRename(t.asInstanceOf[TTuple])

  private def deepTupleRename(t: TTuple) =
    PCanonicalTuple(
      (t._types, this._types).zipped.map { (tfield, pfield) =>
        assert(tfield.index == pfield.index)
        PTupleField(pfield.index, pfield.typ.deepRename(tfield.typ))
      },
      this.required,
    )

  def copiedType: PType = {
    val copiedTypes = types.map(_.copiedType)
    if (types.indices.forall(i => types(i).eq(copiedTypes(i))))
      this
    else {
      PCanonicalTuple(copiedTypes.indices.map(i => _types(i).copy(typ = copiedTypes(i))), required)
    }
  }
}
