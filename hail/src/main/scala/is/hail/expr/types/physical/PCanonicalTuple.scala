package is.hail.expr.types.physical
import is.hail.annotations.UnsafeUtils
import is.hail.expr.types.BaseStruct
import is.hail.expr.types.virtual.{TTuple, Type}
import is.hail.utils._

object PCanonicalTuple {
  def apply(required: Boolean, args: PType*): PCanonicalTuple = PCanonicalTuple(args.iterator.zipWithIndex.map { case (t, i) => PTupleField(i, t)}.toIndexedSeq, required)
}

final case class PCanonicalTuple(_types: IndexedSeq[PTupleField], override val required: Boolean = false) extends PCanonicalBaseStruct(_types.map(_.typ).toArray) with PTuple {
  lazy val fieldIndex: Map[Int, Int] = _types.zipWithIndex.map { case (tf, idx) => tf.index -> idx }.toMap

  def setRequired(required: Boolean) = if(required == this.required) this else PCanonicalTuple(_types, required)

  override def truncate(newSize: Int): PTuple =
    PCanonicalTuple(_types.take(newSize), required)

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean) {
    sb.append("PCTuple[")
    _types.foreachBetween { fd =>
      sb.append(fd.index)
      sb.append(':')
      fd.typ.pretty(sb, indent, compact)
    }(sb += ',')
    sb += ']'
  }

  lazy val tupleFundamentalType: PTuple = {
    val fundamentalFieldTypes = _types.map(tf => tf.copy(typ = tf.typ.fundamentalType))
    if ((_types, fundamentalFieldTypes).zipped
      .forall { case (t, ft) => t == ft })
      this
    else
      PCanonicalTuple(fundamentalFieldTypes, required)
  }

  override def deepRename(t: Type) = deepTupleRename(t.asInstanceOf[TTuple])

  private def deepTupleRename(t: TTuple) = {
    PCanonicalTuple((t._types, this._types).zipped.map( (tfield, pfield) => {
      assert(tfield.index == pfield.index)
      PTupleField(pfield.index, pfield.typ.deepRename(tfield.typ))
    }), this.required)
  }
}
