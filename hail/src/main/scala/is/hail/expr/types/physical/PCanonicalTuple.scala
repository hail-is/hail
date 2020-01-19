package is.hail.expr.types.physical
import is.hail.annotations.UnsafeUtils
import is.hail.expr.types.BaseStruct
import is.hail.utils._

object PCanonicalTuple {
  def apply(required: Boolean, args: PType*): PTuple = PCanonicalTuple(args.iterator.zipWithIndex.map { case (t, i) => PTupleField(i, t)}.toIndexedSeq, required)
}

final case class PCanonicalTuple(_types: IndexedSeq[PTupleField], override val required: Boolean = false) extends PTuple {
  val types = _types.map(_.typ).toArray
  lazy val fieldIndex: Map[Int, Int] = _types.zipWithIndex.map { case (tf, idx) => tf.index -> idx }.toMap

  val (missingIdx: Array[Int], nMissing: Int) = BaseStruct.getMissingIndexAndCount(types.map(_.required))
  val nMissingBytes = UnsafeUtils.packBitsToBytes(nMissing)
  val byteOffsets = new Array[Long](size)
  override val byteSize: Long = PBaseStruct.getByteSizeAndOffsets(types, nMissingBytes, byteOffsets)
  override val alignment: Long = PBaseStruct.alignment(types)

  def copy(required: Boolean = this.required): PTuple = PCanonicalTuple(_types, required)

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

  override def deepPTypeUnifyOnSameVirtualTypes(ptypes: Seq[PType]): PType = {
    val pTupleFields = this._types.map(pTupleField =>
      PTupleField(
        pTupleField.index,
        pTupleField.typ.deepPTypeUnifyOnSameVirtualTypes(ptypes.map(_.asInstanceOf[PTuple]._types(pTupleField.index).typ))
      )
    )

    PCanonicalTuple(pTupleFields, ptypes.forall(_.required))
  }
}
