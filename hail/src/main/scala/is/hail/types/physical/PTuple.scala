package is.hail.types.physical

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{Code, Value, coerce}
import is.hail.expr.ir.{EmitMethodBuilder, SortOrder}
import is.hail.types.virtual.{TTuple, TupleField}
import is.hail.utils._

case class PTupleField(index: Int, typ: PType)

trait PTuple extends PBaseStruct {
  val _types: IndexedSeq[PTupleField]
  val fieldIndex: Map[Int, Int]

  lazy val virtualType: TTuple = TTuple(_types.map(tf => TupleField(tf.index, tf.typ.virtualType)))

  lazy val fields: IndexedSeq[PField] = _types.zipWithIndex.map { case (PTupleField(tidx, t), i) => PField(s"$tidx", t, i) }
  lazy val nFields: Int = fields.size

  protected val tupleFundamentalType: PTuple
  override lazy val fundamentalType: PTuple = tupleFundamentalType

  final def codeOrdering(mb: EmitMethodBuilder[_], other: PType, so: Array[SortOrder], missingFieldsEqual: Boolean): CodeOrdering = {
    assert(other isOfType this, s"$other != $this")
    assert(so == null || so.size == types.size)
    CodeOrdering.rowOrdering(this, other.asInstanceOf[PTuple], mb, so, missingFieldsEqual)
  }

  def identBase: String = "tuple"
}

class CodePTuple(
  val pType: PTuple,
  val offset: Value[Long]
) {
  def apply[T](i: Int): Value[T] =
      new Value[T] {
        def get: Code[T] = coerce[T](Region.loadIRIntermediate(pType.types(i))(pType.loadField(offset, i)))
      }

  def isMissing(i: Int): Code[Boolean] = {
    pType.isFieldMissing(offset, i)
  }

  def withTypesAndIndices: IndexedSeq[(PType, Value[_], Int)] = (0 until pType.nFields).map(i => (pType.types(i), apply(i), i)).toFastIndexedSeq

  def withTypes: IndexedSeq[(PType, Value[_])] = withTypesAndIndices.map(x => (x._1, x._2)).toFastIndexedSeq

  def missingnessPattern: IndexedSeq[Code[Boolean]] = (0 until pType.nFields).map(isMissing)

  def values[T, U, V]: (Value[T], Value[U], Value[V]) = {
    assert(pType.nFields == 3)
    (apply[T](0), apply[U](1), apply[V](2))
  }
}
