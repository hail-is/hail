package is.hail.types.physical

import is.hail.types.virtual.{TTuple, TupleField}

case class PTupleField(index: Int, typ: PType)

trait PTuple extends PBaseStruct {
  val _types: IndexedSeq[PTupleField]
  val fieldIndex: Map[Int, Int]

  lazy val virtualType: TTuple = TTuple(_types.map(tf => TupleField(tf.index, tf.typ.virtualType)))

  lazy val fields: IndexedSeq[PField] = _types.zipWithIndex.map { case (PTupleField(tidx, t), i) =>
    PField(s"$tidx", t, i)
  }

  lazy val nFields: Int = fields.size

  def identBase: String = "tuple"
}
