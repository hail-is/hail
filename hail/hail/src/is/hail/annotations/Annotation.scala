package is.hail.annotations

import is.hail.collection.compat.immutable.ArraySeq
import is.hail.types.virtual._
import is.hail.utils._

import org.apache.spark.sql.Row

object Annotation {
  val empty: Annotation = RowSeq()

  def apply(args: Any*): Annotation =
    RowSeq(args: _*)

  def fromSeq(values: Seq[Any]): Annotation =
    RowSeq.fromSeq(values)

  def copy(t: Type, a: Annotation): Annotation = {
    if (a == null)
      return null

    t match {
      case t: TBaseStruct =>
        val r = a.asInstanceOf[Row]
        RowSeq.fromSeq(ArraySeq.tabulate(r.size)(i => Annotation.copy(t.types(i), r(i))))

      case t: TArray =>
        val arr = a.asInstanceOf[IndexedSeq[Annotation]]
        ArraySeq.tabulate(arr.length)(i => Annotation.copy(t.elementType, arr(i)))

      case t: TSet =>
        a.asInstanceOf[Set[Annotation]].map(Annotation.copy(t.elementType, _))

      case t: TDict =>
        a.asInstanceOf[Map[Annotation, Annotation]]
          .map { case (k, v) => (Annotation.copy(t.keyType, k), Annotation.copy(t.valueType, v)) }

      case t: TInterval =>
        val i = a.asInstanceOf[Interval]
        i.copy(
          start = Annotation.copy(t.pointType, i.start),
          end = Annotation.copy(t.pointType, i.end),
        )

      case t: TNDArray =>
        val nd = a.asInstanceOf[NDArray]
        val rme = nd.getRowMajorElements()
        SafeNDArray(
          nd.shape,
          ArraySeq.tabulate(rme.length)(i => Annotation.copy(t.elementType, rme(i))),
        )

      case TInt32 | TInt64 | TFloat32 | TFloat64 | TBoolean | TString | TCall | _: TLocus | TBinary =>
        a
    }
  }
}

class RowSeq(values: ArraySeq[Any]) extends Row {
  override def length: Int = values.length
  override def get(i: Int): Any = values(i)
  override def copy(): Row = this
  override def toSeq: Seq[Any] = values
}

object RowSeq {
  def apply(values: Any*): Row =
    if (values.isEmpty) Row.empty else new RowSeq(ArraySeq(values: _*))

  def fromSeq(values: Seq[Any]): Row =
    if (values.isEmpty) Row.empty else new RowSeq(ArraySeq.from(values))

  def fromTuple(p: Product): Row =
    if (p.productArity == 0) Row.empty else new RowSeq(ArraySeq.from(p.productIterator))
}
