package is.hail.annotations

import is.hail.types.virtual._
import is.hail.utils._
import org.apache.spark.sql.Row

object Annotation {
  val empty: Annotation = Row()

  def apply(args: Any*): Annotation = Row.fromSeq(args)

  def fromSeq(values: Seq[Any]): Annotation = Row.fromSeq(values)

  def copy(t: Type, a: Annotation): Annotation = {
    if (a == null)
      return null

    t match {
      case t: TBaseStruct =>
        val r = a.asInstanceOf[Row]
        Row.fromSeq(Array.tabulate(r.size)(i => Annotation.copy(t.types(i), r(i))))

      case t: TArray =>
        val arr = a.asInstanceOf[IndexedSeq[Annotation]]
        Array.tabulate(arr.length)(i => Annotation.copy(t.elementType, arr(i))).toFastSeq

      case t: TSet =>
        a.asInstanceOf[Set[Annotation]].map(Annotation.copy(t.elementType, _))

      case t: TDict =>
        a.asInstanceOf[Map[Annotation, Annotation]]
          .map { case (k, v) => (Annotation.copy(t.keyType, k), Annotation.copy(t.valueType, v)) }

      case t: TInterval =>
        val i = a.asInstanceOf[Interval]
        i.copy(start = Annotation.copy(t.pointType, i.start), end = Annotation.copy(t.pointType, i.end))

      case t: TNDArray =>
        val nd = a.asInstanceOf[NDArray]
        val rme = nd.getRowMajorElements()
        SafeNDArray(nd.shape, Array.tabulate(rme.length)(i => Annotation.copy(t.elementType, rme(i))).toFastSeq)

      case TInt32 | TInt64 | TFloat32 | TFloat64 | TBoolean | TString | TCall | _: TLocus | TBinary => a
    }
  }
}
