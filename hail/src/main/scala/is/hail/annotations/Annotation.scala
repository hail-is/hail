package is.hail.annotations

import is.hail.expr.types.virtual._
import is.hail.utils._
import org.apache.spark.sql.Row

import scala.collection.immutable.{TreeMap, TreeSet}

object Annotation {

  final val COL_HEAD = "sa"

  final val ROW_HEAD = "va"

  final val GLOBAL_HEAD = "global"

  final val ENTRY_HEAD = "g"

  val empty: Annotation = Row()

  def emptyIndexedSeq(n: Int): IndexedSeq[Annotation] = Array.fill[Annotation](n)(Annotation.empty)

  def printAnnotation(a: Any, nSpace: Int = 0): String = {
    val spaces = " " * nSpace
    a match {
      case null => "Null"
      case r: Row =>
        "Struct:\n" +
          r.toSeq.zipWithIndex.map { case (elem, index) =>
            s"""$spaces[$index] ${ printAnnotation(elem, nSpace + 4) }"""
          }
            .mkString("\n")
      case a => a.toString + ": " + a.getClass.getSimpleName
    }
  }

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
        Array.tabulate(arr.length)(i => Annotation.copy(t.elementType, arr(i))).toFastIndexedSeq

      case t: TSet =>
        val s = a.asInstanceOf[TreeSet[Any]]
        TreeSet(s.iterator.map(a => Annotation.copy(t.elementType, a)).toFastIndexedSeq: _*)(s.ordering)

      case t: TDict =>
        val m = a.asInstanceOf[TreeMap[Any, Any]]
        TreeMap(
          m.iterator.map { case (k, v) => (Annotation.copy(t.keyType, k), Annotation.copy(t.valueType, v)) }.toFastIndexedSeq: _*)(m.ordering)

      case t: TInterval =>
        val i = a.asInstanceOf[Interval]
        i.copy(start = Annotation.copy(t.pointType, i.start), end = Annotation.copy(t.pointType, i.end))

      case _ => a
    }
  }

  def isSafe(typ: Type, a: Annotation): Boolean = {
    a == null || (typ match {
      case t: TBaseStruct =>
        val r = a.asInstanceOf[Row]
        !r.isInstanceOf[UnsafeRow] && Array.range(0, t.size).forall(i => Annotation.isSafe(t.types(i), r(i)))

      case t: TArray =>
        !a.isInstanceOf[UnsafeIndexedSeq] && a.asInstanceOf[IndexedSeq[Annotation]].forall(Annotation.isSafe(t.elementType, _))

      case t: TSet =>
        a.asInstanceOf[Set[Annotation]].forall(Annotation.isSafe(t.elementType, _))

      case t: TDict =>
        a.asInstanceOf[Map[Annotation, Annotation]]
          .forall { case (k, v) => Annotation.isSafe(t.keyType, k) && Annotation.isSafe(t.valueType, v) }

      case t: TInterval =>
        val i = a.asInstanceOf[Interval]
        Annotation.isSafe(t.pointType, i.start) && Annotation.isSafe(t.pointType, i.end)

      case _ => true
    })
  }
}
