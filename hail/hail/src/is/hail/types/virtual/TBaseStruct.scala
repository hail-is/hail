package is.hail.types.virtual

import is.hail.annotations._
import is.hail.backend.HailStateManager
import is.hail.utils._

import org.apache.spark.sql.Row
import org.json4s.jackson.JsonMethods

object TBaseStruct {

  /** Define an ordering on Row objects. Works with any row r such that the list of types of r is a
    * prefix of types, or types is a prefix of the list of types of r.
    */
  def getOrdering(sm: HailStateManager, types: Array[Type], missingEqual: Boolean = true)
    : ExtendedOrdering =
    ExtendedOrdering.rowOrdering(types.map(_.ordering(sm)), missingEqual)

  def getJoinOrdering(sm: HailStateManager, types: Array[Type], missingEqual: Boolean = false)
    : ExtendedOrdering =
    ExtendedOrdering.rowOrdering(
      types.map(_.mkOrdering(sm, missingEqual = missingEqual)),
      _missingEqual = missingEqual,
    )
}

abstract class TBaseStruct extends Type {
  def types: Array[Type]

  def fields: IndexedSeq[Field]

  lazy val fieldIdx: collection.Map[String, Int] = toMapFast(fields)(_.name, _.index)

  override def children: IndexedSeq[Type] = types

  def size: Int

  def _toPretty: String = {
    val sb = new StringBuilder
    _pretty(sb, 0, compact = true)
    sb.result()
  }

  override def _typeCheck(a: Any): Boolean = a match {
    case row: Row =>
      row.length == types.length &&
      isComparableAt(a)
    case _ => false
  }

  def relaxedTypeCheck(a: Any): Boolean = a match {
    case row: Row =>
      row.length <= types.length &&
      isComparableAt(a)
    case _ => false
  }

  def isComparableAt(a: Annotation): Boolean = a match {
    case row: Row =>
      row.toSeq.zip(types).forall {
        case (v, t) => t.typeCheck(v)
      }
    case _ => false
  }

  def isJoinableWith(other: TBaseStruct): Boolean =
    size == other.size && isCompatibleWith(other)

  def isPrefixOf(other: TBaseStruct): Boolean =
    size <= other.size && isCompatibleWith(other)

  override def isIsomorphicTo(t: Type): Boolean =
    t match {
      case s: TBaseStruct => size == s.size && forallZippedFields(s)(_.typ isIsomorphicTo _.typ)
      case _ => false
    }

  def isCompatibleWith(other: TBaseStruct): Boolean =
    forallZippedFields(other)(_.typ == _.typ)

  private def forallZippedFields(s: TBaseStruct)(p: (Field, Field) => Boolean): Boolean =
    (fields, s.fields).zipped.forall(p)

  def truncate(newSize: Int): TBaseStruct

  override def _showStr(a: Annotation): String = {
    if (types.isEmpty)
      "()"
    else {
      Array.tabulate(size)(i => types(i).showStr(a.asInstanceOf[Row].get(i)))
        .mkString("(", ",", ")")
    }
  }

  override def str(a: Annotation): String = JsonMethods.compact(export(a))

  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double, absolute: Boolean)
    : Boolean =
    a1 == a2 || (a1 != null && a2 != null
      && types.zip(a1.asInstanceOf[Row].toSeq).zip(a2.asInstanceOf[Row].toSeq)
        .forall {
          case ((t, x1), x2) =>
            t.valuesSimilar(x1, x2, tolerance, absolute)
        })

}
