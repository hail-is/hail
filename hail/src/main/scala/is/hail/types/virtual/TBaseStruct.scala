package is.hail.types.virtual

import is.hail.annotations._
import is.hail.backend.HailStateManager
import is.hail.check.Gen
import is.hail.utils._

import org.json4s.jackson.JsonMethods

import scala.reflect.{classTag, ClassTag}

import org.apache.spark.sql.Row

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

  def isIsomorphicTo(other: TBaseStruct): Boolean =
    size == other.size && isCompatibleWith(other)

  def isPrefixOf(other: TBaseStruct): Boolean =
    size <= other.size && isCompatibleWith(other)

  def isCompatibleWith(other: TBaseStruct): Boolean =
    fields.zip(other.fields).forall { case (l, r) => l.typ == r.typ }

  def truncate(newSize: Int): TBaseStruct

  override def _showStr(a: Annotation): String = {
    if (types.isEmpty)
      "()"
    else {
      Array.tabulate(size)(i => types(i).showStr(a.asInstanceOf[Row].get(i)))
        .mkString("(", ",", ")")
    }
  }

  override def str(a: Annotation): String = JsonMethods.compact(toJSON(a))

  override def genNonmissingValue(sm: HailStateManager): Gen[Annotation] = {
    if (types.isEmpty) {
      Gen.const(Annotation.empty)
    } else
      Gen.size.flatMap(fuel =>
        if (types.length > fuel)
          Gen.uniformSequence(types.map(t => Gen.const(null))).map(a => Annotation(a: _*))
        else
          Gen.uniformSequence(types.map(t => t.genValue(sm))).map(a => Annotation(a: _*))
      )
  }

  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double, absolute: Boolean)
    : Boolean =
    a1 == a2 || (a1 != null && a2 != null
      && types.zip(a1.asInstanceOf[Row].toSeq).zip(a2.asInstanceOf[Row].toSeq)
        .forall {
          case ((t, x1), x2) =>
            t.valuesSimilar(x1, x2, tolerance, absolute)
        })

  override def scalaClassTag: ClassTag[Row] = classTag[Row]
}
