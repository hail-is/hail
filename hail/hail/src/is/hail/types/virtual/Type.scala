package is.hail.types.virtual

import is.hail.annotations._
import is.hail.backend.HailStateManager
import is.hail.collection.FastSeq
import is.hail.expr.{JSONAnnotationImpex, SparkAnnotationImpex}
import is.hail.expr.ir._
import is.hail.utils

import org.apache.spark.sql.types.DataType
import org.json4s.{CustomSerializer, JValue}
import org.json4s.JsonAST.JString

class TypeSerializer extends CustomSerializer[Type](_ =>
      (
        { case JString(s) => IRParser.parseType(s) },
        { case t: Type => JString(t.parsableString()) },
      )
    )

abstract class Type extends VType with Serializable {

  def children: IndexedSeq[Type] = FastSeq()

  def clear(): Unit = children.foreach(_.clear())

  def unify(concrete: Type): Boolean =
    this == concrete

  def _isCanonical: Boolean = true

  final def isCanonical: Boolean = _isCanonical && children.forall(_.isCanonical)

  def isPrimitive: Boolean = this match {
    case TInt32 | TInt64 | TFloat32 | TFloat64 | TBoolean => true
    case _ => false
  }

  def isBound: Boolean = children.forall(_.isBound)

  def subst(): Type = this

  def query(fields: String*): Querier = query(fields.toList)

  def query(path: List[String]): Querier = {
    val (_, q) = queryTyped(path)
    q
  }

  def queryTyped(path: Seq[String]): (Type, Querier) =
    if (path.nonEmpty)
      throw new AnnotationPathException(s"invalid path ${path.mkString(".")} from type ${this}")
    else
      (this, identity[Annotation])

  final override def pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit =
    _pretty(sb, indent, compact)

  def _toPretty: String

  def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit =
    sb ++= _toPretty

  def schema: DataType = SparkAnnotationImpex.exportType(this)

  def str(a: Annotation): String = if (a == null) "NA" else a.toString

  def _showStr(a: Annotation): String = str(a)

  def showStr(a: Annotation): String = if (a == null) "NA" else _showStr(a)

  def showStr(a: Annotation, trunc: Int): String = {
    val s = showStr(a)
    if (s.length > trunc)
      s.substring(0, trunc - 3) + "..."
    else
      s
  }

  def `export`(a: Annotation): JValue =
    JSONAnnotationImpex.exportAnnotation(a, this)

  override def toJSON: JValue =
    JString(toString)

  def isRealizable: Boolean = children.forall(_.isRealizable)

  /* compare values for equality, but compare Float and Double values by the absolute value of their
   * difference is within tolerance or with D_== */
  def valuesSimilar(
    a1: Annotation,
    a2: Annotation,
    tolerance: Double = utils.defaultTolerance,
    absolute: Boolean = false,
  ): Boolean = a1 == a2

  def canCompare(other: Type): Boolean = this == other

  def mkOrdering(sm: HailStateManager, missingEqual: Boolean = true): ExtendedOrdering

  @transient protected var ord: ExtendedOrdering = _

  def ordering(sm: HailStateManager): ExtendedOrdering = {
    if (ord == null) ord = mkOrdering(sm)
    ord
  }

  def _typeCheck(a: Any): Boolean

  final def typeCheck(a: Any): Boolean = a == null || _typeCheck(a)

  def valueSubsetter(subtype: Type): Any => Any = {
    assert(this == subtype)
    identity
  }

  def isIsomorphicTo(t: Type): Boolean
}
