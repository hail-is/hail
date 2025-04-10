package is.hail.types.virtual

import is.hail.annotations._
import is.hail.backend.ExecuteContext
import is.hail.expr.{JSONAnnotationImpex, SparkAnnotationImpex}
import is.hail.expr.ir._
import is.hail.utils
import is.hail.utils._
import org.apache.spark.sql.types.DataType
import org.json4s.{CustomSerializer, Formats, JValue, Serializer}
import org.json4s.JsonAST.JString

import is.hail.utils.json4s._

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

  def queryTyped(fields: String*): (Type, Querier) = queryTyped(fields.toList)

  def queryTyped(path: List[String]): (Type, Querier) =
    if (path.nonEmpty)
      throw new AnnotationPathException(s"invalid path ${path.mkString(".")} from type ${this}")
    else
      (this, identity[Annotation])

  final def pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit =
    _pretty(sb, indent, compact)

  def _toPretty: String

  def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit =
    sb.append(_toPretty)

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

  def export(a: Annotation): JValue =
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

  def mkOrdering(missingEqual: Boolean = true): ExtendedOrdering

  lazy val ordering: ExtendedOrdering = mkOrdering()

  def _typeCheck(a: Any): Boolean

  final def typeCheck(a: Any): Boolean = a == null || _typeCheck(a)

  def valueSubsetter(subtype: Type): Any => Any = {
    assert(this == subtype)
    identity
  }

  def isIsomorphicTo(t: Type): Boolean
}

object Type {
  object Json4sFormat extends Json4sFormat[Type, JString] {
    override lazy val reader: Json4sReader[Type, JString] =
      (ctx: ExecuteContext, v: JString) =>
        (_: Formats) => IRParser.parseType(ctx, v.s)

    override lazy val writer: Json4sWriter[Type, JString] =
      (t: Type) => (_: Formats) => JString(t.parsableString())
  }
}
