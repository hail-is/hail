package is.hail.types.virtual

import is.hail.annotations.{Annotation, ExtendedOrdering}
import is.hail.backend.HailStateManager

import org.json4s.jackson.JsonMethods

final case class TStream(elementType: Type) extends TIterable {
  override def pyString(sb: StringBuilder): Unit = {
    sb ++= "stream<"
    elementType.pyString(sb)
    sb += '>'
  }

  def _toPretty = s"Stream[$elementType]"

  override def canCompare(other: Type): Boolean =
    throw new UnsupportedOperationException("Stream comparison is currently undefined.")

  override def unify(concrete: Type): Boolean = concrete match {
    case TStream(celementType) => elementType.unify(celementType)
    case _ => false
  }

  override def subst() = TStream(elementType.subst())

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false): Unit = {
    sb ++= "Stream["
    elementType.pretty(sb, indent, compact)
    sb += ']'
  }

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[IndexedSeq[_]] &&
    a.asInstanceOf[IndexedSeq[_]].forall(elementType.typeCheck)

  override def str(a: Annotation): String = JsonMethods.compact(export(a))

  override def isRealizable = false

  override def mkOrdering(sm: HailStateManager, missingEqual: Boolean): ExtendedOrdering =
    throw new UnsupportedOperationException("Stream comparison is currently undefined.")

  override def isIsomorphicTo(t: Type): Boolean =
    t match {
      case s: TStream => elementType isIsomorphicTo s.elementType
      case _ => false
    }
}
