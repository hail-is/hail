package is.hail.types.virtual

import is.hail.annotations.{Annotation, ExtendedOrdering}
import is.hail.backend.HailStateManager

import org.json4s.jackson.JsonMethods

final case class TSet(elementType: Type) extends TContainer {
  def _toPretty = s"Set[$elementType]"

  override def pyString(sb: StringBuilder): Unit = {
    sb ++= "set<"
    elementType.pyString(sb)
    sb += '>'
  }

  override def canCompare(other: Type): Boolean = other match {
    case TSet(otherType) => elementType.canCompare(otherType)
    case _ => false
  }

  override def unify(concrete: Type): Boolean = concrete match {
    case TSet(celementType) => elementType.unify(celementType)
    case _ => false
  }

  override def subst() = TSet(elementType.subst())

  def _typeCheck(a: Any): Boolean =
    a.isInstanceOf[Set[_]] && a.asInstanceOf[Set[_]].forall(elementType.typeCheck)

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false): Unit = {
    sb ++= "Set["
    elementType.pretty(sb, indent, compact)
    sb += ']'
  }

  override def mkOrdering(sm: HailStateManager, missingEqual: Boolean): ExtendedOrdering =
    ExtendedOrdering.setOrdering(elementType.ordering(sm), missingEqual)

  override def _showStr(a: Annotation): String =
    a.asInstanceOf[Set[Annotation]]
      .map { case elt => elementType.showStr(elt) }
      .mkString("{", ",", "}")

  override def str(a: Annotation): String = JsonMethods.compact(export(a))

  override def valueSubsetter(subtype: Type): Any => Any = {
    assert(elementType == subtype.asInstanceOf[TSet].elementType)
    identity
  }

  override def arrayElementsRepr: TArray = TArray(elementType)

  override def isIsomorphicTo(t: Type): Boolean =
    t match {
      case s: TSet => elementType isIsomorphicTo s.elementType
      case _ => false
    }
}
