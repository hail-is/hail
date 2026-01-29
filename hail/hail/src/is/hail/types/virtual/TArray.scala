package is.hail.types.virtual

import is.hail.annotations.{Annotation, ExtendedOrdering}
import is.hail.backend.HailStateManager

import org.json4s.jackson.JsonMethods

final case class TArray(elementType: Type) extends TContainer {
  override def pyString(sb: StringBuilder): Unit = {
    sb ++= "array<"
    elementType.pyString(sb)
    sb += '>'
  }

  override def _toPretty = s"Array[$elementType]"

  override def canCompare(other: Type): Boolean = other match {
    case TArray(otherType) => elementType.canCompare(otherType)
    case _ => false
  }

  override def unify(concrete: Type): Boolean = concrete match {
    case TArray(celementType) => elementType.unify(celementType)
    case _ => false
  }

  override def subst() = TArray(elementType.subst())

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false): Unit = {
    sb ++= "Array["
    elementType.pretty(sb, indent, compact)
    sb += ']'
  }

  override def _typeCheck(a: Any): Boolean = a.isInstanceOf[IndexedSeq[_]] &&
    a.asInstanceOf[IndexedSeq[_]].forall(elementType.typeCheck)

  override def _showStr(a: Annotation): String =
    a.asInstanceOf[IndexedSeq[Annotation]]
      .map(elt => elementType.showStr(elt))
      .mkString("[", ",", "]")

  override def str(a: Annotation): String = JsonMethods.compact(export(a))

  override def mkOrdering(sm: HailStateManager, missingEqual: Boolean): ExtendedOrdering =
    ExtendedOrdering.iterableOrdering(elementType.ordering(sm), missingEqual)

  override def valueSubsetter(subtype: Type): Any => Any = {
    if (this == subtype)
      return identity

    val subsetElem = elementType.valueSubsetter(subtype.asInstanceOf[TArray].elementType)
    (a: Any) => a.asInstanceOf[IndexedSeq[Any]].map(subsetElem)
  }

  override def arrayElementsRepr: TArray = this

  override def isIsomorphicTo(t: Type): Boolean =
    t match {
      case a: TArray => elementType isIsomorphicTo a.elementType
      case _ => false
    }
}
