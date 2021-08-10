package is.hail.types.virtual

import is.hail.annotations.{Annotation, ExtendedOrdering}
import is.hail.check.Gen
import org.json4s.jackson.JsonMethods

import scala.reflect.{ClassTag, classTag}

final case class TArray(elementType: Type) extends TContainer {
  override def pyString(sb: StringBuilder): Unit = {
    sb.append("array<")
    elementType.pyString(sb)
    sb.append('>')
  }
  override val fundamentalType: TArray = {
    if (elementType == elementType.fundamentalType)
      this
    else
      this.copy(elementType = elementType.fundamentalType)
  }

  def _toPretty = s"Array[$elementType]"

  override def canCompare(other: Type): Boolean = other match {
    case TArray(otherType) => elementType.canCompare(otherType)
    case _ => false
  }

  override def unify(concrete: Type): Boolean = concrete match {
    case TArray(celementType) => elementType.unify(celementType)
    case _ => false
  }

  override def subst() = TArray(elementType.subst())

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("Array[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[IndexedSeq[_]] &&
    a.asInstanceOf[IndexedSeq[_]].forall(elementType.typeCheck)

  override def _showStr(a: Annotation): String =
    a.asInstanceOf[IndexedSeq[Annotation]]
      .map(elt => elementType.showStr(elt))
      .mkString("[", ",", "]")

  override def str(a: Annotation): String = JsonMethods.compact(toJSON(a))

  override def genNonmissingValue: Gen[IndexedSeq[Annotation]] =
    Gen.buildableOf[Array](elementType.genValue).map(x => x: IndexedSeq[Annotation])

  override val ordering: ExtendedOrdering = mkOrdering()

  def mkOrdering(missingEqual: Boolean): ExtendedOrdering =
    ExtendedOrdering.iterableOrdering(elementType.ordering, missingEqual)

  override def scalaClassTag: ClassTag[IndexedSeq[AnyRef]] = classTag[IndexedSeq[AnyRef]]

  override def valueSubsetter(subtype: Type): Any => Any = {
    if (this == subtype)
      return identity

    val subsetElem = elementType.valueSubsetter(subtype.asInstanceOf[TArray].elementType)
    (a: Any) => a.asInstanceOf[IndexedSeq[Any]].map(subsetElem)
  }

  override def arrayElementsRepr: TArray = this
}
