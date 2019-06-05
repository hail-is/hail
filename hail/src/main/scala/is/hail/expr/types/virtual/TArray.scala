package is.hail.expr.types.virtual

import is.hail.annotations.{Annotation, ExtendedOrdering}
import is.hail.check.Gen
import is.hail.expr.types.physical.PArray
import is.hail.utils._
import org.json4s.jackson.JsonMethods

import scala.reflect.{ClassTag, classTag}

final case class TArray(elementType: Type, override val required: Boolean = false) extends TContainer with TStreamable {
  lazy val physicalType: PArray = PArray(elementType.physicalType, required)

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
    case TArray(otherType, _) => elementType.canCompare(otherType)
    case _ => false
  }

  override def subst() = TArray(elementType.subst().setRequired(false))

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("Array[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[IndexedSeq[_]] &&
    a.asInstanceOf[IndexedSeq[_]].forall(elementType.typeCheck)

  override def str(a: Annotation): String = JsonMethods.compact(toJSON(a))

  override def genNonmissingValue: Gen[IndexedSeq[Annotation]] =
    Gen.buildableOf[Array](elementType.genValue).map(x => x: IndexedSeq[Annotation])

  val ordering: ExtendedOrdering =
    ExtendedOrdering.iterableOrdering(elementType.ordering)

  override def scalaClassTag: ClassTag[IndexedSeq[AnyRef]] = classTag[IndexedSeq[AnyRef]]
}
