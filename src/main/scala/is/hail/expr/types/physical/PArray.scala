package is.hail.expr.types.physical

import is.hail.annotations.{UnsafeUtils, _}
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.TArray
import org.json4s.jackson.JsonMethods

import scala.reflect.{ClassTag, _}

final case class PArray(elementType: PType, override val required: Boolean = false) extends PIterable {
  def virtualType: TArray = TArray(elementType.virtualType, required)

  val elementByteSize: Long = UnsafeUtils.arrayElementSize(elementType)

  val contentsAlignment: Long = elementType.alignment.max(4)

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("array<")
    elementType.pyString(sb)
    sb.append('>')
  }
  override val fundamentalType: PArray = {
    if (elementType == elementType.fundamentalType)
      this
    else
      this.copy(elementType = elementType.fundamentalType)
  }

  def _toPretty = s"Array[$elementType]"

  override def canCompare(other: PType): Boolean = other match {
    case PArray(otherType, _) => elementType.canCompare(otherType)
    case _ => false
  }

  override def unify(concrete: PType): Boolean = {
    concrete match {
      case PArray(celementType, _) => elementType.unify(celementType)
      case _ => false
    }
  }

  override def subst() = PArray(elementType.subst().setRequired(false))

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("Array[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[IndexedSeq[_]] &&
    a.asInstanceOf[IndexedSeq[_]].forall(elementType.typeCheck)

  override def genNonmissingValue: Gen[Annotation] =
    Gen.buildableOf[Array](elementType.genValue).map(x => x: IndexedSeq[Annotation])

  val ordering: ExtendedOrdering =
    ExtendedOrdering.iterableOrdering(elementType.ordering)

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(this isOfType other)
    CodeOrdering.iterableOrdering(virtualType, other.asInstanceOf[PArray].virtualType, mb)
  }

  override def scalaClassTag: ClassTag[IndexedSeq[AnyRef]] = classTag[IndexedSeq[AnyRef]]
}
