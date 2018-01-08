package is.hail.expr.typ

import is.hail.annotations.{UnsafeUtils, _}
import is.hail.check.Gen
import is.hail.utils._
import org.json4s.jackson.JsonMethods

import scala.reflect.{ClassTag, _}

final case class TArray(elementType: Type, override val required: Boolean = false) extends TIterable {
  val elementByteSize: Long = UnsafeUtils.arrayElementSize(elementType)

  val contentsAlignment: Long = elementType.alignment.max(4)

  override val fundamentalType: TArray = {
    if (elementType == elementType.fundamentalType)
      this
    else
      this.copy(elementType = elementType.fundamentalType)
  }

  def _toString = s"Array[$elementType]"

  override def canCompare(other: Type): Boolean = other match {
    case TArray(otherType, _) => elementType.canCompare(otherType)
    case _ => false
  }

  override def unify(concrete: Type): Boolean = {
    concrete match {
      case TArray(celementType, _) => elementType.unify(celementType)
      case _ => false
    }
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

  override def genNonmissingValue: Gen[Annotation] =
    Gen.buildableOf[Array](elementType.genValue).map(x => x: IndexedSeq[Annotation])

  def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    annotationOrdering(extendOrderingToNull(missingGreatest)(
      Ordering.Iterable(elementType.ordering(missingGreatest))))

  override def desc: String =
    """
    An ``Array`` is a collection of items that all have the same data type (ex: Int, String) and are indexed. Arrays can be constructed by specifying ``[item1, item2, ...]`` and they are 0-indexed.

    An example of constructing an array and accessing an element is:

    .. code-block:: text
        :emphasize-lines: 2

        let a = [1, 10, 3, 7] in a[1]
        result: 10

    They can also be nested such as Array[Array[Int]]:

    .. code-block:: text
        :emphasize-lines: 2

        let a = [[1, 2, 3], [4, 5], [], [6, 7]] in a[1]
        result: [4, 5]
    """

  override def scalaClassTag: ClassTag[IndexedSeq[AnyRef]] = classTag[IndexedSeq[AnyRef]]
}
