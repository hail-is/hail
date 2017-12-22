package is.hail.expr.typ

import is.hail.annotations.{UnsafeUtils, _}
import is.hail.check.Gen
import is.hail.utils._
import org.json4s.jackson.JsonMethods

import scala.reflect.{ClassTag, _}

/**
  * Created by dking on 12/21/17.
  */
final case class TSet(elementType: Type, override val required: Boolean = false) extends TIterable {
  val elementByteSize: Long = UnsafeUtils.arrayElementSize(elementType)

  val contentsAlignment: Long = elementType.alignment.max(4)

  override val fundamentalType: TArray = TArray(elementType.fundamentalType, required)

  def _toString = s"Set[$elementType]"

  override def canCompare(other: Type): Boolean = other match {
    case TSet(otherType, _) => elementType.canCompare(otherType)
    case _ => false
  }

  override def unify(concrete: Type): Boolean = concrete match {
    case TSet(celementType, _) => elementType.unify(celementType)
    case _ => false
  }

  override def subst() = TSet(elementType.subst())

  def _typeCheck(a: Any): Boolean =
    a.isInstanceOf[Set[_]] && a.asInstanceOf[Set[_]].forall(elementType.typeCheck)

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("Set[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }

  def ordering(missingGreatest: Boolean): Ordering[Annotation] = {
    val elementSortOrd = elementType.ordering(true)
    val itOrd = Ordering.Iterable(elementType.ordering(missingGreatest))
    val setOrdering = new Ordering[Set[Annotation]] {
      def compare(x: Set[Annotation], y: Set[Annotation]): Int = {
        val s1 = x.toArray.sorted(elementSortOrd)
        val s2 = y.toArray.sorted(elementSortOrd)

        itOrd.compare(s1, s2)
      }
    }

    annotationOrdering(extendOrderingToNull(missingGreatest)(setOrdering))
  }

  override def str(a: Annotation): String = JsonMethods.compact(toJSON(a))

  override def genNonmissingValue: Gen[Annotation] = Gen.buildableOf[Set](elementType.genValue)

  override def desc: String =
    """
    A ``Set`` is an unordered collection with no repeated values of a given data type (ex: Int, String). Sets can be constructed by specifying ``[item1, item2, ...].toSet()``.

    .. code-block:: text
        :emphasize-lines: 2

        let s = ["rabbit", "cat", "dog", "dog"].toSet()
        result: Set("cat", "dog", "rabbit")

    They can also be nested such as Set[Set[Int]]:

    .. code-block:: text
        :emphasize-lines: 2

        let s = [[1, 2, 3].toSet(), [4, 5, 5].toSet()].toSet()
        result: Set(Set(1, 2, 3), Set(4, 5))
    """

  override def scalaClassTag: ClassTag[Set[AnyRef]] = classTag[Set[AnyRef]]
}
