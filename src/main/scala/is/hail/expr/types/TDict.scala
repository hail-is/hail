package is.hail.expr.types

import is.hail.annotations.{UnsafeUtils, _}
import is.hail.check.Gen
import is.hail.utils._
import org.apache.spark.sql.Row
import org.json4s.jackson.JsonMethods

import scala.reflect.{ClassTag, _}

final case class TDict(keyType: Type, valueType: Type, override val required: Boolean = false) extends TContainer {
  val elementType: Type = !TStruct("key" -> keyType, "value" -> valueType)

  val elementByteSize: Long = UnsafeUtils.arrayElementSize(elementType)

  val contentsAlignment: Long = elementType.alignment.max(4)

  override val fundamentalType: TArray = TArray(elementType.fundamentalType, required)

  override def canCompare(other: Type): Boolean = other match {
    case TDict(okt, ovt, _) => keyType.canCompare(okt) && valueType.canCompare(ovt)
    case _ => false
  }

  override def children = Seq(keyType, valueType)

  override def unify(concrete: Type): Boolean = {
    concrete match {
      case TDict(kt, vt, _) => keyType.unify(kt) && valueType.unify(vt)
      case _ => false
    }
  }

  override def subst() = TDict(keyType.subst(), valueType.subst())

  def _toString = s"Dict[$keyType, $valueType]"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("Dict[")
    keyType.pretty(sb, indent, compact)
    if (compact)
      sb += ','
    else
      sb.append(", ")
    valueType.pretty(sb, indent, compact)
    sb.append("]")
  }

  def _typeCheck(a: Any): Boolean = a == null || (a.isInstanceOf[Map[_, _]] &&
    a.asInstanceOf[Map[_, _]].forall { case (k, v) => keyType.typeCheck(k) && valueType.typeCheck(v) })

  override def str(a: Annotation): String = JsonMethods.compact(toJSON(a))

  override def genNonmissingValue: Gen[Annotation] =
    Gen.buildableOf2[Map](Gen.zip(keyType.genValue, valueType.genValue))

  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double): Boolean =
    a1 == a2 || (a1 != null && a2 != null &&
      a1.asInstanceOf[Map[Any, _]].outerJoin(a2.asInstanceOf[Map[Any, _]])
        .forall { case (_, (o1, o2)) =>
          o1.liftedZip(o2).exists { case (v1, v2) => valueType.valuesSimilar(v1, v2, tolerance) }
        })

  override def desc: String =
    """
    A ``Dict`` is an unordered collection of key-value pairs. Each key can only appear once in the collection.
    """

  override def scalaClassTag: ClassTag[Map[_, _]] = classTag[Map[_, _]]

  val ordering: ExtendedOrdering =
    ExtendedOrdering.mapOrdering(elementType.ordering)
}
