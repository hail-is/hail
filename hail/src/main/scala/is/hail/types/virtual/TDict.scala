package is.hail.types.virtual

import is.hail.annotations.{Annotation, ExtendedOrdering}
import is.hail.backend.HailStateManager
import is.hail.check.Gen
import is.hail.utils._

import scala.reflect.{classTag, ClassTag}

import org.json4s.jackson.JsonMethods

final case class TDict(keyType: Type, valueType: Type) extends TContainer {
  lazy val elementType: TBaseStruct =
    (TStruct("key" -> keyType, "value" -> valueType)).asInstanceOf[TBaseStruct]

  override def canCompare(other: Type): Boolean = other match {
    case TDict(okt, ovt) => keyType.canCompare(okt) && valueType.canCompare(ovt)
    case _ => false
  }

  override def children = FastSeq(keyType, valueType)

  override def unify(concrete: Type): Boolean =
    concrete match {
      case TDict(kt, vt) => keyType.unify(kt) && valueType.unify(vt)
      case _ => false
    }

  override def subst() = TDict(keyType.subst(), valueType.subst())

  def _toPretty = s"Dict[$keyType, $valueType]"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("dict<")
    keyType.pyString(sb)
    sb.append(", ")
    valueType.pyString(sb)
    sb.append('>')
  }

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false): Unit = {
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
    a.asInstanceOf[Map[_, _]].forall { case (k, v) =>
      keyType.typeCheck(k) && valueType.typeCheck(v)
    })

  override def _showStr(a: Annotation): String =
    a.asInstanceOf[Map[Annotation, Annotation]]
      .map { case (k, v) => s"${keyType.showStr(k)}:${valueType.showStr(v)}" }
      .mkString("{", ",", "}")

  override def str(a: Annotation): String = JsonMethods.compact(toJSON(a))

  override def genNonmissingValue(sm: HailStateManager): Gen[Annotation] =
    Gen.buildableOf2[Map](Gen.zip(keyType.genValue(sm), valueType.genValue(sm)))

  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double, absolute: Boolean)
    : Boolean =
    a1 == a2 || (a1 != null && a2 != null &&
      a1.asInstanceOf[Map[Any, _]].outerJoin(a2.asInstanceOf[Map[Any, _]])
        .forall { case (_, (o1, o2)) =>
          o1.liftedZip(o2).exists { case (v1, v2) =>
            valueType.valuesSimilar(v1, v2, tolerance, absolute)
          }
        })

  override def scalaClassTag: ClassTag[Map[_, _]] = classTag[Map[_, _]]

  override def mkOrdering(sm: HailStateManager, missingEqual: Boolean): ExtendedOrdering =
    ExtendedOrdering.mapOrdering(elementType.ordering(sm), missingEqual)

  override def valueSubsetter(subtype: Type): Any => Any = {
    val subdict = subtype.asInstanceOf[TDict]
    assert(keyType == subdict.keyType)
    if (valueType == subdict.valueType)
      return identity

    val subsetValue = valueType.valueSubsetter(subdict.valueType)
    (a: Any) => a.asInstanceOf[Map[Any, Any]].mapValues(subsetValue)
  }

  override def arrayElementsRepr: TArray = TArray(elementType)

  override def isIsomorphicTo(t: Type): Boolean =
    t match {
      case d: TDict => (keyType isIsomorphicTo d.keyType) && (valueType isIsomorphicTo d.valueType)
      case _ => false
    }
}
