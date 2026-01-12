package is.hail.types.virtual

import is.hail.annotations.ExtendedOrdering
import is.hail.backend.HailStateManager
import is.hail.collection.FastSeq
import is.hail.utils.Interval

case class TInterval(pointType: Type) extends Type {

  override def children = FastSeq(pointType)

  override def _toPretty = s"""Interval[$pointType]"""

  override def pyString(sb: StringBuilder): Unit = {
    sb ++= "interval<"
    pointType.pyString(sb)
    sb += '>'
  }

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false): Unit = {
    sb ++= "Interval["
    pointType.pretty(sb, indent, compact)
    sb += ']'
  }

  override def _typeCheck(a: Any): Boolean = a.isInstanceOf[Interval] && {
    val i = a.asInstanceOf[Interval]
    pointType.typeCheck(i.start) && pointType.typeCheck(i.end)
  }

  override def mkOrdering(sm: HailStateManager, missingEqual: Boolean): ExtendedOrdering =
    Interval.ordering(pointType.ordering(sm), startPrimary = true, missingEqual)

  lazy val structRepresentation: TStruct =
    TStruct(
      "start" -> pointType,
      "end" -> pointType,
      "includesStart" -> TBoolean,
      "includesEnd" -> TBoolean,
    )

  override def unify(concrete: Type): Boolean = concrete match {
    case TInterval(cpointType) => pointType.unify(cpointType)
    case _ => false
  }

  override def subst() = TInterval(pointType.subst())

  override def isIsomorphicTo(t: Type): Boolean =
    t match {
      case i: TInterval => pointType isIsomorphicTo i.pointType
      case _ => false
    }
}
