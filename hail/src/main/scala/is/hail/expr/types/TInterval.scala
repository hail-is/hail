package is.hail.expr.types

import is.hail.annotations._
import is.hail.asm4s.Code
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.physical.PInterval
import is.hail.utils._

import scala.reflect.{ClassTag, classTag}


case class TInterval(pointType: Type, override val required: Boolean = false) extends ComplexType {
  lazy val physicalType: PInterval = PInterval(pointType.physicalType, required)

  override def children = FastSeq(pointType)

  def _toPretty = s"""Interval[$pointType]"""

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("interval<")
    pointType.pyString(sb)
    sb.append('>')
  }
  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("Interval[")
    pointType.pretty(sb, indent, compact)
    sb.append("]")
  }

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Interval] && {
    val i = a.asInstanceOf[Interval]
    pointType.typeCheck(i.start) && pointType.typeCheck(i.end)
  }

  override def genNonmissingValue: Gen[Annotation] = Interval.gen(pointType.ordering, pointType.genValue)

  override def scalaClassTag: ClassTag[Interval] = classTag[Interval]

  val ordering: ExtendedOrdering = Interval.ordering(pointType.ordering, startPrimary=true)

  val representation: TStruct = {
    val rep = TStruct(
      "start" -> pointType,
      "end" -> pointType,
      "includesStart" -> TBooleanRequired,
      "includesEnd" -> TBooleanRequired)
    rep.setRequired(required).asInstanceOf[TStruct]
  }

  override def unify(concrete: Type): Boolean = concrete match {
    case TInterval(cpointType, _) => pointType.unify(cpointType)
    case _ => false
  }

  override def subst() = TInterval(pointType.subst())
}
