package is.hail.expr.types

import is.hail.annotations._
import is.hail.asm4s.Code
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.physical.PInterval
import is.hail.utils._

import scala.reflect.{ClassTag, classTag}


case class TInterval(pointType: Type, override val required: Boolean = false) extends ComplexType {
  def physicalType: PInterval = PInterval(pointType.physicalType, required)

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

  def startOffset(off: Code[Long]): Code[Long] = representation.fieldOffset(off, 0)

  def endOffset(off: Code[Long]): Code[Long] = representation.fieldOffset(off, 1)

  def loadStart(region: Region, off: Long): Long = representation.loadField(region, off, 0)

  def loadStart(region: Code[Region], off: Code[Long]): Code[Long] = representation.loadField(region, off, 0)

  def loadStart(rv: RegionValue): Long = loadStart(rv.region, rv.offset)

  def loadEnd(region: Region, off: Long): Long = representation.loadField(region, off, 1)

  def loadEnd(region: Code[Region], off: Code[Long]): Code[Long] = representation.loadField(region, off, 1)

  def loadEnd(rv: RegionValue): Long = loadEnd(rv.region, rv.offset)

  def startDefined(region: Region, off: Long): Boolean = representation.isFieldDefined(region, off, 0)

  def endDefined(region: Region, off: Long): Boolean = representation.isFieldDefined(region, off, 1)

  def includesStart(region: Region, off: Long): Boolean = region.loadBoolean(representation.loadField(region, off, 2))

  def includesEnd(region: Region, off: Long): Boolean = region.loadBoolean(representation.loadField(region, off, 3))

  def startDefined(region: Code[Region], off: Code[Long]): Code[Boolean] = representation.isFieldDefined(region, off, 0)

  def endDefined(region: Code[Region], off: Code[Long]): Code[Boolean] = representation.isFieldDefined(region, off, 1)

  def includeStart(region: Code[Region], off: Code[Long]): Code[Boolean] =
    region.loadBoolean(representation.loadField(region, off, 2))

  def includeEnd(region: Code[Region], off: Code[Long]): Code[Boolean] =
    region.loadBoolean(representation.loadField(region, off, 3))
}
