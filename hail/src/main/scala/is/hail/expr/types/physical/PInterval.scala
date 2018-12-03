package is.hail.expr.types.physical

import is.hail.annotations.{CodeOrdering, _}
import is.hail.asm4s.Code
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TInterval
import is.hail.utils._

import scala.reflect.{ClassTag, classTag}


case class PInterval(pointType: PType, override val required: Boolean = false) extends ComplexPType {
  lazy val virtualType: TInterval = TInterval(pointType.virtualType, required)

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

  override def scalaClassTag: ClassTag[Interval] = classTag[Interval]

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(other isOfType this)
    CodeOrdering.intervalOrdering(this, other.asInstanceOf[PInterval], mb)
  }


  override def unsafeOrdering(): UnsafeOrdering =
    new UnsafeOrdering {
      private val pOrd = pointType.unsafeOrdering()
      def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
        val sdef1 = startDefined(r1, o1)
        if (sdef1 == startDefined(r2, o2)) {
          val cmp = pOrd.compare(r1, loadStart(r1, o1), r2, loadStart(r2, o2))
          if (cmp == 0) {
            val includesS1 = includesStart(r1, o1)
            if (includesS1 == includesStart(r2, o2)) {
              val edef1 = endDefined(r1, o1)
              if (edef1 == endDefined(r2, o2)) {
                val cmp = pOrd.compare(r1, loadEnd(r1, o1), r2, loadEnd(r2, o2))
                if (cmp == 0) {
                  val includesE1 = includesEnd(r1, o1)
                  if (includesE1 == includesEnd(r2, o2)) {
                    0
                  } else if (includesE1) 1 else -1
                } else cmp
              } else if (edef1) -1 else 1
            } else if (includesS1) -1 else 1
          } else cmp
        } else {
          if (sdef1) -1 else 1
        }
      }
    }

  val representation: PStruct = {
    val rep = PStruct(
      "start" -> pointType,
      "end" -> pointType,
      "includesStart" -> PBooleanRequired,
      "includesEnd" -> PBooleanRequired)
    rep.setRequired(required).asInstanceOf[PStruct]
  }

  override def unify(concrete: PType): Boolean = concrete match {
    case PInterval(cpointType, _) => pointType.unify(cpointType)
    case _ => false
  }

  override def subst() = PInterval(pointType.subst())

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
