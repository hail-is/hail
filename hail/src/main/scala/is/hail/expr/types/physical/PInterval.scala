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

  def _asIdent = s"interval_of_${pointType.asIdent}"
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

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(other isOfType this)
    CodeOrdering.intervalOrdering(this, other.asInstanceOf[PInterval], mb)
  }

  override def unsafeOrdering(): UnsafeOrdering =
    new UnsafeOrdering {
      private val pOrd = pointType.unsafeOrdering()
      def compare(o1: Long, o2: Long): Int = {
        val sdef1 = startDefined(o1)
        if (sdef1 == startDefined(o2)) {
          val cmp = pOrd.compare(loadStart(o1), loadStart(o2))
          if (cmp == 0) {
            val includesS1 = includesStart(o1)
            if (includesS1 == includesStart(o2)) {
              val edef1 = endDefined(o1)
              if (edef1 == endDefined(o2)) {
                val cmp = pOrd.compare(loadEnd(o1), loadEnd(o2))
                if (cmp == 0) {
                  val includesE1 = includesEnd(o1)
                  if (includesE1 == includesEnd(o2)) {
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

  def endPrimaryUnsafeOrdering(): UnsafeOrdering =
    new UnsafeOrdering {
      private val pOrd = pointType.unsafeOrdering()
      def compare(o1: Long, o2: Long): Int = {
        val edef1 = endDefined(o1)
        if (edef1 == endDefined(o2)) {
          val cmp = pOrd.compare(loadEnd(o1), loadEnd(o2))
          if (cmp == 0) {
            val includesE1 = includesEnd(o1)
            if (includesE1 == includesEnd(o2)) {
              val sdef1 = startDefined(o1)
              if (sdef1 == startDefined(o2)) {
                val cmp = pOrd.compare(loadStart(o1), loadStart(o2))
                if (cmp == 0) {
                  val includesS1 = includesStart(o1)
                  if (includesS1 == includesStart(o2)) {
                    0
                  } else if (includesS1) 1 else -1
                } else cmp
              } else if (sdef1) -1 else 1
            } else if (includesE1) -1 else 1
          } else cmp
        } else {
          if (edef1) -1 else 1
        }
      }
    }

  val representation: PStruct = PStruct(
      required,
      "start" -> pointType,
      "end" -> pointType,
      "includesStart" -> PBooleanRequired,
      "includesEnd" -> PBooleanRequired)

  def startOffset(off: Code[Long]): Code[Long] = representation.fieldOffset(off, 0)

  def endOffset(off: Code[Long]): Code[Long] = representation.fieldOffset(off, 1)

  def loadStart(off: Long): Long = representation.loadField(off, 0)

  def loadStart(off: Code[Long]): Code[Long] = representation.loadField(off, 0)

  def loadStart(rv: RegionValue): Long = loadStart(rv.offset)

  def loadEnd(off: Long): Long = representation.loadField(off, 1)

  def loadEnd(rv: RegionValue): Long = loadEnd(rv.offset)

  def startDefined(off: Long): Boolean = representation.isFieldDefined(off, 0)

  def endDefined(off: Long): Boolean = representation.isFieldDefined(off, 1)

  def includesStart(off: Long): Boolean = Region.loadBoolean(representation.loadField(off, 2))

  def includesEnd(off: Long): Boolean = Region.loadBoolean(representation.loadField(off, 3))

  def startDefined(off: Code[Long]): Code[Boolean] = representation.isFieldDefined(off, 0)

  def endDefined(off: Code[Long]): Code[Boolean] = representation.isFieldDefined(off, 1)

  def includeStart(off: Code[Long]): Code[Boolean] =
    Region.loadBoolean(representation.loadField(off, 2))

  def includeEnd(off: Code[Long]): Code[Boolean] =
    Region.loadBoolean(representation.loadField(off, 3))
}
