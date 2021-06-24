package is.hail.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.check.Gen
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, EmitMethodBuilder, IEmitCode}
import is.hail.types.physical.stypes.interfaces.{SIntervalCode, SIntervalValue}
import is.hail.types.virtual.TInterval
import is.hail.utils._

abstract class PInterval extends PType {
  val pointType: PType

  lazy val virtualType: TInterval = TInterval(pointType.virtualType)

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

  def startOffset(off: Code[Long]): Code[Long]

  def endOffset(off: Code[Long]): Code[Long]

  def loadStart(off: Long): Long

  def loadStart(off: Code[Long]): Code[Long]

  def loadEnd(off: Long): Long

  def loadEnd(off: Code[Long]): Code[Long]

  def startDefined(off: Long): Boolean

  def endDefined(off: Long): Boolean

  def includesStart(off: Long): Boolean

  def includesEnd(off: Long): Boolean

  def startDefined(off: Code[Long]): Code[Boolean]

  def endDefined(off: Code[Long]): Code[Boolean]

  def includesStart(off: Code[Long]): Code[Boolean]

  def includesEnd(off: Code[Long]): Code[Boolean]

  override def genNonmissingValue: Gen[Annotation] = Interval.gen(pointType.virtualType.ordering, pointType.genValue)
}