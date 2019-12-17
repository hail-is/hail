package is.hail.expr.types.physical

import is.hail.annotations.{CodeOrdering, _}
import is.hail.asm4s.Code
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TInterval
import is.hail.utils._

import scala.reflect.{ClassTag, classTag}

object PInterval {
  def apply(pointType: PType, required: Boolean = false) = PCanonicalInterval(pointType, required)
}

abstract class PInterval extends ComplexPType {
  val pointType: PType

  lazy val virtualType: TInterval = TInterval(pointType.virtualType, required)

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(other isOfType this)
    CodeOrdering.intervalOrdering(this, other.asInstanceOf[PInterval], mb)
  }

  def copy(required: Boolean): PInterval

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

  def endPrimaryUnsafeOrdering(): UnsafeOrdering =
    new UnsafeOrdering {
      private val pOrd = pointType.unsafeOrdering()
      def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
        val edef1 = endDefined(r1, o1)
        if (edef1 == endDefined(r2, o2)) {
          val cmp = pOrd.compare(r1, loadEnd(r1, o1), r2, loadEnd(r2, o2))
          if (cmp == 0) {
            val includesE1 = includesEnd(r1, o1)
            if (includesE1 == includesEnd(r2, o2)) {
              val sdef1 = startDefined(r1, o1)
              if (sdef1 == startDefined(r2, o2)) {
                val cmp = pOrd.compare(r1, loadStart(r1, o1), r2, loadStart(r2, o2))
                if (cmp == 0) {
                  val includesS1 = includesStart(r1, o1)
                  if (includesS1 == includesStart(r2, o2)) {
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

  def loadStart(region: Region, off: Long): Long

  def loadStart(region: Code[Region], off: Code[Long]): Code[Long]

  def loadStart(rv: RegionValue): Long

  def loadEnd(region: Region, off: Long): Long

  def loadEnd(region: Code[Region], off: Code[Long]): Code[Long]

  def loadEnd(rv: RegionValue): Long

  def startDefined(region: Region, off: Long): Boolean

  def endDefined(region: Region, off: Long): Boolean

  def includesStart(region: Region, off: Long): Boolean

  def includesEnd(region: Region, off: Long): Boolean

  def startDefined(off: Code[Long]): Code[Boolean]

  def endDefined(off: Code[Long]): Code[Boolean]

  def includeStart(off: Code[Long]): Code[Boolean]

  def includeEnd(off: Code[Long]): Code[Boolean]
}
