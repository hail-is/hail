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

  def endPrimaryUnsafeOrdering(): UnsafeOrdering

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
