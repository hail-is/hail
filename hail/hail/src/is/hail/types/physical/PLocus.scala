package is.hail.types.physical

import is.hail.annotations.{Region, UnsafeOrdering}
import is.hail.asm4s._
import is.hail.types.virtual.TLocus
import is.hail.variant.ReferenceGenome

object PLocus {
  def pack(contigIdx: Int, position: Int): Long =
    (contigIdx.toLong << 32) | position.toLong
}

abstract class PLocus extends PType {
  lazy val virtualType: TLocus = TLocus(rg)

  def rg: ReferenceGenome

  def contig(value: Long): String

  def position(value: Code[Long]): Code[Int]

  def position(value: Long): Int

  def packed(value: Long): Long

  def positionType: PInt32

  def unstagedStoreLocus(
    addr: Long,
    contig: String,
    position: Int,
    region: Region,
  ): Unit

  override def unsafeOrdering(rightType: PType): UnsafeOrdering =
    rightType match {
      case right: PLocus if right.getClass != getClass =>
        require(virtualType == right.virtualType, s"$this, $right")
        new UnsafeOrdering {
          override def compare(o1: Long, o2: Long): Int =
            java.lang.Long.compare(packed(o1), right.packed(o2))
        }
      case _ =>
        super.unsafeOrdering(rightType)
    }
}
