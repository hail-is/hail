package is.hail.types.physical

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.types.virtual.TLocus
import is.hail.variant.ReferenceGenome

abstract class PLocus extends PType {
  lazy val virtualType: TLocus = TLocus(rg)

  def rg: ReferenceGenome

  def contig(value: Long): String

  def contigType: PString

  def position(value: Code[Long]): Code[Int]

  def position(value: Long): Int

  def positionType: PInt32

  def unstagedStoreLocus(addr: Long, contig: String, position: Int, region: Region): Unit
}
