package is.hail.types.physical

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.HailStateManager
import is.hail.types.virtual.TLocus
import is.hail.variant._

abstract class PLocus extends PType {
  lazy val virtualType: TLocus = TLocus(rg)

  def rg: String

  def contig(value: Long): String

  def contigType: PString

  def position(value: Code[Long]): Code[Int]

  def position(value: Long): Int

  def positionType: PInt32

  def unstagedStoreLocus(
    sm: HailStateManager,
    addr: Long,
    contig: String,
    position: Int,
    region: Region,
  ): Unit
}
