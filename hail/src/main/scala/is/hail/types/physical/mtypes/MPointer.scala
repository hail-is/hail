package is.hail.types.physical.mtypes
import is.hail.annotations.Region
import is.hail.asm4s.Code

trait MPointer extends MType {
  def byteSize: Long = 8

  def alignment: Long = 8

  // MPointer subclasses are
  override def loadNestedRepr(addr: Code[Long]): Code[Long] = Region.loadAddress(addr)
}
