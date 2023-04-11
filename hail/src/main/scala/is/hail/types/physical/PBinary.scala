package is.hail.types.physical

import is.hail.annotations.{Region, UnsafeOrdering}
import is.hail.asm4s._
import is.hail.backend.HailStateManager
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.virtual.TBinary

abstract class PBinary extends PType {
  lazy val virtualType: TBinary.type = TBinary

  override def unsafeOrdering(sm: HailStateManager): UnsafeOrdering = new UnsafeOrdering {
    def compare(o1: Long, o2: Long): Int = {
      val l1 = loadLength(o1)
      val l2 = loadLength(o2)

      val bOff1 = bytesAddress(o1)
      val bOff2 = bytesAddress(o2)

      val lim = math.min(l1, l2)
      var i = 0

      while (i < lim) {
        val b1 = java.lang.Byte.toUnsignedInt(Region.loadByte(bOff1 + i))
        val b2 = java.lang.Byte.toUnsignedInt(Region.loadByte(bOff2 + i))
        if (b1 != b2)
          return java.lang.Integer.compare(b1, b2)

        i += 1
      }
      Integer.compare(l1, l2)
    }
  }

  def contentAlignment: Long

  def lengthHeaderBytes: Long

  def allocate(region: Region, length: Int): Long

  def allocate(region: Code[Region], length: Code[Int]): Code[Long]

  def contentByteSize(length: Int): Long

  def contentByteSize(length: Code[Int]): Code[Long]

  def loadLength(bAddress: Long): Int

  def loadLength(bAddress: Code[Long]): Code[Int]

  def loadBytes(bAddress: Code[Long], length: Code[Int]): Code[Array[Byte]]

  def loadBytes(bAddress: Code[Long]): Code[Array[Byte]]

  def loadBytes(bAddress: Long): Array[Byte]

  def loadBytes(bAddress: Long, length: Int): Array[Byte]

  def storeLength(boff: Long, len: Int): Unit

  def storeLength(cb: EmitCodeBuilder, boff: Code[Long], len: Code[Int]): Unit

  def bytesAddress(boff: Long): Long

  def bytesAddress(boff: Code[Long]): Code[Long]

  def store(addr: Long, bytes: Array[Byte]): Unit

  def store(cb: EmitCodeBuilder, addr: Code[Long], bytes: Code[Array[Byte]]): Unit
}
