package is.hail.expr.types.physical

import is.hail.annotations.CodeOrdering
import is.hail.annotations.{Region, UnsafeOrdering, _}
import is.hail.asm4s._
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.expr.types.virtual.TBinary

abstract class PBinary extends PType {
  lazy val virtualType: TBinary.type = TBinary

  override def unsafeOrdering(): UnsafeOrdering = new UnsafeOrdering {
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

  def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering = {
    assert(other isOfType this)
    new CodeOrderingCompareConsistentWithOthers {
      type T = Long

      def compareNonnull(x: Code[T], y: Code[T]): Code[Int] = {
        val l1 = mb.newLocal[Int]()
        val l2 = mb.newLocal[Int]()
        val lim = mb.newLocal[Int]()
        val i = mb.newLocal[Int]()
        val cmp = mb.newLocal[Int]()

        Code.memoize(x, "pbin_cord_x", y, "pbin_cord_y") { (x, y) =>
            Code(
              l1 := loadLength(x),
              l2 := loadLength(y),
              lim := (l1 < l2).mux(l1, l2),
              i := 0,
              cmp := 0,
              Code.whileLoop(cmp.ceq(0) && i < lim,
                cmp := Code.invokeStatic2[java.lang.Integer, Int, Int, Int]("compare",
                  Code.invokeStatic1[java.lang.Byte, Byte, Int]("toUnsignedInt", Region.loadByte(bytesAddress(x) + i.toL)),
                  Code.invokeStatic1[java.lang.Byte, Byte, Int]("toUnsignedInt", Region.loadByte(bytesAddress(y) + i.toL))),
                i += 1),
              cmp.ceq(0).mux(Code.invokeStatic2[java.lang.Integer, Int, Int, Int]("compare", l1, l2), cmp))
        }
      }
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

  def storeLength(boff: Code[Long], len: Code[Int]): Code[Unit]

  def bytesAddress(boff: Long): Long

  def bytesAddress(boff: Code[Long]): Code[Long]

  def store(addr: Long, bytes: Array[Byte]): Unit

  def store(addr: Code[Long], bytes: Code[Array[Byte]]): Code[Unit]
}

abstract class PBinaryValue extends PValue {
  def loadLength(): Code[Int]

  def loadBytes(): Code[Array[Byte]]

  def loadByte(i: Code[Int]): Code[Byte]
}

abstract class PBinaryCode extends PCode {
  def pt: PBinary

  def loadLength(): Code[Int]

  def loadBytes(): Code[Array[Byte]]

  def memoize(cb: EmitCodeBuilder, name: String): PBinaryValue

  def memoizeField(cb: EmitCodeBuilder, name: String): PBinaryValue
}
