package is.hail.expr.types.physical

import is.hail.annotations.CodeOrdering
import is.hail.annotations.{Region, UnsafeOrdering, _}
import is.hail.asm4s._
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TBinary
import is.hail.utils._

import scala.reflect.{ClassTag, _}

abstract class PBinary extends PType {
  lazy val virtualType: TBinary = TBinary(required)

  override def unsafeOrdering(): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
      val l1 = PBinary.loadLength(r1, o1)
      val l2 = PBinary.loadLength(r2, o2)

      val bOff1 = PBinary.bytesOffset(o1)
      val bOff2 = PBinary.bytesOffset(o2)

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

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(other isOfType this)
    new CodeOrderingCompareConsistentWithOthers {
      type T = Long

      def compareNonnull(x: Code[T], y: Code[T]): Code[Int] = {
        val l1 = mb.newLocal[Int]
        val l2 = mb.newLocal[Int]
        val lim = mb.newLocal[Int]
        val i = mb.newLocal[Int]
        val cmp = mb.newLocal[Int]

        Code(
          l1 := PBinary.loadLength(x),
          l2 := PBinary.loadLength(y),
          lim := (l1 < l2).mux(l1, l2),
          i := 0,
          cmp := 0,
          Code.whileLoop(cmp.ceq(0) && i < lim,
            cmp := Code.invokeStatic[java.lang.Integer, Int, Int, Int]("compare",
              Code.invokeStatic[java.lang.Byte, Byte, Int]("toUnsignedInt", Region.loadByte(PBinary.bytesOffset(x) + i.toL)),
              Code.invokeStatic[java.lang.Byte, Byte, Int]("toUnsignedInt", Region.loadByte(PBinary.bytesOffset(y) + i.toL))),
            i += 1),
          cmp.ceq(0).mux(Code.invokeStatic[java.lang.Integer, Int, Int, Int]("compare", l1, l2), cmp))
      }
    }
  }
}

object PBinary {
  def apply(required: Boolean = false): PBinary = PCanonicalBinary(required)

  def unapply(t: PBinary): Option[Boolean] = PCanonicalBinary.unapply(t)

  def contentAlignment: Long = PCanonicalBinary.contentAlignment

  def lengthHeaderBytes: Long = PCanonicalBinary.lengthHeaderBytes

  def contentByteSize(length: Int): Long = PCanonicalBinary.contentByteSize(length)

  def contentByteSize(length: Code[Int]): Code[Long] = PCanonicalBinary.contentByteSize(length)

  def loadLength(boff: Long): Int = PCanonicalBinary.loadLength(boff)

  def loadLength(region: Region, boff: Long): Int = PCanonicalBinary.loadLength(boff)

  def loadLength(boff: Code[Long]): Code[Int] = PCanonicalBinary.loadLength(boff)

  def loadLength(region: Code[Region], boff: Code[Long]): Code[Int] = PCanonicalBinary.loadLength(boff)

  def storeLength(boff: Long, len: Int): Unit = PCanonicalBinary.storeLength(boff, len)

  def storeLength(boff: Code[Long], len: Code[Int]): Code[Unit] = PCanonicalBinary.storeLength(boff, len)

  def bytesOffset(boff: Long): Long = PCanonicalBinary.bytesOffset(boff)

  def bytesOffset(boff: Code[Long]): Code[Long] = PCanonicalBinary.bytesOffset(boff)

  def allocate(region: Region, length: Int): Long = PCanonicalBinary.allocate(region, length)

  def allocate(region: Code[Region], length: Code[Int]): Code[Long] = PCanonicalBinary.allocate(region, length)

  def store(addr: Long, bytes: Array[Byte]): Unit = PCanonicalBinary.store(addr, bytes)

  def store(addr: Code[Long], bytes: Code[Array[Byte]]): Code[Unit] = PCanonicalBinary.store(addr, bytes)
}
