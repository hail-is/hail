package is.hail.expr.types

import is.hail.annotations.CodeOrdering
import is.hail.annotations.{Region, UnsafeOrdering, _}
import is.hail.asm4s._
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.utils._

import scala.reflect.{ClassTag, _}

case object TBinaryOptional extends TBinary(false)

case object TBinaryRequired extends TBinary(true)

class TBinary(override val required: Boolean) extends Type {
  def _toPretty = "Binary"

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Array[Byte]]

  override def genNonmissingValue: Gen[Annotation] = Gen.buildableOf(arbitrary[Byte])

  override def scalaClassTag: ClassTag[Array[Byte]] = classTag[Array[Byte]]

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
      val l1 = TBinary.loadLength(r1, o1)
      val l2 = TBinary.loadLength(r2, o2)

      val bOff1 = TBinary.bytesOffset(o1)
      val bOff2 = TBinary.bytesOffset(o2)

      val lim = math.min(l1, l2)
      var i = 0

      while (i < lim) {
        val b1 = r1.loadByte(bOff1 + i)
        val b2 = r2.loadByte(bOff2 + i)
        if (b1 != b2)
          return java.lang.Byte.compare(b1, b2)

        i += 1
      }
      Integer.compare(l1, l2)
    }
  }

  val ordering: ExtendedOrdering =
    ExtendedOrdering.extendToNull(Ordering.Iterable[Byte])

  def codeOrdering(mb: EmitMethodBuilder, other: Type): CodeOrdering = {
    assert(other isOfType this)
    new CodeOrdering {
      type T = Long

      def compareNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Int] = {
        val l1 = mb.newLocal[Int]
        val l2 = mb.newLocal[Int]
        val lim = mb.newLocal[Int]
        val i = mb.newLocal[Int]
        val cmp = mb.newLocal[Int]

        Code(
          l1 := TBinary.loadLength(rx, x),
          l2 := TBinary.loadLength(ry, y),
          lim := (l1 < l2).mux(l1, l2),
          i := 0,
          cmp := 0,
          Code.whileLoop(cmp.ceq(0) && i < lim,
            cmp := Code.invokeStatic[java.lang.Byte, Byte, Byte, Int]("compare",
              rx.loadByte(TBinary.bytesOffset(x) + i.toL),
              ry.loadByte(TBinary.bytesOffset(y) + i.toL)),
            i += 1),
          cmp.ceq(0).mux(Code.invokeStatic[java.lang.Integer, Int, Int, Int]("compare", l1, l2), cmp))
      }
    }
  }

  override def byteSize: Long = 8
}

object TBinary {
  def apply(required: Boolean = false): TBinary = if (required) TBinaryRequired else TBinaryOptional

  def unapply(t: TBinary): Option[Boolean] = Option(t.required)

  def contentAlignment: Long = 4

  def contentByteSize(length: Int): Long = 4 + length

  def contentByteSize(length: Code[Int]): Code[Long] = (const(4) + length).toL

  def loadLength(region: Region, boff: Long): Int =
    region.loadInt(boff)

  def loadLength(region: Code[Region], boff: Code[Long]): Code[Int] =
    region.loadInt(boff)

  def bytesOffset(boff: Long): Long = boff + 4

  def bytesOffset(boff: Code[Long]): Code[Long] = boff + 4L

  def allocate(region: Region, length: Int): Long = {
    region.allocate(contentAlignment, contentByteSize(length))
  }

  def allocate(region: Code[Region], length: Code[Int]): Code[Long] = {
    region.allocate(const(contentAlignment), contentByteSize(length))
  }

}
