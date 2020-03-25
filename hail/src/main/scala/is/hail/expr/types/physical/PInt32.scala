package is.hail.expr.types.physical

import is.hail.annotations.{Region, UnsafeOrdering, _}
import is.hail.asm4s.{Code, coerce, const, _}
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TInt32

case object PInt32Optional extends PInt32(false)
case object PInt32Required extends PInt32(true)

class PInt32(override val required: Boolean) extends PNumeric with PPrimitive {
  lazy val virtualType: TInt32.type = TInt32
  def _asIdent = "int32"
  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PInt32")
  override type NType = PInt32

  override def unsafeOrdering(): UnsafeOrdering = new UnsafeOrdering {
    def compare(o1: Long, o2: Long): Int = {
      Integer.compare(Region.loadInt(o1), Region.loadInt(o2))
    }
  }

  def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering = {
    assert(other isOfType this)
    new CodeOrderingCompareConsistentWithOthers {
      type T = Int

      def compareNonnull(x: Code[T], y: Code[T]): Code[Int] =
        Code.invokeStatic[java.lang.Integer, Int, Int, Int]("compare", x, y)

      override def ltNonnull(x: Code[T], y: Code[T]): Code[Boolean] = x < y

      override def lteqNonnull(x: Code[T], y: Code[T]): Code[Boolean] = x <= y

      override def gtNonnull(x: Code[T], y: Code[T]): Code[Boolean] = x > y

      override def gteqNonnull(x: Code[T], y: Code[T]): Code[Boolean] = x >= y

      override def equivNonnull(x: Code[T], y: Code[T]): Code[Boolean] = x.ceq(y)
    }
  }

  override def byteSize: Long = 4

  override def zero = coerce[PInt32](const(0))

  override def add(a: Code[_], b: Code[_]): Code[PInt32] = {
    coerce[PInt32](coerce[Int](a) + coerce[Int](b))
  }

  override def multiply(a: Code[_], b: Code[_]): Code[PInt32] = {
    coerce[PInt32](coerce[Int](a) * coerce[Int](b))
  }

  def storePrimitiveAtAddress(addr: Code[Long], srcPType: PType, value: Code[_]): Code[Unit] =
    Region.storeInt(addr, coerce[Int](value))
}

object PInt32 {
  def apply(required: Boolean = false) = if (required) PInt32Required else PInt32Optional

  def unapply(t: PInt32): Option[Boolean] = Option(t.required)
}
