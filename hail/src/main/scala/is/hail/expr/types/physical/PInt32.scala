package is.hail.expr.types.physical

import is.hail.annotations.{Region, UnsafeOrdering, _}
import is.hail.asm4s.{Code, TypeInfo, coerce, const, _}
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TInt32
import is.hail.utils._

import scala.reflect.{ClassTag, _}

case object PInt32Optional extends PInt32(false)
case object PInt32Required extends PInt32(true)

class PInt32(override val required: Boolean) extends PIntegral {
  lazy val virtualType: TInt32 = TInt32(required)
  def _asIdent = "int32"
  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PInt32")
  override type NType = PInt32

  override def unsafeOrdering(): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
      Integer.compare(Region.loadInt(o1), Region.loadInt(o2))
    }
  }

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
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

  def storeShallowAtOffset(dstAddress: Code[Long], srcAddress: Code[Long]) =
    Region.storeInt(dstAddress, Region.loadInt(srcAddress))

  def storeShallowAtOffset(dstAddress: Long, srcAddress: Long) =
    Region.storeInt(dstAddress, Region.loadInt(srcAddress))

  def copyFromType(region: Region, srcPType: PType, srcAddress: Long, forceDeep: Boolean): Long =
    srcAddress

  def copyFromType(mb: MethodBuilder, region: Code[Region], srcPType: PType, srcAddress: Code[Long], forceDeep: Boolean): Code[Long] =
    srcAddress

  def copyFromTypeAndStackValue(mb: MethodBuilder, region: Code[Region], srcPType: PType, stackValue: Code[_], forceDeep: Boolean): Code[_] =
    stackValue
}

object PInt32 {
  def apply(required: Boolean = false) = if (required) PInt32Required else PInt32Optional

  def unapply(t: PInt32): Option[Boolean] = Option(t.required)
}
