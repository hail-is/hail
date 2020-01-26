package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s.{Code, TypeInfo, _}
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TFloat32
import is.hail.utils._

import scala.reflect.{ClassTag, _}

case object PFloat32Optional extends PFloat32(false)
case object PFloat32Required extends PFloat32(true)

class PFloat32(override val required: Boolean) extends PNumeric {
  lazy val virtualType: TFloat32 = TFloat32(required)

  override type NType = PFloat32

  def _asIdent = "float32"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PFloat32")

  override def unsafeOrdering(): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
      java.lang.Float.compare(Region.loadFloat(o1), Region.loadFloat(o2))
    }
  }

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(other isOfType this)
    new CodeOrdering {
      type T = Float

      def compareNonnull(x: Code[T], y: Code[T]): Code[Int] =
        Code.invokeStatic[java.lang.Float, Float, Float, Int]("compare", x, y)

      override def ltNonnull(x: Code[T], y: Code[T]): Code[Boolean] = x < y

      override def lteqNonnull(x: Code[T], y: Code[T]): Code[Boolean] = x <= y

      override def gtNonnull(x: Code[T], y: Code[T]): Code[Boolean] = x > y

      override def gteqNonnull(x: Code[T], y: Code[T]): Code[Boolean] = x >= y

      override def equivNonnull(x: Code[T], y: Code[T]): Code[Boolean] = x.ceq(y)
    }
  }

  override def byteSize: Long = 4

  override def zero = coerce[PFloat32](const(0.0f))

  override def add(a: Code[_], b: Code[_]): Code[PFloat32] = {
    coerce[PFloat32](coerce[Float](a) + coerce[Float](b))
  }

  override def multiply(a: Code[_], b: Code[_]): Code[PFloat32] = {
    coerce[PFloat32](coerce[Float](a) * coerce[Float](b))
  }

  def storeShallowAtOffset(dstAddress: Code[Long], srcAddress: Code[Long]): Code[Unit] =
    Region.storeFloat(dstAddress, Region.loadFloat(srcAddress))

  def storeShallowAtOffset(dstAddress: Long, srcAddress: Long) =
    Region.storeFloat(dstAddress, Region.loadFloat(srcAddress))

  def copyFromType(region: Region, srcPType: PType, srcAddress: Long, forceDeep: Boolean): Long =
    srcAddress

  def copyFromType(mb: MethodBuilder, region: Code[Region], srcPType: PType, srcAddress: Code[Long], forceDeep: Boolean): Code[Long] =
    srcAddress

  def copyFromTypeAndStackValue(mb: MethodBuilder, region: Code[Region], srcPType: PType, stackValue: Code[_], forceDeep: Boolean): Code[_] =
    stackValue
}

object PFloat32 {
  def apply(required: Boolean = false): PFloat32 = if (required) PFloat32Required else PFloat32Optional

  def unapply(t: PFloat32): Option[Boolean] = Option(t.required)
}
