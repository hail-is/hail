package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s.Code
import is.hail.asm4s._
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TFloat64
import is.hail.utils._

import scala.reflect.{ClassTag, _}

case object PFloat64Optional extends PFloat64(false)
case object PFloat64Required extends PFloat64(true)

class PFloat64(override val required: Boolean) extends PNumeric with PPrimitive {
  lazy val virtualType: TFloat64 = TFloat64(required)

  override type NType = PFloat64

  def _asIdent = "float64"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PFloat64")

  override def unsafeOrdering(): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
      java.lang.Double.compare(Region.loadDouble(o1), Region.loadDouble(o2))
    }
  }

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(other isOfType this)
    new CodeOrdering {
      type T = Double

      def compareNonnull(x: Code[T], y: Code[T]): Code[Int] =
        Code.invokeStatic[java.lang.Double, Double, Double, Int]("compare", x, y)

      override def ltNonnull(x: Code[T], y: Code[T]): Code[Boolean] = x < y

      override def lteqNonnull(x: Code[T], y: Code[T]): Code[Boolean] = x <= y

      override def gtNonnull(x: Code[T], y: Code[T]): Code[Boolean] = x > y

      override def gteqNonnull(x: Code[T], y: Code[T]): Code[Boolean] = x >= y

      override def equivNonnull(x: Code[T], y: Code[T]): Code[Boolean] = x.ceq(y)
    }
  }

  override def byteSize: Long = 8

  override def zero = coerce[PFloat64](const(0.0))

  override def add(a: Code[_], b: Code[_]): Code[PFloat64] = {
    coerce[PFloat64](coerce[Double](a) + coerce[Double](b))
  }

  override def multiply(a: Code[_], b: Code[_]): Code[PFloat64] = {
    coerce[PFloat64](coerce[Double](a) * coerce[Double](b))
  }
}

object PFloat64 {
  def apply(required: Boolean = false): PFloat64 = if (required) PFloat64Required else PFloat64Optional

  def unapply(t: PFloat64): Option[Boolean] = Option(t.required)
}
