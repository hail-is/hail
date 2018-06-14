package is.hail.expr.types

import is.hail.annotations._
import is.hail.asm4s.Code
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.utils._

import scala.reflect.{ClassTag, _}

case object TFloat32Optional extends TFloat32(false)
case object TFloat32Required extends TFloat32(true)

class TFloat32(override val required: Boolean) extends TNumeric {
  def _toPretty = "Float32"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("float32")
  }

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Float]

  override def str(a: Annotation): String = if (a == null) "NA" else a.asInstanceOf[Float].formatted("%.5e")

  override def genNonmissingValue: Gen[Annotation] = arbitrary[Double].map(_.toFloat)

  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double, absolute: Boolean): Boolean =
    a1 == a2 || (a1 != null && a2 != null && {
      val f1 = a1.asInstanceOf[Float]
      val f2 = a2.asInstanceOf[Float]

      (if (absolute)
        math.abs(f1 - f2) <= tolerance
      else
        D_==(f1, f2, tolerance)) ||
        (f1.isNaN && f2.isNaN) ||
        (f1.isInfinite && f2.isInfinite && ((f1 > 0 && f2 > 0) || (f1 < 0 && f2 < 0)))
    })

  override def scalaClassTag: ClassTag[java.lang.Float] = classTag[java.lang.Float]

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
      java.lang.Float.compare(r1.loadFloat(o1), r2.loadFloat(o2))
    }
  }

  val ordering: ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[Float]])

  def codeOrdering(mb: EmitMethodBuilder, other: Type): CodeOrdering = {
    assert(other isOfType this)
    new CodeOrdering {
      type T = Float

      def compareNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Int] =
        Code.invokeStatic[java.lang.Float, Float, Float, Int]("compare", x, y)

      override def ltNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x < y

      override def lteqNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x <= y

      override def gtNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x > y

      override def gteqNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x >= y

      override def equivNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x.ceq(y)
    }
  }

  override def byteSize: Long = 4
}

object TFloat32 {
  def apply(required: Boolean = false): TFloat32 = if (required) TFloat32Required else TFloat32Optional

  def unapply(t: TFloat32): Option[Boolean] = Option(t.required)
}
