package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s.Code
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.utils._

import scala.reflect.{ClassTag, _}

case object PFloat64Optional extends PFloat64(false)
case object PFloat64Required extends PFloat64(true)

class PFloat64(override val required: Boolean) extends PNumeric {
  override def _toPretty = "Float64"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("float64")
  }

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Double]

  override def str(a: Annotation): String = if (a == null) "NA" else a.asInstanceOf[Double].formatted("%.5e")

  override def genNonmissingValue: Gen[Annotation] = arbitrary[Double]

  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double, absolute: Boolean): Boolean =
    a1 == a2 || (a1 != null && a2 != null && {
      val f1 = a1.asInstanceOf[Double]
      val f2 = a2.asInstanceOf[Double]

      val withinTol =
        if (absolute)
          math.abs(f1 - f2) <= tolerance
        else
          D_==(f1, f2, tolerance)

      f1 == f2 || withinTol || (f1.isNaN && f2.isNaN)
    })

  override def scalaClassTag: ClassTag[java.lang.Double] = classTag[java.lang.Double]

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
      java.lang.Double.compare(r1.loadDouble(o1), r2.loadDouble(o2))
    }
  }

  val ordering: ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[Double]])

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(other isOfType this)
    new CodeOrdering {
      type T = Double

      def compareNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Int] =
        Code.invokeStatic[java.lang.Double, Double, Double, Int]("compare", x, y)

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

  override def byteSize: Long = 8
}

object PFloat64 {
  def apply(required: Boolean = false): PFloat64 = if (required) PFloat64Required else PFloat64Optional

  def unapply(t: PFloat64): Option[Boolean] = Option(t.required)
}
