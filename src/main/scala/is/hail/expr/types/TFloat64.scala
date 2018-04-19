package is.hail.expr.types

import is.hail.annotations.{Region, UnsafeOrdering, _}
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.DoubleNumericConversion
import is.hail.utils._

import scala.reflect.{ClassTag, _}

case object TFloat64Optional extends TFloat64(false)
case object TFloat64Required extends TFloat64(true)

class TFloat64(override val required: Boolean) extends TNumeric {
  override def _toPretty = "Float64"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("float64")
  }

  val conv = DoubleNumericConversion

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Double]

  override def str(a: Annotation): String = if (a == null) "NA" else a.asInstanceOf[Double].formatted("%.5e")

  override def genNonmissingValue: Gen[Annotation] = arbitrary[Double]

  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double, absolute: Boolean): Boolean =
    a1 == a2 || (a1 != null && a2 != null && {
      val f1 = a1.asInstanceOf[Double]
      val f2 = a2.asInstanceOf[Double]

      (if (absolute)
        math.abs(f1 - f2) <= tolerance
      else
        D_==(f1, f2, tolerance)) ||
        (f1.isNaN && f2.isNaN) ||
        (f1.isInfinite && f2.isInfinite && ((f1 > 0 && f2 > 0) || (f1 < 0 && f2 < 0)))
    })

  override def scalaClassTag: ClassTag[java.lang.Double] = classTag[java.lang.Double]

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
      val d1 = r1.loadDouble(o1)
      val d2 = r2.loadDouble(o2)
      if (d1 == d2) 0 else java.lang.Double.compare(d1, d2)
    }
  }

  val ordering: ExtendedOrdering =
    new ExtendedOrdering {
      def compareNonnull(x: T, y: T, missingGreatest: Boolean): Int = {
        val dx = x.asInstanceOf[Double]
        val dy = y.asInstanceOf[Double]
        if (dx == dy) 0 else java.lang.Double.compare(dx, dy)
      }
    }

  override def byteSize: Long = 8
}

object TFloat64 {
  def apply(required: Boolean = false): TFloat64 = if (required) TFloat64Required else TFloat64Optional

  def unapply(t: TFloat64): Option[Boolean] = Option(t.required)
}
