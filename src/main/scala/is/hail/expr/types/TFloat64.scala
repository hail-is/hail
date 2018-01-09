package is.hail.expr.types

import is.hail.annotations.{Region, UnsafeOrdering, _}
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.DoubleNumericConversion
import is.hail.utils._

import scala.reflect.{ClassTag, _}

class TFloat64(override val required: Boolean) extends TNumeric {
  override def _toString = "Float64"

  val conv = DoubleNumericConversion

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Double]

  override def str(a: Annotation): String = if (a == null) "NA" else a.asInstanceOf[Double].formatted("%.5e")

  override def genNonmissingValue: Gen[Annotation] = arbitrary[Double]

  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double): Boolean =
    a1 == a2 || (a1 != null && a2 != null &&
      (D_==(a1.asInstanceOf[Double], a2.asInstanceOf[Double], tolerance) ||
        (a1.asInstanceOf[Double].isNaN && a2.asInstanceOf[Double].isNaN)))

  override def scalaClassTag: ClassTag[java.lang.Double] = classTag[java.lang.Double]

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
      java.lang.Double.compare(r1.loadDouble(o1), r2.loadDouble(o2))
    }
  }

  def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    annotationOrdering(
      extendOrderingToNull(missingGreatest)(implicitly[Ordering[Double]]))

  override def byteSize: Long = 8
}

object TFloat64 {
  def apply(required: Boolean = false): TFloat64 = if (required) TFloat64Required else TFloat64Optional

  def unapply(t: TFloat64): Option[Boolean] = Option(t.required)
}
