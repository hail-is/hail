package is.hail.expr.types

import is.hail.annotations.{Region, UnsafeOrdering, _}
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.FloatNumericConversion
import is.hail.utils._

import scala.reflect.{ClassTag, _}

case object TFloat32Optional extends TFloat32(false)
case object TFloat32Required extends TFloat32(true)

class TFloat32(override val required: Boolean) extends TNumeric {
  def _toString = "Float32"

  val conv = FloatNumericConversion

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Float]

  override def str(a: Annotation): String = if (a == null) "NA" else a.asInstanceOf[Float].formatted("%.5e")

  override def genNonmissingValue: Gen[Annotation] = arbitrary[Double].map(_.toFloat)

  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double): Boolean =
    a1 == a2 || (a1 != null && a2 != null &&
      (D_==(a1.asInstanceOf[Float], a2.asInstanceOf[Float], tolerance) ||
        (a1.asInstanceOf[Double].isNaN && a2.asInstanceOf[Double].isNaN)))

  override def scalaClassTag: ClassTag[java.lang.Float] = classTag[java.lang.Float]

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
      java.lang.Float.compare(r1.loadFloat(o1), r2.loadFloat(o2))
    }
  }

  def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    annotationOrdering(
      extendOrderingToNull(missingGreatest)(implicitly[Ordering[Float]]))

  override def byteSize: Long = 4
}

object TFloat32 {
  def apply(required: Boolean = false): TFloat32 = if (required) TFloat32Required else TFloat32Optional

  def unapply(t: TFloat32): Option[Boolean] = Option(t.required)
}
