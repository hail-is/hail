package is.hail.types.virtual

import is.hail.annotations._
import is.hail.backend.HailStateManager
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.utils._

import scala.reflect._

case object TFloat64 extends TNumeric {
  override def _toPretty = "Float64"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("float64")
  }

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Double]

  override def _showStr(a: Annotation): String = "%.02e".format(a.asInstanceOf[Double])

  override def str(a: Annotation): String = if (a == null) "NA" else "%.5e".format(a.asInstanceOf[Double])

  override def genNonmissingValue(sm: HailStateManager): Gen[Annotation] = arbitrary[Double]

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

  override def mkOrdering(sm: HailStateManager, missingEqual: Boolean): ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[Double]], missingEqual)
}
