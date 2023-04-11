package is.hail.types.virtual

import is.hail.annotations._
import is.hail.backend.HailStateManager
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.types.physical.PFloat32
import is.hail.utils._

import scala.reflect.{ClassTag, _}

case object TFloat32 extends TNumeric {
  def _toPretty = "Float32"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("float32")
  }

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Float]

  override def _showStr(a: Annotation): String = a.asInstanceOf[Float].formatted("%.02e")

  override def str(a: Annotation): String = if (a == null) "NA" else a.asInstanceOf[Float].formatted("%.5e")

  override def genNonmissingValue(sm: HailStateManager): Gen[Annotation] = arbitrary[Double].map(_.toFloat)

  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double, absolute: Boolean): Boolean =
    a1 == a2 || (a1 != null && a2 != null && {
      val f1 = a1.asInstanceOf[Float]
      val f2 = a2.asInstanceOf[Float]

      val withinTol =
        if (absolute)
          math.abs(f1 - f2) <= tolerance
        else
          D_==(f1, f2, tolerance)

      f1 == f2 || withinTol || (f1.isNaN && f2.isNaN)
    })

  override def scalaClassTag: ClassTag[java.lang.Float] = classTag[java.lang.Float]

  override def mkOrdering(sm: HailStateManager, missingEqual: Boolean): ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[Float]], missingEqual)
}
