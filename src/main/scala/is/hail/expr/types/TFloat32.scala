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
  def _toPretty = "Float32"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("float32")
  }
  val conv = FloatNumericConversion

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
      val f1 = r1.loadFloat(o1)
      val f2 = r2.loadFloat(o2)
      if (f1 == f2) 0 else java.lang.Float.compare(f1, f2)
    }
  }

  val ordering: ExtendedOrdering =
    new ExtendedOrdering {
      def compareNonnull(x: T, y: T, missingGreatest: Boolean): Int = {
        val fx = x.asInstanceOf[Float]
        val fy = y.asInstanceOf[Float]
        if (fx == fy) 0 else java.lang.Float.compare(fx, fy)
      }
    }

  override def byteSize: Long = 4
}

object TFloat32 {
  def apply(required: Boolean = false): TFloat32 = if (required) TFloat32Required else TFloat32Optional

  def unapply(t: TFloat32): Option[Boolean] = Option(t.required)
}
