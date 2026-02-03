package is.hail.types.virtual

import is.hail.annotations._
import is.hail.backend.HailStateManager
import is.hail.utils._

case object TFloat32 extends TNumeric {
  override def _toPretty = "Float32"

  override def pyString(sb: StringBuilder): Unit =
    sb ++= "float32"

  override def _typeCheck(a: Any): Boolean = a.isInstanceOf[Float]

  override def _showStr(a: Annotation): String = "%.02e".format(a.asInstanceOf[Float])

  override def str(a: Annotation): String =
    if (a == null) "NA" else "%.5e".format(a.asInstanceOf[Float])

  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double, absolute: Boolean)
    : Boolean =
    a1 == a2 || (a1 != null && a2 != null && {
      val f1 = a1.asInstanceOf[Float]
      val f2 = a2.asInstanceOf[Float]

      val withinTol =
        if (absolute)
          math.abs(f1 - f2) <= tolerance
        else
          D_==(f1.toDouble, f2.toDouble, tolerance)

      f1 == f2 || withinTol || (f1.isNaN && f2.isNaN)
    })

  override def mkOrdering(sm: HailStateManager, missingEqual: Boolean): ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[Float]], missingEqual)

  override def isIsomorphicTo(t: Type): Boolean =
    this == t
}
