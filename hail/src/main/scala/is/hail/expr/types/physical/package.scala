package is.hail.expr.types

import scala.language.implicitConversions

package object physical {
  implicit def pvalueToPCode(pv: PValue): PCode = pv.get
}
