package is.hail

import is.hail.variant.MatrixTable
import scala.language.implicitConversions

package object testUtils {
  implicit def toRichMatrixTable(mt: MatrixTable): RichMatrixTable = new RichMatrixTable(mt)
}
