package is.hail

import is.hail.utils.RichMatrixTable
import is.hail.variant.MatrixTable

import scala.language.implicitConversions

package object testUtils {
  implicit def toRichVariantSampleMatrix(vsm: MatrixTable): RichMatrixTable = new RichMatrixTable(vsm)
}
