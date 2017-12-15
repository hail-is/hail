package is.hail

import is.hail.utils.RichVariantSampleMatrix
import is.hail.variant.MatrixTable

import scala.language.implicitConversions

package object testUtils {
  implicit def toRichVariantSampleMatrix(vsm: MatrixTable): RichVariantSampleMatrix = new RichVariantSampleMatrix(vsm)
}
