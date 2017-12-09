package is.hail

import is.hail.utils.RichVariantSampleMatrix
import is.hail.variant.VariantSampleMatrix

import scala.language.implicitConversions

package object testUtils {
  implicit def toRichVariantSampleMatrix(vsm: VariantSampleMatrix): RichVariantSampleMatrix = new RichVariantSampleMatrix(vsm)
}
