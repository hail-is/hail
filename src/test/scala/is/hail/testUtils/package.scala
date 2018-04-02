package is.hail

import is.hail.table.Table
import is.hail.variant.MatrixTable

import scala.language.implicitConversions

package object testUtils {
  implicit def toRichMatrixTable(mt: MatrixTable): RichMatrixTable = new RichMatrixTable(mt )

  implicit def toRichTable(ht: Table): RichTable = new RichTable(ht)
}
