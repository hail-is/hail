package is.hail.methods

import is.hail.annotations.UnsafeRow
import is.hail.expr.types.TArray
import is.hail.variant.{Locus, MatrixTable, Variant}
import is.hail.utils._

object VerifyBiallelic {
  def apply(vsm: MatrixTable, method: String): MatrixTable = {
    val localRVType = vsm.rvRowType
    val locusField = vsm.rvRowType.fieldByName("locus")
    val allelesField = vsm.rvRowType.fieldByName("alleles")
    val allelesType = allelesField.typ.asInstanceOf[TArray]
    vsm.copy2(
      rvd = vsm.rvd.mapPreservesPartitioning(vsm.rvd.typ) { rv =>
        if (!localRVType.isFieldDefined(rv, allelesField.index))
          fatal(s"in $method: found missing locus")
        if (!localRVType.isFieldDefined(rv, locusField.index))
          fatal(s"in $method: found missing alleles")
        val allelesOffset = localRVType.loadField(rv, allelesField.index)
        val allelesLength = allelesType.loadLength(rv.region, allelesOffset)
        if (allelesLength != 2) {
          val ur = new UnsafeRow(localRVType, rv)
          val locus = ur.getAs[Locus](locusField.index)
          val alleles = ur.getAs[IndexedSeq[String]](allelesField.index)
          fatal(s"in $method: found non-biallelic variant: locus $locus, alleles ${ alleles.mkString(", ") }")
        }
        rv
      })
  }
}
