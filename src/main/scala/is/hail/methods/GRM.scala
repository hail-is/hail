package is.hail.methods

import is.hail.stats.ToHWENormalizedIndexedRowMatrix
import is.hail.utils._
import is.hail.variant.VariantDataset

object GRM {
  def apply(vds: VariantDataset): KinshipMatrix = {

    val (_, mat) = ToHWENormalizedIndexedRowMatrix(vds)

    val nSamples = vds.nSamples
    assert(nSamples == mat.numCols())
    val nVariants = mat.numRows() // mat cached

    val bmat = mat.toBlockMatrixDense().cache()
    val grm = bmat.transpose.multiply(bmat)

    assert(grm.numCols == nSamples
      && grm.numRows == nSamples)

    new KinshipMatrix(vds.hc, grm.toIndexedRowMatrix, vds.sampleIds.map(_.asInstanceOf[String]).toArray, vds.countVariants())
  }
}