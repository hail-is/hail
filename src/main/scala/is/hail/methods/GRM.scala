package is.hail.methods

import is.hail.distributedmatrix.HailBlockMatrix.ops._
import is.hail.stats.ToHWENormalizedIndexedRowMatrix
import is.hail.utils._
import is.hail.variant.VariantDataset

object GRM {
  def apply(vds: VariantDataset): KinshipMatrix = {

    val (_, mat) = ToHWENormalizedIndexedRowMatrix(vds)

    val nSamples = vds.nSamples
    assert(nSamples == mat.numCols())
    val nVariants = mat.numRows() // mat cached

    val bmat = mat.toHailBlockMatrixDense().cache()
    val grm = bmat.t * bmat

    assert(grm.cols == nSamples && grm.rows == nSamples)

    KinshipMatrix(vds.hc, vds.sSignature, grm.toIndexedRowMatrix, vds.sampleIds.toArray, vds.countVariants())
  }
}
