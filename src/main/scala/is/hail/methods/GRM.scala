package is.hail.methods

import is.hail.distributedmatrix.BlockMatrix.ops._
import is.hail.stats.ToHWENormalizedIndexedRowMatrix
import is.hail.utils._
import is.hail.variant.VariantSampleMatrix

object GRM {
  def apply(vds: VariantSampleMatrix): KinshipMatrix = {

    val (_, irm) = ToHWENormalizedIndexedRowMatrix(vds)

    val nSamples = vds.nSamples
    assert(nSamples == irm.numCols())
    val nVariants = irm.numRows() // mat cached

    val bm = irm.toHailBlockMatrix().cache()
    val grm = bm.t * bm

    assert(grm.cols == nSamples && grm.rows == nSamples)

    KinshipMatrix(vds.hc, vds.sSignature, grm.toIndexedRowMatrix, vds.sampleIds.toArray, vds.countVariants())
  }
}
