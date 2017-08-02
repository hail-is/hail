package is.hail.methods

import is.hail.distributedmatrix.DistributedMatrix
import is.hail.stats.ToHWENormalizedIndexedRowMatrix
import is.hail.utils._
import is.hail.variant.VariantDataset
import org.apache.spark.mllib.linalg.distributed.BlockMatrix

object GRM {
  def apply(vds: VariantDataset): KinshipMatrix = {

    val (_, mat) = ToHWENormalizedIndexedRowMatrix(vds)

    val nSamples = vds.nSamples
    assert(nSamples == mat.numCols())
    val nVariants = mat.numRows() // mat cached

    import is.hail.distributedmatrix.DistributedMatrix.implicits._
    val dm = DistributedMatrix[BlockMatrix]
    import dm.ops._

    val bmat = mat.toBlockMatrixDense().cache()
    val grm = bmat.t * bmat

    assert(grm.numCols == nSamples
      && grm.numRows == nSamples)

    KinshipMatrix(vds.hc, vds.sSignature, grm.toIndexedRowMatrix, vds.sampleIds.toArray, vds.countVariants())
  }
}