package is.hail.methods

import is.hail.distributedmatrix.{BlockMatrixIsDistributedMatrix, DistributedMatrix}
import is.hail.utils._
import is.hail.stats.RegressionUtils
import is.hail.variant.{Variant, VariantDataset}
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix, Vectors}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix}


object LDMatrix {
  /**
    * Computes the LD matrix for the given VDS.
    * @param vds VDS on which to compute Pearson correlation between pairs of variants.
    * @return An LDMatrix.
    */
  def apply(vds : VariantDataset, optComputeLocally: Option[Boolean]): LDMatrix = {
    val nSamples = vds.nSamples
    val nVariants = vds.countVariants()


    val filteredNormalizedHardCalls = vds.rdd.flatMap { 
      case (v, (va, gs)) => RegressionUtils.normalizedHardCalls(gs, nSamples).map(x => (v, x))
    }
    
    val variantsKept = filteredNormalizedHardCalls.map(_._1).collect()
    assert(variantsKept.isSorted, "ld_matrix: Array of variants is not sorted. This is a bug")

    val normalizedIndexedRows = filteredNormalizedHardCalls.map(_._2).zipWithIndex()
      .map{ case (values, idx) => IndexedRow(idx, Vectors.dense(values))}
    val normalizedBlockMatrix = new IndexedRowMatrix(normalizedIndexedRows).toBlockMatrixDense()

    val nVariantsKept = variantsKept.length
    val nVariantsDropped = nVariants - nVariantsKept

    info(s"Computing LD matrix with ${variantsKept.length} variants using $nSamples samples. $nVariantsDropped variants were dropped.")

    val localBound = 5000 * 5000
    val nEntries: Long = nVariantsKept * nVariantsKept
    val nSamplesInverse = 1.0 / nSamples

    val computeLocally = optComputeLocally.getOrElse(nEntries <= localBound)

    var indexedRowMatrix: IndexedRowMatrix = null

    if (computeLocally) {
      val localMat: DenseMatrix = normalizedBlockMatrix.toLocalMatrix().asInstanceOf[DenseMatrix]
      val product = localMat multiply localMat.transpose
      indexedRowMatrix =
        BlockMatrixIsDistributedMatrix.from(vds.sparkContext, product, normalizedBlockMatrix.rowsPerBlock,
          normalizedBlockMatrix.colsPerBlock).toIndexedRowMatrix()
    } else {
      import is.hail.distributedmatrix.DistributedMatrix.implicits._
      val dm = DistributedMatrix[BlockMatrix]
      import dm.ops._
      indexedRowMatrix = (normalizedBlockMatrix * normalizedBlockMatrix.t)
        .toIndexedRowMatrix()
    }

    val scaledIndexedRowMatrix = new IndexedRowMatrix(indexedRowMatrix.rows
      .map{case IndexedRow(idx, vals) => IndexedRow(idx, vals.map(d => d * nSamplesInverse))},
      variantsKept.length,
      variantsKept.length)

    persistedVDS.unpersist()

    LDMatrix(scaledIndexedRowMatrix, variantsKept, nSamples)
  }
}

/**
  *
  * @param matrix Spark IndexedRowMatrix. Entry (i, j) encodes the r value between variants i and j.
  * @param variants Array of variants indexing the rows and columns of the matrix.
  * @param nSamples Number of samples used to compute this matrix.
  */
case class LDMatrix(matrix: IndexedRowMatrix, variants: Array[Variant], nSamples: Int) extends SimilarityMatrix {
  def toLocalMatrix(): Matrix = {
    matrix.toBlockMatrixDense().toLocalMatrix()
  }
}