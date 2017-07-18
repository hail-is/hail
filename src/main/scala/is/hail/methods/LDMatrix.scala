package is.hail.methods

import is.hail.distributedmatrix.{BlockMatrixIsDistributedMatrix, DistributedMatrix}
import is.hail.utils._
import is.hail.stats.ToNormalizedIndexedRowMatrix
import is.hail.variant.{Variant, VariantDataset}
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix}
import org.apache.spark.storage.StorageLevel


object LDMatrix {

  /**
    * Computes the LD matrix for the given VDS.
    * @param vds VDS on which to compute Pearson correlation between pairs of variants.
    * @return An LDMatrix.
    */
  def apply(vds : VariantDataset, optComputeLocally: Option[Boolean]): LDMatrix = {
    val nSamples = vds.nSamples
    val originalVariants = vds.variants.collect()
    assert(originalVariants.isSorted, "Array of variants is not sorted. This is a bug.")

    val normalizedIRM = ToNormalizedIndexedRowMatrix(vds)
    normalizedIRM.rows.persist()
    val variantKeptIndices = normalizedIRM.rows.map{case IndexedRow(idx, _) => idx.toInt}.collect()
    assert(variantKeptIndices.isSorted, "Array of kept variants is not sorted. This is a bug.")
    val variantsKept = variantKeptIndices.map(idx => originalVariants(idx))

    val nVariantsKept = variantsKept.length
    val nVariantsDropped = originalVariants.length - nVariantsKept

    info(s"Computing LD Matrix with ${variantsKept.length} variants using $nSamples samples. $nVariantsDropped variants were dropped.")

    //The indices can be expected to be correct from the zip since the VDS is backed by an OrderedRDD of variants.
    val normalizedFilteredRows = normalizedIRM.rows.zipWithIndex()
      .map {case (IndexedRow(_, data), idx) => IndexedRow(idx, data)}

    val normalizedBlockMatrix = new IndexedRowMatrix(normalizedFilteredRows).toBlockMatrixDense()
    normalizedBlockMatrix.persist(StorageLevel.MEMORY_AND_DISK)
    normalizedBlockMatrix.blocks.count()
    normalizedIRM.rows.unpersist()

    val localBound = 5000 * 5000
    val nEntries: Long = variantsKept.length * variantsKept.length

    val computeLocally = optComputeLocally.getOrElse(nEntries <= localBound)

    val nSamplesInverse = 1.0 / nSamples

    var indexedRowMatrix: IndexedRowMatrix = null

    if (computeLocally) {
      val localMat: DenseMatrix = normalizedBlockMatrix.toLocalMatrix().asInstanceOf[DenseMatrix]
      val product = localMat multiply localMat.transpose
      indexedRowMatrix =
        BlockMatrixIsDistributedMatrix.from(normalizedBlockMatrix.blocks.sparkContext, product,
          normalizedBlockMatrix.rowsPerBlock, normalizedBlockMatrix.colsPerBlock).toIndexedRowMatrix()
    }
    else {
      import is.hail.distributedmatrix.DistributedMatrix.implicits._
      val dm = DistributedMatrix[BlockMatrix]
      import dm.ops._
      indexedRowMatrix = (normalizedBlockMatrix * normalizedBlockMatrix.transpose)
        .toIndexedRowMatrix()
    }

    val scaledIndexedRowMatrix = new IndexedRowMatrix(indexedRowMatrix.rows
      .map{case IndexedRow(idx, vals) => IndexedRow(idx, vals.map(d => d * nSamplesInverse))})

    LDMatrix(scaledIndexedRowMatrix, variantsKept, vds.nSamples)
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
