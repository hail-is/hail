package is.hail.methods

import is.hail.utils._
import is.hail.stats.ToNormalizedIndexedRowMatrix
import is.hail.variant.{Variant, VariantDataset}
import is.hail.utils.richUtils.RichIndexedRowMatrix._
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.storage.StorageLevel


object LDMatrix {

  /**
    * Computes the LD matrix for the given VDS.
    * @param vds VDS on which to compute Pearson correlation between pairs of variants.
    * @return An LDMatrix.
    */
  def apply(vds : VariantDataset): LDMatrix = {
    val nSamples = vds.nSamples
    val originalVariants = vds.variants.collect().sorted

    val normalizedIRM = ToNormalizedIndexedRowMatrix(vds)
    normalizedIRM.rows.persist()
    val variantKeptIndices = normalizedIRM.rows.map{case IndexedRow(idx, _) => idx.toInt}.collect().sorted
    val variantsKept = variantKeptIndices.map(idx => originalVariants(idx))

    val numVariantsKept = variantsKept.length
    val numVariantsDropped = originalVariants.length - numVariantsKept

    info(s"Computing LDMatrix with ${variantsKept.length} variants using $nSamples samples. $numVariantsDropped variants were dropped.")

    //The indices can be expected to be correct from the zip since the VDS is backed by an OrderedRDD of variants.
    val normalizedFilteredRows = normalizedIRM.rows.zipWithIndex()
      .map {case (IndexedRow(_, data), idx) => IndexedRow(idx, data)}

    val normalizedBlockMatrix = new IndexedRowMatrix(normalizedFilteredRows).toBlockMatrixDense()
    normalizedBlockMatrix.persist(StorageLevel.MEMORY_AND_DISK)
    normalizedBlockMatrix.blocks.count()
    normalizedIRM.rows.unpersist()

    val indexedRowMatrix = (normalizedBlockMatrix multiply normalizedBlockMatrix.transpose)
      .toIndexedRowMatrix()

    val nSamplesInverse = 1.0 / nSamples
    val scaledIndexedRowMatrix = new IndexedRowMatrix(indexedRowMatrix.rows.
      map{case IndexedRow(idx, vals) => IndexedRow(idx, vals.map(d => d * nSamplesInverse))})

    LDMatrix(scaledIndexedRowMatrix, variantsKept, vds.nSamples)
  }
}

/**
  *
  * @param matrix Spark IndexedRowMatrix. Entry (i, j) encodes the r value between variants i and j.
  * @param variants Array of variants indexing the rows and columns of the matrix.
  * @param nSamples Number of samples used to compute this matrix.
  */
case class LDMatrix(matrix: IndexedRowMatrix, variants: Array[Variant], nSamples: Int)
