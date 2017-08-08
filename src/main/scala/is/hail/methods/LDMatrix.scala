package is.hail.methods

import breeze.linalg.*
import is.hail.distributedmatrix.{BlockMatrixIsDistributedMatrix, DistributedMatrix}
import is.hail.utils._
import is.hail.stats.{Eigendecomposition, RegressionUtils, ToNormalizedIndexedRowMatrix, eigSymD}
import is.hail.variant.{Variant, VariantDataset}
import org.apache.spark.mllib.linalg
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
      .map{case IndexedRow(idx, vals) => IndexedRow(idx, vals.map(d => d * nSamplesInverse))})
    
    LDMatrix(scaledIndexedRowMatrix, variantsKept, nSamples)
  }
}

/**
  *
  * @param matrix Spark IndexedRowMatrix. Entry (i, j) encodes the r value between variants i and j.
  * @param variants Array of variants indexing the rows and columns of the matrix.
  * @param nSamplesUsed Number of samples used to compute this matrix.
  */
case class LDMatrix(matrix: IndexedRowMatrix, variants: Array[Variant], nSamplesUsed: Int) extends {
  def toLocalMatrix: Matrix = matrix.toBlockMatrixDense().toLocalMatrix()
  
  def eigenRRM(vds: VariantDataset, optNEigs: Option[Int]): Eigendecomposition = {
    val variantSet = variants.toSet

    val maxRank = variants.length min nSamplesUsed
    val nEigs = optNEigs.getOrElse(maxRank)
    optNEigs.foreach( k => if (k > nEigs) info(s"Requested $k evects but maximum rank is $maxRank.") )

    if (nEigs.toLong * vds.nSamples > Integer.MAX_VALUE)
      fatal(s"$nEigs eigenvectors times ${vds.nSamples} samples exceeds 2^31 - 1, the maximum size of a local matrix.")
    
    val L = matrix.toLocalMatrix().asBreeze().toDenseMatrix

    info(s"Computing eigenvectors of LD matrix...")
    val eigL = printTime(eigSymD(L))
    
    info(s"Transforming $nEigs variant eigenvectors to sample eigenvectors...")

    // G = normalized genotype matrix (n samples by m variants)
    //   = U * sqrt(S) * V.t
    // U = G * V * inv(sqrt(S))
    // L = 1 / n * G.t * G = V * S_L * V.t
    // K = 1 / m * G * G.t = U * S_K * U.t
    // S_K = S_L * n / m
    // S = S_K * m

    val n = nSamplesUsed.toDouble
    val m = variants.length
    assert(m == eigL.eigenvectors.cols)
    val V = eigL.eigenvectors(::, (m - nEigs) until m)
    val S_K =
      if (nEigs == m)
        eigL.eigenvalues :* (n / m)
      else
        (eigL.eigenvalues((m - nEigs) until m) :* (n / m)).copy
      
    val c2 = 1.0 / math.sqrt(m)
    val sqrtSInv = S_K.map(e => c2 / math.sqrt(e))

    var filteredVDS = vds.filterVariants((v, _, _) => variantSet(v))
    filteredVDS = filteredVDS.persist()
    require(filteredVDS.variants.count() == variantSet.size, "Some variants in LD matrix are missing from VDS")

    // FIXME Clean up this ugliness. Unnecessary back and forth from Breeze to Spark. (Might just need to allow multiplying block matrix by local Breeze matrix.
    val VS = V(* , ::) :* sqrtSInv
    val VSSpark = new linalg.DenseMatrix(VS.rows, VS.cols, VS.data, VS.isTranspose)

    import is.hail.distributedmatrix.DistributedMatrix.implicits._
    val dm = DistributedMatrix[BlockMatrix]
    import dm.ops._

    val sparkG = ToNormalizedIndexedRowMatrix(filteredVDS).toBlockMatrixDense().t
    val sparkU = (sparkG * VSSpark).toLocalMatrix()
    val U = sparkU.asBreeze().toDenseMatrix
    
    filteredVDS.unpersist()
    
    Eigendecomposition(vds.sSignature, vds.sampleIds.toArray, U, S_K)
  }
}