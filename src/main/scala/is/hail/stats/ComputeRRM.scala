package is.hail.stats


import breeze.linalg.DenseMatrix
import is.hail.utils._

import is.hail.variant.{Variant, VariantDataset}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vectors}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}


// diagonal values are approximately m assuming independent variants by Central Limit Theorem
object ComputeGramian {
  def withoutBlock(A: RowMatrix): IndexedRowMatrix = {
    val n = A.numCols().toInt
    val G = A.computeGramianMatrix().toArray
    LocalDenseMatrixToIndexedRowMatrix(new DenseMatrix[Double](n, n, G), A.rows.sparkContext)
  }

  def withBlock(A: IndexedRowMatrix): IndexedRowMatrix = {
    val n = A.numCols().toInt
    val B = A.toBlockMatrixDense().cache()
    val G = B.transpose.multiply(B)
    B.blocks.unpersist()
    G.toIndexedRowMatrix()
  }
}

// diagonal values are approximately 1 assuming independent variants by Central Limit Theorem
object ComputeRRM {

  def apply(vds: VariantDataset, forceBlock: Boolean = false, forceGramian: Boolean = false): (IndexedRowMatrix, Long) = {
    def scaleMatrix(matrix: Matrix, scalar: Double): Matrix = {
      Matrices.dense(matrix.numRows, matrix.numCols, matrix.toArray.map(_ * scalar))
    }

    val useBlock = (forceBlock, forceGramian) match {
      case (false, false) => vds.nSamples > 3000 // for small matrices, computeGramian fits in memory and runs faster than BlockMatrix product
      case (true, true) => fatal("Cannot force both Block and Gramian")
      case (b, _) => b
    }

    var rowCount: Long = -1
    var computedGramian: IndexedRowMatrix = null
    if (useBlock) {
      val A = ToNormalizedIndexedRowMatrix(vds)
      rowCount = A.rows.count()
      computedGramian = ComputeGramian.withBlock(A)
    } else {
      val A = ToNormalizedRowMatrix(vds)
      rowCount = A.numRows()
      computedGramian = ComputeGramian.withoutBlock(A)
    }

    val mRec = 1d / rowCount

    (new IndexedRowMatrix(computedGramian.rows.map(ir => IndexedRow(ir.index, ir.vector.map(_ * mRec)))), rowCount)
  }
}

object LocalDenseMatrixToIndexedRowMatrix {
  def apply(dm: DenseMatrix[Double], sc: SparkContext): IndexedRowMatrix = {
    //TODO Is there a better Breeze to Spark conversion?
    val range = 0 until dm.rows
    val numberedDVs = range.map(rowNum => IndexedRow(rowNum.toLong, dm(rowNum, ::).t))
    new IndexedRowMatrix(sc.parallelize(numberedDVs))
  }
}

// each row has mean 0, norm sqrt(n), variance 1, constant variants are dropped
object ToNormalizedRowMatrix {
  def apply(vds: VariantDataset): RowMatrix = {
    require(vds.wasSplit)
    val n = vds.nSamples
    val rows = vds.rdd.flatMap { case (v, (va, gs)) => RegressionUtils.normalizedHardCalls(gs, n) }.map(Vectors.dense)
    val m = rows.count()
    new RowMatrix(rows, m, n)
  }
}

// each row has mean 0, norm sqrt(n), variance 1
object ToNormalizedIndexedRowMatrix {
  def apply(vds: VariantDataset): IndexedRowMatrix = {
    require(vds.wasSplit)
    val n = vds.nSamples
    val variants = vds.variants.collect()
    val variantIdxBc = vds.sparkContext.broadcast(variants.index)
    val indexedRows = vds.rdd.flatMap { case (v, (va, gs)) => RegressionUtils.normalizedHardCalls(gs, n).map(a => IndexedRow(variantIdxBc.value(v), Vectors.dense(a))) }
    new IndexedRowMatrix(indexedRows, variants.size, n)
  }
}

// each row has mean 0, norm approx sqrt(n), variance approx 1, constant variants are included as zero vector
object ToHWENormalizedIndexedRowMatrix {
  def apply(vds: VariantDataset): (Array[Variant], IndexedRowMatrix) = {
    require(vds.wasSplit)

    val n = vds.nSamples
    val variants = vds.variants.collect()
    val variantIdxBc = vds.sparkContext.broadcast(variants.index)

    val mat = vds.rdd.map { case (v, (va, gs)) =>
      IndexedRow(variantIdxBc.value(v), Vectors.dense(
        RegressionUtils.normalizedHardCalls(gs, n, useHWE = true, variants.size).getOrElse(Array.ofDim[Double](n))))
    }

    (variants, new IndexedRowMatrix(mat.cache(), variants.size, n))
  }

}