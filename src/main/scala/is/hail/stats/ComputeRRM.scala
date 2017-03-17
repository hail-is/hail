package is.hail.stats


import breeze.linalg.DenseMatrix
import is.hail.utils._
import is.hail.utils.richUtils.RichIndexedRowMatrix._

import is.hail.variant.{Variant, VariantDataset}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix, RowMatrix}

import scala.collection.parallel.immutable.ParSeq

// diagonal values are approximately m assuming independent variants by Central Limit Theorem
object ComputeGramian {
  def withoutBlock(A: RowMatrix): BlockMatrix = {
    val n = A.numCols().toInt
    val G = A.computeGramianMatrix().toArray
    LocalDenseMatrixToIndexedRowMatrix(new DenseMatrix[Double](n, n, G), A.rows.sparkContext).toBlockMatrix()
  }

  def withBlock(A: IndexedRowMatrix): BlockMatrix = {
    val n = A.numCols().toInt
    val B = A.toBlockMatrix().cache()
    val G = B.transpose.multiply(B)
    B.blocks.unpersist()
    G
  }
}

// diagonal values are approximately 1 assuming independent variants by Central Limit Theorem
object ComputeRRM {

  def apply(vds: VariantDataset, useBlock: Boolean): (BlockMatrix, Int) = {
    def scaleMatrix(matrix: Matrix, scalar: Double): Matrix = {
      Matrices.dense(matrix.numRows, matrix.numCols, matrix.toArray.map(_ * scalar))
    }

    var rowCount: Long = -1
    var computedGrammian: BlockMatrix = null
    if (useBlock) {
      val A = ToNormalizedIndexedRowMatrix(vds)
      rowCount = A.rows.count()
      computedGrammian = ComputeGramian.withBlock(A)
    } else {
      val A = ToNormalizedRowMatrix(vds)
      rowCount = A.numRows()
      computedGrammian = ComputeGramian.withoutBlock(A)
    }

    val mRec = 1d / rowCount

    val scaledBlockRDD = computedGrammian.blocks.map(tuple => tuple match {case (coords, matrix) => (coords, scaleMatrix(matrix, mRec))})
    (new BlockMatrix(scaledBlockRDD, computedGrammian.rowsPerBlock, computedGrammian.colsPerBlock), rowCount.toInt)
  }
}

object LocalDenseMatrixToIndexedRowMatrix {
  def apply(dm: DenseMatrix[Double], sc: SparkContext): IndexedRowMatrix = {
    //TODO Is there a better Breeze to Spark conversion?
    val range = (0 until dm.cols).par
    val numberedDVs: ParSeq[IndexedRow] = range.map(colNum => IndexedRow(colNum.toLong, (dm(::, colNum))))
    new IndexedRowMatrix(sc.parallelize(numberedDVs.seq))
  }
}

// each row has mean 0, norm sqrt(n), variance 1, constant variants are dropped
object ToNormalizedRowMatrix {
  def apply(vds: VariantDataset): RowMatrix = {
    require(vds.wasSplit)
    val n = vds.nSamples
    val rows = vds.rdd.flatMap { case (v, (va, gs)) => RegressionUtils.toNormalizedGtArray(gs, n) }.map(Vectors.dense)
    val m = rows.count()
    new RowMatrix(rows, m, n)
  }
}

// each row has mean 0, norm sqrt(n), variance 1, constant variants are dropped
object ToNormalizedIndexedRowMatrix {
  def apply(vds: VariantDataset): IndexedRowMatrix = {
    require(vds.wasSplit)
    val n = vds.nSamples
    val variants = vds.variants.collect()
    val variantIdxBc = vds.sparkContext.broadcast(variants.index)
    val indexedRows = vds.rdd.flatMap { case (v, (va, gs)) => RegressionUtils.toNormalizedGtArray(gs, n).map(a => IndexedRow(variantIdxBc.value(v), Vectors.dense(a))) }
    new IndexedRowMatrix(indexedRows, variants.length, n)
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
        RegressionUtils.toHWENormalizedGtArray(gs, n, variants.length).getOrElse(Array.ofDim[Double](n))))
    }

    (variants, new IndexedRowMatrix(mat.cache(), variants.length, n))
  }

}