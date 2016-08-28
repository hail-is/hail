package is.hail.stats

import breeze.linalg._
import is.hail.utils._
import is.hail.variant.VariantDataset
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}

// each row has mean 0, norm sqrt(n), variance 1 (constant variants are dropped)
object ToNormalizedRowMatrix {
  def apply(vds: VariantDataset): RowMatrix = {
    val n = vds.nSamples
    val rows = vds.rdd.flatMap { case (v, (va, gs)) => toNormalizedGtArray(gs, n) }.map(Vectors.dense)
    val m = rows.count()
    new RowMatrix(rows, m, n)
  }
}

// each row has mean 0, norm sqrt(n), variance 1, constant variants are dropped
object ToNormalizedIndexedRowMatrix {
  def apply(vds: VariantDataset): IndexedRowMatrix = {
    val n = vds.nSamples
    val variants = vds.variants.collect()
    val variantIdxBc = vds.sparkContext.broadcast(variants.index)
    val indexedRows = vds.rdd.flatMap { case (v, (va, gs)) => toNormalizedGtArray(gs, n).map(a => IndexedRow(variantIdxBc.value(v), Vectors.dense(a))) }
    new IndexedRowMatrix(indexedRows, variants.length, n)
  }
}

// diagonal values are approximately m assuming independent variants by Central Limit Theorem
object ComputeLocalGrammian {
  def withoutBlock(A: RowMatrix): DenseMatrix[Double] = {
    val n = A.numCols().toInt
    val G = A.computeGramianMatrix().toArray
    new DenseMatrix[Double](n, n, G)
  }

  def withBlock(A: IndexedRowMatrix): DenseMatrix[Double] = {
    val n = A.numCols().toInt
    val B = A.toBlockMatrix().cache()
    val G = B.transpose.multiply(B).toLocalMatrix().toArray
    B.blocks.unpersist()
    new DenseMatrix[Double](n, n, G)
  }
}

// diagonal values are approximately 1 assuming independent variants by Central Limit Theorem
object ComputeRRM {
  def apply(vds: VariantDataset, useBlock: Boolean): (DenseMatrix[Double], Int) = {
    if (useBlock) {
      val A = ToNormalizedIndexedRowMatrix(vds)
      val mRec = 1d / A.rows.count()
      (ComputeLocalGrammian.withBlock(A) :* mRec, A.numRows().toInt)
    } else {
      val A = ToNormalizedRowMatrix(vds)
      val mRec = 1d / A.numRows()
      (ComputeLocalGrammian.withoutBlock(A) :* mRec, A.numRows().toInt)
    }
  }
}
