package is.hail.stats

import breeze.linalg._
import is.hail.utils._
import is.hail.utils.richUtils.RichIndexedRowMatrix._
import is.hail.variant.{Variant, VariantDataset}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix, RowMatrix}

// diagonal values are approximately m assuming independent variants by Central Limit Theorem

object ComputeGrammian {
  /*def withoutBlock(A: RowMatrix): DenseMatrix[Double] = {
    val n = A.numCols().toInt
    val G = A.computeGramianMatrix().toArray
    new DenseMatrix[Double](n, n, G)
  }*/

  def withBlock(A: IndexedRowMatrix): BlockMatrix = {
    val n = A.numCols().toInt
    val B = A.toBlockMatrix().cache()
    //val G = B.transpose.multiply(B).toLocalMatrix().toArray
    //B.blocks.unpersist()
    B.transpose.multiply(B)
    //new DenseMatrix[Double](n, n, G)
  }
}

// diagonal values are approximately 1 assuming independent variants by Central Limit Theorem
object ComputeRRM {
  def apply(vds: VariantDataset, useBlock: Boolean): (BlockMatrix, Int) = {
    //if (useBlock) {
      val A = ToNormalizedIndexedRowMatrix(vds)
      val mRec = 1d / A.rows.count()
      (ComputeGrammian.withBlock(A) :* mRec, A.numRows().toInt)
    /*} else {
      val A = ToNormalizedRowMatrix(vds)
      val mRec = 1d / A.numRows()
      (ComputeLocalGrammian.withoutBlock(A) :* mRec, A.numRows().toInt)
    }*/
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