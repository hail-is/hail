package is.hail.methods

import org.apache.spark.SparkContext
import org.apache.spark.broadcast._
import org.apache.spark.rdd.RDD
import is.hail.utils._
import is.hail.keytable.KeyTable
import is.hail.variant.{Variant, VariantDataset}
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.linalg.Matrix

import scala.collection.generic.CanBuildFrom
import scala.language.higherKinds
import scala.language.implicitConversions
import scala.reflect.ClassTag

object PCRelate {

  /**
    *
    * @param vds
    * @param mean (variant, (sample, mean))
    * @return
    */
  def apply(vds: VariantDataset, mean: RDD[(Variant, (Int, Double))]): RDD[((String, String), Double)] = {
    assert(vds.wasSplit, "PCRelate requires biallelic VDSes")

    // (variant, (sample, gt))
    val g = vds.rdd.flatMap { case (v, (va, gs)) =>
      gs.zipWithIndex.map { case (g, i) =>
        (v, (i, g.nNonRefAlleles.getOrElse[Int](-1): Double)) } }

    val meanPairs = mean.join(mean)
      .filter { case (_, ((i, _), (j, _))) => j >= i }
      .map { case (vi, ((s1, mean1), (s2, mean2))) =>
        ((s1, s2, vi), (mean1, mean2))
    }

    val numerator = g.join(g)
      .filter { case (_, ((i, _), (j, _))) => j >= i }
      .map { case (vi, ((s1, gt1), (s2, gt2))) =>
        ((s1, s2, vi), (gt1, gt2))
    }
      .join(meanPairs)
      .map { case ((s1, s2, vi), ((gt1, gt2), (mean1, mean2))) =>
        ((s1, s2), (gt1 - 2 * mean1) * (gt2 - 2 * mean2))
    }
      .reduceByKey(_ + _)

    val denominator = mean.join(mean)
      .filter { case (_, ((i, _), (j, _))) => j >= i }
      .map { case (vi, ((s1, mean1), (s2, mean2))) =>
        ((s1, s2), Math.sqrt(mean1 * (1 - mean1) * mean2 * (1 - mean2)))
    }
      .reduceByKey(_ + _)

    val sampleIndexToId =
      vds.sampleIds.zipWithIndex.map { case (s, i) => (i, s) }.toMap

    numerator.join(denominator)
      .map { case ((s1, s2), (numerator, denominator)) => ((s1, s2), numerator / denominator / 4) }
      .map { case ((s1, s2), x) => ((sampleIndexToId(s1), sampleIndexToId(s2)), x) }
  }

  trait Vector[T <: Vector[T]] {
    def v: T
    def apply(i: Int): Double
    def plus(that: T): T
    def size: Long
  }
  implicit class ArrayIsVector(val a: Array[Double]) extends Vector[ArrayIsVector] with Serializable {
    type V = ArrayIsVector

    def v: V = a

    def apply(i: Int): Double = a(i)
    def plus(that: V): V = new ArrayIsVector(a.zip(that.a).map { case (x: Double, y: Double) => x + y } )
    def size: Long = a.length
  }
  object Vector {
    def from(rdd: RDD[Double]): ArrayIsVector = {
      require(rdd.count() < Int.MaxValue)
      new ArrayIsVector(rdd.collect())
    }
  }
  type MyVector = Vector[ArrayIsVector]

  trait MMatrix[T <: MMatrix[T]] {
    def m: T

    def multiply(that: T): T
    def multiply(that: DenseMatrix): T
    def transpose: T
    def pointwiseAdd(that: T): T
    def pointwiseSubtract(that: T): T
    def pointwiseMultiply(that: T): T
    def pointwiseDivide(that: T): T
    def scalarMultiply(i: Double): T
    def scalarAdd(i: Double): T

    def vectorExtendAddRowWise(v: MyVector): T

    def toArrayArray: Array[Array[Double]]

    def mapRows[U](f: Array[Double] => U)(implicit uct: ClassTag[U]): RDD[U]
  }
  object BetterBlockMatrix {
    private def localMultiply(dm: DenseMatrix)(bm: BlockMatrix): BlockMatrix =
      bm.toIndexedRowMatrix().multiply(dm).toBlockMatrix()
    private def pointwiseOp(op: (Double, Double) => Double)(x: BlockMatrix, y: BlockMatrix): BlockMatrix = {
      require(x.numRows() == y.numRows())
      require(x.numCols() == y.numCols())
      require(x.rowsPerBlock == y.rowsPerBlock)
      require(x.colsPerBlock == y.colsPerBlock)
      val blocks: RDD[((Int, Int), Matrix)] = x.blocks.join(y.blocks).map { case (block, (m1, m2)) =>
        (block, new DenseMatrix(m1.numRows, m1.numCols, m1.toArray.zip(m2.toArray).map(op.tupled))) }
      new BlockMatrix(blocks, x.rowsPerBlock, x.colsPerBlock, x.numRows(), x.numCols())
    }
    private def elementMap(op: (Double) => Double)(x: BlockMatrix): BlockMatrix = {
      val blocks: RDD[((Int, Int), Matrix)] = x.blocks.map { case (block, m) =>
        (block, new DenseMatrix(m.numRows, m.numCols, m.toArray.map(op)))
      }
      new BlockMatrix(blocks, x.rowsPerBlock, x.colsPerBlock, x.numRows(), x.numCols())
    }
    private def elementMapWithRowIndex(op: (Double, Int) => Double)(x: BlockMatrix): BlockMatrix = {
      val nRows = x.numRows
      val nCols = x.numCols
      val blocks: RDD[((Int, Int), Matrix)] = x.blocks.map { case ((blockRow, blockCol), m) =>
        ((blockRow, blockCol), new DenseMatrix(m.numRows, m.numCols, m.toArray.zipWithIndex.map { case (e, j) =>
          if (blockRow * x.rowsPerBlock + j % x.colsPerBlock < nRows &&
            blockCol * x.colsPerBlock + j / x.colsPerBlock < nCols)
            op(e, blockRow * x.rowsPerBlock + j % x.colsPerBlock)
          else
            e
        }))
      }
      new BlockMatrix(blocks, x.rowsPerBlock, x.colsPerBlock, x.numRows(), x.numCols())
    }
  }
  implicit class BetterBlockMatrix(val bm: BlockMatrix) extends MMatrix[BetterBlockMatrix] {
    import BetterBlockMatrix._
    type M = BetterBlockMatrix

    def m: M = this

    private def lift(f: BlockMatrix => BlockMatrix) =
      new BetterBlockMatrix(f(this.bm))
    private def lift2(f: (BlockMatrix, BlockMatrix) => BlockMatrix)(that: BetterBlockMatrix) =
      new BetterBlockMatrix(f(this.bm, that.bm))

    def multiply(that: M): M = lift2((x: BlockMatrix, y: BlockMatrix) => x.multiply(y))(that)
    def multiply(that: DenseMatrix): M = lift(localMultiply(that))
    def transpose: M = lift(_.transpose)

    def pointwiseAdd(that: M): M =
      lift2(pointwiseOp(_ - _))(that)
    def pointwiseSubtract(that: M): M =
      lift2(pointwiseOp(_ - _))(that)
    def pointwiseMultiply(that: M): M =
      lift2(pointwiseOp(_ * _))(that)
    def pointwiseDivide(that: M): M =
      lift2(pointwiseOp(_ / _))(that)
    def scalarMultiply(i: Double): M =
      lift(elementMap(_ * i))
    def scalarAdd(i: Double): M =
      lift(elementMap(_ + i))
    def sqrt: M =
      lift(elementMap(Math.sqrt _))
    def vectorExtendAddRowWise(v: MyVector): M = {
      require(v.size == this.bm.numRows())
      lift(elementMapWithRowIndex((x,i) => x * v(i)))
    }

    def toArrayArray: Array[Array[Double]] = {
      require(this.bm.numRows() < Int.MaxValue)
      this.bm.blocks.collect()
        .groupBy { case ((blockRow, _), m) => blockRow }
        .mapValues(rowOfBlocks => rowOfBlocks
          .sortBy { case ((_, blockCol), m) => blockCol }
          .map { case (_, m) =>
            val a = m.toArray
            val r = Array.fill(this.bm.rowsPerBlock, this.bm.colsPerBlock)(0.0)
            var col = 0
            while (col < this.bm.colsPerBlock) {
              var row = 0
              while (row < this.bm.rowsPerBlock) {
                r(row)(col) = a(col * this.bm.rowsPerBlock + row)
                row += 1
              }
              col += 1
            }
            r
        }.reduce((x,y) => x.zip(y).map { case (x,y) => x ++ y }.toArray))
        .values
        .reduce((x,y) => x ++ y)
    }

    def mapRows[U](f: Array[Double] => U)(implicit uct: ClassTag[U]): RDD[U] =
      this.bm.toIndexedRowMatrix().rows.map((ir: IndexedRow) => f(ir.vector.toArray))
  }
  implicit def removeMMatrix[M <: MMatrix[M]](mm: MMatrix[M]): M = mm.m
  object MMatrix {
    def from(rdd: RDD[Array[Double]]): MMatrix[BetterBlockMatrix] =
      new IndexedRowMatrix(rdd.zipWithIndex().map { case (x, i) => new IndexedRow(i, new DenseVector(x)) })
        .toBlockMatrix()
  }

  type MyMatrix = MMatrix[BetterBlockMatrix]

  case class Result(phiHat: MyMatrix)

  def pcRelate(vds: VariantDataset, pcs: DenseMatrix): Result = {
    val g = vdsToMeanImputedMatrix(vds)

    val pcsbc = vds.sparkContext.broadcast(pcs)

    val (beta0, betas) = fitBeta(g, pcsbc)

    Result(phiHat(g, muHat(pcsbc, beta0, betas)))
  }

  def vdsToMeanImputedMatrix(vds: VariantDataset): MyMatrix = {
    val rdd = vds.rdd.mapPartitions { stuff =>
      val ols = new OLSMultipleLinearRegression()
      stuff.map { case (v, (va, gs)) =>
        val goptions = gs.map(_.gt.map(_.toDouble)).toArray
        val defined = goptions.flatMap(x => x)
        val mean = defined.sum / defined.length
        goptions.map(_.getOrElse(mean))
      }
    }
    MMatrix.from(rdd)
  }

  /**
    *  g: SNP x Sample
    *  pcs: Sample x D
    *
    *  result: (SNP, SNP x D)
    */
  def fitBeta(g: MyMatrix, pcs: Broadcast[DenseMatrix]): (MyVector, MyMatrix) = {
    println("one sample data")
    println(pcs.value.rowIter.map(x => x.toArray: IndexedSeq[Double]).toArray[IndexedSeq[Double]] : IndexedSeq[IndexedSeq[Double]])
    println(g.m.bm.blocks.collect():IndexedSeq[((Int, Int), Matrix)])
    val rdd: RDD[(Double, Array[Double])] = g.mapRows { row =>
      val aa = pcs.value.rowIter.map(_.toArray).toArray
      val ols = new OLSMultipleLinearRegression()
      ols.newSampleData(row, aa)
      val allBetas = try {
        ols.estimateRegressionParameters().toArray
      } catch {
        case e: org.apache.commons.math3.linear.SingularMatrixException =>
          println(row: Seq[Double])
          println(aa.map(x => x: Seq[Double]) : Seq[Seq[Double]])
          throw e
      }
      (allBetas(0), allBetas.slice(1, allBetas.length))
    }
    val vecRdd = rdd.map(_._1)
    val matRdd = rdd.map(_._2)
    (Vector.from(vecRdd), MMatrix.from(matRdd))
  }

  /**
    *  pcs: Sample x D
    *  betas: SNP x D
    *  beta0: SNP
    *
    *  result: SNP x Sample
    */
  def muHat(pcs: Broadcast[DenseMatrix], beta0: MyVector, betas: MyMatrix): MyMatrix = {
    println("pcs:")
    println(pcs.value)
    println("betas")
    println(betas.bm.blocks.collect():IndexedSeq[((Int, Int), Matrix)])
    println("beta0")
    println(beta0.v.a: IndexedSeq[Double])
    println("betas * pcs")
    println(betas.multiply(pcs.value.transpose).bm.blocks.collect():IndexedSeq[((Int, Int), Matrix)])
    println("betas * pcs + [beta0, beta0, ...]")
    println(betas.multiply(pcs.value.transpose).vectorExtendAddRowWise(beta0).bm.blocks.collect():IndexedSeq[((Int, Int), Matrix)])
    println("(betas * pcs + [beta0, beta0, ...])/2")
    println(betas.multiply(pcs.value.transpose).vectorExtendAddRowWise(beta0).scalarMultiply(1.0/2.0).bm.blocks.collect():IndexedSeq[((Int, Int), Matrix)])
    betas.multiply(pcs.value.transpose).vectorExtendAddRowWise(beta0).scalarMultiply(1.0/2.0)
  }

  /**
    * g: SNP x Sample
    * muHat: SNP x Sample
    **/
  def phiHat(g: MyMatrix, muHat: MyMatrix): MyMatrix = {
    val gMinusMu = g.pointwiseSubtract(muHat.scalarMultiply(2.0))
    val oneMinusMu = muHat.scalarMultiply(-1.0).scalarAdd(1.0)
    val varianceHat = muHat.pointwiseMultiply(oneMinusMu)
    println("mu")
    println(muHat.bm.blocks.collect():IndexedSeq[((Int, Int), Matrix)])
    println("1 - mu")
    println(oneMinusMu.bm.blocks.collect():IndexedSeq[((Int, Int), Matrix)])
    println("mu(1 - mu)")
    println(varianceHat.bm.blocks.collect():IndexedSeq[((Int, Int), Matrix)])
    gMinusMu.multiply(gMinusMu.transpose)
      .pointwiseDivide(varianceHat.multiply(varianceHat.transpose).sqrt)
      .scalarMultiply(1.0/4.0)
  }

}
