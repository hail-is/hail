package is.hail.stats

import is.hail.distributedmatrix.BlockMatrix.ops._
import breeze.linalg.DenseMatrix
import is.hail.annotations.UnsafeRow
import is.hail.expr.typ.TVariant
import is.hail.methods.KinshipMatrix
import is.hail.utils._
import is.hail.variant.{HardCallView, Locus, MatrixTable, Variant}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vectors}

// diagonal values are approximately m assuming independent variants by Central Limit Theorem
object ComputeGramian {
  def withoutBlock(A: RowMatrix): IndexedRowMatrix = {
    val n = A.numCols().toInt
    val G = A.computeGramianMatrix().toArray
    LocalDenseMatrixToIndexedRowMatrix(new DenseMatrix[Double](n, n, G), A.rows.sparkContext)
  }

  def withBlock(A: IndexedRowMatrix): IndexedRowMatrix = {
    val n = A.numCols().toInt
    val B = A.toHailBlockMatrix().cache()
    val G = B.t * B
    B.blocks.unpersist()
    G.toIndexedRowMatrix()
  }
}

// diagonal values are approximately 1 assuming independent variants by Central Limit Theorem
object ComputeRRM {

  def apply(vds: MatrixTable, forceBlock: Boolean = false, forceGramian: Boolean = false): KinshipMatrix = {
    info(s"rrm: Computing Realized Relationship Matrix...")

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

    val rrm = new IndexedRowMatrix(computedGramian.rows.map(ir => IndexedRow(ir.index, ir.vector.map(_ * mRec))))

    info(s"rrm: RRM computed using $rowCount variants.")
    KinshipMatrix(vds.hc, vds.sSignature, rrm, vds.sampleIds.toArray, rowCount)
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
  def apply(vds: MatrixTable): RowMatrix = {
    val n = vds.nSamples

    val rowType = vds.rowType
    val rows = vds.rdd2.mapPartitions { it =>
      val view = HardCallView(rowType)

      it.flatMap { rv =>
        view.setRegion(rv)
        RegressionUtils.normalizedHardCalls(view, n)
          .map(Vectors.dense)
      }
    }.persist()

    new RowMatrix(rows, rows.count(), n)
  }
}

// each row has mean 0, norm sqrt(n), variance 1
object ToNormalizedIndexedRowMatrix {
  def apply(vds: MatrixTable): IndexedRowMatrix = {
    val n = vds.nSamples

    val partStarts = vds.partitionStarts()

    assert(partStarts.length == vds.rdd2.getNumPartitions + 1)
    val partStartsBc = vds.sparkContext.broadcast(partStarts)

    val rowType = vds.rowType
    val indexedRows = vds.rdd2.mapPartitionsWithIndex { case (i, it) =>
      val view = HardCallView(rowType)

      val start = partStartsBc.value(i)
      var j = 0
      it.flatMap { rv =>
        view.setRegion(rv)
        val row = RegressionUtils.normalizedHardCalls(view, n)
          .map { a => IndexedRow(start + j, Vectors.dense(a)) }
        j += 1
        row
      }
    }.persist()

    new IndexedRowMatrix(indexedRows, partStarts.last, n)
  }
}

// each row has mean 0, norm approx sqrt(n), variance approx 1, constant variants are included as zero vector
object ToHWENormalizedIndexedRowMatrix {
  def apply(vsm: MatrixTable): (Array[Variant], IndexedRowMatrix) = {
    val rowType = vsm.rowType

    val n = vsm.nSamples
    // extra leading 0 from scanLeft
    val variantsAndSizes = vsm.rdd2.mapPartitions { it =>
      val tv = rowType.fields(1).typ.asInstanceOf[TVariant]
      val ab = new ArrayBuilder[Variant]
      var n = 0
      it.foreach { rv =>
        ab += Variant.fromRegionValue(rv.region, rowType.loadField(rv, 1))
        n += 1
      }
      Iterator.single((n, ab.result()))
    }.collect()

    val variants = variantsAndSizes.flatMap(_._2)
    val nVariants = variants.length

    val partitionSizes = variantsAndSizes.map(_._1).scanLeft(0)(_ + _)
    assert(partitionSizes.length == vsm.rdd2.getNumPartitions + 1)
    val pSizeBc = vsm.sparkContext.broadcast(partitionSizes)

    val indexedRows = vsm.rdd2.mapPartitionsWithIndex { case (i, it) =>
      val view = HardCallView(rowType)

      val partitionStartIndex = pSizeBc.value(i)
      var indexInPartition = 0
      it.flatMap { rv =>
        view.setRegion(rv)
        val row = RegressionUtils.normalizedHardCalls(view, n, useHWE = true, nVariants)
          .map { a => IndexedRow(partitionStartIndex + indexInPartition, Vectors.dense(a)) }
        indexInPartition += 1
        row
      }
    }.persist()

    (variants, new IndexedRowMatrix(indexedRows, nVariants, n))
  }
}
