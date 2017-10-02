package is.hail.distributedmatrix

import is.hail.check.Arbitrary._
import is.hail.check.Prop._
import is.hail.check.Gen._
import is.hail.check._
import is.hail.utils._
import is.hail.SparkSuite

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.rdd.RDD
import org.testng.annotations.Test
import scala.reflect.ClassTag

import scala.util.Random

class BlockMatrixIsDistributedMatrixSuite extends SparkSuite {
  import is.hail.distributedmatrix.DistributedMatrix.implicits._

  val dm = BlockMatrixIsDistributedMatrix
  import dm.ops._

  def toBM(rows: Seq[Array[Double]]): BlockMatrix =
    new IndexedRowMatrix(sc.parallelize(rows.zipWithIndex.map { case (v, i) => IndexedRow(i, new DenseVector(v)) }),
      rows.size, if (rows.isEmpty) 0 else rows.head.length)
      .toBlockMatrixDense()

  def toBM(rows: Seq[Array[Double]], blockSize: Int): BlockMatrix =
    new IndexedRowMatrix(sc.parallelize(rows.zipWithIndex.map { case (v, i) => IndexedRow(i, new DenseVector(v)) }),
      rows.size, if (rows.isEmpty) 0 else rows.head.length)
      .toBlockMatrixDense(blockSize, blockSize)

  def toBM(rows: Seq[Array[Double]], rowsPerBlock: Int, colsPerBlock: Int): BlockMatrix =
    new IndexedRowMatrix(sc.parallelize(rows.zipWithIndex.map { case (v, i) => IndexedRow(i, new DenseVector(v)) }),
      rows.size, if (rows.isEmpty) 0 else rows.head.length)
      .toBlockMatrixDense(rowsPerBlock, colsPerBlock)

  def toBM(x: BDM[Double], rowsPerBlock: Int, colsPerBlock: Int): BlockMatrix =
    dm.from(sc, new DenseMatrix(x.rows, x.cols, x.toArray), rowsPerBlock, colsPerBlock)

  def toBreeze(bm: BlockMatrix): BDM[Double] = {
    val lm = dm.toLocalMatrix(bm)
    new BDM(lm.numRows, lm.numCols, lm.toArray)
  }

  def toBreeze(a: Array[Double]): BDV[Double] =
    new BDV(a)

  def blockMatrixPreGen(rowsPerBlock: Int, colsPerBlock: Int): Gen[BlockMatrix] = for {
    (l, w) <- Gen.nonEmptySquareOfAreaAtMostSize
    bm <- blockMatrixPreGen(l, w, rowsPerBlock, colsPerBlock)
  } yield bm

  def blockMatrixPreGen(rows: Int, columns: Int, rowsPerBlock: Int, colsPerBlock: Int): Gen[BlockMatrix] = for {
    arrays <- Gen.buildableOfN[Seq, Array[Double]](rows, Gen.buildableOfN(columns, arbDouble.arbitrary))
  } yield toBM(arrays, rowsPerBlock, colsPerBlock)

  val blockMatrixGen = for {
    rowsPerBlock <- Gen.interestingPosInt
    colsPerBlock <- Gen.interestingPosInt
    bm <- blockMatrixPreGen(rowsPerBlock, colsPerBlock)
  } yield bm

  val blockMatrixSquareBlocksGen = for {
    blockSize <- Gen.interestingPosInt
    bm <- blockMatrixPreGen(blockSize, blockSize)
  } yield bm

  val twoMultipliableBlockMatrices = for {
    Array(rows, inner, columns) <- Gen.nonEmptyNCubeOfVolumeAtMostSize(3)
    blockSize <- Gen.interestingPosInt
    x <- blockMatrixPreGen(rows, inner, blockSize, blockSize)
    y <- blockMatrixPreGen(inner, columns, blockSize, blockSize)
  } yield (x, y)

  implicit val arbitraryBlockMatrix =
    Arbitrary(blockMatrixGen)

  @Test
  def pointwiseSubtractCorrect() {
    val m = toBM(Seq(
      Array[Double](1,2,3,4),
      Array[Double](5,6,7,8),
      Array[Double](9,10,11,12),
      Array[Double](13,14,15,16)))

    val expected = Array[IndexedSeq[Double]](
      Array[Double](0,-3,-6,-9),
      Array[Double](3,0,-3,-6),
      Array[Double](6,3,0,-3),
      Array[Double](9,6,3,0)):IndexedSeq[IndexedSeq[Double]]

    val actual = (m :- (m.t)).toLocalMatrix().rowIter.map(x => x.toArray: IndexedSeq[Double]).toArray[IndexedSeq[Double]]: IndexedSeq[IndexedSeq[Double]]
    assert(actual == expected)
  }

  @Test
  def multiplyByLocalMatrix() {
    val l = toBM(Seq(
      Array[Double](1,2,3,4),
      Array[Double](5,6,7,8),
      Array[Double](9,10,11,12),
      Array[Double](13,14,15,16)))

    val r = new DenseMatrix(4, 1, Array[Double](1,2,3,4))

    assert((l.toIndexedRowMatrix().multiply(r).toBlockMatrix().toLocalMatrix().toArray: IndexedSeq[Double]) == ((l * r).toLocalMatrix().toArray: IndexedSeq[Double]))
  }

  @Test
  def multiplyByLocalMatrix2() {
    val l = toBM(Seq(
      Array[Double](-0.0, -0.0, 0.0),
      Array[Double](0.24999999999999994, 0.5000000000000001, -0.5),
      Array[Double](0.4999999999999998, 2.220446049250313E-16, 2.220446049250313E-16),
      Array[Double](0.75, 0.5, -0.5),
      Array[Double](0.25, -0.5, 0.5),
      Array[Double](0.5000000000000001, 1.232595164407831E-32, -2.220446049250313E-16),
      Array[Double](0.75, -0.5000000000000001, 0.5),
      Array[Double](1.0, -0.0, 0.0)))

    val r = new DenseMatrix(3, 4, Array[Double](1.0,0.0,1.0,
      1.0,1.0,1.0,
      1.0,1.0,0.0,
      1.0,0.0,0.0))

    assert((l.toIndexedRowMatrix().multiply(r).toBlockMatrix().toLocalMatrix().toArray: IndexedSeq[Double]) == ((l * r).toLocalMatrix().toArray: IndexedSeq[Double]))
  }

  private def arrayEqualNaNEqualsNaN(x: Array[Double], y: Array[Double]): Boolean = {
    if (x.length != y.length) {
      return false
    } else {
      var i = 0
      while (i < x.length) {
        if (x(i) != y(i) && !(x(i).isNaN && y(i).isNaN)) {
          return false
        }
        i += 1
      }
      return true
    }
  }

  @Test
  def multiplySameAsSpark() {
    forAll(twoMultipliableBlockMatrices) { case (a: BlockMatrix, b: BlockMatrix) =>
      val truth = dm.toLocalMatrix(a * b)
      val expected = dm.toLocalMatrix(a.multiply(b))

      if (arrayEqualNaNEqualsNaN(truth.toArray, expected.toArray))
        true
      else {
        println(s"$truth != $expected")
        false
      }
    }.check()
  }

  @Test
  def rowwiseMultiplication() {
    // row major
    val l = toBM(Seq(
      Array[Double](1,2,3,4),
      Array[Double](5,6,7,8),
      Array[Double](9,10,11,12),
      Array[Double](13,14,15,16)))

    val r = Array[Double](1,2,3,4)

    // col major
    val result = new DenseMatrix(4,4, Array[Double](
      1,5,9,13,
      4,12,20,28,
      9,21,33,45,
      16,32,48,64
    ))

    assert(dm.toLocalMatrix(l --* r) == result)
  }

  @Test
  def rowwiseMultiplicationRandom() {
    val g = for {
      blockSize <- Gen.interestingPosInt
      l <- blockMatrixPreGen(blockSize, blockSize)
      r <- Gen.buildableOfN[Array, Double](l.numCols().toInt, arbitrary[Double])
    } yield (l, r)

    forAll(g) { case (l: BlockMatrix, r: Array[Double]) =>
      val truth = toBreeze(l --* r)
      val repeatedR = (0 until l.numRows().toInt).map(x => r).flatten.toArray
      val repeatedRMatrix = new BDM(r.size, l.numRows().toInt, repeatedR).t
      val expected = toBreeze(l) :* repeatedRMatrix

      if (arrayEqualNaNEqualsNaN(truth.toArray, expected.toArray))
        true
      else {
        println(s"$truth != $expected")
        false
      }
    }.check()
  }

  @Test
  def colwiseMultiplication() {
    // row major
    val l = toBM(Seq(
      Array[Double](1,2,3,4),
      Array[Double](5,6,7,8),
      Array[Double](9,10,11,12),
      Array[Double](13,14,15,16)
    ))

    val r = Array[Double](1,2,3,4)

    // col major
    val result = new DenseMatrix(4,4, Array[Double](
      1,10,27,52,
      2,12,30,56,
      3,14,33,60,
      4,16,36,64
    ))

    assert(dm.toLocalMatrix(l :* r) == result)
  }

  @Test
  def colwiseMultiplicationRandom() {
    val g = for {
      blockSize <- Gen.interestingPosInt
      l <- blockMatrixPreGen(blockSize, blockSize)
      r <- Gen.buildableOfN[Array, Double](l.numRows().toInt, arbitrary[Double])
    } yield (l, r)

    forAll(g) { case (l: BlockMatrix, r: Array[Double]) =>
      val truth = toBreeze(l :* r)
      val repeatedR = (0 until l.numCols().toInt).map(x => r).flatten.toArray
      val repeatedRMatrix = new BDM(r.size, l.numCols().toInt, repeatedR)
      val expected = toBreeze(l) :* repeatedRMatrix

      if (arrayEqualNaNEqualsNaN(truth.toArray, expected.toArray))
        true
      else {
        println(s"${dm.toLocalMatrix(l).toArray.toSeq}\n*\n${r.toSeq}")
        println(s"${truth.toString(10000,10000)}\n!=\n${expected.toString(10000,10000)}")
        false
      }
    }.check()
  }

  @Test
  def colwiseAddition() {
    // row major
    val l = toBM(Seq(
      Array[Double](1,2,3,4),
      Array[Double](5,6,7,8),
      Array[Double](9,10,11,12),
      Array[Double](13,14,15,16)
    ))

    val r = Array[Double](1,2,3,4)

    // col major
    val result = new DenseMatrix(4,4, Array[Double](
      2, 7,12,17,
      3, 8,13,18,
      4, 9,14,19,
      5,10,15,20
    ))

    assert(dm.toLocalMatrix(l :+ r) == result)
  }

  @Test
  def diagonalTestTiny() {
    val l = toBM(Seq(
      Array[Double](1,2,3,4),
      Array[Double](5,6,7,8),
      Array[Double](9,10,11,12),
      Array[Double](13,14,15,16)
    ))

    assert(l.diag.toSeq == Seq(1,6,11,16))
  }

  @Test
  def diagonalTestRandomized() {
    forAll(blockMatrixSquareBlocksGen) { (mat: BlockMatrix) =>
      val lm = mat.toLocalMatrix()
      val diagonalLength = math.min(lm.numRows, lm.numCols)
      val diagonal = (0 until diagonalLength).map(i => lm(i,i)).toArray

      if (mat.diag.toSeq == diagonal.toSeq)
        true
      else {
        println(s"mat: $lm")
        println(s"${mat.diag.toSeq} != ${diagonal.toSeq}")
        false
      }
    }.check()
  }

  @Test
  def fromLocalTest() {
    val numRows = 100
    val numCols = 100
    val breezeLocal: breeze.linalg.DenseMatrix[Double] = breeze.linalg.DenseMatrix.rand[Double](numRows, numCols)
    val sparkLocal = new DenseMatrix(numRows, numCols, breezeLocal.toArray)
    BlockMatrixIsDistributedMatrix.from(sc, sparkLocal, numRows - 1, numCols - 1).blocks.count()
  }

  @Test
  def readWriteIdentityTrivial() {
    val actual = toBM(Seq(
      Array[Double](1,2,3,4),
      Array[Double](5,6,7,8),
      Array[Double](9,10,11,12),
      Array[Double](13,14,15,16)))

    val fname = tmpDir.createTempFile("test")
    dm.write(actual, fname)
    assert(actual.toLocalMatrix() == dm.read(hc, fname).toLocalMatrix())
  }

  @Test
  def scratchTest() { // FIXME: Make this bigger
    val lm = new BDM[Double](1100, 2200, (0 until (1100 * 2200)).map(_.toDouble).toArray)
    val actual = toBM(lm, 1024, 1024)
    
    val fname = "/Users/jbloom/data/block_matrix/test_matrix"
    
    dm.write(actual, fname)
    
    val bm = dm.read(hc, fname)
    
    println(bm, bm.numRowBlocks, bm.numColBlocks, bm.numRows(), bm.numCols())
    
    assert(actual.toLocalMatrix() == dm.read(hc, fname).toLocalMatrix())
  }
  
  @Test
  def readWriteIdentityRandom() {
    forAll(blockMatrixGen) { (bm: BlockMatrix) =>
      val fname = tmpDir.createTempFile("test")
      dm.write(bm, fname)
      assert(bm.toLocalMatrix() == dm.read(hc, fname).toLocalMatrix())
      true
    }.check()
  }

}
