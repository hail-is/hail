package is.hail.distributedmatrix

import is.hail.check.Arbitrary._
import is.hail.check.Prop._
import is.hail.check.Gen._
import is.hail.check._
import is.hail.utils._
import is.hail.SparkSuite
import HailBlockMatrix.ops._

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import org.testng.annotations.Test
import scala.reflect.ClassTag
import org.scalactic._

import scala.util.Random

class HailBlockMatrixSuite extends SparkSuite {

  private val defaultBlockSize = 1024

  def toBM(rows: Seq[Array[Double]]): HailBlockMatrix =
    toBM(rows, defaultBlockSize)

  def toBM(rows: Seq[Array[Double]], blockSize: Int): HailBlockMatrix = {
    val n = rows.length
    val m = if (n == 0) 0 else rows(0).length

    HailBlockMatrix.from(sc, new BDM[Double](m, n, rows.flatten.toArray).t, blockSize)
  }

  def toBM(x: BDM[Double]): HailBlockMatrix =
    toBM(x, defaultBlockSize)

  def toBM(x: BDM[Double], blockSize: Int): HailBlockMatrix =
    HailBlockMatrix.from(sc, x, blockSize)

  def toBreeze(a: Array[Double]): BDV[Double] =
    new BDV(a)

  def blockMatrixPreGen(blockSize: Int): Gen[HailBlockMatrix] = for {
    (l, w) <- Gen.nonEmptySquareOfAreaAtMostSize
    bm <- blockMatrixPreGen(l, w, blockSize)
  } yield bm

  def blockMatrixPreGen(rows: Int, columns: Int, blockSize: Int): Gen[HailBlockMatrix] = for {
    arrays <- Gen.buildableOfN[Seq, Array[Double]](rows, Gen.buildableOfN(columns, arbDouble.arbitrary))
  } yield toBM(arrays, blockSize)

  val squareBlockMatrixGen = for {
    size <- Gen.size
    l <- Gen.interestingPosInt
    s = math.sqrt(math.min(l, size)).toInt
    blockSize <- Gen.interestingPosInt
    bm <- blockMatrixPreGen(s, s, blockSize)
  } yield bm

  val blockMatrixGen = for {
    blockSize <- Gen.interestingPosInt
    bm <- blockMatrixPreGen(blockSize)
  } yield bm

  val twoMultipliableBlockMatrices = for {
    Array(rows, inner, columns) <- Gen.nonEmptyNCubeOfVolumeAtMostSize(3)
    blockSize <- Gen.interestingPosInt
    x <- blockMatrixPreGen(rows, inner, blockSize)
    y <- blockMatrixPreGen(inner, columns, blockSize)
  } yield (x, y)

  implicit val arbitraryHailBlockMatrix =
    Arbitrary(blockMatrixGen)

  @Test
  def pointwiseSubtractCorrect() {
    val m = toBM(Seq(
      Array[Double](1,2,3,4),
      Array[Double](5,6,7,8),
      Array[Double](9,10,11,12),
      Array[Double](13,14,15,16)))

    val expected = new BDM[Double](4,4,Array[Double](
      0,-3,-6,-9,
      3,0,-3,-6,
      6,3,0,-3,
      9,6,3,0)).t

    val actual = (m :- m.t).toLocalMatrix()
    assert(actual == expected)
  }

  @Test
  def multiplyByLocalMatrix() {
    val bl = new BDM[Double](4, 4, Array[Double](
      1,2,3,4,
      5,6,7,8,
      9,10,11,12,
      13,14,15,16)).t
    val l = toBM(bl)

    val r = new BDM[Double](4, 1, Array[Double](1,2,3,4))

    assert(bl * r === (l * r).toLocalMatrix())
  }

  @Test
  def multiplyByLocalMatrix2() {
    val bl = new BDM[Double](3,8,Array[Double](
      -0.0, -0.0, 0.0,
      0.24999999999999994, 0.5000000000000001, -0.5,
      0.4999999999999998, 2.220446049250313E-16, 2.220446049250313E-16,
      0.75, 0.5, -0.5,
      0.25, -0.5, 0.5,
      0.5000000000000001, 1.232595164407831E-32, -2.220446049250313E-16,
      0.75, -0.5000000000000001, 0.5,
      1.0, -0.0, 0.0)).t
    val l = toBM(bl)

    val r = new BDM[Double](3, 4, Array[Double](1.0,0.0,1.0,
      1.0,1.0,1.0,
      1.0,1.0,0.0,
      1.0,0.0,0.0))

    assert(bl * r === (l * r).toLocalMatrix())
  }

  private def arrayEqualNaNEqualsNaN(x: Array[Double], y: Array[Double], absoluteTolerance: Double = 1e-15): Boolean = {
    if (x.length != y.length) {
      return false
    } else {
      var i = 0
      while (i < x.length) {
        if (math.abs(x(i) - y(i)) > absoluteTolerance && !(x(i).isNaN && y(i).isNaN)) {
          println(s"inequality found at $i: ${x(i)} and ${y(i)}")
          return false
        }
        i += 1
      }
      return true
    }
  }

  @Test
  def multiplySameAsBreeze() {
    {
      val a = BDM.rand[Double](4,4)
      val b = BDM.rand[Double](4,4)
      val da = toBM(a, 2)
      val db = toBM(b, 2)

      assert(arrayEqualNaNEqualsNaN((da * db).toLocalMatrix().toArray, (a * b).toArray))
    }

    {
      val a = BDM.rand[Double](9,9)
      val b = BDM.rand[Double](9,9)
      val da = toBM(a, 3)
      val db = toBM(b, 3)

      assert(arrayEqualNaNEqualsNaN((da * db).toLocalMatrix().toArray, (a * b).toArray))
    }

    {
      val a = BDM.rand[Double](9,9)
      val b = BDM.rand[Double](9,9)
      val da = toBM(a, 2)
      val db = toBM(b, 2)

      assert(arrayEqualNaNEqualsNaN((da * db).toLocalMatrix().toArray, (a * b).toArray))
    }

    {
      val a = BDM.rand[Double](2,10)
      val b = BDM.rand[Double](10,2)
      val da = toBM(a, 3)
      val db = toBM(b, 3)

      assert(arrayEqualNaNEqualsNaN((da * db).toLocalMatrix().toArray, (a * b).toArray))
    }
  }

  @Test
  def multiplySameAsBreezeRandomized() {
    forAll(twoMultipliableBlockMatrices) { case (a: HailBlockMatrix, b: HailBlockMatrix) =>
      val truth = (a * b).toLocalMatrix()
      val expected = a.toLocalMatrix() * b.toLocalMatrix()

      if (arrayEqualNaNEqualsNaN(truth.toArray, expected.toArray))
        true
      else {
        println(s"${a.toLocalMatrix()}")
        println(s"${b.toLocalMatrix()}")
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
    val result = new BDM[Double](4,4, Array[Double](
      1,5,9,13,
      4,12,20,28,
      9,21,33,45,
      16,32,48,64
    ))

    assert((l --* r).toLocalMatrix() == result)
  }

  @Test
  def rowwiseMultiplicationRandom() {
    val g = for {
      blockSize <- Gen.interestingPosInt
      l <- blockMatrixPreGen(blockSize)
      r <- Gen.buildableOfN[Array, Double](l.cols.toInt, arbitrary[Double])
    } yield (l, r)

    forAll(g) { case (l: HailBlockMatrix, r: Array[Double]) =>
      val truth = (l --* r).toLocalMatrix()
      val repeatedR = (0 until l.rows.toInt).map(x => r).flatten.toArray
      val repeatedRMatrix = new BDM(r.size, l.rows.toInt, repeatedR).t
      val expected = l.toLocalMatrix() :* repeatedRMatrix

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
    val result = new BDM[Double](4,4, Array[Double](
      1,10,27,52,
      2,12,30,56,
      3,14,33,60,
      4,16,36,64
    ))

    assert((l :* r).toLocalMatrix() == result)
  }

  @Test
  def colwiseMultiplicationRandom() {
    val g = for {
      blockSize <- Gen.interestingPosInt
      l <- blockMatrixPreGen(blockSize)
      r <- Gen.buildableOfN[Array, Double](l.rows.toInt, arbitrary[Double])
    } yield (l, r)

    forAll(g) { case (l: HailBlockMatrix, r: Array[Double]) =>
      val truth = (l :* r).toLocalMatrix()
      val repeatedR = (0 until l.cols.toInt).map(x => r).flatten.toArray
      val repeatedRMatrix = new BDM(r.size, l.cols.toInt, repeatedR)
      val expected = l.toLocalMatrix() :* repeatedRMatrix

      if (arrayEqualNaNEqualsNaN(truth.toArray, expected.toArray))
        true
      else {
        println(s"${l.toLocalMatrix().toArray.toSeq}\n*\n${r.toSeq}")
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
    val result = new BDM[Double](4,4, Array[Double](
      2, 7,12,17,
      3, 8,13,18,
      4, 9,14,19,
      5,10,15,20
    ))

    assert((l :+ r).toLocalMatrix() == result)
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
    forAll(squareBlockMatrixGen) { (mat: HailBlockMatrix) =>
      val lm = mat.toLocalMatrix()
      val diagonalLength = math.min(lm.rows, lm.cols)
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
    val local = BDM.rand[Double](numRows, numCols)
    assert(local === HailBlockMatrix.from(sc, local, numRows - 1).toLocalMatrix())
  }

  @Test
  def readWriteIdentityTrivial() {
    val actual = toBM(Seq(
      Array[Double](1,2,3,4),
      Array[Double](5,6,7,8),
      Array[Double](9,10,11,12),
      Array[Double](13,14,15,16)))

    val fname = tmpDir.createTempFile("test")
    HailBlockMatrix.write(actual, fname)
    assert(actual.toLocalMatrix() == HailBlockMatrix.read(hc, fname).toLocalMatrix())
  }

  @Test
  def readWriteIdentityRandom() {
    forAll(blockMatrixGen) { (bm: HailBlockMatrix) =>
      val fname = tmpDir.createTempFile("test")
      HailBlockMatrix.write(bm, fname)
      assert(bm.toLocalMatrix() == HailBlockMatrix.read(hc, fname).toLocalMatrix())
      true
    }.check()
  }

  @Test
  def transpose() {
    forAll(blockMatrixGen) { (bm: HailBlockMatrix) =>
      val transposed = bm.toLocalMatrix().t
      assert(transposed.rows == bm.cols)
      assert(transposed.cols == bm.rows)
      assert(transposed === bm.t.toLocalMatrix())
      true
    }.check()
  }

  @Test
  def doubleTransposeIsIdentity() {
    forAll(blockMatrixGen.resize(10)) { (bm: HailBlockMatrix) =>
      val bmt = bm.t.cache()
      val bmtt = bm.t.t.cache()
      assert(bmtt.rows == bm.rows)
      assert(bmtt.cols == bm.cols)
      assert(bmtt.toLocalMatrix() === bm.toLocalMatrix())
      assert((bmt * bmtt).toLocalMatrix() === (bmt * bm).toLocalMatrix())
      true
    }.check()
  }

  @Test
  def cachedOpsOK() {
    forAll(twoMultipliableBlockMatrices) { case (a: HailBlockMatrix, b: HailBlockMatrix) =>
      a.cache()
      b.cache()

      val truth = (a * b).toLocalMatrix()
      val expected = a.toLocalMatrix() * b.toLocalMatrix()

      if (!arrayEqualNaNEqualsNaN(truth.toArray, expected.toArray)) {
        println(s"${a.toLocalMatrix()}")
        println(s"${b.toLocalMatrix()}")
        println(s"$truth != $expected")
        assert(false)
      }

      if (!arrayEqualNaNEqualsNaN(a.t.cache().t.toLocalMatrix().toArray, a.toLocalMatrix.toArray)) {
        println(s"${a.t.cache().t.toLocalMatrix()}")
        println(s"${a.toLocalMatrix()}")
        println(s"$truth != $expected")
        assert(false)
      }

      true
    }.check()
  }

}
