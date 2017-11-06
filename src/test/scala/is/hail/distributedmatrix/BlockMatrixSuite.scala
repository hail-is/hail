package is.hail.distributedmatrix

import breeze.linalg.{DenseMatrix => BDM}
import is.hail.SparkSuite
import is.hail.check.Arbitrary._
import is.hail.check.Prop._
import is.hail.check.Gen._
import is.hail.check._
import is.hail.distributedmatrix.BlockMatrix.ops._
import is.hail.utils._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD
import org.testng.annotations.Test

class BlockMatrixSuite extends SparkSuite {

  // row major
  def toLM(rows: Int, cols: Int, data: Array[Double]): BDM[Double] =
    new BDM(rows, cols, data, 0, cols, isTranspose = true)

  def toBM(rows: Int, cols: Int, data: Array[Double]): BlockMatrix =
    toBM(new BDM(rows, cols, data, 0, rows, true))

  def toBM(rows: Seq[Array[Double]]): BlockMatrix =
    toBM(rows, BlockMatrix.defaultBlockSize)

  def toBM(rows: Seq[Array[Double]], blockSize: Int): BlockMatrix = {
    val n = rows.length
    val m = if (n == 0) 0 else rows(0).length

    BlockMatrix.from(sc, new BDM[Double](m, n, rows.flatten.toArray).t, blockSize)
  }

  def toBM(lm: BDM[Double]): BlockMatrix =
    toBM(lm, BlockMatrix.defaultBlockSize)

  def toBM(lm: BDM[Double], blockSize: Int): BlockMatrix =
    BlockMatrix.from(sc, lm, blockSize)
 
  private val defaultBlockSize = choose(0, 1 << 6)
  private val defaultDims = nonEmptySquareOfAreaAtMostSize
  private val defaultTransposed = coin()
  private val defaultElement = arbitrary[Double]

  def blockMatrixGen(
    blockSize: Gen[Int] = defaultBlockSize,
    dims: Gen[(Int, Int)] = defaultDims,
    transposed: Gen[Boolean] = defaultTransposed,
    element: Gen[Double] = defaultElement
  ): Gen[BlockMatrix] = for {
    blockSize <- blockSize
    (rows, columns) <- dims
    transposed <- transposed
    arrays <- buildableOfN[Seq, Array[Double]](rows, buildableOfN(columns, element))
    m = toBM(arrays, blockSize)
  } yield if (transposed) m.t else m

  def squareBlockMatrixGen(
    transposed: Gen[Boolean] = defaultTransposed,
    element: Gen[Double] = defaultElement
  ): Gen[BlockMatrix] = blockMatrixGen(
    blockSize = interestingPosInt.map(math.sqrt(_).toInt),
    dims = for {
      size <- size
      l <- interestingPosInt
      s = math.sqrt(math.min(l, size)).toInt
    } yield (s, s),
    transposed = transposed,
    element = element
  )

  def twoMultipliableBlockMatrices(element: Gen[Double] = defaultElement): Gen[(BlockMatrix, BlockMatrix)] = for {
    Array(rows, inner, cols) <- nonEmptyNCubeOfVolumeAtMostSize(3)
    blockSize <- interestingPosInt.map(math.pow(_, 1.0 / 3.0).toInt)
    transposed <- coin()
    l <- blockMatrixGen(const(blockSize), const(rows -> inner), const(transposed), element)
    r <- blockMatrixGen(const(blockSize), const(inner -> cols), const(transposed), element)
  } yield if (transposed) (r, l) else (l, r)

  implicit val arbitraryBlockMatrix =
    Arbitrary(blockMatrixGen())

  private val defaultRelTolerance = 1e-14

  private def sameDoubleMatrixNaNEqualsNaN(x: BDM[Double], y: BDM[Double], relTolerance: Double = defaultRelTolerance): Boolean =
    findDoubleMatrixMismatchNaNEqualsNaN(x, y, relTolerance) match {
      case Some(_) => false
      case None => true
    }

  private def findDoubleMatrixMismatchNaNEqualsNaN(x: BDM[Double], y: BDM[Double], relTolerance: Double = defaultRelTolerance): Option[(Int, Int)] = {
    assert(x.rows == y.rows && x.cols == y.cols,
      s"dimension mismatch: ${x.rows} x ${x.cols} vs ${y.rows} x ${y.cols}")
    var j = 0
    while (j < x.cols) {
      var i = 0
      while (i < x.rows) {
        if (D_==(x(i, j) - y(i, j), relTolerance) && !(x(i, j).isNaN && y(i, j).isNaN)) {
          println(x.toString(1000, 1000))
          println(y.toString(1000, 1000))
          println(s"inequality found at ($i, $j): ${x(i, j)} and ${y(i, j)}")
          return Some((i, j))
        }
        i += 1
      }
      j += 1
    }
    None
  }

  @Test
  def pointwiseSubtractCorrect() {
    val m = toBM(4, 4, Array[Double](
      1,  2,  3,  4,
      5,  6,  7,  8,
      9,  10, 11, 12,
      13, 14, 15, 16))

    val expected = toLM(4, 4, Array[Double](
      0, -3, -6, -9,
      3, 0,  -3, -6,
      6, 3,  0,  -3,
      9, 6,  3,  0))

    val actual = (m :- m.t).toLocalMatrix()
    assert(actual == expected)
  }

  @Test
  def multiplyByLocalMatrix() {
    val ll = toLM(4, 4, Array[Double](
      1,  2,  3,  4,
      5,  6,  7,  8,
      9,  10, 11, 12,
      13, 14, 15, 16))
    val l = toBM(ll)

    val lr = toLM(4, 1, Array[Double](
      1,
      2,
      3,
      4))

    assert(ll * lr === (l * lr).toLocalMatrix())
  }

  @Test
  def randomMultiplyByLocalMatrix() {
    forAll(twoMultipliableDenseMatrices[Double]) { case (ll, lr) =>
      val l = toBM(ll)
      sameDoubleMatrixNaNEqualsNaN(ll * lr, (l * lr).toLocalMatrix())
    }.check()
  }

  @Test
  def multiplySameAsBreeze() {
    def randomLm(n: Int, m: Int) = denseMatrix[Double](n,m)

    forAll(randomLm(4,4), randomLm(4,4)) { (ll, lr) =>
      val l = toBM(ll, 2)
      val r = toBM(lr, 2)

      sameDoubleMatrixNaNEqualsNaN((l * r).toLocalMatrix(), ll * lr)
    }.check()

    forAll(randomLm(9,9), randomLm(9,9)) { (ll, lr) =>
      val l = toBM(ll, 3)
      val r = toBM(lr, 3)

      sameDoubleMatrixNaNEqualsNaN((l * r).toLocalMatrix(), ll * lr)
    }.check()

    forAll(randomLm(9,9), randomLm(9,9)) { (ll, lr) =>
      val l = toBM(ll, 2)
      val r = toBM(lr, 2)

      sameDoubleMatrixNaNEqualsNaN((l * r).toLocalMatrix(), ll * lr)
    }.check()

    forAll(randomLm(2,10), randomLm(10,2)) { (ll, lr) =>
      val l = toBM(ll, 3)
      val r = toBM(lr, 3)

      sameDoubleMatrixNaNEqualsNaN((l * r).toLocalMatrix(), ll * lr)
    }.check()

    forAll(twoMultipliableDenseMatrices[Double], interestingPosInt) { case ((ll, lr), blockSize) =>
      val l = toBM(ll, blockSize)
      val r = toBM(lr, blockSize)

      sameDoubleMatrixNaNEqualsNaN((l * r).toLocalMatrix(), ll * lr)
    }.check()
  }

  @Test
  def multiplySameAsBreezeRandomized() {
    forAll(twoMultipliableBlockMatrices(nonExtremeDouble)) { case (l: BlockMatrix, r: BlockMatrix) =>
      val actual = (l * r).toLocalMatrix()
      val expected = l.toLocalMatrix() * r.toLocalMatrix()

      findDoubleMatrixMismatchNaNEqualsNaN(actual, expected) match {
        case Some((i, j)) =>
          println(s"blockSize: ${l.blockSize}")
          println(s"${l.toLocalMatrix()}")
          println(s"${r.toLocalMatrix()}")
          println(s"row: ${l.toLocalMatrix()(i,::)}")
          println(s"col: ${r.toLocalMatrix()(::,j)}")
          false
        case None =>
          true
      }
    }.check()
  }

  @Test
  def rowwiseMultiplication() {
    val l = toBM(4, 4, Array[Double](
      1,  2,  3,  4,
      5,  6,  7,  8,
      9,  10, 11, 12,
      13, 14, 15, 16))

    val v = Array[Double](1,2,3,4)

    val result = toLM(4, 4, Array[Double](
      1,  4,   9, 16,
      5,  12, 21, 32,
      9,  20, 33, 48,
      13, 28, 45, 64))

    assert((l --* v).toLocalMatrix() == result)
  }

  @Test
  def rowwiseMultiplicationRandom() {
    val g = for {
      l <- blockMatrixGen()
      v <- buildableOfN[Array, Double](l.cols.toInt, arbitrary[Double])
    } yield (l, v)

    forAll(g) { case (l: BlockMatrix, v: Array[Double]) =>
      val actual = (l --* v).toLocalMatrix()
      val repeatedR = (0 until l.rows.toInt).flatMap(_ => v).toArray
      val repeatedRMatrix = new BDM(v.length, l.rows.toInt, repeatedR).t
      val expected = l.toLocalMatrix() :* repeatedRMatrix

      sameDoubleMatrixNaNEqualsNaN(actual, expected)
    }.check()
  }

  @Test
  def colwiseMultiplication() {
    val l = toBM(4, 4, Array[Double](
      1,  2,  3,  4,
      5,  6,  7,  8,
      9,  10, 11, 12,
      13, 14, 15, 16))

    val v = Array[Double](1,2,3,4)

    val result = toLM(4, 4, Array[Double](
      1,  2,  3,  4,
      10, 12, 14, 16,
      27, 30, 33, 36,
      52, 56, 60, 64))

    assert((l :* v).toLocalMatrix() == result)
  }

  @Test
  def colwiseMultiplicationRandom() {
    val g = for {
      l <- blockMatrixGen()
      v <- buildableOfN[Array, Double](l.rows.toInt, arbitrary[Double])
    } yield (l, v)

    forAll(g) { case (l: BlockMatrix, v: Array[Double]) =>
      val actual = (l :* v).toLocalMatrix()
      val repeatedR = (0 until l.cols.toInt).flatMap(_ => v).toArray
      val repeatedRMatrix = new BDM(v.length, l.cols.toInt, repeatedR)
      val expected = l.toLocalMatrix() :* repeatedRMatrix

      if (sameDoubleMatrixNaNEqualsNaN(actual, expected))
        true
      else {
        println(s"${l.toLocalMatrix().toArray.toSeq}\n*\n${v.toSeq}")
        false
      }
    }.check()
  }

  @Test
  def colwiseAddition() {
    val l = toBM(4, 4, Array[Double](
      1,  2,  3,  4,
      5,  6,  7,  8,
      9,  10, 11, 12,
      13, 14, 15, 16))

    val v = Array[Double](1,2,3,4)

    val result = toLM(4, 4, Array[Double](
      2,  3,  4,  5,
      7,  8,  9,  10,
      12, 13, 14, 15,
      17, 18, 19, 20))

    assert((l :+ v).toLocalMatrix() == result)
  }

  @Test
  def rowwiseAddition() {
    val l = toBM(4, 4, Array[Double](
      1,  2,  3,  4,
      5,  6,  7,  8,
      9,  10, 11, 12,
      13, 14, 15, 16))

    val v = Array[Double](1,2,3,4)

    val result = toLM(4, 4, Array[Double](
      2,  4,  6,  8,
      6,  8,  10, 12,
      10, 12, 14, 16,
      14, 16, 18, 20))

    assert((l --+ v).toLocalMatrix() == result)
  }

  @Test
  def diagonalTestTiny() {
    val m = toBM(4, 4, Array[Double](
      1,  2,  3,  4,
      5,  6,  7,  8,
      9,  10, 11, 12,
      13, 14, 15, 16))

    assert(m.diag.toSeq == Seq(1,6,11,16))
  }

  @Test
  def diagonalTestRandomized() {
    forAll(squareBlockMatrixGen()) { (m: BlockMatrix) =>
      val lm = m.toLocalMatrix()
      val diagonalLength = math.min(lm.rows, lm.cols)
      val diagonal = Array.tabulate(diagonalLength)(i => lm(i,i))

      if (m.diag.toSeq == diagonal.toSeq)
        true
      else {
        println(s"lm: $lm")
        println(s"${m.diag.toSeq} != ${diagonal.toSeq}")
        false
      }
    }.check()
  }

  @Test
  def fromLocalTest() {
    forAll(denseMatrix[Double]) { lm =>
      assert(lm === BlockMatrix.from(sc, lm, lm.rows + 1).toLocalMatrix())
      assert(lm === BlockMatrix.from(sc, lm, lm.rows).toLocalMatrix())
      if (lm.rows > 1) {
        assert(lm === BlockMatrix.from(sc, lm, lm.rows - 1).toLocalMatrix())
        assert(lm === BlockMatrix.from(sc, lm, math.sqrt(lm.rows).toInt).toLocalMatrix())
      }
      true
    }.check()
  }

  @Test
  def readWriteIdentityTrivial() {
    val m = toBM(4, 4, Array[Double](
      1,  2,  3,  4,
      5,  6,  7,  8,
      9,  10, 11, 12,
      13, 14, 15, 16))

    val fname = tmpDir.createTempFile("test")
    m.write(fname)
    assert(m.toLocalMatrix() == BlockMatrix.read(hc, fname).toLocalMatrix())
  }

  @Test
  def readWriteIdentityTrivialTransposed() {
    val m = toBM(4, 4, Array[Double](
      1,  2,  3,  4,
      5,  6,  7,  8,
      9,  10, 11, 12,
      13, 14, 15, 16))

    val fname = tmpDir.createTempFile("test")
    m.t.write(fname)
    assert(m.t.toLocalMatrix() == BlockMatrix.read(hc, fname).toLocalMatrix())
  }

  @Test
  def readWriteIdentityRandom() {
    forAll(blockMatrixGen()) { (m: BlockMatrix) =>
      val fname = tmpDir.createTempFile("test")
      m.write(fname)
      assert(sameDoubleMatrixNaNEqualsNaN(m.toLocalMatrix(), BlockMatrix.read(hc, fname).toLocalMatrix()))
      true
    }.check()
  }

  @Test
  def transpose() {
    forAll(blockMatrixGen()) { (m: BlockMatrix) =>
      val transposed = m.toLocalMatrix().t
      assert(transposed.rows == m.cols)
      assert(transposed.cols == m.rows)
      assert(transposed === m.t.toLocalMatrix())
      true
    }.check()
  }

  @Test
  def doubleTransposeIsIdentity() {
    forAll(blockMatrixGen(element = nonExtremeDouble)) { (m: BlockMatrix) =>
      val mt = m.t.cache()
      val mtt = m.t.t.cache()
      assert(mtt.rows == m.rows)
      assert(mtt.cols == m.cols)
      assert(sameDoubleMatrixNaNEqualsNaN(mtt.toLocalMatrix(), m.toLocalMatrix()))
      assert(sameDoubleMatrixNaNEqualsNaN((mt * mtt).toLocalMatrix(), (mt * m).toLocalMatrix()))
      true
    }.check()
  }

  @Test
  def cachedOpsOK() {
    forAll(twoMultipliableBlockMatrices(nonExtremeDouble)) { case (l: BlockMatrix, r: BlockMatrix) =>
      l.cache()
      r.cache()

      val actual = (l * r).toLocalMatrix()
      val expected = l.toLocalMatrix() * r.toLocalMatrix()

      if (!sameDoubleMatrixNaNEqualsNaN(actual, expected)) {
        println(s"${l.toLocalMatrix()}")
        println(s"${r.toLocalMatrix()}")
        assert(false)
      }

      if (!sameDoubleMatrixNaNEqualsNaN(l.t.cache().t.toLocalMatrix(), l.toLocalMatrix())) {
        println(s"${l.t.cache().t.toLocalMatrix()}")
        println(s"${l.toLocalMatrix()}")
        assert(false)
      }

      true
    }.check()
  }

  @Test
  def toIRMToHBMIdentity() {
    forAll(blockMatrixGen()) { (m: BlockMatrix) =>
      val roundtrip = m.toIndexedRowMatrix().toHailBlockMatrix(m.blockSize)

      val roundtriplm = roundtrip.toLocalMatrix()
      val lm = m.toLocalMatrix()

      if (roundtriplm != lm) {
        println(roundtriplm)
        println(lm)
        assert(false)
      }

      true
    }.check()
  }

  @Test
  def map2RespectsTransposition() {
    val lm = toLM(4, 2, Array[Double](
      1,2,
      3,4,
      5,6,
      7,8))
    val lmt = toLM(2, 4, Array[Double](
      1,3,5,7,
      2,4,6,8))

    val m = toBM(lm)
    val mt = toBM(lmt)

    assert(m.map2(mt.t, _ + _).toLocalMatrix() === lm + lm)
    assert(mt.t.map2(m, _ + _).toLocalMatrix() === lm + lm, s"${mt.toLocalMatrix()}\n${mt.t.toLocalMatrix()}\n${m.toLocalMatrix()}")
  }

  @Test
  def map4RespectsTransposition() {
    val lm = toLM(4, 2, Array[Double](
      1,2,
      3,4,
      5,6,
      7,8))
    val lmt = toLM(2, 4, Array[Double](
      1,3,5,7,
      2,4,6,8))

    val m = toBM(lm)
    val mt = toBM(lmt)

    assert(m.map4(m, mt.t, mt.t.t.t, _ + _ + _ + _).toLocalMatrix() === lm + lm + lm + lm)
    assert(mt.map4(mt, m.t, mt.t.t, _ + _ + _ + _).toLocalMatrix() === lm.t + lm.t + lm.t + lm.t)
  }

  @Test
  def mapRespectsTransposition() {
    val lm = toLM(4, 2, Array[Double](
      1,2,
      3,4,
      5,6,
      7,8))
    val lmt = toLM(2, 4, Array[Double](
      1,3,5,7,
      2,4,6,8))

    val m = toBM(lm)
    val mt = toBM(lmt)

    assert(m.t.map(_ * 4).toLocalMatrix() === lm.t.map(_ * 4))
    assert(m.t.t.map(_ * 4).toLocalMatrix() === lm.map(_ * 4))
    assert(mt.t.map(_ * 4).toLocalMatrix() === lm.map(_ * 4))
  }

  @Test
  def mapWithIndexRespectsTransposition() {
    val lm = toLM(4, 2, Array[Double](
      1,2,
      3,4,
      5,6,
      7,8))
    val lmt = toLM(2, 4, Array[Double](
      1,3,5,7,
      2,4,6,8))

    val m = toBM(lm)
    val mt = toBM(lmt)

    assert(m.t.mapWithIndex((_,_,x) => x * 4).toLocalMatrix() === lm.t.map(_ * 4))
    assert(m.t.t.mapWithIndex((_,_,x) => x * 4).toLocalMatrix() === lm.map(_ * 4))
    assert(mt.t.mapWithIndex((_,_,x) => x * 4).toLocalMatrix() === lm.map(_ * 4))

    assert(m.t.mapWithIndex((i,j,x) => i * 10 + j + x).toLocalMatrix() ===
      mt.mapWithIndex((i,j,x) => i * 10 + j + x).toLocalMatrix())
    assert(m.t.mapWithIndex((i,j,x) => x + j * 2 + i + 1).toLocalMatrix() ===
      lm.t + lm.t)
    assert(mt.mapWithIndex((i,j,x) => x + j * 2 + i + 1).toLocalMatrix() ===
      lm.t + lm.t)
    assert(mt.t.mapWithIndex((i,j,x) => x + i * 2 + j + 1).toLocalMatrix() ===
      lm + lm)
  }

  @Test
  def map2WithIndexRespectsTransposition() {
    val lm = toLM(4, 2, Array[Double](
      1,2,
      3,4,
      5,6,
      7,8))
    val lmt = toLM(2, 4, Array[Double](
      1,3,5,7,
      2,4,6,8))

    val m = toBM(lm)
    val mt = toBM(lmt)

    assert(m.map2WithIndex(mt.t, (_,_,x,y) => x + y).toLocalMatrix() === lm + lm)
    assert(mt.map2WithIndex(m.t, (_,_,x,y) => x + y).toLocalMatrix() === lm.t + lm.t)
    assert(mt.t.map2WithIndex(m, (_,_,x,y) => x + y).toLocalMatrix() === lm + lm)
    assert(m.t.t.map2WithIndex(mt.t, (_,_,x,y) => x + y).toLocalMatrix() === lm + lm)

    assert(m.t.map2WithIndex(mt, (i,j,x,y) => i * 10 + j + x + y).toLocalMatrix() ===
      mt.map2WithIndex(m.t, (i,j,x,y) => i * 10 + j + x + y).toLocalMatrix())
    assert(m.t.map2WithIndex(m.t, (i,j,x,y) => i * 10 + j + x + y).toLocalMatrix() ===
      mt.map2WithIndex(mt, (i,j,x,y) => i * 10 + j + x + y).toLocalMatrix())
    assert(m.t.map2WithIndex(mt, (i,j,x,y) => x + 2 * y + j * 2 + i + 1).toLocalMatrix() ===
      4.0 * lm.t)
    assert(mt.map2WithIndex(m.t, (i,j,x,y) => x + 2 * y + j * 2 + i + 1).toLocalMatrix() ===
      4.0 * lm.t)
    assert(mt.t.map2WithIndex(m.t.t, (i,j,x,y) => 3 * x + 5 * y + i * 2 + j + 1).toLocalMatrix() ===
      9.0 * lm)
  }

  @Test
  def writeBlocksRDD() {
    // FIXME: duplicates matrix in RichIRMSuite
    val rows = 9L
    val cols = 6
    val data = Seq(
      (0L, Vectors.dense(0.0, 1.0, 2.0, 1.0, 3.0, 4.0)),
      (1L, Vectors.dense(3.0, 4.0, 5.0, 1.0, 1.0, 1.0)),
      (2L, Vectors.dense(9.0, 0.0, 2.0, 1.0, 1.0, 1.0)),
      (3L, Vectors.dense(9.0, 0.0, 1.0, 1.0, 1.0, 1.0)),
      (4L, Vectors.dense(9.0, 0.0, 1.0, 1.0, 1.0, 1.0)),
      (5L, Vectors.dense(9.0, 0.0, 1.0, 1.0, 1.0, 1.0)),
      (6L, Vectors.dense(1.0, 2.0, 3.0, 1.0, 1.0, 1.0)),
      (7L, Vectors.dense(4.0, 5.0, 6.0, 1.0, 1.0, 1.0)),
      (8L, Vectors.dense(7.0, 8.0, 9.0, 1.0, 1.0, 1.0))
    ).map(IndexedRow.tupled)
    `
    val indexedRows: RDD[IndexedRow] = sc.parallelize(data, numSlices = 4)

    val irm = new IndexedRowMatrix(indexedRows, rows, cols)
    
    val b = indexedRows.computeBoundaries()
    println(b.toIndexedSeq)
    
    assert(b.last == rows) // will catch missing rows
    
    def dependencies(blockRow: Int) = ???

    // val blockMat = irm.toHailBlockMatrix(2)
    }
  }
}
