package is.hail.linalg

import is.hail.{HailSuite, TestUtils}
import is.hail.check._
import is.hail.check.Arbitrary._
import is.hail.check.Gen._
import is.hail.check.Prop._
import is.hail.expr.ir.{CompileAndEvaluate, GetField, TableCollect, TableLiteral}
import is.hail.linalg.BlockMatrix.ops._
import is.hail.types.virtual.{TFloat64, TInt64, TStruct}
import is.hail.utils._

import scala.language.implicitConversions

import breeze.linalg.{*, diag, DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class BlockMatrixSuite extends HailSuite {

  // row major
  def toLM(nRows: Int, nCols: Int, data: Array[Double]): BDM[Double] =
    new BDM(nRows, nCols, data, 0, nCols, isTranspose = true)

  def toBM(nRows: Int, nCols: Int, data: Array[Double]): BlockMatrix =
    toBM(new BDM(nRows, nCols, data, 0, nRows, true))

  def toBM(rows: Seq[Array[Double]]): BlockMatrix =
    toBM(rows, BlockMatrix.defaultBlockSize)

  def toBM(rows: Seq[Array[Double]], blockSize: Int): BlockMatrix = {
    val n = rows.length
    val m = if (n == 0) 0 else rows(0).length

    BlockMatrix.fromBreezeMatrix(new BDM[Double](m, n, rows.flatten.toArray).t, blockSize)
  }

  def toBM(lm: BDM[Double]): BlockMatrix =
    toBM(lm, BlockMatrix.defaultBlockSize)

  def toBM(lm: BDM[Double], blockSize: Int): BlockMatrix =
    BlockMatrix.fromBreezeMatrix(lm, blockSize)

  private val defaultBlockSize = choose(1, 1 << 6)
  private val defaultDims = nonEmptySquareOfAreaAtMostSize
  private val defaultElement = arbitrary[Double]

  def blockMatrixGen(
    blockSize: Gen[Int] = defaultBlockSize,
    dims: Gen[(Int, Int)] = defaultDims,
    element: Gen[Double] = defaultElement,
  ): Gen[BlockMatrix] = for {
    blockSize <- blockSize
    (nRows, nCols) <- dims
    arrays <- buildableOfN[Seq](nRows, buildableOfN[Array](nCols, element))
    m = toBM(arrays, blockSize)
  } yield m

  def squareBlockMatrixGen(
    element: Gen[Double] = defaultElement
  ): Gen[BlockMatrix] = blockMatrixGen(
    blockSize = interestingPosInt.map(math.sqrt(_).toInt),
    dims = for {
      size <- size
      l <- interestingPosInt
      s = math.sqrt(math.min(l, size)).toInt
    } yield (s, s),
    element = element,
  )

  def twoMultipliableBlockMatrices(element: Gen[Double] = defaultElement)
    : Gen[(BlockMatrix, BlockMatrix)] = for {
    Array(nRows, innerDim, nCols) <- nonEmptyNCubeOfVolumeAtMostSize(3)
    blockSize <-
      interestingPosInt.filter(
        _ > 3
      ) // 1 or 2 cause large numbers of partitions, leading to slow tests
    l <- blockMatrixGen(const(blockSize), const(nRows -> innerDim), element)
    r <- blockMatrixGen(const(blockSize), const(innerDim -> nCols), element)
  } yield (l, r)

  implicit val arbitraryBlockMatrix =
    Arbitrary(blockMatrixGen())

  private val defaultRelTolerance = 1e-14

  private def sameDoubleMatrixNaNEqualsNaN(
    x: BDM[Double],
    y: BDM[Double],
    relTolerance: Double = defaultRelTolerance,
  ): Boolean =
    findDoubleMatrixMismatchNaNEqualsNaN(x, y, relTolerance) match {
      case Some(_) => false
      case None => true
    }

  private def findDoubleMatrixMismatchNaNEqualsNaN(
    x: BDM[Double],
    y: BDM[Double],
    relTolerance: Double = defaultRelTolerance,
  ): Option[(Int, Int)] = {
    assert(
      x.rows == y.rows && x.cols == y.cols,
      s"dimension mismatch: ${x.rows} x ${x.cols} vs ${y.rows} x ${y.cols}",
    )
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
    val m = toBM(
      4,
      4,
      Array[Double](
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16),
    )

    val expected = toLM(
      4,
      4,
      Array[Double](
        0, -3, -6, -9,
        3, 0, -3, -6,
        6, 3, 0, -3,
        9, 6, 3, 0),
    )

    val actual = (m - m.T).toBreezeMatrix()
    assert(actual == expected)
  }

  @Test
  def multiplyByLocalMatrix() {
    val ll = toLM(
      4,
      4,
      Array[Double](
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16),
    )
    val l = toBM(ll)

    val lr = toLM(
      4,
      1,
      Array[Double](
        1,
        2,
        3,
        4,
      ),
    )

    assert(ll * lr === l.dot(lr).toBreezeMatrix())
  }

  @Test
  def randomMultiplyByLocalMatrix() {
    forAll(twoMultipliableDenseMatrices[Double]()) { case (ll, lr) =>
      val l = toBM(ll)
      sameDoubleMatrixNaNEqualsNaN(ll * lr, l.dot(lr).toBreezeMatrix())
    }.check()
  }

  @Test
  def multiplySameAsBreeze() {
    def randomLm(n: Int, m: Int) = denseMatrix[Double](n, m)

    forAll(randomLm(4, 4), randomLm(4, 4)) { (ll, lr) =>
      val l = toBM(ll, 2)
      val r = toBM(lr, 2)

      sameDoubleMatrixNaNEqualsNaN(l.dot(r).toBreezeMatrix(), ll * lr)
    }.check()

    forAll(randomLm(9, 9), randomLm(9, 9)) { (ll, lr) =>
      val l = toBM(ll, 3)
      val r = toBM(lr, 3)

      sameDoubleMatrixNaNEqualsNaN(l.dot(r).toBreezeMatrix(), ll * lr)
    }.check()

    forAll(randomLm(9, 9), randomLm(9, 9)) { (ll, lr) =>
      val l = toBM(ll, 2)
      val r = toBM(lr, 2)

      sameDoubleMatrixNaNEqualsNaN(l.dot(r).toBreezeMatrix(), ll * lr)
    }.check()

    forAll(randomLm(2, 10), randomLm(10, 2)) { (ll, lr) =>
      val l = toBM(ll, 3)
      val r = toBM(lr, 3)

      sameDoubleMatrixNaNEqualsNaN(l.dot(r).toBreezeMatrix(), ll * lr)
    }.check()

    forAll(twoMultipliableDenseMatrices[Double](), interestingPosInt) {
      case ((ll, lr), blockSize) =>
        val l = toBM(ll, blockSize)
        val r = toBM(lr, blockSize)

        sameDoubleMatrixNaNEqualsNaN(l.dot(r).toBreezeMatrix(), ll * lr)
    }.check()
  }

  @Test
  def multiplySameAsBreezeRandomized() {
    forAll(twoMultipliableBlockMatrices(nonExtremeDouble)) {
      case (l: BlockMatrix, r: BlockMatrix) =>
        val actual = l.dot(r).toBreezeMatrix()
        val expected = l.toBreezeMatrix() * r.toBreezeMatrix()

        findDoubleMatrixMismatchNaNEqualsNaN(actual, expected) match {
          case Some((i, j)) =>
            println(s"blockSize: ${l.blockSize}")
            println(s"${l.toBreezeMatrix()}")
            println(s"${r.toBreezeMatrix()}")
            println(s"row: ${l.toBreezeMatrix()(i, ::)}")
            println(s"col: ${r.toBreezeMatrix()(::, j)}")
            false
          case None =>
            true
        }
    }.check()
  }

  @Test
  def rowwiseMultiplication() {
    val l = toBM(
      4,
      4,
      Array[Double](
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16),
    )

    val v = Array[Double](1, 2, 3, 4)

    val result = toLM(
      4,
      4,
      Array[Double](
        1, 4, 9, 16,
        5, 12, 21, 32,
        9, 20, 33, 48,
        13, 28, 45, 64),
    )

    assert(l.rowVectorMul(v).toBreezeMatrix() == result)
  }

  @Test
  def rowwiseMultiplicationRandom() {
    val g = for {
      l <- blockMatrixGen()
      v <- buildableOfN[Array](l.nCols.toInt, arbitrary[Double])
    } yield (l, v)

    forAll(g) { case (l: BlockMatrix, v: Array[Double]) =>
      val actual = l.rowVectorMul(v).toBreezeMatrix()
      val repeatedR = (0 until l.nRows.toInt).flatMap(_ => v).toArray
      val repeatedRMatrix = new BDM(v.length, l.nRows.toInt, repeatedR).t
      val expected = l.toBreezeMatrix() *:* repeatedRMatrix

      sameDoubleMatrixNaNEqualsNaN(actual, expected)
    }.check()
  }

  @Test
  def colwiseMultiplication() {
    val l = toBM(
      4,
      4,
      Array[Double](
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16),
    )

    val v = Array[Double](1, 2, 3, 4)

    val result = toLM(
      4,
      4,
      Array[Double](
        1, 2, 3, 4,
        10, 12, 14, 16,
        27, 30, 33, 36,
        52, 56, 60, 64),
    )

    assert(l.colVectorMul(v).toBreezeMatrix() == result)
  }

  @Test
  def colwiseMultiplicationRandom() {
    val g = for {
      l <- blockMatrixGen()
      v <- buildableOfN[Array](l.nRows.toInt, arbitrary[Double])
    } yield (l, v)

    forAll(g) { case (l: BlockMatrix, v: Array[Double]) =>
      val actual = l.colVectorMul(v).toBreezeMatrix()
      val repeatedR = (0 until l.nCols.toInt).flatMap(_ => v).toArray
      val repeatedRMatrix = new BDM(v.length, l.nCols.toInt, repeatedR)
      val expected = l.toBreezeMatrix() *:* repeatedRMatrix

      if (sameDoubleMatrixNaNEqualsNaN(actual, expected))
        true
      else {
        println(s"${l.toBreezeMatrix().toArray.toSeq}\n*\n${v.toSeq}")
        false
      }
    }.check()
  }

  @Test
  def colwiseAddition() {
    val l = toBM(
      4,
      4,
      Array[Double](
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16),
    )

    val v = Array[Double](1, 2, 3, 4)

    val result = toLM(
      4,
      4,
      Array[Double](
        2, 3, 4, 5,
        7, 8, 9, 10,
        12, 13, 14, 15,
        17, 18, 19, 20),
    )

    assert(l.colVectorAdd(v).toBreezeMatrix() == result)
  }

  @Test
  def rowwiseAddition() {
    val l = toBM(
      4,
      4,
      Array[Double](
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16),
    )

    val v = Array[Double](1, 2, 3, 4)

    val result = toLM(
      4,
      4,
      Array[Double](
        2, 4, 6, 8,
        6, 8, 10, 12,
        10, 12, 14, 16,
        14, 16, 18, 20),
    )

    assert(l.rowVectorAdd(v).toBreezeMatrix() == result)
  }

  @Test
  def diagonalTestTiny() {
    val lm = toLM(
      3,
      4,
      Array[Double](
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12),
    )

    val m = toBM(lm, blockSize = 2)

    assert(m.diagonal().toSeq == Seq(1, 6, 11))
    assert(m.T.diagonal().toSeq == Seq(1, 6, 11))
    assert(m.dot(m.T).diagonal().toSeq == Seq(30, 174, 446))
  }

  @Test
  def diagonalTestRandomized() {
    forAll(squareBlockMatrixGen()) { (m: BlockMatrix) =>
      val lm = m.toBreezeMatrix()
      val diagonalLength = math.min(lm.rows, lm.cols)
      val diagonal = Array.tabulate(diagonalLength)(i => lm(i, i))

      if (m.diagonal().toSeq == diagonal.toSeq)
        true
      else {
        println(s"lm: $lm")
        println(s"${m.diagonal().toSeq} != ${diagonal.toSeq}")
        false
      }
    }.check()
  }

  @Test
  def fromLocalTest() {
    forAll(denseMatrix[Double]().flatMap { m =>
      Gen.zip(Gen.const(m), Gen.choose(math.sqrt(m.rows).toInt, m.rows + 16))
    }) { case (lm, blockSize) =>
      assert(lm === BlockMatrix.fromBreezeMatrix(lm, blockSize).toBreezeMatrix())
      true
    }.check()
  }

  @Test
  def readWriteIdentityTrivial() {
    val m = toBM(
      4,
      4,
      Array[Double](
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16),
    )

    val fname = ctx.createTmpPath("test")
    m.write(ctx, fname)
    assert(m.toBreezeMatrix() == BlockMatrix.read(fs, fname).toBreezeMatrix())

    val fname2 = ctx.createTmpPath("test2")
    m.write(ctx, fname2, forceRowMajor = true)
    assert(m.toBreezeMatrix() == BlockMatrix.read(fs, fname2).toBreezeMatrix())
  }

  @Test
  def readWriteIdentityTrivialTransposed() {
    val m = toBM(
      4,
      4,
      Array[Double](
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16),
    )

    val fname = ctx.createTmpPath("test")
    m.T.write(ctx, fname)
    assert(m.T.toBreezeMatrix() == BlockMatrix.read(fs, fname).toBreezeMatrix())

    val fname2 = ctx.createTmpPath("test2")
    m.T.write(ctx, fname2, forceRowMajor = true)
    assert(m.T.toBreezeMatrix() == BlockMatrix.read(fs, fname2).toBreezeMatrix())
  }

  @Test
  def readWriteIdentityRandom() {
    forAll(blockMatrixGen()) { (m: BlockMatrix) =>
      val fname = ctx.createTmpPath("test")
      m.write(ctx, fname)
      assert(sameDoubleMatrixNaNEqualsNaN(
        m.toBreezeMatrix(),
        BlockMatrix.read(fs, fname).toBreezeMatrix(),
      ))
      true
    }.check()
  }

  @Test
  def transpose() {
    forAll(blockMatrixGen()) { (m: BlockMatrix) =>
      val transposed = m.toBreezeMatrix().t
      assert(transposed.rows == m.nCols)
      assert(transposed.cols == m.nRows)
      assert(transposed === m.T.toBreezeMatrix())
      true
    }.check()
  }

  @Test
  def doubleTransposeIsIdentity() {
    forAll(blockMatrixGen(element = nonExtremeDouble)) { (m: BlockMatrix) =>
      val mt = m.T.cache()
      val mtt = m.T.T.cache()
      assert(mtt.nRows == m.nRows)
      assert(mtt.nCols == m.nCols)
      assert(sameDoubleMatrixNaNEqualsNaN(mtt.toBreezeMatrix(), m.toBreezeMatrix()))
      assert(sameDoubleMatrixNaNEqualsNaN(mt.dot(mtt).toBreezeMatrix(), mt.dot(m).toBreezeMatrix()))
      true
    }.check()
  }

  @Test
  def cachedOpsOK() {
    forAll(twoMultipliableBlockMatrices(nonExtremeDouble)) {
      case (l: BlockMatrix, r: BlockMatrix) =>
        l.cache()
        r.cache()

        val actual = l.dot(r).toBreezeMatrix()
        val expected = l.toBreezeMatrix() * r.toBreezeMatrix()

        if (!sameDoubleMatrixNaNEqualsNaN(actual, expected)) {
          println(s"${l.toBreezeMatrix()}")
          println(s"${r.toBreezeMatrix()}")
          assert(false)
        }

        if (!sameDoubleMatrixNaNEqualsNaN(l.T.cache().T.toBreezeMatrix(), l.toBreezeMatrix())) {
          println(s"${l.T.cache().T.toBreezeMatrix()}")
          println(s"${l.toBreezeMatrix()}")
          assert(false)
        }

        true
    }.check()
  }

  @Test
  def toIRMToHBMIdentity() {
    forAll(blockMatrixGen()) { (m: BlockMatrix) =>
      val roundtrip = m.toIndexedRowMatrix().toHailBlockMatrix(m.blockSize)

      val roundtriplm = roundtrip.toBreezeMatrix()
      val lm = m.toBreezeMatrix()

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
    val lm = toLM(
      4,
      2,
      Array[Double](
        1, 2,
        3, 4,
        5, 6,
        7, 8),
    )
    val lmt = toLM(
      2,
      4,
      Array[Double](
        1, 3, 5, 7,
        2, 4, 6, 8),
    )

    val m = toBM(lm)
    val mt = toBM(lmt)

    assert(m.map2(mt.T, _ + _).toBreezeMatrix() === lm + lm)
    assert(
      mt.T.map2(m, _ + _).toBreezeMatrix() === lm + lm,
      s"${mt.toBreezeMatrix()}\n${mt.T.toBreezeMatrix()}\n${m.toBreezeMatrix()}",
    )
  }

  @Test
  def map4RespectsTransposition() {
    val lm = toLM(
      4,
      2,
      Array[Double](
        1, 2,
        3, 4,
        5, 6,
        7, 8),
    )
    val lmt = toLM(
      2,
      4,
      Array[Double](
        1, 3, 5, 7,
        2, 4, 6, 8),
    )

    val m = toBM(lm)
    val mt = toBM(lmt)

    assert(m.map4(m, mt.T, mt.T.T.T, _ + _ + _ + _).toBreezeMatrix() === lm + lm + lm + lm)
    assert(mt.map4(mt, m.T, mt.T.T, _ + _ + _ + _).toBreezeMatrix() === lm.t + lm.t + lm.t + lm.t)
  }

  @Test
  def mapRespectsTransposition() {
    val lm = toLM(
      4,
      2,
      Array[Double](
        1, 2,
        3, 4,
        5, 6,
        7, 8),
    )
    val lmt = toLM(
      2,
      4,
      Array[Double](
        1, 3, 5, 7,
        2, 4, 6, 8),
    )

    val m = toBM(lm)
    val mt = toBM(lmt)

    assert(m.T.map(_ * 4).toBreezeMatrix() === lm.t.map(_ * 4))
    assert(m.T.T.map(_ * 4).toBreezeMatrix() === lm.map(_ * 4))
    assert(mt.T.map(_ * 4).toBreezeMatrix() === lm.map(_ * 4))
  }

  @Test
  def mapWithIndexRespectsTransposition() {
    val lm = toLM(
      4,
      2,
      Array[Double](
        1, 2,
        3, 4,
        5, 6,
        7, 8),
    )
    val lmt = toLM(
      2,
      4,
      Array[Double](
        1, 3, 5, 7,
        2, 4, 6, 8),
    )

    val m = toBM(lm)
    val mt = toBM(lmt)

    assert(m.T.mapWithIndex((_, _, x) => x * 4).toBreezeMatrix() === lm.t.map(_ * 4))
    assert(m.T.T.mapWithIndex((_, _, x) => x * 4).toBreezeMatrix() === lm.map(_ * 4))
    assert(mt.T.mapWithIndex((_, _, x) => x * 4).toBreezeMatrix() === lm.map(_ * 4))

    assert(m.T.mapWithIndex((i, j, x) => i * 10 + j + x).toBreezeMatrix() ===
      mt.mapWithIndex((i, j, x) => i * 10 + j + x).toBreezeMatrix())
    assert(m.T.mapWithIndex((i, j, x) => x + j * 2 + i + 1).toBreezeMatrix() ===
      lm.t + lm.t)
    assert(mt.mapWithIndex((i, j, x) => x + j * 2 + i + 1).toBreezeMatrix() ===
      lm.t + lm.t)
    assert(mt.T.mapWithIndex((i, j, x) => x + i * 2 + j + 1).toBreezeMatrix() ===
      lm + lm)
  }

  @Test
  def map2WithIndexRespectsTransposition() {
    val lm = toLM(
      4,
      2,
      Array[Double](
        1, 2,
        3, 4,
        5, 6,
        7, 8),
    )
    val lmt = toLM(
      2,
      4,
      Array[Double](
        1, 3, 5, 7,
        2, 4, 6, 8),
    )

    val m = toBM(lm)
    val mt = toBM(lmt)

    assert(m.map2WithIndex(mt.T, (_, _, x, y) => x + y).toBreezeMatrix() === lm + lm)
    assert(mt.map2WithIndex(m.T, (_, _, x, y) => x + y).toBreezeMatrix() === lm.t + lm.t)
    assert(mt.T.map2WithIndex(m, (_, _, x, y) => x + y).toBreezeMatrix() === lm + lm)
    assert(m.T.T.map2WithIndex(mt.T, (_, _, x, y) => x + y).toBreezeMatrix() === lm + lm)

    assert(m.T.map2WithIndex(mt, (i, j, x, y) => i * 10 + j + x + y).toBreezeMatrix() ===
      mt.map2WithIndex(m.T, (i, j, x, y) => i * 10 + j + x + y).toBreezeMatrix())
    assert(m.T.map2WithIndex(m.T, (i, j, x, y) => i * 10 + j + x + y).toBreezeMatrix() ===
      mt.map2WithIndex(mt, (i, j, x, y) => i * 10 + j + x + y).toBreezeMatrix())
    assert(m.T.map2WithIndex(mt, (i, j, x, y) => x + 2 * y + j * 2 + i + 1).toBreezeMatrix() ===
      4.0 * lm.t)
    assert(mt.map2WithIndex(m.T, (i, j, x, y) => x + 2 * y + j * 2 + i + 1).toBreezeMatrix() ===
      4.0 * lm.t)
    assert(mt.T.map2WithIndex(
      m.T.T,
      (i, j, x, y) => 3 * x + 5 * y + i * 2 + j + 1,
    ).toBreezeMatrix() ===
      9.0 * lm)
  }

  @Test
  def filterCols() {
    val lm = new BDM[Double](9, 10, (0 until 90).map(_.toDouble).toArray)

    for { blockSize <- Seq(1, 2, 3, 5, 10, 11) } {
      val bm = BlockMatrix.fromBreezeMatrix(lm, blockSize)
      for {
        keep <- Seq(
          Array(0),
          Array(1),
          Array(9),
          Array(0, 3, 4, 5, 7),
          Array(1, 4, 5, 7, 8, 9),
          Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        )
      } {
        val filteredViaBlock = bm.filterCols(keep.map(_.toLong)).toBreezeMatrix()
        val filteredViaBreeze = lm(::, keep.toFastSeq).copy

        assert(filteredViaBlock === filteredViaBreeze)
      }
    }
  }

  @Test
  def filterColsTranspose() {
    val lm = new BDM[Double](9, 10, (0 until 90).map(_.toDouble).toArray)
    val lmt = lm.t

    for { blockSize <- Seq(2, 3) } {
      val bm = BlockMatrix.fromBreezeMatrix(lm, blockSize).transpose()
      for {
        keep <- Seq(
          Array(0),
          Array(1, 4, 5, 7, 8),
          Array(0, 1, 2, 3, 4, 5, 6, 7, 8),
        )
      } {
        val filteredViaBlock = bm.filterCols(keep.map(_.toLong)).toBreezeMatrix()
        val filteredViaBreeze = lmt(::, keep.toFastSeq).copy

        assert(filteredViaBlock === filteredViaBreeze)
      }
    }
  }

  @Test
  def filterRows() {
    val lm = new BDM[Double](9, 10, (0 until 90).map(_.toDouble).toArray)

    for { blockSize <- Seq(2, 3) } {
      val bm = BlockMatrix.fromBreezeMatrix(lm, blockSize)
      for {
        keep <- Seq(
          Array(0),
          Array(1, 4, 5, 7, 8),
          Array(0, 1, 2, 3, 4, 5, 6, 7, 8),
        )
      } {
        val filteredViaBlock = bm.filterRows(keep.map(_.toLong)).toBreezeMatrix()
        val filteredViaBreeze = lm(keep.toFastSeq, ::).copy

        assert(filteredViaBlock === filteredViaBreeze)
      }
    }
  }

  @Test
  def filterSymmetric() {
    val lm = new BDM[Double](10, 10, (0 until 100).map(_.toDouble).toArray)

    for { blockSize <- Seq(1, 2, 3, 5, 10, 11) } {
      val bm = BlockMatrix.fromBreezeMatrix(lm, blockSize)
      for {
        keep <- Seq(
          Array(0),
          Array(1),
          Array(9),
          Array(0, 3, 4, 5, 7),
          Array(1, 4, 5, 7, 8, 9),
          Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        )
      } {
        val filteredViaBlock = bm.filter(keep.map(_.toLong), keep.map(_.toLong)).toBreezeMatrix()
        val filteredViaBreeze = lm(keep.toFastSeq, keep.toFastSeq).copy

        assert(filteredViaBlock === filteredViaBreeze)
      }
    }
  }

  @Test
  def filter() {
    val lm = new BDM[Double](9, 10, (0 until 90).map(_.toDouble).toArray)

    for { blockSize <- Seq(1, 2, 3, 5, 10, 11) } {
      val bm = BlockMatrix.fromBreezeMatrix(lm, blockSize)
      for {
        keepRows <- Seq(
          Array(1),
          Array(0, 3, 4, 5, 7),
          Array(0, 1, 2, 3, 4, 5, 6, 7, 8),
        )
        keepCols <- Seq(
          Array(2),
          Array(1, 4, 5, 7, 8, 9),
          Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        )
      } {
        val filteredViaBlock =
          bm.filter(keepRows.map(_.toLong), keepCols.map(_.toLong)).toBreezeMatrix()
        val filteredViaBreeze = lm(keepRows.toFastSeq, keepCols.toFastSeq).copy

        assert(filteredViaBlock === filteredViaBreeze)
      }
    }
  }

  @Test
  def writeLocalAsBlockTest() {
    val lm = new BDM[Double](10, 10, (0 until 100).map(_.toDouble).toArray)

    for { blockSize <- Seq(1, 2, 3, 5, 10, 11) } {
      val fname = ctx.createTmpPath("test")
      lm.writeBlockMatrix(fs, fname, blockSize)
      assert(lm === BlockMatrix.read(fs, fname).toBreezeMatrix())
    }
  }

  @Test
  def randomTest() {
    var lm1 =
      BlockMatrix.random(5, 10, 2, staticUID = 1, nonce = 1, gaussian = false).toBreezeMatrix()
    var lm2 =
      BlockMatrix.random(5, 10, 2, staticUID = 1, nonce = 1, gaussian = false).toBreezeMatrix()
    var lm3 =
      BlockMatrix.random(5, 10, 2, staticUID = 2, nonce = 1, gaussian = false).toBreezeMatrix()
    var lm4 =
      BlockMatrix.random(5, 10, 2, staticUID = 1, nonce = 2, gaussian = false).toBreezeMatrix()

    println(lm1)
    assert(lm1 === lm2)
    assert(lm1 !== lm3)
    assert(lm1 !== lm4)
    assert(lm3 !== lm4)
    assert(lm1.data.forall(x => x >= 0 && x <= 1))

    lm1 = BlockMatrix.random(5, 10, 2, staticUID = 1, nonce = 1, gaussian = true).toBreezeMatrix()
    lm2 = BlockMatrix.random(5, 10, 2, staticUID = 1, nonce = 1, gaussian = true).toBreezeMatrix()
    lm3 = BlockMatrix.random(5, 10, 2, staticUID = 2, nonce = 1, gaussian = true).toBreezeMatrix()
    lm4 = BlockMatrix.random(5, 10, 2, staticUID = 1, nonce = 2, gaussian = true).toBreezeMatrix()

    assert(lm1 === lm2)
    assert(lm1 !== lm3)
    assert(lm1 !== lm4)
    assert(lm3 !== lm4)
  }

  @Test
  def testEntriesTable(): Unit = {
    val data = (0 until 90).map(_.toDouble).toArray
    val lm = new BDM[Double](9, 10, data)
    val expectedEntries = data.map(x => ((x % 9).toLong, (x / 9).toLong, x)).toSet
    val expectedSignature = TStruct("i" -> TInt64, "j" -> TInt64, "entry" -> TFloat64)

    for { blockSize <- Seq(1, 4, 10) } {
      val entriesLiteral = TableLiteral(toBM(lm, blockSize).entriesTable(ctx), theHailClassLoader)
      assert(entriesLiteral.typ.rowType == expectedSignature)
      val rows =
        CompileAndEvaluate[IndexedSeq[Row]](ctx, GetField(TableCollect(entriesLiteral), "rows"))
      val entries = rows.map(row => (row.get(0), row.get(1), row.get(2))).toSet
      // block size affects order of rows in table, but sets will be the same
      assert(entries === expectedEntries)
    }
  }

  @Test
  def testEntriesTableWhenKeepingOnlySomeBlocks(): Unit = {
    val data = (0 until 50).map(_.toDouble).toArray
    val lm = new BDM[Double](5, 10, data)
    val bm = toBM(lm, blockSize = 2)

    val rows = CompileAndEvaluate[IndexedSeq[Row]](
      ctx,
      GetField(
        TableCollect(
          TableLiteral(
            bm.filterBlocks(Array(0, 1, 6)).entriesTable(ctx),
            theHailClassLoader,
          )
        ),
        "rows",
      ),
    )
    val expected = rows
      .sortBy(r => (r.get(0).asInstanceOf[Long], r.get(1).asInstanceOf[Long]))
      .map(r => r.get(2).asInstanceOf[Double])

    assert(expected sameElements Array[Double](0, 5, 20, 25, 1, 6, 21, 26, 2, 7, 3, 8))
  }

  @Test
  def testPowSqrt(): Unit = {
    val lm = new BDM[Double](2, 3, Array(0.0, 1.0, 4.0, 9.0, 16.0, 25.0))
    val bm = BlockMatrix.fromBreezeMatrix(lm, blockSize = 2)
    val expected = new BDM[Double](2, 3, Array(0.0, 1.0, 2.0, 3.0, 4.0, 5.0))

    TestUtils.assertMatrixEqualityDouble(bm.pow(0.0).toBreezeMatrix(), BDM.fill(2, 3)(1.0))
    TestUtils.assertMatrixEqualityDouble(bm.pow(0.5).toBreezeMatrix(), expected)
    TestUtils.assertMatrixEqualityDouble(bm.sqrt().toBreezeMatrix(), expected)
  }

  def filteredEquals(bm1: BlockMatrix, bm2: BlockMatrix): Boolean =
    bm1.blocks.collect() sameElements bm2.blocks.collect()

  @Test
  def testSparseFilterEdges(): Unit = {
    val lm = new BDM[Double](12, 12, (0 to 143).map(_.toDouble).toArray)
    val bm = toBM(lm, blockSize = 5)

    val onlyEight = bm.filterBlocks(Array(8)) // Bottom right corner block
    val onlyEightRowEleven = onlyEight.filterRows(Array(11)).toBreezeMatrix()
    val onlyEightColEleven = onlyEight.filterCols(Array(11)).toBreezeMatrix()
    val onlyEightCornerFour = onlyEight.filter(Array(10, 11), Array(10, 11)).toBreezeMatrix()

    assert(onlyEightRowEleven.toArray sameElements Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 131,
      143).map(_.toDouble))
    assert(onlyEightColEleven.toArray sameElements Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 142,
      143).map(_.toDouble))
    assert(onlyEightCornerFour == new BDM[Double](2, 2, Array(130.0, 131.0, 142.0, 143.0)))
  }

  @Test
  def testSparseTransposeMaybeBlocks(): Unit = {
    val lm = new BDM[Double](9, 12, (0 to 107).map(_.toDouble).toArray)
    val bm = toBM(lm, blockSize = 3)
    val sparse = bm.filterBand(0, 0, true)
    assert(sparse.transpose().gp.partitionIndexToBlockIndex.get.toIndexedSeq == IndexedSeq(
      0,
      5,
      10,
    ))
  }

  @Test
  def filterRowsRectangleSum(): Unit = {
    val nRows = 10
    val nCols = 50
    val bm = BlockMatrix.fill(nRows, nCols, 2, 1)
    val banded = bm.filterBand(0, 0, false)
    val rowFilt = banded.filterRows((0L until nRows.toLong by 2L).toArray)
    val summed = rowFilt.rowSum().toBreezeMatrix().toArray
    val expected =
      Array.tabulate(nRows)(x => if (x % 2 == 0) 2.0 else 0) ++ Array.tabulate(nCols - nRows)(x =>
        0.0
      )
    assert(summed sameElements expected)
  }

  @Test
  def testFilterBlocks() {
    val lm = toLM(
      4,
      4,
      Array(
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16),
    )

    val bm = toBM(lm, blockSize = 2)

    val keepArray = Array(
      Array.empty[Int],
      Array(0),
      Array(1, 3),
      Array(2, 3),
      Array(1, 2, 3),
      Array(0, 1, 2, 3),
    )

    val localBlocks =
      Array(lm(0 to 1, 0 to 1), lm(2 to 3, 0 to 1), lm(0 to 1, 2 to 3), lm(2 to 3, 2 to 3))

    for { keep <- keepArray } {
      val fbm = bm.filterBlocks(keep)

      assert(fbm.blocks.count() == keep.length)
      assert(fbm.blocks.collect().forall { case ((i, j), block) =>
        block == localBlocks(fbm.gp.coordinatesBlock(i, j))
      })
    }

    // test multiple block filters
    val bm13 = bm.filterBlocks(Array(1, 3)).cache()

    assert(filteredEquals(bm13, bm13.filterBlocks(Array(1, 3))))
    assert(filteredEquals(bm13, bm.filterBlocks(Array(1, 2, 3)).filterBlocks(Array(0, 1, 3))))
    assert(filteredEquals(bm13, bm13.filterBlocks(Array(0, 1, 2, 3))))
    assert(filteredEquals(
      bm.filterBlocks(Array(1)),
      bm.filterBlocks(Array(1, 2, 3)).filterBlocks(Array(0, 1, 2)).filterBlocks(Array(0, 1, 3)),
    ))
  }

  @Test
  def testSparseBlockMatrixIO() {
    val lm = toLM(
      4,
      4,
      Array(
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16),
    )

    val bm = toBM(lm, blockSize = 2)

    val keepArray = Array(
      Array.empty[Int],
      Array(0),
      Array(1, 3),
      Array(2, 3),
      Array(1, 2, 3),
      Array(0, 1, 2, 3),
    )

    val lm_zero = BDM.zeros[Double](2, 2)

    def filterBlocks(keep: Array[Int]): BDM[Double] = {
      val flm = lm.copy
      (0 to 3).diff(keep).foreach { i =>
        val r = 2 * (i % 2)
        val c = 2 * (i / 2)
        flm(r to r + 1, c to c + 1) := lm_zero
      }
      flm
    }

    // test toBlockMatrix, toIndexedRowMatrix, toRowMatrix, read/write identity
    for { keep <- keepArray } {
      val fbm = bm.filterBlocks(keep)
      val flm = filterBlocks(keep)

      assert(fbm.toBreezeMatrix() === flm)

      assert(flm === fbm.toIndexedRowMatrix().toHailBlockMatrix().toBreezeMatrix())

      val fname = ctx.createTmpPath("test")
      fbm.write(ctx, fname, forceRowMajor = true)

      assert(RowMatrix.readBlockMatrix(fs, fname, 3).toBreezeMatrix() === flm)

      assert(filteredEquals(fbm, BlockMatrix.read(fs, fname)))
    }
  }

  @Test
  def testSparseBlockMatrixMathAndFilter() {
    val lm = toLM(
      4,
      4,
      Array(
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16),
    )

    val bm = toBM(lm, blockSize = 2)

    val keepArray = Array(
      Array.empty[Int],
      Array(0),
      Array(1, 3),
      Array(2, 3),
      Array(1, 2, 3),
      Array(0, 1, 2, 3),
    )

    val lm_zero = BDM.zeros[Double](2, 2)

    def filterBlocks(keep: Array[Int]): BDM[Double] = {
      val flm = lm.copy
      (0 to 3).diff(keep).foreach { i =>
        val r = 2 * (i % 2)
        val c = 2 * (i / 2)
        flm(r to r + 1, c to c + 1) := lm_zero
      }
      flm
    }

    val transposeBI = Array(0 -> 0, 1 -> 2, 2 -> 1, 3 -> 3).toMap

    val v = Array(1.0, 2.0, 3.0, 4.0)

    // test transpose, diagonal, math ops, filter ops
    for { keep <- keepArray } {
      println(s"Test says keep block: ${keep.toIndexedSeq}")
      val fbm = bm.filterBlocks(keep)
      val flm = filterBlocks(keep)

      assert(filteredEquals(fbm.transpose().transpose(), fbm))

      assert(filteredEquals(
        fbm.transpose(),
        bm.transpose().filterBlocks(keep.map(transposeBI).sorted),
      ))

      assert(fbm.diagonal() sameElements diag(fbm.toBreezeMatrix()).toArray)

      assert(filteredEquals(+fbm, +bm.filterBlocks(keep)))
      assert(filteredEquals(-fbm, -bm.filterBlocks(keep)))

      assert(filteredEquals(fbm + fbm, (bm + bm).filterBlocks(keep)))
      assert(filteredEquals(fbm - fbm, (bm - bm).filterBlocks(keep)))
      assert(filteredEquals(fbm * fbm, (bm * bm).filterBlocks(keep)))

      assert(filteredEquals(fbm.rowVectorMul(v), bm.rowVectorMul(v).filterBlocks(keep)))
      assert(filteredEquals(fbm.rowVectorDiv(v), bm.rowVectorDiv(v).filterBlocks(keep)))

      assert(filteredEquals(fbm.colVectorMul(v), bm.colVectorMul(v).filterBlocks(keep)))
      assert(filteredEquals(fbm.colVectorDiv(v), bm.colVectorDiv(v).filterBlocks(keep)))

      assert(filteredEquals(fbm * 2, (bm * 2).filterBlocks(keep)))
      assert(filteredEquals(fbm / 2, (bm / 2).filterBlocks(keep)))

      assert(filteredEquals(fbm.sqrt(), bm.sqrt().filterBlocks(keep)))
      assert(filteredEquals(fbm.pow(3), bm.pow(3).filterBlocks(keep)))

      assert(fbm.dot(fbm).toBreezeMatrix() === flm * flm)

      // densifying ops
      assert((fbm + 2).toBreezeMatrix() === flm + 2.0)
      assert((2 + fbm).toBreezeMatrix() === flm + 2.0)
      assert((fbm - 2).toBreezeMatrix() === flm - 2.0)
      assert((2 - fbm).toBreezeMatrix() === 2.0 - flm)

      assert(fbm.rowVectorAdd(v).toBreezeMatrix() === flm(*, ::) + BDV(v))
      assert(fbm.rowVectorSub(v).toBreezeMatrix() === flm(*, ::) - BDV(v))
      assert(fbm.reverseRowVectorSub(v).toBreezeMatrix() === -(flm(*, ::) - BDV(v)))

      assert(fbm.colVectorAdd(v).toBreezeMatrix() === flm(::, *) + BDV(v))
      assert(fbm.colVectorSub(v).toBreezeMatrix() === flm(::, *) - BDV(v))
      assert(fbm.reverseColVectorSub(v).toBreezeMatrix() === -(flm(::, *) - BDV(v)))

      // filter ops
      assert(fbm.filterRows(Array(1, 2)).toBreezeMatrix() === flm(1 to 2, ::))
      assert(fbm.filterCols(Array(1, 2)).toBreezeMatrix() === flm(::, 1 to 2))
      assert(fbm.filter(Array(1, 2), Array(1, 2)).toBreezeMatrix() === flm(1 to 2, 1 to 2))
    }

    val bm0 = bm.filterBlocks(Array(0))
    val bm13 = bm.filterBlocks(Array(1, 3))
    val bm23 = bm.filterBlocks(Array(2, 3))
    val bm123 = bm.filterBlocks(Array(1, 2, 3))

    val lm0 = filterBlocks(Array(0))
    val lm13 = filterBlocks(Array(1, 3))
    val lm23 = filterBlocks(Array(2, 3))
    val lm123 = filterBlocks(Array(1, 2, 3))

    // test +/- with mismatched blocks
    assert(filteredEquals(bm0 + bm13, bm.filterBlocks(Array(0, 1, 3))))

    assert((bm0 + bm).toBreezeMatrix() === lm0 + lm)
    assert((bm + bm0).toBreezeMatrix() === lm + lm0)
    assert(
      (bm0 + 2.0 * bm13 + 3.0 * bm23 + 5.0 * bm123).toBreezeMatrix() ===
        lm0 + 2.0 * lm13 + 3.0 * lm23 + 5.0 * lm123
    )
    assert(
      (bm123 + 2.0 * bm13 + 3.0 * bm23 + 5.0 * bm0).toBreezeMatrix() ===
        lm123 + 2.0 * lm13 + 3.0 * lm23 + 5.0 * lm0
    )

    assert((bm0 - bm).toBreezeMatrix() === lm0 - lm)
    assert((bm - bm0).toBreezeMatrix() === lm - lm0)
    assert(
      (bm0 - 2.0 * bm13 - 3.0 * bm23 - 5.0 * bm123).toBreezeMatrix() ===
        lm0 - 2.0 * lm13 - 3.0 * lm23 - 5.0 * lm123
    )
    assert(
      (bm123 - 2.0 * bm13 - 3.0 * bm23 - 5.0 * bm0).toBreezeMatrix() ===
        lm123 - 2.0 * lm13 - 3.0 * lm23 - 5.0 * lm0
    )

    // test * with mismatched blocks
    assert(filteredEquals(bm0 * bm13, bm.filterBlocks(Array.empty[Int])))
    assert(filteredEquals(bm13 * bm23, (bm * bm).filterBlocks(Array(3))))
    assert(filteredEquals(bm13 * bm, (bm * bm).filterBlocks(Array(1, 3))))
    assert(filteredEquals(bm * bm13, (bm * bm).filterBlocks(Array(1, 3))))

    // test unsupported ops
    val notSupported: String = "not supported for block-sparse matrices"

    val v0 = Array(0.0, Double.NaN, Double.PositiveInfinity, Double.NegativeInfinity)

    TestUtils.interceptFatal(notSupported)(bm0 / bm0)
    TestUtils.interceptFatal(notSupported)(bm0.reverseRowVectorDiv(v))
    TestUtils.interceptFatal(notSupported)(bm0.reverseColVectorDiv(v))
    TestUtils.interceptFatal(notSupported)(1 / bm0)

    TestUtils.interceptFatal(notSupported)(bm0.rowVectorDiv(v0))
    TestUtils.interceptFatal(notSupported)(bm0.colVectorDiv(v0))
    TestUtils.interceptFatal("multiplication by scalar NaN")(bm0 * Double.NaN)
    TestUtils.interceptFatal("division by scalar 0.0")(bm0 / 0)

    TestUtils.interceptFatal(notSupported)(bm0.pow(-1))
  }

  @Test
  def testRealizeBlocks() {
    val lm = toLM(
      4,
      4,
      Array(
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16),
    )

    val bm = toBM(lm, blockSize = 2)

    val keepArray = Array(
      Array.empty[Int],
      Array(0),
      Array(1, 3),
      Array(2, 3),
      Array(1, 2, 3),
      Array(0, 1, 2, 3),
    )

    val lm_zero = BDM.zeros[Double](2, 2)

    def filterBlocks(keep: Array[Int]): BDM[Double] = {
      val flm = lm.copy
      (0 to 3).diff(keep).foreach { i =>
        val r = 2 * (i % 2)
        val c = 2 * (i / 2)
        flm(r to r + 1, c to c + 1) := lm_zero
      }
      flm
    }

    assert(filteredEquals(bm.densify(), bm))
    assert(filteredEquals(bm.realizeBlocks(None), bm))

    for { keep <- keepArray } {
      val fbm = bm.filterBlocks(keep)
      val flm = filterBlocks(keep)

      assert(filteredEquals(fbm.densify(), toBM(flm, blockSize = 2)))
      assert(filteredEquals(fbm.realizeBlocks(Some(keep)), fbm))

      val bis = (keep ++ Array(0, 2)).distinct.sorted
      assert(filteredEquals(
        fbm.realizeBlocks(Some(Array(0, 2))),
        toBM(flm, blockSize = 2).filterBlocks(bis),
      ))
    }
  }
}
