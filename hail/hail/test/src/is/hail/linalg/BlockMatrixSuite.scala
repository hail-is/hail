package is.hail.linalg

import is.hail.HailSuite
import is.hail.collection.implicits.toRichIterable
import is.hail.expr.ir.{CompileAndEvaluate, TableLiteral}
import is.hail.expr.ir.defs.{GetField, TableCollect}
import is.hail.expr.ir.lowering.LoweringPipeline
import is.hail.linalg.BlockMatrix.ops._
import is.hail.linalg.implicits._
import is.hail.scalacheck._
import is.hail.types.virtual.{TFloat64, TInt64, TStruct}
import is.hail.utils._

import breeze.linalg.{*, diag, DenseMatrix, DenseVector => BDV}
import org.apache.spark.sql.Row
import org.scalacheck._
import org.scalacheck.Arbitrary._
import org.scalacheck.Gen.{size, _}
import org.scalatest
import org.scalatestplus.scalacheck.CheckerAsserting.assertingNatureOfAssertion
import org.scalatestplus.scalacheck.ScalaCheckDrivenPropertyChecks
import org.testng.annotations.Test

class BlockMatrixSuite extends HailSuite with ScalaCheckDrivenPropertyChecks {

  val interestingPosInt: Gen[Int] =
    oneOf(
      oneOf(1, 2, Int.MaxValue - 1, Int.MaxValue),
      choose(1, 100),
      posNum[Int],
    )

  val nonExtremeDouble: Gen[Double] =
    oneOf(
      oneOf(1e30, -1.0, -1e-30, 0.0, 1e-30, 1.0, 1e30),
      choose(-100.0, 100.0),
      choose(-1e150, 1e150),
    )

  def blockMatrixGen(
    blockSize: Gen[Int] = defaultBlockSize,
    dims: Gen[(Int, Int)] = defaultDims,
    element: Gen[Double] = arbitrary[Double],
  ): Gen[BlockMatrix] =
    for {
      (nRows, nCols) <- dims
      arrays <- containerOfN[Seq, Array[Double]](nRows, containerOfN[Array, Double](nCols, element))
      blockSize <- blockSize
      m = toBM(arrays, blockSize)
    } yield m

  val squareBlockMatrixGen: Gen[BlockMatrix] =
    blockMatrixGen(
      blockSize = interestingPosInt.map(n => math.sqrt(n.toDouble).toInt),
      dims = for {
        size <- size
        l <- interestingPosInt
        s = math.sqrt(math.min(l, math.max(size, 1)).toDouble).toInt
      } yield (s, s),
      element = arbitrary[Double],
    )

  val twoMultipliableBlockMatrices: Gen[(BlockMatrix, BlockMatrix)] =
    for {
      Array(nRows, innerDim, nCols) <- genNonEmptyNCubeOfVolumeAtMostSize(3)
      blockSize <- interestingPosInt
      if blockSize > 3 // 1 or 2 cause large numbers of partitions, leading to slow tests
      l <- blockMatrixGen(const(blockSize), const(nRows -> innerDim), nonExtremeDouble)
      r <- blockMatrixGen(const(blockSize), const(innerDim -> nCols), nonExtremeDouble)
    } yield (l, r)

  implicit val arbBlockMatrix: Arbitrary[BlockMatrix] =
    Arbitrary(blockMatrixGen())

  private val defaultBlockSize = choose(1, 1 << 6)
  private val defaultDims = genNonEmptySquareOfAreaAtMostSize

  // row major
  def toLM(nRows: Int, nCols: Int, data: Array[Double]): DenseMatrix[Double] =
    new DenseMatrix(nRows, nCols, data, 0, nCols, isTranspose = true)

  def toBM(nRows: Int, nCols: Int, data: Array[Double]): BlockMatrix =
    toBM(new DenseMatrix(nRows, nCols, data, 0, nRows, true))

  def toBM(rows: Seq[Array[Double]]): BlockMatrix =
    toBM(rows, BlockMatrix.defaultBlockSize)

  def toBM(rows: Seq[Array[Double]], blockSize: Int): BlockMatrix = {
    val n = rows.length
    val m = if (rows.isEmpty) 0 else rows.head.length
    BlockMatrix.fromBreezeMatrix(
      ctx,
      new DenseMatrix[Double](m, n, rows.flatten.toArray).t,
      blockSize,
    )
  }

  def toBM(lm: DenseMatrix[Double]): BlockMatrix =
    toBM(lm, BlockMatrix.defaultBlockSize)

  def toBM(lm: DenseMatrix[Double], blockSize: Int): BlockMatrix =
    BlockMatrix.fromBreezeMatrix(ctx, lm, blockSize)

  private val defaultRelTolerance = 1e-14

  private def assertDoubleMatrixNaNEqualsNaN(
    x: DenseMatrix[Double],
    y: DenseMatrix[Double],
    relTolerance: Double = defaultRelTolerance,
  ): Unit = {
    assert(
      x.rows == y.rows && x.cols == y.cols,
      s"dimension mismatch: ${x.rows} x ${x.cols} vs ${y.rows} x ${y.cols}",
    )
    scalatest.Inspectors.forAll(0 until x.cols) { j =>
      scalatest.Inspectors.forAll(0 until x.rows) { i =>
        assert(
          !(D_==(x(i, j) - y(i, j), relTolerance) && !(x(i, j).isNaN && y(i, j).isNaN)),
          s"x=${x.toString(1000, 1000)}\n" ++
            s"y=${y.toString(1000, 1000)}\n" ++
            s"inequality found at ($i, $j): ${x(i, j)} and ${y(i, j)}",
        )
      }
    }
  }

  @Test
  def pointwiseSubtractCorrect(): Unit = {
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
  def multiplyByLocalMatrix(): Unit = {
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

    assert(ll * lr === l.dot(ctx, lr).toBreezeMatrix())
  }

  @Test
  def randomMultiplyByLocalMatrix(): Unit =
    forAll(genMultipliableDenseMatrices) { case (ll, lr) =>
      val l = toBM(ll)
      assertDoubleMatrixNaNEqualsNaN(ll * lr, l.dot(ctx, lr).toBreezeMatrix())
    }

  @Test
  def multiplySameAsBreeze(): Unit = {
    forAll(genDenseMatrix(4, 4), genDenseMatrix(4, 4)) { (ll, lr) =>
      val l = toBM(ll, 2)
      val r = toBM(lr, 2)

      assertDoubleMatrixNaNEqualsNaN(l.dot(r).toBreezeMatrix(), ll * lr)
    }

    forAll(genDenseMatrix(9, 9), genDenseMatrix(9, 9)) { (ll, lr) =>
      val l = toBM(ll, 3)
      val r = toBM(lr, 3)

      assertDoubleMatrixNaNEqualsNaN(l.dot(r).toBreezeMatrix(), ll * lr)
    }

    forAll(genDenseMatrix(9, 9), genDenseMatrix(9, 9)) { (ll, lr) =>
      val l = toBM(ll, 2)
      val r = toBM(lr, 2)

      assertDoubleMatrixNaNEqualsNaN(l.dot(r).toBreezeMatrix(), ll * lr)
    }

    forAll(genDenseMatrix(2, 10), genDenseMatrix(10, 2)) { (ll, lr) =>
      val l = toBM(ll, 3)
      val r = toBM(lr, 3)

      assertDoubleMatrixNaNEqualsNaN(l.dot(r).toBreezeMatrix(), ll * lr)
    }

    forAll(genMultipliableDenseMatrices, interestingPosInt) { case ((ll, lr), blockSize) =>
      val l = toBM(ll, blockSize)
      val r = toBM(lr, blockSize)

      assertDoubleMatrixNaNEqualsNaN(l.dot(r).toBreezeMatrix(), ll * lr)
    }
  }

  @Test
  def multiplySameAsBreezeRandomized(): Unit = {
    forAll(twoMultipliableBlockMatrices) {
      case (l: BlockMatrix, r: BlockMatrix) =>
        val actual = l.dot(r).toBreezeMatrix()
        val expected = l.toBreezeMatrix() * r.toBreezeMatrix()
        assertDoubleMatrixNaNEqualsNaN(actual, expected)
    }
  }

  @Test
  def rowwiseMultiplication(): Unit = {
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

    assert(l.rowVectorMul(ctx, v).toBreezeMatrix() == result)
  }

  @Test
  def rowwiseMultiplicationRandom(): Unit = {
    val g = for {
      l <- blockMatrixGen()
      v <- containerOfN[Array, Double](l.nCols.toInt, arbitrary[Double])
    } yield (l, v)

    forAll(g) { case (l: BlockMatrix, v: Array[Double]) =>
      val actual = l.rowVectorMul(ctx, v).toBreezeMatrix()
      val repeatedR = (0 until l.nRows.toInt).flatMap(_ => v).toArray
      val repeatedRMatrix = new DenseMatrix(v.length, l.nRows.toInt, repeatedR).t
      val expected = l.toBreezeMatrix() *:* repeatedRMatrix

      assertDoubleMatrixNaNEqualsNaN(actual, expected)
    }
  }

  @Test
  def colwiseMultiplication(): Unit = {
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

    assert(l.colVectorMul(ctx, v).toBreezeMatrix() == result)
  }

  @Test
  def colwiseMultiplicationRandom(): Unit = {
    val g = for {
      l <- blockMatrixGen()
      v <- containerOfN[Array, Double](l.nRows.toInt, arbitrary[Double])
    } yield (l, v)

    forAll(g) { case (l: BlockMatrix, v: Array[Double]) =>
      val actual = l.colVectorMul(ctx, v).toBreezeMatrix()
      val repeatedR = (0 until l.nCols.toInt).flatMap(_ => v).toArray
      val repeatedRMatrix = new DenseMatrix(v.length, l.nCols.toInt, repeatedR)
      val expected = l.toBreezeMatrix() *:* repeatedRMatrix
      assertDoubleMatrixNaNEqualsNaN(actual, expected)
    }
  }

  @Test
  def colwiseAddition(): Unit = {
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

    assert(l.colVectorAdd(ctx, v).toBreezeMatrix() == result)
  }

  @Test
  def rowwiseAddition(): Unit = {
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

    assert(l.rowVectorAdd(ctx, v).toBreezeMatrix() == result)
  }

  @Test
  def diagonalTestTiny(): Unit = {
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
  def diagonalTestRandomized(): Unit =
    forAll(squareBlockMatrixGen) { (m: BlockMatrix) =>
      val lm = m.toBreezeMatrix()
      val diagonalLength = math.min(lm.rows, lm.cols)
      val diagonal = Array.tabulate(diagonalLength)(i => lm(i, i))

      assert(
        m.diagonal().toSeq == diagonal.toSeq,
        s"lm: $lm\n${m.diagonal().toSeq} != ${diagonal.toSeq}",
      )
    }

  @Test
  def fromLocalTest(): Unit =
    forAll(arbitrary[DenseMatrix[Double]].flatMap { m =>
      Gen.zip(Gen.const(m), Gen.choose(math.sqrt(m.rows.toDouble).toInt, m.rows + 16))
    }) { case (lm, blockSize) =>
      assert(lm === BlockMatrix.fromBreezeMatrix(ctx, lm, blockSize).toBreezeMatrix())
    }

  @Test
  def readWriteIdentityTrivial(): Unit = {
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
    assert(m.toBreezeMatrix() == BlockMatrix.read(ctx, fname).toBreezeMatrix())

    val fname2 = ctx.createTmpPath("test2")
    m.write(ctx, fname2, forceRowMajor = true)
    assert(m.toBreezeMatrix() == BlockMatrix.read(ctx, fname2).toBreezeMatrix())
  }

  @Test
  def readWriteIdentityTrivialTransposed(): Unit = {
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
    assert(m.T.toBreezeMatrix() == BlockMatrix.read(ctx, fname).toBreezeMatrix())

    val fname2 = ctx.createTmpPath("test2")
    m.T.write(ctx, fname2, forceRowMajor = true)
    assert(m.T.toBreezeMatrix() == BlockMatrix.read(ctx, fname2).toBreezeMatrix())
  }

  @Test
  def readWriteIdentityRandom(): Unit = {
    forAll(blockMatrixGen()) { (m: BlockMatrix) =>
      val fname = ctx.createTmpPath("test")
      m.write(ctx, fname)
      assertDoubleMatrixNaNEqualsNaN(
        m.toBreezeMatrix(),
        BlockMatrix.read(ctx, fname).toBreezeMatrix(),
      )
    }
  }

  @Test
  def transpose(): Unit = {
    forAll(blockMatrixGen()) { (m: BlockMatrix) =>
      val transposed = m.toBreezeMatrix().t
      assert(transposed.rows == m.nCols)
      assert(transposed.cols == m.nRows)
      assert(transposed === m.T.toBreezeMatrix())
    }
  }

  @Test
  def doubleTransposeIsIdentity(): Unit = {
    forAll(blockMatrixGen(element = nonExtremeDouble)) { (m: BlockMatrix) =>
      val mt = m.T.cache()
      val mtt = m.T.T.cache()
      assert(mtt.nRows == m.nRows)
      assert(mtt.nCols == m.nCols)
      assertDoubleMatrixNaNEqualsNaN(mtt.toBreezeMatrix(), m.toBreezeMatrix())
      assertDoubleMatrixNaNEqualsNaN(mt.dot(mtt).toBreezeMatrix(), mt.dot(m).toBreezeMatrix())
    }
  }

  @Test
  def cachedOpsOK(): Unit =
    forAll(twoMultipliableBlockMatrices) {
      case (l: BlockMatrix, r: BlockMatrix) =>
        l.cache()
        r.cache()

        val actual = l.dot(r).toBreezeMatrix()
        val expected = l.toBreezeMatrix() * r.toBreezeMatrix()
        assertDoubleMatrixNaNEqualsNaN(actual, expected)
        assertDoubleMatrixNaNEqualsNaN(l.T.cache().T.toBreezeMatrix(), l.toBreezeMatrix())
    }

  @Test
  def toIRMToHBMIdentity(): Unit =
    forAll(blockMatrixGen()) { (m: BlockMatrix) =>
      val roundtrip = m.toIndexedRowMatrix().toHailBlockMatrix(m.blockSize)

      val roundtriplm = roundtrip.toBreezeMatrix()
      val lm = m.toBreezeMatrix()

      assert(roundtriplm == lm)
    }

  @Test
  def map2RespectsTransposition(): Unit = {
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
  def map4RespectsTransposition(): Unit = {
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
  def mapRespectsTransposition(): Unit = {
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
  def mapWithIndexRespectsTransposition(): Unit = {
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
  def map2WithIndexRespectsTransposition(): Unit = {
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
  def filterCols(): Unit = {
    val lm = new DenseMatrix[Double](9, 10, (0 until 90).map(_.toDouble).toArray)

    scalatest.Inspectors.forAll(Seq(1, 2, 3, 5, 10, 11)) { blockSize =>
      val bm = BlockMatrix.fromBreezeMatrix(ctx, lm, blockSize)
      scalatest.Inspectors.forAll {
        Seq(
          Array(0),
          Array(1),
          Array(9),
          Array(0, 3, 4, 5, 7),
          Array(1, 4, 5, 7, 8, 9),
          Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        )
      } { keep =>
        val filteredViaBlock = bm.filterCols(keep.map(_.toLong)).toBreezeMatrix()
        val filteredViaBreeze = lm(::, keep.toFastSeq).copy
        assert(filteredViaBlock === filteredViaBreeze)
      }
    }
  }

  @Test
  def filterColsTranspose(): Unit = {
    val lm = new DenseMatrix[Double](9, 10, (0 until 90).map(_.toDouble).toArray)
    val lmt = lm.t

    scalatest.Inspectors.forAll(Seq(2, 3)) { blockSize =>
      val bm = BlockMatrix.fromBreezeMatrix(ctx, lm, blockSize).transpose()
      scalatest.Inspectors.forAll {
        Seq(
          Array(0),
          Array(1, 4, 5, 7, 8),
          Array(0, 1, 2, 3, 4, 5, 6, 7, 8),
        )
      } { keep =>
        val filteredViaBlock = bm.filterCols(keep.map(_.toLong)).toBreezeMatrix()
        val filteredViaBreeze = lmt(::, keep.toFastSeq).copy
        assert(filteredViaBlock === filteredViaBreeze)
      }
    }
  }

  @Test
  def filterRows(): Unit = {
    val lm = new DenseMatrix[Double](9, 10, (0 until 90).map(_.toDouble).toArray)

    scalatest.Inspectors.forAll(Seq(2, 3)) { blockSize =>
      val bm = BlockMatrix.fromBreezeMatrix(ctx, lm, blockSize)
      scalatest.Inspectors.forAll {
        Seq(
          Array(0),
          Array(1, 4, 5, 7, 8),
          Array(0, 1, 2, 3, 4, 5, 6, 7, 8),
        )
      } { keep =>
        val filteredViaBlock = bm.filterRows(keep.map(_.toLong)).toBreezeMatrix()
        val filteredViaBreeze = lm(keep.toFastSeq, ::).copy

        assert(filteredViaBlock === filteredViaBreeze)
      }
    }
  }

  @Test
  def filterSymmetric(): Unit = {
    val lm = new DenseMatrix[Double](10, 10, (0 until 100).map(_.toDouble).toArray)

    scalatest.Inspectors.forAll(Seq(1, 2, 3, 5, 10, 11)) { blockSize =>
      val bm = BlockMatrix.fromBreezeMatrix(ctx, lm, blockSize)
      scalatest.Inspectors.forAll {
        Seq(
          Array(0),
          Array(1),
          Array(9),
          Array(0, 3, 4, 5, 7),
          Array(1, 4, 5, 7, 8, 9),
          Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        )
      } { keep =>
        val filteredViaBlock = bm.filter(keep.map(_.toLong), keep.map(_.toLong)).toBreezeMatrix()
        val filteredViaBreeze = lm(keep.toFastSeq, keep.toFastSeq).copy

        assert(filteredViaBlock === filteredViaBreeze)
      }
    }
  }

  @Test
  def filter(): Unit = {
    val lm = new DenseMatrix[Double](9, 10, (0 until 90).map(_.toDouble).toArray)
    scalatest.Inspectors.forAll(Seq(1, 2, 3, 5, 10, 11)) { blockSize =>
      val bm = BlockMatrix.fromBreezeMatrix(ctx, lm, blockSize)
      scalatest.Inspectors.forAll {
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
        } yield (keepRows, keepCols)
      } { case (keepRows, keepCols) =>
        val filteredViaBlock =
          bm.filter(keepRows.map(_.toLong), keepCols.map(_.toLong)).toBreezeMatrix()
        val filteredViaBreeze = lm(keepRows.toFastSeq, keepCols.toFastSeq).copy

        assert(filteredViaBlock === filteredViaBreeze)
      }
    }
  }

  @Test
  def writeLocalAsBlockTest(): Unit = {
    val lm = new DenseMatrix[Double](10, 10, (0 until 100).map(_.toDouble).toArray)

    scalatest.Inspectors.forAll(Seq(1, 2, 3, 5, 10, 11)) { blockSize =>
      val fname = ctx.createTmpPath("test")
      lm.writeBlockMatrix(fs, fname, blockSize)
      assert(lm === BlockMatrix.read(ctx, fname).toBreezeMatrix())
    }
  }

  @Test
  def randomTest(): Unit = {
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
    val lm = new DenseMatrix[Double](9, 10, data)
    val expectedEntries = data.map(x => ((x % 9).toLong, (x / 9).toLong, x)).toSet
    val expectedSignature = TStruct("i" -> TInt64, "j" -> TInt64, "entry" -> TFloat64)

    scalatest.Inspectors.forAll(Seq(1, 4, 10)) { blockSize =>
      val entriesLiteral = TableLiteral(toBM(lm, blockSize).entriesTable(ctx), theHailClassLoader)
      assert(entriesLiteral.typ.rowType == expectedSignature)
      val rows =
        CompileAndEvaluate[IndexedSeq[Row]](
          ctx,
          GetField(TableCollect(entriesLiteral), "rows"),
          lower = LoweringPipeline.relationalLowerer,
        )
      val entries = rows.map(row => (row.get(0), row.get(1), row.get(2))).toSet
      // block size affects order of rows in table, but sets will be the same
      assert(entries === expectedEntries)
    }
  }

  @Test
  def testEntriesTableWhenKeepingOnlySomeBlocks(): Unit = {
    val data = (0 until 50).map(_.toDouble).toArray
    val lm = new DenseMatrix[Double](5, 10, data)
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
      lower = LoweringPipeline.relationalLowerer,
    )
    val expected = rows
      .sortBy(r => (r.get(0).asInstanceOf[Long], r.get(1).asInstanceOf[Long]))
      .map(r => r.get(2).asInstanceOf[Double])

    assert(expected sameElements Array[Double](0, 5, 20, 25, 1, 6, 21, 26, 2, 7, 3, 8))
  }

  @Test
  def testPowSqrt(): Unit = {
    val lm = new DenseMatrix[Double](2, 3, Array(0.0, 1.0, 4.0, 9.0, 16.0, 25.0))
    val bm = BlockMatrix.fromBreezeMatrix(ctx, lm, blockSize = 2)
    val expected = new DenseMatrix[Double](2, 3, Array(0.0, 1.0, 2.0, 3.0, 4.0, 5.0))

    assertMatrixEqualityDouble(bm.pow(0.0).toBreezeMatrix(), DenseMatrix.fill(2, 3)(1.0))
    assertMatrixEqualityDouble(bm.pow(0.5).toBreezeMatrix(), expected)
    assertMatrixEqualityDouble(bm.sqrt().toBreezeMatrix(), expected)
  }

  def filteredEquals(bm1: BlockMatrix, bm2: BlockMatrix): Boolean =
    bm1.blocks.collect() sameElements bm2.blocks.collect()

  @Test
  def testSparseFilterEdges(): Unit = {
    val lm = new DenseMatrix[Double](12, 12, (0 to 143).map(_.toDouble).toArray)
    val bm = toBM(lm, blockSize = 5)

    val onlyEight = bm.filterBlocks(Array(8)) // Bottom right corner block
    val onlyEightRowEleven = onlyEight.filterRows(Array(11)).toBreezeMatrix()
    val onlyEightColEleven = onlyEight.filterCols(Array(11)).toBreezeMatrix()
    val onlyEightCornerFour = onlyEight.filter(Array(10, 11), Array(10, 11)).toBreezeMatrix()

    assert(onlyEightRowEleven.toArray sameElements Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 131,
      143).map(_.toDouble))
    assert(onlyEightColEleven.toArray sameElements Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 142,
      143).map(_.toDouble))
    assert(onlyEightCornerFour == new DenseMatrix[Double](2, 2, Array(130.0, 131.0, 142.0, 143.0)))
  }

  @Test
  def testSparseTransposeMaybeBlocks(): Unit = {
    val lm = new DenseMatrix[Double](9, 12, (0 to 107).map(_.toDouble).toArray)
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
    val bm = BlockMatrix.fill(nRows.toLong, nCols.toLong, 2, 1)
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
  def testFilterBlocks(): Unit = {
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

    scalatest.Inspectors.forAll(keepArray) { keep =>
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
  def testSparseBlockMatrixIO(): Unit = {
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

    val lm_zero = DenseMatrix.zeros[Double](2, 2)

    def filterBlocks(keep: Array[Int]): DenseMatrix[Double] = {
      val flm = lm.copy
      (0 to 3).diff(keep).foreach { i =>
        val r = 2 * (i % 2)
        val c = 2 * (i / 2)
        flm(r to r + 1, c to c + 1) := lm_zero
      }
      flm
    }

    // test toBlockMatrix, toIndexedRowMatrix, toRowMatrix, read/write identity
    scalatest.Inspectors.forAll(keepArray) { keep =>
      val fbm = bm.filterBlocks(keep)
      val flm = filterBlocks(keep)

      assert(fbm.toBreezeMatrix() === flm)

      assert(flm === fbm.toIndexedRowMatrix().toHailBlockMatrix().toBreezeMatrix())

      val fname = ctx.createTmpPath("test")
      fbm.write(ctx, fname, forceRowMajor = true)

      assert(RowMatrix.readBlockMatrix(ctx, fname, 3).toBreezeMatrix() === flm)

      assert(filteredEquals(fbm, BlockMatrix.read(ctx, fname)))
    }
  }

  @Test
  def testSparseBlockMatrixMathAndFilter(): Unit = {
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

    val lm_zero = DenseMatrix.zeros[Double](2, 2)

    def filterBlocks(keep: Array[Int]): DenseMatrix[Double] = {
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
    scalatest.Inspectors.forAll(keepArray) { keep =>
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

      assert(filteredEquals(fbm.rowVectorMul(ctx, v), bm.rowVectorMul(ctx, v).filterBlocks(keep)))
      assert(filteredEquals(fbm.rowVectorDiv(ctx, v), bm.rowVectorDiv(ctx, v).filterBlocks(keep)))

      assert(filteredEquals(fbm.colVectorMul(ctx, v), bm.colVectorMul(ctx, v).filterBlocks(keep)))
      assert(filteredEquals(fbm.colVectorDiv(ctx, v), bm.colVectorDiv(ctx, v).filterBlocks(keep)))

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

      assert(fbm.rowVectorAdd(ctx, v).toBreezeMatrix() === flm(*, ::) + BDV(v))
      assert(fbm.rowVectorSub(ctx, v).toBreezeMatrix() === flm(*, ::) - BDV(v))
      assert(fbm.reverseRowVectorSub(ctx, v).toBreezeMatrix() === -(flm(*, ::) - BDV(v)))

      assert(fbm.colVectorAdd(ctx, v).toBreezeMatrix() === flm(::, *) + BDV(v))
      assert(fbm.colVectorSub(ctx, v).toBreezeMatrix() === flm(::, *) - BDV(v))
      assert(fbm.reverseColVectorSub(ctx, v).toBreezeMatrix() === -(flm(::, *) - BDV(v)))

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

    interceptFatal(notSupported)(bm0 / bm0)
    interceptFatal(notSupported)(bm0.reverseRowVectorDiv(ctx, v))
    interceptFatal(notSupported)(bm0.reverseColVectorDiv(ctx, v))
    interceptFatal(notSupported)(1 / bm0)

    interceptFatal(notSupported)(bm0.rowVectorDiv(ctx, v0))
    interceptFatal(notSupported)(bm0.colVectorDiv(ctx, v0))
    interceptFatal("multiplication by scalar NaN")(bm0 * Double.NaN)
    interceptFatal("division by scalar 0.0")(bm0 / 0)

    interceptFatal(notSupported)(bm0.pow(-1))
  }

  @Test
  def testRealizeBlocks(): Unit = {
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

    val lm_zero = DenseMatrix.zeros[Double](2, 2)

    def filterBlocks(keep: Array[Int]): DenseMatrix[Double] = {
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

    scalatest.Inspectors.forAll(keepArray) { keep =>
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
