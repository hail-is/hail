package is.hail.distributedmatrix

import breeze.linalg.{DenseMatrix => BDM}
import is.hail.{SparkSuite, TestUtils}
import is.hail.check.Arbitrary._
import is.hail.check.Prop._
import is.hail.check.Gen._
import is.hail.check._
import is.hail.distributedmatrix.BlockMatrix.ops._
import is.hail.utils._
import is.hail.utils.richUtils.RichDenseMatrixDouble
import org.testng.annotations.Test

class BlockMatrixSuite extends SparkSuite {

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

    BlockMatrix.from(sc, new BDM[Double](m, n, rows.flatten.toArray).t, blockSize)
  }

  def toBM(lm: BDM[Double]): BlockMatrix =
    toBM(lm, BlockMatrix.defaultBlockSize)

  def toBM(lm: BDM[Double], blockSize: Int): BlockMatrix =
    BlockMatrix.from(sc, lm, blockSize)
 
  private val defaultBlockSize = choose(1, 1 << 6)
  private val defaultDims = nonEmptySquareOfAreaAtMostSize
  private val defaultElement = arbitrary[Double]

  def blockMatrixGen(
    blockSize: Gen[Int] = defaultBlockSize,
    dims: Gen[(Int, Int)] = defaultDims,
    element: Gen[Double] = defaultElement
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
    element = element
  )

  def twoMultipliableBlockMatrices(element: Gen[Double] = defaultElement): Gen[(BlockMatrix, BlockMatrix)] = for {
    Array(nRows, innerDim, nCols) <- nonEmptyNCubeOfVolumeAtMostSize(3)
    blockSize <- interestingPosInt.map(math.pow(_, 1.0 / 3.0).toInt)
    l <- blockMatrixGen(const(blockSize), const(nRows -> innerDim), element)
    r <- blockMatrixGen(const(blockSize), const(innerDim -> nCols), element)
  } yield (l, r)

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
    forAll(twoMultipliableDenseMatrices[Double]()) { case (ll, lr) =>
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
      v <- buildableOfN[Array](l.nCols.toInt, arbitrary[Double])
    } yield (l, v)

    forAll(g) { case (l: BlockMatrix, v: Array[Double]) =>
      val actual = (l --* v).toLocalMatrix()
      val repeatedR = (0 until l.nRows.toInt).flatMap(_ => v).toArray
      val repeatedRMatrix = new BDM(v.length, l.nRows.toInt, repeatedR).t
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
      v <- buildableOfN[Array](l.nRows.toInt, arbitrary[Double])
    } yield (l, v)

    forAll(g) { case (l: BlockMatrix, v: Array[Double]) =>
      val actual = (l :* v).toLocalMatrix()
      val repeatedR = (0 until l.nCols.toInt).flatMap(_ => v).toArray
      val repeatedRMatrix = new BDM(v.length, l.nCols.toInt, repeatedR)
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
    forAll(denseMatrix[Double]()) { lm =>
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
    
    val fname2 = tmpDir.createTempFile("test2")
    m.write(fname2, forceRowMajor = true)
    assert(m.toLocalMatrix() == BlockMatrix.read(hc, fname2).toLocalMatrix())
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
    
    val fname2 = tmpDir.createTempFile("test2")
    m.t.write(fname2, forceRowMajor = true)
    assert(m.t.toLocalMatrix() == BlockMatrix.read(hc, fname2).toLocalMatrix())    
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
      assert(transposed.rows == m.nCols)
      assert(transposed.cols == m.nRows)
      assert(transposed === m.t.toLocalMatrix())
      true
    }.check()
  }

  @Test
  def doubleTransposeIsIdentity() {
    forAll(blockMatrixGen(element = nonExtremeDouble)) { (m: BlockMatrix) =>
      val mt = m.t.cache()
      val mtt = m.t.t.cache()
      assert(mtt.nRows == m.nRows)
      assert(mtt.nCols == m.nCols)
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
  def readWriteBDM() {
    val lm = BDM.rand[Double](256, 129) // 33024 doubles
    val fname = tmpDir.createTempFile("test")

    lm.write(hc, fname)    
    val lm2 = RichDenseMatrixDouble.read(hc, fname)
    
    assert(lm === lm2)
  }
  
  @Test
  def filterCols() {
    val lm = new BDM[Double](9, 10, (0 until 90).map(_.toDouble).toArray)

    for { blockSize <- Seq(1, 2, 3, 5, 10, 11)
    } {
      val bm = BlockMatrix.from(sc, lm, blockSize)      
      for { keep <- Seq(
        Array(0),
        Array(1),
        Array(9),
        Array(0, 3, 4, 5, 7),
        Array(1, 4, 5, 7, 8, 9),
        Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
      } {
        val filteredViaBlock = bm.filterCols(keep.map(_.toLong)).toLocalMatrix()
        val filteredViaBreeze = lm(::, keep.toIndexedSeq).copy
        
        assert(filteredViaBlock === filteredViaBreeze)
      }
    }
  }
  
  @Test
  def filterColsTranspose() {
    val lm = new BDM[Double](9, 10, (0 until 90).map(_.toDouble).toArray)
    val lmt = lm.t

    for { blockSize <- Seq(2, 3)
    } {
      val bm = BlockMatrix.from(sc, lm, blockSize).transpose()   
      for { keep <- Seq(
        Array(0),
        Array(1, 4, 5, 7, 8),
        Array(0, 1, 2, 3, 4, 5, 6, 7, 8))
      } {
        val filteredViaBlock = bm.filterCols(keep.map(_.toLong)).toLocalMatrix()
        val filteredViaBreeze = lmt(::, keep.toIndexedSeq).copy
        
        assert(filteredViaBlock === filteredViaBreeze)
      }
    }
  }
  
  @Test
  def filterRows() {
    val lm = new BDM[Double](9, 10, (0 until 90).map(_.toDouble).toArray)
    
    for { blockSize <- Seq(2, 3)
    } {
      val bm = BlockMatrix.from(sc, lm, blockSize) 
      for { keep <- Seq(
        Array(0),
        Array(1, 4, 5, 7, 8),
        Array(0, 1, 2, 3, 4, 5, 6, 7, 8))
      } {
        val filteredViaBlock = bm.filterRows(keep.map(_.toLong)).toLocalMatrix()
        val filteredViaBreeze = lm(keep.toIndexedSeq, ::).copy
        
        assert(filteredViaBlock === filteredViaBreeze)
      }
    }
  }
  
  @Test
  def filterSymmetric() {
    val lm = new BDM[Double](10, 10, (0 until 100).map(_.toDouble).toArray)

    for { blockSize <- Seq(1, 2, 3, 5, 10, 11)
    } {
      val bm = BlockMatrix.from(sc, lm, blockSize)      
      for { keep <- Seq(
        Array(0),
        Array(1),
        Array(9),
        Array(0, 3, 4, 5, 7),
        Array(1, 4, 5, 7, 8, 9),
        Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
      } {
        val filteredViaBlock = bm.filter(keep.map(_.toLong), keep.map(_.toLong)).toLocalMatrix()
        val filteredViaBreeze = lm(keep.toIndexedSeq, keep.toIndexedSeq).copy
        
        assert(filteredViaBlock === filteredViaBreeze)
      }
    }
  }
  
  @Test
  def filter() {
    val lm = new BDM[Double](9, 10, (0 until 90).map(_.toDouble).toArray)

    for { blockSize <- Seq(1, 2, 3, 5, 10, 11)
    } {
      val bm = BlockMatrix.from(sc, lm, blockSize)      
      for {
        keepRows <- Seq(
          Array(1),
          Array(0, 3, 4, 5, 7),
          Array(0, 1, 2, 3, 4, 5, 6, 7, 8))  
        keepCols <- Seq(
          Array(2),
          Array(1, 4, 5, 7, 8, 9),
          Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))  
      } {
        val filteredViaBlock = bm.filter(keepRows.map(_.toLong), keepCols.map(_.toLong)).toLocalMatrix()
        val filteredViaBreeze = lm(keepRows.toIndexedSeq, keepCols.toIndexedSeq).copy
        
        assert(filteredViaBlock === filteredViaBreeze)
      }
    }
  }
  
  @Test
  def writeLocalAsBlockTest() {
    val lm = new BDM[Double](10, 10, (0 until 100).map(_.toDouble).toArray)
    
    for { blockSize <- Seq(1, 2, 3, 5, 10, 11) } {
      val fname = tmpDir.createTempFile("test")
      lm.writeBlockMatrix(hc, fname, blockSize)
      assert(lm === BlockMatrix.read(hc, fname).toLocalMatrix())
    }
  }
  
  @Test
  def randomTest() {
    var lm1 = BlockMatrix.random(hc, 5, 10, 2, seed = 1).toLocalMatrix()
    var lm2 = BlockMatrix.random(hc, 5, 10, 2, seed = 1).toLocalMatrix()
    var lm3 = BlockMatrix.random(hc, 5, 10, 2, seed = 2).toLocalMatrix()
    
    assert(lm1 === lm2)
    assert(lm1 !== lm3)
    assert(lm1.data.forall(x => x >= 0 && x <= 1))
    
    lm1 = BlockMatrix.random(hc, 5, 10, 2, seed = 1, gaussian = true).toLocalMatrix()
    lm2 = BlockMatrix.random(hc, 5, 10, 2, seed = 1, gaussian = true).toLocalMatrix()
    lm3 = BlockMatrix.random(hc, 5, 10, 2, seed = 2, gaussian = true).toLocalMatrix()
    
    assert(lm1 === lm2)
    assert(lm1 !== lm3)
  }
  
  @Test
  def writeSubsetTest() {
    val lm = new BDM[Double](9, 10, (0 until 90).map(_.toDouble).toArray)

    def prefix(blockSize: Double): String = "/parts/part-" + (if (blockSize <= 3) "0" else "")
    
    for {blockSize <- Seq(2, 4, 8)} {
      val bm = BlockMatrix.from(sc, lm, blockSize)
      val pre = prefix(blockSize)

      val allFile = tmpDir.createTempFile("all")
      val someFile = tmpDir.createTempFile("some")
      
      bm.write(allFile)
      bm.write(someFile, optKeep = Some(Array(1, 3)))

      assert(!hc.hadoopConf.exists(someFile + pre + "0"))
      assert( hc.hadoopConf.exists( allFile + pre + "0"))

      assert(!hc.hadoopConf.exists(someFile + pre + "2"))
      assert( hc.hadoopConf.exists( allFile + pre + "2"))

      assert(TestUtils.fileByteEquality(someFile + pre + "1", allFile + pre + "1"))
      assert(TestUtils.fileByteEquality(someFile + pre + "3", allFile + pre + "3"))
      assert(TestUtils.fileByteEquality(someFile + "/metadata.json", allFile + "/metadata.json"))
    }
  }
}
