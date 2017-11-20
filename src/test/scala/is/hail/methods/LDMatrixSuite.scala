package is.hail.methods

import breeze.linalg.{DenseMatrix, convert, norm}
import breeze.stats.mean
import is.hail.utils._
import is.hail.variant.{Genotype, Locus, Variant, VariantDataset}
import is.hail.{SparkSuite, TestUtils, stats}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._
import org.testng.Assert
import org.testng.annotations.Test

class LDMatrixSuite extends SparkSuite {
  val m = 50
  val n = 60
  val seed: Int = scala.util.Random.nextInt()
  lazy val vds: VariantDataset = hc.baldingNicholsModel(1, n, m, seed = seed)
  lazy val distLDMatrix: LDMatrix = LDMatrix.apply(vds, Some(false))
  lazy val localLDMatrix: LDMatrix = vds.ldMatrix(forceLocal = true)

  /**
    * Tests that entries in LDMatrix agree with those computed by LDPrune.computeR. Also tests
    * that index i in the matrix corresponds to index i in the array.
    */
  @Test def testEntries() {
    val nSamples = vds.nSamples

    val variantsTable = vds.typedRDD[Locus, Variant, Genotype].map { case (v, (_, gs)) =>
      (v, LDPrune.toBitPackedVector(gs.hardCallIterator, nSamples))}.collectAsMap()

    val indexToBPV = distLDMatrix.variants.map(v => variantsTable(v).get)
    val distLDMatrixLocal = distLDMatrix.matrix.toHailBlockMatrix().toLocalMatrix()
    val localLDMatrixLocal = localLDMatrix.matrix.toHailBlockMatrix().toLocalMatrix()
    val numVariants = distLDMatrixLocal.rows

    for(i <- 0 until numVariants; j <- 0 until numVariants) {
      val computedR = LDPrune.computeR(indexToBPV(i), indexToBPV(j))
      val distMatrixR = distLDMatrixLocal(i, j)
      val localMatrixR = localLDMatrixLocal(i, j)

      Assert.assertEquals(computedR, distMatrixR, .000001)
      Assert.assertEquals(computedR, localMatrixR, .000001)
    }
  }

  /**
    * Tests that variants are ordered in array.
    */
  @Test def testOrderingOfVariants() {
    assert(distLDMatrix.variants.isSorted)
    assert(localLDMatrix.variants.isSorted)
  }

  /**
    * Tests that the method agrees with local Breeze math.
    */
  @Test def localTest() {
    val genotypes: DenseMatrix[Int] = DenseMatrix((0, 2, 1),
                                                  (1, 0, 0))

    def localLDCompute(G: DenseMatrix[Int]): DenseMatrix[Double] = {
      val W = convert(G, Double).t

      for (i <- 0 until W.cols) {
        W(::, i) -= mean(W(::, i))
        W(::, i) *= math.sqrt(n) / norm(W(::, i))
      }

      val ld = (W.t * W) / n.toDouble
      ld
    }

    val localLD = localLDCompute(genotypes)

    val vds = stats.vdsFromGtMatrix(hc)(genotypes.t)
    val distLdSpark = vds.ldMatrix().matrix.toBlockMatrix().toLocalMatrix()
    val distLDBreeze = new DenseMatrix[Double](distLdSpark.numRows, distLdSpark.numCols, distLdSpark.toArray)

    TestUtils.assertMatrixEqualityDouble(distLDBreeze, localLD)

  }

  @Test
  def readWriteIdentityLocalLDMatrix() {
    val actual = localLDMatrix

    val fname = tmpDir.createTempFile("test")
    actual.write(fname)
    assert(actual.toLocalMatrix == LDMatrix.read(hc, fname).toLocalMatrix)
  }

  @Test
  def readWriteIdentityDistLDMatrix() {
    val actual = distLDMatrix

    val fname = tmpDir.createTempFile("test")
    actual.write(fname)
    assert(actual.toLocalMatrix == LDMatrix.read(hc, fname).toLocalMatrix)
  }

  private def readCSV(fname: String): Array[Array[Double]] =
    hc.hadoopConf.readLines(fname) { it =>
      it.map(_.value)
        .map(_.split(",").map(_.toDouble))
        .toArray[Array[Double]]
    }

  private def exportImportAssert(export: (String) => Unit, expected: Array[Double]*) {
    val fname = tmpDir.createTempFile("test")
    export(fname)
    assert(readCSV(fname) === expected.toArray[Array[Double]])
  }

  private def rowArrayToIRM(a: Array[Array[Double]]): IndexedRowMatrix = {
    val rows = a.length
    val cols = if (rows == 0) 0 else a(0).length
    new IndexedRowMatrix(
      sc.parallelize(a.zipWithIndex.map { case (a, i) => new IndexedRow(i, new DenseVector(a)) }),
      rows,
      cols)
  }

  private def rowArrayToLDMatrix(a: Array[Array[Double]]): LDMatrix = {
    val m = rowArrayToIRM(a)
    LDMatrix(hc, m, (0 until m.numRows().toInt).map(_ => Variant.gen.sample()).toArray[Variant], m.numCols().toInt)
  }

  @Test
  def exportSimple() {
    val fullExpected = Array(
      Array(1.0, 2.0, 3.0),
      Array(4.0, 5.0, 6.0),
      Array(7.0, 8.0, 9.0))
    val ldm = rowArrayToLDMatrix(fullExpected)

    exportImportAssert(ldm.export(_, ",", header=None, parallelWrite=false),
      fullExpected:_*)

    exportImportAssert(ldm.exportLowerTriangle(_, ",", header=None, parallelWrite=false),
      Array(4.0),
      Array(7.0, 8.0))

    exportImportAssert(ldm.exportStrictLowerTriangle(_, ",", header=None, parallelWrite=false),
      Array(1.0),
      Array(4.0, 5.0),
      Array(7.0, 8.0, 9.0))

    exportImportAssert(ldm.exportStrictUpperTriangle(_, ",", header=None, parallelWrite=false),
      Array(1.0, 2.0, 3.0),
      Array(5.0, 6.0),
      Array(9.0))

    exportImportAssert(ldm.exportUpperTriangle(_, ",", header=None, parallelWrite=false),
      Array(2.0, 3.0),
      Array(6.0))
  }

  @Test
  def exportAllZeros() {
    val allZeros = Array(
      Array(0.0, 0.0, 0.0),
      Array(0.0, 0.0, 0.0),
      Array(0.0, 0.0, 0.0))
    val ldm = rowArrayToLDMatrix(allZeros)

    exportImportAssert(ldm.export(_, ",", header=None, parallelWrite=false),
      Array(0.0, 0.0, 0.0),
      Array(0.0, 0.0, 0.0),
      Array(0.0, 0.0, 0.0))

    exportImportAssert(ldm.exportLowerTriangle(_, ",", header=None, parallelWrite=false),
      Array(0.0),
      Array(0.0, 0.0))

    exportImportAssert(ldm.exportStrictLowerTriangle(_, ",", header=None, parallelWrite=false),
      Array(0.0),
      Array(0.0, 0.0),
      Array(0.0, 0.0, 0.0))

    exportImportAssert(ldm.exportStrictUpperTriangle(_, ",", header=None, parallelWrite=false),
      Array(0.0, 0.0, 0.0),
      Array(0.0, 0.0),
      Array(0.0))

    exportImportAssert(ldm.exportUpperTriangle(_, ",", header=None, parallelWrite=false),
      Array(0.0, 0.0),
      Array(0.0))
  }

  @Test
  def exportBigish() {
    val lm = distLDMatrix.toLocalMatrix
    val expected = (0 until lm.rows)
      .map(i => (0 until lm.cols).map(j =>
        lm(i, j)).toArray[Double])
      .toArray[Array[Double]]

    exportImportAssert(distLDMatrix.export(_, ",", header=None, parallelWrite=false),
      expected:_*)

    exportImportAssert(distLDMatrix.exportLowerTriangle(_, ",", header=None, parallelWrite=false),
      expected.zipWithIndex
        .map { case (a,i) =>
          a.zipWithIndex.filter { case (_, j) => j < i }.map(_._1).toArray[Double] }
        .filter(_.nonEmpty)
        .toArray[Array[Double]]:_*)

    exportImportAssert(distLDMatrix.exportStrictLowerTriangle(_, ",", header=None, parallelWrite=false),
      expected.zipWithIndex
        .map { case (a,i) =>
          a.zipWithIndex.filter { case (_, j) => j <= i }.map(_._1).toArray[Double] }
        .toArray[Array[Double]]:_*)

    exportImportAssert(distLDMatrix.exportStrictUpperTriangle(_, ",", header=None, parallelWrite=false),
      expected.zipWithIndex
        .map { case (a,i) =>
          a.zipWithIndex.filter { case (_, j) => j >= i }.map(_._1).toArray[Double] }
        .toArray[Array[Double]]:_*)

    exportImportAssert(distLDMatrix.exportUpperTriangle(_, ",", header=None, parallelWrite=false),
      expected.zipWithIndex
        .map { case (a,i) =>
          a.zipWithIndex.filter { case (_, j) => j > i }.map(_._1).toArray[Double] }
        .filter(_.nonEmpty)
        .toArray[Array[Double]]:_*)
  }
}
