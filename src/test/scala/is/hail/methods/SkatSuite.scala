package is.hail.methods

import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.io.annotators.IntervalList
import is.hail.{SparkSuite, TestUtils}
import is.hail.stats.RegressionUtils._
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant._
import is.hail.stats.vdsFromGtMatrix

import breeze.linalg._
import breeze.numerics.sigmoid
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.testng.annotations.Test

import scala.sys.process._
import scala.language.postfixOps

case class SkatAggForR(xs: ArrayBuilder[DenseVector[Double]], weights: ArrayBuilder[Double])

class SkatSuite extends SparkSuite {
  def skatInR(vds: VariantDataset,
    keyExpr: String,
    weightExpr: String,
    yExpr: String,
    covExpr: Array[String],
    useLogistic: Boolean,
    useDosages: Boolean): Array[Row] = {
    
    def readRResults(file: String): Array[Array[Double]] = {
      hadoopConf.readLines(file) {
        _.map {
          _.map {
            _.split(" ").map(_.toDouble)
          }.value
        }.toArray
      }
    }
  
    def formGenotypeMatrixAndWeightVector(
      xwArray: Array[(DenseVector[Double], Double)], n: Int): (DenseMatrix[Double], DenseVector[Double]) = {
      
      val m = xwArray.length
      val genotypeData = Array.ofDim[Double](m * n)
      val weightData = Array.ofDim[Double](m)
  
      var i = 0
      while (i < m) {
        val (x, w) = xwArray(i)
        weightData(i) = w
        val data = x.toArray
        var j = 0
        while (j < n) {
          genotypeData(i * n + j) = data(j)
          j += 1
        }
        i += 1
      }

      (new DenseMatrix(n, m, genotypeData), DenseVector(weightData))
    }
    
    def runInR(keyGsWeightRdd:  RDD[(Annotation, Iterable[(DenseVector[Double], Double)])], keyType: Type,
      y: DenseVector[Double], cov: DenseMatrix[Double]): Array[Row] = {

      val inputFilePheno = tmpDir.createLocalTempFile("skatPhenoVec", ".txt")
      hadoopConf.writeTextFile(inputFilePheno) {
        _.write(TestUtils.matrixToString(y.toDenseMatrix, " "))
      }

      val inputFileCov = tmpDir.createLocalTempFile("skatCovMatrix", ".txt")
      hadoopConf.writeTextFile(inputFileCov) {
        _.write(TestUtils.matrixToString(cov, " "))
      }

      val skatRDD = keyGsWeightRdd.collect()
        .map { case (key, xw) =>
          val (genotypeMatrix, weightVector) = formGenotypeMatrixAndWeightVector(xw.toArray, y.size)

          val inputFileG = tmpDir.createLocalTempFile("skatGMatrix", ".txt")
          hadoopConf.writeTextFile(inputFileG) {
            _.write(TestUtils.matrixToString(genotypeMatrix, " "))
          }

          val inputFileW = tmpDir.createLocalTempFile("skatWeightVec", ".txt")
          hadoopConf.writeTextFile(inputFileW) {
            _.write(TestUtils.matrixToString(weightVector.toDenseMatrix, " "))
          }

          val resultsFile = tmpDir.createLocalTempFile("results", ".txt")

          val datatype = if (useLogistic) "D" else "C"

          val rScript = s"Rscript src/test/resources/skatTest.R " +
            s"${ uriPath(inputFileG) } ${ uriPath(inputFileCov) } " +
            s"${ uriPath(inputFilePheno) } ${ uriPath(inputFileW) } " +
            s"${ uriPath(resultsFile) } " + datatype

          rScript !
          val results = readRResults(resultsFile)

          Row(key, results(0)(0), results(0)(1))
        }
      
      skatRDD
    }

    val (y, cov, completeSampleIndex) = getPhenoCovCompleteSamples(vds, yExpr, covExpr)

    val (keyGsWeightRdd, keyType) =
      Skat.computeKeyGsWeightRdd(vds, if (useDosages) "plDosage(g.PL)" else "g.GT.nNonRefAlleles()",
        completeSampleIndex, keyExpr, weightExpr)

    runInR(keyGsWeightRdd, keyType, y, cov)
  }

  // 18 complete samples from sample2.vcf
  // 5 genes with sizes 24, 13, 66, 27, and 51
  lazy val vdsSkat: VariantDataset = {
    val covSkat = hc.importTable("src/test/resources/skat.cov",
      impute = true).keyBy("Sample")

    val phenoSkat = hc.importTable("src/test/resources/skat.pheno",
      types = Map("Pheno" -> TFloat64()), missing = "0").keyBy("Sample")

    val intervalsSkat = IntervalList.read(hc, "src/test/resources/skat.interval_list")

    val rg = GenomeReference.GRCh37

    val weightsSkat = hc.importTable("src/test/resources/skat.weights",
      types = Map("locus" -> TLocus(rg), "weight" -> TFloat64())).keyBy("locus")

    hc.importVCF("src/test/resources/sample2.vcf")
      .filterMulti()
      .annotateVariantsTable(intervalsSkat, root = "va.gene") // intervals do not overlap
      .annotateVariantsTable(weightsSkat, root = "va.weight")
      .annotateSamplesTable(covSkat, root = "sa.cov")
      .annotateSamplesTable(phenoSkat, root = "sa.pheno")
      .annotateSamplesExpr("sa.pheno = if (sa.pheno == 1.0) false else if (sa.pheno == 2.0) true else NA: Boolean")
  }
  
  // A larger deterministic example using the Balding-Nichols model (only hardcalls)
  lazy val vdsBN: VariantDataset = {
    val seed = 0
    val nSamples = 500
    val nVariants = 50
  
    val rand = scala.util.Random
    rand.setSeed(seed)
    
    val cov1Array: Array[Double] = Array.fill[Double](nSamples)(rand.nextGaussian())
    val cov2Array: Array[Double] = Array.fill[Double](nSamples)(rand.nextGaussian())
  
    val vdsBN0 = hc.baldingNicholsModel(1, nSamples, nVariants, seed = seed)
  
    val G: DenseMatrix[Double] = TestUtils.vdsToMatrixDouble(vdsBN0)
    val pi: DenseVector[Double] = sigmoid(sum(G(*, ::)) - nVariants.toDouble)
    val phenoArray: Array[Boolean] = pi.toArray.map(_ > rand.nextDouble())
    
    vdsBN0
      .annotateSamplesF(TFloat64(), List("cov", "Cov1"), s => cov1Array(s.asInstanceOf[String].toInt))
      .annotateSamplesF(TFloat64(), List("cov", "Cov2"), s => cov2Array(s.asInstanceOf[String].toInt))
      .annotateSamplesF(TBoolean(), List("pheno"), s => phenoArray(s.asInstanceOf[String].toInt))
      .annotateVariantsExpr("va.gene = v.start % 3") // three genes
      .annotateVariantsExpr("va.AF = gs.map(g => g.GT).callStats(GT => v).AF")
      .annotateVariantsExpr("va.weight = let af = if (va.AF[0] <= va.AF[1]) va.AF[0] else va.AF[1] in " +
        "dbeta(af, 1.0, 25.0)**2")
  }
  
  def hailVsRTest(useBN: Boolean = false, useDosages: Boolean = false, logistic: Boolean = false,
    displayValues: Boolean = false, qstatTol: Double = 1e-5) {
    
    require(!(useBN && useDosages))
    
    val vds = if (useBN) vdsBN else vdsSkat
    
    val hailKT = vds.skat("va.gene", "va.weight", "sa.pheno",
      if (useDosages) "plDosage(g.PL)" else "g.GT.nNonRefAlleles()",
      Array("sa.cov.Cov1", "sa.cov.Cov2"), logistic)

    hailKT.typeCheck()
    
    val resultHail = hailKT.rdd.collect()

    val resultsR = skatInR(vds, "va.gene", "va.weight", "sa.pheno",
      Array("sa.cov.Cov1", "sa.cov.Cov2"), logistic, useDosages)

    var i = 0
    while (i < resultsR.length) {
      val size = resultHail(i).getAs[Int](1)
      val qstat = resultHail(i).getAs[Double](2)
      val pval = resultHail(i).getAs[Double](3)
      val fault = resultHail(i).getAs[Int](4)

      val qstatR = resultsR(i).getAs[Double](1)
      val pvalR = resultsR(i).getAs[Double](2)
      
      if (displayValues) {
        println(f"HAIL qstat: $qstat%2.9f  pval: $pval  fault: $fault  size: $size")
        println(f"   R qstat: $qstatR%2.9f  pval: $pvalR")
      }

      assert(fault == 0)
      assert(D_==(qstat, qstatR, qstatTol))
      assert(math.abs(pval - pvalR) < math.max(qstatTol, 2e-6)) // R Davies accuracy is only up to 1e-6
      
      i += 1
    }
  }
  
  // testing linear
  @Test def linear() { hailVsRTest() }
  @Test def linearDosages() { hailVsRTest(useDosages = true) }
  @Test def linearBN() { hailVsRTest(useBN = true) }
  
  // testing logistic
  @Test def logistic() { hailVsRTest(logistic = true) }
  @Test def logisticDosages() { hailVsRTest(useDosages = true, logistic = true) }
  @Test def logisticBN() { hailVsRTest(useBN = true, logistic = true, qstatTol = 1e-4) }
  
  //testing size and maxSize
  @Test def maxSizeTest() {
    val maxSize = 27
    
    val kt = vdsSkat.skat("va.gene", y = "sa.pheno", x = "g.GT.nNonRefAlleles()", weightExpr = "1", maxSize = maxSize)
      
    val ktMap = kt.rdd.collect().map{ case Row(key, size, qstat, pval, fault) => 
        key.asInstanceOf[String] -> (size.asInstanceOf[Int], qstat == null, pval == null, fault == null) }.toMap
    
    assert(ktMap("Gene1") == (24, false, false, false))
    assert(ktMap("Gene2") == (13, false, false, false))
    assert(ktMap("Gene3") == (66, true, true, true))
    assert(ktMap("Gene4") == (27, false, false, false))
    assert(ktMap("Gene5") == (51, true, true, true))
  }
  
  @Test def smallNLargeNEqualityTest() {
    val rand = scala.util.Random
    rand.setSeed(0)
    
    val n = 10 // samples
    val m = 5 // variants
    val k = 3 // covariates
    
    val st = Array.tabulate(m){ _ => 
      SkatTuple(rand.nextDouble(),
        DenseVector(Array.fill(n)(rand.nextDouble())),
        DenseVector(Array.fill(k)(rand.nextDouble())))
    }
        

    val (qSmall, gramianSmall) = Skat.computeGramianSmallN(st)
    val (qLarge, gramianLarge) = Skat.computeGramianLargeN(st)
      
    assert(D_==(qSmall, qLarge))
    TestUtils.assertMatrixEqualityDouble(gramianSmall, gramianLarge)
  }
  
  @Test def testToKeyGsWeightRdd() {
    val v1 = Array(0, 1, 0, 2)
    val v2 = Array(0, 2, 1, 0)
    val v3 = Array(1, 0, 0, 1)

    val vds0 = vdsFromGtMatrix(hc)(new DenseMatrix[Int](4, 3, v1 ++ v2 ++ v3))
    
    // annotations from table
    val kt = IntervalList.read(hc, "src/test/resources/skat2.interval_list")
    val vds = vds0
      .annotateVariantsTable(kt, root = "va.key", product = true)
      .annotateVariantsExpr("va.key = va.key.toSet(), va.weight = v.start")
      .explodeVariants("va.key")
      .annotateVariantsExpr("va.key = va.key.toInt32()")
    
    // annotations from expr
    val vds2 = vds0
      .annotateVariantsExpr("va.key = [9, v.start % 2, v.start // 2 + 1].toSet(), va.weight = v.start") // v1 -> {9, 1}, v2 -> {9, 0, 2}, v3 -> {9, 1, 2}
      .explodeVariants("va.key") // 0 -> {v2}, 1 -> {v1, v3}, 2 -> {v2, v3}, 9 -> {v1, v2, v3}
    
    // table/explode and annotate/explode give same keys
    assert(vds.same(vds2))
    
    val (keyGsWeightRdd, keyType) = Skat.computeKeyGsWeightRdd(vds, "g.GT.nNonRefAlleles()",
      completeSampleIndex = Array(1, 3), keyExpr = "va.key", weightExpr = "va.weight")
    
    val keyToSet = keyGsWeightRdd.collect().map { case (key, it) => key.asInstanceOf[Int] -> it.toSet }.toMap
    
    // groups of SkatTuples are as expected
    assert(keyToSet(0) == Set((DenseVector(2.0, 0.0), 2)))
    assert(keyToSet(1) == Set((DenseVector(1.0, 2.0), 1), (DenseVector(0.0, 1.0), 3)))
    assert(keyToSet(2) == Set((DenseVector(2.0, 0.0), 2), (DenseVector(0.0, 1.0), 3)))
    assert(keyToSet(9) == Set((DenseVector(1.0, 2.0), 1), (DenseVector(2.0, 0.0), 2), (DenseVector(0.0, 1.0), 3)))
    
    assert(keyType == TInt32())
  }
}