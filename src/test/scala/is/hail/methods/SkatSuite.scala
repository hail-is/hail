package is.hail.methods

import is.hail.utils._
import is.hail.expr._
import is.hail.{SparkSuite, TestUtils}
import is.hail.io.annotators.IntervalList
import is.hail.variant.{GenomeReference, Genotype, VariantDataset}
import is.hail.annotations.Annotation

import scala.sys.process._
import breeze.linalg._
import breeze.numerics.sigmoid

import scala.language.postfixOps
import is.hail.stats.RegressionUtils._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.testng.annotations.Test

case class SkatAggForR(xs: ArrayBuilder[Vector[Double]], weights: ArrayBuilder[Double])

class SkatSuite extends SparkSuite {
  def skatInR(vds: VariantDataset,
    variantKeys: String,
    singleKey: Boolean,
    yExpr: String,
    covExpr: Array[String],
    weightExpr: Option[String],
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
  
    def resultOp(st: Array[(Vector[Double], Double)], n: Int): (DenseMatrix[Double], DenseVector[Double]) = {
      val m = st.length
      val xArray = Array.ofDim[Double](m * n)
      val wArray = Array.ofDim[Double](m)
  
      var i = 0
      while (i < m) {
        val (xw, w) = st(i)
        wArray(i) = w
        xw match {
          case xw: SparseVector[Double] =>
            val index = xw.index
            val data = xw.data
            var j = 0
            while (j < index.length) {
              xArray(i * n + index(j)) = data(j)
              j += 1
            }
          case xw: DenseVector[Double] =>
            val data = xw.data
            var j = 0
            while (j < n) {
              xArray(i * n + j) = data(j)
              j += 1
            }
        }
        i += 1
  
      }
  
      (new DenseMatrix(n, m, xArray), new DenseVector(wArray))
    }
    
    def runInR(keyedRdd:  RDD[(Annotation, Iterable[(Vector[Double], Double)])], keyType: Type,
      y: DenseVector[Double], cov: DenseMatrix[Double],
      resultOp: (Array[(Vector[Double], Double)], Int) => (DenseMatrix[Double], DenseVector[Double])): Array[Row] = {

      val n = y.size
      val k = cov.cols
      val d = n - k

      if (d < 1)
        fatal(s"$n samples and $k ${ plural(k, "covariate") } including intercept implies $d degrees of freedom.")

      // fit null model
      val qr.QR(q, r) = qr.reduced.impl_reduced_DM_Double(cov)
      val beta = r \ (q.t * y)
      val res = y - cov * beta
      val sigmaSq = (res dot res) / d

      val inputFilePheno = tmpDir.createLocalTempFile("skatPhenoVec", ".txt")
      hadoopConf.writeTextFile(inputFilePheno) {
        _.write(TestUtils.matrixToString(y.toDenseMatrix, " "))
      }

      val inputFileCov = tmpDir.createLocalTempFile("skatCovMatrix", ".txt")
      hadoopConf.writeTextFile(inputFileCov) {
        _.write(TestUtils.matrixToString(cov, " "))
      }

      val skatRDD = keyedRdd.collect()
        .map { case (key, vs) =>
          val (xs, weights) = resultOp(vs.toArray, n)

          //write files to a location R script can read
          val inputFileG = tmpDir.createLocalTempFile("skatGMatrix", ".txt")
          hadoopConf.writeTextFile(inputFileG) {
            _.write(TestUtils.matrixToString(xs, " "))
          }

          val inputFileW = tmpDir.createLocalTempFile("skatWeightVec", ".txt")
          hadoopConf.writeTextFile(inputFileW) {
            _.write(TestUtils.matrixToString(weights.toDenseMatrix, " "))
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
    val n = y.size
    val sampleMask = Array.fill[Boolean](vds.nSamples)(false)
    completeSampleIndex.foreach(i => sampleMask(i) = true)
    val filteredVds = vds.filterSamplesMask(sampleMask)

    val (keyedRdd, keysType) =
      Skat.toKeyGsWeightRdd(filteredVds, variantKeys, singleKey, weightExpr, useDosages)
    
    runInR(keyedRdd, keysType, y, cov, resultOp)
  }

  // 18 complete samples from sample2.vcf, 5 genes
  // Using specified weights in Hail and R
  lazy val vdsSkat: VariantDataset = {
    val covSkat = hc.importTable("src/test/resources/skat.cov",
      impute = true).keyBy("Sample")

    val phenoSkat = hc.importTable("src/test/resources/skat.pheno",
      types = Map("Pheno" -> TFloat64), missing = "0").keyBy("Sample")

    val intervalsSkat = IntervalList.read(hc, "src/test/resources/skat.interval_list")

    val rg = GenomeReference.GRCh37

    val weightsSkat = hc.importTable("src/test/resources/skat.weights",
      types = Map("locus" -> TLocus(rg), "weight" -> TFloat64)).keyBy("locus")

    hc.importVCF("src/test/resources/sample2.vcf")
      .filterMulti()
      .annotateVariantsTable(intervalsSkat, root = "va.genes") // intervals do not overlap
      .annotateVariantsTable(weightsSkat, root = "va.weight")
      .annotateSamplesTable(covSkat, root = "sa.cov")
      .annotateSamplesTable(phenoSkat, root = "sa.pheno")
      .annotateSamplesExpr("sa.pheno = if (sa.pheno == 1.0) false else if (sa.pheno == 2.0) true else NA: Boolean")
  }
  
  // R uses a small sample correction for logistic below 2000 samples, Hail does not
  // So here we make a large deterministic example using the Balding-Nichols model (only hardcalls)
  // Using default weights in both Hail and R
  lazy val vdsBN: VariantDataset = {
    val seed = 0
    val nSamples = 2001
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
      .annotateSamples(TFloat64, List("cov", "Cov1"), s => cov1Array(s.asInstanceOf[String].toInt))
      .annotateSamples(TFloat64, List("cov", "Cov2"), s => cov2Array(s.asInstanceOf[String].toInt))
      .annotateSamples(TBoolean, List("pheno"), s => phenoArray(s.asInstanceOf[String].toInt))
      .annotateVariantsExpr("va.genes = [v.start % 2, v.start % 3].toSet") // three overlapping genes
  }
  
  def hailVsRTest(useBN: Boolean, useDosages: Boolean, useLogistic: Boolean, useLargeN: Boolean,
    displayValues: Boolean = false, tol: Double = 1e-5) {
   
    require(useBN || !useLogistic)
    require(!(useBN && useDosages))
    
    val (vds, singleKey, weightExpr) = if (useBN) (vdsBN, false, None) else (vdsSkat, true, Some("va.weight"))
    
    val hailKT = vds.skat("va.genes", singleKey = singleKey, "sa.pheno", Array("sa.cov.Cov1", "sa.cov.Cov2"),
      weightExpr, useLogistic, useDosages, forceLargeN=useLargeN)

    hailKT.typeCheck()
    
    val resultHail = hailKT.rdd.collect()

    val resultsR = skatInR(vds, "va.genes", singleKey = singleKey, "sa.pheno", Array("sa.cov.Cov1", "sa.cov.Cov2"),
      weightExpr, useLogistic, useDosages)

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
      assert(D_==(qstat, qstatR, tol))
      assert(math.abs(pval - pvalR) < 2e-6) // R Davies accuracy is only up to 1e-6
      
      i += 1
    }
  }
  
  @Test def linearHardcalls() {
    val useBN = false
    val useDosages = false
    val useLargeN = false
    val useLogistic = false
    hailVsRTest(useBN, useDosages, useLogistic, useLargeN)
  }

  @Test def linearDosages() {
    val useBN = false
    val useDosages = true
    val useLargeN = false
    val useLogistic = false
    hailVsRTest(useBN, useDosages, useLogistic, useLargeN)
  }

  @Test def linearLargeNHardCalls() {
    val useBN = false
    val useDosages = false
    val useLargeN = true
    val useLogistic = false
    hailVsRTest(useBN, useDosages, useLogistic, useLargeN)
  }

  @Test def linearLargeNDosages() {
    val useBN = false
    val useDosages = true
    val useLargeN = true
    val useLogistic = false
    hailVsRTest(useBN, useDosages, useLogistic, useLargeN)
  }
  
  @Test def linearHardcallsBN() {
    val useBN= true
    val useDosages = false
    val useLargeN = false
    val useLogistic = false
    hailVsRTest(useBN, useDosages, useLogistic, useLargeN)
  }

  @Test def linearLargeNHardcallsBN() {
    val useBN = true
    val useDosages = false
    val useLargeN = true
    val useLogistic = false
    hailVsRTest(useBN, useDosages, useLogistic, useLargeN)
  }
  
  @Test def logisticHardCallsBN() {
    val useBN = true
    val useDosages = false
    val useLargeN = false
    val useLogistic = true
    hailVsRTest(useBN, useDosages, useLogistic, useLargeN)
  }

  @Test def logisticLargeNHardCalls() {
    val useBN = true
    val useDosages = false
    val useLargeN = true
    val useLogistic = true
    hailVsRTest(useBN, useDosages, useLogistic, useLargeN)
  }
}
