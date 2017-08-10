package is.hail.methods

import is.hail.utils._
import java.io._

import scala.io.Source
import is.hail.expr._
import is.hail.SparkSuite
import is.hail.io.annotators.IntervalList
import is.hail.keytable.KeyTable
import is.hail.variant.{Genotype, VariantDataset}

import scala.sys.process._
import breeze.linalg._
import breeze.numerics.abs

import scala.language.postfixOps
import is.hail.annotations.Annotation
import is.hail.methods.Skat.polymorphicSkat
import is.hail.stats.{RegressionUtils, SkatModel}
import org.apache.spark.sql.Row
import org.testng.annotations.Test
import org.scalatest.Assertions


object SkatAggForR {
  val zeroValDense = SkatAggForR(new ArrayBuilder[DenseVector[Double]](), new ArrayBuilder[Double])
  val zeroValSparse = SkatAggForR(new ArrayBuilder[SparseVector[Double]](), new ArrayBuilder[Double])

  def seqOp[T <: Vector[Double]](safr: SkatAggForR[T], info: (T, Double)): SkatAggForR[T] = {
    val (x, weight) = info
    SkatAggForR(safr.xs + x, safr.weights + weight)
  }

  def combOp[T <: Vector[Double]](safr: SkatAggForR[T], safr2: SkatAggForR[T]): SkatAggForR[T] =
    SkatAggForR(safr.xs ++ safr2.xs.result(), safr.weights ++ safr2.weights.result())

  def sparseResultOp(safr: SkatAggForR[SparseVector[Double]], n: Int): (DenseMatrix[Double], DenseVector[Double]) = {
    val m = safr.xs.size
    val xArray = Array.ofDim[Double](m * n)

    var i = 0
    while (i < m) {
      val index = safr.xs(i).index
      val data = safr.xs(i).data
      var j = 0
      while (j < index.length) {
        xArray(i * n + index(j)) = data(j)
        j += 1
      }
      i += 1
    }

    (new DenseMatrix(n, m, xArray), new DenseVector(safr.weights.result()))
  }

  def denseResultOp(safr: SkatAggForR[DenseVector[Double]], n: Int): (DenseMatrix[Double], DenseVector[Double]) = {
    val m = safr.xs.size
    val xArray = Array.ofDim[Double](m * n)

    var i = 0
    while (i < m) {
      val data = safr.xs(i).data
      var j = 0
      while (j < n) {
        xArray(i * n + j) = data(j)
        j += 1
      }
      i += 1
    }

    (new DenseMatrix(n, m, xArray), new DenseVector(safr.weights.result()))
  }
}

case class SkatAggForR[T <: Vector[Double]](xs: ArrayBuilder[T], weights: ArrayBuilder[Double])

class SkatSuite extends SparkSuite {
  def testInR(vds: VariantDataset,
    keyName: String,
    variantKeys: String,
    singleKey: Boolean,
    weightExpr: String,
    yExpr: String,
    covExpr: Array[String],
    useDosages: Boolean): Array[Row] = {

    if (!useDosages) {
      polymorphicTestInR(vds, keyName, variantKeys, singleKey, weightExpr, yExpr, covExpr,
        RegressionUtils.hardCalls(_, _), SkatAggForR.zeroValSparse, SkatAggForR.sparseResultOp _)
    }
    else {
      val dosages = (gs: Iterable[Genotype], n: Int) => RegressionUtils.dosages(gs, (0 until n).toArray)
      polymorphicTestInR(vds, keyName, variantKeys, singleKey, weightExpr, yExpr, covExpr,
        dosages, SkatAggForR.zeroValDense, SkatAggForR.denseResultOp _)
    }
  }

  def polymorphicTestInR[T <: Vector[Double]](vds: VariantDataset,
    keyName: String,
    variantKeys: String,
    singleKey: Boolean,
    weightExpr: String,
    yExpr: String,
    covExpr: Array[String],
    getGenotype: (Iterable[Genotype], Int) => T, zero: SkatAggForR[T],
    resultOp: (SkatAggForR[T], Int) => (DenseMatrix[Double], DenseVector[Double])): Array[Row] = {


    //get variables
    val (y, cov, completeSamples) = RegressionUtils.getPhenoCovCompleteSamples(vds, yExpr, covExpr)

    val n = y.size
    val k = cov.cols
    val d = n - k - 1

    val filteredVds = vds.filterSamplesList(completeSamples.toSet)

    val (keysType, keysQuerier) = filteredVds.queryVA(variantKeys)
    val (weightType, weightQuerier) = filteredVds.queryVA(weightExpr)

    val typedWeightQuerier = weightType match {
      case TFloat64 => weightQuerier.asInstanceOf[Annotation => Double]
      case TFloat32 => weightQuerier.asInstanceOf[Annotation => Double]
      case TInt64 => weightQuerier.asInstanceOf[Annotation => Double]
      case TInt32 => weightQuerier.asInstanceOf[Annotation => Double]
      case _ => fatal("Weight must be a numeric type")
    }

    val (keyType, keyedRdd) =
      if (singleKey) {
        (keysType, filteredVds.rdd.flatMap { case (v, (va, gs)) =>
          (Option(keysQuerier(va)), Option(typedWeightQuerier(va))) match {
            case (Some(key), Some(w)) =>
              val x = getGenotype(gs, n)
              Some((key, (x, w)))
            case _ => None
          }
        })
      } else {
        val keyType = keysType match {
          case TArray(e) => e
          case TSet(e) => e
          case _ => fatal(s"With single_key=False, variant keys must be of type Set[T] or Array[T], got $keysType")
        }
        (keyType, filteredVds.rdd.flatMap { case (v, (va, gs)) =>
          val keys = Option(keysQuerier(va).asInstanceOf[Iterable[_]]).getOrElse(Iterable.empty)
          val optWeight = Option(typedWeightQuerier(va))
          if (keys.isEmpty || optWeight.isEmpty)
            Iterable.empty
          else {
            val x = getGenotype(gs, n)
            keys.map((_, (x, optWeight.get)))
          }
        })
      }

    val aggregatedKT = keyedRdd.aggregateByKey(zero)(SkatAggForR.seqOp, SkatAggForR.combOp)

    val inputFilePheno = tmpDir.createLocalTempFile("skatPhenoVec", ".txt")
    hadoopConf.writeTextFile(inputFilePheno) {
      _.write(largeMatrixToString(y.toDenseMatrix, " "))
    }

    val inputFileCov = tmpDir.createLocalTempFile("skatCovMatrix", ".txt")
    hadoopConf.writeTextFile(inputFileCov) {
      _.write(largeMatrixToString(cov, " "))
    }

    aggregatedKT.collect().map { case (key, safr) => {
      val (xs, weights) = resultOp(safr, n)

      //write files to a location R script can read
      val inputFileG = tmpDir.createLocalTempFile("skatGMatrix", ".txt")
      hadoopConf.writeTextFile(inputFileG) {
        _.write(largeMatrixToString(xs, " "))
      }

      val inputFileW = tmpDir.createLocalTempFile("skatWeightVec", ".txt")
      hadoopConf.writeTextFile(inputFileW) {
        _.write(largeMatrixToString(weights.toDenseMatrix, " "))
      }

      val resultsFile = tmpDir.createLocalTempFile("results", ".txt")

      val rScript = s"Rscript src/test/resources/skatTest.R " +
        s"${ uriPath(inputFileG) } ${ uriPath(inputFileCov) } " +
        s"${ uriPath(inputFilePheno) } ${ uriPath(inputFileW) } " +
        s"${ uriPath(resultsFile) } " + "C"

      rScript !
      val results = readResults(resultsFile)

      Row(key, results(0)(0), results(0)(1))
    }
    }

  }


  def covariates = hc.importTable("src/test/resources/regressionLinear.cov",
    types = Map("Cov1" -> TFloat64, "Cov2" -> TFloat64)).keyBy("Sample")

  def phenotypes = hc.importTable("src/test/resources/regressionLinear.pheno",
    types = Map("Pheno" -> TFloat64), missing = "0").keyBy("Sample")

  def intervals = IntervalList.read(hc, "src/test/resources/regressionLinear.interval_list")

  def vds: VariantDataset = hc.importVCF("src/test/resources/regressionLinear.vcf")
    .annotateVariantsTable(intervals, root = "va.genes", product = true)
    .annotateSamplesTable(phenotypes, root = "sa.pheno")
    .annotateSamplesTable(covariates, root = "sa.cov")
    //  .annotateVariantsExpr("va.weight = v.start.toDouble")
    .annotateSamplesExpr("sa.pheno = if (sa.pheno == 1.0) false else if (sa.pheno == 2.0) true else NA: Boolean")
    .filterSamplesExpr("sa.pheno.isDefined() && sa.cov.Cov1.isDefined() && sa.cov.Cov2.isDefined()")
    .annotateVariantsExpr("va.AF = gs.callStats(g=> v).AF")
    .annotateVariantsExpr("va.weight = let af = if (va.AF[0] <= va.AF[1]) va.AF[0] else va.AF[1] in dbeta(af,1.0,25.0)**2")

  @Test def hardcallsSmallTest() {

    val useDosages = false

    val kt = vds.skat("gene", "va.genes", singleKey = false, Option("va.weight"),
      "sa.pheno", covariates = Array("sa.cov.Cov1", "sa.cov.Cov2"), useDosages)

    val resultsArray = testInR(vds, "gene", "va.genes", singleKey = false,
      "va.weight", "sa.pheno", Array("sa.cov.Cov1", "sa.cov.Cov2"), useDosages)

    val rows = kt.rdd.collect()

    val tol = 1e-5
    var i = 0

    while (i < resultsArray.size) {

      val qstat = rows(i).get(1).asInstanceOf[Double]
      val pval = rows(i).get(2).asInstanceOf[Double]

      val qstatR = resultsArray(i)(1).asInstanceOf[Double]
      val pvalR = resultsArray(i)(2).asInstanceOf[Double]
      if (pval <= 1 && pval >= 0) {

        assert(D_==(qstat, qstatR, tol))
        assert(D_==(pval, pvalR, tol))
      }

      i += 1
    }

  }

  @Test def dosagesSmallTest() {

    val useDosages = true

    val resultsArray = testInR(vds, "gene", "va.genes", singleKey = false,
      "va.weight", "sa.pheno", Array("sa.cov.Cov1", "sa.cov.Cov2"), useDosages)

    val kt = vds.skat("gene", "va.genes", singleKey = false, Option("va.weight"),
      "sa.pheno", covariates = Array("sa.cov.Cov1", "sa.cov.Cov2"), useDosages)


    val rows = kt.rdd.collect()

    val tol = 1e-5
    var i = 0

    while (i < resultsArray.size) {
      val qstat = rows(i).get(1).asInstanceOf[Double]
      val pval = rows(i).get(2).asInstanceOf[Double]

      val qstatR = resultsArray(i)(1).asInstanceOf[Double]
      val pvalR = resultsArray(i)(2).asInstanceOf[Double]

      if (pval <= 1 && pval >= 0) {
        assert(D_==(qstat, qstatR, tol))
        assert(D_==(pval, pvalR, tol))
      }
      i += 1
    }

  }

  @Test def largeNSmallTest() {

    val useDosages = false

    val kt = polymorphicSkat[SparseVector[Double]](vds, "gene", "va.genes", singleKey = false, Option("va.weight"), "sa.pheno",
      Array("sa.cov.Cov1", "sa.cov.Cov2"), RegressionUtils.hardCalls(_, _), SkatAgg.zeroValSparse,
      SkatAgg.largeNResultOp)

    val resultsArray = testInR(vds, "gene", "va.genes", singleKey = false,
      "va.weight", "sa.pheno", Array("sa.cov.Cov1", "sa.cov.Cov2"), useDosages)

    val rows = kt.rdd.collect()

    val tol = 1e-5
    var i = 0

    while (i < resultsArray.size) {
      val qstat = rows(i).get(1).asInstanceOf[Double]
      val pval = rows(i).get(2).asInstanceOf[Double]

      val qstatR = resultsArray(i)(1).asInstanceOf[Double]
      val pvalR = resultsArray(i)(2).asInstanceOf[Double]

      if (pval <= 1 && pval >= 0) {
        assert(D_==(qstat, qstatR, tol))
        assert(D_==(pval, pvalR, tol))
      }
      i += 1
    }

  }

  //Routines for running programs in R for comparison

  def readResults(file: String) = {
    hadoopConf.readLines(file) {
      _.map {
        _.map {
          _.split(" ").map(_.toDouble)
        }.value
      }.toArray
    }
  }

  def largeMatrixToString(A: DenseMatrix[Double], seperator: String): String = {
    var string: String = ""
    for (i <- 0 until A.rows) {
      for (j <- 0 until A.cols) {
        string = string + seperator + A(i, j).toString()
      }
      string = string + "\n"
    }
    string
  }

  //Generates random data
  def buildWeightValues(filepath: String): Unit = {

    //read in chrom:pos values
    val fileSource = Source.fromFile(filepath)
    val linesArray = fileSource.getLines.toArray
    fileSource.close()

    //write in randomized weights
    val fileObject = new PrintWriter(new File(filepath))

    for (i <- -1 until linesArray.size) {
      if (i == -1) {
        fileObject.write("Pos\tWeights\n")
      }
      else {
        val pos = linesArray(i)
        val randomWeight = scala.util.Random.nextDouble()
        fileObject.write(s"$pos\t$randomWeight\n")

      }
    }
    fileObject.close()
  }

  def buildCovariateMatrix(filepath: String, covariateCount: Int): Unit = {
    val fileObject = new PrintWriter(new File(filepath))

    val startIndex = 96
    val endIndex = 116

    for (i <- startIndex - 1 to endIndex) {

      if (i == startIndex - 1) {
        fileObject.write("Sample\t")
        for (j <- 1 to covariateCount) {
          if (j == covariateCount) {
            fileObject.write(s"Cov$j")
          }
          else {
            fileObject.write(s"Cov$j\t")
          }
        }
        fileObject.write("\n")
      }
      else {
        fileObject.write("HG%05d\t".format(i))
        for (j <- 1 to covariateCount) {
          if (j == covariateCount) {
            fileObject.write("%d".format(scala.util.Random.nextInt(25)))
          }
          else {
            fileObject.write("%d\t".format(scala.util.Random.nextInt(25)))
          }
        }
        fileObject.write("\n")
      }
    }

    fileObject.close()
  }

  //Dataset for big Test

  def covariatesSkat = hc.importTable("src/test/resources/skat.cov",
    impute = true).keyBy("Sample")

  def phenotypesSkat = hc.importTable("src/test/resources/skat.pheno",
    types = Map("Pheno" -> TFloat64), missing = "0").keyBy("Sample")

  def intervalsSkat = IntervalList.read(hc, "src/test/resources/skat.interval_list")

  def weightsSkat = hc.importTable("src/test/resources/skat.weights",
    types = Map("locus" -> TLocus, "weight" -> TFloat64)).keyBy("locus")

  def vdsSkat: VariantDataset = hc.importVCF("src/test/resources/sample2.vcf")
    .annotateVariantsTable(intervalsSkat, root = "va.genes", product = true)
    .annotateVariantsTable(weightsSkat, root = "va.weight")
    .annotateSamplesTable(phenotypesSkat, root = "sa.pheno0")
    .annotateSamplesTable(covariatesSkat, root = "sa.cov")
    .annotateSamplesExpr("sa.pheno = if (sa.pheno0 == 1.0) false else if (sa.pheno0 == 2.0) true else NA: Boolean")


  @Test def hardcallsBigTest() {

    val useDosages = false
    val covariates = new Array[String](2)
    for (i <- 1 to 2) {
      covariates(i - 1) = "sa.cov.Cov%d".format(i)
    }

    val kt = vdsSkat.splitMulti().skat("gene", "va.genes", singleKey = false, Option("va.weight"),
      "sa.pheno0", covariates, useDosages)
    val resultsArray = testInR(vdsSkat.splitMulti(), "gene", "va.genes", singleKey = false,
      "va.weight", "sa.pheno0", covariates, useDosages)

    val rows = kt.rdd.collect()

    var i = 0
    val tol = 1e-5

    while (i < resultsArray.size) {

      val qstat = rows(i).get(1).asInstanceOf[Double]
      val pval = rows(i).get(2).asInstanceOf[Double]

      val qstatR = resultsArray(i)(1).asInstanceOf[Double]
      val pvalR = resultsArray(i)(2).asInstanceOf[Double]

      if (pval <= 1 && pval >= 0) {
        assert(D_==(qstat, qstatR, tol))
        assert(D_==(pval, pvalR, tol))
      }
      i += 1
    }
  }

  @Test def dosagesBigTest() {
    val useDosages = true
    val covariates = new Array[String](2)
    for (i <- 1 to 2) {
      covariates(i - 1) = "sa.cov.Cov%d".format(i)
    }

    val kt = vdsSkat.splitMulti().skat("gene", "va.genes", singleKey = false, Option("va.weight"),
      "sa.pheno0", covariates, useDosages)
    val resultsArray = testInR(vdsSkat.splitMulti(), "gene", "va.genes", singleKey = false,
      "va.weight", "sa.pheno0", covariates, useDosages)

    val rows = kt.rdd.collect()

    var i = 0
    val tol = 1e-5

    while (i < resultsArray.size) {
      val qstat = rows(i).get(1).asInstanceOf[Double]
      val pval = rows(i).get(2).asInstanceOf[Double]

      val qstatR = resultsArray(i)(1).asInstanceOf[Double]
      val pvalR = resultsArray(i)(2).asInstanceOf[Double]

      if (pval <= 1 && pval >= 0) {
        assert(D_==(qstat, qstatR, tol))
        assert(D_==(pval, pvalR, tol))
      }
      i += 1
    }
  }

  @Test def largeNBigTest() {
    val useDosages = true

    val covariates = new Array[String](2)
    for (i <- 1 to 2) {
      covariates(i - 1) = "sa.cov.Cov%d".format(i)
    }

    val dosages = (gs: Iterable[Genotype], n: Int) => RegressionUtils.dosages(gs, (0 until n).toArray)

    val kt = polymorphicSkat[DenseVector[Double]](vdsSkat.splitMulti(), "gene", "va.genes", singleKey = false,
      Option("va.weight"), "sa.pheno0", Array("sa.cov.Cov1", "sa.cov.Cov2"), dosages, SkatAgg.zeroValDense,
      SkatAgg.largeNResultOp)

    val resultsArray = testInR(vdsSkat.splitMulti(), "gene", "va.genes", singleKey = false,
      "va.weight", "sa.pheno0", covariates, useDosages)

    val rows = kt.rdd.collect()

    var i = 0
    val tol = 1e-5

    while (i < resultsArray.size) {
      val qstat = rows(i).get(1).asInstanceOf[Double]
      val pval = rows(i).get(2).asInstanceOf[Double]

      val qstatR = resultsArray(i)(1).asInstanceOf[Double]
      val pvalR = resultsArray(i)(2).asInstanceOf[Double]
      if (pval <= 1 && pval >= 0) {
        assert(D_==(qstat, qstatR, tol))
        assert(D_==(pval, pvalR, tol))
      }
      i += 1
    }
  }
}


