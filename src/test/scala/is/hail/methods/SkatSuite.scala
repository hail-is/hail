package is.hail.methods

import is.hail.SparkSuite
import is.hail.TestUtils._
import is.hail.utils._
import is.hail.check.Prop._
import org.testng.annotations.Test
import breeze.linalg._
import scala.language.postfixOps
import scala.sys.process._
import java.io.File

import breeze.numerics._
import com.google.common.primitives.Doubles
import org.scalatest.testng.TestNGSuite

class SkatSuite extends SparkSuite {

  val header = "==============================================="

  def readResults(file: String) = {
    hadoopConf.readLines(file) {
      _.map {
        _.map { _.split(" ").map(_.toDouble)}.value
      }.toArray
    }
  }
  def largeMatrixToString(A: DenseMatrix[Double], seperator: String): String =
  {
    val rows = A.rows
    val cols = A.cols
    var string:String = ""
    for(i <- 0 until rows )
    {
      for(j <- 0 until cols)
      {
         string = string + seperator + A(i,j).toString()
      }
      string = string + "\n"
    }
    string
  }

  def randomizedInput(samples: Int, variants: Int, covariates: Int):
      (DenseMatrix[Int], DenseMatrix[Double], DenseVector[Double],
       DenseVector[Double])
  = {
    var G = DenseMatrix.zeros[Int](samples, variants)
    var cov = DenseMatrix.zeros[Double](samples, covariates)
    var pheno = DenseVector.zeros[Double](samples)
    var weights = DenseVector.zeros[Double](variants)

    //generate new seed
    val Pgenerator = new util.Random()

    for(i <- 0 until samples ;j <- 0 until variants)
      G(i,j) = Pgenerator.nextInt(2)
    for(i <- 0 until samples ;j <- 0 until covariates)
      cov(i,j) = Pgenerator.nextDouble()
    for(i <- 0 until samples)
      pheno(i) = Pgenerator.nextDouble()
    for(i <- 0 until variants)
      weights(i) = Pgenerator.nextDouble()
    (G, cov, pheno, weights)
  }

  @Test def smallScaleTest(): Unit = {
    val testsToAverageOver = 1

    println(header)
    println("Starting small scale skat test")
    println(header)

    println(header)
    println("Generating Input")
    println(header)
    val G = DenseMatrix((0, 1, 1), (1, 2, 0), (1, 0, 2), (2, 2, 1),
      (0, 2, 2), (1, 0, 0), (0, 0, 2))
    val covariates = DenseMatrix((.3, .2), (.5, .2), (.9, .3), (.2, .84),
      (2.0, .2), (.4, .1), (.7, .8))
    val phenotypes = DenseVector(1.3, 2.3, 4.5, -.4, .8, 6.1, -2)
    val weights = DenseVector(1.0, 4.0, 9.0)

    //write files to a location R script can read
    val inputFileG = tmpDir.createLocalTempFile("skatGMatrix", ".txt")
    hadoopConf.writeTextFile(inputFileG) { _.write(G.toString())}

    val inputFileCov = "/Users/charlesc/Documents/Software/R/skatCovMatrixNewFormat.txt"
    hadoopConf.writeTextFile(inputFileCov) { _.write(covariates.toString())}

    val inputFilePheno = tmpDir.createLocalTempFile("skatPhenoVec", ".txt")
    hadoopConf.writeTextFile(inputFilePheno)
      { _.write((phenotypes).toDenseMatrix.toString())}

    val inputFileW = tmpDir.createLocalTempFile("skatWeightVec",".txt")
    hadoopConf.writeTextFile(inputFileW)
      { _.write(weights.toDenseMatrix.toString())}

    val resultsFile = tmpDir.createLocalTempFile("results", ".txt")

    println(header)
    println("Starting Scala Routines")
    println(header)

    val SKAT = new skat(convert(G, Double), covariates, phenotypes, weights)
    val (skatNullModel,firstTiming1) = time{SKAT.fitCovariates()}
    var t1 = 0.0.toLong
    for (i <- 1 to testsToAverageOver) {
      val (s, t) = time {
        SKAT.fitCovariates()
      }
      t1 += t
    }


    println(header)
    println("Finished Regression in %s".format(formatTime(t1/testsToAverageOver)))
    println("   speed of first run  %s".format(formatTime(firstTiming1)))
    println(header)
    var ((skatStat,pValue),firstTiming2) = time {skatNullModel.computeSkatStats()}
    var t2 = 0.0.toLong
    for (i <- 1 to testsToAverageOver){
      val ((s,p), t) = time {
        skatNullModel.computeSkatStats()
      }
      t2 += t
    }
    println(header)
    println("Finished computing SKAT stats in %s".format(formatTime(t2/testsToAverageOver)))
    println("             speed of first run  %s".format(formatTime(firstTiming2)))
    println(header)

    println(f"Variance component statistic: $skatStat")
    println(f"                     p value: $pValue")

    println(header)
    println("Starting R Routines")
    println(header)

    val rScript = s"Rscript src/test/resources/skatTest.R " +
      s"${ uriPath(inputFileG) } ${ uriPath(inputFileCov) } " +
      s"${ uriPath(inputFilePheno) } ${ uriPath(inputFileW) } " +
      s"${ uriPath(resultsFile) }"

    rScript !
    val tolerance = 1e-5
    val results = readResults(resultsFile)
    assert(abs(skatStat - results(0)(0)) < tolerance)
    assert(abs(pValue - results(0)(1)) < tolerance)


  }

  @Test def randomizedTest() {
    val testsToAverageOver = 1

    println(header)
    println("Starting randomize skat test")
    println(header)

    println(header)
    println("Generating Input")
    println(header)
    val samples = 25
    val variants = 50
    val covariateCount = 20

    var (genotypes, covariates, phenotypes, weights) =
      randomizedInput(samples, variants, covariateCount)


    //write files to a location R script can read

    val inputFileG = tmpDir.createLocalTempFile("skatGMatrix", ".txt")
    //val inputFileG = "/Users/charlesc/Documents/Software/R/skatGMatrixNewFormat.txt"
    hadoopConf.writeTextFile(inputFileG) { _.write(largeMatrixToString(convert(genotypes, Double)," "))}

    val inputFileCov = tmpDir.createLocalTempFile("skatCovMatrix", ".txt")
    //val inputFileCov = "/Users/charlesc/Documents/Software/R/skatCovMatrixNewFormat.txt"
    hadoopConf.writeTextFile(inputFileCov) { _.write(largeMatrixToString(covariates," "))}

    val inputFilePheno = tmpDir.createLocalTempFile("skatPhenoVec", ".txt")
    //    val inputFilePheno = "/Users/charlesc/Documents/Software/R/skatPhenoNewFormat.txt"
    hadoopConf.writeTextFile(inputFilePheno)
    { _.write(largeMatrixToString((phenotypes).toDenseMatrix," "))}

    val inputFileW = tmpDir.createLocalTempFile("skatWeightVec",".txt")
    hadoopConf.writeTextFile(inputFileW)
    { _.write(largeMatrixToString((weights.toDenseMatrix)," "))}

    val resultsFile = tmpDir.createLocalTempFile("results", ".txt")


    println(header)
    println("Starting scala routines")
    println(header)
    val SKAT = new skat(convert(genotypes, Double), covariates,
                        phenotypes, weights)
    val (skatNullModel,firstTiming1) = time{SKAT.fitCovariates()}
    var t1 = 0.0.toLong
    for (i <- 1 to testsToAverageOver) {
      val (s, t) = time {
        SKAT.fitCovariates()
      }
      t1 += t
    }

    println(header)
    println("Finished Regression in %s".format(formatTime(t1/testsToAverageOver)))
    println("   speed of first run  %s".format(formatTime(firstTiming1)))
    println(header)
    var ((skatStat,pValue),firstTiming2) = time {skatNullModel.computeSkatStats()}
    var t2 = 0.0.toLong
    for (i <- 1 to testsToAverageOver){
      val ((s,p), t) = time {
        skatNullModel.computeSkatStats()
      }
      t2 += t
    }
    
    println(header)
    println("Finished computing SKAT stats in %s".format(formatTime(t2/testsToAverageOver)))
    println("             speed of first run  %s".format(formatTime(firstTiming2)))
    println(header)

    println(f"Variance component statistic: $skatStat")
    println(f"                     p value: $pValue")

    println(header)
    println("Starting R Routines")
    println(header)

    val rScript = s"Rscript src/test/resources/skatTest.R " +
      s"${ uriPath(inputFileG) } ${ uriPath(inputFileCov) } " +
      s"${ uriPath(inputFilePheno) } ${ uriPath(inputFileW) } " +
      s"${ uriPath(resultsFile) }"

    rScript !
    val tolerance = 1e-5
    val results = readResults(resultsFile)
    assert(abs(skatStat - results(0)(0)) < tolerance)
    assert(abs(pValue - results(0)(1)) < tolerance)

  }

}
