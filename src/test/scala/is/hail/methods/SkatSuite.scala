package is.hail.methods

import is.hail.SparkSuite
import is.hail.TestUtils._
import is.hail.check.Prop._
import org.testng.annotations.Test
import breeze.linalg._
import java.io.File
import breeze.numerics._

class SkatSuite extends SparkSuite {

  @Test def test() {
    val header = "==============================================="

    println(header)
    println("Starting Skat tests")
    println(header)
    val save = true

    val G = DenseMatrix((0,1,1),(1,2,0),
                        (1,0,2),(2,2,1))
    val cov = DenseMatrix((.3,.2),(.5,.2),(.9,.3),(.2,.84))
    val w = DenseVector(1.0,4.0,9.0)
    val p = DenseVector(1.3,2.3,4.5,-.4)
    if (save)
      {
        csvwrite(new File("/Users/Jon/Desktop/CharlieTempfolder/skatGMatrix" +
         ".txt"), convert(G, Double), separator = ' ')
        csvwrite(new File("/Users/Jon/Desktop/CharlieTempfolder/skatCovMatrix" +
         ".txt"), cov, separator = ' ')
        csvwrite(new File("/Users/Jon/Desktop/CharlieTempfolder/skatWeightVec" +
         ".txt"), w.toDenseMatrix, separator = ' ')
        csvwrite(new File("/Users/Jon/Desktop/CharlieTempfolder/skatPhenoVec" +
         ".txt"), p.toDenseMatrix, separator = ' ')
      }

    val SKAT = new Skat(convert(G,Double),w,cov,p)
    //test failing VCS test
    interceptFatal("Regression hasn't run") {
      SKAT.VarianceComponentStatistic()
    }
    interceptFatal("VCS has not been computed") {
      SKAT.pvalue()
    }

    //non-failing
    val sol = SKAT.regressCovariates()
    val test = Some(4.2)
    println(test.get)
    println("Least Squares Solution")
    SKAT.parameterVector match{
      case Some(p) => println(p)
      case _ => println("something messed up")

    }

    val results = SKAT.VarianceComponentStatistic()
    println(results)
    SKAT.pvalue()



  }

}
