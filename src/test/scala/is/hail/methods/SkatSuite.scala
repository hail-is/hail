package is.hail.methods

import is.hail.annotations.Annotation
import is.hail.io.annotators.IntervalList
import is.hail.{SparkSuite, TestUtils}
import is.hail.stats.RegressionUtils._
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant._
import is.hail.stats.vdsFromCallMatrix
import breeze.linalg._
import breeze.numerics.sigmoid
import is.hail.expr.types._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.testng.annotations.Test

import scala.sys.process._
import scala.language.postfixOps

case class SkatAggForR(xs: ArrayBuilder[DenseVector[Double]], weights: ArrayBuilder[Double])

class SkatSuite extends SparkSuite {

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
}
