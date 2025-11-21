package is.hail.methods

import is.hail.HailSuite
import is.hail.utils._

import breeze.linalg._
import org.testng.annotations.Test

class SkatSuite extends HailSuite {

  @Test def smallNLargeNEqualityTest(): Unit = {
    val rand = scala.util.Random
    rand.setSeed(0)

    val n = 10 // samples
    val m = 5 // variants
    val k = 3 // covariates

    val st = Array.tabulate(m) { _ =>
      SkatTuple(
        rand.nextDouble(),
        DenseVector(Array.fill(n)(rand.nextDouble())),
        DenseVector(Array.fill(k)(rand.nextDouble())),
      )
    }

    val (qSmall, gramianSmall) = Skat.computeGramianSmallN(st)
    val (qLarge, gramianLarge) = Skat.computeGramianLargeN(st)

    assert(D_==(qSmall, qLarge))
    assertMatrixEqualityDouble(gramianSmall, gramianLarge)
  }
}
