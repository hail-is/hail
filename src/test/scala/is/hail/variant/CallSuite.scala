package is.hail.variant

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test
import is.hail.variant.Call._

class CallSuite extends TestNGSuite {
  @Test def testClass() {
    val cNoCall = -1
    assert(isNotCalled(cNoCall))
    val cHomRef = 0
    assert(isHomRef(cHomRef) && isCalled(cHomRef) && nNonRefAlleles(cHomRef) == Option(0))
    val cHet = 1
    assert(isHet(cHet) && isCalled(cHet) && nNonRefAlleles(cHet) == Option(1))
    val cHomVar = 2
    assert(isHomVar(cHomVar) && isCalled(cHomVar) && nNonRefAlleles(cHomVar) == Option(2))
  }
}
