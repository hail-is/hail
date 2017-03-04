package is.hail.variant

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class CallSuite extends TestNGSuite {
  @Test def testClass() {
    val cNoCall = Call(-1)
    assert(cNoCall.isNotCalled)
    val cHomRef = Call(0)
    assert(cHomRef.isHomRef && cHomRef.isCalled && cHomRef.nNonRefAlleles == Option(0))
    val cHet = Call(1)
    assert(cHet.isHet && cHet.isCalled && cHet.nNonRefAlleles == Option(1))
    val cHomVar = Call(2)
    assert(cHomVar.isHomVar && cHomVar.isCalled && cHomVar.nNonRefAlleles == Option(2))
  }
}
