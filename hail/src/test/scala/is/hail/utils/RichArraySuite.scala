package is.hail.utils

import is.hail.utils.richUtils.RichArray
import is.hail.{SparkSuite, TestUtils}
import org.testng.annotations.Test

class RichArraySuite extends SparkSuite {
  @Test def testArrayImpex() {
    val file = tmpDir.createTempFile("test")
    val a = Array.fill[Double](100)(util.Random.nextDouble())
    val a2 = new Array[Double](100)
    
    RichArray.exportToDoubles(hc, file, a, bufSize = 32)
    RichArray.importFromDoubles(hc, file, a2, bufSize = 16)
    assert(a === a2)
        
    TestUtils.interceptFatal("Premature") {
      RichArray.importFromDoubles(hc, file, new Array[Double](101), bufSize = 64)
    }
  }
}
