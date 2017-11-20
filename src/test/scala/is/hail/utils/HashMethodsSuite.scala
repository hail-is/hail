package is.hail.utils

import is.hail.SparkSuite
import org.testng.annotations.Test

class HashMethodsSuite extends SparkSuite {
  @Test def testMultGF(): Unit = {
    val ph = new PolyHash
    import ph._

    val x1 = 3705006673L.toInt
    val y1 = 2551778209L.toInt
    assert(multGF(x1, y1) == 2272553040L.toInt)

    val x2 = 1791774536L.toInt
    val y2 = 201716548L.toInt
    assert(multGF(x2, y2) == 1195407259L.toInt)
  }

}
