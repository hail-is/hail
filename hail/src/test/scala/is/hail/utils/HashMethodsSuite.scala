package is.hail.utils

import is.hail.HailSuite

import org.testng.annotations.Test

class HashMethodsSuite extends HailSuite {
  @Test def testMultGF() {
    import PolyHash._

    // multGF should agree with the Mathematica function
    /* f[a_, b_] := PolynomialRemainder[g[a, b], x^32 + x^7 + x^3 + x^2 + 1, x, Modulus -> 2] /. x
     * -> 2 */
    // where
    /* g[a_, b_] := Expand[FromDigits[IntegerDigits[a, 2], x] * FromDigits[IntegerDigits[b, 2], x],
     * Modulus -> 2] */
    val x1 = 3705006673L.toInt
    val y1 = 2551778209L.toInt
    assert(multGF(x1, y1) == 2272553040L.toInt)

    val x2 = 1791774536L.toInt
    val y2 = 201716548L.toInt
    assert(multGF(x2, y2) == 1195407259L.toInt)
  }
}
