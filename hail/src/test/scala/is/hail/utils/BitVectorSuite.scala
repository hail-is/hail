package is.hail.utils

import is.hail.check._
import is.hail.check.Prop._

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class BitVectorSuite extends TestNGSuite {

  @Test def test() {
    val bv0 = new BitVector(0)
    assert(bv0.length == 0)

    intercept[ArrayIndexOutOfBoundsException] {
      bv0(-1)
    }

    intercept[ArrayIndexOutOfBoundsException] {
      bv0(0)
    }

    val g =
      for {
        n <- Gen.choose(1, 1000)
        s <- Gen.buildableOf[Set](Gen.choose(0, n - 1))
      } yield {
        val bv = new BitVector(n)

        assert(bv.length == n)

        intercept[ArrayIndexOutOfBoundsException] {
          bv(-1)
        }

        intercept[ArrayIndexOutOfBoundsException] {
          bv(n)
        }

        // check empty
        for (i <- 0 until n)
          assert(!bv(i))

        for (i <- s)
          bv.set(i)

        for (i <- 0 until n)
          assert(bv(i) == s(i))

        // test clear
        bv.clear()
        for (i <- 0 until n)
          assert(!bv(i))

        // fill again and test reset
        for (i <- s)
          bv.set(i)
        for (i <- s)
          bv.reset(i)

        for (i <- 0 until n)
          assert(!bv(i))

        ()
      }
    val p = forAll(g)(_ => true)
    p.check()
  }
}
