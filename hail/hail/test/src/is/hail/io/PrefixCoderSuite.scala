package is.hail.io

import is.hail.HailSuite

import org.scalatestplus.scalacheck.CheckerAsserting.assertingNatureOfAssertion
import org.scalatestplus.scalacheck.ScalaCheckDrivenPropertyChecks
import org.testng.annotations.Test

class PrefixCoderSuite extends HailSuite with ScalaCheckDrivenPropertyChecks {
  def helper(v: Any): Array[Byte] = {
    val coder = new PrefixCoder
    v match {
      case v: Boolean => coder.encodeBool(v)
      case v: Int => coder.encodeInt(v)
      case v: Long => coder.encodeLong(v)
      case v: Float => coder.encodeFloat(v)
      case v: Double => coder.encodeDouble(v)
      case v: String => coder.writeBytes(v.getBytes())
    }
    coder.toByteArray()
  }

  @Test def testIntCoding(): Unit = forAll { (a: Int, b: Int) =>
    val abytes = helper(a)
    val bbytes = helper(b)

    val icmp = java.lang.Integer.signum(java.lang.Integer.compare(a, b))
    val acmp = java.lang.Integer.signum(java.util.Arrays.compareUnsigned(abytes, bbytes))

    assert(icmp == acmp)
  }

  @Test def testLongCoding(): Unit =
    forAll { (a: Long, b: Long) =>
      val abytes = helper(a)
      val bbytes = helper(b)

      val icmp = java.lang.Integer.signum(java.lang.Long.compare(a, b))
      val acmp = java.lang.Integer.signum(java.util.Arrays.compareUnsigned(abytes, bbytes))

      assert(icmp == acmp)
    }

  @Test def testFloatCoding(): Unit =
    forAll { (a: Float, b: Float) =>
      val abytes = helper(a)
      val bbytes = helper(b)

      val icmp = java.lang.Integer.signum(java.lang.Float.compare(a, b))
      val acmp = java.lang.Integer.signum(java.util.Arrays.compareUnsigned(abytes, bbytes))

      assert(icmp == acmp)
    }

  @Test def testDoubleCoding(): Unit =
    forAll { (a: Double, b: Double) =>
      val abytes = helper(a)
      val bbytes = helper(b)

      val icmp = java.lang.Integer.signum(java.lang.Double.compare(a, b))
      val acmp = java.lang.Integer.signum(java.util.Arrays.compareUnsigned(abytes, bbytes))

      assert(icmp == acmp)
    }

  @Test def testStringCoding(): Unit =
    forAll { (a: String, b: String) =>
      val abytes = helper(a)
      val bbytes = helper(b)

      val icmp = java.lang.Integer.signum(a.compareTo(b))
      val acmp = java.lang.Integer.signum(java.util.Arrays.compareUnsigned(abytes, bbytes))

      assert(icmp == acmp)
    }
}
