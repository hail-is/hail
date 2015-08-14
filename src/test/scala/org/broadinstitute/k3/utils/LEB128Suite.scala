package org.broadinstitute.k3.utils

import org.scalacheck.Properties
import org.scalacheck.Prop._
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

import scala.collection.mutable

import org.broadinstitute.k3.Utils._

object LEB128Suite {
  val b = new mutable.ArrayBuilder.ofByte

  def readWriteEqual(i: Int): Boolean = {
    b.clear()
    b.writeULEB128(i)
    i == b.result().iterator.readULEB128()
  }

  object Spec extends Properties("LEB128") {
    property("readWrite") = forAll { n: Int =>
      (n >= 0) ==> readWriteEqual(n) }
  }
}

class LEB128Suite extends TestNGSuite {
  import LEB128Suite._

  def testReadWrite(i: Int) {
    assert(readWriteEqual(i))
  }

  @Test def test() {
    testReadWrite(0)
    testReadWrite(0x7f)
    testReadWrite(0x80)
    testReadWrite(0xe5)
    (0 until 31).foreach(i =>
      testReadWrite(0x7eadbeef >>> i))

    Spec.check
  }
}
