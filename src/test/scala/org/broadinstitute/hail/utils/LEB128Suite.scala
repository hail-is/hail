package org.broadinstitute.hail.utils

import org.scalacheck.Properties
import org.scalacheck.Prop._
import org.scalacheck
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

import scala.collection.mutable

import org.broadinstitute.hail.Utils._

object LEB128Suite {
  val b = new mutable.ArrayBuilder.ofByte

  def ulebReadWriteEqual(i: Int): Boolean = {
    b.clear()
    b.writeULEB128(i)
    i == b.result().iterator.readULEB128()
  }

  def slebReadWriteEqual(i: Int): Boolean = {
    b.clear()
    b.writeSLEB128(i)
    i == b.result().iterator.readSLEB128()
  }

  object Spec extends Properties("LEB128") {
    property("readWrite") = forAll { n: Int =>
      (n >= 0) ==> ulebReadWriteEqual(n)
    }

    property("readWrite") = forAll { n: Int =>
      slebReadWriteEqual(n)
    }
  }

}

class LEB128Suite extends TestNGSuite {

  import LEB128Suite._

  def testReadWrite(i: Int) {
    assert(ulebReadWriteEqual(i))
    assert(slebReadWriteEqual(i))
  }

  def testSLEBReadWrite(i: Int) {
    assert(slebReadWriteEqual(i))
  }

  // FIXME add to ScalaCheckSuite
  def check(props: Properties) {
    assert(scalacheck.Test.check(scalacheck.Test.Parameters.default, props).status == scalacheck.Test.Passed)
  }

  @Test def test() {
    testReadWrite(0)
    testReadWrite(0x7f)
    testReadWrite(0x80)
    testReadWrite(0xe5)
    (0 until 31).foreach(i =>
      testReadWrite(0x7eadbeef >>> i))

    testSLEBReadWrite(-1)
    testSLEBReadWrite(-129)
    (0 until 31).foreach(i =>
      testSLEBReadWrite(0xdeadbeef >>> i))

    check(Spec)
  }
}
