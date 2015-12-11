package org.broadinstitute.hail.variant

import org.broadinstitute.hail.ScalaCheckSuite
import org.scalacheck.Properties
import org.scalacheck.Prop._
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test
import org.broadinstitute.hail.Utils._

import scala.collection.mutable

object GenotypeSuite {
  val b = new mutable.ArrayBuilder.ofByte

  def readWriteEqual(g: Genotype): Boolean = {
    b.clear()
    g.write(b)
    g == Genotype.read(b.result().iterator)
  }

  object Spec extends Properties("Genotype") {
    property("readWrite") = forAll { g: Genotype =>
      readWriteEqual(g)
    }

    property("call") = forAll { g: Genotype =>
      g.call.isDefined == g.isCalled
    }
  }
}

class GenotypeSuite extends TestNGSuite with ScalaCheckSuite {

  import GenotypeSuite._

  def testReadWrite(g: Genotype) {
    assert(readWriteEqual(g))
  }

  @Test def testGenotype() {
    intercept[IllegalArgumentException] {
      Genotype(-2, (2, 0), 2, (0, 0, 0))
    }
    /*
    intercept[IllegalArgumentException] {
      Genotype(-1, (2, 0), 2, (0, 0, 0))
    }
    intercept[IllegalArgumentException] {
      Genotype(-1, (2, 0), 1, null)
    }
    intercept[IllegalArgumentException] {
      Genotype(1, (2, 0), 3, (0, 100, 1000))
    }
    */

    val noCall = Genotype(-1, (2, 0), 2, null)
    val homRef = Genotype(0, (10, 0), 10, (0, 1000, 100))
    val het = Genotype(1, (5, 5), 12, (100, 0, 1000))
    val homVar = Genotype(2, (2, 10), 12, (100, 1000, 0))

    assert(noCall.isNotCalled && !noCall.isCalled && !noCall.isHomRef && !noCall.isHet && !noCall.isHomVar)
    assert(!homRef.isNotCalled && homRef.isCalled && homRef.isHomRef && !homRef.isHet && !homRef.isHomVar)
    assert(!het.isNotCalled && het.isCalled && !het.isHomRef && het.isHet && !het.isHomVar)
    assert(!homVar.isNotCalled && homVar.isCalled && !homVar.isHomRef && !homVar.isHet && homVar.isHomVar)

    assert(noCall.call.isEmpty)
    assert(homRef.call.isDefined)
    assert(het.call.isDefined)
    assert(homVar.call.isDefined)

    testReadWrite(noCall)
    testReadWrite(homRef)
    testReadWrite(het)
    testReadWrite(homVar)

    assert(D_==(Genotype(-1, (0, 0), 0, null).pAB(), 1.0))
    assert(D_==(Genotype(1, (16, 16), 33, (100, 0, 100)).pAB(), 1.0))
    assert(D_==(Genotype(1, (5, 8), 13, (200, 0, 100)).pAB(), 0.423950))

    check(Spec)
  }
}
