package org.broadinstitute.k3.variant

import org.scalacheck.{Prop, Properties}
import org.scalacheck.Prop._
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

import scala.collection.mutable

object GenotypeSuite {
  val b = new mutable.ArrayBuilder.ofByte

  def readWriteEqual(g: Genotype): Boolean = {
    b.clear()
    g.write(b)
    g == Genotype.read(b.result().iterator)
  }

  import Genotype.arbGenotype
  object Spec extends Properties("Genotype") {
    property("readWrite") = forAll[Genotype, Boolean] { g: Genotype =>
      readWriteEqual(g)
    }
  }
}

class GenotypeSuite extends TestNGSuite {

  import GenotypeSuite._

  def testReadWrite(g: Genotype) {
    assert(readWriteEqual(g))
  }

  @Test def testGenotype(): Unit = {
    Spec.check

    val noCall = Genotype(-1, (2, 0), 2, null)
    val homRef = Genotype(0, (10, 0), 10, (0, 1000, 100))
    val het = Genotype(1, (5, 5), 12, (100, 0, 1000))
    val homVar = Genotype(2, (2, 10), 12, (100, 1000, 0))

    assert(noCall.notCalled && !noCall.called && !noCall.isHomRef && !noCall.isHet && !noCall.isHomVar)
    assert(!homRef.notCalled && homRef.called && homRef.isHomRef && !homRef.isHet && !homRef.isHomVar)
    assert(!het.notCalled && het.called && !het.isHomRef && het.isHet && !het.isHomVar)
    assert(!homVar.notCalled && homVar.called && !homVar.isHomRef && !homVar.isHet && homVar.isHomVar)

    testReadWrite(noCall)
    testReadWrite(homRef)
    testReadWrite(het)
    testReadWrite(homVar)
  }
}
