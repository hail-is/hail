package org.broadinstitute.k3.variant

import org.broadinstitute.k3.utils.ByteStream
import org.scalatest.Suite
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

import scala.collection.mutable

class GenotypeSuite extends TestNGSuite {
  @Test def testGenotype(): Unit = {
    val noCall = Genotype(-1, (2, 0), 2, null)
    val homRef = Genotype(0, (10, 0), 10, (0, 1000, 100))
    val het = Genotype(1, (5, 5), 12, (100, 0, 1000))
    val homVar = Genotype(2, (2, 10), 12, (100, 1000, 0))

    assert(noCall.isNotCalled)
    assert(homRef.isCalled && homRef.isHomRef)
    assert(het.isCalled && het.isHet)
    assert(homVar.isCalled && homVar.isHomVar)

    val b = new mutable.ArrayBuilder.ofByte
    noCall.write(b)
    assert(noCall === Genotype.read(new ByteStream(b.result())))

    b.clear()
    homRef.write(b)
    assert(homRef === Genotype.read(new ByteStream(b.result())))

    b.clear()
    het.write(b)
    assert(het === Genotype.read(new ByteStream(b.result())))

    b.clear()
    homVar.write(b)
    assert(homVar === Genotype.read(new ByteStream(b.result())))
  }
}
