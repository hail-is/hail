package org.broadinstitute.hail.variant

import org.broadinstitute.hail.check.{Gen, Properties}
import org.broadinstitute.hail.check.Prop._
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

object GenotypeStreamSuite {

  object Spec extends Properties("GenotypeStream") {
    property("iterateBuild") = forAll(for (
      v <- Variant.gen;
      gs <- Gen.buildableOf[Iterable[Genotype], Genotype](Genotype.gen(v)))
    yield (v, gs)) { case (v: Variant, it: Iterable[Genotype]) =>
      val b = new GenotypeStreamBuilder(v)
      b ++= it
      val gs = b.result()
      val a2 = gs.toArray
      it.sameElements(a2)
    }
  }

}

class GenotypeStreamSuite extends TestNGSuite {

  import GenotypeStreamSuite._

  @Test def testGenotypeStream() {
    Spec.check()
  }
}
