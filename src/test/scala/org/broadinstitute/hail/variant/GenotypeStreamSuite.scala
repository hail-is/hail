package org.broadinstitute.hail.variant

import org.broadinstitute.hail.PropertySuite
import org.broadinstitute.hail.check.Gen
import org.broadinstitute.hail.check.Prop._

class GenotypeStreamProperties extends PropertySuite {

  property("iterateBuild") = forAll(for (
    v <- Variant.gen;
    gs <- Gen.buildableOf[Iterable, Genotype](Genotype.genExtreme(v.nAlleles)))
    yield (v, gs)) { case (v, it) =>
    val b = new GenotypeStreamBuilder(v.nAlleles)
    b ++= it
    val gs = b.result()
    val a2 = gs.toArray
    it.sameElements(a2)
  }
}
