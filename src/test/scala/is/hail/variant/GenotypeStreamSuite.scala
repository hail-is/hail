package is.hail.variant

import is.hail.check.{Gen, Properties}
import is.hail.check.Prop._
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

object GenotypeStreamSuite {

  import is.hail.utils._

  object Spec extends Properties("GenotypeStream") {

    property("iterateBuild") = forAll(for (
      v <- Variant.gen;
      gs <- Gen.buildableOf[Iterable, Genotype](Genotype.genExtreme(v.nAlleles)))
      yield (v, gs)) { case (v: Variant, it: Iterable[Genotype]) =>
      val b = new GenotypeStreamBuilder(v.nAlleles)
      b ++= it
      val gs = b.result()
      it.iterator.sameElements(gs.iterator) &&
//        gs.iterator.sameElements(gs.genericIterator) &&
        gs.iterator.map(_.unboxedGT).sameElements(gs.hardCallIterator)
    }
  }
}

class GenotypeStreamSuite extends TestNGSuite {

  import GenotypeStreamSuite._

  @Test def testGenotypeStream() {
    Spec.check()
  }
}
