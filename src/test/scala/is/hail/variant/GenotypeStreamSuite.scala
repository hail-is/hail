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
      val a1 = gs.toArray
      val a2 = gs.hardCallIterator.toArray
      it.sameElements(a1) && a1.map(_.unboxedGT).sameElements(a2)
    }
  }
}

class GenotypeStreamSuite extends TestNGSuite {

  import GenotypeStreamSuite._

  @Test def testGenotypeStream() {
    Spec.check()
  }
}
