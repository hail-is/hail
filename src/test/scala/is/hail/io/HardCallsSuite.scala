package is.hail.io

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.check.Prop._
import is.hail.utils._
import is.hail.variant.{VSMSubgen, VariantSampleMatrix, _}
import org.testng.annotations.Test

class HardCallsSuite extends SparkSuite {
  def gtTriples(vds: VariantDataset): Set[(Variant, Annotation, (Option[Int], Option[Boolean]))] =
    vds.expand()
      .map { case (v, s, g) => (v, s, (Genotype.gt(g), Genotype.fakeRef(g))) }
      .collect()
      .toSet

  @Test def test() {
    val p = forAll(VariantSampleMatrix.gen(hc, VSMSubgen.random).map(_.hardCalls())) { vds =>
      vds.rdd.forall { case (v, (va, gs)) =>
        gs.forall { g =>
          Genotype.ad(g).isEmpty &&
            Genotype.dp(g).isEmpty &&
            Genotype.gq(g).isEmpty &&
            Genotype.pl(g).isEmpty &&
            !Genotype.isLinearScale(g).getOrElse(false)
        }
      } && gtTriples(vds) == gtTriples(vds)
    }
    p.check()
  }
}
