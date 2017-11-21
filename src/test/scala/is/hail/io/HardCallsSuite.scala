package is.hail.io

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.check.Prop._
import is.hail.utils._
import is.hail.variant.{VSMSubgen, VariantSampleMatrix, _}
import org.testng.annotations.Test

class HardCallsSuite extends SparkSuite {
  def gtTriples(vds: VariantDataset): Set[(Annotation, Annotation, Option[Int])] =
    vds.expand()
      .map { case (v, va, g) => (v, va, Genotype.gt(g.asInstanceOf[Genotype])) }
      .collect()
      .toSet

  @Test def test() {
    val p = forAll(VariantSampleMatrix.gen(hc, VSMSubgen.random).map(_.hardCalls())) { vds =>
      vds.rdd.forall { case (v, (va, gs)) =>
        gs.forall { g1 =>
          val g = g1.asInstanceOf[Genotype]
          Genotype.ad(g).isEmpty &&
            Genotype.dp(g).isEmpty &&
            Genotype.gq(g).isEmpty &&
            Genotype.pl(g).isEmpty
        }
      } && gtTriples(vds) == gtTriples(vds)
    }
    p.check()
  }
}
