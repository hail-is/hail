package is.hail.io

import is.hail.SparkSuite
import is.hail.check.Prop._
import is.hail.utils._
import is.hail.variant.{VSMSubgen, VariantSampleMatrix, _}
import org.testng.annotations.Test

class HardCallsSuite extends SparkSuite {
  def gtTriples(vds: VariantDataset): Set[(Variant, String, (Option[Int], Boolean))] =
    vds.mapValues { g => (g.gt, g.fakeRef) }
      .expand()
      .collect()
      .toSet

  @Test def test() {
    val p = forAll(VariantSampleMatrix.gen(hc, VSMSubgen.random).map(_.hardCalls())) { vds =>
      vds.rdd.forall { case (v, (va, gs)) =>
        gs.forall { g =>
          g.ad.isEmpty &&
            g.dp.isEmpty &&
            g.gq.isEmpty &&
            g.pl.isEmpty &&
            !g.isLinearScale
        }
      } && gtTriples(vds) == gtTriples(vds)
    }
    p.check()
  }
}
