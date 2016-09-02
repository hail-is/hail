package org.broadinstitute.hail.io

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant.{VSMSubgen, VariantSampleMatrix}
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.driver.{HardCalls, State}
import org.testng.annotations.Test

class HardCallsSuite extends SparkSuite {
  def gtTriples(vds: VariantDataset): Set[(Variant, String, (Option[Int], Boolean))] =
    vds.mapValues { g => (g.gt, g.fakeRef) }
      .expand()
      .collect()
      .toSet

  @Test def test() {
    val p = forAll(VariantSampleMatrix.gen(sc, VSMSubgen.random)) { vds =>
      var s = State(sc, sqlContext, vds)
      s = HardCalls.run(s)

      s.vds.rdd.forall { case (v, (va, gs)) =>
          gs.forall { g =>
            g.ad.isEmpty &&
            g.dp.isEmpty &&
            g.gq.isEmpty &&
            g.pl.isEmpty &&
            !g.isDosage
          }
      } && gtTriples(vds) == gtTriples(s.vds)
    }
    p.check()
  }
}
