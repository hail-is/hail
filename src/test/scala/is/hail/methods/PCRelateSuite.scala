package is.hail.methods

import is.hail.SparkSuite
import org.testng.annotations.Test
import is.hail.check._
import is.hail.check.Prop._
import is.hail.variant.VariantDataset
import is.hail.variant.VSMSubgen
import is.hail.stats._

class PCRelateSuite extends SparkSuite {
  @Test def noErrors() {
    val vds: VariantDataset = BaldingNicholsModel(hc, 3, 50, 10000, None, None, 0, None, BetaDist(0.5,0.5)).splitMulti()
    val pcs = SamplePCA.justScores(vds, 2)
    println(PCRelate.pcRelate(vds, pcs).phiHat.bm.blocks.take(1).toSeq)
  }

  // @Test def foo() {
  //   val vds: VariantDataset = BaldingNicholsModel(hc, 5, 1000, 10000, None, None, 0, None, UniformDist(0.0,1.0)).splitMulti()
  //   val pcs = PCRelate.MMatrix.from(vds.sparkContext, SamplePCA.justScores(vds, 10))
  //   println(PCRelate.pcRelate(vds, pcs).phiHat.m.bm.toLocalMatrix().toArray)
  // }
}
