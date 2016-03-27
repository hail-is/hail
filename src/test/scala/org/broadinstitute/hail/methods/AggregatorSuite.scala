package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.Utils._
import org.testng.annotations.Test

class AggregatorSuite extends SparkSuite {

  @Test def testRows() {
    val vds = LoadVCF(sc, "src/test/resources/sample2.vcf")
    var s = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])
    s = VariantQC.run(s, Array.empty[String])
    s = AnnotateVariants.run(s, Array("-c", "va.test.callrate = gs.fraction(g.isCalled), va.test.MAC = gs.stats(g.nNonRefAlleles).sum, va.test.MAF = gs.stats(g.nNonRefAlleles).sum.toDouble / gs.count(g.isCalled) / 2.0, va.test.gqstats = gs.stats(g.gq), va.test.gqhetstats = gs.statsif(g.isHet, g.gq)"))

    val qCallRate = s.vds.queryVA("test", "callrate")
    val qCallRateQC = s.vds.queryVA("qc", "callRate")
    val qMAC = s.vds.queryVA("test", "MAC")
    val qMACQC = s.vds.queryVA("qc", "MAC")
    val qMAF = s.vds.queryVA("test", "MAF")
    val qMAFQC = s.vds.queryVA("qc", "MAF")
    val gqStatsMean = s.vds.queryVA("test", "gqstats", "mean")
    val gqStatsMeanQC = s.vds.queryVA("qc", "gqMean")
    val gqStatsStDev = s.vds.queryVA("test", "gqstats", "stdev")
    val gqStatsStDevQC = s.vds.queryVA("qc", "gqStDev")
    val gqStatsHetMean = s.vds.queryVA("test", "gqhetstats", "mean")
    val gqStatsHetMeanQC = s.vds.queryVA("qc", "gqMeanHet")
    val gqStatsHetStDev = s.vds.queryVA("test", "gqhetstats", "stdev")
    val gqStatsHetStDevQC = s.vds.queryVA("qc", "gqStDevHet")

    s.vds.rdd.collect()
      .foreach {
        case (v, va, gs) =>
          assert(qCallRate(va) == qCallRateQC(va))
          assert(qMAC(va) == qMACQC(va))
          assert(qMAF(va) == qMAFQC(va))
          assert(gqStatsMean(va).zip(gqStatsMeanQC(va)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
          assert(gqStatsStDev(va).zip(gqStatsStDevQC(va)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
          assert(gqStatsHetMean(va).zip(gqStatsHetMeanQC(va)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
          assert(gqStatsHetStDev(va).zip(gqStatsHetStDevQC(va)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })

      }
  }

  @Test def testColumns() {
    val vds = LoadVCF(sc, "src/test/resources/sample2.vcf")
    var s = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])

    s = SampleQC.run(s, Array.empty[String])

    s = AnnotateSamples.run(s, Array("-c", "sa.test.callrate = gs.fraction(g.isCalled), sa.test.gqstats = " +
      "gs.stats(g.gq), sa.test.gqhetstats = gs.statsif(g.isHet, g.gq)"))

    val qCallRate = s.vds.querySA("test", "callrate")
    val qCallRateQC = s.vds.querySA("qc", "callRate")
    val gqStatsMean = s.vds.querySA("test", "gqstats", "mean")
    val gqStatsMeanQC = s.vds.querySA("qc", "gqMean")
    val gqStatsStDev = s.vds.querySA("test", "gqstats", "stdev")
    val gqStatsStDevQC = s.vds.querySA("qc", "gqStDev")
    val gqStatsHetMean = s.vds.querySA("test", "gqhetstats", "mean")
    val gqStatsHetMeanQC = s.vds.querySA("qc", "gqMeanHet")
    val gqStatsHetStDev = s.vds.querySA("test", "gqhetstats", "stdev")
    val gqStatsHetStDevQC = s.vds.querySA("qc", "gqStDevHet")

    s.vds.sampleAnnotations
      .foreach {
        sa =>
          assert(qCallRate(sa) == qCallRateQC(sa))
          assert(gqStatsMean(sa).zip(gqStatsMeanQC(sa)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
          assert(gqStatsStDev(sa).zip(gqStatsStDevQC(sa)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
          assert(gqStatsHetMean(sa).zip(gqStatsHetMeanQC(sa)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
          assert(gqStatsHetStDev(sa).zip(gqStatsHetStDevQC(sa)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
      }
  }
}
