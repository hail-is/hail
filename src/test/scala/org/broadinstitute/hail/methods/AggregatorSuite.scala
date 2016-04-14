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
    s = AnnotateVariants.run(s, Array("expr", "-c",
      "va.test.callrate = gs.fraction(g.isCalled), va.test.MAC = gs.stats(g.nNonRefAlleles).sum, " +
        "va.test.MAF = gs.stats(g.nNonRefAlleles).sum.toDouble / gs.count(g.isCalled) / 2.0, " +
        "va.test.gqstats = gs.stats(g.gq), va.test.gqhetstats = gs.statsif(g.isHet, g.gq)"))

    val qCallRate = s.vds.queryVA("va.test.callrate")._2
    val qCallRateQC = s.vds.queryVA("va.qc.callRate")._2
    val qMAC = s.vds.queryVA("va.test.MAC")._2
    val qMACQC = s.vds.queryVA("va.qc.MAC")._2
    val qMAF = s.vds.queryVA("va.test.MAF")._2
    val qMAFQC = s.vds.queryVA("va.qc.MAF")._2
    val gqStatsMean = s.vds.queryVA("va.test.gqstats.mean")._2
    val gqStatsMeanQC = s.vds.queryVA("va.qc.gqMean")._2
    val gqStatsStDev = s.vds.queryVA("va.test.gqstats.stdev")._2
    val gqStatsStDevQC = s.vds.queryVA("va.qc.gqStDev")._2
    val gqStatsHetMean = s.vds.queryVA("va.test.gqhetstats.mean")._2
    val gqStatsHetMeanQC = s.vds.queryVA("va.qc.gqMeanHet")._2
    val gqStatsHetStDev = s.vds.queryVA("va.test.gqhetstats.stdev")._2
    val gqStatsHetStDevQC = s.vds.queryVA("va.qc.gqStDevHet")._2

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

    s = AnnotateSamples.run(s, Array("expr", "-c", "sa.test.callrate = gs.fraction(g.isCalled), sa.test.gqstats = " +
      "gs.stats(g.gq), sa.test.gqhetstats = gs.statsif(g.isHet, g.gq)"))

    val qCallRate = s.vds.querySA("sa.test.callrate")._2
    val qCallRateQC = s.vds.querySA("sa.qc.callRate")._2
    val gqStatsMean = s.vds.querySA("sa.test.gqstats.mean")._2
    val gqStatsMeanQC = s.vds.querySA("sa.qc.gqMean")._2
    val gqStatsStDev = s.vds.querySA("sa.test.gqstats.stdev")._2
    val gqStatsStDevQC = s.vds.querySA("sa.qc.gqStDev")._2
    val gqStatsHetMean = s.vds.querySA("sa.test.gqhetstats.mean")._2
    val gqStatsHetMeanQC = s.vds.querySA("sa.qc.gqMeanHet")._2
    val gqStatsHetStDev = s.vds.querySA("sa.test.gqhetstats.stdev")._2
    val gqStatsHetStDevQC = s.vds.querySA("sa.qc.gqStDevHet")._2

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

//    s = AnnotateSamples.run(s, Array("expr", "-c", "sa.test.findmapped = gs.findmap(v.start == 16052684, g), " +
//      "sa.test.collected = gs.collect(g.isHet, g)"))
//    val qfm = s.vds.querySA("test", "findmapped")
//    val qcol = s.vds.querySA("test", "collected")
//    s.vds.sampleAnnotations
//      .foreach {
//        sa =>
//          println(qfm(sa))
//          println(qcol(sa))
//
//      }
  }
}
