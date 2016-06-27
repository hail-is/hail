package org.broadinstitute.hail.methods

import org.apache.spark.util.StatCounter
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
      "va.test.callrate = gs.fraction(g.isCalled), va.test.AC = gs.stats(g.nNonRefAlleles).sum, " +
        "va.test.AF = gs.stats(g.nNonRefAlleles).sum.toDouble / gs.count(g.isCalled) / 2.0, " +
        "va.test.gqstats = gs.stats(g.gq), va.test.gqhetstats = gs.statsif(g.isHet, g.gq)"))

    val qCallRate = s.vds.queryVA("va.test.callrate")._2
    val qCallRateQC = s.vds.queryVA("va.qc.callRate")._2
    val qAC = s.vds.queryVA("va.test.AC")._2
    val qACQC = s.vds.queryVA("va.qc.AC")._2
    val qAF = s.vds.queryVA("va.test.AF")._2
    val qAFQC = s.vds.queryVA("va.qc.AF")._2
    val gqStatsMean = s.vds.queryVA("va.test.gqstats.mean")._2
    val gqStatsMeanQC = s.vds.queryVA("va.qc.gqMean")._2
    val gqStatsStDev = s.vds.queryVA("va.test.gqstats.stdev")._2
    val gqStatsStDevQC = s.vds.queryVA("va.qc.gqStDev")._2
    val gqStatsHetMean = s.vds.queryVA("va.test.gqhetstats.mean")._2
    val gqStatsHetStDev = s.vds.queryVA("va.test.gqhetstats.stdev")._2

    s.vds.rdd.collect()
      .foreach {
        case (v, va, gs) =>
          assert(qCallRate(va) == qCallRateQC(va))
          assert(qAC(va) == qACQC(va))
          assert(qAF(va) == qAFQC(va))
          assert(gqStatsMean(va).zip(gqStatsMeanQC(va)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
          assert(gqStatsStDev(va).zip(gqStatsStDevQC(va)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })

          val gqSC = gs.aggregate(new StatCounter())({ case (s, g) =>
            if (g.isHet)
              g.gq.foreach(x => s.merge(x))
            s
          }, { case (s1, s2) => s1.merge(s2) })

          assert(gqStatsHetMean(va).zip(Option(gqSC.mean)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
          assert(gqStatsHetStDev(va).zip(Option(gqSC.stdev)).forall {
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
    val gqStatsHetStDev = s.vds.querySA("sa.test.gqhetstats.stdev")._2


    val gqHetMap = s.vds.aggregateBySample(new StatCounter())({ case (s, g) =>
      if (g.isHet)
        g.gq.foreach(x => s.merge(x))
      s
    }, { case (s1, s2) => s1.merge(s2) })
      .collect().toMap

    s.vds.sampleIdsAndAnnotations
      .foreach {
        case (s, sa) =>
          assert(qCallRate(sa) == qCallRateQC(sa))
          assert(gqStatsMean(sa).zip(gqStatsMeanQC(sa)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
          assert(gqStatsStDev(sa).zip(gqStatsStDevQC(sa)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
          assert(gqStatsHetMean(sa).zip(gqHetMap.get(s).map(_.mean)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
          assert(gqStatsHetStDev(sa).zip(gqHetMap.get(s).map(_.stdev)).forall {
            case (a, b) => D_==(a.asInstanceOf[Double], b.asInstanceOf[Double])
          })
      }
  }
}
