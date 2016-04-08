package org.broadinstitute.hail.methods

import org.apache.spark.util.StatCounter
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.testng.annotations.Test


class MapReduceSuite extends SparkSuite {
  @Test def testVariants(): Unit = {

    var s = SplitMulti.run(State(sc, sqlContext, LoadVCF(sc, "src/test/resources/sample2.vcf")), Array.empty[String])
    s = VariantQC.run(s, Array.empty[String])
    s = SampleQC.run(s, Array.empty[String])

    s = MapReduce.run(s, Array("-c", "global.mafDist = variants.stats(va.qc.MAF), global.singStats = samples.count(sa.qc.nSingleton > 2)"))
    s = MapReduce.run(s, Array("-c", "global.anotherAnnotation.sumOver2 = global.mafDist.sum / 2"))
    s = MapReduce.run(s, Array("-c", "global.macDist = variants.stats(va.qc.MAC)"))
    s = MapReduce.run(s, Array("-c", "global.CRStats = samples.stats(sa.qc.callRate)"))


    val vds = s.vds
    val qSingleton = vds.querySA("qc", "nSingleton")
    val qSingletonGlobal = vds.queryGlobal("singStats")

    val sCount = vds.sampleAnnotations.count(sa => {
      qSingleton(sa).exists(_.asInstanceOf[Int] > 2)
    })

    assert(qSingletonGlobal(vds.globalAnnotation).contains(sCount))

    val qMaf = vds.queryVA("qc", "MAF")
    val mafSC = vds.variantsAndAnnotations.map(_._2)
        .aggregate(new StatCounter())({case (statC, va) =>
        val maf = qMaf(va)
        maf.foreach(o => statC.merge(o.asInstanceOf[Double]))
        statC
        }, { case (sc1, sc2) => sc1.merge(sc2)})

    assert(vds.queryGlobal("mafDist")(vds.globalAnnotation)
      .contains(Annotation(mafSC.mean, mafSC.stdev, mafSC.min, mafSC.max, mafSC.count, mafSC.sum)))

    assert(vds.queryGlobal("anotherAnnotation", "sumOver2")(vds.globalAnnotation).contains(mafSC.sum / 2))

    val qMac = vds.queryVA("qc", "MAC")
    val macSC = vds.variantsAndAnnotations.map(_._2)
      .aggregate(new StatCounter())({case (statC, va) =>
        val mac = qMac(va)
        mac.foreach(o => statC.merge(o.asInstanceOf[Int]))
        statC
      }, { case (sc1, sc2) => sc1.merge(sc2)})

    assert(vds.queryGlobal("macDist")(vds.globalAnnotation)
      .contains(Annotation(macSC.mean, macSC.stdev, macSC.min.toInt,
        macSC.max.toInt, macSC.count, macSC.sum.round.toInt)))

    val qCR = vds.querySA("qc", "callRate")
    val crSC = vds.sampleAnnotations
      .aggregate(new StatCounter())({case (statC, sa) =>
        val cr = qCR(sa)
        cr.foreach(o => statC.merge(o.asInstanceOf[Double]))
        statC
      }, { case (sc1, sc2) => sc1.merge(sc2)})

    assert(vds.queryGlobal("CRStats")(vds.globalAnnotation)
      .contains(Annotation(crSC.mean, crSC.stdev, crSC.min,
        crSC.max, crSC.count, crSC.sum)))

  }
}
