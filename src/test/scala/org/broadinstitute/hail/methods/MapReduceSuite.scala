package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver._
import org.testng.annotations.Test

class MapReduceSuite extends SparkSuite {
  @Test def testVariants(): Unit = {

    val vds = LoadVCF(sc, "src/test/resources/sample2.vcf")
    var s = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])
    s = VariantQC.run(s, Array.empty[String])
    s = SampleQC.run(s, Array.empty[String])

    println("first run (do both):")
    s = MapReduce.run(s, Array("-c", "a.mafDist = variants.stats(va.qc.MAF), a.singStats = samples.stats(sa.qc.nSingleton)"))
    println("second run (no aggregates):")
    s = MapReduce.run(s, Array("-c", "a.mafDist.sumOver2 = a.mafDist.sum / 2"))
    println("third run (just variants):")
    s = MapReduce.run(s, Array("-c", "a.macDist = variants.stats(va.qc.MAC)"))
    println("fourth run:")
    s = MapReduce.run(s, Array("-c", "a.CRStats = samples.stats(sa.qc.callRate)"))

//    s = ShowGlobalAnnotations.run(s)
//    val sb = new StringBuilder
//    s.vds.taSignature.pretty(sb, 0, Vector("a"), 0)
//    println(sb.result())
    s = ShowAnnotations.run(s)
  }
}
