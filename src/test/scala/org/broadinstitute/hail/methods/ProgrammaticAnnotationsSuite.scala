package org.broadinstitute.hail.methods

import org.apache.spark.sql.Row
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.driver._
import org.testng.annotations.Test

class ProgrammaticAnnotationsSuite extends SparkSuite {

  @Test def testSamples() {

    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
      .cache()

    var s = State(sc, sqlContext, vds)
    s = SplitMulti.run(s)
    s = SampleQC.run(s)
    s = AnnotateSamples.run(s, Array("-c", "sa.userdefined.missingness = (1 - sa.qc.callRate) * 100, sa.anotherthing = 5, sa.`hi there i have spaces`.another = true"))
    val qCall = s.vds.querySA("qc", "callRate")
    val qMiss = s.vds.querySA("userdefined", "missingness")
    val q3 = s.vds.querySA("anotherthing")
    val q4 = s.vds.querySA("hi there i have spaces", "another")
    val head = s.vds.sampleAnnotations.head
    println(qCall(head))
    println(qMiss(head))
    println(q3(head))
    println(q4(head))
  }

  @Test def testVariants() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
      .cache()

    var s = State(sc, sqlContext, vds)
    s = s.copy(vds = s.vds.filterVariants((v, va) => v.start == 10019093))
    s = SplitMulti.run(s)
    s = VariantQC.run(s)
    s = AnnotateVariants.run(s, Array("-c", "va.a.b = va.pass"))
    val vaa = s.vds.variantsAndAnnotations.collect()
    println(s.vds.vaSignature.dType)
//    println(s.vds.vaSignature.printSchema("va", 2, "va"))
    val q5 = s.vds.queryVA("qc", "MAC")
    println(s.vds.variantsAndAnnotations.collect().head._2.asInstanceOf[Row].get(8))
//    println(Annotation.printAnnotation(s.vds.variantsAndAnnotations.collect().head._2))
//    s.vds.variantsAndAnnotations.map(_._2).foreach(i => println("qc.mac is " + q5(i)))
//    s.vds.variantsAndAnnotations.map(_._2).foreach(i => println("qc.mac is " + q5(i)))

  }
}
