package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver._
import org.testng.annotations.Test

class ProgrammaticAnnotationsSuite extends SparkSuite {

  @Test def testSamples() {

    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
      .cache()

    var s = State(sc, sqlContext, vds)
    s = SplitMulti.run(s)
    s = SampleQC.run(s)
    s = AnnotateSamples.run(s, Array("expr", "-c", "sa.userdefined.missingness = (1 - sa.qc.callRate) * 100, sa.anotherthing = 5, sa.`hi there i have spaces`.another = true"))
    val qCall = s.vds.querySA("qc", "callRate")
    val qMiss = s.vds.querySA("userdefined", "missingness")
    val q3 = s.vds.querySA("anotherthing")
    val q4 = s.vds.querySA("hi there i have spaces", "another")
    val head = s.vds.sampleAnnotations.head
    s.vds.sampleAnnotations.foreach { sa =>
      qCall(sa) == qMiss(sa).map(x => (1 - x.asInstanceOf[Double]) * 100) &&
      q3(sa) == Some(5) &&
      q4(sa) == Some(true)
    }
  }

  @Test def testVariants() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
      .cache()

    var s = State(sc, sqlContext, vds)
    s = s.copy(vds = s.vds.filterVariants((v, va, gs) => v.start == 10019093))
    s = SplitMulti.run(s)
    s = VariantQC.run(s)
    s = AnnotateVariants.run(s, Array("expr", "-c", "va.a.b.c.d.e = va.qc.callRate * 100, va.a.c = if (va.pass) 1 else 0, va.`weird spaces name` = 5 / (va.qual - 5)"))
    val vaa = s.vds.variantsAndAnnotations.collect()
    val q = s.vds.queryVA("a", "b", "c", "d", "e")
    val q2 = s.vds.queryVA("a", "c")
    val q3 = s.vds.queryVA("weird spaces name")
    val qCallRate = s.vds.queryVA("qc", "callRate")
    val qPass = s.vds.queryVA("pass")
    val qQual = s.vds.queryVA("qual")
    s.vds.variantsAndAnnotations
      .collect()
      .foreach { case (v, va) =>
        assert(q(va) == qCallRate(va).map(_.asInstanceOf[Double] * 100) &&
          q2(va) == qPass(va).map(_.asInstanceOf[Boolean] match {
            case true => 1
            case false => 0
          }) &&
          q3(va) == qQual(va).map(x => 5 / (x.asInstanceOf[Double] - 5)))
      }
  }
}
