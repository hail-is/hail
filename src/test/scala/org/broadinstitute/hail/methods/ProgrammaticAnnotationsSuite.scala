package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.io.vcf.LoadVCF
import org.testng.annotations.Test

class ProgrammaticAnnotationsSuite extends SparkSuite {

  @Test def testSamples() {

    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
      .cache()

    var s = State(sc, sqlContext, vds)
    s = SplitMulti.run(s)
    s = SampleQC.run(s)
    s = AnnotateSamples.run(s, Array("expr", "-c", "sa.userdefined.missingness = (1 - sa.qc.callRate) * 100, sa.anotherthing = 5, sa.`hi there i have spaces`.another = true"))
    val qCall = s.vds.querySA("sa.qc.callRate")._2
    val qMiss = s.vds.querySA("sa.userdefined.missingness")._2
    val q3 = s.vds.querySA("sa.anotherthing")._2
    val q4 = s.vds.querySA("sa.`hi there i have spaces`.another")._2
    val head = s.vds.sampleAnnotations.head
    s.vds.sampleAnnotations.foreach { sa =>
      assert(D_==(qCall(sa).get.asInstanceOf[Double], qMiss(sa).map(x => (100 - x.asInstanceOf[Double]) / 100).get) &&
      q3(sa) == Some(5) &&
      q4(sa) == Some(true))
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
    val q = s.vds.queryVA("va.a.b.c.d.e")._2
    val q2 = s.vds.queryVA("va.a.c")._2
    val q3 = s.vds.queryVA("va.`weird spaces name`")._2
    val qCallRate = s.vds.queryVA("va.qc.callRate")._2
    val qPass = s.vds.queryVA("va.pass")._2
    val qQual = s.vds.queryVA("va.qual")._2
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
