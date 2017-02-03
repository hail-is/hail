package is.hail.methods

import is.hail.SparkSuite
import is.hail.driver._
import is.hail.utils._
import is.hail.io.vcf.LoadVCF
import org.testng.annotations.Test

class ProgrammaticAnnotationsSuite extends SparkSuite {

  @Test def testSamples() {

    val vds = hc.importVCF("src/test/resources/sample.vcf")
      .cache()
      .splitMulti()
      .sampleQC()
      .annotateSamplesExpr(
        "sa.userdefined.missingness = (1 - sa.qc.callRate) * 100, sa.anotherthing = 5, " +
          "sa.`hi there i have spaces`.another = true")

    val qCall = vds.querySA("sa.qc.callRate")._2
    val qMiss = vds.querySA("sa.userdefined.missingness")._2
    val q3 = vds.querySA("sa.anotherthing")._2
    val q4 = vds.querySA("sa.`hi there i have spaces`.another")._2
    val head = vds.sampleAnnotations.head
    vds.sampleAnnotations.foreach { sa =>
      assert(D_==(qCall(sa).get.asInstanceOf[Double], qMiss(sa).map(x => (100 - x.asInstanceOf[Double]) / 100).get) &&
        q3(sa).contains(5) &&
        q4(sa).contains(true))
    }
  }

  @Test def testVariants() {
    val vds = hc.importVCF("src/test/resources/sample.vcf")
      .cache()
      .filterVariantsExpr("v.start == 10019093")
      .splitMulti()
      .variantQC()
      .annotateVariantsExpr(
        "va.a.b.c.d.e = va.qc.callRate * 100, va.a.c = if (va.pass) 1 else 0, va.`weird spaces name` = 5 / (va.qual - 5)")

    val vaa = vds.variantsAndAnnotations.collect()
    val q = vds.queryVA("va.a.b.c.d.e")._2
    val q2 = vds.queryVA("va.a.c")._2
    val q3 = vds.queryVA("va.`weird spaces name`")._2
    val qCallRate = vds.queryVA("va.qc.callRate")._2
    val qPass = vds.queryVA("va.pass")._2
    val qQual = vds.queryVA("va.qual")._2
    vds.variantsAndAnnotations
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
