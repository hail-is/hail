package is.hail.methods

import is.hail.SparkSuite
import is.hail.utils._
import is.hail.testUtils._
import org.testng.annotations.Test

class ProgrammaticAnnotationsSuite extends SparkSuite {

  @Test def testSamples() {
    var vds = hc.importVCF("src/test/resources/sample.vcf")
    vds = SplitMulti(vds)
    vds = SampleQC(vds)
    vds = vds
      .annotateSamplesExpr(
        "sa.userdefined.missingness = (1 - sa.qc.callRate) * 100, sa.anotherthing = 5, " +
          "sa.`hi there i have spaces`.another = true")
      .cache()

    val qCall = vds.querySA("sa.qc.callRate")._2
    val qMiss = vds.querySA("sa.userdefined.missingness")._2
    val q3 = vds.querySA("sa.anotherthing")._2
    val q4 = vds.querySA("sa.`hi there i have spaces`.another")._2
    val head = vds.sampleAnnotations.head
    vds.sampleAnnotations.foreach { sa =>
      assert(D_==(qCall(sa).asInstanceOf[Double], Option(qMiss(sa)).map(x => (100 - x.asInstanceOf[Double]) / 100).get) &&
        q3(sa) == 5 &&
        q4(sa) == true)
    }
  }

  @Test def testVariants() {
    var vds = hc.importVCF("src/test/resources/sample.vcf")
    vds = vds.filterVariantsExpr("v.start == 10019093")
    vds = SplitMulti(vds)
    vds = VariantQC(vds)
    vds = vds
      .annotateVariantsExpr(
        "va.a.b.c.d.e = va.qc.callRate * 100, va.a.c = if (va.filters.isEmpty) 1 else 0, va.`weird spaces name` = 5 / (va.qual - 5)")
      .cache()

    val vaa = vds.variantsAndAnnotations.collect()
    val q = vds.queryVA("va.a.b.c.d.e")._2
    val q2 = vds.queryVA("va.a.c")._2
    val q3 = vds.queryVA("va.`weird spaces name`")._2
    val qCallRate = vds.queryVA("va.qc.callRate")._2
    val qPass = vds.queryVA("va.filters.isEmpty")._2
    val qQual = vds.queryVA("va.qual")._2
    vds.variantsAndAnnotations
      .collect()
      .foreach { case (v, va) =>
        assert(Option(q(va)) == Option(qCallRate(va)).map(_.asInstanceOf[Double] * 100) &&
          Option(q2(va)) == Option(qPass(va)).map(_.asInstanceOf[Boolean].toInt) &&
          Option(q3(va)) == Option(qQual(va)).map(x => 5 / (x.asInstanceOf[Double] - 5)))
      }
  }
}
