package org.broadinstitute.hail.driver.example

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver.{AnnotateSamples, ImportVCF, State}
import org.testng.annotations.Test

class CaseControlCountSuite extends SparkSuite {
  @Test def test() {
    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("src/test/resources/casecontrolcount.vcf"))
    s = AnnotateSamples.run(s, Array("table",
      "-i", "src/test/resources/casecontrolstatus.tsv",
      "--root", "sa",
      "--sample-expr", "Sample",
      "--types", "case: Boolean"))
    s = CaseControlCount.run(s, Array[String]())

    val qCase = s.vds.queryVA("va.nCase")._2
    val qControl = s.vds.queryVA("va.nControl")._2

    val r = s.vds.mapWithAll { case (v, va, s, sa, g) =>
      (v.start, (qCase(va).get.asInstanceOf[Int],
        qControl(va).get.asInstanceOf[Int]))
    }.collectAsMap()

    assert(r == Map(1 ->(1, 0),
      2 ->(0, 2)))
  }
}
