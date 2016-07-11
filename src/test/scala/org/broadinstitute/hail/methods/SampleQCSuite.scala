package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver.{State, _}
import org.testng.annotations.Test
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.utils._
import scala.io.Source

class SampleQCSuite extends SparkSuite {
  @Test def testStoreAfterFilter() {
    var s = State(sc, sqlContext)

    val sampleQCFile = tmpDir.createTempFile("sampleqc", extension = ".tsv")
    val exportSamplesFile = tmpDir.createTempFile("exportsamples", extension = ".tsv")

    s = ImportVCF.run(s, Array("src/test/resources/multipleChromosomes.vcf"))
    s = SplitMulti.run(s, Array.empty[String])
    s = FilterSamplesExpr.run(s, Array("--keep", "-c", """"HG" ~ s.id"""))
    s = SampleQC.run(s, Array("-o", sampleQCFile))
    s = ExportSamples.run(s, Array("-o", exportSamplesFile, "-c",
      """Sample = s.id,
        |nNotCalled = sa.qc.nNotCalled,
        |nHomRef = sa.qc.nHomRef,
        |nHet = sa.qc.nHet,
        |nHomVar = sa.qc.nHomVar""".stripMargin))

    val sampleQCLines = readFile(sampleQCFile, hadoopConf) { s =>
      Source.fromInputStream(s)
        .getLines()
        .map { line =>
          val fields = line.split("\t")
          Array(fields(0), fields(3), fields(4), fields(5), fields(6)).mkString("\t")
        }
        .toSet
    }
    val exportSamplesLines = readFile(exportSamplesFile, hadoopConf) { s =>
      Source.fromInputStream(s)
        .getLines().toSet
    }

    assert(exportSamplesLines == sampleQCLines)
  }
}
