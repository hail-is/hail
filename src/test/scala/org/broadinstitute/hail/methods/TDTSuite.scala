package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.io.vcf.LoadVCF
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.Variant
import org.testng.annotations.Test


class TDTSuite extends SparkSuite {

  @Test def test() {

    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, "src/test/resources/fake_vcf_forTDT_testing.vcf")
    s = SplitMulti.run(s)
    s = TDTCommand.run(s, "-r", "va.tdt", "-f", "src/test/resources/tdt_ped_test3_6.fam")
    val out = tmpDir.createLocalTempFile("out", "txt")
    s = FilterVariantsExpr.run(s, "--keep", "-c", "v.contig != \"Y\" && v.contig != \"MT\"")
    ExportVariants.run(s, "-o", out, "-c", "CHROM=v.contig, POSITION=v.start, REF=v.ref, ALT=v.alt, " +
      "Transmitted = va.tdt.nTransmitted, Untransmitted = va.tdt.nUntransmitted, `Chi-Square` = va.tdt.chiSquare.orElse(0.0)")

    def parse(file: String) = {
      hadoopConf.readLines(file) { lines =>
        lines.drop(1).map(_.value).map { line =>
          val Array(chr, pos, ref, alt, nTrans, nUntrans, chiSq) = line.split("\t")


          val v = Variant(chr, pos.toInt, ref, alt)
          val tdt = TDTResult(nTrans.toInt, nUntrans.toInt, chiSq.toDouble)
          (v, tdt)

        }.toMap
      }
    }

    assert(parse("src/test/resources/TDT_testAnswers.tdt.txt") == parse(out))
  }
}
