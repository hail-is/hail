package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.vcf.LoadVCF
import org.broadinstitute.hail.variant.Variant


class TDTSuite extends SparkSuite {

  @Test def test() {
    // command option l : formats the line
    // command B -- go to declaration
    // CMD E -- open up recent files
    // CMD O -- browse classes (similar)
    // ctrl shift P -- show type
    // CMD R -- run

    val vds2 = LoadVCF(sc, "src/test/resources/fake_vcf_forTDT_testing.vcf")
    val ped2 = Pedigree.read("src/test/resources/tdt_ped_test3_6.fam", sc.hadoopConfiguration, vds2.sampleIds)
    val tdt = TDT(vds2, ped2.completeTrios)

    val parsedFile = readLines("src/test/resources/TDT_testAnswers.tdt.txt", hadoopConf) { lines =>
      lines.drop(1).map(_.value).map {line =>
        val Array(chr, pos, ref, alt, nTrans, nUntrans, chiSq) = line.split("\t")


        val v = Variant(chr, pos.toInt, ref, alt)
        val tdt = TDTResult(nTrans.toInt, nUntrans.toInt, chiSq.toDouble)
        (v, tdt)

          //parse variant, nT, nU, CS
          // convert to map
        }.toMap
    }

    val tdtmap = tdt.collectAsMap()
    assert(tdtmap == parsedFile)
  }

}
