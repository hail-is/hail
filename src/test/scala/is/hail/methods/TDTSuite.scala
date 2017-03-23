package is.hail.methods

import is.hail.SparkSuite
import is.hail.utils._
import is.hail.variant.Variant
import org.testng.annotations.Test

class TDTSuite extends SparkSuite {

  @Test def test() {

    val out = tmpDir.createLocalTempFile("out", "txt")
    hc.importVCF("src/test/resources/tdt.vcf", nPartitions = Some(4))
      .splitMulti()
      .tdt(Pedigree.fromFam("src/test/resources/tdt.fam", hadoopConf))
      .filterVariantsExpr("v.contig != \"Y\" && v.contig != \"MT\"")
      .exportVariants(out, "CHROM=v.contig, POSITION=v.start, REF=v.ref, ALT=v.alt(), " +
        "T = va.tdt.nTransmitted, U = va.tdt.nUntransmitted, Chi2 = va.tdt.chi2, Pval = va.tdt.pval")

    def parse(file: String) = {
      hadoopConf.readLines(file) { lines =>
        lines.drop(1).map(_.value).map { line =>
          val Array(chr, pos, ref, alt, t, u, chi2, pval) = line.split("\t")

          val v = Variant(chr, pos.toInt, ref, alt)
          val tdt = TDTResult(t.toInt, u.toInt, chi2.toDouble, pval.toDouble)
          (v, tdt)

        }.toMap
      }
    }

    val res1 = parse("src/test/resources/tdt_results.tsv")
    val res2 = parse(out)

    def same(tdt1: TDTResult, tdt2: TDTResult): Boolean =
      tdt1.nTransmitted == tdt1.nTransmitted &&
        tdt1.nUntransmitted == tdt1.nUntransmitted &&
        D_==(tdt1.chi2, tdt2.chi2, 1e-3) &&
        D_==(tdt1.pval, tdt2.pval, 1e-3)

    assert(res1.keysIterator.forall(v => same(res1(v), res2(v))))
  }
}
