package is.hail.methods

import is.hail.SparkSuite
import org.testng.annotations.Test

import is.hail.utils._

class SummarizeSuite extends SparkSuite {

  @Test def test() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf", nPartitions = Some(8))

    val summary = vds.summarize()

    assert(D_==(summary.callRate.get, vds.queryGenotypes("gs.fraction(g => isDefined(g.GT))")._1.asInstanceOf[Double]))
    assert(summary.multiallelics ==
      vds.queryVariants("variants.filter(v => v.nAlleles() > 2).count()")._1)
    assert(summary.complex ==
      vds.queryVariants("variants.flatMap(v => v.altAlleles).filter(aa => aa.isComplex()).count()")._1)
    assert(summary.snps ==
      vds.queryVariants("variants.flatMap(v => v.altAlleles).filter(aa => aa.isSNP()).count()")._1)
    assert(summary.mnps ==
      vds.queryVariants("variants.flatMap(v => v.altAlleles).filter(aa => aa.isMNP()).count()")._1)
    assert(summary.insertions ==
      vds.queryVariants("variants.flatMap(v => v.altAlleles).filter(aa => aa.isInsertion()).count()")._1)
    assert(summary.deletions ==
      vds.queryVariants("variants.flatMap(v => v.altAlleles).filter(aa => aa.isDeletion()).count()")._1)
    assert(summary.star ==
      vds.queryVariants("variants.flatMap(v => v.altAlleles).filter(aa => aa.isStar()).count()")._1)

    assert(summary.contigs == vds.queryVariants("variants.map(v => v.contig).collect().toSet()")._1)

    assert(summary.maxAlleles == vds.queryVariants("variants.map(v => v.nAlleles()).max()")._1)

    assert(summary.samples == vds.count()._1)
    assert(summary.variants == vds.count()._2)
  }
}
