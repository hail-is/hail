package is.hail.variant

import is.hail.SparkSuite
import org.testng.annotations.Test

class GenomeReferenceSuite extends SparkSuite {
  @Test def test() {
    val grch37 = GenomeReference.GRCh37

    assert(grch37.inX("X") && grch37.inY("Y") && grch37.isMitochondrial("MT"))
    assert(grch37.contigs.find(_.name == "1").get.length == 249250621)

    val parXLocus = Array(Locus("X", 2499520), Locus("X", 155260460))
    val parYLocus = Array(Locus("Y", 50001), Locus("Y", 59035050))
    val nonParXLocus = Array(Locus("X", 50), Locus("X", 50000000))
    val nonParYLocus = Array(Locus("Y", 5000), Locus("Y", 10000000))

    assert(parXLocus.forall(grch37.inXPar) && parYLocus.forall(grch37.inYPar))
    assert(!nonParXLocus.forall(grch37.inXPar) && !nonParYLocus.forall(grch37.inYPar))

    val vds = hc.importVCF("src/test/resources/sample.vcf")
    assert(vds.filterVariants { case (v, va, gs) => v.isMitochondrial }.countVariants() == 0)
  }
}
