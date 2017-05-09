package is.hail.variant

import is.hail.SparkSuite
import org.testng.annotations.Test

class GenomeReferenceSuite extends SparkSuite {
  @Test def test() {
    val grch37 = GenomeReference.GRCh37

    assert(grch37.inX("X") && grch37.inY("Y") && grch37.isMitochondrial("MT"))
    assert(grch37.contigs.find(_.name == "1").get.length == 249250621)

    val parXLocus37 = Array(Locus("X", 2499520), Locus("X", 155260460))
    val parYLocus37 = Array(Locus("Y", 50001), Locus("Y", 59035050))
    val nonParXLocus37 = Array(Locus("X", 50), Locus("X", 50000000))
    val nonParYLocus37 = Array(Locus("Y", 5000), Locus("Y", 10000000))

    assert(parXLocus37.forall(grch37.inXPar) && parYLocus37.forall(grch37.inYPar))
    assert(!nonParXLocus37.forall(grch37.inXPar) && !nonParYLocus37.forall(grch37.inYPar))


    val grch38 = GenomeReference.GRCh38

    assert(grch38.inX("chrX") && grch38.inY("chrY") && grch38.isMitochondrial("chrM"))
    assert(grch38.contigs.find(_.name == "chr1").get.length == 248956422)

    val parXLocus38 = Array(Locus("chrX", 2781479)(grch38), Locus("chrX", 156030895)(grch38))
    val parYLocus38 = Array(Locus("chrY", 50001)(grch38), Locus("chrY", 57217415)(grch38))
    val nonParXLocus38 = Array(Locus("chrX", 50)(grch38), Locus("chrX", 50000000)(grch38))
    val nonParYLocus38 = Array(Locus("chrY", 5000)(grch38), Locus("chrY", 10000000)(grch38))

    assert(parXLocus38.forall(grch38.inXPar) && parYLocus38.forall(grch38.inYPar))
    assert(!nonParXLocus38.forall(grch38.inXPar) && !nonParYLocus38.forall(grch38.inYPar))

    val localGenomeReference = hc.genomeReference
    val vds = hc.importVCF("src/test/resources/sample.vcf")
    assert(vds.filterVariants { case (v, va, gs) => v.isMitochondrial(localGenomeReference) }.countVariants() == 0)

    val f = tmpDir.createTempFile(extension = "vds")
    vds.write(f)

    hc.read(f).countVariants()

    hc.read("src/test/resources/sample.vds").countVariants() // Make sure can still read old VDS
  }
}
