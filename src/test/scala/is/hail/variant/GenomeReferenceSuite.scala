package is.hail.variant

import is.hail.SparkSuite
import is.hail.expr.{TInterval, TLocus, TStruct, TVariant}
import is.hail.utils.Interval
import org.testng.annotations.Test

class GenomeReferenceSuite extends SparkSuite {
  @Test def testGRCh37() {
    val grch37 = GenomeReference.GRCh37

    assert(grch37.inX("X") && grch37.inY("Y") && grch37.isMitochondrial("MT"))
    assert(grch37.contigs.find(_.name == "1").get.length == 249250621)

    val parXLocus = Array(Locus("X", 2499520), Locus("X", 155260460))
    val parYLocus = Array(Locus("Y", 50001), Locus("Y", 59035050))
    val nonParXLocus = Array(Locus("X", 50), Locus("X", 50000000))
    val nonParYLocus = Array(Locus("Y", 5000), Locus("Y", 10000000))

    assert(parXLocus.forall(grch37.inXPar) && parYLocus.forall(grch37.inYPar))
    assert(!nonParXLocus.forall(grch37.inXPar) && !nonParYLocus.forall(grch37.inYPar))
  }

  @Test def testGRCh38() {
    val grch38 = GenomeReference.GRCh38

    assert(grch38.inX("chrX") && grch38.inY("chrY") && grch38.isMitochondrial("chrM"))
    assert(grch38.contigs.find(_.name == "chr1").get.length == 248956422)

    val parXLocus38 = Array(Locus("chrX", 2781479), Locus("chrX", 156030895))
    val parYLocus38 = Array(Locus("chrY", 50001), Locus("chrY", 57217415))
    val nonParXLocus38 = Array(Locus("chrX", 50), Locus("chrX", 50000000))
    val nonParYLocus38 = Array(Locus("chrY", 5000), Locus("chrY", 10000000))

    assert(parXLocus38.forall(grch38.inXPar) && parYLocus38.forall(grch38.inYPar))
    assert(!nonParXLocus38.forall(grch38.inXPar) && !nonParYLocus38.forall(grch38.inYPar))
  }

  @Test def testAssertions() {
    intercept[IllegalArgumentException](GenomeReference("test", Array(Contig("1", 5), Contig("2", 5), Contig("3", 5)),
      Set("X"), Set.empty[String], Set.empty[String], Array.empty[Interval[Locus]]))
    intercept[IllegalArgumentException](GenomeReference("test", Array(Contig("1", 5), Contig("2", 5), Contig("3", 5)),
      Set.empty[String], Set("Y"), Set.empty[String], Array.empty[Interval[Locus]]))
    intercept[IllegalArgumentException](GenomeReference("test", Array(Contig("1", 5), Contig("2", 5), Contig("3", 5)),
      Set.empty[String], Set.empty[String], Set("MT"), Array.empty[Interval[Locus]]))
    intercept[IllegalArgumentException](GenomeReference("test", Array.empty[Contig],
      Set.empty[String], Set.empty[String], Set.empty[String], Array.empty[Interval[Locus]]))
    intercept[IllegalArgumentException](GenomeReference("test", Array(Contig("1", 5), Contig("2", 5), Contig("3", 5)),
      Set.empty[String], Set.empty[String], Set("MT"), Array(Interval(Locus("X", 1), Locus("X", 5)))))
  }

  @Test def testVariant() {
    val gr = GenomeReference.GRCh37

    val v1 = Variant("1", 50000, "A", "T")
    val v2 = Variant("X", 2499520, "T", "G")
    val v3 = Variant("Y", 50001, "G", "C")
    val v4 = Variant("MT", 30, "T", "G")
    val v5 = Variant("X", 50, "G", "A")
    val v6 = Variant("Y", 5000, "C", "T")

    for (v <- Array(v1, v2, v3, v4, v5, v6)) {
      assert(v.isAutosomal == v.isAutosomal(gr))
      assert(v.isAutosomalOrPseudoAutosomal == v.isAutosomalOrPseudoAutosomal(gr))
      assert(v.isMitochondrial == v.isMitochondrial(gr))
      assert(v.inXPar == v.inXPar(gr))
      assert(v.inYPar == v.inYPar(gr))
      assert(v.inXNonPar == v.inXNonPar(gr))
      assert(v.inYNonPar == v.inYNonPar(gr))
    }
  }

  @Test def testParser() {
    val gr = GenomeReference("foo", Array(Contig("1", 10000)))
    GenomeReference.addReference(gr)

    val vds = hc.importVCF("src/test/resources/sample.vcf")
      .annotateVariantsExpr("va.v = NA: Variant('foo'), va.l = NA: Locus('foo'), va.i = NA: Interval('foo')")

    val vas = vds.vaSignature.asInstanceOf[TStruct]

    assert(vas.field("v").typ == TVariant(gr))
    assert(vas.field("l").typ == TLocus(gr))
    assert(vas.field("i").typ == TInterval(gr))
  }
}
