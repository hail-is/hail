package is.hail.variant

import java.io.FileNotFoundException

import is.hail.expr.{TInterval, TLocus, TStruct, TVariant}
import is.hail.keytable.KeyTable
import is.hail.utils.Interval
import is.hail.{SparkSuite, TestUtils}
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class GenomeReferenceSuite extends SparkSuite {
  @Test def testGRCh37() {
    val grch37 = GenomeReference.GRCh37
    assert(GenomeReference.hasReference("GRCh37"))

    assert(grch37.inX("X") && grch37.inY("Y") && grch37.isMitochondrial("MT"))
    assert(grch37.contigLength("1") == 249250621)

    val parXLocus = Array(Locus("X", 2499520), Locus("X", 155260460))
    val parYLocus = Array(Locus("Y", 50001), Locus("Y", 59035050))
    val nonParXLocus = Array(Locus("X", 50), Locus("X", 50000000))
    val nonParYLocus = Array(Locus("Y", 5000), Locus("Y", 10000000))

    assert(parXLocus.forall(grch37.inXPar) && parYLocus.forall(grch37.inYPar))
    assert(!nonParXLocus.forall(grch37.inXPar) && !nonParYLocus.forall(grch37.inYPar))
  }

  @Test def testGRCh38() {
    val grch38 = GenomeReference.GRCh38
    assert(GenomeReference.hasReference("GRCh38"))

    assert(grch38.inX("chrX") && grch38.inY("chrY") && grch38.isMitochondrial("chrM"))
    assert(grch38.contigLength("chr1") == 248956422)

    val parXLocus38 = Array(Locus("chrX", 2781479), Locus("chrX", 156030895))
    val parYLocus38 = Array(Locus("chrY", 50001), Locus("chrY", 57217415))
    val nonParXLocus38 = Array(Locus("chrX", 50), Locus("chrX", 50000000))
    val nonParYLocus38 = Array(Locus("chrY", 5000), Locus("chrY", 10000000))

    assert(parXLocus38.forall(grch38.inXPar) && parYLocus38.forall(grch38.inYPar))
    assert(!nonParXLocus38.forall(grch38.inXPar) && !nonParYLocus38.forall(grch38.inYPar))
  }

  @Test def testAssertions() {
    TestUtils.interceptFatal("The following X contig names are absent from the reference:")(GenomeReference("test", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5),
      Set("X"), Set.empty[String], Set.empty[String], Array.empty[Interval[Locus]]))
    TestUtils.interceptFatal("No lengths given for the following contigs:")(GenomeReference("test", Array("1", "2", "3"), Map("1" -> 5),
      Set.empty[String], Set.empty[String], Set.empty[String], Array.empty[Interval[Locus]]))
    TestUtils.interceptFatal("Contigs found in `lengths' that are not present in `contigs'")(GenomeReference("test", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5, "4" -> 100),
      Set.empty[String], Set.empty[String], Set.empty[String], Array.empty[Interval[Locus]]))
    TestUtils.interceptFatal("The following Y contig names are absent from the reference:")(GenomeReference("test", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5),
      Set.empty[String], Set("Y"), Set.empty[String], Array.empty[Interval[Locus]]))
    TestUtils.interceptFatal("The following mitochondrial contig names are absent from the reference:")(GenomeReference("test", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5),
      Set.empty[String], Set.empty[String], Set("MT"), Array.empty[Interval[Locus]]))
    TestUtils.interceptFatal("Must have at least one contig in the reference genome.")(GenomeReference("test", Array.empty[String], Map.empty[String, Int],
      Set.empty[String], Set.empty[String], Set.empty[String], Array.empty[Interval[Locus]]))
    TestUtils.interceptFatal("The contig name for PAR interval")(GenomeReference("test", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5),
      Set.empty[String], Set.empty[String], Set.empty[String], Array(Interval(Locus("X", 1), Locus("X", 5)))))
    TestUtils.interceptFatal("in both X and Y contigs.")(GenomeReference("test", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5),
      Set("1"), Set("1"), Set.empty[String], Array.empty[Interval[Locus]]))
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
    val gr = GenomeReference("foo", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5),
      Set.empty[String], Set.empty[String], Set.empty[String], Array.empty[Interval[Locus]])
    GenomeReference.addReference(gr)

    val vds = hc.importVCF("src/test/resources/sample.vcf")
      .annotateVariantsExpr("va.v = NA: Variant(foo), va.l = NA: Locus(foo), va.i = NA: Interval(foo)")

    val vas = vds.vaSignature.asInstanceOf[TStruct]

    assert(vas.field("v").typ == TVariant(gr))
    assert(vas.field("l").typ == TLocus(gr))
    assert(vas.field("i").typ == TInterval(gr))
  }

  @Test def testDefaultReference() {
    GenomeReference.setDefaultReference(hc, "GRCh38")
    assert(GenomeReference.defaultReference.name == "GRCh38")

    GenomeReference.setDefaultReference(hc, "src/test/resources/fake_ref_genome.json")
    assert(GenomeReference.defaultReference.name == "my_reference_genome")
    GenomeReference.setDefaultReference(hc, "GRCh37")

    TestUtils.interceptFatal("Cannot add reference genome. Reference genome `GRCh38' already exists.")(GenomeReference.setDefaultReference(hc, "src/main/resources/reference/grch38.json"))
    intercept[FileNotFoundException](GenomeReference.setDefaultReference(hc, "grch38.json"))
  }

  @Test def testFuncReg() {
    val sig = TStruct(("v37", TVariant(GenomeReference.GRCh37)), ("v38", TVariant(GenomeReference.GRCh38)))

    val data1 = Array(Row(Variant("X", 154931044, "A", "G"), Variant("chrX", 156030895, "A", "G")))
    val kt1 = KeyTable(hc, sc.parallelize(data1), sig)
    kt1.typeCheck()
    assert(kt1.forall("v37.inXPar() && v38.inXPar()"))

    val data2 = Array(Row(Variant("Y", 2649520, "A", "G"), Variant("chrY", 2649520, "A", "G")))
    val kt2 = KeyTable(hc, sc.parallelize(data2), sig)
    kt2.typeCheck()
    assert(kt2.forall("v37.inYPar() && v38.inYPar()"))

    val data3 = Array(Row(Variant("X", 157701382, "A", "G"), Variant("chrX", 157701382, "A", "G")))
    val kt3 = KeyTable(hc, sc.parallelize(data3), sig)
    kt3.typeCheck()
    assert(kt3.forall("v37.inXNonPar() && v38.inXNonPar()"))

    val data4 = Array(Row(Variant("Y", 2781480, "A", "G"), Variant("chrY", 2781480, "A", "G")))
    val kt4 = KeyTable(hc, sc.parallelize(data4), sig)
    kt4.typeCheck()
    assert(kt4.forall("v37.inYNonPar() && v38.inYNonPar()"))

    val data5 = Array(
      Row(Variant("1", 2781480, "A", "G"), Variant("X", 2781480, "A", "G"), Variant("chr1", 2781480, "A", "G"), Variant("chrX", 2781480, "A", "G")),
      Row(Variant("6", 2781480, "A", "G"), Variant("Y", 2781480, "A", "G"), Variant("chr6", 2781480, "A", "G"), Variant("chrY", 2781480, "A", "G")),
      Row(Variant("21", 2781480, "A", "G"), Variant("MT", 2781480, "A", "G"), Variant("chr21", 2781480, "A", "G"), Variant("chrM", 2781480, "A", "G")))
    val kt5 = KeyTable(hc, sc.parallelize(data5), TStruct(
      ("v37a", TVariant(GenomeReference.GRCh37)), ("v37na", TVariant(GenomeReference.GRCh37)),
      ("v38a", TVariant(GenomeReference.GRCh38)), ("v38na", TVariant(GenomeReference.GRCh38))))
    kt5.typeCheck()
    assert(kt5.forall("v37a.isAutosomal() && !v37na.isAutosomal() && v38a.isAutosomal() && !v38na.isAutosomal()"))
  }
}
