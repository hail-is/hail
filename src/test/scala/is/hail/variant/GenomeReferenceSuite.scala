package is.hail.variant

import java.io.FileNotFoundException

import is.hail.expr.typ.{TInterval, TLocus, TStruct, TVariant}
import is.hail.table.Table
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
    TestUtils.interceptFatal("Must have at least one contig in the reference genome.")(GenomeReference("test", Array.empty[String], Map.empty[String, Int]))
    TestUtils.interceptFatal("No lengths given for the following contigs:")(GenomeReference("test", Array("1", "2", "3"), Map("1" -> 5)))
    TestUtils.interceptFatal("Contigs found in `lengths' that are not present in `contigs'")(GenomeReference("test", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5, "4" -> 100)))
    TestUtils.interceptFatal("The following X contig names are absent from the reference:")(GenomeReference("test", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5), xContigs = Set("X")))
    TestUtils.interceptFatal("The following Y contig names are absent from the reference:")(GenomeReference("test", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5), yContigs = Set("Y")))
    TestUtils.interceptFatal("The following mitochondrial contig names are absent from the reference:")(GenomeReference("test", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5), mtContigs = Set("MT")))
    TestUtils.interceptFatal("The contig name for PAR interval")(GenomeReference("test", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5), parInput = Array((Locus("X", 1), Locus("X", 5)))))
    TestUtils.interceptFatal("in both X and Y contigs.")(GenomeReference("test", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5), xContigs = Set("1"), yContigs = Set("1")))
  }

  @Test def testVariant() {
    val gr = GenomeReference.GRCh37

    val v1 = Variant("1", 50000, "A", "T")
    val v2 = Variant("X", 2499520, "T", "G")
    val v3 = Variant("Y", 59034055, "G", "C")
    val v4 = Variant("MT", 30, "T", "G")
    val v5 = Variant("X", 50, "G", "A")
    val v6 = Variant("Y", 5000, "C", "T")

    val expected = Array(
      Array(true, true, false, false, false, false, false),
      Array(false, true, false, true, false, false, false),
      Array(false, true, false, false, true, false, false),
      Array(false, false, true, false, false, false, false),
      Array(false, false, false, false, false, true, false),
      Array(false, false, false, false, false, false, true))

    for ((v, e) <- Array(v1, v2, v3, v4, v5, v6).zip(expected)) {
      assert(v.isAutosomal(gr) == e(0))
      assert(v.isAutosomalOrPseudoAutosomal(gr) == e(1))
      assert(v.isMitochondrial(gr) == e(2))
      assert(v.inXPar(gr) == e(3))
      assert(v.inYPar(gr) == e(4))
      assert(v.inXNonPar(gr) == e(5))
      assert(v.inYNonPar(gr) == e(6))
    }
  }

  @Test def testParser() {
    val gr = GenomeReference("foo", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5))
    GenomeReference.addReference(gr)

    val vds = hc.importVCF("src/test/resources/sample.vcf")
      .annotateVariantsExpr("va.v = NA: Variant(foo), va.l = NA: Locus(foo), va.i = NA: Interval[Locus(foo)]")

    val vas = vds.vaSignature.asInstanceOf[TStruct]

    assert(vas.field("v").typ == TVariant(gr))
    assert(vas.field("l").typ == TLocus(gr))
    assert(vas.field("i").typ == TInterval(TLocus(gr)))

    GenomeReference.removeReference("foo")
  }

  @Test def testDefaultReference() {
    GenomeReference.setDefaultReference(hc, "GRCh38")
    assert(GenomeReference.defaultReference.name == "GRCh38")

    GenomeReference.setDefaultReference(hc, "src/test/resources/fake_ref_genome.json")
    assert(GenomeReference.defaultReference.name == "my_reference_genome")
    GenomeReference.setDefaultReference(hc, "GRCh37")

    TestUtils.interceptFatal("Cannot add reference genome. `GRCh38' already exists.")(GenomeReference.setDefaultReference(hc, "src/main/resources/reference/grch38.json"))
    intercept[FileNotFoundException](GenomeReference.setDefaultReference(hc, "grch38.json"))
    TestUtils.interceptFatal("is a built-in Hail reference")(GenomeReference.removeReference("GRCh37"))
  }

  @Test def testFuncReg() {
    val sig = TStruct(("v37", TVariant(GenomeReference.GRCh37)), ("v38", TVariant(GenomeReference.GRCh38)))

    val data1 = Array(Row(Variant("X", 154931044, "A", "G"), Variant("chrX", 156030895, "A", "G")))
    val kt1 = Table(hc, sc.parallelize(data1), sig)
    kt1.typeCheck()
    assert(kt1.forall("v37.inXPar() && v38.inXPar()"))

    val data2 = Array(Row(Variant("Y", 2649520, "A", "G"), Variant("chrY", 2649520, "A", "G")))
    val kt2 = Table(hc, sc.parallelize(data2), sig)
    kt2.typeCheck()
    assert(kt2.forall("v37.inYPar() && v38.inYPar()"))

    val data3 = Array(Row(Variant("X", 157701382, "A", "G"), Variant("chrX", 157701382, "A", "G")))
    val kt3 = Table(hc, sc.parallelize(data3), sig)
    kt3.typeCheck()
    assert(kt3.forall("v37.inXNonPar() && v38.inXNonPar()"))

    val data4 = Array(Row(Variant("Y", 2781480, "A", "G"), Variant("chrY", 2781480, "A", "G")))
    val kt4 = Table(hc, sc.parallelize(data4), sig)
    kt4.typeCheck()
    assert(kt4.forall("v37.inYNonPar() && v38.inYNonPar()"))

    val data5 = Array(
      Row(Variant("1", 2781480, "A", "G"), Variant("X", 2781480, "A", "G"), Variant("chr1", 2781480, "A", "G"), Variant("chrX", 2781480, "A", "G")),
      Row(Variant("6", 2781480, "A", "G"), Variant("Y", 2781480, "A", "G"), Variant("chr6", 2781480, "A", "G"), Variant("chrY", 2781480, "A", "G")),
      Row(Variant("21", 2781480, "A", "G"), Variant("MT", 2781480, "A", "G"), Variant("chr21", 2781480, "A", "G"), Variant("chrM", 2781480, "A", "G")))
    val kt5 = Table(hc, sc.parallelize(data5), TStruct(
      ("v37a", TVariant(GenomeReference.GRCh37)), ("v37na", TVariant(GenomeReference.GRCh37)),
      ("v38a", TVariant(GenomeReference.GRCh38)), ("v38na", TVariant(GenomeReference.GRCh38))))
    kt5.typeCheck()
    assert(kt5.forall("v37a.isAutosomal() && !v37na.isAutosomal() && v38a.isAutosomal() && !v38na.isAutosomal()"))
  }

  @Test def testConstructors() {
    val kt = hc.importTable("src/test/resources/sampleAnnotations.tsv")
    val ktann = kt.annotate("""v1 = Variant(GRCh38)("chrX:156030895:A:T"), v2 = Variant(GRCh37)("X:154931044:A:T"),
    |v3 = Variant(GRCh37)("1", 3, "A", "T"), l1 = Locus(GRCh38)("1", 100), l2 = Locus(GRCh37)("1:100"),
    |i1 = LocusInterval(GRCh37)("1:5-10"), i2 = LocusInterval(GRCh38)("chrX", 156030890, 156030895)""".stripMargin)

    assert(ktann.signature.field("v1").typ == GenomeReference.GRCh38.variant &&
    ktann.signature.field("v2").typ == GenomeReference.GRCh37.variant &&
    ktann.signature.field("v3").typ == GenomeReference.GRCh37.variant &&
    ktann.signature.field("l1").typ == GenomeReference.GRCh38.locus &&
    ktann.signature.field("l2").typ == GenomeReference.GRCh37.locus &&
    ktann.signature.field("i1").typ == GenomeReference.GRCh37.interval &&
    ktann.signature.field("i2").typ == GenomeReference.GRCh38.interval)

    assert(ktann.forall("v1.inXPar() && v2.inXPar() && v3.isAutosomal() &&" +
      "l1.position == 100 && l2.position == 100 &&" +
      """i1.start == Locus(GRCh37)("1", 5) && !i2.contains(l1)"""))

    val gr = GenomeReference("foo2", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5))
    GenomeReference.addReference(gr)
    GenomeReference.setDefaultReference(gr)
    assert(kt.annotate("""v1 = Variant("chrX:156030895:A:T")""").signature.field("v1").typ == gr.variant)
    assert(kt.annotate("""i1 = Interval(Locus(foo2)("1:100"), Locus(foo2)("1:104"))""").signature.field("i1").typ == gr.interval)

    GenomeReference.setDefaultReference(GenomeReference.GRCh37)
    GenomeReference.removeReference("foo2")
  }

  @Test def testContigRemap() {
    val mapping = Map("23" -> "foo")
    TestUtils.interceptFatal("do not have remapped contigs in the reference genome")(GenomeReference.GRCh37.validateContigRemap(mapping))
  }

  @Test def testComparisonOps() {
    val gr = GenomeReference.GRCh37

    // Test contigs
    assert(gr.compare("3", "18") < 0)
    assert(gr.compare("18", "3") > 0)
    assert(gr.compare("7", "7") == 0)

    assert(gr.compare("3", "X") < 0)
    assert(gr.compare("X", "3") > 0)
    assert(gr.compare("X", "X") == 0)

    assert(gr.compare("X", "Y") < 0)
    assert(gr.compare("Y", "X") > 0)
    assert(gr.compare("Y", "MT") < 0)

    assert(gr.compare("18", "SPQR") < 0)
    assert(gr.compare("MT", "SPQR") < 0)

    // Test variants
    val v1 = Variant("1", 500, "A", "T")
    val v2 = Variant("1", 600, "T", "A")
    val v3 = Variant("1", 600, "G", "A")
    val v4 = Variant("1", 600, "G", Array("A", "C"))
    val v5 = Variant("2", 700, "T", "C")

    assert(gr.compare(v1, v2) < 0)
    assert(gr.compare(v2, v1) > 0)
    assert(gr.compare(v1, v1) == 0)

    assert(gr.compare(v2, v3) > 0)
    assert(gr.compare(v3, v2) < 0)
    assert(gr.compare(v3, v4) < 0)

    assert(gr.compare(v5, v1) > 0)

    // Test loci
    val l1 = Locus("1", 25)
    val l2 = Locus("1", 13000)
    val l3 = Locus("2", 26)

    assert(gr.compare(l1, l3) < 1)
    assert(gr.compare(l1, l1) == 0)
    assert(gr.compare(l3, l1) > 0)
    assert(gr.compare(l2, l1) > 0)
  }

  @Test def testWriteToFile() {
    val tmpFile = tmpDir.createTempFile("grWrite", ".json")

    val gr = GenomeReference.GRCh37
    gr.copy(name = "GRCh37_2").write(hc, tmpFile)
    val gr2 = GenomeReference.fromFile(hc, tmpFile)

    assert((gr.contigs sameElements gr2.contigs) &&
      gr.lengths == gr2.lengths &&
      gr.xContigs == gr2.xContigs &&
      gr.yContigs == gr2.yContigs &&
      gr.mtContigs == gr2.mtContigs &&
      (gr.parInput sameElements gr2.parInput))

    GenomeReference.removeReference("GRCh37_2")
  }
}
