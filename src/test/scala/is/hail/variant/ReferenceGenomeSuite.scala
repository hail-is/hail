package is.hail.variant

import java.io.FileNotFoundException

import is.hail.asm4s.FunctionBuilder
import is.hail.check.Prop._
import is.hail.check.Properties
import is.hail.expr.ir.EmitFunctionBuilder
import is.hail.expr.types.{TInterval, TLocus, TStruct}
import is.hail.io.reference.FASTAReader
import is.hail.table.Table
import is.hail.utils.{HailException, Interval, SerializableHadoopConfiguration}
import is.hail.testUtils._
import is.hail.{SparkSuite, TestUtils}
import org.apache.spark.SparkException
import org.apache.spark.sql.Row
import org.testng.annotations.Test
import org.apache.hadoop

class ReferenceGenomeSuite extends SparkSuite {
  @Test def testGRCh37() {
    val grch37 = ReferenceGenome.GRCh37
    assert(ReferenceGenome.hasReference("GRCh37"))

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
    val grch38 = ReferenceGenome.GRCh38
    assert(ReferenceGenome.hasReference("GRCh38"))

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
    TestUtils.interceptFatal("Must have at least one contig in the reference genome.")(ReferenceGenome("test", Array.empty[String], Map.empty[String, Int]))
    TestUtils.interceptFatal("No lengths given for the following contigs:")(ReferenceGenome("test", Array("1", "2", "3"), Map("1" -> 5)))
    TestUtils.interceptFatal("Contigs found in `lengths' that are not present in `contigs'")(ReferenceGenome("test", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5, "4" -> 100)))
    TestUtils.interceptFatal("The following X contig names are absent from the reference:")(ReferenceGenome("test", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5), xContigs = Set("X")))
    TestUtils.interceptFatal("The following Y contig names are absent from the reference:")(ReferenceGenome("test", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5), yContigs = Set("Y")))
    TestUtils.interceptFatal("The following mitochondrial contig names are absent from the reference:")(ReferenceGenome("test", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5), mtContigs = Set("MT")))
    TestUtils.interceptFatal("The contig name for PAR interval")(ReferenceGenome("test", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5), parInput = Array((Locus("X", 1), Locus("X", 5)))))
    TestUtils.interceptFatal("in both X and Y contigs.")(ReferenceGenome("test", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5), xContigs = Set("1"), yContigs = Set("1")))
  }

  @Test def testParser() {
    val rg = ReferenceGenome("foo", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5))
    ReferenceGenome.addReference(rg)

    val vds = hc.importVCF("src/test/resources/sample.vcf")
      .annotateRowsExpr("l" -> "NA: Locus(foo)", "i" -> "NA: Interval[Locus(foo)]")

    val vas = vds.rowType.asInstanceOf[TStruct]

    assert(vas.field("l").typ == TLocus(rg))
    assert(vas.field("i").typ == TInterval(TLocus(rg)))

    ReferenceGenome.removeReference("foo")
  }

  @Test def testDefaultReference() {
    ReferenceGenome.setDefaultReference(hc, "GRCh38")
    assert(ReferenceGenome.defaultReference.name == "GRCh38")

    ReferenceGenome.setDefaultReference(hc, "src/test/resources/fake_ref_genome.json")
    assert(ReferenceGenome.defaultReference.name == "my_reference_genome")
    ReferenceGenome.setDefaultReference(hc, "GRCh37")

    TestUtils.interceptFatal("Cannot add reference genome. `GRCh38' already exists.")(ReferenceGenome.setDefaultReference(hc, "src/main/resources/reference/grch38.json"))
    intercept[FileNotFoundException](ReferenceGenome.setDefaultReference(hc, "grch38.json"))
    TestUtils.interceptFatal("is a built-in Hail reference")(ReferenceGenome.removeReference("GRCh37"))
  }

  @Test def testFuncReg() {
    val sig = TStruct(("v37", TLocus(ReferenceGenome.GRCh37)), ("v38", TLocus(ReferenceGenome.GRCh38)))

    val data1 = Array(Row(Locus("X", 154931044), Locus("chrX", 156030895)))
    val kt1 = Table(hc, sc.parallelize(data1), sig)
    kt1.typeCheck()
    assert(kt1.forall("row.v37.inXPar() && row.v38.inXPar()"))

    val data2 = Array(Row(Locus("Y", 2649520), Locus("chrY", 2649520)))
    val kt2 = Table(hc, sc.parallelize(data2), sig)
    kt2.typeCheck()
    assert(kt2.forall("row.v37.inYPar() && row.v38.inYPar()"))

    val data3 = Array(Row(Locus("X", 157701382), Locus("chrX", 157701382)))
    val kt3 = Table(hc, sc.parallelize(data3), sig)
    kt3.typeCheck()
    assert(kt3.forall("row.v37.inXNonPar() && row.v38.inXNonPar()"))

    val data4 = Array(Row(Locus("Y", 2781480), Locus("chrY", 2781480)))
    val kt4 = Table(hc, sc.parallelize(data4), sig)
    kt4.typeCheck()
    assert(kt4.forall("row.v37.inYNonPar() && row.v38.inYNonPar()"))

    val data5 = Array(
      Row(Locus("1", 2781480), Locus("X", 2781480), Locus("chr1", 2781480), Locus("chrX", 2781480)),
      Row(Locus("6", 2781480), Locus("Y", 2781480), Locus("chr6", 2781480), Locus("chrY", 2781480)),
      Row(Locus("21", 2781480), Locus("MT", 2781480), Locus("chr21", 2781480), Locus("chrM", 2781480)))
    val kt5 = Table(hc, sc.parallelize(data5), TStruct(
      ("v37a", TLocus(ReferenceGenome.GRCh37)), ("v37na", TLocus(ReferenceGenome.GRCh37)),
      ("v38a", TLocus(ReferenceGenome.GRCh38)), ("v38na", TLocus(ReferenceGenome.GRCh38))))
    kt5.typeCheck()
    assert(kt5.forall("row.v37a.isAutosomal() && !row.v37na.isAutosomal() && " +
      "row.v38a.isAutosomal() && !row.v38na.isAutosomal()"))
  }

  @Test def testConstructors() {
    val kt = hc.importTable("src/test/resources/sampleAnnotations.tsv")
    val ktann = kt.annotate("l1" -> """Locus(GRCh38)("chr1", 100)""",
      "l2" -> """Locus(GRCh37)("1:100")""",
      "i1" -> """LocusInterval(GRCh37)("1:5-10")""",
      "i2" -> """LocusInterval(GRCh38)("chrX", 156030890, 156030895, true, false)""")

    assert(ktann.signature.field("l1").typ == ReferenceGenome.GRCh38.locusType &&
    ktann.signature.field("l2").typ == ReferenceGenome.GRCh37.locusType &&
    ktann.signature.field("i1").typ == TInterval(ReferenceGenome.GRCh37.locusType) &&
    ktann.signature.field("i2").typ == TInterval(ReferenceGenome.GRCh38.locusType))

    assert(ktann.forall("row.l1.position == 100 && row.l2.position == 100 &&" +
      """row.i1.start == Locus(GRCh37)("1", 5) && !row.i2.contains(row.l1)"""))

    val rg = ReferenceGenome("foo2", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5))
    ReferenceGenome.addReference(rg)
    ReferenceGenome.setDefaultReference(rg)
    assert(kt.annotate("i1" -> """Interval(Locus(foo2)("1:100"), Locus(foo2)("1:104"), true, false)""").signature.field("i1").typ == TInterval(rg.locusType))

    // check for invalid contig names or positions
    intercept[SparkException](kt.annotate("l1bad" -> """Locus("MT:17000") """).collect())
    intercept[SparkException](kt.annotate("l1bad" -> """Locus("foo:17000") """).collect())
    intercept[SparkException](kt.annotate("l1bad" -> """Locus("foo", 17000) """).collect())

    intercept[SparkException](kt.annotate("i1bad" -> """LocusInterval("MT:4789-17000") """).collect())
    intercept[SparkException](kt.annotate("i1bad" -> """LocusInterval("foo:4789-17000") """).collect())
    intercept[SparkException](kt.annotate("i1bad" -> """LocusInterval("foo", 1, 10, true, false) """).collect())
    intercept[SparkException](kt.annotate("i1bad" -> """LocusInterval("MT", 5, 17000, true, false) """).collect())

    intercept[HailException](kt.annotate("i1bad" -> """Interval(Locus(foo2)("1:100"), Locus(GRCh37)("1:104"), true, false)""").collect())

    ReferenceGenome.setDefaultReference(ReferenceGenome.GRCh37)
    ReferenceGenome.removeReference("foo2")
  }

  @Test def testContigRemap() {
    val mapping = Map("23" -> "foo")
    TestUtils.interceptFatal("have remapped contigs in reference genome")(ReferenceGenome.GRCh37.validateContigRemap(mapping))
  }

  @Test def testComparisonOps() {
    val rg = ReferenceGenome.GRCh37

    // Test contigs
    assert(rg.compare("3", "18") < 0)
    assert(rg.compare("18", "3") > 0)
    assert(rg.compare("7", "7") == 0)

    assert(rg.compare("3", "X") < 0)
    assert(rg.compare("X", "3") > 0)
    assert(rg.compare("X", "X") == 0)

    assert(rg.compare("X", "Y") < 0)
    assert(rg.compare("Y", "X") > 0)
    assert(rg.compare("Y", "MT") < 0)

    assert(rg.compare("18", "SPQR") < 0)
    assert(rg.compare("MT", "SPQR") < 0)

    // Test loci
    val l1 = Locus("1", 25)
    val l2 = Locus("1", 13000)
    val l3 = Locus("2", 26)

    assert(rg.compare(l1, l3) < 1)
    assert(rg.compare(l1, l1) == 0)
    assert(rg.compare(l3, l1) > 0)
    assert(rg.compare(l2, l1) > 0)
  }

  @Test def testWriteToFile() {
    val tmpFile = tmpDir.createTempFile("grWrite", ".json")

    val rg = ReferenceGenome.GRCh37
    rg.copy(name = "GRCh37_2").write(hc, tmpFile)
    val gr2 = ReferenceGenome.fromFile(hc, tmpFile)

    assert((rg.contigs sameElements gr2.contigs) &&
      rg.lengths == gr2.lengths &&
      rg.xContigs == gr2.xContigs &&
      rg.yContigs == gr2.yContigs &&
      rg.mtContigs == gr2.mtContigs &&
      (rg.parInput sameElements gr2.parInput))

    ReferenceGenome.removeReference("GRCh37_2")
  }

  @Test def testWriteRG() {
    val outKT = tmpDir.createTempFile("grWrite", "kt")
    val outKT2 = tmpDir.createTempFile("grWrite", "kt")
    val outVDS = tmpDir.createTempFile("grWrite", "vds")

    val kt = hc.importTable("src/test/resources/sampleAnnotations.tsv")
    val vds = hc.importVCF("src/test/resources/sample.vcf")

    val rg = ReferenceGenome("foo", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5))
    ReferenceGenome.addReference(rg)
    kt.annotate("l1" -> """Locus(foo)("1:3")""").write(outKT)
    vds.annotateRowsExpr("l2" -> """Locus(foo)("1:3")""").write(outVDS)
    ReferenceGenome.removeReference("foo")

    val rg2 = ReferenceGenome("foo", Array("1"), Map("1" -> 5))
    ReferenceGenome.addReference(rg2)
    kt.annotate("l1" -> """Locus(foo)("1:3")""").write(outKT2)
    ReferenceGenome.removeReference("foo")

    assert(hc.readTable(outKT).signature.field("l1").typ == TLocus(rg))
    assert(hc.read(outVDS).rowType.fieldOption("l2").get.typ == TLocus(rg))
    TestUtils.interceptFatal("`foo' already exists and is not identical to the imported reference from")(hc.readTable(outKT2))
    ReferenceGenome.removeReference("foo")
  }

  @Test def testFasta() {
    val fastaFile = "src/test/resources/fake_reference.fasta"
    val fastaFileGzip = "src/test/resources/fake_reference.fasta.gz"
    val indexFile = "src/test/resources/fake_reference.fasta.fai"

    val rg = ReferenceGenome("test", Array("a", "b", "c"), Map("a" -> 25, "b" -> 15, "c" -> 10))
    ReferenceGenome.addReference(rg)

    val fr = FASTAReader(hc, rg, fastaFile, indexFile, 3, 5)
    val frGzip = FASTAReader(hc, rg, fastaFileGzip, indexFile, 3, 5)

    object Spec extends Properties("Fasta Random") {
      property("cache gives same base as from file") = forAll(Locus.gen(rg)) { l =>
        val contig = l.contig
        val pos = l.position
        val expected = fr.reader.value.getSubsequenceAt(contig, pos, pos).getBaseString
        fr.lookup(contig, pos, 0, 0) == expected && frGzip.lookup(contig, pos, 0, 0) == expected
      }

      val ordering = TLocus(rg).ordering
      property("interval test") = forAll(Interval.gen(ordering, Locus.gen(rg))) { i =>
        val start = i.start.asInstanceOf[Locus]
        val end = i.end.asInstanceOf[Locus]

        def getHtsjdkIntervalSequence: String = {
          val sb = new StringBuilder
          var pos = start
          while (ordering.lteq(pos, end) && pos != null) {
            val endPos = if (pos.contig != end.contig) rg.contigLength(pos.contig) else end.position
            sb ++= fr.reader.value.getSubsequenceAt(pos.contig, pos.position, endPos).getBaseString
            pos =
              if (rg.contigsIndex(pos.contig) == rg.contigs.length - 1)
                null
              else
                Locus(rg.contigs(rg.contigsIndex(pos.contig) + 1), 1)
          }
          sb.result()
        }

        fr.lookup(Interval(start, end, includesStart = true, includesEnd = true)) == getHtsjdkIntervalSequence
      }
    }

    Spec.check()

    assert(fr.lookup("a", 25, 0, 5) == "A")
    assert(fr.lookup("b", 1, 5, 0) == "T")
    assert(fr.lookup("c", 5, 10, 10) == "GGATCCGTGC")
    assert(fr.lookup(Interval(Locus("a", 1), Locus("a", 5), includesStart = true, includesEnd = false)) == "AGGT")
    assert(fr.lookup(Interval(Locus("a", 20), Locus("b", 5), includesStart = false, includesEnd = false)) == "ACGTATAAT")
    assert(fr.lookup(Interval(Locus("a", 20), Locus("c", 5), includesStart = false, includesEnd = false)) == "ACGTATAATTAAATTAGCCAGGAT")

    rg.addSequence(hc, fastaFile, indexFile)
    val table = hc.importTable("src/test/resources/fake_reference.tsv")
    assert(table.annotate("baseComputed" -> """getReferenceSequence(test)(row.contig, row.pos.toInt32(), 0, 0)""")
      .forall("row.base == row.baseComputed"))

    ReferenceGenome.removeReference(rg.name)

    val rg2 = ReferenceGenome.fromFASTAFile(hc, "test2", fastaFileGzip, indexFile)
    assert(table.annotate("baseComputed" -> """getReferenceSequence(test2)(row.contig, row.pos.toInt32(), 0, 0)""")
      .forall("row.base == row.baseComputed"))
    ReferenceGenome.removeReference(rg2.name)
  }

  @Test def testSerializeOnFB() {
    val grch38 = ReferenceGenome.GRCh38
    val fb = EmitFunctionBuilder[String, Boolean]

    val rgfield = fb.newLazyField(grch38.codeSetup(fb))
    fb.emit(rgfield.invoke[String, Boolean]("isValidContig", fb.getArg[String](1)))

    val f = fb.result()()
    assert(f("X") == grch38.isValidContig("X"))
  }

  @Test def testSerializeWithFastaOnFB() {
    val fastaFile = "src/test/resources/fake_reference.fasta"
    val indexFile = "src/test/resources/fake_reference.fasta.fai"

    val rg = ReferenceGenome("test", Array("a", "b", "c"), Map("a" -> 25, "b" -> 15, "c" -> 10))
    ReferenceGenome.addReference(rg)
    rg.addSequence(hc, fastaFile, indexFile)

    val fb = EmitFunctionBuilder[String, Int, Int, Int, String]

    val rgfield = fb.newLazyField(rg.codeSetup(fb))
    fb.emit(rgfield.invoke[String, Int, Int, Int, String]("getSequence", fb.getArg[String](1), fb.getArg[Int](2), fb.getArg[Int](3), fb.getArg[Int](4)))

    val f = fb.result()()
    assert(f("a", 25, 0, 5) == rg.getSequence("a", 25, 0, 5))
    ReferenceGenome.removeReference(rg.name)
  }

  @Test def testSerializeWithLiftoverOnFB() {
    val grch37 = ReferenceGenome.GRCh37
    val liftoverFile = "src/test/resources/grch37_to_grch38_chr20.over.chain.gz"

    grch37.addLiftover(hc, liftoverFile, "GRCh38")

    val fb = EmitFunctionBuilder[String, Locus, Double, Locus]
    val rgfield = fb.newLazyField(grch37.codeSetup(fb))
    fb.emit(rgfield.invoke[String, Locus, Double, Locus]("liftoverLocus", fb.getArg[String](1), fb.getArg[Locus](2), fb.getArg[Double](3)))

    val f = fb.result()()
    assert(f("GRCh38", Locus("20", 60001), 0.95) == grch37.liftoverLocus("GRCh38", Locus("20", 60001), 0.95))
    grch37.removeLiftover("GRCh38")
  }
}
