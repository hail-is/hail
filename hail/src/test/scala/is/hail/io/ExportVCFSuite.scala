package is.hail.io

import is.hail.annotations.Annotation
import is.hail.check.Gen
import is.hail.check.Prop._
import is.hail.expr.types.virtual._
import is.hail.io.vcf.ExportVCF
import is.hail.utils._
import is.hail.variant.{Locus, MatrixTable, VSMSubgen}
import is.hail.{HailSuite, TestUtils}
import org.testng.annotations.Test

import scala.io.Source
import scala.language.postfixOps

class ExportVCFSuite extends HailSuite {

  @Test def testSameAsOrigBGzip() {
    val vcfFile = "src/test/resources/multipleChromosomes.vcf"
    val outFile = tmpDir.createTempFile("export", "vcf")

    val vdsOrig = TestUtils.importVCF(hc, vcfFile, nPartitions = Some(10))

    ExportVCF(vdsOrig, outFile)

    assert(vdsOrig.same(TestUtils.importVCF(hc, outFile, nPartitions = Some(10)),
      tolerance = 1e-3))
  }

  @Test def testSorted() {
    val vcfFile = "src/test/resources/multipleChromosomes.vcf"
    val outFile = tmpDir.createTempFile("sort", "vcf.bgz")

    val vdsOrig = TestUtils.importVCF(hc, vcfFile, nPartitions = Some(10))

    ExportVCF(vdsOrig, outFile)

    val vdsNew = TestUtils.importVCF(hc, outFile, nPartitions = Some(10))

    implicit val locusAllelesOrdering = vdsNew.rowKeyStruct.ordering.toOrdering

    assert(sFS.readFile(outFile) { s =>
      Source.fromInputStream(s)
        .getLines()
        .filter(line => !line.isEmpty && line(0) != '#')
        .map(line => line.split("\t")).take(5).map(a => Annotation(Locus(a(0), a(1).toInt), FastIndexedSeq(a(3), a(4)))).toArray
    }.isSorted)
  }

  @Test def testReadWrite() {
    val out = tmpDir.createTempFile("foo", "vcf.bgz")
    val out2 = tmpDir.createTempFile("foo2", "vcf.bgz")
    val p = forAll(MatrixTable.gen(hc, VSMSubgen.random), Gen.choose(1, 10),
      Gen.choose(1, 10)) { case (vds, nPar1, nPar2) =>
      sFS.delete(out, recursive = true)
      sFS.delete(out2, recursive = true)
      ExportVCF(vds, out)
      val vds2 = TestUtils.importVCF(hc, out, nPartitions = Some(nPar1), rg = Some(vds.referenceGenome))
      ExportVCF(vds, out2)
      TestUtils.importVCF(hc, out2, nPartitions = Some(nPar2), rg = Some(vds.referenceGenome)).same(vds2)
    }

    p.check()
  }

  @Test def testEmptyReadWrite() {
    val vds = TestUtils.importVCF(hc, "src/test/resources/sample.vcf").dropRows()
    val out = tmpDir.createTempFile("foo", "vcf")
    val out2 = tmpDir.createTempFile("foo", "vcf.bgz")

    ExportVCF(vds, out)
    ExportVCF(vds, out2)

    assert(sFS.getFileSize(out) > 0)
    assert(sFS.getFileSize(out2) > 0)
    assert(TestUtils.importVCF(hc, out).same(vds))
    assert(TestUtils.importVCF(hc, out2).same(vds))
  }

  @Test def testVCFFormatHeader() {
    val out = tmpDir.createTempFile("export", "vcf")
    val vcfFile = "src/test/resources/sample2.vcf"

    val metadata = hc.parseVCFMetadata(vcfFile)

    ExportVCF(TestUtils.importVCF(hc, vcfFile), out, metadata = Some(metadata))

    val outFormatHeader = sFS.readFile(out) { in =>
      Source.fromInputStream(in)
        .getLines()
        .filter(_.startsWith("##FORMAT"))
        .mkString("\n")
    }

    val vcfFormatHeader =
      """##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
        |##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths for the ref and alt alleles in the order listed">
        |##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
        |##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
        |##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Normalized, Phred-scaled likelihoods for genotypes as defined in the VCF specification">""".stripMargin

    assert(outFormatHeader == vcfFormatHeader)
  }

  @Test def testGenotypes() {
    val vds = TestUtils.importVCF(hc, "src/test/resources/sample.vcf")

    val out = tmpDir.createLocalTempFile("foo", "vcf")
    ExportVCF(vds, out)
    sFS.readLines(out) { lines =>
      lines.foreach { l =>
        if (l.value.startsWith("20\t13029920")) {
          assert(l.value.contains("GT:AD:DP:GQ:PL\t1/1:0,6:6:18:234,18,0\t1/1:0,4:4:12:159,12,0\t" +
            "1/1:0,4:4:12:163,12,0\t1/1:0,12:12:36:479,36,0\t1/1:0,4:4:12:149,12,0\t1/1:0,6:6:18:232,18,0\t" +
            "1/1:0,6:6:18:242,18,0\t1/1:0,3:3:9:119,9,0\t1/1:0,9:9:27:374,27,0\t./.:1,0:1\t1/1:0,3:3:9:133,9,0"))
        }
      }
    }
  }

  def genFormatFieldVCF: Gen[Type] = Gen.oneOf[Type](
    TInt32(), TFloat32(), TFloat64(), TString(), TCall(),
    TArray(TInt32()), TArray(TFloat32()), TArray(TFloat64()), TArray(TString()), TArray(TCall()),
    TSet(TInt32()), TSet(TFloat32()), TSet(TFloat64()), TSet(TString()), TSet(TCall()))

  def genFormatStructVCF: Gen[TStruct] =
    Gen.buildableOf[Array](
      Gen.zip(Gen.identifier, genFormatFieldVCF))
      .filter(fields => fields.map(_._1).areDistinct())
      .map(fields => TStruct(fields
        .iterator
        .zipWithIndex
        .map { case ((k, t), i) => Field(k, t, i) }
        .toFastIndexedSeq))

  @Test def testContigs() {
    val vds = TestUtils.importVCF(hc, "src/test/resources/sample.vcf", dropSamples = true)

    val out = tmpDir.createLocalTempFile("foo", "vcf")
    ExportVCF(vds, out)
    assert(sFS.readLines(out) { lines =>
      lines.filter(_.value.startsWith("##contig=<ID=10")).forall { l =>
        l.value == "##contig=<ID=10,length=135534747,assembly=GRCh37>"
      }
    })
  }

  @Test def testMetadata() {
    val vcfFile = "src/test/resources/multipleChromosomes.vcf"
    val outFile = tmpDir.createTempFile("export", "vcf")
    val vdsOrig = TestUtils.importVCF(hc, vcfFile, nPartitions = Some(10))

    val md = Some(Map(
      "filters" -> Map("LowQual" -> Map("Description" -> "Low quality")),
      "format" -> Map("GT" -> Map("Description" -> "Genotype call.", "Number" -> "foo")),
      "fakeField" -> Map.empty[String, Map[String, String]]))

    ExportVCF(vdsOrig, outFile, metadata = md)
    assert(sFS.readLines(outFile) { lines =>
      lines.filter(l => l.value.startsWith("##FORMAT=<ID=GT") || l.value.startsWith("##FILTER=<ID=LowQual")).forall { l =>
        l.value == "##FORMAT=<ID=GT,Number=foo,Type=String,Description=\"Genotype call.\">" ||
          l.value == "##FILTER=<ID=LowQual,Description=\"Low quality\">"
      }
    })
  }
}
