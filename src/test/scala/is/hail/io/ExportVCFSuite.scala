package is.hail.io

import is.hail.{SparkSuite, TestUtils}
import is.hail.annotations.Annotation
import is.hail.check.Gen
import is.hail.check.Prop._
import is.hail.expr._
import is.hail.utils._
import is.hail.variant.{VSMSubgen, Variant, VariantSampleMatrix}
import org.testng.annotations.Test

import scala.io.Source
import scala.language.postfixOps

class ExportVCFSuite extends SparkSuite {  
  @Test def testSameAsOrigBGzip() {
    val vcfFile = "src/test/resources/multipleChromosomes.vcf"
    val outFile = tmpDir.createTempFile("export", "vcf")

    val vdsOrig = hc.importVCF(vcfFile, nPartitions = Some(10))

    vdsOrig.exportVCF(outFile)

    assert(vdsOrig.same(hc.importVCF(outFile, nPartitions = Some(10)),
      tolerance = 1e-3))
  }

  @Test def testSameAsOrigNoCompression() {
    val vcfFile = "src/test/resources/multipleChromosomes.vcf"
    val outFile = tmpDir.createTempFile("export", "vcf")
    val outFile2 = tmpDir.createTempFile("export2", "vcf")

    val vdsOrig = hc.importVCF(vcfFile, nPartitions = Some(10))

    vdsOrig.exportVCF(outFile)

    val vdsNew = hc.importVCF(outFile, nPartitions = Some(10))

    assert(vdsOrig.same(vdsNew))

    val infoType = vdsNew.vaSignature.getAsOption[TStruct]("info").get
    val infoSize = infoType.size
    val toAdd = Annotation.fromSeq(Array.fill[Any](infoSize)(null))
    val (newVASignature, inserter) = vdsNew.insertVA(infoType, "info")

    val vdsNewMissingInfo = vdsNew.mapAnnotations(newVASignature,
      (v, va, gs) => inserter(va, toAdd))

    vdsNewMissingInfo.exportVCF(outFile2)

    assert(hc.importVCF(outFile2).same(vdsNewMissingInfo, 1e-2))
  }

  @Test def testSorted() {
    val vcfFile = "src/test/resources/multipleChromosomes.vcf"
    val outFile = tmpDir.createTempFile("sort", "vcf.bgz")

    val vdsOrig = hc.importVCF(vcfFile, nPartitions = Some(10))

    vdsOrig.exportVCF(outFile)

    val vdsNew = hc.importVCF(outFile, nPartitions = Some(10))

    assert(hadoopConf.readFile(outFile) { s =>
      Source.fromInputStream(s)
        .getLines()
        .filter(line => !line.isEmpty && line(0) != '#')
        .map(line => line.split("\t")).take(5).map(a => Variant(a(0), a(1).toInt, a(3), a(4))).toArray
    }.isSorted)
  }

  @Test def testReadWrite() {
    val out = tmpDir.createTempFile("foo", "vcf.bgz")
    val out2 = tmpDir.createTempFile("foo2", "vcf.bgz")
    val p = forAll(VariantSampleMatrix.gen(hc, VSMSubgen.random), Gen.choose(1, 10),
      Gen.choose(1, 10)) { case (vds, nPar1, nPar2) =>
      hadoopConf.delete(out, recursive = true)
      hadoopConf.delete(out2, recursive = true)
      vds.exportVCF(out)
      val vds2 = hc.importVCF(out, nPartitions = Some(nPar1))
      vds.exportVCF(out2)
      hc.importVCF(out2, nPartitions = Some(nPar2)).same(vds2)
    }

    p.check()
  }

  @Test def testEmptyReadWrite() {
    val vds = hc.importVCF("src/test/resources/sample.vcf").dropVariants()
    val out = tmpDir.createTempFile("foo", "vcf")
    val out2 = tmpDir.createTempFile("foo", "vcf.bgz")

    vds.exportVCF(out)
    vds.exportVCF(out2)

    assert(hadoopConf.getFileSize(out) > 0)
    assert(hadoopConf.getFileSize(out2) > 0)
    assert(hc.importVCF(out).same(vds))
    assert(hc.importVCF(out2).same(vds))
  }

  @Test def testGeneratedInfo() {
    val out = tmpDir.createTempFile("export", "vcf")
    hc.importVCF("src/test/resources/sample2.vcf")
      .annotateVariantsExpr("va.info.AC = va.info.AC, va.info.another = 5")
      .exportVCF(out)

    hadoopConf.readFile(out) { in =>
      Source.fromInputStream(in)
        .getLines()
        .filter(_.startsWith("##INFO"))
        .foreach { line =>
          assert(line.contains("Description="))
        }
    }
  }
  
  @Test def testVCFFormatHeader() {
    val out = tmpDir.createTempFile("export", "vcf")
    val vcfFile = "src/test/resources/sample2.vcf"

    val metadata = hc.parseVCFMetadata(vcfFile)

    hc.importVCF(vcfFile)
      .exportVCF(out, metadata = Some(metadata))

    val outFormatHeader = hadoopConf.readFile(out) { in =>
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

  @Test def testCastLongToIntAndOtherTypes() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf")
    
    // cast Long to Int
    val out = tmpDir.createTempFile("out", "vcf")
    vds
      .annotateVariantsExpr("va.info.AC_pass = gs.filter(g => g.gq >= 20 && g.dp >= 10 && " +
        "(!g.isHet() || ( (g.ad[1]/g.ad.sum()) >= 0.2 ) )).count()")
      .exportVCF(out)

    hadoopConf.readFile(out) { in =>
      Source.fromInputStream(in)
        .getLines()
        .filter(l => l.startsWith("##INFO") && l.contains("AC_pass"))
        .foreach { line =>
          assert(line.contains("Type=Integer"))
          assert(line.contains("Number=1"))
        }
    }

    // other valid types
    val out2 = tmpDir.createTempFile("out2", "vcf")
    vds
      .annotateVariantsExpr(
        "va.info.array = [\"foo\", \"bar\"]," +
          "va.info.set = [4, 5].toSet, " +
          "va.info.float = let x = 5.0 in x.toFloat64(), " +
          "va.info.bool = true")
      .exportVCF(out2)
    
    hadoopConf.readFile(out2) { in =>
      Source.fromInputStream(in)
        .getLines()
        .filter(l => l.startsWith("##INFO"))
        .foreach { l =>
          if (l.contains("array")) {
            assert(l.contains("Type=String"))
            assert(l.contains("Number=."))
          } else
          if (l.contains("set")) {
            assert(l.contains("Type=Integer"))
            assert(l.contains("Number=."))  
          } else
          if (l.contains("float")) {
            assert(l.contains("Type=Float"))
            assert(l.contains("Number=1"))
          } else
          if (l.contains("bool")) {
            assert(l.contains("Type=Flag"))
            assert(l.contains("Number=0"))
          }
        }
    }
  }

  @Test def testErrors() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf", dropSamples = true)
    
    val out = tmpDir.createLocalTempFile("foo", "vcf")
    TestUtils.interceptFatal("INFO field 'foo': VCF does not support type") {
      vds
        .annotateVariantsExpr("va.info.foo = [[1]]")
        .exportVCF(out)
    }
    
    TestUtils.interceptFatal("INFO field 'foo': VCF does not support type") {
      vds
        .annotateVariantsExpr("va.info.foo = [Call(3)]")
        .exportVCF(out)
    }

    TestUtils.interceptFatal("INFO field 'foo': VCF does not support type") {
      vds
        .annotateVariantsExpr("va.info.foo = v")
        .exportVCF(out)
    }
    
    TestUtils.interceptSpark("Cannot convert Long to Int") {
      vds
        .annotateVariantsExpr("va.info.foo = 2147483648L")
        .exportVCF(out)
    }

    TestUtils.interceptFatal("INFO field 'foo': VCF does not support type") {
      vds
        .annotateVariantsExpr("va.info.foo = [true]")
        .exportVCF(out)
    }

    TestUtils.interceptFatal("INFO field 'foo': VCF does not support type") {
      vds
        .annotateVariantsExpr("va.info.foo = {INT: 5}")
        .exportVCF(out)
    }
    
    TestUtils.interceptFatal("export_vcf requires g to have type TStruct") {
      vds
        .annotateGenotypesExpr("g = 5")
        .exportVCF(out)
    }

    TestUtils.interceptFatal("Invalid type for format field `BOOL'. Found Boolean.") {
      vds
        .annotateGenotypesExpr("g = {BOOL: true}")
        .exportVCF(out)
    }
    
    TestUtils.interceptFatal("Invalid type for format field `AA'.") {
      vds
        .annotateGenotypesExpr("g = {AA: [[0]]}")
        .exportVCF(out)
    }
  }

  @Test def testInfoFieldSemicolons() {
    val vds = hc.importVCF("src/test/resources/sample.vcf", dropSamples = true)
      .annotateVariantsExpr("va.info = {foo: 5, bar: NA: Int}")

    val out = tmpDir.createLocalTempFile("foo", "vcf")
    vds.exportVCF(out)
    hadoopConf.readLines(out) { lines =>
      lines.foreach { l =>
        if (!l.value.startsWith("#")) {
          assert(l.value.contains("foo=5"))
          assert(!l.value.contains("foo=5;"))
        }
      }
    }
  }
  
  @Test def testGenotypes() {
    val vds = hc.importVCF("src/test/resources/sample.vcf")

    val out = tmpDir.createLocalTempFile("foo", "vcf")
    vds.exportVCF(out)
    hadoopConf.readLines(out) { lines =>
      lines.foreach { l =>
        if (l.value.startsWith("20\t13029920")) {
          assert(l.value.contains("GT:AD:DP:GQ:PL\t1/1:0,6:6:18:234,18,0\t1/1:0,4:4:12:159,12,0\t" +
            "1/1:0,4:4:12:163,12,0\t1/1:0,12:12:36:479,36,0\t1/1:0,4:4:12:149,12,0\t1/1:0,6:6:18:232,18,0\t" +
            "1/1:0,6:6:18:242,18,0\t1/1:0,3:3:9:119,9,0\t1/1:0,9:9:27:374,27,0\t./.:1,0:1:.:.\t1/1:0,3:3:9:133,9,0"))
        }
      }
    }
  }
  
  def genFormatFieldVCF: Gen[Type] = Gen.oneOf[Type](
      TInt32(), TFloat32(), TFloat64(), TString(), TCall(),
      TArray(TInt32()), TArray(TFloat32()), TArray(TFloat64()), TArray(TString()), TArray(TCall()),
      TSet(TInt32()), TSet(TFloat32()), TSet(TFloat64()), TSet(TString()), TSet(TCall()))
  
  def genFormatStructVCF: Gen[TStruct] =
    Gen.buildableOf[Array, (String, Type)](
      Gen.zip(Gen.identifier, genFormatFieldVCF))
      .filter(fields => fields.map(_._1).areDistinct())
      .map(fields => TStruct(fields
        .iterator
        .zipWithIndex
        .map { case ((k, t), i) => Field(k, t, i) }
        .toIndexedSeq))
  
  @Test def testWriteGenericFormatField() {
    val genericFormatFieldVCF:  VSMSubgen = VSMSubgen.random.copy(
      vaSigGen = Gen.const(TStruct.empty()),
      tSigGen = genFormatStructVCF,
      tGen = (t: Type, v: Annotation) => t.genValue)
    
    val out = tmpDir.createTempFile("foo", "vcf.bgz")
    val p = forAll(VariantSampleMatrix.gen(hc, genericFormatFieldVCF)) { vsm =>
        hadoopConf.delete(out, recursive = true)
        vsm.exportVCF(out)
      
        true
      }

    p.check()
  }
  
  @Test def testContigs() {
    val vds = hc.importVCF("src/test/resources/sample.vcf", dropSamples = true)

    val out = tmpDir.createLocalTempFile("foo", "vcf")
    vds.exportVCF(out)
    assert(hadoopConf.readLines(out) { lines =>
      lines.filter(_.value.startsWith("##contig=<ID=10")).forall { l =>
        l.value == "##contig=<ID=10,length=135534747,assembly=GRCh37>"
      }
    })
  }

  @Test def testMetadata() {
    val vcfFile = "src/test/resources/multipleChromosomes.vcf"
    val outFile = tmpDir.createTempFile("export", "vcf")
    val vdsOrig = hc.importVCF(vcfFile, nPartitions = Some(10))

    val md = Some(Map(
      "filters" -> Map("LowQual" -> Map("Description" -> "Low quality")),
      "format" -> Map("GT" -> Map("Description" -> "Genotype call.", "Number" -> "foo")),
      "fakeField" -> Map.empty[String, Map[String, String]]))

    vdsOrig.exportVCF(outFile, metadata = md)
    assert(hadoopConf.readLines(outFile) { lines =>
      lines.filter(l => l.value.startsWith("##FORMAT=<ID=GT") || l.value.startsWith("##FILTER=<ID=LowQual")).forall { l =>
        l.value == "##FORMAT=<ID=GT,Number=foo,Type=String,Description=\"Genotype call.\">" ||
        l.value == "##FILTER=<ID=LowQual,Description=\"Low quality\">"
      }
    })
  }
}
