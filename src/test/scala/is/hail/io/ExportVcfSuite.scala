package is.hail.io

import is.hail.{SparkSuite, TestUtils}
import is.hail.annotations.Annotation
import is.hail.check.Arbitrary.arbitrary
import is.hail.check.Gen
import is.hail.check.Prop._
import is.hail.expr._
import is.hail.utils._
import is.hail.variant.{Locus, VSMSubgen, Variant, VariantSampleMatrix}
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
    hc.importVCF("src/test/resources/sample2.vcf")
      .exportVCF(out)

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

  @Test def testCastLongToInt() {
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
          }
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
    val vds = hc.importVCF("src/test/resources/sample2.vcf")
    
    println(Int.MaxValue)
    
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
  
  def genFormatFieldVCF: Gen[Type] = Gen.oneOf[Type](
      TInt32(), TFloat32(), TFloat64(), TString(), TCall(),
      TArray(TInt32()), TArray(TFloat32()), TArray(TFloat64()), TArray(TString()), TArray(TCall()),
      TSet(TInt32()), TSet(TFloat32()), TSet(TFloat64()), TSet(TString()), TSet(TCall()))
  
  def genFormatStructVCF: Gen[TStruct] =
    Gen.buildableOf[Array, (String, Type, Map[String, String])](
      Gen.zip(Gen.identifier,
        genFormatFieldVCF,
        Gen.option(
          Gen.buildableOf2[Map, String, String](
            Gen.zip(arbitrary[String].filter(s => !s.isEmpty), arbitrary[String])), someFraction = 0.05)
          .map(o => o.getOrElse(Map.empty[String, String]))))
      .filter(fields => fields.map(_._1).areDistinct())
      .map(fields => TStruct(fields
        .iterator
        .zipWithIndex
        .map { case ((k, t, m), i) => Field(k, t, i, m) }
        .toIndexedSeq))
  
  @Test def testWriteGenericFormatField() {
    val genericFormatFieldVCF:  VSMSubgen[Locus, Variant, Annotation] = VSMSubgen.random.copy(
      tSigGen = genFormatStructVCF,
      tGen = (t: Type, v: Variant) => t.genValue.resize(20))
    
    val untestedFields = Set("rsid", "qual", "info")
    
    val out = tmpDir.createTempFile("foo", "vcf.bgz")
    val p = forAll(VariantSampleMatrix.gen(hc, genericFormatFieldVCF), Gen.choose(1, 10),
      Gen.choose(1, 10)) { case (vkds, nPar1, nPar2) =>
      println(vkds.rowType)
      
      vkds.vaSignature match {
        case t: TStruct if untestedFields.intersect(t.fieldNames.toSet).isEmpty => 
        case _ =>
           hadoopConf.delete(out, recursive = true)
           vkds.exportVCF(out)
      }
      true
    }

    p.check()
  }
}
