package is.hail.variant

import is.hail.SparkSuite
import is.hail.check.Prop._
import is.hail.utils._
import is.hail.expr.{TDouble, TGenotype, TInt, TString, TStruct}
import org.testng.annotations.Test

class GenericDatasetSuite extends SparkSuite {

  @Test def testReadWrite() {
    val path = tmpDir.createTempFile(extension = ".vds")

    val vds = hc.importVCF("src/test/resources/sample.vcf.bgz", nPartitions = Some(4))
    assert(!vds.isGenericGenotype)

    val gds = vds.toGDS
    assert(gds.isGenericGenotype && gds.genotypeSignature == TGenotype)

    gds.write(path)

    intercept[FatalException] {
      hc.read(path)
    }

    assert(gds same hc.readGDS(path))

    val p = forAll(VariantSampleMatrix.genGeneric(hc)) { gds =>
      val f = tmpDir.createTempFile(extension = "vds")
      gds.write(f)
      hc.readGDS(f).same(gds)
    }

    p.check()
  }

  @Test def testAnnotateFilterExpr() {
    val gds = hc.importVCFGeneric("src/test/resources/sample.vcf.bgz", nPartitions = Some(4))
    val gdsAnnotated = gds.annotateGenotypesExpr("g.a = 5, g.b = 7.0, g.c = \"foo\"")

    val gsig = gdsAnnotated.genotypeSignature.asInstanceOf[TStruct]
    val expTypes = Array(TInt, TDouble, TString)
    val expNames = Array("a", "b", "c")

    val (_, aQuerier) = gdsAnnotated.queryGA("g.a")
    val (_, bQuerier) = gdsAnnotated.queryGA("g.b")
    val (_, cQuerier) = gdsAnnotated.queryGA("g.c")

    assert(expNames.zip(expTypes).forall { case (n, t) => gsig.hasField(n) && gsig.fields(gsig.fieldIdx(n)).typ == t })
    assert(gdsAnnotated.rdd.forall { case (v, (va, gs)) => gs.forall { a =>
      aQuerier(a).asInstanceOf[Int] == 5 &&
        bQuerier(a).asInstanceOf[Double] == 7.0 &&
        cQuerier(a).asInstanceOf[String] == "foo"
    }
    })

    assert(gdsAnnotated.filterGenotypes("g.a != 5").rdd.forall { case (v, (va, gs)) => gs.forall(_ == null) })
  }

  @Test def testExportVCF() {
    val gds_exportvcf_path = tmpDir.createTempFile(extension = ".vcf")
    val gds = hc.importVCFGeneric("src/test/resources/sample.vcf.bgz", nPartitions = Some(4))
    gds.exportVCF(gds_exportvcf_path)
    assert(gds.same(hc.importVCFGeneric(gds_exportvcf_path)))

    // not TGenotype or TStruct signature
    intercept[FatalException] {
      val path = tmpDir.createTempFile(extension = ".vcf")
      gds
        .annotateGenotypesExpr("g = 5")
        .exportVCF(path)
    }

    // struct field
    intercept[FatalException] {
      val path = tmpDir.createTempFile(extension = ".vcf")
      gds
        .annotateGenotypesExpr("g.a = 5, g.b = 7.0, g.c = \"foo\", g.d = {gene: 5}")
        .exportVCF(path)
    }

    // nested arrays
    intercept[FatalException] {
      val path = tmpDir.createTempFile(extension = ".vcf")
      gds
        .annotateGenotypesExpr("g.a = 5, g.b = 7.0, g.c = \"foo\", g.d = [[1, 5], [2], [3, 4]]")
        .exportVCF(path)
    }

    // nested set
    intercept[FatalException] {
      val path = tmpDir.createTempFile(extension = ".vcf")
      gds
        .annotateGenotypesExpr("g.dpset = g.PL.map(pl => [pl]).toSet()")
        .exportVCF(path)
    }

    val path = tmpDir.createTempFile(extension = ".vcf")
    val path2 = tmpDir.createTempFile(extension = ".vds")

    gds
      .annotateGenotypesExpr("g = Genotype(g.GT)")
      .toVDS
      .exportVCF(path)

    hc.importVCF(path).write(path2)
  }

  @Test def testCount() {
    val vcf = "src/test/resources/sample.vcf.bgz"
    val vds = hc.importVCF(vcf)
    val gds = hc.importVCFGeneric(vcf).annotateGenotypesExpr("g = g.GT")
    assert(vds.count(countGenotypes = true) == gds.count(countGenotypes = true))
  }

  @Test def testPersistCoalesce() {
    val vcf = "src/test/resources/sample.vcf.bgz"

    val gds_cache = hc.importVCFGeneric(vcf).cache()
    val gds_persist = hc.importVCFGeneric(vcf).persist("MEMORY_AND_DISK")
    val gds_coalesce = hc.importVCFGeneric(vcf).coalesce(5)

    assert(gds_cache.storageLevel == "MEMORY_ONLY" &&
      gds_persist.storageLevel == "MEMORY_AND_DISK" &&
      gds_coalesce.nPartitions == 5)
  }

  @Test def testExportGenotypes() {
    val gds = hc.importVCFGeneric("src/test/resources/sample.vcf.bgz").annotateGenotypesExpr("g = g.GT")
    val path = tmpDir.createTempFile("testExportGenotypes", ".tsv")
    gds.exportGenotypes(path, "s, v, g", false)
    val countResult = gds.count(countGenotypes = true).nCalled.getOrElse(0)
    assert(sc.textFile(path).count() == countResult)
  }
}
