package is.hail.variant

import is.hail.SparkSuite
import is.hail.check.Prop._
import is.hail.expr.types._
import is.hail.utils._
import is.hail.testUtils._
import is.hail.io.vcf.ExportVCF
import org.testng.annotations.Test

class GenericDatasetSuite extends SparkSuite {

  @Test def testReadWrite() {
    val path = tmpDir.createTempFile(extension = "vds")

    val p = forAll(MatrixTable.genGeneric(hc)) { gds =>
      val f = tmpDir.createTempFile(extension = "vds")
      gds.write(f)
      hc.readGDS(f).same(gds)
    }

    p.check()
  }

  @Test def testAnnotateFilterExpr() {
    val gds = hc.importVCF("src/test/resources/sample.vcf.bgz", nPartitions = Some(4))
    val gdsAnnotated = gds.annotateGenotypesExpr("g.a = 5, g.b = 7.0, g.c = \"foo\"")

    val gsig = gdsAnnotated.genotypeSignature.asInstanceOf[TStruct]
    val expTypes = Array(TInt32(), TFloat64(), TString())
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
    val gds_exportvcf_path = tmpDir.createTempFile(extension = "vcf")
    val gds = hc.importVCF("src/test/resources/sample.vcf.bgz", nPartitions = Some(4))
    ExportVCF(gds, gds_exportvcf_path)
    assert(gds.same(hc.importVCF(gds_exportvcf_path)))

    // not TGenotype or TStruct signature
    intercept[HailException] {
      val path = tmpDir.createTempFile(extension = "vcf")
      ExportVCF(gds
        .annotateGenotypesExpr("g = 5"),
        path)
    }

    // struct field
    intercept[HailException] {
      val path = tmpDir.createTempFile(extension = ".vcf")
      ExportVCF(gds
        .annotateGenotypesExpr("g.a = 5, g.b = 7.0, g.c = \"foo\", g.d = {gene: 5}"),
        path)
    }

    // nested arrays
    intercept[HailException] {
      val path = tmpDir.createTempFile(extension = ".vcf")
      ExportVCF(gds
        .annotateGenotypesExpr("g.a = 5, g.b = 7.0, g.c = \"foo\", g.d = [[1, 5], [2], [3, 4]]"),
        path)
    }

    // nested set
    intercept[HailException] {
      val path = tmpDir.createTempFile(extension = ".vcf")
      ExportVCF(gds
        .annotateGenotypesExpr("g.dpset = g.PL.map(pl => [pl]).toSet()"),
        path)
    }

    val path = tmpDir.createTempFile(extension = ".vcf")
    val path2 = tmpDir.createTempFile(extension = ".vds")

    ExportVCF(gds, path)

    hc.importVCF(path).write(path2)
  }

  @Test def testPersistCoalesce() {
    val vcf = "src/test/resources/sample.vcf.bgz"

    val gds_cache = hc.importVCF(vcf).cache()
    val gds_persist = hc.importVCF(vcf).persist("MEMORY_AND_DISK")
    val gds_coalesce = hc.importVCF(vcf).coalesce(5)

    assert(gds_cache.storageLevel == "MEMORY_ONLY" &&
      gds_persist.storageLevel == "MEMORY_AND_DISK" &&
      gds_coalesce.nPartitions == 5)
  }
}
