package is.hail.variant

import is.hail.SparkSuite
import is.hail.check.Prop._
import is.hail.utils._
import is.hail.expr.TGenotype
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
}
