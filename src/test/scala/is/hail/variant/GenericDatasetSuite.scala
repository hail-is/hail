package is.hail.variant

import is.hail.SparkSuite
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

    val gds2 = hc.readGDS(path)
    assert(gds same gds2)
  }
}
