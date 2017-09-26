package is.hail.io

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.variant.VariantSampleMatrix
import org.apache.spark.SparkException
import org.testng.annotations.Test
import is.hail.utils._

class ImportMatrixSuite extends SparkSuite {

  @Test def testHeadersNotIdentical() {
    val files =hc.hadoopConf.globAll(List("src/test/resources/sampleheader*.txt"))
    val e = intercept[SparkException] {
      val vsm = LoadMatrix(hc, files(0), files)
    }
    assert(e.getMessage.contains("invalid sample ids"))

  }

  @Test def testMissingVals() {
    val files =hc.hadoopConf.globAll(List("src/test/resources/samplesmissing.txt"))
    val e = intercept[SparkException] {
      val vsm = LoadMatrix(hc, files(0), files)
      vsm.rdd.count()
    }
    assert(e.getMessage.contains("number of elements"))
  }

  @Test def testLoadMatrix() {
    val files =hc.hadoopConf.globAll(List("src/test/resources/samplematrix*.txt"))
    val vsm = LoadMatrix(hc, files(0), files)

    assert(vsm.sampleIds.length == 20)
    vsm.rdd.collect()(1)._2._2.foreach {i => assert(i==0)}

  }

  @Test def testReadWrite() {
    val files =hc.hadoopConf.globAll(List("src/test/resources/samplematrix*.txt"))
    val vsm = LoadMatrix(hc, files(0), files)

    val tmp1 = tmpDir.createTempFile("readtest", ".vds")
    vsm.write(tmp1, true)

    val vsm2 = VariantSampleMatrix.read(hc,tmp1).toGDS
    assert(vsm2.same(vsm))
  }

}