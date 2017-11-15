package is.hail.io

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.check.Gen
import is.hail.expr._
import is.hail.variant.{VSMSubgen, VariantSampleMatrix}
import org.apache.spark.SparkException
import org.testng.annotations.Test
import is.hail.utils._

class ImportMatrixSuite extends SparkSuite {

  @Test def testHeadersNotIdentical() {
    val files = hc.hadoopConf.globAll(List("src/test/resources/sampleheader*.txt"))
    val e = intercept[SparkException] {
      val vsm = LoadMatrix(hc, files)
    }
    assert(e.getMessage.contains("invalid sample ids"))
  }

  @Test def testMissingVals() {
    val files = hc.hadoopConf.globAll(List("src/test/resources/samplesmissing.txt"))
    val e = intercept[SparkException] {
      val vsm = LoadMatrix(hc, files)
      vsm.rdd.count()
    }
    assert(e.getMessage.contains("number of elements"))
  }

  @Test def testLoadMatrix() {
    val files = hc.hadoopConf.globAll(List("src/test/resources/samplematrix*.txt"))
    val vsm = LoadMatrix(hc, files)

    assert(vsm.sampleIds.length == 20)
    vsm.rdd.collect()(1)._2._2.foreach { i => assert(i == 0) }
  }

  @Test def testTypes() {
    val files = hc.hadoopConf.globAll(List("src/test/resources/samplematrix*.txt"))
    val vsm = LoadMatrix(hc, files, cellType = TFloat64())

    assert(vsm.sampleIds.length == 20)
    vsm.rdd.collect()(1)._2._2.foreach { i => assert(i == 0) }

    val vsm2 = LoadMatrix(hc, files, cellType = TFloat32())

    assert(vsm2.sampleIds.length == 20)
    vsm2.rdd.collect()(1)._2._2.foreach { i => assert(i == 0) }

    val vsm3 = LoadMatrix(hc, files, cellType = TInt32())

    assert(vsm3.sampleIds.length == 20)
    vsm3.rdd.collect()(1)._2._2.foreach { i => assert(i == 0) }

    val vsm4 = LoadMatrix(hc, files, cellType = TString())

    assert(vsm4.sampleIds.length == 20)
    vsm4.rdd.collect()(1)._2._2.foreach { i => assert(i == "0") }
  }

  @Test def testTypes2() {
    VSMSubgen[String, String, Annotation](
      sSigGen = Gen.const(TString()),
      saSigGen = Gen.const(TStruct.empty()),
      vSigGen = Gen.const(TString()),
      vaSigGen: Gen[Type],
      globalSigGen: Gen[Type],
      tSigGen: Gen[Type],
      sGen: (Type) => Gen[Annotation],
      saGen: (Type) => Gen[Annotation],
      vaGen: (Type) => Gen[Annotation],
      globalGen: (Type) => Gen[Annotation],
      vGen: (Type) => Gen[RK],
      tGen: (Type, RK) => Gen[T],
      wasSplit: Boolean = false,
      makeKOk: (Type) => OrderedKey[RPK, RK]) {
  }

  @Test def testReadWrite() {
    val files = hc.hadoopConf.globAll(List("src/test/resources/samplematrix*.txt"))
    val vsm = LoadMatrix(hc, files)

    val tmp1 = tmpDir.createTempFile("readtest", ".vds")
    vsm.write(tmp1, true)

    val vsm2 = VariantSampleMatrix.read(hc, tmp1).toGDS
    assert(vsm2.same(vsm))
  }

}