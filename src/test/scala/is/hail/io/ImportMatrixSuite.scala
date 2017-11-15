package is.hail.io

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.check.Gen
import is.hail.check.Prop.forAll
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
    val genMatrix = VSMSubgen[String, String, Annotation](
      sSigGen = Gen.const(TString()),
      saSigGen = Gen.const(TStruct.empty()),
      vSigGen = Gen.const(TString()),
      vaSigGen = Gen.const(TStruct.empty()),
      globalSigGen = Gen.const(TStruct.empty()),
      tSigGen = Gen.zip(Gen.oneOf[Type](TInt32(), TInt64(), TFloat32(), TFloat64(), TString()), Gen.coin(0.2))
        .map{ case (typ, req) => typ.setRequired(req) },
      sGen = (t: Type) => Gen.identifier.map(s => s: Annotation),
      saGen = (t: Type) => t.genValue,
      vaGen = (t: Type) => t.genValue,
      globalGen = (t: Type) => t.genValue,
      vGen = (t: Type) => Gen.identifier,
      tGen = (t: Type, v: String) => t.genValue,
      makeKOk = _ => null)
    forAll(VariantSampleMatrix.gen(hc, genMatrix)) { vds =>

      val truth = {
        val f = tmpDir.createTempFile(extension="txt")
        vds.makeKT("v = v", "g = g", Array("v")).export(f)
        hc.importMatrix(f)
      }

      val actual = {
        val f = tmpDir.createTempFile(extension="vcf")
        truth.toVDS.exportVCF(f)
        hc.importVCF(f)
      }

      truth.same(actual)
    }.check()

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