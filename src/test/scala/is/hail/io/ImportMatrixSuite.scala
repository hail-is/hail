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

  def exportAsTable(vsm: VariantSampleMatrix[String, String, Annotation], file: String) {
    val cellTypeBc = sc.broadcast(vsm.genotypeSignature)
    vsm.makeKT("v = v", "g = g", Array("v")).rdd.mapPartitions { it =>
      val sb = new StringBuilder()
      val cellType = cellTypeBc.value

      it.map { r =>
        sb.clear()

        (0 until r.size).foreachBetween { i =>
          sb.append(TableAnnotationImpex.exportAnnotation(r.get(i), if  (i == 0) TString() else cellType))
        }(sb += '\t')

        sb.result()
      }
    }.writeTable(file, tmpDir.toString(), header = Option(vsm.sampleIds.map(_.toString()).mkString("\t")))
  }

  @Test def testTypes() {
    implicit val KOk = TString().typedOrderedKey[String, String]
    val genMatrix = VSMSubgen[String, String, Annotation](
      sSigGen = Gen.const(TString()),
      saSigGen = Gen.const(TStruct.empty()),
      vSigGen = Gen.const(TString()),
      vaSigGen = Gen.const(TStruct.empty()),
      globalSigGen = Gen.const(TStruct.empty()),
      tSigGen = Gen.zip(Gen.oneOf[Type](TInt32(), TInt64(), TFloat32(), TFloat64(), TString()), Gen.coin(0.2))
        .map{ case (typ, req) => typ.setRequired(req) },
      sGen = (t: Type) => Gen.identifier.map(s => s: Annotation),
      saGen = (t: Type) => t.genNonmissingValue,
      vaGen = (t: Type) => t.genNonmissingValue,
      globalGen = (t: Type) => t.genNonmissingValue,
      vGen = (t: Type) => Gen.identifier,
      tGen = (t: Type, v: String) => t.genValue,
      makeKOk = _ => KOk)

    forAll(VariantSampleMatrix.gen(hc, genMatrix)) { vsm =>
      val actual: VariantSampleMatrix[String, String, Annotation] = {
        val f = tmpDir.createTempFile(extension="txt")
        exportAsTable(vsm, f)
        LoadMatrix(hc, Array(f), cellType = vsm.genotypeSignature)
      }
      assert(vsm.same(actual))

      val tmp1 = tmpDir.createTempFile(extension = "vds")
      vsm.write(tmp1, true)

      val vsm2 = VariantSampleMatrix.read(hc, tmp1).asInstanceOf[VariantSampleMatrix[String, String, Annotation]]
      assert(vsm.same(vsm2))
      true
    }.check()

  }
}