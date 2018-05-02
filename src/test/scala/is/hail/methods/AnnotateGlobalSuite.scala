package is.hail.methods

import is.hail.{SparkSuite, TestUtils}
import is.hail.annotations.Annotation
import is.hail.expr.types._
import is.hail.utils._
import is.hail.testUtils._
import org.apache.spark.sql.Row
import org.apache.spark.util.StatCounter
import org.testng.annotations.Test

class AnnotateGlobalSuite extends SparkSuite {
  @Test def testTable() {
    val out1 = tmpDir.createTempFile("file1", ".txt")

    val toWrite1 = Array(
      "GENE\tPLI\tEXAC_LOF_COUNT",
      "Gene1\t0.12312\t2",
      "Gene2\t0.99123\t0",
      "Gene3\tNA\tNA",
      "Gene4\t0.9123\t10",
      "Gene5\t0.0001\t202")

    hadoopConf.writeTextFile(out1) { out =>
      toWrite1.foreach(line => out.write(line + "\n"))
    }

    val kt = hc.importTable(out1, impute = true)
    val vds = hc.importVCF("src/test/resources/sample.vcf")
      .annotateGlobal(kt.collect().toFastIndexedSeq, TArray(kt.signature), "genes")

    val (t, res) = vds.queryGlobal("global.genes")

    assert(t == TArray(TStruct(
      ("GENE", TString()),
      ("PLI", TFloat64()),
      ("EXAC_LOF_COUNT", TInt32()))))

    assert(res == IndexedSeq(
      Annotation("Gene1", "0.12312".toDouble, 2),
      Annotation("Gene2", "0.99123".toDouble, 0),
      Annotation("Gene3", null, null),
      Annotation("Gene4", "0.9123".toDouble, 10),
      Annotation("Gene5", "0.0001".toDouble, 202)
    ))

  }
}
