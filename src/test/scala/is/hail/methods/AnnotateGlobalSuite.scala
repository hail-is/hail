package is.hail.methods

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.expr.typ.{TArray, TStruct}
import is.hail.utils._
import is.hail.testUtils._
import org.apache.spark.util.StatCounter
import org.testng.annotations.Test

class AnnotateGlobalSuite extends SparkSuite {
  @Test def test() {

    var vds = hc.importVCF("src/test/resources/sample2.vcf")
    vds = SplitMulti(vds)
    vds = VariantQC(vds)
    vds = SampleQC(vds)

    val (afDist, _) = vds.queryVariants("variants.map(v => va.qc.AF).stats()")
    val (singStats, _) = vds.querySamples("samples.filter(s => sa.qc.nSingleton > 2).count()")
    val (acDist, _) = vds.queryVariants("variants.map(v => va.qc.AC).stats()")
    val (crStats, _) = vds.querySamples("samples.map(s => sa.qc.callRate).stats()")

    val qSingleton = vds.querySA("sa.qc.nSingleton")._2

    val sCount = vds.sampleAnnotations.count(sa =>
      qSingleton(sa).asInstanceOf[Long] > 2)

    assert(singStats == sCount)

    val qAF = vds.queryVA("va.qc.AF")._2
    val afSC = vds.variantsAndAnnotations.map(_._2)
      .aggregate(new StatCounter())({ case (statC, va) =>
        val af = Option(qAF(va))
        af.foreach(o => statC.merge(o.asInstanceOf[Double]))
        statC
      }, { case (sc1, sc2) => sc1.merge(sc2) })

    assert(afDist == Annotation(afSC.mean, afSC.stdev, afSC.min, afSC.max, afSC.count, afSC.sum))

    val qAC = vds.queryVA("va.qc.AC")._2
    val acSC = vds.variantsAndAnnotations.map(_._2)
      .aggregate(new StatCounter())({ case (statC, va) =>
        val ac = Option(qAC(va))
        ac.foreach(o => statC.merge(o.asInstanceOf[Int]))
        statC
      }, { case (sc1, sc2) => sc1.merge(sc2) })

    assert(acDist == Annotation(acSC.mean, acSC.stdev, acSC.min, acSC.max, acSC.count, acSC.sum))

    val qCR = vds.querySA("sa.qc.callRate")._2
    val crSC = vds.sampleAnnotations
      .aggregate(new StatCounter())({ case (statC, sa) =>
        val cr = Option(qCR(sa))
        cr.foreach(o => statC.merge(o.asInstanceOf[Double]))
        statC
      }, { case (sc1, sc2) => sc1.merge(sc2) })

    assert(crStats == Annotation(crSC.mean, crSC.stdev, crSC.min, crSC.max, crSC.count, crSC.sum))
  }

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
      .annotateGlobal(kt.collect(), TArray(kt.signature), "global.genes")

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
