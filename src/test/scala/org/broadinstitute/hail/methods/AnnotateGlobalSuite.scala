package org.broadinstitute.hail.methods

import org.apache.spark.util.StatCounter
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.expr._
import org.testng.annotations.Test


class AnnotateGlobalSuite extends SparkSuite {
  @Test def test(): Unit = {

    var s = SplitMulti.run(State(sc, sqlContext, LoadVCF(sc, "src/test/resources/sample2.vcf")), Array.empty[String])
    s = VariantQC.run(s, Array.empty[String])
    s = SampleQC.run(s, Array.empty[String])

    s = AnnotateGlobal.run(s, Array("expr", "-c", "global.afDist = variants.stats(va.qc.AF), global.singStats = samples.count(sa.qc.nSingleton > 2)"))
    s = AnnotateGlobal.run(s, Array("expr", "-c", "global.anotherAnnotation.sumOver2 = global.afDist.sum / 2"))
    s = AnnotateGlobal.run(s, Array("expr", "-c", "global.acDist = variants.stats(va.qc.AC)"))
    s = AnnotateGlobal.run(s, Array("expr", "-c", "global.CRStats = samples.stats(sa.qc.callRate)"))


    val vds = s.vds
    val qSingleton = vds.querySA("sa.qc.nSingleton")._2
    val qSingletonGlobal = vds.queryGlobal("global.singStats")._2

    val sCount = vds.sampleAnnotations.count(sa => {
      qSingleton(sa).exists(_.asInstanceOf[Int] > 2)
    })

    assert(qSingletonGlobal.contains(sCount))

    val qAF = vds.queryVA("va.qc.AF")._2
    val afSC = vds.variantsAndAnnotations.map(_._2)
      .aggregate(new StatCounter())({ case (statC, va) =>
        val af = qAF(va)
        af.foreach(o => statC.merge(o.asInstanceOf[Double]))
        statC
      }, { case (sc1, sc2) => sc1.merge(sc2) })

    assert(vds.queryGlobal("global.afDist")._2
      .contains(Annotation(afSC.mean, afSC.stdev, afSC.min, afSC.max, afSC.count, afSC.sum)))

    assert(vds.queryGlobal("global.anotherAnnotation.sumOver2")._2.contains(afSC.sum / 2))

    val qAC = vds.queryVA("va.qc.AC")._2
    val acSC = vds.variantsAndAnnotations.map(_._2)
      .aggregate(new StatCounter())({ case (statC, va) =>
        val ac = qAC(va)
        ac.foreach(o => statC.merge(o.asInstanceOf[Int]))
        statC
      }, { case (sc1, sc2) => sc1.merge(sc2) })

    assert(vds.queryGlobal("global.acDist")._2
      .contains(Annotation(acSC.mean, acSC.stdev, acSC.min.toInt,
        acSC.max.toInt, acSC.count, acSC.sum.round.toInt)))

    val qCR = vds.querySA("sa.qc.callRate")._2
    val crSC = vds.sampleAnnotations
      .aggregate(new StatCounter())({ case (statC, sa) =>
        val cr = qCR(sa)
        cr.foreach(o => statC.merge(o.asInstanceOf[Double]))
        statC
      }, { case (sc1, sc2) => sc1.merge(sc2) })

    assert(vds.queryGlobal("global.CRStats")._2
      .contains(Annotation(crSC.mean, crSC.stdev, crSC.min,
        crSC.max, crSC.count, crSC.sum)))

  }

  @Test def testLists() {
    val out1 = tmpDir.createTempFile("file1", ".txt")
    val out2 = tmpDir.createTempFile("file2", ".txt")

    val toWrite1 = Array("Gene1", "Gene2", "Gene3", "Gene4", "Gene5")
    val toWrite2 = Array("1", "5", "4", "2", "2")

    writeTextFile(out1, sc.hadoopConfiguration) { out =>
      toWrite1.foreach { line =>
        out.write(line + "\n")
      }
    }

    writeTextFile(out2, sc.hadoopConfiguration) { out =>
      toWrite2.foreach { line =>
        out.write(line + "\n")
      }
    }

    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("src/test/resources/sample.vcf"))
    s = AnnotateGlobal.run(s, Array("list", "-i", out1, "-r", "global.geneList", "--as-set"))
    s = AnnotateGlobal.run(s, Array("list", "-i", out2, "-r", "global.array"))

    val (_, anno1) = s.vds.queryGlobal("global.geneList")
    val (_, anno2) = s.vds.queryGlobal("global.array")
    assert(anno1.contains(toWrite1.toSet))
    assert(anno2.contains(toWrite2: IndexedSeq[Any]))
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

    writeTextFile(out1, sc.hadoopConfiguration) { out =>
      toWrite1.foreach(line => out.write(line + "\n"))
    }

    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("src/test/resources/sample.vcf"))
    s = AnnotateGlobal.run(s, Array("table", "-i", out1, "-r", "global.genes", "-t", "PLI: Double, EXAC_LOF_COUNT: Int"))

    val (t, res) = s.vds.queryGlobal("global.genes")

    assert(t == TArray(TStruct(
      ("GENE", TString),
      ("PLI", TDouble),
      ("EXAC_LOF_COUNT", TInt))))

    assert(res.contains(IndexedSeq(
      Annotation("Gene1", "0.12312".toDouble, 2),
      Annotation("Gene2", "0.99123".toDouble, 0),
      Annotation("Gene3", null, null),
      Annotation("Gene4", "0.9123".toDouble, 10),
      Annotation("Gene5", "0.0001".toDouble, 202)
    )))

  }
}
