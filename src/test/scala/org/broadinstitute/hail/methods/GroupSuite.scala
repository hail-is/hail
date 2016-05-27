package org.broadinstitute.hail.methods

import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr.{TString, TStruct}
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.Utils._

class GroupSuite extends SparkSuite {

  @Test def hardCoded() {
    val genotypeArray = Array(
      Array(0, -1, 0, 2, 1, 0, 1, 1, 0), //A
      Array(1, -1, 1, 2, 1, 0, -1, 0, 0), //B
      Array(0, -1, 2, 2, 1, 0, -1, 0, 0), //A
      Array(-1, -1, 0, 2, 1, 0, -1, 1, 0), //B
      Array(2, -1, 0, 2, 1, 0, -1, 0, 0)) //A

    val variants = for (i <- 1 to genotypeArray.length) yield {Variant("1", i, "A", "T")}
    val sampleIds = for (i <- 1 to genotypeArray(0).length) yield s"Sample_$i"
    val phenotypes = for (i <- 1 to genotypeArray(0).length) yield {
      if (i % 3 == 0)
        "true"
      else if (i % 3 == 1)
        "false"
      else
        "NA"
    }

    val genes = Array("A", "B", "A", "B", "A")

    val sumAnswer = sc.parallelize(Array((IndexedSeq("A"), Array((Some(2.0), Some(4.0)), (None, None), (Some(2.0), Some(4.0)), (Some(6.0), Some(0.0)), (Some(3.0), Some(3.0)), (Some(0.0), Some(6.0)), (Some(1.0), Some(1.0)), (Some(1.0), Some(5.0)), (Some(0.0), Some(6.0)))),
                          (IndexedSeq("B"), Array((Some(1.0), Some(1.0)), (None, None), (Some(1.0), Some(3.0)), (Some(4.0), Some(0.0)), (Some(2.0), Some(2.0)), (Some(0.0), Some(4.0)), (None, None), (Some(1.0), Some(3.0)), (Some(0.0), Some(4.0))))))

    val carrierAnswer = sc.parallelize(Array((IndexedSeq("A"), Array((Some(1.0), Some(0.0)), (None, None), (Some(1.0), Some(0.0)), (Some(1.0), Some(0.0)), (Some(1.0), Some(0.0)), (Some(0.0), Some(1.0)), (Some(1.0), Some(0.0)), (Some(1.0), Some(0.0)), (Some(0.0), Some(1.0)))),
                              (IndexedSeq("B"), Array((Some(1.0), Some(0.0)), (None, None), (Some(1.0), Some(0.0)), (Some(1.0), Some(0.0)), (Some(1.0), Some(0.0)), (Some(0.0), Some(1.0)), (None, None), (Some(1.0), Some(0.0)), (Some(0.0), Some(1.0))))))

    val geneTable = tmpDir.createTempFile("hardCodedCreateGroup",".txt")
    val phenotypeTable = tmpDir.createTempFile("phenotypes", ".txt")

    writeTextFile(geneTable, sc.hadoopConfiguration) { w =>
      w.write(s"Variant\tGene\n")
      variants.zipWithIndex.foreach { case (v, i) =>
      val sb = new StringBuilder()
      sb.append(v.contig)
      sb.append(":")
      sb.append(v.start)
      sb.append(":")
      sb.append(v.ref)
      sb.append(":")
      sb.append(v.alt)
      sb.append("\t")
      sb.append(genes(i))
      sb.append("\n")
      w.write(sb.result())
    }}

    writeTextFile(phenotypeTable, sc.hadoopConfiguration) { w =>
      w.write(s"Sample\tPhenotype1\n")
      phenotypes.zipWithIndex.foreach { case (p, i) =>
        val sb = new StringBuilder()
        sb.append(sampleIds(i))
        sb.append("\t")
        sb.append(p)
        sb.append("\n")
        w.write(sb.result())
      }
    }

    val rdd = sc.parallelize(variants.zipWithIndex.map { case (v, i) =>
      val b = new GenotypeStreamBuilder(v, true)
      val genoBuilder = new GenotypeBuilder(v)
      val genotypes = genotypeArray(i)
      val genotypeStream = for (gt <- genotypes) yield Genotype(gt)

      val annotation = Annotation(genes(i))
      (v, annotation, genotypeStream.toIterable)
    }
    )

    val signatures = TStruct("gene" -> TString)

    val vds: VariantDataset = VariantSampleMatrix[Genotype](VariantMetadata(sampleIds.toArray), rdd).copy(vaSignature = signatures, wasSplit = true)
    var s = State(sc, sqlContext, vds = vds)
    s = CreateGroup.run(s, Array("-k", "va.gene", "-v", "g.nNonRefAlleles", "-a", "sum"))

    val sumResults = s.group.map{case (k, v) => (k.map(_.toString), v)}
    val answerSum = sumAnswer.fullOuterJoin(sumResults).map { case (k, (v1, v2)) =>
      if (v1.isEmpty || v2.isEmpty)
        false
      else {
        v1.get.zip(v2.get).zipWithIndex.forall { case ((d1, d2), i) =>
          if (d1 == d2)
            true
          else {
            println(s"sum k=$k i=$i truth.1=${d1._1.getOrElse("NA")} hail.1=${d2._1.getOrElse("NA")} truth.2=${d1._2.getOrElse("NA")} hail.2=${d2._2.getOrElse("NA")}")
            false
          }
        }
      }
    }.fold(true)(_ && _)

    var c = State(sc, sqlContext, vds = vds)
    c = CreateGroup.run(c, Array("-k", "va.gene", "-v", "g.nNonRefAlleles", "-a", "carrier"))

    val carrierResults = c.group.map{case (k, v) => (k.map(_.toString), v)}
    val answerCarrier = carrierAnswer.fullOuterJoin(carrierResults).map { case (k, (v1, v2)) =>
      if (v1.isEmpty || v2.isEmpty)
        false
      else {
        v1.get.zip(v2.get).zipWithIndex.forall { case ((d1, d2), i) =>
          if (d1 == d2)
            true
          else {
            println(s"carrier k=$k i=$i truth.1=${d1._1.getOrElse("NA")} hail.1=${d2._1.getOrElse("NA")} truth.2=${d1._2.getOrElse("NA")} hail.2=${d2._2.getOrElse("NA")}")
            false
          }
        }
      }
    }.fold(true)(_ && _)

    assert(answerSum && answerCarrier)

    // FIXME: test linear regression results
    val tmpOutputLinReg = tmpDir.createTempFile(prefix = "groupLinReg_test", extension = ".tsv")
    val tmpOutputFisher = tmpDir.createTempFile(prefix = "groupFisher_test", extension = ".tsv")
    c = AnnotateSamplesTable.run(c, Array("-i", phenotypeTable, "-r", "sa.mypheno", "-t", "Phenotype1: Boolean"))
    c = GroupTest.run(c, Array("linreg", "-o", tmpOutputLinReg, "-y", "sa.mypheno.Phenotype1"))
    c = GroupTest.run(c, Array("fisher", "-o", tmpOutputFisher, "-y", "sa.mypheno.Phenotype1"))
  }
}
