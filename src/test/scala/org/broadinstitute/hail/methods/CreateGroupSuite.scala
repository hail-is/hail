package org.broadinstitute.hail.methods

import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr.{TString, TStruct}
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.Utils._

class CreateGroupSuite extends SparkSuite {

  @Test def hardCoded() {
    val genotypeArray = Array(
      Array(0, -1, 0, 2, 1, 0, 1, 1, 0), //A
      Array(1, -1, 1, 2, 1, 0, -1, 0, 0), //B
      Array(0, -1, 2, 2, 1, 0, -1, 0, 0), //A
      Array(-1, -1, 0, 2, 1, 0, -1, 1, 0), //B
      Array(2, -1, 0, 2, 1, 0, -1, 0, 0)) //A

    val variants = for (i <- 1 to genotypeArray.length) yield {Variant("1", i, "A", "T")}
    val sampleIds = for (i <- 1 to genotypeArray(0).length) yield s"Sample_$i"
    val genes = Array("A", "B", "A", "B", "A")

    val sumAnswer = sc.parallelize(Array((IndexedSeq("A"), Array(Some(2.0), None, Some(2.0), Some(6.0), Some(3.0), Some(0.0), Some(1.0), Some(1.0), Some(0.0))),
                          (IndexedSeq("B"), Array(Some(1.0), None, Some(1.0), Some(4.0), Some(2.0), Some(0.0), None, Some(1.0), Some(0.0)))))

    val carrierAnswer = sc.parallelize(Array((IndexedSeq("A"), Array(Some(1.0), None, Some(1.0), Some(1.0), Some(1.0), Some(0.0), Some(1.0), Some(1.0), Some(0.0))),
                              (IndexedSeq("B"), Array(Some(1.0), None, Some(1.0), Some(1.0), Some(1.0), Some(0.0), None, Some(1.0), Some(0.0)))))

    val geneTable = tmpDir.createTempFile("hardCodedCreateGroup",".txt")

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
            println(s"k=$k i=$i d1=${d1.getOrElse("NA")} d2=${d2.getOrElse("NA")}")
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
            println(s"k=$k i=$i d1=${d1.getOrElse("NA")} d2=${d2.getOrElse("NA")}")
            false
          }
        }
      }
    }.fold(true)(_ && _)

    assert(answerSum && answerCarrier)
  }
}
