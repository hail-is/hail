package org.broadinstitute.k3.methods

import java.io.{File, FileWriter}

import org.apache.spark.rdd.RDD
import org.broadinstitute.k3.Utils._
import org.broadinstitute.k3.variant._

import org.apache.spark.{SparkConf, SparkContext}

object TryOutMendel {

  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local").setAppName("test")
    conf.set("spark.sql.parquet.compression.codec", "uncompressed")
    // FIXME KryoSerializer causes jacoco to throw IllegalClassFormatException exception
    // conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    val sc = new SparkContext(conf)

    val ped = Pedigree.read("src/test/resources/sample_mendel.fam")
    val vds = LoadVCF(sc, "sparky", "src/test/resources/sample_mendel.vcf")

    println(ped)
    println(ped.trioMap.keys)
    ped.kidsOfParent.foreach(println)
    vds.sampleIds.foreach(println)

    val mendel = Mendel(vds, ped)

    //mdl.mendel.collect().foreach{ case (s, v, c, m) => if (c != 0) println(c) }

    mendel.write("src/test/resources/sample_mendel.mendel")

    sc.stop()

  }
}

object Mendel {

  // ((trio, variant), (role -> genotype, code))
  def apply(vds: VariantDataset, ped: Pedigree): Mendel = {

    def getCode(kidGT: Genotype, dadGT: Genotype, momGT: Genotype, onX: Boolean): Int = {
      if (kidGT.isHomRef)
        if (onX && momGT.isHomVar)
          9
        else if (!dadGT.isHomVar && !momGT.isHomVar)
          0
        else if (dadGT.isHomVar && !momGT.isHomVar)
          6
        else if (!dadGT.isHomVar && momGT.isHomVar)
          7
        else
          8
      else if (kidGT.isHet)
        if (dadGT.isHet || momGT.isHet)
          0
        else if (dadGT.isHomRef && momGT.isHomRef)
          2
        else if (dadGT.isHomVar && momGT.isHomVar)
          1
        else
          0
      else if (kidGT.isHomVar)
        if (onX && momGT.isHomRef)
          10
        else if (!dadGT.isHomRef && !momGT.isHomRef)
          0
        else if (dadGT.isHomRef && !momGT.isHomRef)
          3
        else if (!dadGT.isHomRef && momGT.isHomRef)
          4
        else
          5
      else
        0
    }

    def roleInTrio(id: String, t: Trio): Int = // dad = 1, mom = 2
      if (t.dadID.contains(id)) 1 else 2

    val sampleIds = vds.sparkContext.broadcast(vds.sampleIds)
    val localPed = vds.sparkContext.broadcast(ped)
    val indexOfSample = vds.sparkContext.broadcast(vds.sampleIds.zipWithIndex.toMap)

    new Mendel(ped, vds.sampleIds,
      vds
      .flatMapWithKeys(
        (v, s, g) => {
          val id = sampleIds.value(s)
          val trio = localPed.value.trioMap(id)
          val triosAsKid = if (trio.hasDadMom) List((id, 0)) else List()
          val triosAsParent = localPed.value.kidsOfParent(id)
            .map(localPed.value.trioMap(_))
            .filter(_.hasDadMom)
            .map(t => (t.kidID, roleInTrio(id, t)))

          (triosAsKid ++ triosAsParent).map { case (k, role) => ((v, indexOfSample.value(k)), (role, g))
          }
        })
      .groupByKey()
      .mapValues(_.toMap)
      .flatMap { case ((v, s), m) => {
         val code = getCode(m(0), m(1), m(2), v.onX)
         if (code == 0)
           List()
         else
           List((s, v, code, m))
      }
    })
  }


  def codeImplicatesDad = Map(
    1 -> true, 2 -> true,
    3 -> true, 4 -> false, 5 -> false,
    6 -> true, 7 -> false, 8 -> false,
    9 -> false, 10 -> false)

  def codeImplicatesMom = Map(
    1 -> true, 2 -> true,
    3 -> false, 4 -> true, 5 -> false,
    6 -> true, 7 -> true, 8 -> false,
    9 -> true, 10 -> true)
}

case class Mendel(ped: Pedigree, sampleIds: Array[String], mendel: RDD[(Int, Variant, Int, Map[Int, Genotype])]) {

  def errorString(v: Variant, m: Map[Int, Genotype]): String =
    m(1).gtString(v) + " x " + m(2).gtString(v) + " -> " + m(0).gtString(v)

  def toMendalM(men: (Int, Variant, Int, Map[Int, Genotype])): MendelM = {
    def toShortString(v: Variant): String = v.contig + ":" + v.start + ":" + v.ref + ":" + v.alt

    men match {
      case (s, v, c, m) => {
        val kidId = sampleIds(s)
        val famId = ped.trioMap(kidId).famID
        new MendelM(famId, kidId, v.contig, toShortString(v), c, errorString(v, m))
      }
    }
  }

  def write(filename: String) {
    val header = "FID\tKID\tCHR\tSNP\tCODE\tERROR\n"

    cleanly[FileWriter, Unit](new FileWriter(new File(filename)),
    fw => fw.close(), {
      fw => {
        fw.write(header)
        mendel.collect().map(toMendalM).foreach(_.write(fw))
      }
    })
  }
}

case class MendelM(famID: Option[String], kidID: String, contig: String, varID: String, code: Int, error: String) {
  def write(fw: FileWriter) = {
    fw.write(famID.getOrElse("0") + "\t" + kidID + "\t" + contig + "\t" + varID + "\t" + code + "\t" + error + "\n")
  }
}

case class MendelF(famID: Option[String], dadID: String, momID: String, nKid: Int, nError: Int) {
  def write(fw: FileWriter) = {
    fw.write(famID.getOrElse("0") + "\t" + dadID + "\t" + momID + "\t" + nKid + "\t" + nError + "\n")
  }
}

case class MendelI(famID: Option[String], indivID: String, nError: Int) {
  def write(fw: FileWriter) = {
    fw.write(famID.getOrElse("0") + "\t" + indivID + "\t" + nError + "\n")
  }
}

case class MendelL(chrom: String, varID: String, nError: Int) {
  def write(fw: FileWriter) = {
    fw.write(chrom + "\t" + varID + "\t" + nError + "\n")
  }
}
