package org.broadinstitute.k3.methods

import java.io.{File, FileWriter}

import org.apache.spark.rdd.RDD
import org.broadinstitute.k3.Utils._
import org.broadinstitute.k3.variant._

import org.apache.spark.{SparkConf, SparkContext}


case class MendelError(variant: Variant, sample: Int, code: Int, kidGeno: Genotype, dadGeno: Genotype, momGeno: Genotype) {
  def errorString: String = dadGeno.gtString(variant) + " x " + momGeno.gtString(variant) + " -> " + kidGeno.gtString(variant)
}

object MendelDataSet {

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

  def apply(vds: VariantDataset, ped: Pedigree): MendelDataSet = {
    def roleInTrio(id: String, t: Trio): Int = // dad = 1, mom = 2
      if (t.dadID.contains(id)) 1 else 2

    val bcSampleIds = vds.sparkContext.broadcast(vds.sampleIds)
    val bcSampleIndices = vds.sparkContext.broadcast(vds.sampleIds.zipWithIndex.toMap)
    val bcPed = vds.sparkContext.broadcast(ped)

    new MendelDataSet(
      ped,
      vds.sampleIds,
      vds
        .flatMapWithKeys(
          (v, s, g) => {
            val id = bcSampleIds.value(s)
            val trio = bcPed.value.trioMap(id)
            val triosAsKid = if (trio.hasDadMom) List((id, 0)) else Nil
            val triosAsParent = bcPed.value.kidsOfParent(id)
              .map(bcPed.value.trioMap(_))
              .filter(_.hasDadMom)
              .map(t => (t.kidID, roleInTrio(id, t)))

            (triosAsKid ++ triosAsParent).map { case (k, role) => ((v, bcSampleIndices.value(k)), (role, g))
            }
          })
        .groupByKey()
        .mapValues(_.toMap)
        .flatMap { case ((v, s), m) => {
          val code = getCode(m(0), m(1), m(2), v.onX)
          if (code != 0)
            Some(new MendelError(v, s, code, m(0), m(1), m(2)))
          else
            None
        }
      }
    )
  }
}

case class MendelDataSet(ped: Pedigree, sampleIds: Array[String], mds: RDD[MendelError]) {
  // FIXME: how to handle variants, families, and individuals in which their were no errors?

  def nErrorPerVariant: RDD[(Variant, Int)] = {
    mds
      .map(mdl => (mdl.variant, 1))
      .reduceByKey(_ + _)
  }

  def nErrorPerFamily: RDD[(String, Int)] = {
    mds
      .flatMap(mdl => {
      val kidID = sampleIds(mdl.sample)
      val famID = ped.trioMap(kidID).famID

      famID.map((_, 1))
      })
      .reduceByKey(_ + _)
  }

  def nErrorPerIndiv: RDD[(Int, Int)] = {
    val dadOf = mds.sparkContext.broadcast(ped.dadOf(sampleIds))
    val momOf = mds.sparkContext.broadcast(ped.momOf(sampleIds))

    def samplesImplicated(mdl: MendelError): List[Int] = {
      val s = mdl.sample
      val c = mdl.code

      if (c == 2 || c == 1)
        List(s, dadOf.value(s), momOf.value(s))
      else if (c == 6 || c == 3)
        List(s, dadOf.value(s))
      else if (c == 4 || c == 7 || c == 9 || c == 10)
        List(s, momOf.value(s))
      else
        List(s)
    }

    mds
      .flatMap(samplesImplicated)
      .map((_, 1))
      .reduceByKey(_ + _)
  }

  def write(filename: String) {
    cleanly[FileWriter, Unit](new FileWriter(new File(filename)),
    fw => fw.close(), {
      fw => {
        fw.write("FID\tKID\tCHR\tSNP\tCODE\tERROR\n")
        mds
          .map(MendelLine(ped, sampleIds, _))
          .collect() // FIXME: write in parallel
          .foreach(_.write(fw))
      }
    })
  }

  def writeVariant(filename: String) {
    cleanly[FileWriter, Unit](new FileWriter(new File(filename)),
    fw => fw.close(), {
      fw => {
        fw.write("CHR\tSNP\tN\n")
        nErrorPerVariant
          .map{ case (v,n) => MendelVariantLine(v, n) } // FIXME: feed more directly using apply(tuple)?
          .collect() // FIXME: write in parallel
          .foreach(_.write(fw))
      }
    })
  }

  def writeFamily(filename: String) {
    cleanly[FileWriter, Unit](new FileWriter(new File(filename)),
    fw => fw.close(), {
      fw => {
        fw.write("FID\tPAT\tMAT\tCHLD\tN\n")
        nErrorPerFamily
          .map{ case (famID, nError) => MendelFamilyLine(famID, "?", "?", -1, nError) } // FIXME: feed more directly using apply(tuple)?
          .collect() // FIXME: write in parallel
          .foreach(_.write(fw))
      }
    })
  }

  def writeIndiv(filename: String) {
    cleanly[FileWriter, Unit](new FileWriter(new File(filename)),
    fw => fw.close(), {
      fw => {
        fw.write("FID\tIID\tN\n")
        nErrorPerIndiv
          .map { case (s, nError) => MendelIndivLine(ped, sampleIds(s), nError) } // FIXME: feed more directly using apply(tuple)?
          .collect() // FIXME: write in parallel
          .foreach(_.write(fw))
      }
    })
  }
}

object MendelLine {
  def apply(ped: Pedigree, sampleIds: Array[String], mdl: MendelError) = {
    def toShortString(v: Variant): String = v.contig + ":" + v.start + ":" + v.ref + ":" + v.alt

    def errorString(v: Variant, m: Map[Int, Genotype]): String =
      m(1).gtString(v) + " x " + m(2).gtString(v) + " -> " + m(0).gtString(v)

    val kidId = sampleIds(mdl.sample)
    val famId = ped.trioMap(kidId).famID

    new MendelLine(famId, kidId, mdl.variant.contig, toShortString(mdl.variant), mdl.code, mdl.errorString)
  }
}

case class MendelLine(famID: Option[String], kidID: String, contig: String, varID: String, code: Int, error: String) {
  def write(fw: FileWriter) = {
    fw.write(famID.getOrElse("0") + "\t" + kidID + "\t" + contig + "\t" + varID + "\t" + code + "\t" + error + "\n")
  }
}

object MendelFamilyLine {
  def apply(ped: Pedigree, sampleIds: Array[String], famID: String, nError: Integer): MendelFamilyLine =
    new MendelFamilyLine(famID, "FIXME", "FIXME", -1, nError)
}

case class MendelFamilyLine(famID: String, dadID: String, momID: String, nKid: Int, nError: Int) {
  def write(fw: FileWriter) = {
    fw.write(famID + "\t" + dadID + "\t" + momID + "\t" + nKid + "\t" + nError + "\n")
  }
}

object MendelIndivLine {
  def apply(ped: Pedigree, kidID: String, nError: Integer): MendelIndivLine = {
    val famID = ped.trioMap(kidID).famID
    new MendelIndivLine(famID, kidID, nError)
  }
}

case class MendelIndivLine(famID: Option[String], indivID: String, nError: Int) {
  def write(fw: FileWriter) = {
    fw.write(famID.getOrElse("0") + "\t" + indivID + "\t" + nError + "\n")
  }
}

object MendelVariantLine {
  def toShortString(v: Variant): String = v.contig + ":" + v.start + ":" + v.ref + ":" + v.alt

  def apply(v: Variant, nError: Integer) = new MendelVariantLine(v.contig, toShortString(v), nError)
}

case class MendelVariantLine(chrom: String, varID: String, nError: Int) {
  def write(fw: FileWriter) = {
    fw.write(chrom + "\t" + varID + "\t" + nError + "\n")
  }
}
