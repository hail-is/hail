package org.broadinstitute.k3.methods

import java.io.{File, FileWriter}

import org.apache.spark.rdd.RDD
import org.broadinstitute.k3.Utils._
import org.broadinstitute.k3.variant._

import org.broadinstitute.k3.variant.GenotypeCall._


import org.apache.spark.{SparkConf, SparkContext}


case class MendelError(variant: Variant, sample: Int, code: Int, kidGeno: Genotype, dadGeno: Genotype, momGeno: Genotype) {
  def errorString: String = dadGeno.gtString(variant) + " x " + momGeno.gtString(variant) + " -> " + kidGeno.gtString(variant)
}

object MendelErrors {

  // FIXME: Decide between getCode and matchCode
  def getCode(kidGeno: Genotype, dadGeno: Genotype, momGeno: Genotype, onX: Boolean): Int = {
    if (kidGeno.isHomRef)
      if (onX && momGeno.isHomVar)
        9
      else if (!dadGeno.isHomVar && !momGeno.isHomVar)
        0
      else if (dadGeno.isHomVar && !momGeno.isHomVar)
        6
      else if (!dadGeno.isHomVar && momGeno.isHomVar)
        7
      else
        8
    else if (kidGeno.isHet)
      if (dadGeno.isHet || momGeno.isHet)
        0
      else if (dadGeno.isHomRef && momGeno.isHomRef)
        2
      else if (dadGeno.isHomVar && momGeno.isHomVar)
        1
      else
        0
    else if (kidGeno.isHomVar)
      if (onX && momGeno.isHomRef)
        10
      else if (!dadGeno.isHomRef && !momGeno.isHomRef)
        0
      else if (dadGeno.isHomRef && !momGeno.isHomRef)
        3
      else if (!dadGeno.isHomRef && momGeno.isHomRef)
        4
      else
        5
    else
      0
  }

  def matchCode(kidGeno: Genotype, dadGeno: Genotype, momGeno: Genotype, onX: Boolean): Int = {
    (dadGeno.gtCall, momGeno.gtCall, kidGeno.gtCall, onX) match {
      case (HomRef, HomRef, HomRef,     _) => 0 // FIXME: does including these cases at the top speed things up?
      case (   Het,      _,    Het,     _) => 0
      case (     _,    Het,    Het,     _) => 0

      case (HomRef, HomRef,    Het, false) => 2
      case (HomVar, HomVar,    Het, false) => 1

      case (HomRef, HomRef, HomVar, false) => 5
      case (HomRef,      _, HomVar, false) => 3
      case (     _, HomRef, HomVar, false) => 4

      case (HomVar, HomVar, HomRef, false) => 8
      case (HomVar,      _, HomRef, false) => 6
      case (     _, HomVar, HomRef, false) => 7

      case (     _, HomVar, HomRef,  true) => 9
      case (     _, HomRef, HomVar,  true) => 10

      case _                               => 0
    }
  }

  def apply(vds: VariantDataset, ped: Pedigree): MendelErrors = {
    def roleInTrio(id: String, t: Trio): Int = // dad = 1, mom = 2
      if (t.dadID.contains(id)) 1 else 2

    val bcSampleIds = vds.sparkContext.broadcast(vds.sampleIds)
    val bcSampleIndices = vds.sparkContext.broadcast(vds.sampleIds.zipWithIndex.toMap)
    val bcPed = vds.sparkContext.broadcast(ped)

    new MendelErrors(
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
          val code = matchCode(m(0), m(1), m(2), v.onX)
          if (code != 0)
            Some(new MendelError(v, s, code, m(0), m(1), m(2)))
          else
            None
        }
      }
    )
  }
}

case class MendelErrors(ped: Pedigree, sampleIds: Array[String], mendalErrors: RDD[MendelError]) {
  // FIXME: how to handle variants, families, and individuals in which their were no errors?

  def nErrorPerVariant: RDD[(Variant, Int)] = {
    mendalErrors
      .map(mdl => (mdl.variant, 1))
      .reduceByKey(_ + _)
  }

  def nErrorPerFamily: RDD[(String, Int)] = {
    mendalErrors
      .flatMap(mdl => {
      val kidID = sampleIds(mdl.sample)
      val famID = ped.trioMap(kidID).famID

      famID.map((_, 1))
    })
      .reduceByKey(_ + _)
  }

  def nErrorPerIndiv: RDD[(Int, Int)] = {
    val dadOf = mendalErrors.sparkContext.broadcast(ped.dadOf(sampleIds))
    val momOf = mendalErrors.sparkContext.broadcast(ped.momOf(sampleIds))

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

    mendalErrors
      .flatMap(samplesImplicated)
      .map((_, 1))
      .reduceByKey(_ + _)
  }

  def writeMendel(filename: String) {
    def mendelLine(m: MendelError): String = {
      val v = m.variant
      val errorString = m.dadGeno.gtString(v) + " x " + m.momGeno.gtString(v) + " -> " + m.kidGeno.gtString(v)
      val kidID = sampleIds(m.sample)
      val famID = ped.trioMap(kidID).famID

      famID.getOrElse("0") + "\t" + kidID + "\t" + v.contig + "\t" +
        v.shortString + "\t" + m.code + "\t" + m.errorString
    }
    val lines = mendalErrors.map(mendelLine)
    writeTableWithSpark(filename, lines, "FID\tKID\tCHR\tSNP\tCODE\tERROR\n")
  }

  def writeMendelL(filename: String) {
    def variantLine(v: Variant, nError: Int) = {
      v.contig + "\t" + v.shortString + "\t" + nError
    }
    val lines = nErrorPerVariant.map((variantLine _).tupled)
    writeTableWithSpark(filename, lines, "CHR\tSNP\tN\n")
  }

  def writeMendelF(filename: String) {
    def famLine(famID: String, nError: Int): String = {
      famID + "\t" + "Fix" + "\t" + "Fix" + "\t" + "Fix" + "\t" + nError + "\n"
    }
    val lines = nErrorPerFamily.map((famLine _).tupled).collect()
    writeTableWithFileWriter(filename, lines, "FID\tPAT\tMAT\tCHLD\tN\n")
  }

  def writeMendelI(filename: String) {
    def indivLine(s: Int, nError: Int): String = {
      val indivID = sampleIds(s)
      val famID = ped.trioMap(indivID).famID
      famID.getOrElse("0") + "\t" + indivID + "\t" + nError + "\n"
    }
    val lines = nErrorPerIndiv.map((indivLine _).tupled).collect()
    writeTableWithFileWriter(filename, lines, "FID\tIID\tN\n")
  }
}
