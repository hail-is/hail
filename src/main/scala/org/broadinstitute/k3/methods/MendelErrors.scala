package org.broadinstitute.k3.methods

import org.apache.spark.rdd.RDD
import org.broadinstitute.k3.Utils._
import org.broadinstitute.k3.variant._

import org.broadinstitute.k3.variant.GenotypeType._

import org.broadinstitute.k3.methods.Role.{Kid, Dad, Mom}

case class MendelError(variant: Variant, sample: Int, code: Int, gKid: Genotype, gDad: Genotype, gMom: Genotype)

object MendelErrors {

  def variantString(v: Variant): String = v.contig + ":" + v.start + ":" + v.ref + ":" + v.alt

  def getCode(gKid: Genotype, gDad: Genotype, gMom: Genotype, onX: Boolean): Int = {
    (gDad.gtType, gMom.gtType, gKid.gtType, onX) match {
      case (HomRef, HomRef,    Het, false) => 2  // Autosome, Het
      case (HomVar, HomVar,    Het, false) => 1
      case (HomRef, HomRef, HomVar, false) => 5  // Autosome, HomVar
      case (HomRef,      _, HomVar, false) => 3
      case (     _, HomRef, HomVar, false) => 4
      case (HomVar, HomVar, HomRef, false) => 8  // Autosome, HomRef
      case (HomVar,      _, HomRef, false) => 6
      case (     _, HomVar, HomRef, false) => 7
      case (     _, HomVar, HomRef,  true) => 9  // X, HomRef
      case (     _, HomRef, HomVar,  true) => 10 // X, HomVar
      case _                               => 0  // No error
    }
  }

  def apply(vds: VariantDataset, ped: Pedigree): MendelErrors = {
    val completeTrios = vds.sparkContext.broadcast(ped.completeTrios)

    new MendelErrors(ped, vds.sampleIds,
      vds
      .flatMapWithKeys{ (v, s, g) =>
        completeTrios.value
          .filter(_.contains(s))
          .map(t => ((v, t.kid), (t.role(s).get, g)))
       }
      .groupByKey()
      .mapValues(_.toMap)
      .flatMap { case ((v, s), gOf) =>
        val code = getCode(gOf(Kid), gOf(Dad), gOf(Mom), v.onX)
        if (code != 0)
          Some(new MendelError(v, s, code, gOf(Kid), gOf(Dad), gOf(Mom)))
        else
          None
      }
      .cache()
    )
  }
}

case class MendelErrors(ped:          Pedigree,
                        sampleIds:    Array[String],
                        mendelErrors: RDD[MendelError]) {

  def sc = mendelErrors.sparkContext

  val dadOf = sc.broadcast(ped.dadOf)
  val momOf = sc.broadcast(ped.momOf)
  val famOf = sc.broadcast(ped.famOf)
  val sampleIdsBc = sc.broadcast(sampleIds)

  def nErrorPerVariant: RDD[(Variant, Int)] = {
    mendelErrors
      .map(_.variant)
      .countByValueRDD()
  }

  def nErrorPerNuclearFamily: RDD[((Int, Int), Int)] = {
    val parentsRDD = sc.parallelize(ped.nuclearFams.keys.toSeq)
    mendelErrors
      .map(me => ((dadOf.value(me.sample), momOf.value(me.sample)), 1))
      .union(parentsRDD.map((_, 0)))
      .reduceByKey(_ + _)
  }

  def nErrorPerIndiv: RDD[(Int, Int)] = {
    val indivRDD = sc.parallelize(ped.trioMap.keys.toSeq)
    def implicatedSamples(me: MendelError): List[Int] = {
      val s = me.sample
      val c = me.code
      if      (c == 2 || c == 1)                       List(s, dadOf.value(s), momOf.value(s))
      else if (c == 6 || c == 3)                       List(s, dadOf.value(s))
      else if (c == 4 || c == 7 || c == 9 || c == 10)  List(s, momOf.value(s))
      else                                             List(s)
    }
    mendelErrors
      .flatMap(implicatedSamples)
      .map((_, 1))
      .union(indivRDD.map((_, 0)))
      .reduceByKey(_ + _)
  }

  def writeMendel(filename: String) {
    def toLine(me: MendelError): String = {
      val v = me.variant
      val s = me.sample
      val errorString = me.gDad.gtString(v) + " x " + me.gMom.gtString(v) + " -> " + me.gKid.gtString(v)
      famOf.value.getOrElse(s, "0") + "\t" + sampleIdsBc.value(s) + "\t" + v.contig + "\t" +
        MendelErrors.variantString(v) + "\t" + me.code + "\t" + errorString
    }
    mendelErrors.map(toLine)
      .writeTable(filename, "FID\tKID\tCHR\tSNP\tCODE\tERROR\n")
  }

  def writeMendelL(filename: String) {
    def toLine(v: Variant, nError: Int) = v.contig + "\t" + MendelErrors.variantString(v) + "\t" + nError
    nErrorPerVariant.map((toLine _).tupled)
      .writeTable(filename, "CHR\tSNP\tN\n")
  }

  def writeMendelF(filename: String) {
    val nuclearFams = sc.broadcast(ped.nuclearFams.force)
    def toLine(parents: (Int, Int), nError: Int): String = {
      val (dad, mom) = parents
      famOf.value.getOrElse(dad, "0") + "\t" + sampleIdsBc.value(dad) + "\t" + sampleIdsBc.value(mom) + "\t" +
        nuclearFams.value((dad, mom)).size + "\t" + nError + "\n"
    }
    val lines = nErrorPerNuclearFamily.map((toLine _).tupled).collect()
    writeTable(filename, sc.hadoopConfiguration, lines, "FID\tPAT\tMAT\tCHLD\tN\n")
  }

  def writeMendelI(filename: String) {
    def toLine(s: Int, nError: Int): String =
      famOf.value.getOrElse(s, "0") + "\t" + sampleIdsBc.value(s) + "\t" + nError + "\n"
    val lines = nErrorPerIndiv.map((toLine _).tupled).collect()
    writeTable(filename, sc.hadoopConfiguration, lines, "FID\tIID\tN\n")
  }
}
