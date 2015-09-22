package org.broadinstitute.k3.methods

import org.apache.spark.rdd.RDD
import org.broadinstitute.k3.Utils._
import org.broadinstitute.k3.variant._

import org.broadinstitute.k3.variant.GenotypeType._

import org.broadinstitute.k3.methods.Role._

case class MendelError(variant: Variant, sample: Int, code: Int, gKid: Genotype, gDad: Genotype, gMom: Genotype)

object MendelErrors {

  // FIXME: Decide between getCode, matchCode, and nestedMatchCode
  def getCode(gKid: Genotype, gDad: Genotype, gMom: Genotype, onX: Boolean): Int = {
    if (gKid.isHomRef)
      if      (onX            &&  gMom.isHomVar) 9
      else if (!gDad.isHomVar && !gMom.isHomVar) 0
      else if ( gDad.isHomVar && !gMom.isHomVar) 6
      else if (!gDad.isHomVar &&  gMom.isHomVar) 7
      else                                       8
    else if (gKid.isHet)
      if      (gDad.isHet    || gMom.isHet)      0
      else if (gDad.isHomRef && gMom.isHomRef)   2
      else if (gDad.isHomVar && gMom.isHomVar)   1
      else                                       0
    else if (gKid.isHomVar)
      if      (onX            &&  gMom.isHomRef) 10
      else if (!gDad.isHomRef && !gMom.isHomRef) 0
      else if ( gDad.isHomRef && !gMom.isHomRef) 3
      else if (!gDad.isHomRef &&  gMom.isHomRef) 4
      else                                       5
    else                                         0
  }

  def matchCode(gKid: Genotype, gDad: Genotype, gMom: Genotype, onX: Boolean): Int = {
    (gDad.gtType, gMom.gtType, gKid.gtType, onX) match {
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

  def nestedMatchCode(gKid: Genotype, gDad: Genotype, gMom: Genotype, onX: Boolean): Int = {
    (gDad.gtType, gMom.gtType, gKid.gtType, onX) match {
      case (HomRef, HomRef, HomRef, _) => 0

      case   (dad, mom, Het, x) => (dad, mom, x) match {
        case (     _,    Het,     _)   => 0
        case (Het   ,      _,     _)   => 0
        case (HomRef, HomRef, false)   => 2
        case (HomVar, HomVar, false)   => 1
        case _                         => 0
      }
      case (dad, mom, HomVar, x) => (dad, mom, x) match {
        case (HomRef, HomRef, false)   => 5
        case (HomRef,      _, false)   => 3
        case (     _, HomRef, false)   => 4
        case (     _, HomRef,  true)   => 10
        case _                         => 0
      }
      case (dad, mom, HomRef, x) => (dad, mom, x) match {
        case (HomVar, HomVar, false)   => 8
        case (HomVar,      _, false)   => 6
        case (     _, HomVar, false)   => 7
        case (     _, HomVar,  true)   => 9
        case _                         => 0
      }
      case _                           => 0
    }
  }

  def apply(vds: VariantDataset, ped: Pedigree): MendelErrors = {
    val completeTrios = vds.sparkContext.broadcast(ped.completeTrios)

    new MendelErrors(ped, vds.sampleIds, vds.variants,
      vds
      .flatMapWithKeys{ (v, s, g) =>
        completeTrios.value
          .filter(_.contains(s))
          .map(t => ((v, t.kid), (t.role(s).get, g)))
       }
      .groupByKey()
      .mapValues(_.toMap)
      .flatMap { case ((v, s), gOf) => { //FIXME: IntelliJ wants me to remove this unnecessary pair of braces.  Thoughts?
        val code = getCode(gOf(Kid), gOf(Dad), gOf(Mom), v.onX)
        if (code != 0)
          Some(new MendelError(v, s, code, gOf(Kid), gOf(Dad), gOf(Mom)))
        else
          None
        }
      }
    )
  }
}

case class MendelErrors(ped:          Pedigree,
                        sampleIds:    Array[String],
                        variants:     RDD[Variant],
                        mendelErrors: RDD[MendelError]) {

  def nErrorPerVariant: RDD[(Variant, Int)] = {
    mendelErrors
      .map(_.variant)
      .countByValueRDD()
      .union(variants.map((_, 0)))
      .reduceByKey(_ + _)
  }

  def nErrorPerNuclearFamily: RDD[((Int, Int), Int)] = {
    val dadOf = mendelErrors.sparkContext.broadcast(ped.dadOf)
    val momOf = mendelErrors.sparkContext.broadcast(ped.momOf)
    val parentsRDD = mendelErrors.sparkContext.parallelize(ped.nuclearFams.keys.toSeq)
    mendelErrors
      .map(me => (dadOf.value(me.sample), momOf.value(me.sample)))
      .countByValueRDD()
      .union(parentsRDD.map((_, 0)))
      .reduceByKey(_ + _)
  }

  def nErrorPerIndiv: RDD[(Int, Int)] = {
    val dadOf = mendelErrors.sparkContext.broadcast(ped.dadOf)
    val momOf = mendelErrors.sparkContext.broadcast(ped.momOf)
    val indivRDD = mendelErrors.sparkContext.parallelize(ped.trioMap.keys.toSeq)
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
      .countByValueRDD()
      .union(indivRDD.map((_, 0)))
      .reduceByKey(_ + _)
  }

  def writeMendel(filename: String) {
    val bcSampleIds = mendelErrors.sparkContext.broadcast(sampleIds)
    val famOf = mendelErrors.sparkContext.broadcast(ped.famOf)
    def toLine(me: MendelError): String = {
      val v = me.variant
      val s = me.sample
      val errorString = me.gDad.gtString(v) + " x " + me.gMom.gtString(v) + " -> " + me.gKid.gtString(v)
      famOf.value.getOrElse(s, "0") + "\t" + bcSampleIds.value(s) + "\t" + v.contig + "\t" +
        v.shortString + "\t" + me.code + "\t" + errorString
    }
    val lines = mendelErrors.map(toLine)
    writeTableWithSpark(filename, lines, "FID\tKID\tCHR\tSNP\tCODE\tERROR\n")
  }

  def writeMendelL(filename: String) {
    def toLine(v: Variant, nError: Int) = v.contig + "\t" + v.shortString + "\t" + nError
    val lines = nErrorPerVariant.map((toLine _).tupled)
    writeTableWithSpark(filename, lines, "CHR\tSNP\tN\n")
  }

  //FIXME: this is
  def writeMendelF(filename: String) {
    val bcSampleIds = nErrorPerNuclearFamily.sparkContext.broadcast(sampleIds)
    val famOf = nErrorPerNuclearFamily.sparkContext.broadcast(ped.famOf)
    val nuclearFams = nErrorPerNuclearFamily.sparkContext.broadcast(ped.nuclearFams)

    //FIXME: plink only prints nCHLD, but the list of kids may be useful, currently not used anywhere else
    def toLine(parents: (Int, Int), nError: Int): String = {
      val (dad, mom) = parents
      famOf.value.getOrElse(dad, "0") + "\t" + bcSampleIds.value(dad) + "\t" + bcSampleIds.value(mom) + "\t" +
        nuclearFams.value((dad, mom)).size + "\t" + nError + "\n"
    }
    val lines = nErrorPerNuclearFamily.map((toLine _).tupled).collect()
    writeTable(filename, lines, "FID\tPAT\tMAT\tCHLD\tN\n")
  }

  def writeMendelI(filename: String) {
    val bcSampleIds = nErrorPerIndiv.sparkContext.broadcast(sampleIds)
    val famOf = nErrorPerIndiv.sparkContext.broadcast(ped.famOf)

    def toLine(s: Int, nError: Int): String =
      famOf.value.getOrElse(s, "0") + "\t" + bcSampleIds.value(s) + "\t" + nError + "\n"
    val lines = nErrorPerIndiv.map((toLine _).tupled).collect()
    writeTable(filename, lines, "FID\tIID\tN\n")
  }
}
