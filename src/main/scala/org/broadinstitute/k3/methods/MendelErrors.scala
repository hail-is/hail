package org.broadinstitute.k3.methods

import org.apache.spark.rdd.RDD
import org.broadinstitute.k3.Utils._
import org.broadinstitute.k3.variant._

import org.broadinstitute.k3.variant.GenotypeType._

object Role extends Enumeration {
  type Role = Value
  val Kid = Value("0")
  val Dad = Value("1")
  val Mom = Value("2")
}

import org.broadinstitute.k3.methods.Role._

case class MendelError(variant: Variant, sample: Int, code: Int, gKid: Genotype, gDad: Genotype, gMom: Genotype) {
  def errorString: String = gDad.gtString(variant) + " x " + gMom.gtString(variant) + " -> " + gKid.gtString(variant)
}

object MendelErrors {

  // FIXME: Decide between getCode and matchCode
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

  def apply(vds: VariantDataset, ped: Pedigree): MendelErrors = {
    val bcPed = vds.sparkContext.broadcast(ped)

    new MendelErrors(ped, vds.sampleIds, vds
      .flatMapWithKeys{ (v, s, g) =>
        bcPed.value
          .completeTriosContaining(s)
          .map(t => ((v, t.kid), (t.role(s).get, g)))
       }
      .groupByKey()
      .mapValues(_.toMap)
      .flatMap { case ((v, s), gOf) => {
        val code = matchCode(gOf(Kid), gOf(Dad), gOf(Mom), v.onX)
        if (code != 0)
          Some(new MendelError(v, s, code, gOf(Kid), gOf(Dad), gOf(Mom)))
        else
          None
        }
      }
    )
  }
}

case class MendelErrors(ped: Pedigree, sampleIds: Array[String], mendelErrors: RDD[MendelError]) {
  // FIXME: how to handle variants, families, and individuals in which their were no errors?

  def nErrorPerVariant: RDD[(Variant, Int)] = {
    mendelErrors
      .map(_.variant)
      .countByValueRDD()
  }

  def nErrorPerFamily: RDD[(String, Int)] = {
    val bcPed = mendelErrors.sparkContext.broadcast(ped)

    mendelErrors
      .flatMap(me => ped.famOf.get(me.sample))
      .countByValueRDD()
  }

  def nErrorPerIndiv: RDD[(Int, Int)] = { // FIXME: how to broadcast the def?
    val bcPed = mendelErrors.sparkContext.broadcast(ped)

    def implicatedSamples(me: MendelError): List[Int] = {
      val s = me.sample
      val c = me.code
      if      (c == 2 || c == 1)                       List(s, bcPed.value.dadOf(s), bcPed.value.momOf(s))
      else if (c == 6 || c == 3)                       List(s, bcPed.value.dadOf(s))
      else if (c == 4 || c == 7 || c == 9 || c == 10)  List(s, bcPed.value.momOf(s))
      else                                             List(s)
    }

    mendelErrors
      .flatMap(implicatedSamples)
      .countByValueRDD()
  }

  def writeMendel(filename: String) {
    def toLine(me: MendelError): String = {
      val v = me.variant
      val s = me.sample
      val errorString = me.gDad.gtString(v) + " x " + me.gMom.gtString(v) + " -> " + me.gKid.gtString(v)

      ped.trioMap(s).fam.getOrElse("0") + "\t" + sampleIds(s) + "\t" + v.contig + "\t" +
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

  def writeMendelF(filename: String) {
    def toLine(fam: String, nError: Int): String = fam + "\t" + "Fix" + "\t" + "Fix" + "\t" + "Fix" + "\t" + nError + "\n"
    val lines = nErrorPerFamily.map((toLine _).tupled).collect()
    writeTable(filename, lines, "FID\tPAT\tMAT\tCHLD\tN\n")
  }

  def writeMendelI(filename: String) {
    def toLine(s: Int, nError: Int): String = ped.trioMap(s).fam.getOrElse("0") + "\t" + sampleIds(s) + "\t" + nError + "\n"
    val lines = nErrorPerIndiv.map((toLine _).tupled).collect()
    writeTable(filename, lines, "FID\tIID\tN\n")
  }
}
