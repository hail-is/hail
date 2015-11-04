package org.broadinstitute.hail.methods

import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant._

import org.broadinstitute.hail.variant.GenotypeType._

import org.broadinstitute.hail.methods.Role._

case class MendelError(variant: Variant, sample: Int, code: Int, gKid: Genotype, gDad: Genotype, gMom: Genotype)

object MendelErrors {

  def variantString(v: Variant): String = v.contig + ":" + v.start + ":" + v.ref + ":" + v.alt

  def getCode(gKid: Genotype, gDad: Genotype, gMom: Genotype, isHemizygous: Boolean): Int = {
    (gDad.gtType, gMom.gtType, gKid.gtType, isHemizygous) match {
      case (HomRef, HomRef,    Het, false) => 2  // Kid is het and not hemizygous
      case (HomVar, HomVar,    Het, false) => 1
      case (HomRef, HomRef, HomVar, false) => 5  // Kid is homvar and not hemizygous
      case (HomRef,      _, HomVar, false) => 3
      case (     _, HomRef, HomVar, false) => 4
      case (HomVar, HomVar, HomRef, false) => 8  // Kid is homref and not hemizygous
      case (HomVar,      _, HomRef, false) => 6
      case (     _, HomVar, HomRef, false) => 7
      case (     _, HomVar, HomRef,  true) => 9  // Kid is homref and hemizygous
      case (     _, HomRef, HomVar,  true) => 10 // Kid is homvar and hemizygous
      case _                               => 0  // No error
    }
  }

  def apply(vds: VariantDataset, ped: Pedigree): MendelErrors = {
    require(ped.sexDefinedForAll)
    
    val sexOf = vds.sparkContext.broadcast(ped.sexOf)

    val sampleKidRole: Map[Int, Array[(Int, Role)]] =
      ped.completeTrios.flatMap{
        t => List((t.kid, (t.kid, Kid)), (t.mom.get, (t.kid, Mom)), (t.dad.get, (t.kid, Dad)))
      }
      .groupBy(_._1)
      .mapValues(a => a.map(_._2))
      .map(identity)

    val sampleKidRoleBc = vds.sparkContext.broadcast(sampleKidRole)

    new MendelErrors(ped, vds.sampleIds,
      vds
      .flatMapWithKeys { (v, s, g) => sampleKidRoleBc.value.get(s) match {
        case Some(arr) => arr.map { case (k, r) => ((v, k), (r, g)) }
        case None => None
        }
      }
      .groupByKey()
      .mapValues(_.toMap)
      .flatMap { case ((v, s), gOf) =>
        val code = getCode(gOf(Kid), gOf(Dad), gOf(Mom), v.isHemizygous(sexOf.value(s)))
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
  require(ped.sexDefinedForAll)

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
