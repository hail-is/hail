package org.broadinstitute.hail.methods

import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.variant.GenotypeType._

case class MendelError(variant: Variant, trio: CompleteTrio, code: Int,
                       gtKid: GenotypeType, gtDad: GenotypeType, gtMom: GenotypeType)

object MendelErrors {

  def variantString(v: Variant): String = v.contig + ":" + v.start + ":" + v.ref + ":" + v.alt

  def getCode(gts: Array[GenotypeType], isHemizygous: Boolean): Int = {
    (gts(1), gts(2), gts(0), isHemizygous) match { // gtDad, gtMom, gtKid, isHemizygous
      case (HomRef, HomRef,    Het, false) => 2    // Kid is het and not hemizygous
      case (HomVar, HomVar,    Het, false) => 1
      case (HomRef, HomRef, HomVar, false) => 5    // Kid is homvar and not hemizygous
      case (HomRef,      _, HomVar, false) => 3
      case (     _, HomRef, HomVar, false) => 4
      case (HomVar, HomVar, HomRef, false) => 8    // Kid is homref and not hemizygous
      case (HomVar,      _, HomRef, false) => 6
      case (     _, HomVar, HomRef, false) => 7
      case (     _, HomVar, HomRef,  true) => 9    // Kid is homref and hemizygous
      case (     _, HomRef, HomVar,  true) => 10   // Kid is homvar and hemizygous
      case _                               => 0    // No error
    }
  }

  def apply(vds: VariantDataset, trios: Array[CompleteTrio]): MendelErrors = {
    require(trios.forall(_.sex.isDefined))

    val sampleTrioRoles: Array[List[(Int, Int)]] = Array.fill[List[(Int, Int)]](vds.nSamples)(List())
    trios.zipWithIndex.foreach { case (t, ti) =>
      sampleTrioRoles(t.kid) ::= (ti, 0)
      sampleTrioRoles(t.dad) ::= (ti, 1)
      sampleTrioRoles(t.mom) ::= (ti, 2)
    }

    val sc = vds.sparkContext
    val sampleTrioRolesBc = sc.broadcast(sampleTrioRoles)
    val triosBc = sc.broadcast(trios)
    val trioSexBc = sc.broadcast(trios.flatMap(_.sex))

    val zeroVal: Array[Array[GenotypeType]] =
      Array.fill[Array[GenotypeType]](trios.size)(Array.fill[GenotypeType](3)(NoCall))

    def seqOp(a: Array[Array[GenotypeType]], s: Int, g: Genotype): Array[Array[GenotypeType]] = {
      sampleTrioRolesBc.value(s).foreach{ case (ti, ri) => a(ti)(ri) = g.gtType }
      a
    }

    def mergeOp(a: Array[Array[GenotypeType]], b: Array[Array[GenotypeType]]): Array[Array[GenotypeType]] = {
      for (ti <- a.indices)
        for (ri <- 0 to 2)
          if (b(ti)(ri) != NoCall)
            a(ti)(ri) = b(ti)(ri)
      a
    }

    new MendelErrors(trios, vds.sampleIds,
      vds
      .aggregateByVariantWithKeys(zeroVal)(
        (a, v, s, g) => seqOp(a, s, g),
        mergeOp)
      .flatMap{ case (v, a) =>
        a.indices.flatMap{
          ti => {
            val code = getCode(a(ti), v.isHemizygous(trioSexBc.value(ti)))
            if (code != 0)
              Some(new MendelError(v, triosBc.value(ti), code, a(ti)(0), a(ti)(1), a(ti)(2)))
            else
              None
          }
        }
      }
      .cache()
    )
  }
}

case class MendelErrors(trios:        Array[CompleteTrio],
                        sampleIds:    Array[String],
                        mendelErrors: RDD[MendelError]) {

  val sc = mendelErrors.sparkContext

  val nuclearFams: Map[(Int, Int), Iterable[Int]] =
    trios
      .map(t => ((t.dad, t.mom), t.kid))
      .toMap
      .groupByKey

  val triosBc = sc.broadcast(trios)
  val famOfBc = sc.broadcast(trios.flatMap(t => t.fam.map(f => (t.kid, f))).toMap)
  val sampleIdsBc = sc.broadcast(sampleIds)

  def nErrorPerVariant: RDD[(Variant, Int)] = {
    mendelErrors
      .map(_.variant)
      .countByValueRDD()
  }

  def nErrorPerNuclearFamily: RDD[((Int, Int), Int)] = {
    val parentsRDD = sc.parallelize(nuclearFams.keys.toSeq)
    mendelErrors
      .map(me => ((me.trio.dad, me.trio.mom), 1))
      .union(parentsRDD.map((_, 0)))
      .reduceByKey(_ + _)
  }

  def nErrorPerIndiv: RDD[(Int, Int)] = {
    val indivRDD = sc.parallelize(trios.flatMap(t => List(t.kid, t.dad, t.mom)).distinct)
    def implicatedSamples(me: MendelError): List[Int] = {
      val t = me.trio
      val c = me.code
      if      (c == 2 || c == 1)                       List(t.kid, t.dad, t.mom)
      else if (c == 6 || c == 3)                       List(t.kid, t.dad)
      else if (c == 4 || c == 7 || c == 9 || c == 10)  List(t.kid, t.mom)
      else                                             List(t.kid)
    }
    mendelErrors
      .flatMap(implicatedSamples)
      .map((_, 1))
      .union(indivRDD.map((_, 0)))
      .reduceByKey(_ + _)
  }

  def writeMendel(filename: String) {
    def gtString(v: Variant, gt: GenotypeType): String = {
      if (gt == HomRef)
        v.ref + "/" + v.ref
      else if (gt == Het)
        v.ref + "/" + v.alt
      else if (gt == HomVar)
        v.alt + "/" + v.alt
      else
        "./."
    }
    def toLine(me: MendelError): String = {
      val v = me.variant
      val t = me.trio
      val errorString = gtString(v, me.gtDad) + " x " + gtString(v, me.gtMom) + " -> " + gtString(v, me.gtKid)
      t.fam.getOrElse("0") + "\t" + sampleIdsBc.value(t.kid) + "\t" + v.contig + "\t" +
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
    val nuclearFamsBc = sc.broadcast(nuclearFams.force)
    def toLine(parents: (Int, Int), nError: Int): String = {
      val (dad, mom) = parents
      famOfBc.value.getOrElse(dad, "0") + "\t" + sampleIdsBc.value(dad) + "\t" + sampleIdsBc.value(mom) + "\t" +
        nuclearFamsBc.value((dad, mom)).size + "\t" + nError + "\n"
    }
    val lines = nErrorPerNuclearFamily.map((toLine _).tupled).collect()
    writeTable(filename, sc.hadoopConfiguration, lines, "FID\tPAT\tMAT\tCHLD\tN\n")
  }

  def writeMendelI(filename: String) {
    def toLine(s: Int, nError: Int): String =
      famOfBc.value.getOrElse(s, "0") + "\t" + sampleIdsBc.value(s) + "\t" + nError + "\n"
    val lines = nErrorPerIndiv.map((toLine _).tupled).collect()
    writeTable(filename, sc.hadoopConfiguration, lines, "FID\tIID\tN\n")
  }
}
