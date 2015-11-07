package org.broadinstitute.hail.methods

import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant._

import org.broadinstitute.hail.variant.GenotypeType._

case class MendelError(variant: Variant, sample: Int, code: Int,
                       gtKid: GenotypeType, gtDad: GenotypeType, gtMom: GenotypeType)

object MendelErrors {

  def variantString(v: Variant): String = v.contig + ":" + v.start + ":" + v.ref + ":" + v.alt

  def getCode(gts: Array[GenotypeType], isHemizygous: Boolean): Int = {
    (gts(1), gts(2), gts(0), isHemizygous) match {
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

    val trios = ped.completeTrios
    val nTrios = trios.size
    val trioSamples = ped.samplesInCompleteTrios
    val nTrioSamples = trioSamples.size
    val trioSampleIndex: Map[Int, Int] = trioSamples.zipWithIndex.toMap
    val isTrioSample = (0 to vds.nSamples).map(trioSamples.contains(_)).toArray

    val sampleIndexTrioRoles: Array[List[(Int, Int)]] = {
      val a: Array[List[(Int, Int)]] = Array.fill[List[(Int, Int)]](nTrioSamples)(List())
      trios.indices.flatMap { ti => {
          val t = trios(ti)
          List((trioSampleIndex(t.kid), (ti, 0)), (trioSampleIndex(t.dad.get), (ti, 1)), (trioSampleIndex(t.mom.get), (ti, 2)))
        }
      }
      .foreach{ case (si, tiri) => a(si) ::= tiri }
      a
    }

    val sc = vds.sparkContext
    val trioSamplesBc = sc.broadcast(trioSamples)
    val trioSampleIndexBc = sc.broadcast(trioSampleIndex)
    val trioSexBc = sc.broadcast(trios.flatMap(t => t.sex))
    val sampleIndexTrioRolesBc = sc.broadcast(sampleIndexTrioRoles)
    val trioKidBc = sc.broadcast(trios.zipWithIndex.map{ case (t, ti) => (ti, t.kid) }.toMap)

    val zeroVal: Array[Array[GenotypeType]] =
      Array.fill[Array[GenotypeType]](nTrios)(Array.fill[GenotypeType](3)(NoCall))

    def seqOp(a: Array[Array[GenotypeType]], s: Int, g: Genotype): Array[Array[GenotypeType]] = {
      sampleIndexTrioRolesBc.value(trioSampleIndexBc.value(s)).foreach { case (ti, ri) => a(ti)(ri) = g.gtType }
      a
    }

    def mergeOp(a: Array[Array[GenotypeType]], b: Array[Array[GenotypeType]]): Array[Array[GenotypeType]] = {
      for (ti <- a.indices)
        for (ri <- 0 to 2)
          if (b(ti)(ri) != NoCall)
            a(ti)(ri) = b(ti)(ri)
      a
    }

    new MendelErrors(ped, vds.sampleIds,
      vds
      .filterSamples(isTrioSample)
      .aggregateByVariantWithKeys(zeroVal)(
        (a, v, s, g) => seqOp(a, s, g),
        mergeOp)
      .flatMap{ case (v, a) =>
        a.indices.flatMap{
          ti => {
            val code = getCode(a(ti), v.isHemizygous(trioSexBc.value(ti)))
            if (code != 0) {
              val s = trioKidBc.value(ti)
              Some(new MendelError(v, s, code, a(ti)(0), a(ti)(1), a(ti)(2)))
            }
            else
              None
          }
        }
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
      val s = me.sample
      val errorString = gtString(v, me.gtDad) + " x " + gtString(v, me.gtMom) + " -> " + gtString(v, me.gtKid)
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
