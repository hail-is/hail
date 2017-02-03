package is.hail.methods

import org.apache.spark.rdd.RDD
import is.hail.utils.MultiArray2
import is.hail.utils._
import is.hail.variant.CopyState._
import is.hail.variant._
import is.hail.variant.GenotypeType._

import scala.collection.mutable

case class MendelError(variant: Variant, trio: CompleteTrio, code: Int,
  gtKid: GenotypeType, gtDad: GenotypeType, gtMom: GenotypeType) {

  def gtString(v: Variant, gt: GenotypeType): String =
    if (gt == HomRef)
      v.ref + "/" + v.ref
    else if (gt == Het)
      v.ref + "/" + v.alt
    else if (gt == HomVar)
      v.alt + "/" + v.alt
    else
      "./."

  def implicatedSamplesWithCounts: Iterator[(String, (Int, Int))] = {
    if (code == 2 || code == 1) Iterator(trio.kid, trio.dad, trio.mom)
    else if (code == 6 || code == 3 || code == 11 || code == 12) Iterator(trio.kid, trio.dad)
    else if (code == 4 || code == 7 || code == 9 || code == 10) Iterator(trio.kid, trio.mom)
    else Iterator(trio.kid)
  }
    .map((_, (1, if (variant.altAllele.isSNP) 1 else 0)))

  def toLineMendel(sampleIds: IndexedSeq[String]): String = {
    val v = variant
    val t = trio
    val errorString = gtString(v, gtDad) + " x " + gtString(v, gtMom) + " -> " + gtString(v, gtKid)
    t.fam.getOrElse("0") + "\t" + t.kid + "\t" + v.contig + "\t" +
      v.contig + ":" + v.start + ":" + v.ref + ":" + v.alt + "\t" + code + "\t" + errorString
  }
}

object MendelErrors {

  def getCode(gts: IndexedSeq[GenotypeType], copyState: CopyState): Int = {
    (gts(1), gts(2), gts(0), copyState) match {
      // (gtDad, gtMom, gtKid)
      case (HomRef, HomRef, Het, Auto) => 2 // Kid is Het
      case (HomVar, HomVar, Het, Auto) => 1
      case (HomRef, HomRef, HomVar, Auto) => 5 // Kid is HomVar
      case (HomRef, _, HomVar, Auto) => 3
      case (_, HomRef, HomVar, Auto) => 4
      case (HomVar, HomVar, HomRef, Auto) => 8 // Kid is HomRef
      case (HomVar, _, HomRef, Auto) => 6
      case (_, HomVar, HomRef, Auto) => 7
      case (_, HomVar, HomRef, HemiX) => 9 // Kid is hemizygous
      case (_, HomRef, HomVar, HemiX) => 10
      case (HomVar, _, HomRef, HemiY) => 11
      case (HomRef, _, HomVar, HemiY) => 12
      case _ => 0 // No error
    }
  }

  def apply(vds: VariantDataset, preTrios: IndexedSeq[CompleteTrio]): MendelErrors = {

    val trios = preTrios.filter(_.sex.isDefined)
    val nSamplesDiscarded = preTrios.size - trios.size

    if (nSamplesDiscarded > 0)
      warn(s"$nSamplesDiscarded ${ plural(nSamplesDiscarded, "sample") } discarded from .fam: sex of child is missing.")

    val sampleTrioRoles = mutable.Map.empty[String, List[(Int, Int)]]
    trios.zipWithIndex.foreach { case (t, ti) =>
      sampleTrioRoles += (t.kid -> sampleTrioRoles.getOrElse(t.kid, List.empty[(Int, Int)]).::(ti, 0))
      sampleTrioRoles += (t.dad -> sampleTrioRoles.getOrElse(t.dad, List.empty[(Int, Int)]).::(ti, 1))
      sampleTrioRoles += (t.mom -> sampleTrioRoles.getOrElse(t.mom, List.empty[(Int, Int)]).::(ti, 2))
    }

    val sc = vds.sparkContext
    val sampleTrioRolesBc = sc.broadcast(sampleTrioRoles)
    val triosBc = sc.broadcast(trios)
    // all trios have defined sex, see filter above
    val trioSexBc = sc.broadcast(trios.map(_.sex.get))

    val zeroVal: MultiArray2[GenotypeType] = MultiArray2.fill(trios.length, 3)(NoCall)

    def seqOp(a: MultiArray2[GenotypeType], s: String, g: Genotype): MultiArray2[GenotypeType] = {
      sampleTrioRolesBc.value.get(s).foreach(l => l.foreach { case (ti, ri) => a.update(ti, ri, g.gtType) })
      a
    }

    def mergeOp(a: MultiArray2[GenotypeType], b: MultiArray2[GenotypeType]): MultiArray2[GenotypeType] = {
      for ((i, j) <- a.indices)
        if (b(i, j) != NoCall)
          a(i, j) = b(i, j)
      a
    }

    new MendelErrors(trios, vds.sampleIds,
      vds
        .aggregateByVariantWithKeys(zeroVal)(
          (a, v, s, g) => seqOp(a, s, g),
          mergeOp)
        .flatMap { case (v, a) =>
          a.rows.flatMap { case (row) => val code = getCode(row, v.copyState(trioSexBc.value(row.i)))
            if (code != 0)
              Some(new MendelError(v, triosBc.value(row.i), code, row(0), row(1), row(2)))
            else
              None
          }
        }
        .cache()
    )
  }
}

case class MendelErrors(trios: IndexedSeq[CompleteTrio],
  sampleIds: IndexedSeq[String],
  mendelErrors: RDD[MendelError]) {

  val sc = mendelErrors.sparkContext
  val trioFam = trios.iterator.flatMap(t => t.fam.map(f => (t.kid, f))).toMap
  val nuclearFams = Pedigree.nuclearFams(trios)

  def nErrorPerVariant: RDD[(Variant, Int)] = {
    mendelErrors
      .map(_.variant)
      .countByValueRDD()
  }

  def nErrorPerNuclearFamily: RDD[((String, String), (Int, Int))] = {
    val parentsRDD = sc.parallelize(nuclearFams.keys.toSeq)
    mendelErrors
      .map(me => ((me.trio.dad, me.trio.mom), (1, if (me.variant.altAllele.isSNP) 1 else 0)))
      .union(parentsRDD.map((_, (0, 0))))
      .reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2))
  }

  def nErrorPerIndiv: RDD[(String, (Int, Int))] = {
    val indivRDD = sc.parallelize(trios.flatMap(t => Iterator(t.kid, t.dad, t.mom)).distinct)
    mendelErrors
      .flatMap(_.implicatedSamplesWithCounts)
      .union(indivRDD.map((_, (0, 0))))
      .reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2))
  }

  def writeMendel(filename: String, tmpDir: String) {
    val sampleIdsBc = sc.broadcast(sampleIds)
    mendelErrors.map(_.toLineMendel(sampleIdsBc.value))
      .writeTable(filename, tmpDir, Some("FID\tKID\tCHR\tSNP\tCODE\tERROR"))
  }

  def writeMendelF(filename: String) {
    val trioFamBc = sc.broadcast(trioFam)
    val nuclearFamsBc = sc.broadcast(nuclearFams)
    val lines = nErrorPerNuclearFamily.map { case ((dad, mom), (n, nSNP)) =>
      trioFamBc.value.getOrElse(dad, "0") + "\t" + dad + "\t" + mom + "\t" +
        nuclearFamsBc.value((dad, mom)).size + "\t" + n + "\t" + nSNP
    }.collect()
    sc.hadoopConfiguration.writeTable(filename, lines, Some("FID\tPAT\tMAT\tCHLD\tN\tNSNP"))
  }

  def writeMendelI(filename: String) {
    val trioFamBc = sc.broadcast(trioFam)
    val sampleIdsBc = sc.broadcast(sampleIds)
    val lines = nErrorPerIndiv.map { case (s, (n, nSNP)) =>
      trioFamBc.value.getOrElse(s, "0") + "\t" + s + "\t" + n + "\t" + nSNP
    }.collect()
    sc.hadoopConfiguration.writeTable(filename, lines, Some("FID\tIID\tN\tNSNP"))
  }

  def writeMendelL(filename: String, tmpDir: String) {
    nErrorPerVariant.map { case (v, n) =>
      v.contig + "\t" + v.contig + ":" + v.start + ":" + v.ref + ":" + v.alt + "\t" + n
    }.writeTable(filename, tmpDir, Some("CHR\tSNP\tN"))
  }
}
